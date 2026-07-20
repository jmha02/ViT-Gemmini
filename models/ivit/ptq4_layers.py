"""Relay helpers for flexi PTQ4ViT (eval/specs/ptq4_deit.py semantics).

Gemmini vs CPU:
  Gemmini int8 MM: patch embed, QKV split blocks, Q@K, proj, fc1, twin V/Gelu MM, head
  CPU float: LayerNorm, Softmax, GELU, twin quant/combine scales
"""

from __future__ import annotations

import numpy as np
from tvm import relay
from tvm.relay.frontend.common import infer_shape as _infer_shape


I8_MIN = -128.0
I8_MAX = 127.0


def _const_f32(value):
    if isinstance(value, (float, int)):
        return relay.const(float(value), "float32")
    return relay.const(np.asarray(value, dtype=np.float32), "float32")


def _const_i32(value):
    if isinstance(value, (float, int)):
        return relay.const(int(value), "int32")
    return relay.const(np.asarray(value, dtype=np.int32), "int32")


def _param(name: str, dtype: str, shape=()):
    return relay.var(name, shape=shape, dtype=dtype)


def _scalar_f32(name: str):
    return _param(name, "float32", ())


def _gemmini_gemm_int32(data_int8: relay.Expr, weight_int8_kn: relay.Expr) -> relay.Expr:
    from tvm.contrib.gemmini.legalize import gemmini_gemm

    weight_shape = tuple(int(dim) for dim in _infer_shape(weight_int8_kn))
    if len(weight_shape) != 2:
        raise RuntimeError(f"gemmini_gemm expects rank-2 weights, got {weight_shape}")
    units = weight_shape[1]
    zero_bias = relay.const(np.zeros((units,), dtype="int32"), "int32")
    ones = relay.const(np.ones((units,), dtype="float32"), "float32")
    weights_nk = relay.transpose(weight_int8_kn, axes=[1, 0])
    return gemmini_gemm(
        data_int8,
        weights_nk,
        zero_bias,
        relay.const(np.float32(1.0), "float32"),
        relay.const(np.int32(0), "int32"),
        ones,
        relay.const(np.int32(0), "int32"),
        relay.const(np.float32(1.0), "float32"),
        relay.const(np.int32(0), "int32"),
    )


def gemmini_batch_matmul_transpose_b_int32(lhs_i8, rhs_i8):
    """Gemmini-backed int8 batch matmul: lhs @ rhs^T for rank-3 tensors."""
    lhs_shape = _infer_shape(lhs_i8)
    rhs_shape = _infer_shape(rhs_i8)
    if len(lhs_shape) != 3 or len(rhs_shape) != 3:
        raise RuntimeError("Gemmini batch matmul expects rank-3 tensors")
    if int(lhs_shape[0]) != int(rhs_shape[0]):
        raise RuntimeError("Gemmini batch matmul inputs must have matching batch size")
    if int(lhs_shape[2]) != int(rhs_shape[2]):
        raise RuntimeError("Gemmini batch matmul inputs must have matching reduction dim")

    batch = int(lhs_shape[0])
    lhs_slices = relay.split(lhs_i8, batch, axis=0)
    rhs_slices = relay.split(rhs_i8, batch, axis=0)
    outs = []
    for idx in range(batch):
        lhs = relay.squeeze(lhs_slices[idx], axis=[0])
        rhs = relay.squeeze(rhs_slices[idx], axis=[0])
        rhs_t = relay.transpose(rhs, axes=[1, 0])
        outs.append(relay.expand_dims(_gemmini_gemm_int32(lhs, rhs_t), axis=0))
    return relay.concatenate(outs, axis=0)


def gemmini_batch_matmul_int32(lhs_i8, rhs_i8):
    """Gemmini-backed int8 batch matmul: lhs @ rhs for rank-3 (no transpose)."""
    lhs_shape = _infer_shape(lhs_i8)
    rhs_shape = _infer_shape(rhs_i8)
    if len(lhs_shape) != 3 or len(rhs_shape) != 3:
        raise RuntimeError("Gemmini batch matmul expects rank-3 tensors")
    if int(lhs_shape[0]) != int(rhs_shape[0]):
        raise RuntimeError("Gemmini batch matmul inputs must have matching batch size")
    if int(lhs_shape[2]) != int(rhs_shape[1]):
        raise RuntimeError("Gemmini batch matmul reduction dims mismatch")

    batch = int(lhs_shape[0])
    lhs_slices = relay.split(lhs_i8, batch, axis=0)
    rhs_slices = relay.split(rhs_i8, batch, axis=0)
    outs = []
    for idx in range(batch):
        lhs = relay.squeeze(lhs_slices[idx], axis=[0])
        rhs = relay.squeeze(rhs_slices[idx], axis=[0])
        # _gemmini_gemm expects weight as (K, N); rhs is (K, N) already for A@B
        outs.append(relay.expand_dims(_gemmini_gemm_int32(lhs, rhs), axis=0))
    return relay.concatenate(outs, axis=0)


def ptq4_q_sym_i8(x, scale_expr):
    """Symmetric int8 quant: clamp(round(x / scale), -128, 127)."""
    rounded = relay.round(relay.divide(x, scale_expr))
    return relay.cast(relay.clip(rounded, I8_MIN, I8_MAX), "int8")


def ptq4_layer_norm(x, prefix: str, dim: int, eps: float = 1e-5):
    weight = _param(f"{prefix}_weight", "float32", (dim,))
    bias = _param(f"{prefix}_bias", "float32", (dim,))
    return relay.nn.layer_norm(x, gamma=weight, beta=bias, axis=-1, epsilon=eps)


def ptq4_int_mm_linear(x, prefix: str, *, out_features: int):
    """IntMmLinear: quantize → gemmini MM → *output_scale + bias."""
    lead_shape = list(_infer_shape(x)[:-1])
    in_features = int(_infer_shape(x)[-1])
    a_interval = _scalar_f32(f"{prefix}_a_interval")
    x_i8 = ptq4_q_sym_i8(x, a_interval)
    x_flat = relay.reshape(x_i8, [-1, in_features])
    weight = _param(f"{prefix}_weight_q_t", "int8", (in_features, out_features))
    acc = _gemmini_gemm_int32(x_flat, weight)
    acc_f = relay.cast(acc, "float32")
    scale = _param(f"{prefix}_output_scale", "float32", (out_features,))
    bias = _param(f"{prefix}_bias", "float32", (out_features,))
    y = relay.add(relay.multiply(acc_f, scale), bias)
    return relay.reshape(y, lead_shape + [out_features])


def ptq4_patch_embed(
    x,
    prefix: str,
    *,
    batch_size: int,
    in_chans: int,
    img_size: int,
    patch_size: int,
    embed_dim: int,
):
    """PatchEmbed: unfold → int8 quant → gemmini MM → scale+bias."""
    grid = img_size // patch_size
    num_patches = grid * grid
    patch_dim = in_chans * patch_size * patch_size
    # x: [B, C, H, W] → [B, N, K]
    patches = relay.reshape(x, [batch_size, in_chans, grid, patch_size, grid, patch_size])
    patches = relay.transpose(patches, axes=[0, 2, 4, 1, 3, 5])
    patches = relay.reshape(patches, [batch_size, num_patches, patch_dim])

    a_interval = _scalar_f32(f"{prefix}_a_interval")
    x_i8 = ptq4_q_sym_i8(patches, a_interval)
    x_flat = relay.reshape(x_i8, [-1, patch_dim])
    weight = _param(f"{prefix}_weight_q_t", "int8", (patch_dim, embed_dim))
    acc = _gemmini_gemm_int32(x_flat, weight)
    acc_f = relay.cast(acc, "float32")
    weight_scale = _param(f"{prefix}_weight_scale", "float32", (embed_dim,))
    bias = _param(f"{prefix}_bias", "float32", (embed_dim,))
    out_scale = relay.multiply(a_interval, weight_scale)
    y = relay.add(relay.multiply(acc_f, out_scale), bias)
    return relay.reshape(y, [batch_size, num_patches, embed_dim])


def ptq4_qkv_split_linear(x, prefix: str, *, dim: int, num_heads: int):
    """QkvSplitLinear → (q, k, v) each [B, H, N, Hd] int8."""
    batch, seq, _ = [int(v) for v in _infer_shape(x)]
    head_dim = dim // num_heads
    num_blocks = 3 * num_heads
    a_interval = _scalar_f32(f"{prefix}_a_interval")
    x_i8 = ptq4_q_sym_i8(x, a_interval)
    x_flat = relay.reshape(x_i8, [-1, dim])

    blocks = []
    for g in range(num_blocks):
        w = _param(f"{prefix}_w{g}", "int8", (dim, head_dim))
        d = _param(f"{prefix}_d{g}", "int32", (head_dim,))
        s = _scalar_f32(f"{prefix}_s{g}")
        acc = relay.add(_gemmini_gemm_int32(x_flat, w), d)
        scaled = relay.multiply(relay.cast(acc, "float32"), s)
        q = relay.cast(relay.clip(relay.round(scaled), I8_MIN, I8_MAX), "int8")
        blocks.append(relay.reshape(q, [batch, seq, head_dim]))

    # stack → [3, H, B, N, Hd] then transpose to head-major Q/K/V
    stacked = relay.stack(blocks, axis=0)  # [3H, B, N, Hd]
    stacked = relay.reshape(stacked, [3, num_heads, batch, seq, head_dim])
    q = relay.transpose(relay.take(stacked, _const_i32(0), axis=0), axes=[1, 0, 2, 3])
    k = relay.transpose(relay.take(stacked, _const_i32(1), axis=0), axes=[1, 0, 2, 3])
    v = relay.transpose(relay.take(stacked, _const_i32(2), axis=0), axes=[1, 0, 2, 3])
    return q, k, v


def ptq4_int_mm_matmul(q_i8, k_i8, prefix: str, *, num_heads: int):
    """IntMmMatMul: Q @ K^T * per-head output_scale → float scores."""
    b, h, n, d = [int(v) for v in _infer_shape(q_i8)]
    q3 = relay.reshape(q_i8, [b * h, n, d])
    k3 = relay.reshape(k_i8, [b * h, n, d])
    acc3 = gemmini_batch_matmul_transpose_b_int32(q3, k3)
    acc = relay.cast(relay.reshape(acc3, [b, h, n, n]), "float32")
    scale = _param(f"{prefix}_output_scale", "float32", (num_heads,))
    scale_b = relay.reshape(scale, [1, num_heads, 1, 1])
    return relay.multiply(acc, scale_b)


def ptq4_scaled_softmax(scores, head_dim: int):
    scale = _const_f32(float(head_dim) ** -0.5)
    return relay.nn.softmax(relay.multiply(scores, scale), axis=-1)


def ptq4_twin_softmax_matmul(attn, v_i8, prefix: str, *, num_heads: int):
    """TwinSoftmaxMatMul: split attn into hi/lo int8 twins × V."""
    b, h, n, d = [int(v) for v in _infer_shape(v_i8)]
    split = _scalar_f32(f"{prefix}_split")
    a_interval = _scalar_f32(f"{prefix}_a_interval")
    b_interval = _param(f"{prefix}_b_interval", "float32", (num_heads,))
    qm1 = _const_f32(127.0)

    # Match torch.clamp(a, split, 1.0) / torch.clamp(a, 0.0, split)
    hi_src = relay.minimum(relay.maximum(attn, split), _const_f32(1.0))
    lo_src = relay.minimum(relay.maximum(attn, _const_f32(0.0)), split)

    q_hi = relay.cast(relay.clip(relay.round(relay.multiply(hi_src, qm1)), 0.0, I8_MAX), "int8")
    q_lo = relay.cast(
        relay.clip(relay.round(relay.divide(lo_src, a_interval)), 0.0, I8_MAX),
        "int8",
    )

    q_hi3 = relay.reshape(q_hi, [b * h, n, n])
    q_lo3 = relay.reshape(q_lo, [b * h, n, n])
    v3 = relay.reshape(v_i8, [b * h, n, d])
    acc_hi = gemmini_batch_matmul_int32(q_hi3, v3)
    acc_lo = gemmini_batch_matmul_int32(q_lo3, v3)
    acc_hi_f = relay.cast(relay.reshape(acc_hi, [b, h, n, d]), "float32")
    acc_lo_f = relay.cast(relay.reshape(acc_lo, [b, h, n, d]), "float32")
    combined = relay.add(
        relay.divide(acc_hi_f, qm1),
        relay.multiply(acc_lo_f, a_interval),
    )
    scale_b = relay.reshape(b_interval, [1, num_heads, 1, 1])
    return relay.multiply(combined, scale_b)


def ptq4_twin_gelu_linear(x, prefix: str, *, out_features: int):
    """TwinGeluLinear: pos/neg twin quant → two gemmini MMs → combine."""
    lead_shape = list(_infer_shape(x)[:-1])
    in_features = int(_infer_shape(x)[-1])
    a_interval = _scalar_f32(f"{prefix}_a_interval")
    a_neg_interval = _scalar_f32(f"{prefix}_a_neg_interval")
    weight = _param(f"{prefix}_weight_q_t", "int8", (in_features, out_features))
    weight_scale = _param(f"{prefix}_weight_scale", "float32", (out_features,))
    bias = _param(f"{prefix}_bias", "float32", (out_features,))

    q_pos = relay.cast(
        relay.clip(relay.round(relay.divide(relay.maximum(x, _const_f32(0.0)), a_interval)), 0.0, I8_MAX),
        "int8",
    )
    q_neg = relay.cast(
        relay.clip(
            relay.round(relay.divide(relay.minimum(x, _const_f32(0.0)), a_neg_interval)),
            -I8_MAX,
            0.0,
        ),
        "int8",
    )
    xp = relay.reshape(q_pos, [-1, in_features])
    xn = relay.reshape(q_neg, [-1, in_features])
    acc_pos = relay.cast(_gemmini_gemm_int32(xp, weight), "float32")
    acc_neg = relay.cast(_gemmini_gemm_int32(xn, weight), "float32")
    pos_scale = relay.multiply(a_interval, weight_scale)
    neg_scale = relay.multiply(a_neg_interval, weight_scale)
    out = relay.add(relay.multiply(acc_pos, pos_scale), relay.multiply(acc_neg, neg_scale))
    out = relay.add(out, bias)
    return relay.reshape(out, lead_shape + [out_features])


def ptq4_attention(x, prefix: str, *, dim: int, num_heads: int, seq_len: int):
    head_dim = dim // num_heads
    q, k, v = ptq4_qkv_split_linear(x, f"{prefix}_attn_qkv", dim=dim, num_heads=num_heads)
    scores = ptq4_int_mm_matmul(q, k, f"{prefix}_attn_matmul1", num_heads=num_heads)
    attn = ptq4_scaled_softmax(scores, head_dim)
    ctx = ptq4_twin_softmax_matmul(attn, v, f"{prefix}_attn_matmul2", num_heads=num_heads)
    batch = int(_infer_shape(x)[0])
    ctx = relay.transpose(ctx, axes=[0, 2, 1, 3])
    ctx = relay.reshape(ctx, [batch, seq_len, dim])
    return ptq4_int_mm_linear(ctx, f"{prefix}_attn_proj", out_features=dim)


def ptq4_gelu(x):
    """GELU approx matching torch.nn.GELU (erf form)."""
    # 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = _const_f32(float(np.sqrt(0.5)))
    return relay.multiply(
        relay.multiply(_const_f32(0.5), x),
        relay.add(_const_f32(1.0), relay.erf(relay.multiply(x, inv_sqrt2))),
    )


def ptq4_mlp(x, prefix: str, *, dim: int, mlp_ratio: int = 4):
    hidden = int(dim * mlp_ratio)
    h = ptq4_int_mm_linear(x, f"{prefix}_mlp_fc1", out_features=hidden)
    h = ptq4_gelu(h)
    return ptq4_twin_gelu_linear(h, f"{prefix}_mlp_fc2", out_features=dim)


def ptq4_block(x, prefix: str, *, dim: int, num_heads: int, seq_len: int, mlp_ratio: int = 4):
    y = ptq4_layer_norm(x, f"{prefix}_norm1", dim)
    y = ptq4_attention(y, prefix, dim=dim, num_heads=num_heads, seq_len=seq_len)
    x = relay.add(x, y)
    y = ptq4_layer_norm(x, f"{prefix}_norm2", dim)
    y = ptq4_mlp(y, prefix, dim=dim, mlp_ratio=mlp_ratio)
    return relay.add(x, y)
