from __future__ import annotations

import re

from tvm import relay

from . import layers


_ONLY_REPQ_DEIT_BLOCK_RE = re.compile(r"^only_block(\d+)$")


def _parse_only_repq_deit_block(debug_unit: str | None, depth: int) -> int | None:
    if not debug_unit:
        return None
    match = _ONLY_REPQ_DEIT_BLOCK_RE.match(debug_unit)
    if not match:
        return None
    block_idx = int(match.group(1))
    if not 0 <= block_idx < depth:
        raise RuntimeError(f"Unsupported RepQ DeiT standalone block index: {block_idx}")
    return block_idx


def _layer_norm_var(module_name: str, dim: int) -> tuple[relay.Expr, relay.Expr]:
    prefix = layers.sanitize_name(module_name)
    weight = relay.var(prefix + "_weight", shape=[dim], dtype="float32")
    bias = relay.var(prefix + "_bias", shape=[dim], dtype="float32")
    return weight, bias


def _reshape_qkv(qkv: relay.Expr, batch_size: int, num_tokens: int, num_heads: int, head_dim: int):
    qkv = relay.reshape(qkv, [batch_size, num_tokens, 3, num_heads, head_dim])
    qkv = relay.transpose(qkv, axes=[2, 0, 3, 1, 4])
    qkv = relay.split(qkv, 3, axis=0)

    def _pack(part):
        part = relay.squeeze(part, axis=[0])
        return relay.reshape(part, [batch_size * num_heads, num_tokens, head_dim])

    q = _pack(qkv[0])
    k = _pack(qkv[1])
    v = _pack(qkv[2])
    return q, k, v


def _repq_block(
    data: relay.Expr,
    block_idx: int,
    embed_dim: int,
    num_heads: int,
    batch_size: int,
    num_tokens: int,
) -> relay.Expr:
    head_dim = embed_dim // num_heads
    block_prefix = f"blocks.{block_idx}"
    shortcut = data

    norm1_weight, norm1_bias = _layer_norm_var(block_prefix + ".norm1", embed_dim)
    x = layers.layer_norm_float(data, norm1_weight, norm1_bias)

    qkv = layers.exact_uniform_linear(x, block_prefix + ".attn.qkv", add_bias=True)
    q, k, v = _reshape_qkv(qkv, batch_size, num_tokens, num_heads, head_dim)

    matmul1_meta = layers.RepQContext.get(block_prefix + ".attn.matmul1")
    attn = layers.exact_uniform_matmul(
        q,
        relay.transpose(k, axes=[0, 2, 1]),
        a_scale=matmul1_meta.input_scale,
        a_zero_point=matmul1_meta.input_zero_point,
        b_scale=matmul1_meta.b_scale,
        b_zero_point=matmul1_meta.b_zero_point,
        n_bits=matmul1_meta.n_bits,
    )
    attn = attn * relay.const((head_dim ** -0.5), "float32")
    attn = relay.nn.softmax(attn, axis=-1)

    matmul2_meta = layers.RepQContext.get(block_prefix + ".attn.matmul2")
    if layers.RepQContext.use_gemmini_ops:
        x = layers.approx_log_matmul(
            attn,
            v,
            log_delta=matmul2_meta.log_scale,
            value_scale=matmul2_meta.b_scale,
            value_zero_point=matmul2_meta.b_zero_point,
            n_bits=matmul2_meta.n_bits,
        )
    else:
        attn_q = layers.repq_log_quant(attn, matmul2_meta.log_scale, matmul2_meta.n_bits)
        v_q = layers.uniform_quant_dequant(
            v,
            scale=matmul2_meta.b_scale,
            zero_point=matmul2_meta.b_zero_point,
            n_bits=matmul2_meta.n_bits,
            axis=-1,
        )
        x = layers.batched_float_matmul(attn_q, v_q)
    x = relay.reshape(x, [batch_size, num_heads, num_tokens, head_dim])
    x = relay.transpose(x, axes=[0, 2, 1, 3])
    x = relay.reshape(x, [batch_size, num_tokens, embed_dim])

    x = layers.exact_uniform_linear(x, block_prefix + ".attn.proj", add_bias=True)
    x = x + shortcut

    shortcut = x
    norm2_weight, norm2_bias = _layer_norm_var(block_prefix + ".norm2", embed_dim)
    x = layers.layer_norm_float(x, norm2_weight, norm2_bias)
    x = layers.exact_uniform_linear(x, block_prefix + ".mlp.fc1", add_bias=True)
    x = layers.gelu_float(x)
    x = layers.exact_uniform_linear(x, block_prefix + ".mlp.fc2", add_bias=True)
    return relay.annotation.stop_fusion(x + shortcut)


def RepQVisionTransformer(
    data_shape,
    patch_size: int = 16,
    num_patches: int = 196,
    num_classes: int = 1000,
    embed_dim: int = 192,
    depth: int = 12,
    num_heads: int = 3,
    debug_unit: str | None = None,
):
    batch_size = data_shape[0]
    num_tokens = num_patches + 1
    data = relay.var("data", shape=data_shape, dtype="float32")

    if debug_unit in ("only_embed", "only_embedding"):
        debug_unit = "post_patch_embed"

    standalone_block_idx = _parse_only_repq_deit_block(debug_unit, depth)
    if standalone_block_idx is not None:
        block_input = relay.var(
            "data",
            shape=[batch_size, num_tokens, embed_dim],
            dtype="float32",
        )
        body = _repq_block(
            block_input,
            block_idx=standalone_block_idx,
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_size=batch_size,
            num_tokens=num_tokens,
        )
        return relay.Function(relay.analysis.free_vars(body), body)

    if debug_unit in ("only_head", "only_classifier", "classifier_only"):
        x = relay.var(
            "data",
            shape=[batch_size, num_tokens, embed_dim],
            dtype="float32",
        )
        norm_weight, norm_bias = _layer_norm_var("norm", embed_dim)
        x = layers.layer_norm_float(x, norm_weight, norm_bias)
        x = relay.strided_slice(
            x,
            begin=[0, 0, 0],
            end=[batch_size, 1, embed_dim],
            strides=[1, 1, 1],
        )
        x = relay.reshape(x, [batch_size, embed_dim])
        x = layers.exact_uniform_linear(x, "head", add_bias=True)
        return relay.Function(relay.analysis.free_vars(x), x)

    x = layers.patchify_conv2d_exact(
        data,
        module_name="patch_embed.proj",
        kernel_size=(patch_size, patch_size),
        stride=(patch_size, patch_size),
        out_channels=embed_dim,
    )
    x = relay.reshape(x, [batch_size, num_patches, embed_dim])
    if debug_unit == "post_patch_embed":
        return relay.Function(relay.analysis.free_vars(x), x)

    cls_token = relay.var("cls_token", shape=(1, 1, embed_dim), dtype="float32")
    cls_tokens = relay.repeat(cls_token, batch_size, axis=0)
    x = relay.concatenate([cls_tokens, x], axis=1)
    if debug_unit == "post_concat":
        return relay.Function(relay.analysis.free_vars(x), x)

    pos_embed = relay.var("pos_embed", shape=(1, num_tokens, embed_dim), dtype="float32")
    x = x + pos_embed
    x = relay.annotation.stop_fusion(x)
    if debug_unit == "post_addpos":
        return relay.Function(relay.analysis.free_vars(x), x)

    for block_idx in range(depth):
        x = _repq_block(
            x,
            block_idx=block_idx,
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_size=batch_size,
            num_tokens=num_tokens,
        )
        if debug_unit == f"post_block{block_idx}":
            return relay.Function(relay.analysis.free_vars(x), x)

    norm_weight, norm_bias = _layer_norm_var("norm", embed_dim)
    x = layers.layer_norm_float(x, norm_weight, norm_bias)
    x = relay.strided_slice(x, begin=[0, 0, 0], end=[batch_size, 1, embed_dim], strides=[1, 1, 1])
    x = relay.reshape(x, [batch_size, embed_dim])
    if debug_unit == "pre_head":
        return relay.Function(relay.analysis.free_vars(x), x)
    x = layers.exact_uniform_linear(x, "head", add_bias=True)
    if debug_unit == "head_float":
        return relay.Function(relay.analysis.free_vars(x), x)
    return relay.Function(relay.analysis.free_vars(x), x)


def get_repq_deit_tiny_model(data_shape, debug_unit: str | None = None):
    return RepQVisionTransformer(
        data_shape=data_shape,
        patch_size=16,
        num_patches=196,
        num_classes=1000,
        embed_dim=192,
        depth=12,
        num_heads=3,
        debug_unit=debug_unit,
    )
