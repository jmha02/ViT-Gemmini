#!/usr/bin/env python3
"""Export flexi-faithful PTQ4-DeiT ONNX.

Float LN / Softmax / GELU stay mathematically identical to flexi PTQ4ViT, but
are fused into ivit customs to cut ORT dispatch (no RVV in ORT kernels):

  - ivit.FloatLayerNorm
  - ivit.SymQuantizeI8
  - ivit.TwinSoftmaxMatMul  (Softmax + twin + Gemmini MM)
  - ivit.TwinGeluLinear     (Erf-GELU + twin + Gemmini MM)
  - ivit.GemminiMatMulInteger
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPORT_DIR = Path(__file__).resolve().parent
IVIT_DIR = REPO_ROOT / "models" / "ivit"
sys.path.insert(0, str(EXPORT_DIR))
sys.path.insert(0, str(IVIT_DIR))

from export_deit_ivit_onnx import OnnxBuilder  # noqa: E402
from ptq4_checkpoint import (  # noqa: E402
    DEFAULT_PTQ4_DEIT_T_CHECKPOINT,
    load_ptq4_state_dict,
)


def _scalar(sd, key, default=1.0) -> float:
    if key not in sd:
        return float(default)
    return float(sd[key].detach().cpu().numpy().reshape(-1)[0])


def _vec(sd, key) -> np.ndarray:
    return sd[key].detach().cpu().numpy().astype(np.float32).reshape(-1)


def _arr(sd, key, dtype=np.float32):
    return sd[key].detach().cpu().numpy().astype(dtype)


def _q_sym_i8(g: OnnxBuilder, x: str, scale: float, name: str) -> str:
    return g.add_custom(
        "SymQuantizeI8",
        "ivit",
        [x, g.scalar_f32(f"{name}_s", float(scale))],
        [f"{name}_i8"],
    )


def _int_mm_linear(
    g: OnnxBuilder,
    x: str,
    sd,
    prefix: str,
    *,
    in_features: int,
    out_features: int,
    lead_shape: list[int],
    name: str,
) -> str:
    a_interval = _scalar(sd, f"{prefix}.a_interval")
    weight = _arr(sd, f"{prefix}.weight_q_t", np.int8)
    scale = _vec(sd, f"{prefix}.output_scale")
    bias = _vec(sd, f"{prefix}.bias")
    x_i8 = _q_sym_i8(g, x, a_interval, f"{name}_q")
    flat = g.add(
        "Reshape",
        [x_i8, g.init_tensor(f"{name}_rfin", np.array([-1, in_features], dtype=np.int64))],
        [f"{name}_flat"],
    )
    mm = g.gemmini_matmul_integer(flat, weight, f"{name}_mm")
    mm_f = g.add("Cast", [mm], [f"{name}_mm_f"], to=TensorProto.FLOAT)
    scaled = g.add("Mul", [mm_f, g.init_tensor(f"{name}_scale", scale)], [f"{name}_scaled"])
    y = g.add("Add", [scaled, g.init_tensor(f"{name}_bias", bias)], [f"{name}_y"])
    out_shape = np.array(list(lead_shape) + [out_features], dtype=np.int64)
    return g.add("Reshape", [y, g.init_tensor(f"{name}_rfout", out_shape)], [f"{name}_out"])


def _layer_norm(g: OnnxBuilder, x: str, sd, prefix: str, dim: int, name: str) -> str:
    """Float LayerNorm via fused ivit.FloatLayerNorm (same math as ReduceMean chain)."""
    w = _vec(sd, f"{prefix}.weight")
    b = _vec(sd, f"{prefix}.bias")
    return g.add_custom(
        "FloatLayerNorm",
        "ivit",
        [
            x,
            g.init_tensor(f"{name}_w", w),
            g.init_tensor(f"{name}_b", b),
            g.scalar_f32(f"{name}_eps", 1e-5),
        ],
        [f"{name}_out"],
    )


def _gelu(g: OnnxBuilder, x: str, name: str) -> str:
    # Kept for reference; MLP path absorbs Erf-GELU into TwinGeluLinear.
    inv_sqrt2 = g.scalar_f32(f"{name}_is2", float(np.sqrt(0.5)))
    half = g.scalar_f32(f"{name}_half", 0.5)
    one = g.scalar_f32(f"{name}_one", 1.0)
    scaled = g.add("Mul", [x, inv_sqrt2], [f"{name}_scaled"])
    erf = g.add("Erf", [scaled], [f"{name}_erf"])
    term = g.add("Add", [one, erf], [f"{name}_term"])
    half_x = g.add("Mul", [half, x], [f"{name}_hx"])
    return g.add("Mul", [half_x, term], [f"{name}_out"])


def _qkv_split(g: OnnxBuilder, x: str, sd, prefix: str, *, dim: int, num_heads: int, seq: int, name: str):
    head_dim = dim // num_heads
    a_interval = _scalar(sd, f"{prefix}.a_interval")
    x_i8 = _q_sym_i8(g, x, a_interval, f"{name}_q")
    x_flat = g.add("Reshape", [x_i8, g.init_tensor(f"{name}_rs", np.array([-1, dim], dtype=np.int64))], [f"{name}_flat"])
    blocks = []
    for gi in range(3 * num_heads):
        w = _arr(sd, f"{prefix}.w{gi}", np.int8)
        d = _arr(sd, f"{prefix}.d{gi}", np.int32)
        s = _scalar(sd, f"{prefix}.s{gi}")
        mm = g.gemmini_matmul_integer(x_flat, w, f"{name}_w{gi}")
        mm_b = g.add("Add", [mm, g.init_tensor(f"{name}_d{gi}", d)], [f"{name}_acc{gi}"])
        mm_f = g.add("Cast", [mm_b], [f"{name}_f{gi}"], to=TensorProto.FLOAT)
        scaled = g.add("Mul", [mm_f, g.scalar_f32(f"{name}_s{gi}", s)], [f"{name}_sc{gi}"])
        clipped = g.add(
            "Clip",
            [g.add("Round", [scaled], [f"{name}_r{gi}"]), g.scalar_f32(f"{name}_lo{gi}", -128.0), g.scalar_f32(f"{name}_hi{gi}", 127.0)],
            [f"{name}_c{gi}"],
        )
        qi = g.add("Cast", [clipped], [f"{name}_i8{gi}"], to=TensorProto.INT8)
        blocks.append(g.add("Reshape", [qi, g.init_tensor(f"{name}_brs{gi}", np.array([1, seq, head_dim], dtype=np.int64))], [f"{name}_blk{gi}"]))

    # Stack manually via Concat + Reshape → [3,H,B,N,Hd] then Gather
    cat = g.add("Concat", blocks, [f"{name}_cat"], axis=0)  # [3H, B=1, N, Hd] wait blocks are [1,seq,hd]
    # each block [1, seq, hd]; concat axis0 → [3H, seq, hd] if we squeeze batch... 
    # Our reshape used [1, seq, hd]. Concat axis0 → [3H, seq, hd]
    stacked = g.add(
        "Reshape",
        [cat, g.init_tensor(f"{name}_st", np.array([3, num_heads, 1, seq, head_dim], dtype=np.int64))],
        [f"{name}_stacked"],
    )
    # Take Q/K/V and transpose to [B,H,N,Hd]
    outs = []
    for p, tag in enumerate(["q", "k", "v"]):
        part = g.add("Gather", [stacked, g.init_tensor(f"{name}_{tag}_idx", np.array(p, dtype=np.int64))], [f"{name}_{tag}_raw"], axis=0)
        # [H,1,N,Hd] → [1,H,N,Hd]
        outs.append(g.add("Transpose", [part], [f"{name}_{tag}"], perm=[1, 0, 2, 3]))
    return outs[0], outs[1], outs[2]


def _qk_matmul(g: OnnxBuilder, q: str, k: str, sd, prefix: str, *, num_heads: int, seq: int, head_dim: int, name: str) -> str:
    scale = _vec(sd, f"{prefix}.output_scale")
    q3 = g.add(
        "Reshape",
        [q, g.init_tensor(f"{name}_qrs", np.array([num_heads, seq, head_dim], dtype=np.int64))],
        [f"{name}_q3"],
    )
    k3 = g.add(
        "Reshape",
        [k, g.init_tensor(f"{name}_krs", np.array([num_heads, seq, head_dim], dtype=np.int64))],
        [f"{name}_k3"],
    )
    scores = []
    for i in range(num_heads):
        # Gather (not Split-13) — onnxruntime-riscv lacks Split(13).
        qi = g.add(
            "Gather",
            [q3, g.init_tensor(f"{name}_qi_idx{i}", np.array(i, dtype=np.int64))],
            [f"{name}_qi{i}"],
            axis=0,
        )
        ki = g.add(
            "Gather",
            [k3, g.init_tensor(f"{name}_ki_idx{i}", np.array(i, dtype=np.int64))],
            [f"{name}_ki{i}"],
            axis=0,
        )
        kt_i = g.add("Transpose", [ki], [f"{name}_kti{i}"], perm=[1, 0])
        mm = g.gemmini_matmul_integer_inputs(qi, kt_i, f"{name}_mm{i}")
        mm_f = g.add("Cast", [mm], [f"{name}_mmf{i}"], to=TensorProto.FLOAT)
        sc = g.add("Mul", [mm_f, g.scalar_f32(f"{name}_sc{i}", float(scale[i]))], [f"{name}_scs{i}"])
        # Reshape instead of Unsqueeze (opset-13 Unsqueeze needs axes input; riscv ORT is picky).
        scores.append(
            g.add(
                "Reshape",
                [sc, g.init_tensor(f"{name}_su_rs{i}", np.array([1, seq, seq], dtype=np.int64))],
                [f"{name}_su{i}"],
            )
        )
    cat = g.add("Concat", scores, [f"{name}_cat"], axis=0)  # [H, N, N]
    return g.add(
        "Reshape",
        [cat, g.init_tensor(f"{name}_out_rs", np.array([1, num_heads, seq, seq], dtype=np.int64))],
        [f"{name}_out"],
    )


def _attention(g: OnnxBuilder, x: str, sd, prefix: str, *, dim: int, num_heads: int, seq: int, name: str) -> str:
    head_dim = dim // num_heads
    q, k, v = _qkv_split(g, x, sd, f"{prefix}.attn.qkv", dim=dim, num_heads=num_heads, seq=seq, name=f"{name}_qkv")
    scores = _qk_matmul(g, q, k, sd, f"{prefix}.attn.matmul1", num_heads=num_heads, seq=seq, head_dim=head_dim, name=f"{name}_mm1")
    scale = float(head_dim ** -0.5)
    # TwinSoftmaxMatMul absorbs Softmax(+scale) + twin quant + 2x Gemmini MM.
    split = _scalar(sd, f"{prefix}.attn.matmul2.split")
    a_interval = _scalar(sd, f"{prefix}.attn.matmul2.a_interval")
    b_interval = _vec(sd, f"{prefix}.attn.matmul2.b_interval")
    ctx = g.add_custom(
        "TwinSoftmaxMatMul",
        "ivit",
        [
            scores,
            v,
            g.scalar_f32(f"{name}_split", split),
            g.scalar_f32(f"{name}_ai", a_interval),
            g.init_tensor(f"{name}_bi", b_interval),
            g.scalar_f32(f"{name}_sm_s", scale),
        ],
        [f"{name}_ctx"],
    )
    ctx_t = g.add("Transpose", [ctx], [f"{name}_ctxt"], perm=[0, 2, 1, 3])
    ctx_r = g.add("Reshape", [ctx_t, g.init_tensor(f"{name}_crs", np.array([1, seq, dim], dtype=np.int64))], [f"{name}_ctxr"])
    return _int_mm_linear(
        g,
        ctx_r,
        sd,
        f"{prefix}.attn.proj",
        in_features=dim,
        out_features=dim,
        lead_shape=[1, seq],
        name=f"{name}_proj",
    )


def _mlp(g: OnnxBuilder, x: str, sd, prefix: str, *, dim: int, mlp_ratio: int, seq: int, name: str) -> str:
    hidden = dim * mlp_ratio
    h = _int_mm_linear(
        g,
        x,
        sd,
        f"{prefix}.mlp.fc1",
        in_features=dim,
        out_features=hidden,
        lead_shape=[1, seq],
        name=f"{name}_fc1",
    )
    # TwinGeluLinear absorbs Erf-GELU + twin quant + 2x Gemmini MM.
    w = _arr(sd, f"{prefix}.mlp.fc2.weight_q_t", np.int8)
    return g.add_custom(
        "TwinGeluLinear",
        "ivit",
        [
            h,
            g.init_tensor(f"{name}_w", w),
            g.scalar_f32(f"{name}_ai", _scalar(sd, f"{prefix}.mlp.fc2.a_interval")),
            g.scalar_f32(f"{name}_an", _scalar(sd, f"{prefix}.mlp.fc2.a_neg_interval")),
            g.init_tensor(f"{name}_ws", _vec(sd, f"{prefix}.mlp.fc2.weight_scale")),
            g.init_tensor(f"{name}_b", _vec(sd, f"{prefix}.mlp.fc2.bias")),
        ],
        [f"{name}_fc2"],
    )


def _block(g: OnnxBuilder, x: str, sd, bi: int, *, dim: int, num_heads: int, seq: int, mlp_ratio: int) -> str:
    p = f"blocks.{bi}"
    n1 = _layer_norm(g, x, sd, f"{p}.norm1", dim, f"b{bi}_n1")
    a = _attention(g, n1, sd, p, dim=dim, num_heads=num_heads, seq=seq, name=f"b{bi}_attn")
    x = g.add("Add", [x, a], [f"b{bi}_res1"])
    n2 = _layer_norm(g, x, sd, f"{p}.norm2", dim, f"b{bi}_n2")
    m = _mlp(g, n2, sd, p, dim=dim, mlp_ratio=mlp_ratio, seq=seq, name=f"b{bi}_mlp")
    return g.add("Add", [x, m], [f"b{bi}_out"])


def _patch_embed(g: OnnxBuilder, x: str, sd, *, embed_dim: int, name: str = "patch") -> str:
    # x: [1,3,224,224] → patches [1,196,768]
    # Use Reshape/Transpose matching flexi PatchEmbed
    a_interval = _scalar(sd, "patch_embed.a_interval")
    weight = _arr(sd, "patch_embed.weight_q_t", np.int8)
    wscale = _vec(sd, "patch_embed.weight_scale")
    bias = _vec(sd, "patch_embed.bias")
    # [1,3,14,16,14,16] -> [1,14,14,3,16,16] -> [1,196,768]
    r1 = g.add("Reshape", [x, g.init_tensor(f"{name}_r1", np.array([1, 3, 14, 16, 14, 16], dtype=np.int64))], [f"{name}_r1o"])
    t1 = g.add("Transpose", [r1], [f"{name}_t1"], perm=[0, 2, 4, 1, 3, 5])
    patches = g.add("Reshape", [t1, g.init_tensor(f"{name}_r2", np.array([1, 196, 768], dtype=np.int64))], [f"{name}_pat"])
    x_i8 = _q_sym_i8(g, patches, a_interval, f"{name}_q")
    flat = g.add("Reshape", [x_i8, g.init_tensor(f"{name}_rf", np.array([-1, 768], dtype=np.int64))], [f"{name}_flat"])
    mm = g.gemmini_matmul_integer(flat, weight, f"{name}_mm")
    mm_f = g.add("Cast", [mm], [f"{name}_mmf"], to=TensorProto.FLOAT)
    out_scale = (a_interval * wscale).astype(np.float32)
    scaled = g.add("Mul", [mm_f, g.init_tensor(f"{name}_os", out_scale)], [f"{name}_sc"])
    y = g.add("Add", [scaled, g.init_tensor(f"{name}_b", bias)], [f"{name}_y"])
    return g.add("Reshape", [y, g.init_tensor(f"{name}_ro", np.array([1, 196, embed_dim], dtype=np.int64))], [f"{name}_out"])


def build_ptq4_onnx(sd, *, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4) -> onnx.ModelProto:
    g = OnnxBuilder()
    seq = 197
    x = "image"
    pe = _patch_embed(g, x, sd, embed_dim=embed_dim)
    cls = g.init_tensor("cls_token", _arr(sd, "cls_token", np.float32))
    pos = g.init_tensor("pos_embed", _arr(sd, "pos_embed", np.float32))
    tokens = g.add("Concat", [cls, pe], ["tokens"], axis=1)
    x = g.add("Add", [tokens, pos], ["embed"])
    for bi in range(depth):
        x = _block(g, x, sd, bi, dim=embed_dim, num_heads=num_heads, seq=seq, mlp_ratio=mlp_ratio)
    x = _layer_norm(g, x, sd, "norm", embed_dim, "final_norm")
    # Gather cls token (avoid Slice-10 quirks on onnxruntime-riscv).
    cls_raw = g.add(
        "Gather",
        [x, g.init_tensor("cls_idx", np.array(0, dtype=np.int64))],
        ["cls_tok_raw"],
        axis=1,
    )  # [1, embed]
    cls_flat = g.add("Reshape", [cls_raw, g.init_tensor("cls_rs", np.array([1, embed_dim], dtype=np.int64))], ["cls_flat"])
    logits = _int_mm_linear(
        g,
        cls_flat,
        sd,
        "head",
        in_features=embed_dim,
        out_features=1000,
        lead_shape=[1],
        name="head",
    )

    inputs = [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 224, 224])]
    outputs = [helper.make_tensor_value_info(logits, TensorProto.FLOAT, [1, 1000])]
    graph = helper.make_graph(g.nodes, "ptq4_deit", inputs, outputs, g.initializers, value_info=g.value_infos)
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 13),
            helper.make_opsetid("ivit", 1),
        ],
    )
    # onnxruntime-riscv rejects IR > 7 ("Unknown model file format version").
    model.ir_version = 7
    onnx.checker.check_model(model, full_check=False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_PTQ4_DEIT_T_CHECKPOINT)
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "build/ort/ptq4_deit_tiny_int8.onnx")
    args = ap.parse_args()
    sd = load_ptq4_state_dict(args.checkpoint)
    model = build_ptq4_onnx(sd)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(f"wrote {args.output} nodes={len(model.graph.node)}")


if __name__ == "__main__":
    main()
