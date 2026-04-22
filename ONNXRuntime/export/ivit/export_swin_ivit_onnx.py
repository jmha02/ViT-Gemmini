#!/usr/bin/env python3
"""
Swin-Tiny (I-ViT quantized) -> ONNX INT8 graph builder (DeiT-style custom-op flow).

This exporter builds a graph that uses:
- Gemmini-friendly INT8 ops: QLinearConv, ivit.GemminiMatMulInteger
- quantized add: com.microsoft.QLinearAdd
- I-ViT custom ops (domain=ivit): QLayernorm, Shiftmax, ShiftGELU

The intent is to keep Swin on the same export style as DeiT I-ViT.
Model depth is fixed to Swin-Tiny baseline: (2, 2, 6, 2).
"""

from __future__ import annotations

import argparse
import math
import sys
from functools import partial
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


SCRIPT_DIR = Path(__file__).resolve().parent
ONNXRT_DIR = SCRIPT_DIR.parent.parent
REPO_ROOT = ONNXRT_DIR.parent
TOOLS_DIR = ONNXRT_DIR / "tools"
DEFAULT_OUTPUT = REPO_ROOT / "build" / "ort" / "swin_tiny_int8.onnx"
DEFAULT_IMAGE = REPO_ROOT / "scripts" / "gemmini" / "test_cat.jpg"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from export_deit_ivit_onnx import (  # noqa: E402
    OnnxBuilder,
    get_scale,
    get_weight,
    to_int64,
    to_int32,
)
from ivit_model_io import load_checkpoint_into_model, load_checkpoint_state_dict


IMG_SIZE = 224
PATCH_SIZE = 4
WINDOW_SIZE = 7
IN_CHANNELS = 3
EMBED_DIM = 96
NUM_CLASSES = 1000
BATCH = 1
STAGE_HEADS = (3, 6, 12, 24)
SWIN_DEPTHS = (2, 2, 6, 2)
DEAD_CHANNEL_THRESHOLD = 0.01


def _fix_norm_bias_and_scale(sd: dict, weight_key: str, bias_key: str, sf_key: str) -> tuple[np.ndarray, np.ndarray]:
    weight = get_weight(sd, weight_key).reshape(-1).astype(np.float32)
    bias_int = to_int64(get_weight(sd, bias_key)).reshape(-1)
    norm_sf = np.asarray(get_scale(sd, sf_key), dtype=np.float32).reshape(-1)

    dead_mask = np.abs(weight) < DEAD_CHANNEL_THRESHOLD
    int64_info = np.iinfo(np.int64)
    bad_bias_mask = (~np.isfinite(bias_int)) | (bias_int < int64_info.min) | (bias_int > int64_info.max)
    bad_sf_mask = (~np.isfinite(norm_sf)) | (norm_sf == 0)
    fix_mask = dead_mask & (bad_bias_mask | bad_sf_mask)
    if not np.any(fix_mask):
        return bias_int.astype(np.int64), norm_sf.astype(np.float32)

    valid_mask = ~fix_mask
    if np.any(valid_mask):
        mean_sf = float(np.mean(np.abs(norm_sf[valid_mask])))
    else:
        mean_sf = float(np.mean(np.abs(norm_sf)))
    if mean_sf == 0.0:
        mean_sf = 1.0

    fixed_bias = bias_int.copy()
    fixed_bias[fix_mask] = 0

    fixed_sf = norm_sf.copy()
    signs = np.sign(fixed_sf[fix_mask])
    signs[signs == 0] = 1.0
    fixed_sf[fix_mask] = mean_sf * signs
    return fixed_bias.astype(np.int64), fixed_sf.astype(np.float32)


def _prepare_swin_state_dict(
    checkpoint: Path | None,
    allow_random_init: bool,
    depths: tuple[int, int, int, int],
) -> dict:
    import torch

    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        print(f"Loaded Swin checkpoint state_dict directly: {checkpoint}")
        return load_checkpoint_state_dict(checkpoint)

    sys.path.insert(0, str(REPO_ROOT / "I-ViT"))
    from models.model_utils import freeze_model
    from models.quantization_utils import IntLayerNorm
    from models.swin_quant import SwinTransformer

    model = SwinTransformer(
        patch_size=PATCH_SIZE,
        window_size=WINDOW_SIZE,
        embed_dim=EMBED_DIM,
        depths=depths,
        num_heads=STAGE_HEADS,
        norm_layer=partial(IntLayerNorm, eps=1e-6),
    )
    model.eval()

    if not allow_random_init:
        raise RuntimeError("Swin export requires --checkpoint (or --allow-random-init)")
    print("[WARN] Exporting Swin with random-initialized weights.")

    freeze_model(model)
    with torch.no_grad():
        model(torch.randn(BATCH, IN_CHANNELS, IMG_SIZE, IMG_SIZE))

    return model.state_dict()


def _slice_static(
    g: OnnxBuilder,
    x: str,
    starts: list[int],
    ends: list[int],
    axes: list[int],
    name_prefix: str,
    steps: list[int] | None = None,
) -> str:
    s_n = g.init_tensor(f"{name_prefix}_starts", np.array(starts, dtype=np.int64))
    e_n = g.init_tensor(f"{name_prefix}_ends", np.array(ends, dtype=np.int64))
    a_n = g.init_tensor(f"{name_prefix}_axes", np.array(axes, dtype=np.int64))
    if steps is None:
        out = f"{name_prefix}_out"
        g.add("Slice", [x, s_n, e_n, a_n], [out])
        return out
    st_n = g.init_tensor(f"{name_prefix}_steps", np.array(steps, dtype=np.int64))
    out = f"{name_prefix}_out"
    g.add("Slice", [x, s_n, e_n, a_n, st_n], [out])
    return out


def _roll_nhwc(
    g: OnnxBuilder,
    x: str,
    h: int,
    w: int,
    c: int,
    shift_h: int,
    shift_w: int,
    name_prefix: str,
) -> str:
    out = x

    if shift_h != 0:
        if shift_h < 0:
            s = (-shift_h) % h
            part1 = _slice_static(g, out, [0, s, 0, 0], [BATCH, h, w, c], [0, 1, 2, 3], f"{name_prefix}_h_neg_p1")
            part2 = _slice_static(g, out, [0, 0, 0, 0], [BATCH, s, w, c], [0, 1, 2, 3], f"{name_prefix}_h_neg_p2")
            rolled = f"{name_prefix}_h_neg_roll"
            g.add("Concat", [part1, part2], [rolled], axis=1)
            out = rolled
        else:
            s = shift_h % h
            part1 = _slice_static(g, out, [0, h - s, 0, 0], [BATCH, h, w, c], [0, 1, 2, 3], f"{name_prefix}_h_pos_p1")
            part2 = _slice_static(g, out, [0, 0, 0, 0], [BATCH, h - s, w, c], [0, 1, 2, 3], f"{name_prefix}_h_pos_p2")
            rolled = f"{name_prefix}_h_pos_roll"
            g.add("Concat", [part1, part2], [rolled], axis=1)
            out = rolled

    if shift_w != 0:
        if shift_w < 0:
            s = (-shift_w) % w
            part1 = _slice_static(g, out, [0, 0, s, 0], [BATCH, h, w, c], [0, 1, 2, 3], f"{name_prefix}_w_neg_p1")
            part2 = _slice_static(g, out, [0, 0, 0, 0], [BATCH, h, s, c], [0, 1, 2, 3], f"{name_prefix}_w_neg_p2")
            rolled = f"{name_prefix}_w_neg_roll"
            g.add("Concat", [part1, part2], [rolled], axis=2)
            out = rolled
        else:
            s = shift_w % w
            part1 = _slice_static(g, out, [0, 0, w - s, 0], [BATCH, h, w, c], [0, 1, 2, 3], f"{name_prefix}_w_pos_p1")
            part2 = _slice_static(g, out, [0, 0, 0, 0], [BATCH, h, w - s, c], [0, 1, 2, 3], f"{name_prefix}_w_pos_p2")
            rolled = f"{name_prefix}_w_pos_roll"
            g.add("Concat", [part1, part2], [rolled], axis=2)
            out = rolled

    return out


def _tokens_to_nhwc(g: OnnxBuilder, x: str, h: int, w: int, c: int, name_prefix: str) -> str:
    shape_n = g.init_tensor(f"{name_prefix}_shape", np.array([BATCH, h, w, c], dtype=np.int64))
    out = f"{name_prefix}_out"
    g.add("Reshape", [x, shape_n], [out])
    return out


def _nhwc_to_tokens(g: OnnxBuilder, x: str, h: int, w: int, c: int, name_prefix: str) -> str:
    shape_n = g.init_tensor(f"{name_prefix}_shape", np.array([BATCH, h * w, c], dtype=np.int64))
    out = f"{name_prefix}_out"
    g.add("Reshape", [x, shape_n], [out])
    return out


def _window_partition(
    g: OnnxBuilder,
    x_nhwc: str,
    h: int,
    w: int,
    c: int,
    window: int,
    name_prefix: str,
) -> tuple[str, int]:
    n_w = (h // window) * (w // window)
    shape1 = g.init_tensor(
        f"{name_prefix}_shape1",
        np.array([BATCH, h // window, window, w // window, window, c], dtype=np.int64),
    )
    r1 = f"{name_prefix}_r1"
    g.add("Reshape", [x_nhwc, shape1], [r1])
    t1 = f"{name_prefix}_t1"
    g.add("Transpose", [r1], [t1], perm=[0, 1, 3, 2, 4, 5])
    shape2 = g.init_tensor(
        f"{name_prefix}_shape2",
        np.array([BATCH * n_w, window * window, c], dtype=np.int64),
    )
    out = f"{name_prefix}_out"
    g.add("Reshape", [t1, shape2], [out])
    return out, n_w


def _window_reverse(
    g: OnnxBuilder,
    x_windows: str,
    h: int,
    w: int,
    c: int,
    window: int,
    n_w: int,
    name_prefix: str,
) -> str:
    _ = n_w  # static by shape.
    shape1 = g.init_tensor(
        f"{name_prefix}_shape1",
        np.array([BATCH, h // window, w // window, window, window, c], dtype=np.int64),
    )
    r1 = f"{name_prefix}_r1"
    g.add("Reshape", [x_windows, shape1], [r1])
    t1 = f"{name_prefix}_t1"
    g.add("Transpose", [r1], [t1], perm=[0, 1, 3, 2, 4, 5])
    shape2 = g.init_tensor(
        f"{name_prefix}_shape2",
        np.array([BATCH, h, w, c], dtype=np.int64),
    )
    out = f"{name_prefix}_out"
    g.add("Reshape", [t1, shape2], [out])
    return out


def _dynamic_gemmini_matmul(
    g: OnnxBuilder,
    a: str,
    b: str,
    name_prefix: str,
) -> str:
    return g.gemmini_matmul_integer_inputs(a, b, name_prefix)


def _collapse_conv_scales(
    w_int8: np.ndarray,
    b_int32: np.ndarray,
    w_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    if w_scales.size == 1:
        return w_int8, b_int32, float(w_scales[0])

    scalar = float(np.max(np.abs(w_scales)))
    if scalar == 0.0:
        scalar = 1.0
    w_float = w_int8.astype(np.float32) * w_scales.reshape(-1, 1, 1, 1)
    w_new = np.clip(np.round(w_float / scalar), -128, 127).astype(np.int8)
    b_new = np.round(b_int32.astype(np.float32) * (w_scales / scalar)).astype(np.int32)
    return w_new, b_new, scalar


def _build_swin_ivit_graph(
    sd: dict,
    depths: tuple[int, int, int, int],
    patch_embed_op: str,
) -> onnx.ModelProto:
    g = OnnxBuilder()

    patch_grid = IMG_SIZE // PATCH_SIZE  # 56
    cur_h = patch_grid
    cur_w = patch_grid
    cur_c = EMBED_DIM

    # Input quantization
    input_scale = get_scale(sd, "qact_input.act_scaling_factor")
    x = g.quantize("image", input_scale, "input")

    # Exact patch embedding via Gemmini matmul preserves per-channel conv scales.
    # The systolic QLinearConv kernel only accepts scalar scales, so the conv path
    # collapses the checkpoint's per-channel conv scales to one shared scale.
    conv_w = get_weight(sd, "patch_embed.proj.weight_integer").astype(np.int8)
    conv_b = to_int32(get_weight(sd, "patch_embed.proj.bias_integer")).reshape(-1)
    conv_w_sc_raw = np.asarray(get_scale(sd, "patch_embed.proj.conv_scaling_factor"), dtype=np.float32).reshape(-1)
    patch_qact_scale = get_scale(sd, "patch_embed.qact.act_scaling_factor")
    patch_before_norm_scale = get_scale(sd, "patch_embed.qact_before_norm.act_scaling_factor") if "patch_embed.norm.bias_integer" in sd else patch_qact_scale
    if patch_embed_op == "matmul":
        x = g.patch_embed_matmul(
            x,
            input_scale,
            conv_w,
            conv_w_sc_raw,
            conv_b,
            patch_before_norm_scale,
            "int8",
            PATCH_SIZE,
            IMG_SIZE,
            "patch_embed",
        )
    elif patch_embed_op == "qlinearconv":
        conv_w_qc, conv_b_qc, conv_w_scale = _collapse_conv_scales(conv_w, conv_b, conv_w_sc_raw)
        x = g.qlinear_conv(
            x,
            input_scale,
            conv_w_qc,
            conv_w_scale,
            conv_b_qc,
            patch_before_norm_scale,
            "patch_embed",
            kernel_shape=(PATCH_SIZE, PATCH_SIZE),
            strides=(PATCH_SIZE, PATCH_SIZE),
        )
        patch_embed_t = "patch_embed_to_tokens_t"
        g.add("Transpose", [x], [patch_embed_t], perm=[0, 2, 3, 1])
        patch_embed_shape = g.init_tensor(
            "patch_embed_to_tokens_shape",
            np.array([BATCH, cur_h * cur_w, cur_c], dtype=np.int64),
        )
        patch_embed_tokens = "patch_embed_to_tokens"
        g.add("Reshape", [patch_embed_t, patch_embed_shape], [patch_embed_tokens])
        x = patch_embed_tokens
    else:
        raise ValueError(f"Unsupported patch_embed_op: {patch_embed_op}")
    cur_scale = patch_before_norm_scale
    cur_dtype = "int8"

    # Optional patch norm (used by Swin-T default)
    if "patch_embed.norm.bias_integer" in sd:
        ln_bias, ln_sf = _fix_norm_bias_and_scale(
            sd,
            "patch_embed.norm.weight",
            "patch_embed.norm.bias_integer",
            "patch_embed.norm.norm_scaling_factor",
        )
        ln = g.q_layernorm(x, ln_bias, "patch_norm", input_dtype="int8")
        x = g.requantize_from_int64(ln, ln_sf, patch_qact_scale, "int16", "patch_norm_req")
        cur_scale = patch_qact_scale
        cur_dtype = "int16"

    # Top qact1
    top_qact1_scale = get_scale(sd, "qact1.act_scaling_factor")
    if "absolute_pos_embed" in sd:
        pos_scale = get_scale(sd, "qact_pos.act_scaling_factor")
        pos_f = get_weight(sd, "absolute_pos_embed").astype(np.float32)
        pos_i8 = np.clip(np.round(pos_f / pos_scale), -128, 127).astype(np.int8)
        pos_name = g.init_tensor("absolute_pos_embed_i8", pos_i8)
        x = g.add_float_int16(x, cur_scale, pos_name, pos_scale, top_qact1_scale, "top_pos_add")
    else:
        if cur_dtype == "int16":
            x = g.requantize_int16(x, cur_scale, top_qact1_scale, "top_qact1_req")
        else:
            x = g.requantize_int8_to_int16(x, cur_scale, top_qact1_scale, "top_qact1_req")
    cur_scale = top_qact1_scale
    cur_dtype = "int16"

    for stage_idx, stage_depth in enumerate(depths):
        num_heads = STAGE_HEADS[stage_idx]
        head_dim = cur_c // num_heads

        for blk_idx in range(stage_depth):
            p = f"layers.{stage_idx}.blocks.{blk_idx}"

            # Norm1 -> qact1
            n1_bias, n1_sf = _fix_norm_bias_and_scale(
                sd,
                f"{p}.norm1.weight",
                f"{p}.norm1.bias_integer",
                f"{p}.norm1.norm_scaling_factor",
            )
            ln1 = g.q_layernorm(x, n1_bias, f"{p}_norm1", input_dtype=cur_dtype)
            blk_qact1_scale = get_scale(sd, f"{p}.qact1.act_scaling_factor")
            ln1_req = g.requantize_from_int64(ln1, n1_sf, blk_qact1_scale, "int8", f"{p}_norm1_req")

            # Token -> NHWC -> (optional shift) -> windows
            x_img = _tokens_to_nhwc(g, ln1_req, cur_h, cur_w, cur_c, f"{p}_tok2img")
            shift = WINDOW_SIZE // 2 if (blk_idx % 2 == 1 and min(cur_h, cur_w) > WINDOW_SIZE) else 0
            if shift:
                x_img = _roll_nhwc(g, x_img, cur_h, cur_w, cur_c, -shift, -shift, f"{p}_shift")
            x_win, n_w = _window_partition(g, x_img, cur_h, cur_w, cur_c, WINDOW_SIZE, f"{p}_winpart")

            # QKV projection (GemminiMatMulInteger path, DeiT-style)
            qkv_w = get_weight(sd, f"{p}.attn.qkv.weight_integer").astype(np.int8).T
            qkv_b = to_int32(get_weight(sd, f"{p}.attn.qkv.bias_integer")).reshape(-1)
            qkv_ws = get_scale(sd, f"{p}.attn.qkv.fc_scaling_factor")
            attn_qact1_scale = get_scale(sd, f"{p}.attn.qact1.act_scaling_factor")
            qkv = g.qlinear_matmul_bias(
                x_win,
                blk_qact1_scale,
                qkv_w,
                qkv_ws,
                qkv_b,
                attn_qact1_scale,
                "int8",
                f"{p}_qkv",
            )

            qkv_shape = g.init_tensor(
                f"{p}_qkv_shape",
                np.array([BATCH * n_w, WINDOW_SIZE * WINDOW_SIZE, 3, num_heads, head_dim], dtype=np.int64),
            )
            qkv_5d = f"{p}_qkv_5d"
            g.add("Reshape", [qkv, qkv_shape], [qkv_5d])
            qkv_t = f"{p}_qkv_t"
            g.add("Transpose", [qkv_5d], [qkv_t], perm=[2, 0, 3, 1, 4])  # [3, nW, H, N, d]

            idx0 = g.init_tensor(f"{p}_idx0", np.array(0, dtype=np.int64))
            idx1 = g.init_tensor(f"{p}_idx1", np.array(1, dtype=np.int64))
            idx2 = g.init_tensor(f"{p}_idx2", np.array(2, dtype=np.int64))
            q = f"{p}_q"
            k = f"{p}_k"
            v = f"{p}_v"
            g.add("Gather", [qkv_t, idx0], [q], axis=0)
            g.add("Gather", [qkv_t, idx1], [k], axis=0)
            g.add("Gather", [qkv_t, idx2], [v], axis=0)

            # Attn logits
            kt = f"{p}_kt"
            g.add("Transpose", [k], [kt], perm=[0, 1, 3, 2])
            matmul1_scale = get_scale(sd, f"{p}.attn.matmul_1.act_scaling_factor")
            attn_scores = _dynamic_gemmini_matmul(g, q, kt, f"{p}_attn_scores")

            attn_logits_scale = float(matmul1_scale) * (head_dim ** -0.5)
            attn_qact_attn1_scale = get_scale(sd, f"{p}.attn.qact_attn1.act_scaling_factor")
            attn_req = g.requantize_int32(attn_scores, attn_logits_scale, attn_qact_attn1_scale, f"{p}_attn_req")

            # Relative position bias (quantized table + broadcast add)
            rel_table = get_weight(sd, f"{p}.attn.relative_position_bias_table").astype(np.float32)
            rel_index = get_weight(sd, f"{p}.attn.relative_position_index").astype(np.int64).reshape(-1)
            rel_bias = rel_table[rel_index].reshape(WINDOW_SIZE * WINDOW_SIZE, WINDOW_SIZE * WINDOW_SIZE, num_heads)
            rel_bias = rel_bias.transpose(2, 0, 1)  # [heads, 49, 49]
            rel_table_scale = get_scale(sd, f"{p}.attn.qact_table.act_scaling_factor")
            rel_bias_i8 = np.clip(np.round(rel_bias / rel_table_scale), -128, 127).astype(np.int8)
            rel_bias_name = g.init_tensor(
                f"{p}_rel_bias_i8",
                rel_bias_i8.reshape(1, num_heads, WINDOW_SIZE * WINDOW_SIZE, WINDOW_SIZE * WINDOW_SIZE),
            )

            attn_qact2_scale = get_scale(sd, f"{p}.attn.qact2.act_scaling_factor")
            attn_bias_added = g.add_float_int32(
                attn_req,
                attn_qact_attn1_scale,
                rel_bias_name,
                rel_table_scale,
                attn_qact2_scale,
                f"{p}_attn_bias_add",
            )

            # Shifted-window attention mask (only for odd blocks in larger stages)
            if shift and f"{p}.attn_mask" in sd:
                attn_mask = get_weight(sd, f"{p}.attn_mask").astype(np.float32)  # [nW, 49, 49]
                attn_mask_i8 = np.round(attn_mask / attn_qact2_scale).astype(np.int32)
                mask_name = g.init_tensor(
                    f"{p}_attn_mask_i32",
                    attn_mask_i8.reshape(n_w, 1, WINDOW_SIZE * WINDOW_SIZE, WINDOW_SIZE * WINDOW_SIZE),
                )
                attn_mask_added = f"{p}_attn_mask_add"
                g.add("Add", [attn_bias_added, mask_name], [attn_mask_added])
                attn_bias_added = attn_mask_added

            # Shiftmax custom op
            attn_softmax = g.shiftmax(attn_bias_added, attn_qact2_scale, f"{p}_softmax", input_dtype="int32")

            # Attn @ V
            matmul2_scale = get_scale(sd, f"{p}.attn.matmul_2.act_scaling_factor")
            attn_v = _dynamic_gemmini_matmul(g, attn_softmax, v, f"{p}_attn_v")
            attn_qact3_scale = get_scale(sd, f"{p}.attn.qact3.act_scaling_factor")
            attn_v_req = g.requantize_int32(attn_v, matmul2_scale, attn_qact3_scale, f"{p}_attn_v_req")
            attn_v_perm = f"{p}_attn_v_perm"
            g.add("Transpose", [attn_v_req], [attn_v_perm], perm=[0, 2, 1, 3])
            attn_flat_shape = g.init_tensor(
                f"{p}_attn_flat_shape",
                np.array([BATCH * n_w, WINDOW_SIZE * WINDOW_SIZE, cur_c], dtype=np.int64),
            )
            attn_flat = f"{p}_attn_flat"
            g.add("Reshape", [attn_v_perm, attn_flat_shape], [attn_flat])

            # Projection
            proj_w = get_weight(sd, f"{p}.attn.proj.weight_integer").astype(np.int8).T
            proj_b = to_int32(get_weight(sd, f"{p}.attn.proj.bias_integer")).reshape(-1)
            proj_ws = get_scale(sd, f"{p}.attn.proj.fc_scaling_factor")
            attn_qact4_scale = get_scale(sd, f"{p}.attn.qact4.act_scaling_factor")
            proj = g.qlinear_matmul_bias(
                attn_flat,
                attn_qact3_scale,
                proj_w,
                proj_ws,
                proj_b,
                attn_qact4_scale,
                "int16",
                f"{p}_proj",
            )

            # Windows -> image -> (reverse shift) -> tokens
            proj_img = _window_reverse(g, proj, cur_h, cur_w, cur_c, WINDOW_SIZE, n_w, f"{p}_winrev")
            if shift:
                proj_img = _roll_nhwc(g, proj_img, cur_h, cur_w, cur_c, shift, shift, f"{p}_unshift")
            proj_tokens = _nhwc_to_tokens(g, proj_img, cur_h, cur_w, cur_c, f"{p}_img2tok")

            # Residual 1
            blk_qact2_scale = get_scale(sd, f"{p}.qact2.act_scaling_factor")
            x2 = g.residual_add(
                proj_tokens,
                attn_qact4_scale,
                x,
                cur_scale,
                blk_qact2_scale,
                f"{p}_res1",
            )

            # Norm2 -> qact3
            n2_bias, n2_sf = _fix_norm_bias_and_scale(
                sd,
                f"{p}.norm2.weight",
                f"{p}.norm2.bias_integer",
                f"{p}.norm2.norm_scaling_factor",
            )
            ln2 = g.q_layernorm(x2, n2_bias, f"{p}_norm2", input_dtype="int32")
            blk_qact3_scale = get_scale(sd, f"{p}.qact3.act_scaling_factor")
            ln2_req = g.requantize_from_int64(ln2, n2_sf, blk_qact3_scale, "int8", f"{p}_norm2_req")

            # MLP fc1 -> ShiftGELU -> fc2
            fc1_w = get_weight(sd, f"{p}.mlp.fc1.weight_integer").astype(np.int8).T
            fc1_b = to_int32(get_weight(sd, f"{p}.mlp.fc1.bias_integer")).reshape(-1)
            fc1_ws = get_scale(sd, f"{p}.mlp.fc1.fc_scaling_factor")
            mlp_qact_gelu_scale = get_scale(sd, f"{p}.mlp.qact_gelu.act_scaling_factor")
            fc1 = g.qlinear_matmul_bias(
                ln2_req,
                blk_qact3_scale,
                fc1_w,
                fc1_ws,
                fc1_b,
                mlp_qact_gelu_scale,
                "int8",
                f"{p}_mlp_fc1",
            )

            gelu_i32 = g.shift_gelu(fc1, mlp_qact_gelu_scale, f"{p}_mlp_gelu")
            gelu_out_scale = get_scale(sd, f"{p}.mlp.act.act_scaling_factor")
            mlp_qact1_scale = get_scale(sd, f"{p}.mlp.qact1.act_scaling_factor")
            gelu_req = g.requantize_int32(gelu_i32, gelu_out_scale, mlp_qact1_scale, f"{p}_mlp_gelu_req")

            fc2_w = get_weight(sd, f"{p}.mlp.fc2.weight_integer").astype(np.int8).T
            fc2_b = to_int32(get_weight(sd, f"{p}.mlp.fc2.bias_integer")).reshape(-1)
            fc2_ws = get_scale(sd, f"{p}.mlp.fc2.fc_scaling_factor")
            mlp_qact2_scale = get_scale(sd, f"{p}.mlp.qact2.act_scaling_factor")
            fc2 = g.qlinear_matmul_bias(
                gelu_req,
                mlp_qact1_scale,
                fc2_w,
                fc2_ws,
                fc2_b,
                mlp_qact2_scale,
                "int16",
                f"{p}_mlp_fc2",
            )

            blk_qact4_scale = get_scale(sd, f"{p}.qact4.act_scaling_factor")
            x = g.residual_add(fc2, mlp_qact2_scale, x2, blk_qact2_scale, blk_qact4_scale, f"{p}_res2")
            cur_scale = blk_qact4_scale
            cur_dtype = "int32"

        # Patch merging at stage end (except last stage)
        ds_prefix = f"layers.{stage_idx}.downsample"
        if stage_idx < len(depths) - 1 and f"{ds_prefix}.reduction.weight_integer" in sd:
            x_img = _tokens_to_nhwc(g, x, cur_h, cur_w, cur_c, f"{ds_prefix}_tok2img")
            x0 = _slice_static(g, x_img, [0, 0, 0, 0], [BATCH, cur_h, cur_w, cur_c], [0, 1, 2, 3], f"{ds_prefix}_x0", [1, 2, 2, 1])
            x1 = _slice_static(g, x_img, [0, 1, 0, 0], [BATCH, cur_h, cur_w, cur_c], [0, 1, 2, 3], f"{ds_prefix}_x1", [1, 2, 2, 1])
            x2m = _slice_static(g, x_img, [0, 0, 1, 0], [BATCH, cur_h, cur_w, cur_c], [0, 1, 2, 3], f"{ds_prefix}_x2", [1, 2, 2, 1])
            x3 = _slice_static(g, x_img, [0, 1, 1, 0], [BATCH, cur_h, cur_w, cur_c], [0, 1, 2, 3], f"{ds_prefix}_x3", [1, 2, 2, 1])
            merged = f"{ds_prefix}_merged"
            g.add("Concat", [x0, x1, x2m, x3], [merged], axis=3)

            next_h = cur_h // 2
            next_w = cur_w // 2
            next_c = cur_c * 2

            merged_tok = _nhwc_to_tokens(g, merged, next_h, next_w, cur_c * 4, f"{ds_prefix}_img2tok")
            ds_n_bias, ds_n_sf = _fix_norm_bias_and_scale(
                sd,
                f"{ds_prefix}.norm.weight",
                f"{ds_prefix}.norm.bias_integer",
                f"{ds_prefix}.norm.norm_scaling_factor",
            )
            ds_ln = g.q_layernorm(merged_tok, ds_n_bias, f"{ds_prefix}_norm", input_dtype=cur_dtype)
            ds_qact1_scale = get_scale(sd, f"{ds_prefix}.qact1.act_scaling_factor")
            ds_ln_req = g.requantize_from_int64(ds_ln, ds_n_sf, ds_qact1_scale, "int8", f"{ds_prefix}_norm_req")

            red_w = get_weight(sd, f"{ds_prefix}.reduction.weight_integer").astype(np.int8).T
            red_ws = get_scale(sd, f"{ds_prefix}.reduction.fc_scaling_factor")
            red_qact2_scale = get_scale(sd, f"{ds_prefix}.qact2.act_scaling_factor")
            red_bias = np.zeros(next_c, dtype=np.int32)
            x = g.qlinear_matmul_bias(
                ds_ln_req,
                ds_qact1_scale,
                red_w,
                red_ws,
                red_bias,
                red_qact2_scale,
                "int8",
                f"{ds_prefix}_reduction",
            )

            cur_h, cur_w, cur_c = next_h, next_w, next_c
            cur_scale = red_qact2_scale
            cur_dtype = "int8"

    # Final norm
    final_n_bias, final_n_sf = _fix_norm_bias_and_scale(
        sd,
        "norm.weight",
        "norm.bias_integer",
        "norm.norm_scaling_factor",
    )
    final_ln = g.q_layernorm(x, final_n_bias, "final_norm", input_dtype=cur_dtype)
    final_qact2_scale = get_scale(sd, "qact2.act_scaling_factor")
    final_ln_req = g.requantize_from_int64(final_ln, final_n_sf, final_qact2_scale, "int8", "final_norm_req")

    # AvgPool in float, then quantize to qact3
    final_dq = g.dequantize(final_ln_req, final_qact2_scale, "final_pool")
    final_t = "final_pool_t"
    g.add("Transpose", [final_dq], [final_t], perm=[0, 2, 1])  # [1, C, L]
    pooled = "final_pool_mean"
    g.add("ReduceMean", [final_t], [pooled], axes=[2], keepdims=1)
    final_qact3_scale = get_scale(sd, "qact3.act_scaling_factor")
    pooled_q = g.quantize(pooled, final_qact3_scale, "final_qact3")

    flat_shape = g.init_tensor("final_flat_shape", np.array([BATCH, cur_c], dtype=np.int64))
    flat = "final_flat"
    g.add("Reshape", [pooled_q, flat_shape], [flat])

    # Head GemminiMatMulInteger + dequantize to float logits
    head_w = get_weight(sd, "head.weight_integer").astype(np.int8).T
    head_b = to_int32(get_weight(sd, "head.bias_integer")).reshape(-1)
    head_ws = get_scale(sd, "head.fc_scaling_factor")

    head_int32 = g.gemmini_matmul_integer(flat, head_w, "head_mm")
    head_b_name = g.init_tensor("head_bias", head_b)
    g.add("Add", [head_int32, head_b_name], ["head_biased"])
    g.add("Cast", ["head_biased"], ["head_cast_f32"], to=TensorProto.FLOAT)

    head_ws_arr = np.asarray(head_ws, dtype=np.float32)
    if head_ws_arr.ndim == 0:
        head_scale = g.scalar_f32("head_dq_scale", float(final_qact3_scale) * float(head_ws_arr))
    else:
        head_scale = g.init_tensor("head_dq_scale", (float(final_qact3_scale) * head_ws_arr.reshape(-1)).astype(np.float32))
    logits = "logits"
    g.add("Mul", ["head_cast_f32", head_scale], [logits])

    input_vi = helper.make_tensor_value_info("image", TensorProto.FLOAT, [BATCH, IN_CHANNELS, IMG_SIZE, IMG_SIZE])
    output_vi = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [BATCH, NUM_CLASSES])
    graph = helper.make_graph(g.nodes, "ivit_swin_tiny", [input_vi], [output_vi], initializer=g.initializers)
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 13),
            helper.make_opsetid("com.microsoft", 1),
            helper.make_opsetid("ivit", 1),
        ],
    )
    model.ir_version = 7
    return model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Swin-Tiny I-ViT-style INT8 ONNX graph with custom ops",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Swin I-ViT QAT checkpoint path")
    parser.add_argument("--allow-random-init", action="store_true", help="Allow export with random weights")
    parser.add_argument(
        "--patch-embed-op",
        choices=("matmul", "qlinearconv"),
        default="matmul",
        help="Patch embedding lowering to use for the initial projection",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output ONNX path")
    args = parser.parse_args()

    depths = SWIN_DEPTHS
    print(f"Swin depths (fixed): {depths}")
    print(f"Patch embed op: {args.patch_embed_op}")
    sd = _prepare_swin_state_dict(args.checkpoint, args.allow_random_init, depths)
    model = _build_swin_ivit_graph(sd, depths, args.patch_embed_op)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    size_mb = args.output.stat().st_size / 1024.0 / 1024.0
    print(f"Saved: {args.output} ({size_mb:.1f} MB)")

    try:
        onnx.checker.check_model(model)
        print("ONNX structure check: OK")
    except Exception as ex:
        print(f"ONNX checker warning (custom ops can trigger this): {ex}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
