from __future__ import annotations

import re

import numpy as np
from tvm import relay

from . import layers


SWIN_TINY_DEPTHS = (2, 2, 6, 2)
SWIN_TINY_HEADS = (3, 6, 12, 24)
SWIN_LAYER_NORM_EPS = 1e-5
_ONLY_REPQ_SWIN_STAGE_BLOCK_RE = re.compile(r"^only_stage(\d+)_block(\d+)$")
_ONLY_REPQ_SWIN_GLOBAL_BLOCK_RE = re.compile(r"^only_block(\d+)$")


def _resolve_only_repq_swin_block(debug_unit: str | None, depths) -> tuple[int, int] | None:
    if not debug_unit:
        return None

    stage_match = _ONLY_REPQ_SWIN_STAGE_BLOCK_RE.match(debug_unit)
    if stage_match:
        stage_idx = int(stage_match.group(1))
        block_idx = int(stage_match.group(2))
        if not 0 <= stage_idx < len(depths):
            raise RuntimeError(f"Unsupported RepQ Swin standalone stage index: {stage_idx}")
        if not 0 <= block_idx < depths[stage_idx]:
            raise RuntimeError(
                f"Unsupported RepQ Swin standalone block index {block_idx} for stage {stage_idx}"
            )
        return stage_idx, block_idx

    global_match = _ONLY_REPQ_SWIN_GLOBAL_BLOCK_RE.match(debug_unit)
    if not global_match:
        return None

    global_block_idx = int(global_match.group(1))
    if not 0 <= global_block_idx < sum(depths):
        raise RuntimeError(f"Unsupported RepQ Swin standalone block index: {global_block_idx}")

    running = 0
    for stage_idx, stage_depth in enumerate(depths):
        if global_block_idx < running + stage_depth:
            return stage_idx, global_block_idx - running
        running += stage_depth

    raise RuntimeError(f"Could not resolve RepQ Swin block index: {global_block_idx}")


def _standalone_repq_swin_block_signature(
    embed_dim: int, depths, num_heads, window_size: int, stage_idx: int, block_idx: int
) -> dict[str, object]:
    h = 56
    w = 56
    dim = embed_dim

    for current_stage_idx, stage_depth in enumerate(depths):
        for current_block_idx in range(stage_depth):
            if current_stage_idx == stage_idx and current_block_idx == block_idx:
                shift_size = (
                    0
                    if (current_block_idx % 2 == 0 or min(h, w) <= window_size)
                    else window_size // 2
                )
                return {
                    "input_shape": [1, h * w, dim],
                    "dim": dim,
                    "num_heads": num_heads[current_stage_idx],
                    "input_resolution": (h, w),
                    "shift_size": shift_size,
                }
        if current_stage_idx < len(depths) - 1:
            h //= 2
            w //= 2
            dim *= 2

    raise RuntimeError(
        f"Could not determine RepQ Swin signature for stage {stage_idx} block {block_idx}"
    )


def _layer_norm_var(module_name: str, dim: int) -> tuple[relay.Expr, relay.Expr]:
    prefix = layers.sanitize_name(module_name)
    weight = relay.var(prefix + "_weight", shape=[dim], dtype="float32")
    bias = relay.var(prefix + "_bias", shape=[dim], dtype="float32")
    return weight, bias


def _slice_nhwc(x, batch, h0, h1, w0, w1, c):
    return relay.strided_slice(
        x,
        begin=[0, h0, w0, 0],
        end=[batch, h1, w1, c],
        strides=[1, 1, 1, 1],
    )


def _cyclic_shift_neg(x, batch, h, w, c, shift_size):
    if shift_size == 0:
        return x
    x1 = _slice_nhwc(x, batch, shift_size, h, shift_size, w, c)
    x2 = _slice_nhwc(x, batch, shift_size, h, 0, shift_size, c)
    x3 = _slice_nhwc(x, batch, 0, shift_size, shift_size, w, c)
    x4 = _slice_nhwc(x, batch, 0, shift_size, 0, shift_size, c)
    top = relay.concatenate([x1, x2], axis=2)
    bottom = relay.concatenate([x3, x4], axis=2)
    return relay.concatenate([top, bottom], axis=1)


def _cyclic_shift_pos(x, batch, h, w, c, shift_size):
    if shift_size == 0:
        return x
    x1 = _slice_nhwc(x, batch, h - shift_size, h, w - shift_size, w, c)
    x2 = _slice_nhwc(x, batch, h - shift_size, h, 0, w - shift_size, c)
    x3 = _slice_nhwc(x, batch, 0, h - shift_size, w - shift_size, w, c)
    x4 = _slice_nhwc(x, batch, 0, h - shift_size, 0, w - shift_size, c)
    top = relay.concatenate([x1, x2], axis=2)
    bottom = relay.concatenate([x3, x4], axis=2)
    return relay.concatenate([top, bottom], axis=1)


def _window_partition(x, batch, h, w, c, window_size):
    x = relay.reshape(
        x,
        [batch, h // window_size, window_size, w // window_size, window_size, c],
    )
    x = relay.transpose(x, [0, 1, 3, 2, 4, 5])
    return relay.reshape(
        x,
        [batch * (h // window_size) * (w // window_size), window_size * window_size, c],
    )


def _window_reverse(x, batch, h, w, c, window_size):
    x = relay.reshape(
        x,
        [batch, h // window_size, w // window_size, window_size, window_size, c],
    )
    x = relay.transpose(x, [0, 1, 3, 2, 4, 5])
    return relay.reshape(x, [batch, h, w, c])


def _relative_position_index(window_size):
    coords_h = np.arange(window_size)
    coords_w = np.arange(window_size)
    coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))
    coords_flatten = coords.reshape(2, -1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = np.transpose(relative_coords, (1, 2, 0))
    relative_coords[:, :, 0] += window_size - 1
    relative_coords[:, :, 1] += window_size - 1
    relative_coords[:, :, 0] *= 2 * window_size - 1
    return relative_coords.sum(-1).astype("int32").reshape(-1)


def _attention_mask(input_resolution, window_size, shift_size):
    if shift_size == 0:
        return None

    h, w = input_resolution
    img_mask = np.zeros((1, h, w, 1), dtype=np.float32)
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    count = 0
    for hs in h_slices:
        for ws in w_slices:
            img_mask[:, hs, ws, :] = count
            count += 1

    nwh = h // window_size
    nww = w // window_size
    mask_windows = img_mask.reshape(1, nwh, window_size, nww, window_size, 1)
    mask_windows = np.transpose(mask_windows, (0, 1, 3, 2, 4, 5)).reshape(
        -1, window_size * window_size
    )
    attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
    return np.where(attn_mask != 0, -100.0, 0.0).astype("float32")


def _reshape_qkv(qkv: relay.Expr, num_windows: int, tokens_per_window: int, num_heads: int, head_dim: int):
    qkv = relay.reshape(qkv, [num_windows, tokens_per_window, 3, num_heads, head_dim])
    qkv = relay.transpose(qkv, [2, 0, 3, 1, 4])
    qkv = relay.split(qkv, 3, axis=0)

    def _pack(part):
        part = relay.squeeze(part, axis=[0])
        return relay.reshape(part, [num_windows * num_heads, tokens_per_window, head_dim])

    return _pack(qkv[0]), _pack(qkv[1]), _pack(qkv[2])


def _repq_swin_block(
    data,
    name: str,
    dim: int,
    num_heads: int,
    input_resolution: tuple[int, int],
    window_size: int,
    shift_size: int,
    batch_size: int,
):
    h, w = input_resolution
    num_windows = batch_size * (h // window_size) * (w // window_size)
    tokens_per_window = window_size * window_size
    head_dim = dim // num_heads

    shortcut = data
    norm1_weight, norm1_bias = _layer_norm_var(name + ".norm1", dim)
    x = layers.layer_norm_float(data, norm1_weight, norm1_bias, eps=SWIN_LAYER_NORM_EPS)
    x = relay.reshape(x, [batch_size, h, w, dim])
    x = _cyclic_shift_neg(x, batch_size, h, w, dim, shift_size)
    x = _window_partition(x, batch_size, h, w, dim, window_size)

    qkv = layers.exact_uniform_linear(x, name + ".attn.qkv", add_bias=True)
    q, k, v = _reshape_qkv(qkv, num_windows, tokens_per_window, num_heads, head_dim)
    q = q * relay.const((head_dim ** -0.5), "float32")

    matmul1_meta = layers.RepQContext.get(name + ".attn.matmul1")
    attn = layers.exact_uniform_matmul(
        q,
        relay.transpose(k, axes=[0, 2, 1]),
        a_scale=matmul1_meta.input_scale,
        a_zero_point=matmul1_meta.input_zero_point,
        b_scale=matmul1_meta.b_scale,
        b_zero_point=matmul1_meta.b_zero_point,
        n_bits=matmul1_meta.n_bits,
    )
    attn = relay.reshape(attn, [num_windows, num_heads, tokens_per_window, tokens_per_window])

    rel_pos_name = layers.sanitize_name(name + ".attn") + "_relative_position_bias_table"
    rel_pos_table = relay.var(
        rel_pos_name,
        shape=((2 * window_size - 1) * (2 * window_size - 1), num_heads),
        dtype="float32",
    )
    rel_index = relay.const(_relative_position_index(window_size), dtype="int32")
    rel_pos = relay.take(rel_pos_table, rel_index, axis=0)
    rel_pos = relay.reshape(rel_pos, [tokens_per_window, tokens_per_window, num_heads])
    rel_pos = relay.transpose(rel_pos, [2, 0, 1])
    rel_pos = relay.expand_dims(rel_pos, axis=0)
    if num_windows > 1:
        rel_pos = relay.repeat(rel_pos, num_windows, axis=0)
    attn = attn + rel_pos

    if shift_size > 0:
        attn_mask = relay.const(_attention_mask(input_resolution, window_size, shift_size), "float32")
        attn_mask = relay.expand_dims(attn_mask, axis=1)
        attn = attn + attn_mask

    attn = relay.nn.softmax(attn, axis=-1)
    attn = relay.reshape(attn, [num_windows * num_heads, tokens_per_window, tokens_per_window])

    matmul2_meta = layers.RepQContext.get(name + ".attn.matmul2")
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
    x = relay.reshape(x, [num_windows, num_heads, tokens_per_window, head_dim])
    x = relay.transpose(x, [0, 2, 1, 3])
    x = relay.reshape(x, [num_windows, tokens_per_window, dim])

    x = _window_reverse(x, batch_size, h, w, dim, window_size)
    x = _cyclic_shift_pos(x, batch_size, h, w, dim, shift_size)
    x = relay.reshape(x, [batch_size, h * w, dim])
    x = layers.exact_uniform_linear(x, name + ".attn.proj", add_bias=True)
    x = x + shortcut

    shortcut = x
    norm2_weight, norm2_bias = _layer_norm_var(name + ".norm2", dim)
    x = layers.layer_norm_float(x, norm2_weight, norm2_bias, eps=SWIN_LAYER_NORM_EPS)
    x = layers.exact_uniform_linear(x, name + ".mlp.fc1", add_bias=True)
    x = layers.gelu_float(x)
    x = layers.exact_uniform_linear(x, name + ".mlp.fc2", add_bias=True)
    return relay.annotation.stop_fusion(x + shortcut)


def _patch_merging(data, name: str, input_resolution: tuple[int, int], dim: int, batch_size: int):
    h, w = input_resolution
    x = relay.reshape(data, [batch_size, h, w, dim])

    x0 = relay.strided_slice(x, begin=[0, 0, 0, 0], end=[batch_size, h, w, dim], strides=[1, 2, 2, 1])
    x1 = relay.strided_slice(x, begin=[0, 1, 0, 0], end=[batch_size, h, w, dim], strides=[1, 2, 2, 1])
    x2 = relay.strided_slice(x, begin=[0, 0, 1, 0], end=[batch_size, h, w, dim], strides=[1, 2, 2, 1])
    x3 = relay.strided_slice(x, begin=[0, 1, 1, 0], end=[batch_size, h, w, dim], strides=[1, 2, 2, 1])
    x = relay.concatenate([x0, x1, x2, x3], axis=3)
    x = relay.reshape(x, [batch_size, (h // 2) * (w // 2), 4 * dim])

    norm_weight, norm_bias = _layer_norm_var(name + ".norm", 4 * dim)
    x = layers.layer_norm_float(x, norm_weight, norm_bias, eps=SWIN_LAYER_NORM_EPS)
    # RepQ scale reparameterization can materialize a bias on the patch-merging
    # reduction even though the original Swin layer is bias-free.
    x = layers.exact_uniform_linear(x, name + ".reduction", add_bias=True)
    return relay.reshape(x, [batch_size, (h // 2) * (w // 2), 2 * dim])


def RepQSwinTiny(data_shape, embed_dim: int = 96, debug_unit: str | None = None):
    batch_size = data_shape[0]
    data = relay.var("data", shape=data_shape, dtype="float32")

    if debug_unit in ("only_embed", "only_embedding"):
        debug_unit = "post_stem"

    standalone_block = _resolve_only_repq_swin_block(debug_unit, SWIN_TINY_DEPTHS)
    if standalone_block is not None:
        stage_idx, block_idx = standalone_block
        signature = _standalone_repq_swin_block_signature(
            embed_dim=embed_dim,
            depths=SWIN_TINY_DEPTHS,
            num_heads=SWIN_TINY_HEADS,
            window_size=7,
            stage_idx=stage_idx,
            block_idx=block_idx,
        )
        x = relay.var(
            "data",
            shape=signature["input_shape"],
            dtype="float32",
        )
        body = _repq_swin_block(
            x,
            name=f"layers.{stage_idx}.blocks.{block_idx}",
            dim=signature["dim"],
            num_heads=signature["num_heads"],
            input_resolution=signature["input_resolution"],
            window_size=7,
            shift_size=signature["shift_size"],
            batch_size=batch_size,
        )
        return relay.Function(relay.analysis.free_vars(body), body)

    if debug_unit in ("only_head", "only_classifier", "classifier_only"):
        final_dim = embed_dim * (2 ** (len(SWIN_TINY_DEPTHS) - 1))
        final_hw = 56 // (2 ** (len(SWIN_TINY_DEPTHS) - 1))
        x = relay.var(
            "data",
            shape=[batch_size, final_hw * final_hw, final_dim],
            dtype="float32",
        )
        norm_weight, norm_bias = _layer_norm_var("norm", final_dim)
        x = layers.layer_norm_float(x, norm_weight, norm_bias, eps=SWIN_LAYER_NORM_EPS)
        x = relay.mean(x, axis=1)
        x = layers.exact_uniform_linear(x, "head.fc", add_bias=True)
        return relay.Function(relay.analysis.free_vars(x), x)

    x = layers.patchify_conv2d_exact(
        data,
        module_name="patch_embed.proj",
        kernel_size=(4, 4),
        stride=(4, 4),
        out_channels=embed_dim,
    )
    x = relay.reshape(x, [batch_size, 56 * 56, embed_dim])
    if debug_unit == "post_patch_embed_proj":
        return relay.Function(relay.analysis.free_vars(x), x)
    patch_norm_weight, patch_norm_bias = _layer_norm_var("patch_embed.norm", embed_dim)
    x = layers.layer_norm_float(x, patch_norm_weight, patch_norm_bias, eps=SWIN_LAYER_NORM_EPS)
    if debug_unit == "post_patch_embed_norm":
        return relay.Function(relay.analysis.free_vars(x), x)
    x = relay.annotation.stop_fusion(x)
    if debug_unit == "post_stem":
        return relay.Function(relay.analysis.free_vars(x), x)

    h = 56
    w = 56
    dim = embed_dim
    global_block_idx = 0
    for stage_idx, stage_depth in enumerate(SWIN_TINY_DEPTHS):
        for block_idx in range(stage_depth):
            name = f"layers.{stage_idx}.blocks.{block_idx}"
            shift_size = 0 if (block_idx % 2 == 0 or min(h, w) <= 7) else 3
            x = _repq_swin_block(
                x,
                name=name,
                dim=dim,
                num_heads=SWIN_TINY_HEADS[stage_idx],
                input_resolution=(h, w),
                window_size=7,
                shift_size=shift_size,
                batch_size=batch_size,
            )
            if debug_unit == f"post_stage{stage_idx}_block{block_idx}":
                return relay.Function(relay.analysis.free_vars(x), x)
            if debug_unit == f"post_block{global_block_idx}":
                return relay.Function(relay.analysis.free_vars(x), x)
            global_block_idx += 1
        if stage_idx < len(SWIN_TINY_DEPTHS) - 1:
            downsample_name = f"layers.{stage_idx + 1}.downsample"
            x = _patch_merging(x, downsample_name, (h, w), dim, batch_size)
            if debug_unit == f"post_stage{stage_idx}_downsample":
                return relay.Function(relay.analysis.free_vars(x), x)
            h //= 2
            w //= 2
            dim *= 2

    norm_weight, norm_bias = _layer_norm_var("norm", dim)
    x = layers.layer_norm_float(x, norm_weight, norm_bias, eps=SWIN_LAYER_NORM_EPS)
    if debug_unit == "pre_head":
        return relay.Function(relay.analysis.free_vars(x), x)
    x = relay.mean(x, axis=1)
    if debug_unit == "pre_head_float":
        return relay.Function(relay.analysis.free_vars(x), x)
    x = layers.exact_uniform_linear(x, "head.fc", add_bias=True)
    if debug_unit == "head_float":
        return relay.Function(relay.analysis.free_vars(x), x)
    return relay.Function(relay.analysis.free_vars(x), x)


def get_repq_swin_tiny_model(data_shape, debug_unit: str | None = None):
    return RepQSwinTiny(data_shape=data_shape, debug_unit=debug_unit)
