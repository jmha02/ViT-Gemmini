"""PTQ4-DeiT Relay graph builder aligned with flexi eval/specs/ptq4_deit.py."""

from __future__ import annotations

import re

from tvm import relay

from . import ptq4_layers as L
from .ptq4_checkpoint import param_name


_ONLY_BLOCK_RE = re.compile(r"^only_block(\d+)$")


def _parse_only_block(debug_unit, depth):
    if not debug_unit:
        return None
    match = _ONLY_BLOCK_RE.match(debug_unit)
    if not match:
        return None
    block_idx = int(match.group(1))
    if not 0 <= block_idx < depth:
        raise RuntimeError(f"Unsupported PTQ4 standalone block index: {block_idx}")
    return block_idx


def PTQ4_VisionTransformer(
    data_shape,
    *,
    embed_dim: int = 192,
    depth: int = 12,
    num_heads: int = 3,
    mlp_ratio: int = 4,
    num_classes: int = 1000,
    img_size: int = 224,
    patch_size: int = 16,
    in_chans: int = 3,
    debug_unit=None,
):
    """Build Relay expr for PTQ4 DeiT. Input dtype float32 NCHW."""
    batch_size = int(data_shape[0])
    seq_len = (img_size // patch_size) ** 2 + 1
    data = relay.var("data", shape=data_shape, dtype="float32")

    only_block = _parse_only_block(debug_unit, depth)
    if only_block is not None:
        # Standalone block: input [B, seq, dim] float32
        block_in = relay.var("data", shape=(batch_size, seq_len, embed_dim), dtype="float32")
        out = L.ptq4_block(
            block_in,
            f"blocks_{only_block}",
            dim=embed_dim,
            num_heads=num_heads,
            seq_len=seq_len,
            mlp_ratio=mlp_ratio,
        )
        return relay.Function(relay.analysis.free_vars(out), out)

    x = L.ptq4_patch_embed(
        data,
        "patch_embed",
        batch_size=batch_size,
        in_chans=in_chans,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
    )
    cls = relay.var("cls_token", shape=(1, 1, embed_dim), dtype="float32")
    pos = relay.var("pos_embed", shape=(1, seq_len, embed_dim), dtype="float32")
    cls_b = relay.repeat(cls, batch_size, axis=0)
    x = relay.add(relay.concatenate([cls_b, x], axis=1), pos)

    for i in range(depth):
        x = L.ptq4_block(
            x,
            f"blocks_{i}",
            dim=embed_dim,
            num_heads=num_heads,
            seq_len=seq_len,
            mlp_ratio=mlp_ratio,
        )
        if debug_unit == f"only_block{i}_out":
            return relay.Function(relay.analysis.free_vars(x), x)

    x = L.ptq4_layer_norm(x, "norm", embed_dim)
    # CLS token
    x = relay.strided_slice(x, begin=[0, 0, 0], end=[batch_size, 1, embed_dim])
    x = relay.reshape(x, [batch_size, embed_dim])
    logits = L.ptq4_int_mm_linear(x, "head", out_features=num_classes)
    return relay.Function(relay.analysis.free_vars(logits), logits)


def expected_param_names_for_block(block_idx: int, *, dim: int, num_heads: int, mlp_ratio: int = 4) -> list[str]:
    """Relay param names used by a single block (for debugging)."""
    p = f"blocks_{block_idx}"
    head_dim = dim // num_heads
    hidden = dim * mlp_ratio
    names = [
        f"{p}_norm1_weight",
        f"{p}_norm1_bias",
        f"{p}_attn_qkv_a_interval",
    ]
    for g in range(3 * num_heads):
        names += [f"{p}_attn_qkv_w{g}", f"{p}_attn_qkv_d{g}", f"{p}_attn_qkv_s{g}"]
    names += [
        f"{p}_attn_matmul1_output_scale",
        f"{p}_attn_matmul2_split",
        f"{p}_attn_matmul2_a_interval",
        f"{p}_attn_matmul2_b_interval",
        f"{p}_attn_proj_a_interval",
        f"{p}_attn_proj_weight_q_t",
        f"{p}_attn_proj_output_scale",
        f"{p}_attn_proj_bias",
        f"{p}_norm2_weight",
        f"{p}_norm2_bias",
        f"{p}_mlp_fc1_a_interval",
        f"{p}_mlp_fc1_weight_q_t",
        f"{p}_mlp_fc1_output_scale",
        f"{p}_mlp_fc1_bias",
        f"{p}_mlp_fc2_a_interval",
        f"{p}_mlp_fc2_a_neg_interval",
        f"{p}_mlp_fc2_weight_q_t",
        f"{p}_mlp_fc2_weight_scale",
        f"{p}_mlp_fc2_bias",
    ]
    _ = (head_dim, hidden, param_name)  # silence unused in lint-ish envs
    return names
