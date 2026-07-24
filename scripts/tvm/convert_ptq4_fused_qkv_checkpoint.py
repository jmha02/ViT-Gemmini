#!/usr/bin/env python3
"""Convert flexi PTQ4 split-QKV checkpoint to fused I-ViT-style QKV params."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _tensor(sd: dict, key: str) -> torch.Tensor:
    if key not in sd:
        raise KeyError(f"Missing checkpoint key: {key}")
    return sd[key]


def fuse_qkv_block(sd: dict, prefix: str, *, dim: int, num_heads: int) -> None:
    head_dim = dim // num_heads
    out_dim = 3 * dim
    num_blocks = 3 * num_heads

    weight_integer = np.zeros((out_dim, dim), dtype=np.int8)
    bias_integer = np.zeros(out_dim, dtype=np.int32)
    requant_scale = np.zeros(out_dim, dtype=np.float32)

    for g in range(num_blocks):
        lo, hi = g * head_dim, (g + 1) * head_dim
        w_g = _tensor(sd, f"{prefix}.w{g}").detach().cpu().numpy().astype(np.int8)
        d_g = _tensor(sd, f"{prefix}.d{g}").detach().cpu().numpy().astype(np.int32).reshape(-1)
        s_g = float(_tensor(sd, f"{prefix}.s{g}").detach().cpu().numpy().reshape(-1)[0])
        weight_integer[lo:hi, :] = w_g.T
        bias_integer[lo:hi] = d_g
        requant_scale[lo:hi] = s_g

    sd[f"{prefix}.weight_integer"] = torch.from_numpy(weight_integer)
    sd[f"{prefix}.bias_integer"] = torch.from_numpy(bias_integer)
    sd[f"{prefix}.requant_scale"] = torch.from_numpy(requant_scale)
    sd[f"{prefix}.kernel_scale"] = torch.ones(out_dim, dtype=torch.float32)
    sd[f"{prefix}.a_interval"] = _tensor(sd, f"{prefix}.a_interval")

    for g in range(num_blocks):
        for suffix in ("w", "d", "s"):
            sd.pop(f"{prefix}.{suffix}{g}", None)


def add_linear_qnn_fields(sd: dict, prefix: str) -> None:
    if f"{prefix}.weight_q_t" not in sd:
        return
    w = _tensor(sd, f"{prefix}.weight_q_t").detach().cpu().numpy().astype(np.int8)
    out_f = w.shape[1]
    sd[f"{prefix}.weight_integer"] = torch.from_numpy(w.T.copy())
    if f"{prefix}.bias_integer" not in sd:
        sd[f"{prefix}.bias_integer"] = torch.zeros(out_f, dtype=torch.int32)
    a_interval = float(_tensor(sd, f"{prefix}.a_interval").detach().cpu().numpy().reshape(-1)[0])
    out_scale = _tensor(sd, f"{prefix}.output_scale").detach().cpu().numpy().astype(np.float32).reshape(-1)
    kernel_scale = out_scale / max(a_interval, 1e-12)
    sd[f"{prefix}.kernel_scale"] = torch.from_numpy(kernel_scale.astype(np.float32))



def add_weight_integer_only(sd: dict, prefix: str) -> None:
    if f"{prefix}.weight_q_t" not in sd:
        return
    w = _tensor(sd, f"{prefix}.weight_q_t").detach().cpu().numpy().astype(np.int8)
    sd[f"{prefix}.weight_integer"] = torch.from_numpy(w.T.copy())

def convert_state_dict(sd: dict, *, embed_dim: int, depth: int, num_heads: int) -> dict:
    out = {k: v for k, v in sd.items()}
    for i in range(depth):
        fuse_qkv_block(out, f"blocks.{i}.attn.qkv", dim=embed_dim, num_heads=num_heads)
        for linear_prefix in (f"blocks.{i}.attn.proj", f"blocks.{i}.mlp.fc1"):
            add_linear_qnn_fields(out, linear_prefix)
        # fc2 uses TwinGeluLinear (weight_q_t + weight_scale); keep split layout.
    add_weight_integer_only(out, "patch_embed")
    add_linear_qnn_fields(out, "head")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("/root/flexi/eval/data/ptq4_deit_t/e2e_model.pt"))
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "build/checkpoints/ptq4_deit_tiny_fused_qkv.pt")
    ap.add_argument("--embed-dim", type=int, default=192)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--num-heads", type=int, default=3)
    args = ap.parse_args()

    blob = torch.load(str(args.input), map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        sd = blob["state_dict"]
        wrapper = {**blob}
    else:
        sd = blob
        wrapper = {"state_dict": sd}

    converted = convert_state_dict(sd, embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads)
    wrapper["state_dict"] = converted
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(wrapper, str(args.output))
    print(f"Wrote fused checkpoint: {args.output}")


if __name__ == "__main__":
    main()
