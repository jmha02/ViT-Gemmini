"""Helpers for flexi PTQ4ViT checkpoints (e2e_model.pt layout)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

DEFAULT_PTQ4_DEIT_T_CHECKPOINT = Path("/root/flexi/eval/data/ptq4_deit_t/e2e_model.pt")
DEFAULT_PTQ4_DEIT_T_INPUT = Path("/root/flexi/eval/data/ptq4_deit_t/model_input.f32.bin")
DEFAULT_PTQ4_DEIT_T_OUTPUT = Path("/root/flexi/eval/data/ptq4_deit_t/model_output.f32.bin")


def load_ptq4_state_dict(path: str | Path) -> dict:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError(f"Unsupported PTQ4 checkpoint format: {type(ckpt)}")


def param_name(key: str) -> str:
    return key.replace(".", "_")


def tensor_value(tensor, dtype=None) -> np.ndarray:
    arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def build_ptq4_param_dict(state_dict: dict) -> dict[str, np.ndarray]:
    """Map checkpoint tensors to Relay var names (dots → underscores)."""
    params: dict[str, np.ndarray] = {}
    for key, value in state_dict.items():
        arr = tensor_value(value)
        # Relay scalar params use shape ()
        if arr.ndim == 0:
            params[param_name(key)] = arr
        else:
            params[param_name(key)] = arr
    return params


def random_ptq4_state_dict(
    *,
    embed_dim: int = 192,
    depth: int = 12,
    num_heads: int = 3,
    mlp_ratio: int = 4,
    num_classes: int = 1000,
    img_size: int = 224,
    patch_size: int = 16,
    in_chans: int = 3,
    seed: int = 0,
) -> dict:
    """Random-init state_dict matching flexi PTQ4VisionTransformer layout."""
    import sys
    from pathlib import Path as P

    sys.path.insert(0, str(P("/root/flexi/eval/specs")))
    import ptq4_deit as M  # noqa: WPS433

    torch.manual_seed(seed)
    model = M.PTQ4VisionTransformer(
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        num_classes=num_classes,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
    )
    # Fill int8 weights / scales with small random values so forward is defined.
    with torch.no_grad():
        for name, buf in model.named_buffers():
            if buf.dtype == torch.int8:
                buf.copy_(torch.randint(-8, 9, buf.shape, dtype=torch.int8))
            elif buf.dtype == torch.int32:
                buf.zero_()
            elif buf.dtype == torch.float32:
                if buf.numel() == 0:
                    continue
                if "interval" in name or "scale" in name or "split" in name:
                    buf.fill_(0.1 if "split" not in name else 0.5)
                else:
                    buf.normal_(0.0, 0.02)
        for name, param in model.named_parameters():
            param.normal_(0.0, 0.02)
    return model.state_dict()
