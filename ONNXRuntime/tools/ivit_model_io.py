#!/usr/bin/env python3
"""Shared I-ViT model/checkpoint helpers for ONNX Runtime tooling."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
IVIT_ROOT = REPO_ROOT / "I-ViT"

if str(IVIT_ROOT) not in sys.path:
    sys.path.insert(0, str(IVIT_ROOT))


SCALE_BUFFER_SUFFIXES = {"act_scaling_factor", "norm_scaling_factor"}


def load_checkpoint_state_dict(path: str | Path) -> dict:
    """Load a raw checkpoint state_dict and strip any DistributedDataParallel prefix."""
    import torch

    ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")

    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint does not contain a state_dict dictionary")

    if state_dict and all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def create_model(model_name: str):
    """Instantiate a supported I-ViT model without loading weights."""
    from models.swin_quant import swin_small_patch4_window7_224, swin_tiny_patch4_window7_224
    from models.vit_quant import deit_small_patch16_224, deit_tiny_patch16_224

    builders = {
        "deit_tiny_patch16_224": deit_tiny_patch16_224,
        "deit_small_patch16_224": deit_small_patch16_224,
        "swin_tiny_patch4_window7_224": swin_tiny_patch4_window7_224,
        "swin_small_patch4_window7_224": swin_small_patch4_window7_224,
    }
    try:
        return builders[model_name](pretrained=False)
    except KeyError as exc:
        supported = ", ".join(sorted(builders))
        raise ValueError(f"Unsupported model_name={model_name!r}; supported: {supported}") from exc


def _get_parent_module(model, tensor_name: str):
    parent = model
    parts = tensor_name.split(".")
    for attr in parts[:-1]:
        parent = getattr(parent, attr)
    return parent, parts[-1]


def resize_model_scale_buffers(model, state_dict: dict) -> list[tuple[str, tuple[int, ...], tuple[int, ...]]]:
    """
    Resize placeholder scale buffers to the exact checkpoint tensor shape.

    I-ViT models register scale buffers as scalar placeholders (`zeros(1)`), but
    checkpoints may store them as 0-D tensors or per-channel vectors. Rather than
    collapsing those values, update the model buffer shapes before `load_state_dict`
    so the checkpoint can be loaded losslessly.
    """

    model_buffers = dict(model.named_buffers())
    resized: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for name, tensor in state_dict.items():
        current = model_buffers.get(name)
        if current is None:
            continue
        if name.rsplit(".", 1)[-1] not in SCALE_BUFFER_SUFFIXES:
            continue
        if tuple(current.shape) == tuple(tensor.shape):
            continue

        parent, attr = _get_parent_module(model, name)
        value = tensor.detach().clone()
        if current.dtype != value.dtype:
            value = value.to(dtype=current.dtype)
        setattr(parent, attr, value)
        resized.append((name, tuple(current.shape), tuple(value.shape)))

    return resized


def load_checkpoint_into_model(model, checkpoint: str | Path | dict, strict: bool = False):
    """Load a checkpoint into a model after aligning known scale-buffer shapes."""
    if isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = load_checkpoint_state_dict(checkpoint)
    resized = resize_model_scale_buffers(model, state_dict)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return state_dict, resized, incompatible


def restore_quantact_ranges(model) -> int:
    """
    Reconstruct QuantAct min/max buffers from loaded act_scaling_factor values.

    I-ViT checkpoints persist the learned scaling factors but not the derived
    range tensors used by QuantAct during inference. TVM parity scripts restore
    these ranges before running PyTorch reference inference; do the same here so
    ONNX/host comparisons use the correct integer model semantics.
    """
    restored = 0
    for module in model.modules():
        if not hasattr(module, "activation_bit") or not hasattr(module, "act_scaling_factor"):
            continue
        scale = module.act_scaling_factor.detach().to(device="cpu")
        levels = float(2 ** (int(module.activation_bit) - 1) - 1)
        module.min_val = (-scale * levels).clone()
        module.max_val = (scale * levels).clone()
        restored += 1
    return restored


def build_reference_model(model_name: str, checkpoint: str | Path):
    """Create, load, eval, and freeze an I-ViT model for inference."""
    from models.model_utils import freeze_model

    model = create_model(model_name)
    state_dict, resized, incompatible = load_checkpoint_into_model(model, checkpoint, strict=False)
    restore_quantact_ranges(model)
    model.eval()
    freeze_model(model)
    return model, state_dict, resized, incompatible
