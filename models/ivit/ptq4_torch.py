"""Thin torch re-export / loader for flexi PTQ4 DeiT (numeric checks)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

FLEXI_SPECS = Path("/root/flexi/eval/specs")
if str(FLEXI_SPECS) not in sys.path:
    sys.path.insert(0, str(FLEXI_SPECS))

import ptq4_deit as flexi_ptq4  # noqa: E402

from .ptq4_checkpoint import DEFAULT_PTQ4_DEIT_T_CHECKPOINT, load_ptq4_state_dict


def build_ptq4_deit_tiny(checkpoint: str | Path | None = None) -> torch.nn.Module:
    model = flexi_ptq4.PTQ4DeiTTiny().eval()
    path = Path(checkpoint) if checkpoint is not None else DEFAULT_PTQ4_DEIT_T_CHECKPOINT
    if path.is_file():
        sd = load_ptq4_state_dict(path)
        model.load_state_dict(sd, strict=True)
    return model


def build_ptq4_deit_small(checkpoint: str | Path | None = None) -> torch.nn.Module:
    model = flexi_ptq4.PTQ4DeiTSmall().eval()
    if checkpoint is not None and Path(checkpoint).is_file():
        sd = load_ptq4_state_dict(checkpoint)
        model.load_state_dict(sd, strict=True)
    return model
