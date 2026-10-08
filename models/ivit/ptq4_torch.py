"""Thin torch re-export / loader for flexi PTQ4 DeiT (numeric checks)."""

from __future__ import annotations

import sys
import os
from pathlib import Path

import torch

FLEXI_EVAL_ROOT = Path(
    os.environ.get("FLEXI_EVAL_ROOT", Path(__file__).resolve().parents[2] / "eval")
).expanduser()
FLEXI_SPECS = FLEXI_EVAL_ROOT / "specs"
if not FLEXI_SPECS.is_dir():
    raise RuntimeError(f"PTQ4 model specs not found at {FLEXI_SPECS}; set FLEXI_EVAL_ROOT.")
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
