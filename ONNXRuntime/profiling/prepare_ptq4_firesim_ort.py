#!/usr/bin/env python3
"""Prepare PTQ4-DeiT ORT FireSim artifacts (f32 input)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import onnx


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX = REPO_ROOT / "build/ort/ptq4_deit_tiny_int8.onnx"
DEFAULT_INPUT = Path("/root/flexi/eval/data/ptq4_deit_t/model_input.f32.bin")
DEFAULT_OUT = REPO_ROOT / "build/ort_ptq4_firesim_artifacts/ort_ptq4_deit/only_full"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    ap.add_argument("--input-bin", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.onnx.is_file():
        raise SystemExit(f"ONNX not found: {args.onnx} (run export_ptq4_deit_onnx.py first)")
    if not args.input_bin.is_file():
        raise SystemExit(f"input bin not found: {args.input_bin}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_out = args.output_dir / "only_full.onnx"
    shutil.copy2(args.onnx, model_out)

    # Ensure input is f32 [1,3,224,224]
    arr = np.fromfile(args.input_bin, dtype="<f4")
    if arr.size != 1 * 3 * 224 * 224:
        raise SystemExit(f"expected 150528 f32 values, got {arr.size}")
    (args.output_dir / "input.bin").write_bytes(arr.astype("<f4").tobytes())

    m = onnx.load(str(model_out))
    print(f"wrote {model_out} nodes={len(m.graph.node)}")
    print(f"wrote {args.output_dir / 'input.bin'} bytes={arr.nbytes}")


if __name__ == "__main__":
    main()
