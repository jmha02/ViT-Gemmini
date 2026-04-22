#!/usr/bin/env python3
"""Run a RepQ ONNX model with host ORT and print ImageNet top-k."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ivit_input import preprocess_image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_IMAGE = REPO_ROOT / "scripts" / "gemmini" / "test_cat.jpg"
DEFAULT_LABELS = REPO_ROOT / "scripts" / "gemmini" / "imagenet_classes.txt"
DEFAULT_HOST_CUSTOM_OP_LIB = REPO_ROOT / "build" / "ort" / "ort_ivit_ops" / "libivit_ops_host.so"


def load_labels(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run RepQ ONNX inference on host ORT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, required=True, help="ONNX model path")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Input image path")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="ImageNet labels path")
    parser.add_argument("--topk", type=int, default=5, help="Number of top classes to print")
    parser.add_argument(
        "--model-name",
        type=str,
        default="deit_tiny_patch16_224",
        help="Model name used to select preprocessing parameters",
    )
    parser.add_argument(
        "--custom-op-library",
        type=Path,
        default=DEFAULT_HOST_CUSTOM_OP_LIB,
        help="Optional host custom-op shared library",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.labels.is_file():
        raise FileNotFoundError(f"Labels not found: {args.labels}")

    sess_options = ort.SessionOptions()
    if args.custom_op_library.is_file():
        sess_options.register_custom_ops_library(str(args.custom_op_library))

    session = ort.InferenceSession(
        str(args.model),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: preprocess_image(args.image, model_name=args.model_name)})[0][0]

    labels = load_labels(args.labels)
    indices = np.argsort(logits)[::-1][: args.topk]
    for rank, index in enumerate(indices, start=1):
        print(f"{rank:>2d}. {int(index):>3d}  {labels[int(index)]:<30s}  {float(logits[int(index)]):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
