#!/usr/bin/env python3
"""Run PyTorch reference inference, with optional host ORT comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

from ivit_model_io import build_reference_model
from ivit_input import preprocess_image


DEFAULT_IMAGE = REPO_ROOT / "scripts" / "gemmini" / "test_cat.jpg"
DEFAULT_LABELS = REPO_ROOT / "scripts" / "gemmini" / "imagenet_classes.txt"
DEFAULT_HOST_CUSTOM_OP_LIB = REPO_ROOT / "build" / "ort" / "ort_ivit_ops" / "libivit_ops_host.so"


def load_labels(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle]


def format_topk(logits: np.ndarray, labels: list[str], topk: int) -> list[tuple[int, str, float]]:
    indices = np.argsort(logits)[::-1][:topk]
    return [(int(idx), labels[int(idx)], float(logits[int(idx)])) for idx in indices]


def print_topk(title: str, rows: list[tuple[int, str, float]]) -> None:
    print(title)
    for rank, (idx, label, score) in enumerate(rows, start=1):
        print(f"  {rank:>2d}. {idx:>3d}  {label:<30s}  {score:.6f}")


def run_pytorch(model_name: str, checkpoint: Path, image: Path) -> tuple[np.ndarray, list[tuple[str, tuple[int, ...], tuple[int, ...]]], object]:
    model, _, resized, incompatible = build_reference_model(model_name, checkpoint)
    input_tensor = torch.from_numpy(preprocess_image(image))
    with torch.no_grad():
        logits = model(input_tensor).detach().cpu().numpy()[0]
    return logits, resized, incompatible


def run_ort(model_path: Path, image: Path, custom_op_library: Path) -> np.ndarray:
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.register_custom_ops_library(str(custom_op_library))
    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    logits = session.run(None, {"image": preprocess_image(image)})[0][0]
    return np.asarray(logits, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run I-ViT PyTorch reference inference and optional host ORT comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", required=True, choices=["deit_tiny_patch16_224", "swin_tiny_patch4_window7_224"])
    parser.add_argument("--checkpoint", type=Path, required=True, help="QAT checkpoint path")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Input image path")
    parser.add_argument("--topk", type=int, default=5, help="Number of top classes to print")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="ImageNet class-name file")
    parser.add_argument("--onnx", type=Path, default=None, help="Optional ONNX model for host ORT comparison")
    parser.add_argument("--custom-op-library", type=Path, default=DEFAULT_HOST_CUSTOM_OP_LIB, help="Host custom-op shared library")
    args = parser.parse_args()

    labels = load_labels(args.labels)

    print(f"Model      : {args.model_name}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Image      : {args.image}")
    pt_logits, resized, incompatible = run_pytorch(args.model_name, args.checkpoint, args.image)
    print(f"Resized scale buffers: {len(resized)}")
    print(f"Missing keys         : {len(incompatible.missing_keys)}")
    print(f"Unexpected keys      : {len(incompatible.unexpected_keys)}")
    print_topk("\nPyTorch reference top-k:", format_topk(pt_logits, labels, args.topk))

    if args.onnx is None:
        return 0
    if not args.custom_op_library.is_file():
        raise FileNotFoundError(f"Host custom-op library not found: {args.custom_op_library}")

    ort_logits = run_ort(args.onnx, args.image, args.custom_op_library)
    print_topk("\nHost ORT top-k:", format_topk(ort_logits, labels, args.topk))

    pt_top1 = int(np.argmax(pt_logits))
    ort_top1 = int(np.argmax(ort_logits))
    pt_top5 = set(np.argsort(pt_logits)[::-1][: args.topk].tolist())
    ort_top5 = set(np.argsort(ort_logits)[::-1][: args.topk].tolist())
    print("\nComparison:")
    print(f"  top-1 match   : {pt_top1 == ort_top1}")
    print(f"  top-{args.topk} overlap: {len(pt_top5 & ort_top5)}/{args.topk}")
    print(f"  max|diff|     : {float(np.max(np.abs(pt_logits - ort_logits))):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
