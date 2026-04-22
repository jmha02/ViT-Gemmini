#!/usr/bin/env python3
"""Preprocess an image into a raw float32 tensor for ort_test."""

from __future__ import annotations

import argparse
from pathlib import Path

from ivit_input import write_tensor_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a preprocessed 1x3x224x224 float32 tensor for ORT runner"
    )
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output raw tensor path")
    parser.add_argument(
        "--model-name",
        default="deit_tiny_patch16_224",
        help="Model name used to select preprocessing parameters",
    )
    args = parser.parse_args()

    out = write_tensor_file(args.image, args.output, model_name=args.model_name)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
