#!/usr/bin/env python3
"""Shared I-ViT input preprocessing helpers for ONNX Runtime flows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SHAPE = (1, 3, 224, 224)
INPUT_SIZE = 224
MODEL_RESIZE_SHORT_SIDE = {
    "deit": 256,
    "swin": 248,
}


def resolve_resize_short_side(model_name: str | None) -> int:
    if model_name is None:
        return MODEL_RESIZE_SHORT_SIDE["deit"]
    if model_name.startswith("swin"):
        return MODEL_RESIZE_SHORT_SIDE["swin"]
    return MODEL_RESIZE_SHORT_SIDE["deit"]


def preprocess_image(path: str | Path, model_name: str | None = None) -> np.ndarray:
    """Apply the model-specific ImageNet preprocessing pipeline."""
    img = Image.open(path).convert("RGB")
    width, height = img.size
    resize_short_side = resolve_resize_short_side(model_name)
    scale = resize_short_side / min(width, height)
    resized = img.resize((int(width * scale), int(height * scale)), Image.BILINEAR)

    new_width, new_height = resized.size
    left = (new_width - INPUT_SIZE) // 2
    top = (new_height - INPUT_SIZE) // 2
    cropped = resized.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))

    arr = np.asarray(cropped, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
    return np.ascontiguousarray(arr.astype(np.float32))


def write_tensor_file(
    image_path: str | Path,
    out_path: str | Path,
    model_name: str | None = None,
) -> Path:
    """Preprocess an image and write raw float32 NCHW tensor bytes."""
    out_path = Path(out_path)
    tensor = preprocess_image(image_path, model_name=model_name)
    tensor.tofile(out_path)
    return out_path
