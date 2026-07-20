#!/usr/bin/env python3
"""Prepare TVM-aligned ORT FireSim artifacts for I-DeiT-S and I-Swin-S.

The Small ONNX exports take float32 ``image`` inputs and start with a
QuantizeLinear node. The TVM FireSim binaries, like the Tiny ORT FireSim
success cases, use the quantized image boundary. This script extracts
``input_quant -> logits`` subgraphs and writes the exact TVM input bytes.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FLEXI_ROOT = Path("/root/flexi/third-party/I-ViT-Gemmini")


def _read_tvm_input(main_c: Path) -> bytes:
    text = main_c.read_text()
    size_match = re.search(r"#define INPUT_SIZE_BYTES \((\d+)\)", text)
    array_match = re.search(
        r"static const uint8_t input_data\[INPUT_SIZE_BYTES\].*?= \{(.*?)\};",
        text,
        flags=re.DOTALL,
    )
    if size_match is None or array_match is None:
        raise RuntimeError(f"Could not parse TVM input_data from {main_c}")
    values = bytes(int(token) for token in re.findall(r"\b\d+\b", array_match.group(1)))
    expected = int(size_match.group(1))
    if len(values) != expected:
        raise RuntimeError(f"{main_c}: parsed {len(values)} bytes, expected {expected}")
    return values


def _static_shape(value_info: onnx.ValueInfoProto) -> list[int]:
    shape: list[int] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            raise RuntimeError(f"Dynamic input shape is unsupported: {value_info.name}")
        shape.append(dim.dim_value)
    return shape


def _check_int8_input(model_path: Path) -> int:
    model = onnx.load(str(model_path))
    if len(model.graph.input) != 1:
        raise RuntimeError(f"{model_path}: expected one graph input")
    graph_input = model.graph.input[0]
    elem_type = graph_input.type.tensor_type.elem_type
    if elem_type != TensorProto.INT8:
        raise RuntimeError(f"{model_path}: graph input is {elem_type}, expected INT8")
    return int(np.prod(_static_shape(graph_input), dtype=np.int64))


def _extract_full(
    extractor: Path,
    base_model: Path,
    output_dir: Path,
    tvm_main_c: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "only_full.onnx"
    subprocess.run(
        [
            sys.executable,
            str(extractor),
            "--model",
            str(base_model),
            "--out",
            str(model_path),
            "--input-tensor",
            "input_quant",
            "--output-tensor",
            "logits",
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    expected = _check_int8_input(model_path)
    data = _read_tvm_input(tvm_main_c)
    if len(data) != expected:
        raise RuntimeError(
            f"{model_path}: TVM input has {len(data)} bytes, graph expects {expected}"
        )
    (output_dir / "input.bin").write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flexi-root", type=Path, default=DEFAULT_FLEXI_ROOT)
    parser.add_argument(
        "--deit-model",
        type=Path,
        default=DEFAULT_FLEXI_ROOT / "build" / "ort" / "ivit_deit_small_int8.onnx",
    )
    parser.add_argument(
        "--swin-model",
        type=Path,
        default=DEFAULT_FLEXI_ROOT / "build" / "ort" / "swin_small_int8.onnx",
    )
    parser.add_argument(
        "--deit-tvm-main",
        type=Path,
        default=DEFAULT_FLEXI_ROOT / "build" / "tvm_spike_deit_s_random" / "main.c",
    )
    parser.add_argument(
        "--swin-tvm-main",
        type=Path,
        default=REPO_ROOT / "build" / "tvm_spike_swin_s_random_o2" / "main.c",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "build" / "ort_small_firesim_artifacts",
    )
    args = parser.parse_args()

    extractor = args.flexi_root / "scripts" / "onnxrt" / "profiling" / "extract_onnx_prefix.py"
    required = [extractor, args.deit_model, args.swin_model, args.deit_tvm_main, args.swin_tvm_main]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_root = args.output_root.resolve()
    _extract_full(
        extractor,
        args.deit_model.resolve(),
        output_root / "ort_ivit_deit" / "only_full",
        args.deit_tvm_main.resolve(),
    )
    _extract_full(
        extractor,
        args.swin_model.resolve(),
        output_root / "ort_ivit_swin" / "only_full",
        args.swin_tvm_main.resolve(),
    )
    print(f"Small ORT FireSim artifacts written to {output_root}")


if __name__ == "__main__":
    main()
