#!/usr/bin/env python3
"""Prepare FQ-DeiT ORT FireSim artifacts aligned with TVM int8 input."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FLEXI_ROOT = Path(os.environ.get("VIT_GEMMINI_ROOT", REPO_ROOT)).expanduser()
EXTRACTOR = DEFAULT_FLEXI_ROOT / "ONNXRuntime" / "profiling" / "extract_onnx_prefix.py"


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


def _element_size_bytes(elem_type: int) -> int:
    if elem_type in (TensorProto.INT8, TensorProto.UINT8, TensorProto.BOOL):
        return 1
    if elem_type in (TensorProto.FLOAT16, TensorProto.INT16, TensorProto.UINT16):
        return 2
    if elem_type in (TensorProto.FLOAT, TensorProto.INT32, TensorProto.UINT32):
        return 4
    if elem_type in (TensorProto.DOUBLE, TensorProto.INT64, TensorProto.UINT64):
        return 8
    raise RuntimeError(f"Unsupported ONNX input element type: {elem_type}")


def prepare_fq_artifacts(
    *,
    onnx_model: Path,
    tvm_main_c: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "only_full.onnx"
    subprocess.run(
        [
            sys.executable,
            str(EXTRACTOR),
            "--model",
            str(onnx_model),
            "--out",
            str(model_path),
            "--input-tensor",
            "input_out",
            "--output-tensor",
            "act_out_out",
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    model = onnx.load(str(model_path))
    input_vi = model.graph.input[0]
    elem_type = input_vi.type.tensor_type.elem_type
    expected = int(np.prod(_static_shape(input_vi), dtype=np.int64)) * _element_size_bytes(elem_type)
    data = _read_tvm_input(tvm_main_c)
    if len(data) != expected:
        raise RuntimeError(
            f"{model_path}: TVM input has {len(data)} bytes, graph expects {expected}"
        )
    (output_dir / "input.bin").write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=REPO_ROOT / "build" / "ort" / "fq_deit_tiny_int8.onnx",
    )
    parser.add_argument(
        "--tvm-main",
        type=Path,
        default=DEFAULT_FLEXI_ROOT / "build" / "tvm_spike_fq_deit_t_random" / "main.c",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "build" / "ort_fq_firesim_artifacts",
    )
    args = parser.parse_args()

    for path in (args.onnx_model, args.tvm_main, EXTRACTOR):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = args.output_root.resolve() / "ort_fq_deit" / "only_full"
    prepare_fq_artifacts(
        onnx_model=args.onnx_model.resolve(),
        tvm_main_c=args.tvm_main.resolve(),
        output_dir=output_dir,
    )
    print(f"FQ ORT FireSim artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
