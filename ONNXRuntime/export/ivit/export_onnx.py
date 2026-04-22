#!/usr/bin/env python3
"""
Export ONNX models for ONNX Runtime + Gemmini Spike flow.

Supported model names:
  - deit_tiny_patch16_224
  - swin_tiny_patch4_window7_224
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ONNXRT_DIR = SCRIPT_DIR.parent.parent
REPO_ROOT = ONNXRT_DIR.parent
DEFAULT_BUILD_DIR = REPO_ROOT / "build" / "ort"
DEIT_EXPORTER = SCRIPT_DIR / "export_deit_ivit_onnx.py"
SWIN_EXPORTER = SCRIPT_DIR / "export_swin_ivit_onnx.py"
IVIT_PYTORCH_ROOT = REPO_ROOT / "I-ViT"
SWIN_DEPTHS_FIXED = (2, 2, 6, 2)


def _load_ckpt_state_dict(path: Path):
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

    if state_dict and all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _print_onnx_op_counts(model_path: Path, label: str) -> Counter:
    import onnx

    model = onnx.load(str(model_path))
    counts = Counter(node.op_type for node in model.graph.node)
    print(f"\n[{label}] {model_path}")
    print(f"  Total nodes: {len(model.graph.node)}")
    for op in (
        "MatMulInteger",
        "QLinearMatMul",
        "QLinearConv",
        "QLayernorm",
        "Shiftmax",
        "ShiftGELU",
        "MatMul",
        "Conv",
    ):
        print(f"  {op:14s}: {counts.get(op, 0)}")
    return counts


def export_deit(checkpoint: Path | None, output: Path) -> None:
    if not DEIT_EXPORTER.is_file():
        raise FileNotFoundError(f"DeiT exporter not found: {DEIT_EXPORTER}")

    cmd = [sys.executable, str(DEIT_EXPORTER), "--output", str(output)]
    if checkpoint is not None:
        cmd.extend(["--checkpoint", str(checkpoint)])
    print("Running DeiT I-ViT exporter:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    _print_onnx_op_counts(output, "DeiT INT8")


def export_swin_custom(
    checkpoint: Path | None,
    output: Path,
    allow_random_init: bool,
) -> None:
    if not SWIN_EXPORTER.is_file():
        raise FileNotFoundError(f"Swin exporter not found: {SWIN_EXPORTER}")

    cmd = [
        sys.executable,
        str(SWIN_EXPORTER),
        "--output",
        str(output),
    ]
    if checkpoint is not None:
        cmd.extend(["--checkpoint", str(checkpoint)])
    if allow_random_init:
        cmd.append("--allow-random-init")

    print("Running Swin I-ViT custom-op exporter:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    _print_onnx_op_counts(output, "Swin-T INT8 (custom-op)")


def _lift_swin_matmul_rhs_to_initializers(
    src_model: Path,
    dst_model: Path,
    probe_runs: int = 2,
) -> dict[str, int]:
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import helper, numpy_helper

    model = onnx.load(str(src_model))
    init_names = {t.name for t in model.graph.initializer}
    matmul_nodes = [n for n in model.graph.node if n.op_type == "MatMul" and len(n.input) >= 2]
    rhs_candidates = []
    seen = set()
    for node in matmul_nodes:
        rhs = node.input[1]
        if rhs in init_names or rhs in seen:
            continue
        rhs_candidates.append(rhs)
        seen.add(rhs)

    if not rhs_candidates:
        onnx.save(model, str(dst_model))
        return {"total_matmul": len(matmul_nodes), "rhs_candidates": 0, "constants": 0, "replaced": 0}

    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    for rhs in rhs_candidates:
        probe.graph.output.append(helper.make_empty_tensor_value_info(rhs))

    with tempfile.NamedTemporaryFile(prefix="swin_rhs_probe_", suffix=".onnx", delete=False) as tf:
        probe_path = Path(tf.name)
    onnx.save(probe, str(probe_path))

    try:
        sess = ort.InferenceSession(str(probe_path), providers=["CPUExecutionProvider"])
        input_meta = sess.get_inputs()[0]
        input_shape = [
            d if isinstance(d, int) and d > 0 else 1
            for d in input_meta.shape
        ]

        rng = np.random.default_rng(0)
        runs = []
        for _ in range(max(2, probe_runs)):
            feed = {input_meta.name: rng.standard_normal(input_shape).astype(np.float32)}
            runs.append(sess.run(rhs_candidates, feed))
    finally:
        probe_path.unlink(missing_ok=True)

    const_rhs: dict[str, np.ndarray] = {}
    for idx, rhs in enumerate(rhs_candidates):
        ref = runs[0][idx]
        if all(np.array_equal(ref, out[idx]) for out in runs[1:]):
            const_rhs[rhs] = ref

    used_names = set(init_names)
    used_names.update(i.name for i in model.graph.input)
    used_names.update(o.name for o in model.graph.output)
    for node in model.graph.node:
        used_names.update(node.output)

    replaced = 0
    for node in matmul_nodes:
        rhs = node.input[1]
        if rhs not in const_rhs:
            continue

        base = (node.name or f"MatMul_{replaced}").replace("/", "_")
        new_name = f"{base}_rhs_const"
        suffix = 0
        while new_name in used_names:
            suffix += 1
            new_name = f"{base}_rhs_const_{suffix}"
        used_names.add(new_name)

        rhs_arr = const_rhs[rhs]
        if rhs_arr.dtype != np.float32:
            rhs_arr = rhs_arr.astype(np.float32)
        model.graph.initializer.append(numpy_helper.from_array(rhs_arr, name=new_name))
        node.input[1] = new_name
        replaced += 1

    onnx.save(model, str(dst_model))
    return {
        "total_matmul": len(matmul_nodes),
        "rhs_candidates": len(rhs_candidates),
        "constants": len(const_rhs),
        "replaced": replaced,
    }


def _quantize_swin_static_qoperator(
    src_model: Path,
    dst_model: Path,
    calibration_samples: int,
) -> list[str]:
    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    sess = ort.InferenceSession(str(src_model), providers=["CPUExecutionProvider"])
    input_meta = sess.get_inputs()[0]
    input_shape = [
        d if isinstance(d, int) and d > 0 else 1
        for d in input_meta.shape
    ]

    class _RandomCalibrationDataReader(CalibrationDataReader):
        def __init__(self, input_name: str, shape: list[int], num_samples: int):
            rng = np.random.default_rng(1)
            self._items = [
                {input_name: rng.standard_normal(shape).astype(np.float32)}
                for _ in range(max(1, num_samples))
            ]
            self._idx = 0

        def get_next(self):
            if self._idx >= len(self._items):
                return None
            item = self._items[self._idx]
            self._idx += 1
            return item

    op_type_candidates = [
        ["MatMul", "Gemm", "Conv"],
        ["MatMul", "Gemm"],
    ]
    last_error: Exception | None = None

    for op_types in op_type_candidates:
        try:
            quantize_static(
                str(src_model),
                str(dst_model),
                _RandomCalibrationDataReader(input_meta.name, input_shape, calibration_samples),
                quant_format=QuantFormat.QOperator,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=False,
                op_types_to_quantize=op_types,
                calibrate_method=CalibrationMethod.MinMax,
            )
            return op_types
        except Exception as ex:
            last_error = ex
            if "Conv" in op_types:
                print(
                    "[WARN] Swin Conv quantization failed; retrying with MatMul/Gemm only. "
                    f"Reason: {ex}"
                )
                continue
            raise

    if last_error is not None:
        raise last_error
    return []


def _force_qlinear_zero_points_to_zero(
    src_model: Path,
    dst_model: Path,
) -> int:
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(src_model))
    init = {t.name: t for t in model.graph.initializer}
    updated = 0

    zero_point_indices_by_op = {
        "QLinearMatMul": (2, 5, 7),
        "QLinearConv": (2, 5, 7),
    }

    for node in model.graph.node:
        zp_indices = zero_point_indices_by_op.get(node.op_type)
        if zp_indices is None:
            continue
        max_idx = max(zp_indices)
        if len(node.input) <= max_idx:
            continue
        for idx in zp_indices:
            zp_name = node.input[idx]
            tensor = init.get(zp_name)
            if tensor is None:
                continue
            arr = numpy_helper.to_array(tensor)
            if arr.size == 1 and arr.item() != 0:
                tensor.CopyFrom(numpy_helper.from_array(np.zeros_like(arr), name=zp_name))
                updated += 1

    onnx.save(model, str(dst_model))
    return updated


def export_swin_legacy_qop(
    checkpoint: Path | None,
    fp32_output: Path,
    int8_output: Path | None,
    allow_random_init: bool,
    depths: tuple[int, int, int, int],
    enable_rhs_lift: bool,
    force_zero_points: bool,
    calibration_samples: int,
) -> None:
    import torch
    from functools import partial

    sys.path.insert(0, str(IVIT_PYTORCH_ROOT))
    from models.model_utils import freeze_model
    from models.quantization_utils import IntLayerNorm
    from models.swin_quant import SwinTransformer

    model = SwinTransformer(
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=depths,
        num_heads=(3, 6, 12, 24),
        norm_layer=partial(IntLayerNorm, eps=1e-6),
    )
    model.eval()

    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        state_dict = _load_ckpt_state_dict(checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded Swin-T checkpoint: {checkpoint}")
        print(f"  missing keys   : {len(missing)}")
        print(f"  unexpected keys: {len(unexpected)}")
    else:
        if not allow_random_init:
            raise RuntimeError(
                "Swin-T export requires --checkpoint (or use --allow-random-init)"
            )
        print("[WARN] Exporting Swin-T with random-initialized weights.")

    freeze_model(model)
    dummy = torch.randn(1, 3, 224, 224)

    print(f"Exporting Swin-T FP32 ONNX: {fp32_output}")
    torch.onnx.export(
        model,
        dummy,
        str(fp32_output),
        opset_version=13,
        input_names=["image"],
        output_names=["logits"],
        dynamo=False,  # keep legacy exporter to avoid onnxscript dependency
    )
    _print_onnx_op_counts(fp32_output, "Swin-T FP32")

    if int8_output is None:
        return

    quant_input = fp32_output
    if enable_rhs_lift:
        lifted = int8_output.with_suffix(".lifted.onnx")
        stats = _lift_swin_matmul_rhs_to_initializers(fp32_output, lifted)
        print(
            "Lifted Swin MatMul RHS constants:"
            f" total_matmul={stats['total_matmul']},"
            f" rhs_candidates={stats['rhs_candidates']},"
            f" constants={stats['constants']},"
            f" replaced={stats['replaced']}"
        )
        quant_input = lifted

    raw_quant = int8_output.with_suffix(".qop_raw.onnx") if force_zero_points else int8_output

    print(
        "Quantizing Swin-T ONNX to INT8 QOperator "
        "(I-ViT-like per-tensor flow, "
        f"calibration_samples={calibration_samples}): {raw_quant}"
    )
    quantized_ops = _quantize_swin_static_qoperator(
        src_model=quant_input,
        dst_model=raw_quant,
        calibration_samples=calibration_samples,
    )
    print(f"Quantized op types: {quantized_ops}")

    if force_zero_points:
        updated = _force_qlinear_zero_points_to_zero(raw_quant, int8_output)
        print(f"Forced QLinearMatMul/QLinearConv zero-points to 0 (a/b/y tensors updated: {updated})")
        raw_quant.unlink(missing_ok=True)

    if enable_rhs_lift and quant_input != fp32_output:
        quant_input.unlink(missing_ok=True)

    counts = _print_onnx_op_counts(int8_output, "Swin-T INT8")
    if counts.get("QLinearMatMul", 0) == 0 and counts.get("MatMulInteger", 0) == 0:
        print(
            "[WARN] Swin-T INT8 ONNX does not include QLinearMatMul/MatMulInteger; "
            "Gemmini acceleration may be limited."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export ONNX model for ORT Spike flow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        default="deit_tiny_patch16_224",
        choices=["deit_tiny_patch16_224", "swin_tiny_patch4_window7_224"],
        help="Model to export",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="QAT checkpoint path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output INT8 ONNX path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generated ONNX files",
    )
    parser.add_argument(
        "--fp32-output",
        type=Path,
        default=None,
        help="(Swin only) output FP32 ONNX path before quantization",
    )
    parser.add_argument(
        "--skip-quantize",
        action="store_true",
        help="(Swin only) skip INT8 quantization step",
    )
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="(Swin only) allow export without checkpoint",
    )
    parser.add_argument(
        "--swin-calib-samples",
        type=int,
        default=4,
        help="(Swin only) random calibration samples for static quantization",
    )
    parser.add_argument(
        "--swin-export-style",
        default="custom",
        choices=["custom", "legacy-qop"],
        help="(Swin only) ONNX export path; use 'custom' for DeiT-style ivit custom-op graph",
    )
    parser.add_argument(
        "--no-rhs-lift",
        action="store_true",
        help="(Swin only) disable MatMul RHS constant lifting pass",
    )
    parser.add_argument(
        "--no-force-zp0",
        action="store_true",
        help="(Swin only) disable forcing QLinearMatMul/QLinearConv zero-points to zero",
    )
    args = parser.parse_args()

    build_dir = args.output_dir or DEFAULT_BUILD_DIR
    build_dir.mkdir(parents=True, exist_ok=True)

    if args.model_name == "deit_tiny_patch16_224":
        output = args.output or (build_dir / "ivit_tiny_int8.onnx")
        export_deit(args.checkpoint, output)
        print("\nDeiT export done.")
        return 0

    depths = SWIN_DEPTHS_FIXED
    print(f"Swin depths (fixed): {depths}")

    if args.swin_export_style == "custom":
        if args.skip_quantize:
            raise ValueError("--skip-quantize is only valid for --swin-export-style legacy-qop")
        if args.fp32_output is not None:
            raise ValueError("--fp32-output is only valid for --swin-export-style legacy-qop")
        if args.no_rhs_lift or args.no_force_zp0:
            raise ValueError("--no-rhs-lift/--no-force-zp0 are only valid for legacy-qop")
        if args.swin_calib_samples != 4:
            raise ValueError("--swin-calib-samples is only valid for --swin-export-style legacy-qop")

        output = args.output or (build_dir / "swin_tiny_int8.onnx")
        export_swin_custom(
            checkpoint=args.checkpoint,
            output=output,
            allow_random_init=args.allow_random_init,
        )
        print("\nSwin-T custom-op export done.")
        return 0

    fp32_output = args.fp32_output or (build_dir / "swin_tiny_fp32.onnx")
    int8_output = None if args.skip_quantize else (args.output or (build_dir / "swin_tiny_int8.onnx"))
    export_swin_legacy_qop(
        checkpoint=args.checkpoint,
        fp32_output=fp32_output,
        int8_output=int8_output,
        allow_random_init=args.allow_random_init,
        depths=depths,
        enable_rhs_lift=not args.no_rhs_lift,
        force_zero_points=not args.no_force_zp0,
        calibration_samples=args.swin_calib_samples,
    )
    print("\nSwin-T legacy-qop export done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
