#!/usr/bin/env python3
"""Validate I-ViT ORT custom ops against Python reference implementations."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_LIB = REPO_ROOT / "build" / "ort" / "ort_ivit_ops" / "libivit_ops_host.so"


def shift_exp(data: np.ndarray, x0: int, n: int) -> np.ndarray:
    data = data.astype(np.int32, copy=True)
    data = data + (data >> 1) - (data >> 4)
    floor_val = n * x0
    data = np.maximum(data, floor_val)
    q = np.trunc(data / x0).astype(np.int32)
    r = data - q * x0
    exp_int = (r >> 1) - x0
    shift = np.maximum(n - q, 0)
    out = np.zeros_like(exp_int, dtype=np.int64)
    valid = shift <= 31
    out[valid] = exp_int[valid].astype(np.int64) << shift[valid]
    return out.astype(np.int64)


def ref_shiftmax(x: np.ndarray, x0: int) -> np.ndarray:
    x32 = x.astype(np.int32)
    mx = np.max(x32, axis=-1, keepdims=True)
    exp_buf = shift_exp(x32 - mx, x0, 16)
    exp_sum = np.sum(exp_buf, axis=-1, keepdims=True)
    exp_sum = np.maximum(exp_sum, 1)
    factor = np.int64(0x7FFFFFFF) // exp_sum
    out = (((factor * exp_buf) + (np.int64(1) << 23)) >> 24).clip(-128, 127)
    return out.astype(np.int8)


def ref_shiftgelu(x: np.ndarray, x0: int) -> np.ndarray:
    x32 = x.astype(np.int32)
    mx = np.max(x32, axis=-1, keepdims=True)
    exp_buf = shift_exp(x32 - mx, x0, 23)
    exp_max_neg = shift_exp(-mx, x0, 23)
    exp_sum = np.maximum(exp_buf + exp_max_neg, 1)
    sig = ((np.int64(0x7FFFFFFF) // exp_sum) * exp_buf) >> 24
    return (x32.astype(np.int64) * sig).astype(np.int32)


def ref_qlayernorm(x: np.ndarray, bias: np.ndarray, out_dtype=np.int32) -> np.ndarray:
    x64 = x.astype(np.int64, copy=False)
    c = x64.shape[-1]
    mean = np.rint(np.sum(x64, axis=-1, keepdims=True, dtype=np.int64).astype(np.float32) / c).astype(np.int64)
    centered64 = x64 - mean
    var = np.sum(centered64 * centered64, axis=-1, keepdims=True, dtype=np.int64)

    std = np.full_like(var, 1 << 16, dtype=np.int64)
    safe_var = np.where(var == 0, 1, var).astype(np.int64, copy=False)
    for _ in range(10):
        std = (std + (safe_var // std)) // 2
    std = np.maximum(std, 1)

    norm_scale = (np.int64(0x7FFFFFFF) // std).astype(np.int64, copy=False)
    out = ((norm_scale * centered64) // 2) + bias.astype(np.int64)
    return out.astype(out_dtype)


def ref_requantize_int32(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scaled = x.astype(np.float32) * scale.astype(np.float32)
    rounded = np.rint(scaled)
    return np.clip(rounded, -128, 127).astype(np.int8)


def round_half_up(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.copysign(np.floor(np.abs(x) + 0.5), x)


def ref_requantize_fixedpoint(x: np.ndarray, scale: np.ndarray, out_dtype) -> np.ndarray:
    scale64 = np.asarray(scale, dtype=np.float64)
    mant, exp = np.frexp(scale64)
    mant = round_half_up(np.ldexp(mant, 31)).astype(np.int64)
    exp = (31 - exp).astype(np.int64)
    out = np.rint(x.astype(np.float64) * mant.astype(np.float64) / np.exp2(exp.astype(np.float64)))
    info = np.iinfo(out_dtype)
    return np.clip(out, info.min, info.max).astype(out_dtype)


def ref_repq_log_quant(x: np.ndarray, delta: float, n_bits: int) -> np.ndarray:
    levels = 2 ** n_bits
    x = x.astype(np.float32, copy=False)
    out = np.zeros_like(x, dtype=np.float32)
    positive = x > 0
    q = np.rint(-np.log2(x[positive] / delta) * 2.0)
    valid = q < levels
    q = np.clip(q, 0, levels - 1).astype(np.int32)
    odd_mask = np.where(q % 2 == 0, 1.0, np.float32(np.sqrt(2.0)))
    dequant = np.exp2(-np.ceil(q.astype(np.float32) / 2.0)) * odd_mask * np.float32(delta)
    tmp = np.zeros_like(q, dtype=np.float32)
    tmp[valid] = dequant[valid]
    out[positive] = tmp
    return out


def build_session(model: onnx.ModelProto, custom_op_lib: Path) -> ort.InferenceSession:
    sess_options = ort.SessionOptions()
    sess_options.register_custom_ops_library(str(custom_op_lib))
    return ort.InferenceSession(
        model.SerializeToString(),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


def run_and_check(
    name: str,
    model: onnx.ModelProto,
    feeds: dict[str, np.ndarray],
    output_name: str,
    reference: np.ndarray,
    custom_op_lib: Path,
) -> None:
    session = build_session(model, custom_op_lib)
    output = session.run([output_name], feeds)[0]
    if not np.array_equal(output, reference):
        diff = np.abs(output.astype(np.int64) - reference.astype(np.int64))
        raise AssertionError(
            f"{name} mismatch: max_diff={diff.max()} shape={output.shape}"
        )
    print(f"[PASS] {name}")


def make_shiftmax_model(x0: int, shape: list[int]) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.INT8, shape)
    y = helper.make_tensor_value_info("y", TensorProto.INT8, shape)
    node = helper.make_node("Shiftmax", ["x"], ["y"], domain="ivit", x0=x0)
    graph = helper.make_graph([node], "shiftmax", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("ivit", 1)])


def make_shiftmax_int32_model(x0: int, shape: list[int]) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.INT32, shape)
    y = helper.make_tensor_value_info("y", TensorProto.INT8, shape)
    node = helper.make_node("ShiftmaxInt32", ["x"], ["y"], domain="ivit", x0=x0)
    graph = helper.make_graph([node], "shiftmax_int32", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("ivit", 1)])


def make_shiftgelu_model(x0: int, shape: list[int]) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.INT8, shape)
    y = helper.make_tensor_value_info("y", TensorProto.INT32, shape)
    node = helper.make_node("ShiftGELU", ["x"], ["y"], domain="ivit", x0=x0)
    graph = helper.make_graph([node], "shiftgelu", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("ivit", 1)])


def make_qlayernorm_model(shape: list[int], channels: int) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.INT8, shape)
    bias = helper.make_tensor_value_info("bias", TensorProto.INT32, [channels])
    y = helper.make_tensor_value_info("y", TensorProto.INT32, shape)
    node = helper.make_node("QLayernorm", ["x", "bias"], ["y"], domain="ivit")
    graph = helper.make_graph([node], "qlayernorm", [x, bias], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("ivit", 1)])


def make_qlayernorm_int32_model(shape: list[int], channels: int) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.INT32, shape)
    bias = helper.make_tensor_value_info("bias", TensorProto.INT32, [channels])
    y = helper.make_tensor_value_info("y", TensorProto.INT32, shape)
    node = helper.make_node("QLayernormInt32", ["x", "bias"], ["y"], domain="ivit")
    graph = helper.make_graph([node], "qlayernorm_int32", [x, bias], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("ivit", 1)])


def make_qlayernorm_i64_model(shape: list[int], channels: int) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.INT8, shape)
    bias = helper.make_tensor_value_info("bias", TensorProto.INT64, [channels])
    y = helper.make_tensor_value_info("y", TensorProto.INT64, shape)
    node = helper.make_node("QLayernormI64", ["x", "bias"], ["y"], domain="ivit")
    graph = helper.make_graph([node], "qlayernorm_i64", [x, bias], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("ivit", 1)])


def make_qlayernorm_int32_i64_model(shape: list[int], channels: int) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.INT32, shape)
    bias = helper.make_tensor_value_info("bias", TensorProto.INT64, [channels])
    y = helper.make_tensor_value_info("y", TensorProto.INT64, shape)
    node = helper.make_node("QLayernormInt32I64", ["x", "bias"], ["y"], domain="ivit")
    graph = helper.make_graph([node], "qlayernorm_int32_i64", [x, bias], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("ivit", 1)])


def make_requantize_model(x_shape: list[int], scale_shape: list[int]) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.INT32, x_shape)
    scale = helper.make_tensor_value_info("scale", TensorProto.FLOAT, scale_shape)
    y = helper.make_tensor_value_info("y", TensorProto.INT8, x_shape)
    node = helper.make_node("RequantizeInt32", ["x", "scale"], ["y"], domain="ivit")
    graph = helper.make_graph([node], "requantize", [x, scale], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("ivit", 1)])


def make_requantize_int64_model(x_shape: list[int], scale_shape: list[int], out_dtype: int) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.INT64, x_shape)
    scale = helper.make_tensor_value_info("scale", TensorProto.FLOAT, scale_shape)
    y = helper.make_tensor_value_info("y", out_dtype, x_shape)
    op_name = "RequantizeInt64ToInt16" if out_dtype == TensorProto.INT16 else "RequantizeInt64"
    node = helper.make_node(op_name, ["x", "scale"], ["y"], domain="ivit")
    graph = helper.make_graph([node], "requantize_int64", [x, scale], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("ivit", 1)])


def make_repq_log_quant_model(x_shape: list[int], n_bits: int) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)
    delta = helper.make_tensor_value_info("delta", TensorProto.FLOAT, [])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, x_shape)
    node = helper.make_node("RepQLogQuant", ["x", "delta"], ["y"], domain="ivit", n_bits=n_bits)
    graph = helper.make_graph([node], "repq_log_quant", [x, delta], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("ivit", 1)])


def make_matmul_model(a_shape: list[int], b: np.ndarray) -> onnx.ModelProto:
    a = helper.make_tensor_value_info("a", TensorProto.INT8, a_shape)
    y_shape = list(a_shape[:-1])
    y_shape.append(int(b.shape[-1]))
    y = helper.make_tensor_value_info("y", TensorProto.INT32, y_shape)
    b_info = helper.make_tensor_value_info("b", TensorProto.INT8, list(b.shape))
    a_zp = numpy_helper.from_array(np.array(0, dtype=np.int8), name="a_zp")
    b_zp = numpy_helper.from_array(np.array(0, dtype=np.int8), name="b_zp")
    node = helper.make_node("GemminiMatMulInteger", ["a", "b", "a_zp", "b_zp"], ["y"], domain="ivit")
    graph = helper.make_graph([node], "gemmini_matmul", [a, b_info], [y], initializer=[a_zp, b_zp])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("ivit", 1)])


def validate(custom_op_lib: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)

    softmax_scale = 0.0625
    softmax_x0 = int(-1.0 / softmax_scale - 1.0)
    softmax_x = rng.integers(-128, 128, size=(2, 3, 17), dtype=np.int8)
    run_and_check(
        "Shiftmax",
        make_shiftmax_model(softmax_x0, [2, 3, 17]),
        {"x": softmax_x},
        "y",
        ref_shiftmax(softmax_x, softmax_x0),
        custom_op_lib,
    )

    softmax_x32 = rng.integers(-(1 << 20), 1 << 20, size=(2, 3, 17), dtype=np.int32)
    run_and_check(
        "ShiftmaxInt32",
        make_shiftmax_int32_model(softmax_x0, [2, 3, 17]),
        {"x": softmax_x32},
        "y",
        ref_shiftmax(softmax_x32, softmax_x0),
        custom_op_lib,
    )

    gelu_scale = 0.09375
    gelu_x0 = int(-1.0 / (gelu_scale * 1.702) - 1.0)
    gelu_x = rng.integers(-128, 128, size=(4, 19), dtype=np.int8)
    run_and_check(
        "ShiftGELU",
        make_shiftgelu_model(gelu_x0, [4, 19]),
        {"x": gelu_x},
        "y",
        ref_shiftgelu(gelu_x, gelu_x0),
        custom_op_lib,
    )

    ln_x = rng.integers(-128, 128, size=(3, 5, 16), dtype=np.int8)
    ln_bias = rng.integers(-(1 << 20), 1 << 20, size=(16,), dtype=np.int32)
    run_and_check(
        "QLayernorm",
        make_qlayernorm_model([3, 5, 16], 16),
        {"x": ln_x, "bias": ln_bias},
        "y",
        ref_qlayernorm(ln_x, ln_bias),
        custom_op_lib,
    )

    ln_x32 = rng.integers(-(1 << 20), 1 << 20, size=(3, 5, 16), dtype=np.int32)
    run_and_check(
        "QLayernormInt32",
        make_qlayernorm_int32_model([3, 5, 16], 16),
        {"x": ln_x32, "bias": ln_bias},
        "y",
        ref_qlayernorm(ln_x32, ln_bias),
        custom_op_lib,
    )

    ln_bias64 = rng.integers(-(1 << 40), 1 << 40, size=(16,), dtype=np.int64)
    run_and_check(
        "QLayernormI64",
        make_qlayernorm_i64_model([3, 5, 16], 16),
        {"x": ln_x, "bias": ln_bias64},
        "y",
        ref_qlayernorm(ln_x, ln_bias64, out_dtype=np.int64),
        custom_op_lib,
    )

    run_and_check(
        "QLayernormInt32I64",
        make_qlayernorm_int32_i64_model([3, 5, 16], 16),
        {"x": ln_x32, "bias": ln_bias64},
        "y",
        ref_qlayernorm(ln_x32, ln_bias64, out_dtype=np.int64),
        custom_op_lib,
    )

    rq_x = rng.integers(-(1 << 20), 1 << 20, size=(2, 7, 11), dtype=np.int32)
    rq_scale_scalar = np.array(0.0007324219, dtype=np.float32)
    run_and_check(
        "RequantizeInt32 scalar",
        make_requantize_model([2, 7, 11], []),
        {"x": rq_x, "scale": rq_scale_scalar},
        "y",
        ref_requantize_int32(rq_x, rq_scale_scalar),
        custom_op_lib,
    )

    rq_scale_vec = rng.uniform(1e-4, 2e-3, size=(11,)).astype(np.float32)
    run_and_check(
        "RequantizeInt32 per-channel",
        make_requantize_model([2, 7, 11], [11]),
        {"x": rq_x, "scale": rq_scale_vec},
        "y",
        ref_requantize_int32(rq_x, rq_scale_vec.reshape(1, 1, -1)),
        custom_op_lib,
    )

    rq_x64 = rng.integers(-(1 << 30), 1 << 30, size=(2, 7, 11), dtype=np.int64)
    rq64_scale_scalar = np.array(0.0007324219, dtype=np.float32)
    run_and_check(
        "RequantizeInt64 scalar",
        make_requantize_int64_model([2, 7, 11], [], TensorProto.INT8),
        {"x": rq_x64, "scale": rq64_scale_scalar},
        "y",
        ref_requantize_fixedpoint(rq_x64, rq64_scale_scalar, np.int8),
        custom_op_lib,
    )

    rq64_scale_vec = rng.uniform(1e-5, 2e-3, size=(11,)).astype(np.float32)
    run_and_check(
        "RequantizeInt64ToInt16 per-channel",
        make_requantize_int64_model([2, 7, 11], [11], TensorProto.INT16),
        {"x": rq_x64, "scale": rq64_scale_vec},
        "y",
        ref_requantize_fixedpoint(rq_x64, rq64_scale_vec.reshape(1, 1, -1), np.int16),
        custom_op_lib,
    )

    repq_x = rng.uniform(1e-4, 1.0, size=(2, 3, 17)).astype(np.float32)
    repq_delta = np.array(0.875, dtype=np.float32)
    run_and_check(
        "RepQLogQuant",
        make_repq_log_quant_model([2, 3, 17], 8),
        {"x": repq_x, "delta": repq_delta},
        "y",
        ref_repq_log_quant(repq_x, float(repq_delta), 8),
        custom_op_lib,
    )

    mm_a2 = rng.integers(-128, 128, size=(13, 32), dtype=np.int8)
    mm_b2 = rng.integers(-128, 128, size=(32, 9), dtype=np.int8)
    run_and_check(
        "GemminiMatMulInteger rank2",
        make_matmul_model([13, 32], mm_b2),
        {"a": mm_a2, "b": mm_b2},
        "y",
        (mm_a2.astype(np.int32) @ mm_b2.astype(np.int32)).astype(np.int32),
        custom_op_lib,
    )

    mm_a3 = rng.integers(-128, 128, size=(2, 7, 32), dtype=np.int8)
    mm_b3 = rng.integers(-128, 128, size=(32, 15), dtype=np.int8)
    ref_mm3 = np.matmul(mm_a3.astype(np.int32), mm_b3.astype(np.int32)).astype(np.int32)
    run_and_check(
        "GemminiMatMulInteger rank3",
        make_matmul_model([2, 7, 32], mm_b3),
        {"a": mm_a3, "b": mm_b3},
        "y",
        ref_mm3,
        custom_op_lib,
    )

    mm_a4 = rng.integers(-128, 128, size=(2, 3, 5, 16), dtype=np.int8)
    mm_b4 = rng.integers(-128, 128, size=(2, 3, 16, 9), dtype=np.int8)
    ref_mm4 = np.matmul(mm_a4.astype(np.int32), mm_b4.astype(np.int32)).astype(np.int32)
    run_and_check(
        "GemminiMatMulInteger rank4 batched",
        make_matmul_model([2, 3, 5, 16], mm_b4),
        {"a": mm_a4, "b": mm_b4},
        "y",
        ref_mm4,
        custom_op_lib,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate I-ViT ONNX Runtime custom ops against Python references",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--custom-op-lib", type=Path, default=DEFAULT_LIB)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.custom_op_lib.is_file():
        raise FileNotFoundError(
            f"Custom op library not found: {args.custom_op_lib}. Build it with "
            f"`make -C ONNXRuntime/ort_ivit_ops host`."
        )

    validate(args.custom_op_lib, args.seed)
    print("All custom op checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
