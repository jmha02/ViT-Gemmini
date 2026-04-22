from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from tvm import relay
from tvm.relay.frontend.common import infer_shape as _infer_shape


@dataclass
class RepQModuleMeta:
    n_bits: int
    input_scale: Any | None = None
    input_zero_point: Any | None = None
    weight_scale: Any | None = None
    weight_zero_point: Any | None = None
    log_scale: Any | None = None
    b_scale: Any | None = None
    b_zero_point: Any | None = None


class RepQContext:
    meta: dict[str, RepQModuleMeta] = {}
    params: dict[str, np.ndarray] = {}
    batch_matmul_unroll_limit: int = 16
    use_gemmini_ops: bool = True

    @staticmethod
    def set_meta(meta: dict[str, RepQModuleMeta]) -> None:
        RepQContext.meta = meta

    @staticmethod
    def set_params(params: dict[str, np.ndarray]) -> None:
        RepQContext.params = params

    @staticmethod
    def set_artifacts(meta: dict[str, RepQModuleMeta], params: dict[str, np.ndarray]) -> None:
        RepQContext.meta = meta
        RepQContext.params = params

    @staticmethod
    def set_use_gemmini_ops(enabled: bool) -> None:
        RepQContext.use_gemmini_ops = enabled

    @staticmethod
    def get(name: str) -> RepQModuleMeta:
        if name not in RepQContext.meta:
            raise KeyError(f"RepQ metadata not found for module: {name}")
        return RepQContext.meta[name]

    @staticmethod
    def get_param(name: str) -> np.ndarray:
        if name not in RepQContext.params:
            raise KeyError(f"RepQ parameter not found: {name}")
        return RepQContext.params[name]


def sanitize_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_").replace(":", "_")


def _as_array(value: Any, dtype: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(dtype)
    return np.asarray(value, dtype=dtype)


def _const_float32(value: Any) -> relay.Expr:
    return relay.const(_as_array(value, "float32"), "float32")


def _const_int32(value: Any) -> relay.Expr:
    return relay.const(_as_array(value, "int32"), "int32")


def _const_int8(value: Any) -> relay.Expr:
    return relay.const(_as_array(value, "int8"), "int8")


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _broadcast_const(value: Any, rank: int, axis: int, dtype: str) -> relay.Expr:
    array = _as_array(value, dtype)
    if array.ndim == 0 or array.size == 1:
        return relay.const(array.reshape(()), dtype)

    axis_index = axis if axis >= 0 else rank + axis
    if axis_index < 0 or axis_index >= rank:
        raise RuntimeError(f"Invalid broadcast axis {axis} for rank-{rank} tensor")

    shape = [1] * rank
    shape[axis_index] = int(array.reshape(-1).shape[0])
    return relay.const(array.reshape(shape), dtype)


def _reshape_const_2d(value: Any, rows: int, cols: int, dtype: str) -> relay.Expr:
    array = _as_array(value, dtype).reshape(rows, cols)
    return relay.const(array, dtype)


def _scalar_or_row_expr(expr: relay.Expr, value: Any, cols: int, dtype: str) -> relay.Expr:
    array = _as_array(value, dtype).reshape(-1)
    if array.size == 1:
        return relay.reshape(expr, [])
    if array.size != cols:
        raise RuntimeError(f"Expected {cols} elements but got {array.size}")
    return relay.reshape(expr, [1, cols])


def layer_norm_float(data: relay.Expr, weight: relay.Expr, bias: relay.Expr, eps: float = 1e-6) -> relay.Expr:
    mean = relay.mean(data, axis=-1, keepdims=True)
    centered = data - mean
    var = relay.mean(centered * centered, axis=-1, keepdims=True)
    inv_std = relay.rsqrt(var + relay.const(np.float32(eps), "float32"))
    normalized = centered * inv_std
    return normalized * weight + bias


def gelu_float(data: relay.Expr) -> relay.Expr:
    inv_sqrt2 = relay.const(np.float32(1.0 / np.sqrt(2.0)), "float32")
    half = relay.const(np.float32(0.5), "float32")
    one = relay.const(np.float32(1.0), "float32")
    return data * half * (one + relay.erf(data * inv_sqrt2))


def uniform_quant_dequant(
    data: relay.Expr,
    scale: Any,
    zero_point: Any,
    n_bits: int,
    axis: int = -1,
) -> relay.Expr:
    rank = len(_infer_shape(data))
    scale_expr = _broadcast_const(scale, rank, axis, "float32")
    zp_expr = _broadcast_const(zero_point, rank, axis, "float32")
    levels = relay.const(np.float32((1 << n_bits) - 1), "float32")
    quant = _clip_expr(
        relay.round(data / scale_expr) + zp_expr,
        relay.const(0.0, "float32"),
        levels,
    )
    return (quant - zp_expr) * scale_expr


def _clip_expr(data: relay.Expr, clip_min: relay.Expr, clip_max: relay.Expr) -> relay.Expr:
    return relay.minimum(relay.maximum(data, clip_min), clip_max)


def repq_log_quant(data: relay.Expr, delta: Any, n_bits: int) -> relay.Expr:
    levels = float(1 << n_bits)
    delta_expr = _const_float32(delta)
    safe = relay.maximum(data, relay.const(np.float32(1e-12), "float32"))
    x_int = relay.round(relay.negative(relay.log(safe / delta_expr) / relay.log(relay.const(np.float32(2.0), "float32"))) * relay.const(np.float32(2.0), "float32"))
    mask = relay.greater_equal(x_int, relay.const(np.float32(levels), "float32"))
    x_quant = _clip_expr(
        x_int,
        relay.const(0.0, "float32"),
        relay.const(np.float32(levels - 1.0), "float32"),
    )
    odd = relay.equal(relay.mod(x_quant, relay.const(np.float32(2.0), "float32")), relay.const(1.0, "float32"))
    odd_scale = relay.where(
        odd,
        relay.const(np.float32(np.sqrt(2.0)), "float32"),
        relay.const(np.float32(1.0), "float32"),
    )
    exponent = relay.negative(relay.ceil(x_quant / relay.const(np.float32(2.0), "float32")))
    quantized = relay.power(relay.const(np.float32(2.0), "float32"), exponent) * odd_scale * delta_expr
    return relay.where(mask, relay.zeros_like(quantized), quantized)


def _uniform_quant_shift(
    data: relay.Expr,
    scale: Any,
    zero_point: Any,
    n_bits: int,
    axis: int = -1,
) -> tuple[relay.Expr, relay.Expr]:
    rank = len(_infer_shape(data))
    scale_expr = _broadcast_const(scale, rank, axis, "float32")
    zp_expr = _broadcast_const(zero_point, rank, axis, "float32")

    qmin = relay.ceil(relay.negative(zp_expr))
    qmax = relay.floor(relay.const(np.float32((1 << n_bits) - 1), "float32") - zp_expr)
    q = _clip_expr(relay.round(data / scale_expr), qmin, qmax)
    shift_center = qmin + relay.const(np.float32(128.0), "float32")
    shifted = relay.cast(q - shift_center, "int8")
    return shifted, relay.cast(shift_center, "int32")


def _gemmini_gemm_int32(data_int8: relay.Expr, weight_int8_kn: relay.Expr) -> relay.Expr:
    if not RepQContext.use_gemmini_ops:
        weight_nk = relay.transpose(weight_int8_kn, axes=[1, 0])
        units = tuple(int(dim) for dim in _infer_shape(weight_nk))[0]
        return relay.nn.dense(data_int8, weight_nk, units=units, out_dtype="int32")

    from tvm.contrib.gemmini.legalize import gemmini_gemm

    weight_shape = tuple(int(dim) for dim in _infer_shape(weight_int8_kn))
    if len(weight_shape) != 2:
        raise RuntimeError(f"gemmini_gemm expects rank-2 weights, got {weight_shape}")
    units = weight_shape[1]
    zero_bias = relay.const(np.zeros((units,), dtype="int32"), "int32")
    ones = relay.const(np.ones((units,), dtype="float32"), "float32")
    weights_nk = relay.transpose(weight_int8_kn, axes=[1, 0])
    return gemmini_gemm(
        data_int8,
        weights_nk,
        zero_bias,
        relay.const(np.float32(1.0), "float32"),
        relay.const(np.int32(0), "int32"),
        ones,
        relay.const(np.int32(0), "int32"),
        relay.const(np.float32(1.0), "float32"),
        relay.const(np.int32(0), "int32"),
    )


def _safe_scalar(value: float) -> float:
    return float(max(value, 1e-8))


def _sym_scale_from_uniform(scale: Any, zero_point: Any, n_bits: int) -> float:
    scale_arr = _as_array(scale, "float32").reshape(-1)
    zp_arr = _as_array(zero_point, "float32").reshape(-1)
    qmin = -zp_arr
    qmax = float((1 << n_bits) - 1) - zp_arr
    max_abs = np.maximum(np.abs(qmin * scale_arr), np.abs(qmax * scale_arr))
    return _safe_scalar(float(np.max(max_abs)) / 127.0)


def _sym_scale_from_log(delta: Any) -> float:
    delta_arr = _as_array(delta, "float32").reshape(-1)
    return _safe_scalar(float(np.max(np.abs(delta_arr))) / 127.0)


def _quantize_sym_int8_expr(data: relay.Expr, scale: float) -> relay.Expr:
    scaled = relay.round(data / relay.const(np.float32(scale), "float32"))
    clipped = _clip_expr(
        scaled,
        relay.const(np.float32(-127.0), "float32"),
        relay.const(np.float32(127.0), "float32"),
    )
    return relay.cast(clipped, "int8")


def _dequantize_sym_int8_expr(data: relay.Expr, scale: float) -> relay.Expr:
    return relay.cast(data, "float32") * relay.const(np.float32(scale), "float32")


def _quantize_sym_int8_array(array: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.round(array / np.float32(scale)), -127, 127).astype("int8")


def _requant_output_scale(a_scale: float, b_scale: float, depth: int) -> float:
    max_accum = max(int(depth), 1) * 127.0 * 127.0
    return _safe_scalar(a_scale * b_scale * (max_accum / 120.0))


def _gemmini_qnn_dense_dequant(
    data_float: relay.Expr,
    weight_int8_nk: relay.Expr,
    input_scale: float,
    weight_scale: float,
    depth: int,
) -> relay.Expr:
    units = int(_infer_shape(weight_int8_nk)[0])
    data_int8 = _quantize_sym_int8_expr(data_float, input_scale)
    dense = relay.qnn.op.dense(
        data_int8,
        weight_int8_nk,
        relay.const(0, "int32"),
        relay.const(0, "int32"),
        relay.const(np.float32(input_scale), "float32"),
        relay.const(np.float32(weight_scale), "float32"),
        units,
        "int32",
    )
    zero_bias = relay.const(np.zeros((units,), dtype="int32"), "int32")
    dense = relay.nn.bias_add(dense, zero_bias, axis=-1)
    dense_scale = _safe_scalar(input_scale * weight_scale)
    output_scale = _requant_output_scale(input_scale, weight_scale, depth)
    requant = relay.qnn.op.requantize(
        dense,
        relay.const(np.float32(dense_scale), "float32"),
        relay.const(0, "int32"),
        relay.const(np.float32(output_scale), "float32"),
        relay.const(0, "int32"),
        axis=-1,
        out_dtype="int8",
    )
    return _dequantize_sym_int8_expr(requant, output_scale)


def _gemmini_batch_matmul_dequant(
    lhs_float: relay.Expr,
    rhs_float: relay.Expr,
    lhs_scale: float,
    rhs_scale: float,
    transpose_b: bool = False,
) -> relay.Expr:
    lhs_int8 = _quantize_sym_int8_expr(lhs_float, lhs_scale)
    rhs_int8 = _quantize_sym_int8_expr(rhs_float, rhs_scale)
    batch_matmul = relay.nn.batch_matmul(
        lhs_int8,
        rhs_int8,
        out_dtype="int32",
        transpose_b=transpose_b,
    )
    return relay.cast(batch_matmul, "float32") * relay.const(
        np.float32(lhs_scale * rhs_scale), "float32"
    )


def _approx_uniform_matmul_2d(
    a: relay.Expr,
    b: relay.Expr,
    a_scale: Any,
    a_zero_point: Any,
    b_scale: Any,
    b_zero_point: Any,
    n_bits: int,
) -> relay.Expr:
    a_shape = tuple(int(dim) for dim in _infer_shape(a))
    b_shape = tuple(int(dim) for dim in _infer_shape(b))
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise RuntimeError("RepQ Gemmini uniform matmul expects rank-2 tensors")
    if a_shape[1] != b_shape[0]:
        raise RuntimeError(f"MatMul shape mismatch: {a_shape} vs {b_shape}")

    a_sym_scale = _sym_scale_from_uniform(a_scale, a_zero_point, n_bits)
    b_sym_scale = _sym_scale_from_uniform(b_scale, b_zero_point, n_bits)
    weight_nk = _quantize_sym_int8_expr(relay.transpose(b, axes=[1, 0]), b_sym_scale)
    return _gemmini_qnn_dense_dequant(a, weight_nk, a_sym_scale, b_sym_scale, depth=a_shape[1])


def _approx_uniform_matmul(
    a: relay.Expr,
    b: relay.Expr,
    a_scale: Any,
    a_zero_point: Any,
    b_scale: Any,
    b_zero_point: Any,
    n_bits: int,
) -> relay.Expr:
    a_shape = tuple(int(dim) for dim in _infer_shape(a))
    b_shape = tuple(int(dim) for dim in _infer_shape(b))

    if len(a_shape) == 2 and len(b_shape) == 2:
        return _approx_uniform_matmul_2d(a, b, a_scale, a_zero_point, b_scale, b_zero_point, n_bits)

    if len(a_shape) == 3 and len(b_shape) == 2:
        batch = a_shape[0]
        a_parts = relay.split(a, batch, axis=0)
        outputs = []
        for idx in range(batch):
            a_i = relay.squeeze(a_parts[idx], axis=[0])
            y_i = _approx_uniform_matmul_2d(
                a_i,
                b,
                a_scale=a_scale,
                a_zero_point=a_zero_point,
                b_scale=b_scale,
                b_zero_point=b_zero_point,
                n_bits=n_bits,
            )
            outputs.append(relay.expand_dims(y_i, axis=0))
        return relay.concatenate(outputs, axis=0)

    if len(a_shape) != 3 or len(b_shape) != 3:
        raise RuntimeError(f"RepQ Gemmini uniform matmul expects rank-2 or rank-3 tensors, got {a_shape} and {b_shape}")
    if a_shape[0] != b_shape[0] or a_shape[2] != b_shape[1]:
        raise RuntimeError(f"RepQ Gemmini batched matmul shape mismatch: {a_shape} vs {b_shape}")

    a_sym_scale = _sym_scale_from_uniform(a_scale, a_zero_point, n_bits)
    b_sym_scale = _sym_scale_from_uniform(b_scale, b_zero_point, n_bits)
    return _gemmini_batch_matmul_dequant(a, b, a_sym_scale, b_sym_scale, transpose_b=False)


def approx_log_matmul(
    attn: relay.Expr,
    value: relay.Expr,
    log_delta: Any,
    value_scale: Any,
    value_zero_point: Any,
    n_bits: int,
) -> relay.Expr:
    attn_shape = tuple(int(dim) for dim in _infer_shape(attn))
    value_shape = tuple(int(dim) for dim in _infer_shape(value))
    if len(attn_shape) != 3 or len(value_shape) != 3:
        raise RuntimeError(f"RepQ Gemmini log matmul expects rank-3 tensors, got {attn_shape} and {value_shape}")
    if attn_shape[0] != value_shape[0] or attn_shape[2] != value_shape[1]:
        raise RuntimeError(f"RepQ Gemmini log matmul shape mismatch: {attn_shape} vs {value_shape}")

    attn_q = repq_log_quant(attn, log_delta, n_bits)
    attn_sym_scale = _sym_scale_from_log(log_delta)
    value_sym_scale = _sym_scale_from_uniform(value_scale, value_zero_point, n_bits)
    return _gemmini_batch_matmul_dequant(
        attn_q,
        value,
        attn_sym_scale,
        value_sym_scale,
        transpose_b=False,
    )


def approx_patchify_conv2d(
    data_nchw: relay.Expr,
    module_name: str,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    out_channels: int,
) -> relay.Expr:
    if kernel_size != stride:
        raise RuntimeError("RepQ Gemmini patchify conv currently expects stride == kernel size")

    meta = RepQContext.get(module_name)
    prefix = sanitize_name(module_name)
    weight_shape = tuple(int(dim) for dim in RepQContext.get_param(prefix + "_weight").shape)
    weight = relay.var(prefix + "_weight", shape=weight_shape, dtype="float32")
    bias_shape = tuple(int(dim) for dim in RepQContext.get_param(prefix + "_bias").shape)
    bias = relay.var(prefix + "_bias", shape=bias_shape, dtype="float32")

    input_sym_scale = _sym_scale_from_uniform(meta.input_scale, meta.input_zero_point, meta.n_bits)
    weight_sym_scale = _sym_scale_from_uniform(meta.weight_scale, meta.weight_zero_point, meta.n_bits)
    data_nhwc = relay.layout_transform(data_nchw, src_layout="NCHW", dst_layout="NHWC")
    data_int8 = _quantize_sym_int8_expr(data_nhwc, input_sym_scale)

    weight = relay.transpose(weight, axes=[2, 3, 1, 0])
    weight_int8 = _quantize_sym_int8_expr(weight, weight_sym_scale)
    conv = relay.qnn.op.conv2d(
        data_int8,
        weight_int8,
        relay.const(0, "int32"),
        relay.const(0, "int32"),
        relay.const(np.float32(input_sym_scale), "float32"),
        relay.const(np.float32(weight_sym_scale), "float32"),
        kernel_size=kernel_size,
        channels=out_channels,
        strides=stride,
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="int32",
    )
    zero_bias = relay.const(np.zeros((out_channels,), dtype="int32"), "int32")
    conv = relay.nn.bias_add(conv, zero_bias, axis=-1)
    patch_dim = int(np.prod(kernel_size))
    in_channels = int(weight_shape[1])
    output_scale = _requant_output_scale(input_sym_scale, weight_sym_scale, depth=in_channels * patch_dim)
    conv = relay.qnn.op.requantize(
        conv,
        relay.const(np.float32(input_sym_scale * weight_sym_scale), "float32"),
        relay.const(0, "int32"),
        relay.const(np.float32(output_scale), "float32"),
        relay.const(0, "int32"),
        axis=-1,
        out_dtype="int8",
    )
    conv = _dequantize_sym_int8_expr(conv, output_scale)
    return relay.nn.bias_add(conv, bias, axis=-1)


def _exact_uniform_matmul_2d(
    a: relay.Expr,
    b: relay.Expr,
    a_scale: Any,
    a_zero_point: Any,
    b_scale: Any,
    b_zero_point: Any,
    n_bits: int,
) -> relay.Expr:
    a_shape = tuple(int(dim) for dim in _infer_shape(a))
    b_shape = tuple(int(dim) for dim in _infer_shape(b))
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise RuntimeError("RepQ exact uniform matmul expects rank-2 tensors")
    if a_shape[1] != b_shape[0]:
        raise RuntimeError(f"MatMul shape mismatch: {a_shape} vs {b_shape}")

    rows, depth = a_shape
    _, cols = b_shape

    a_shifted, a_shift_center = _uniform_quant_shift(a, a_scale, a_zero_point, n_bits, axis=-1)
    b_shifted, b_shift_center = _uniform_quant_shift(b, b_scale, b_zero_point, n_bits, axis=-1)

    gemm = relay.cast(_gemmini_gemm_int32(a_shifted, b_shifted), "int32")
    row_sum = relay.sum(relay.cast(a_shifted, "int32"), axis=1, keepdims=True)
    col_sum = relay.sum(relay.cast(b_shifted, "int32"), axis=0)

    if np.asarray(a_zero_point).size != 1:
        raise RuntimeError("RepQ exact uniform matmul currently expects scalar A zero-point")
    if np.asarray(a_scale).size != 1:
        raise RuntimeError("RepQ exact uniform matmul currently expects scalar A scale")

    a_shift_center_scalar = relay.reshape(a_shift_center, [])
    b_shift_center_row = _scalar_or_row_expr(b_shift_center, b_zero_point, cols, "float32")
    col_term = relay.reshape(a_shift_center_scalar * col_sum, [1, cols])
    offset_term = _scalar_or_row_expr(
        relay.const(np.int32(depth), "int32") * a_shift_center_scalar * b_shift_center,
        b_zero_point,
        cols,
        "float32",
    )
    exact = gemm + row_sum * b_shift_center_row + col_term + offset_term

    scale = relay.const(np.float32(np.asarray(a_scale).reshape(-1)[0]), "float32")
    b_scale_expr = _const_float32(_as_array(b_scale, "float32").reshape(-1))
    b_scale_row = _scalar_or_row_expr(b_scale_expr, b_scale, cols, "float32")
    return relay.cast(exact, "float32") * scale * b_scale_row


def exact_uniform_matmul(
    a: relay.Expr,
    b: relay.Expr,
    a_scale: Any,
    a_zero_point: Any,
    b_scale: Any,
    b_zero_point: Any,
    n_bits: int,
) -> relay.Expr:
    if RepQContext.use_gemmini_ops:
        return _approx_uniform_matmul(
            a,
            b,
            a_scale=a_scale,
            a_zero_point=a_zero_point,
            b_scale=b_scale,
            b_zero_point=b_zero_point,
            n_bits=n_bits,
        )

    a_shape = tuple(int(dim) for dim in _infer_shape(a))
    b_shape = tuple(int(dim) for dim in _infer_shape(b))

    if len(a_shape) == 2 and len(b_shape) == 2:
        return _exact_uniform_matmul_2d(a, b, a_scale, a_zero_point, b_scale, b_zero_point, n_bits)

    if len(a_shape) == 3 and len(b_shape) == 2:
        batch = a_shape[0]
        a_parts = relay.split(a, batch, axis=0)
        outputs = []
        for idx in range(batch):
            a_i = relay.squeeze(a_parts[idx], axis=[0])
            y_i = _exact_uniform_matmul_2d(
                a_i,
                b,
                a_scale=a_scale,
                a_zero_point=a_zero_point,
                b_scale=b_scale,
                b_zero_point=b_zero_point,
                n_bits=n_bits,
            )
            outputs.append(relay.expand_dims(y_i, axis=0))
        return relay.concatenate(outputs, axis=0)

    if len(a_shape) != 3 or len(b_shape) != 3:
        raise RuntimeError(f"RepQ exact uniform matmul expects rank-2 or rank-3 tensors, got {a_shape} and {b_shape}")
    if a_shape[0] != b_shape[0]:
        raise RuntimeError(f"RepQ exact uniform batched matmul expects matching batch dims, got {a_shape} and {b_shape}")

    batch = a_shape[0]
    if batch > RepQContext.batch_matmul_unroll_limit:
        a_q = uniform_quant_dequant(
            a,
            scale=a_scale,
            zero_point=a_zero_point,
            n_bits=n_bits,
            axis=-1,
        )
        b_q = uniform_quant_dequant(
            b,
            scale=b_scale,
            zero_point=b_zero_point,
            n_bits=n_bits,
            axis=-1,
        )
        return batched_float_matmul(a_q, b_q)

    a_parts = relay.split(a, batch, axis=0)
    b_parts = relay.split(b, batch, axis=0)
    outputs = []
    for idx in range(batch):
        a_i = relay.squeeze(a_parts[idx], axis=[0])
        b_i = relay.squeeze(b_parts[idx], axis=[0])
        y_i = _exact_uniform_matmul_2d(
            a_i,
            b_i,
            a_scale=a_scale,
            a_zero_point=a_zero_point,
            b_scale=b_scale,
            b_zero_point=b_zero_point,
            n_bits=n_bits,
        )
        outputs.append(relay.expand_dims(y_i, axis=0))
    return relay.concatenate(outputs, axis=0)


def batched_float_matmul(a: relay.Expr, b: relay.Expr) -> relay.Expr:
    a_shape = tuple(int(dim) for dim in _infer_shape(a))
    b_shape = tuple(int(dim) for dim in _infer_shape(b))
    if len(a_shape) != 3 or len(b_shape) != 3:
        raise RuntimeError(f"batched_float_matmul expects rank-3 tensors, got {a_shape} and {b_shape}")
    if a_shape[0] != b_shape[0] or a_shape[2] != b_shape[1]:
        raise RuntimeError(f"batched_float_matmul shape mismatch: {a_shape} vs {b_shape}")

    return relay.nn.batch_matmul(a, b, transpose_b=False)


def exact_uniform_linear(data: relay.Expr, module_name: str, add_bias: bool = True) -> relay.Expr:
    meta = RepQContext.get(module_name)
    prefix = sanitize_name(module_name)
    weight_shape = tuple(int(dim) for dim in RepQContext.get_param(prefix + "_weight").shape)
    weight = relay.var(prefix + "_weight", shape=weight_shape, dtype="float32")
    weight_t = relay.transpose(weight, axes=[1, 0])
    out = exact_uniform_matmul(
        data,
        weight_t,
        a_scale=meta.input_scale,
        a_zero_point=meta.input_zero_point,
        b_scale=meta.weight_scale,
        b_zero_point=meta.weight_zero_point,
        n_bits=meta.n_bits,
    )
    if add_bias:
        bias_shape = tuple(int(dim) for dim in RepQContext.get_param(prefix + "_bias").shape)
        bias = relay.var(prefix + "_bias", shape=bias_shape, dtype="float32")
        out = relay.nn.bias_add(out, bias, axis=-1)
    return out


def patchify_conv2d_exact(
    data_nchw: relay.Expr,
    module_name: str,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    out_channels: int,
) -> relay.Expr:
    if RepQContext.use_gemmini_ops:
        return approx_patchify_conv2d(
            data_nchw,
            module_name=module_name,
            kernel_size=kernel_size,
            stride=stride,
            out_channels=out_channels,
        )

    if kernel_size != stride:
        raise RuntimeError("RepQ exact patchify conv currently expects stride == kernel size")

    batch, in_channels, height, width = (int(dim) for dim in _infer_shape(data_nchw))
    k_h, k_w = kernel_size
    out_h = height // k_h
    out_w = width // k_w
    patch_dim = in_channels * k_h * k_w

    data = relay.layout_transform(data_nchw, src_layout="NCHW", dst_layout="NHWC")
    patches = relay.reshape(data, [batch, out_h, k_h, out_w, k_w, in_channels])
    patches = relay.transpose(patches, axes=[0, 1, 3, 2, 4, 5])
    patches = relay.reshape(patches, [batch * out_h * out_w, patch_dim])

    prefix = sanitize_name(module_name)
    weight_shape = tuple(int(dim) for dim in RepQContext.get_param(prefix + "_weight").shape)
    weight = relay.var(prefix + "_weight", shape=weight_shape, dtype="float32")
    # Patch tensors are flattened in [kh, kw, c] order, so transpose OIHW -> OHWI first.
    weight = relay.transpose(weight, axes=[0, 2, 3, 1])
    weight = relay.reshape(weight, [out_channels, patch_dim])
    out = exact_uniform_matmul(
        patches,
        relay.transpose(weight, axes=[1, 0]),
        a_scale=RepQContext.get(module_name).input_scale,
        a_zero_point=RepQContext.get(module_name).input_zero_point,
        b_scale=RepQContext.get(module_name).weight_scale,
        b_zero_point=RepQContext.get(module_name).weight_zero_point,
        n_bits=RepQContext.get(module_name).n_bits,
    )
    bias_shape = tuple(int(dim) for dim in RepQContext.get_param(prefix + "_bias").shape)
    bias = relay.var(prefix + "_bias", shape=bias_shape, dtype="float32")
    out = relay.nn.bias_add(out, bias, axis=-1)
    return relay.reshape(out, [batch, out_h, out_w, out_channels])
