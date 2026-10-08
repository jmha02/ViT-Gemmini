"""
I-ViT DeiT → ONNX QOperator graph builder.

Constructs a fully-INT8 ONNX graph directly from the I-ViT QAT checkpoint,
using the same weights (weight_integer) and scales (scaling_factor) as the
TVM-Gemmini pipeline (TVM_benchmark/convert_model.py + quantized_vit.py).

Graph topology:
  float32 input → QuantizeLinear → int8
  → Gemmini patch-embed matmul → int16
  → [12 × Transformer Block]
      QLayernormInt16 → GemminiMatMulInteger(QKV) → GemminiMatMulInteger(Q@K^T) → Shiftmax
      → GemminiMatMulInteger(Attn@V) → GemminiMatMulInteger(proj) → QLinearAddInt16 (residual)
      → QLayernormInt16 → GemminiMatMulInteger(fc1) → ShiftGELU → GemminiMatMulInteger(fc2)
      → QLinearAddInt16 (residual)
  → QLayernormInt16 (final) → GemminiMatMulInteger(head) → DequantizeLinear → float32

Gemmini Systolic provider handles: QLinearConv, QLinearMatMul
Custom ops (ivit domain):          ivit.QLayernorm, ivit.QLayernormInt16, ivit.Shiftmax,
                                    ivit.ShiftGELU, ivit.GemminiMatMulInteger,
                                    ivit.RequantizeInt32, ivit.RequantizeInt32ToInt16,
                                    ivit.RequantizeInt16, ivit.RequantizeInt8ToInt16,
                                    ivit.QLinearAddInt16

Usage:
    cd third-party/I-ViT-Gemmini
    python3 ONNXRuntime/export/ivit/export_deit_ivit_onnx.py --checkpoint "$IVIT_CHECKPOINT"
    python3 ONNXRuntime/export/ivit/export_deit_ivit_onnx.py --checkpoint "$IVIT_CHECKPOINT" --verify
"""

import argparse
import os
import sys
import math
from pathlib import Path
import numpy as np

import onnx
from onnx import helper, TensorProto, numpy_helper

SCRIPT_DIR = Path(__file__).resolve().parent
ONNXRT_DIR = SCRIPT_DIR.parent.parent
REPO_ROOT = ONNXRT_DIR.parent
TOOLS_DIR = ONNXRT_DIR / "tools"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from ivit_model_io import create_model, load_checkpoint_state_dict

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = REPO_ROOT / "build" / "ort"
OUTPUT_ONNX = OUTPUT_DIR / "ivit_tiny_int8.onnx"
DEFAULT_CKPT = os.environ.get("IVIT_CHECKPOINT")
DEFAULT_IMAGE = REPO_ROOT / "scripts" / "gemmini" / "test_cat.jpg"
DEFAULT_HOST_CUSTOM_OP_LIB = REPO_ROOT / "build" / "ort" / "ort_ivit_ops" / "libivit_ops_host.so"

MODEL_SPECS = {
    "deit_tiny_patch16_224": {"embed_dim": 192, "depth": 12, "num_heads": 3, "tag": "tiny"},
    "deit_small_patch16_224": {"embed_dim": 384, "depth": 12, "num_heads": 6, "tag": "small"},
}
SUPPORTED_MODEL_NAMES = tuple(MODEL_SPECS)

# Defaults are kept as module globals because the graph builder uses these
# constants throughout the hand-written ONNX topology.
EMBED_DIM   = MODEL_SPECS["deit_tiny_patch16_224"]["embed_dim"]
DEPTH       = MODEL_SPECS["deit_tiny_patch16_224"]["depth"]
NUM_HEADS   = MODEL_SPECS["deit_tiny_patch16_224"]["num_heads"]
HEAD_DIM    = EMBED_DIM // NUM_HEADS  # 64
MLP_RATIO   = 4
HIDDEN_DIM  = EMBED_DIM * MLP_RATIO  # 768
PATCH_SIZE  = 16
IMG_SIZE    = 224
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2   # 196
SEQ_LEN     = NUM_PATCHES + 1                 # 197
IN_CHANNELS = 3
NUM_CLASSES = 1000

INT8_ZERO  = np.array(0, dtype=np.int8)
FLOAT_ZERO = np.array(0.0, dtype=np.float32)


def configure_model(model_name: str) -> dict:
    try:
        spec = MODEL_SPECS[model_name]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_MODEL_NAMES)
        raise ValueError(f"Unsupported model_name={model_name!r}; supported: {supported}") from exc

    global EMBED_DIM, DEPTH, NUM_HEADS, HEAD_DIM, HIDDEN_DIM
    EMBED_DIM = spec["embed_dim"]
    DEPTH = spec["depth"]
    NUM_HEADS = spec["num_heads"]
    HEAD_DIM = EMBED_DIM // NUM_HEADS
    HIDDEN_DIM = EMBED_DIM * MLP_RATIO
    return spec

# ── Graph builder ─────────────────────────────────────────────────────────────

class OnnxBuilder:
    """Accumulates ONNX nodes, initializers and intermediate value_infos."""

    def __init__(self):
        self.nodes       = []
        self.initializers = []
        self.value_infos = []
        self._counter    = {}

    # ── Naming helpers ────────────────────────────────────────────────────────

    def uid(self, prefix):
        n = self._counter.get(prefix, 0)
        self._counter[prefix] = n + 1
        return f"{prefix}_{n}" if n else prefix

    # ── Initializer registration ──────────────────────────────────────────────

    def init_tensor(self, name, data: np.ndarray):
        self.initializers.append(numpy_helper.from_array(data, name=name))
        return name

    def scalar_f32(self, name, value: float):
        return self.init_tensor(name, np.array(value, dtype=np.float32))

    def scalar_i8(self, name, value: int):
        return self.init_tensor(name, np.array(value, dtype=np.int8))

    def scalar_i16(self, name, value: int):
        return self.init_tensor(name, np.array(value, dtype=np.int16))

    def vec_f32(self, name, arr):
        return self.init_tensor(name, np.array(arr, dtype=np.float32))

    def scale_tensor(self, name, value):
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0 or arr.size == 1:
            return self.scalar_f32(name, float(arr.reshape(())))
        return self.init_tensor(name, arr.reshape(-1).astype(np.float32))

    # ── Low-level ONNX node helpers ───────────────────────────────────────────

    def add(self, op, inputs, outputs, **attrs):
        self.nodes.append(helper.make_node(op, inputs, outputs, **attrs))
        return outputs[0]

    def add_custom(self, op, domain, inputs, outputs, **attrs):
        node = helper.make_node(op, inputs, outputs, domain=domain, **attrs)
        self.nodes.append(node)
        return outputs[0]

    # ── Shape annotations (optional but useful for debuggers) ─────────────────

    def note_int8(self, name, shape):
        self.value_infos.append(helper.make_tensor_value_info(name, TensorProto.INT8, shape))

    def note_int32(self, name, shape):
        self.value_infos.append(helper.make_tensor_value_info(name, TensorProto.INT32, shape))

    def note_int16(self, name, shape):
        self.value_infos.append(helper.make_tensor_value_info(name, TensorProto.INT16, shape))

    def note_int64(self, name, shape):
        self.value_infos.append(helper.make_tensor_value_info(name, TensorProto.INT64, shape))

    def note_float(self, name, shape):
        self.value_infos.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))

    # ── Standard quantization ops ─────────────────────────────────────────────

    def quantize(self, x, scale_val: float, name_prefix: str) -> str:
        """float32 → int8  (QuantizeLinear, zero_point=0)."""
        scale_n = self.scalar_f32(f"{name_prefix}_scale", scale_val)
        zp_n    = self.scalar_i8(f"{name_prefix}_zp", 0)
        out     = f"{name_prefix}_quant"
        self.add("QuantizeLinear", [x, scale_n, zp_n], [out])
        return out

    def dequantize(self, x, scale_val: float, name_prefix: str) -> str:
        """int8 → float32  (DequantizeLinear, zero_point=0)."""
        scale_n = self.scalar_f32(f"{name_prefix}_dq_scale", scale_val)
        zp_n    = self.scalar_i8(f"{name_prefix}_dq_zp", 0)
        out     = f"{name_prefix}_dequant"
        self.add("DequantizeLinear", [x, scale_n, zp_n], [out])
        return out

    def dequantize_general(self, x, scale_val, name_prefix: str) -> str:
        """Signed integer tensor → float32 via Cast + Mul."""
        scale_n = self.scale_tensor(f"{name_prefix}_dq_scale", scale_val)
        x_f = f"{name_prefix}_cast_f32"
        self.add("Cast", [x], [x_f], to=TensorProto.FLOAT)
        out = f"{name_prefix}_dequant"
        self.add("Mul", [x_f, scale_n], [out])
        return out

    def quantize_to_int32(self, x_float, scale_val, name_prefix: str) -> str:
        """float32 → int32 with ONNX Round semantics, matching TVM float fallback."""
        scale_n = self.scale_tensor(f"{name_prefix}_q_scale", scale_val)
        scaled = f"{name_prefix}_scaled"
        self.add("Div", [x_float, scale_n], [scaled])
        rounded = f"{name_prefix}_rounded"
        self.add("Round", [scaled], [rounded])
        out = f"{name_prefix}_out"
        self.add("Cast", [rounded], [out], to=TensorProto.INT32)
        return out

    def quantize_to_int16(self, x_float, scale_val, name_prefix: str) -> str:
        """float32 → int16 with symmetric clipping."""
        scale_n = self.scale_tensor(f"{name_prefix}_q_scale", scale_val)
        scaled = f"{name_prefix}_scaled"
        self.add("Div", [x_float, scale_n], [scaled])
        rounded = f"{name_prefix}_rounded"
        self.add("Round", [scaled], [rounded])
        clip_min = self.scalar_f32(f"{name_prefix}_clip_min", -32768.0)
        clip_max = self.scalar_f32(f"{name_prefix}_clip_max", 32767.0)
        clipped = f"{name_prefix}_clipped"
        self.add("Clip", [rounded, clip_min, clip_max], [clipped])
        out = f"{name_prefix}_out"
        self.add("Cast", [clipped], [out], to=TensorProto.INT16)
        return out

    def quantize_to_int8(self, x_float, scale_val, name_prefix: str) -> str:
        """float32 → int8 with symmetric clipping."""
        scale_n = self.scale_tensor(f"{name_prefix}_q_scale", scale_val)
        scaled = f"{name_prefix}_scaled"
        self.add("Div", [x_float, scale_n], [scaled])
        rounded = f"{name_prefix}_rounded"
        self.add("Round", [scaled], [rounded])
        clip_min = self.scalar_f32(f"{name_prefix}_clip_min", -128.0)
        clip_max = self.scalar_f32(f"{name_prefix}_clip_max", 127.0)
        clipped = f"{name_prefix}_clipped"
        self.add("Clip", [rounded, clip_min, clip_max], [clipped])
        out = f"{name_prefix}_out"
        self.add("Cast", [clipped], [out], to=TensorProto.INT8)
        return out

    def add_float_int32(self, x1, x1_scale, x2, x2_scale, out_scale, name_prefix: str) -> str:
        """
        Match TVM/PyTorch residual semantics:
          dequantize(x1) + dequantize(x2) -> quantize(..., out_dtype=int32)
        """
        x1_f = self.dequantize_general(x1, x1_scale, f"{name_prefix}_x1")
        x2_f = self.dequantize_general(x2, x2_scale, f"{name_prefix}_x2")
        summed = f"{name_prefix}_sum_f32"
        self.add("Add", [x1_f, x2_f], [summed])
        return self.quantize_to_int32(summed, out_scale, f"{name_prefix}_q")

    def add_float_int16(self, x1, x1_scale, x2, x2_scale, out_scale, name_prefix: str) -> str:
        """Dequantize-add-requantize to int16, used by DeiT/Swin 16-bit qact stages."""
        x1_f = self.dequantize_general(x1, x1_scale, f"{name_prefix}_x1")
        x2_f = self.dequantize_general(x2, x2_scale, f"{name_prefix}_x2")
        summed = f"{name_prefix}_sum_f32"
        self.add("Add", [x1_f, x2_f], [summed])
        return self.quantize_to_int16(summed, out_scale, f"{name_prefix}_q")

    def requantize_float(self, x_int8, in_scale: float, out_scale: float,
                          name_prefix: str) -> str:
        """int8 → float32 → int8 requantize via DQ+Q (float intermediate)."""
        mid = self.dequantize(x_int8, in_scale, name_prefix)
        return self.quantize(mid, out_scale, name_prefix + "_req")

    def requantize_int32(self, x_int32, in_scale, out_scale: float,
                          name_prefix: str) -> str:
        in_scale_arr = np.asarray(in_scale, dtype=np.float32)
        req_scale_arr = in_scale_arr / np.float32(out_scale)
        if in_scale_arr.ndim == 0:
            scale_n = self.scalar_f32(f"{name_prefix}_req_scale", float(req_scale_arr))
        else:
            scale_n = self.init_tensor(
                f"{name_prefix}_req_scale", req_scale_arr.reshape(-1).astype(np.float32)
            )
        out = f"{name_prefix}_out"
        self.add_custom("RequantizeInt32", "ivit", [x_int32, scale_n], [out])
        return out

    def requantize_int32_to_int16(self, x_int32, in_scale, out_scale: float,
                                   name_prefix: str) -> str:
        in_scale_arr = np.asarray(in_scale, dtype=np.float32)
        req_scale_arr = in_scale_arr / np.float32(out_scale)
        if in_scale_arr.ndim == 0:
            scale_n = self.scalar_f32(f"{name_prefix}_req_scale", float(req_scale_arr))
        else:
            scale_n = self.init_tensor(
                f"{name_prefix}_req_scale", req_scale_arr.reshape(-1).astype(np.float32)
            )
        out = f"{name_prefix}_out"
        self.add_custom("RequantizeInt32ToInt16", "ivit", [x_int32, scale_n], [out])
        return out

    def requantize_int8_to_int16(self, x_int8, in_scale: float, out_scale: float,
                                  name_prefix: str) -> str:
        scale_n = self.scalar_f32(f"{name_prefix}_req_scale", float(np.float32(in_scale) / np.float32(out_scale)))
        out = f"{name_prefix}_out"
        self.add_custom("RequantizeInt8ToInt16", "ivit", [x_int8, scale_n], [out])
        return out

    def requantize_int16(self, x_int16, in_scale: float, out_scale: float,
                          name_prefix: str) -> str:
        scale_n = self.scalar_f32(f"{name_prefix}_req_scale", float(np.float32(in_scale) / np.float32(out_scale)))
        out = f"{name_prefix}_out"
        self.add_custom("RequantizeInt16", "ivit", [x_int16, scale_n], [out])
        return out

    def requantize_int16_to_int8(self, x_int16, in_scale: float, out_scale: float,
                                  name_prefix: str) -> str:
        scale_n = self.scalar_f32(f"{name_prefix}_req_scale", float(np.float32(in_scale) / np.float32(out_scale)))
        out = f"{name_prefix}_out"
        self.add_custom("RequantizeInt16ToInt8", "ivit", [x_int16, scale_n], [out])
        return out

    def requantize_from_int64(self, x_int64, in_scale, out_scale: float, out_dtype: str, name_prefix: str) -> str:
        """
        int64 → int{8,16} requantize using the same fixed-point approximation that
        I-ViT QuantAct uses during checkpoint inference.
        """
        req_scale_arr = np.asarray(in_scale, dtype=np.float32) / np.float32(out_scale)
        if req_scale_arr.ndim == 0 or req_scale_arr.size == 1:
            scale_n = self.scalar_f32(f"{name_prefix}_req_scale", float(req_scale_arr.reshape(())))
        else:
            scale_n = self.init_tensor(
                f"{name_prefix}_req_scale", req_scale_arr.reshape(-1).astype(np.float32)
            )
        out = f"{name_prefix}_out"
        if out_dtype == "int16":
            self.add_custom("RequantizeInt64ToInt16", "ivit", [x_int64, scale_n], [out])
            return out
        if out_dtype == "int8":
            self.add_custom("RequantizeInt64", "ivit", [x_int64, scale_n], [out])
            return out
        raise ValueError(f"Unsupported int64 requantize output dtype: {out_dtype}")

    def patch_embed_matmul(
        self,
        x,
        x_scale: float,
        w_data: np.ndarray,
        w_scale,
        bias_data: np.ndarray,
        y_scale: float,
        out_dtype: str,
        patch_size: int,
        image_size: int,
        name_prefix: str,
    ) -> str:
        """
        Exact non-overlapping patch embedding via reshape + GemminiMatMulInteger.

        For kernel=stride=patch_size, convolution is equivalent to flattening each
        patch in `[C, patch, patch]` order and applying a per-output-channel dense.
        This preserves the checkpoint's per-channel conv scales exactly, unlike
        collapsing them to a single QLinearConv weight scale.
        """
        out_channels, in_channels, kh, kw = w_data.shape
        if kh != patch_size or kw != patch_size:
            raise ValueError("patch_embed_matmul expects kernel spatial shape == patch_size")
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        patch_grid = image_size // patch_size
        patch_dim = in_channels * patch_size * patch_size
        num_patches = patch_grid * patch_grid

        split_shape = self.init_tensor(
            f"{name_prefix}_split_shape",
            np.array([1, in_channels, patch_grid, patch_size, patch_grid, patch_size], dtype=np.int64),
        )
        split = f"{name_prefix}_split"
        self.add("Reshape", [x, split_shape], [split])

        reorder = f"{name_prefix}_reorder"
        self.add("Transpose", [split], [reorder], perm=[0, 2, 4, 1, 3, 5])

        token_shape = self.init_tensor(
            f"{name_prefix}_token_shape",
            np.array([1, num_patches, patch_dim], dtype=np.int64),
        )
        tokens = f"{name_prefix}_tokens"
        self.add("Reshape", [reorder, token_shape], [tokens])

        flat_w = w_data.reshape(out_channels, patch_dim).T.astype(np.int8)
        mm = self.gemmini_matmul_integer(tokens, flat_w, f"{name_prefix}_mm")
        bias_name = self.init_tensor(f"{name_prefix}_bias", bias_data.astype(np.int32))
        biased = f"{name_prefix}_biased"
        self.add("Add", [mm, bias_name], [biased])

        natural_scale = (np.float32(x_scale) * np.asarray(w_scale, dtype=np.float32).reshape(-1)).astype(np.float32)
        if out_dtype == "int16":
            return self.requantize_int32_to_int16(biased, natural_scale, y_scale, f"{name_prefix}_req")
        if out_dtype == "int8":
            return self.requantize_int32(biased, natural_scale, y_scale, f"{name_prefix}_req")
        raise ValueError(f"Unsupported patch embedding output dtype: {out_dtype}")

    def gemmini_matmul_integer(
        self,
        a,
        b_data: np.ndarray,
        name_prefix: str,
        *,
        force_cpu: bool = False,
    ) -> str:
        """ivit.GemminiMatMulInteger: A(int8) @ B(int8) → int32."""
        a_zp_n = self.scalar_i8(f"{name_prefix}_a_zp", 0)
        b_name = self.init_tensor(f"{name_prefix}_weight", b_data.astype(np.int8))
        return self.gemmini_matmul_integer_inputs(
            a,
            b_name,
            name_prefix,
            a_zp_n=a_zp_n,
            force_cpu=force_cpu,
        )

    def gemmini_matmul_integer_inputs(
        self,
        a,
        b,
        name_prefix: str,
        a_zp_n: str | None = None,
        *,
        force_cpu: bool = False,
    ) -> str:
        """ivit.GemminiMatMulInteger: A(int8) @ B(int8) → int32."""
        if a_zp_n is None:
            a_zp_n = self.scalar_i8(f"{name_prefix}_a_zp", 0)
        b_zp_n = self.scalar_i8(f"{name_prefix}_b_zp", 0)
        out = f"{name_prefix}_out"
        self.add_custom(
            "GemminiMatMulInteger",
            "ivit",
            [a, b, a_zp_n, b_zp_n],
            [out],
            force_cpu=int(force_cpu),
        )
        return out

    # ── QLinear ops ───────────────────────────────────────────────────────────

    def qlinear_matmul(self, a, a_scale: float,
                        b_data: np.ndarray, b_scale,  # scalar or 1-D array
                        y_scale: float,
                        name_prefix: str,
                        bias_data: np.ndarray = None) -> str:
        """
        QLinearMatMul: A(int8) @ B(int8) → Y(int8).

        b_scale can be a scalar (per-tensor) or 1-D array (per-column).
        Bias is NOT part of QLinearMatMul; we add it separately in int32 space
        via MatMulInteger+bias then requantize if needed. However, ONNX
        QLinearMatMul has no bias input, so we represent fc with bias as
        separate MatMulInteger + Add + Requantize nodes for correctness.

        Here, QLinearMatMul is used for ops where bias is absent OR small
        enough that we skip it (attention matmuls).
        For proj/QKV/fc/head with bias: see qlinear_matmul_bias().
        """
        a_s_n  = self.scalar_f32(f"{name_prefix}_a_scale", a_scale)
        a_zp_n = self.scalar_i8(f"{name_prefix}_a_zp", 0)

        b_name = self.init_tensor(f"{name_prefix}_weight", b_data.astype(np.int8))
        if np.isscalar(b_scale) or (isinstance(b_scale, np.ndarray) and b_scale.ndim == 0):
            b_s_n = self.scalar_f32(f"{name_prefix}_b_scale", float(b_scale))
        else:
            b_s_n = self.vec_f32(f"{name_prefix}_b_scale", b_scale)
        b_zp_n = self.scalar_i8(f"{name_prefix}_b_zp", 0)

        y_s_n  = self.scalar_f32(f"{name_prefix}_y_scale", y_scale)
        y_zp_n = self.scalar_i8(f"{name_prefix}_y_zp", 0)

        out = f"{name_prefix}_out"
        self.add("QLinearMatMul",
                 [a, a_s_n, a_zp_n, b_name, b_s_n, b_zp_n, y_s_n, y_zp_n],
                 [out])
        return out

    def qlinear_matmul_bias(self, a, a_scale: float,
                             b_data: np.ndarray, b_scale,
                             bias_data: np.ndarray,
                             y_scale: float,
                             out_dtype: str,
                             name_prefix: str) -> str:
        """
        Quantized MatMul with bias via GemminiMatMulInteger + int32 bias add + requantize.

        This correctly models:
          out_int32 = A_int8 @ B_int8 + bias_int32
          out_int8  = saturate(round(out_int32 * (a_scale * b_scale / y_scale)))
        """
        mm_out = self.gemmini_matmul_integer(a, b_data, f"{name_prefix}_mm")

        # Add int32 bias
        bias_name = self.init_tensor(f"{name_prefix}_bias", bias_data.astype(np.int32))
        biased    = f"{name_prefix}_biased_int32"
        self.add("Add", [mm_out, bias_name], [biased])

        b_scale_arr = np.asarray(b_scale, dtype=np.float32)
        natural_scale = (np.float32(a_scale) * b_scale_arr).astype(np.float32)
        if out_dtype == "int16":
            return self.requantize_int32_to_int16(biased, natural_scale, y_scale, f"{name_prefix}_req")
        if out_dtype == "int8":
            return self.requantize_int32(biased, natural_scale, y_scale, f"{name_prefix}_req")
        raise ValueError(f"Unsupported dense output dtype: {out_dtype}")

    def qlinear_conv(self, x, x_scale: float,
                     w_data: np.ndarray, w_scale,
                     bias_data: np.ndarray,
                     y_scale: float,
                     name_prefix: str,
                     kernel_shape, strides, pads=(0,0,0,0)) -> str:
        """QLinearConv: x(int8) * w(int8) + bias(int32) → y(int8)."""
        x_s_n  = self.scalar_f32(f"{name_prefix}_x_scale", x_scale)
        x_zp_n = self.scalar_i8(f"{name_prefix}_x_zp", 0)

        w_name = self.init_tensor(f"{name_prefix}_w", w_data.astype(np.int8))
        if np.isscalar(w_scale) or (isinstance(w_scale, np.ndarray) and w_scale.ndim == 0):
            w_s_n = self.scalar_f32(f"{name_prefix}_w_scale", float(w_scale))
        else:
            w_s_n = self.vec_f32(f"{name_prefix}_w_scale", w_scale)
        w_zp_n = self.scalar_i8(f"{name_prefix}_w_zp", 0)

        y_s_n  = self.scalar_f32(f"{name_prefix}_y_scale", y_scale)
        y_zp_n = self.scalar_i8(f"{name_prefix}_y_zp", 0)

        bias_name = self.init_tensor(f"{name_prefix}_bias",
                                     bias_data.astype(np.int32))

        out = f"{name_prefix}_out"
        self.add("QLinearConv",
                 [x, x_s_n, x_zp_n, w_name, w_s_n, w_zp_n, y_s_n, y_zp_n,
                  bias_name],
                 [out],
                 kernel_shape=list(kernel_shape),
                 strides=list(strides),
                 pads=list(pads))
        return out

    # ── Custom ops (ivit domain) ───────────────────────────────────────────────

    def shiftmax(self, x, input_scale: float, name_prefix: str, input_dtype: str = "int8") -> str:
        """
        ivit.Shiftmax{Int32}: int{8,32} → int8
        x0 = int(-1 / input_scale - 1)  (TVM ShiftExp parameter for n=16)
        """
        x0 = int(-1.0 / input_scale - 1.0)
        out = f"{name_prefix}_out"
        op_name = "ShiftmaxInt32" if input_dtype == "int32" else "Shiftmax"
        self.add_custom(op_name, "ivit", [x], [out], x0=x0)
        return out

    def shift_gelu(self, x, scaling_factor: float, name_prefix: str) -> str:
        """
        ivit.ShiftGELU: int8 → int32
        scaling_factor = mlp.qact_gelu.act_scaling_factor (input scale)
        x0 = int(-1 / (scaling_factor * 1.702) - 1)
        Output int32 has natural scale = scaling_factor / 128.0
        """
        x0 = int(-1.0 / (scaling_factor * 1.702) - 1.0)
        out = f"{name_prefix}_out"
        self.add_custom("ShiftGELU", "ivit", [x], [out], x0=x0)
        return out

    def q_layernorm(self, x, bias_int_arr: np.ndarray, name_prefix: str, input_dtype: str = "int8") -> str:
        """
        ivit.QLayernorm* custom op wrapper.

        int32-bias variants preserve the original DeiT export path:
          int{8,16,32} + int32 bias -> int32

        int64-bias variants are used for Swin checkpoints whose LayerNorm
        bias_integer values exceed int32 range:
          int{8,16,32} + int64 bias -> int64
        """
        if np.issubdtype(np.asarray(bias_int_arr).dtype, np.int64):
            bias_name = self.init_tensor(f"{name_prefix}_bias", np.asarray(bias_int_arr, dtype=np.int64))
            if input_dtype == "int16":
                op_name = "QLayernormInt16I64"
            elif input_dtype == "int32":
                op_name = "QLayernormInt32I64"
            else:
                op_name = "QLayernormI64"
        else:
            bias_name = self.init_tensor(f"{name_prefix}_bias", np.asarray(bias_int_arr, dtype=np.int32))
            if input_dtype == "int16":
                op_name = "QLayernormInt16"
            elif input_dtype == "int32":
                op_name = "QLayernormInt32"
            else:
                op_name = "QLayernorm"
        out = f"{name_prefix}_out"
        self.add_custom(op_name, "ivit", [x, bias_name], [out])
        return out

    # ── Quantized add ─────────────────────────────────────────────────────────

    def qlinear_add(self, x1, x1_scale: float, x2, x2_scale: float,
                    out_scale: float, name_prefix: str) -> str:
        x1_s = self.scalar_f32(f"{name_prefix}_x1_scale", x1_scale)
        x1_z = self.scalar_i8(f"{name_prefix}_x1_zp", 0)
        x2_s = self.scalar_f32(f"{name_prefix}_x2_scale", x2_scale)
        x2_z = self.scalar_i8(f"{name_prefix}_x2_zp", 0)
        y_s = self.scalar_f32(f"{name_prefix}_y_scale", out_scale)
        y_z = self.scalar_i8(f"{name_prefix}_y_zp", 0)
        out = f"{name_prefix}_out"
        self.nodes.append(
            helper.make_node(
                "QLinearAdd",
                [x1, x1_s, x1_z, x2, x2_s, x2_z, y_s, y_z],
                [out],
                domain="com.microsoft",
            )
        )
        return out

    def qlinear_add_int16(self, x1, x1_scale: float, x2, x2_scale: float,
                          out_scale: float, name_prefix: str) -> str:
        x1_s = self.scalar_f32(f"{name_prefix}_x1_scale", x1_scale)
        x1_z = self.scalar_i16(f"{name_prefix}_x1_zp", 0)
        x2_s = self.scalar_f32(f"{name_prefix}_x2_scale", x2_scale)
        x2_z = self.scalar_i16(f"{name_prefix}_x2_zp", 0)
        y_s = self.scalar_f32(f"{name_prefix}_y_scale", out_scale)
        y_z = self.scalar_i16(f"{name_prefix}_y_zp", 0)
        out = f"{name_prefix}_out"
        self.add_custom(
            "QLinearAddInt16",
            "ivit",
            [x1, x1_s, x1_z, x2, x2_s, x2_z, y_s, y_z],
            [out],
        )
        return out

    def residual_add(self, x1, x1_scale: float,
                     x2, x2_scale: float,
                     out_scale: float,
                     name_prefix: str) -> str:
        """
        Float residual path returning int32, matching TVM add_float semantics.
        """
        return self.add_float_int32(x1, x1_scale, x2, x2_scale, out_scale, name_prefix)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

DEAD_CHANNEL_THRESHOLD = 0.01


def get_scale(sd, key: str):
    """
    Get quantization scale from checkpoint.

    - `act_scaling_factor`: scalar (first element if tensor shape is [1]).
    - other scaling factors (`conv/fc/norm`): keep full vector if non-scalar.
    """
    t = sd[key]
    if hasattr(t, "cpu"):
        t = t.cpu()
    v = t.numpy().astype(np.float32).reshape(-1)

    if "act_scaling_factor" in key:
        return float(v[0])
    if v.size == 1:
        return float(v[0])
    return v


def get_weight(sd, key: str) -> np.ndarray:
    return sd[key].cpu().numpy()


def to_int32(arr: np.ndarray) -> np.ndarray:
    """Cast float32 array to int32 via int64 to avoid RuntimeWarning on overflow.
    Matches TVM behavior: truncate/wrap for out-of-range values.
    """
    return arr.astype(np.int64).astype(np.int32)


def to_int64(arr: np.ndarray) -> np.ndarray:
    """Cast float32 array to int64 using nearest integer, matching TVM LayerNorm bias handling."""
    return np.rint(arr).astype(np.int64)


def fix_norm_bias_and_scale(sd, block_idx: int, norm_name: str):
    """
    Apply dead-channel fix used by TVM conversion for LayerNorm.

    Dead channel criterion: |weight| < DEAD_CHANNEL_THRESHOLD.
    For dead channels:
    - bias_integer -> 0
    - norm_scaling_factor -> mean(abs(valid_channels)) with original sign
    """
    prefix = f"blocks.{block_idx}.{norm_name}"
    weight = get_weight(sd, f"{prefix}.weight").reshape(-1).astype(np.float32)
    bias_int = to_int32(get_weight(sd, f"{prefix}.bias_integer")).reshape(-1)
    norm_sf = np.asarray(get_scale(sd, f"{prefix}.norm_scaling_factor"), dtype=np.float32).reshape(-1)

    dead_mask = np.abs(weight) < DEAD_CHANNEL_THRESHOLD
    if not np.any(dead_mask):
        return bias_int, norm_sf

    valid_mask = ~dead_mask
    if np.any(valid_mask):
        mean_sf = float(np.mean(np.abs(norm_sf[valid_mask])))
    else:
        mean_sf = float(np.mean(np.abs(norm_sf)))
    if mean_sf == 0.0:
        mean_sf = 1.0

    fixed_bias = bias_int.copy()
    fixed_bias[dead_mask] = 0

    fixed_sf = norm_sf.copy()
    signs = np.sign(fixed_sf[dead_mask])
    signs[signs == 0] = 1.0
    fixed_sf[dead_mask] = mean_sf * signs

    return fixed_bias.astype(np.int32), fixed_sf.astype(np.float32)


def fix_final_norm_bias_and_scale(sd):
    """Dead-channel fix for final LayerNorm (`norm.*`)."""
    weight = get_weight(sd, "norm.weight").reshape(-1).astype(np.float32)
    bias_int = to_int32(get_weight(sd, "norm.bias_integer")).reshape(-1)
    norm_sf = np.asarray(get_scale(sd, "norm.norm_scaling_factor"), dtype=np.float32).reshape(-1)

    dead_mask = np.abs(weight) < DEAD_CHANNEL_THRESHOLD
    if not np.any(dead_mask):
        return bias_int, norm_sf

    valid_mask = ~dead_mask
    if np.any(valid_mask):
        mean_sf = float(np.mean(np.abs(norm_sf[valid_mask])))
    else:
        mean_sf = float(np.mean(np.abs(norm_sf)))
    if mean_sf == 0.0:
        mean_sf = 1.0

    fixed_bias = bias_int.copy()
    fixed_bias[dead_mask] = 0

    fixed_sf = norm_sf.copy()
    signs = np.sign(fixed_sf[dead_mask])
    signs[signs == 0] = 1.0
    fixed_sf[dead_mask] = mean_sf * signs

    return fixed_bias.astype(np.int32), fixed_sf.astype(np.float32)


# ── Graph construction ────────────────────────────────────────────────────────

def build_ivit_graph(sd, model_name="deit_tiny_patch16_224") -> onnx.ModelProto:
    """
    Build full I-ViT DeiT ONNX graph from I-ViT QAT state_dict.
    Matches the quantization config extracted by TVM_benchmark/convert_model.py.
    """
    configure_model(model_name)
    g = OnnxBuilder()

    # ── Input ──────────────────────────────────────────────────────────────────
    # float32 image (1, 3, 224, 224) normalized with ImageNet mean/std
    input_scale = get_scale(sd, "qact_input.act_scaling_factor")

    x = g.quantize("image", input_scale, "input")
    # x: int8 (1, 3, 224, 224)

    # ── Patch Embedding ────────────────────────────────────────────────────────
    conv_w = get_weight(sd, "patch_embed.proj.weight_integer").astype(np.int8)  # (192, 3, 16, 16)
    conv_b = get_weight(sd, "patch_embed.proj.bias_integer").reshape(-1).astype(np.int32)  # (192,)
    conv_w_scale = np.asarray(get_scale(sd, "patch_embed.proj.conv_scaling_factor"),
                              dtype=np.float32).reshape(-1)
    embed_out_scale = get_scale(sd, "patch_embed.qact.act_scaling_factor")

    patch_flat = g.patch_embed_matmul(
        x,
        input_scale,
        conv_w,
        conv_w_scale,
        conv_b,
        embed_out_scale,
        "int16",
        PATCH_SIZE,
        IMG_SIZE,
        "patch_embed",
    )
    # patch_flat: int16 (1, 196, 192)

    # ── CLS token ─────────────────────────────────────────────────────────────
    # Quantize cls_token (float32 from checkpoint) using the patch-embed qact scale.
    # In I-ViT, cls token is concatenated before qact1 and therefore shares the
    # incoming activation scale from patch embedding.
    qact1_scale = get_scale(sd, "qact1.act_scaling_factor")
    cls_float = get_weight(sd, "cls_token")  # (1, 1, 192) float32
    cls_int16 = np.clip(np.round(cls_float / embed_out_scale), -32768, 32767).astype(np.int16)
    cls_name  = g.init_tensor("cls_token", cls_int16)

    # Concat cls + patches: both are int16 at patch-embed qact scale.
    seq_out = "seq_tokens"
    g.add("Concat", [cls_name, patch_flat], [seq_out], axis=1)

    # ── Position Embedding ─────────────────────────────────────────────────────
    pos_scale = get_scale(sd, "qact_pos.act_scaling_factor")
    pos_float = get_weight(sd, "pos_embed")  # (1, 197, 192) float32
    pos_int16 = np.clip(np.round(pos_float / pos_scale), -32768, 32767).astype(np.int16)
    pos_name  = g.init_tensor("pos_embed", pos_int16)

    # qact1 fuses seq(pos-less) + pos into an int16 output at qact1_scale.
    seq_final = g.add_float_int16(
        seq_out, embed_out_scale,
        pos_name, pos_scale,
        qact1_scale,
        "pos_add"
    )
    # seq_final: int16 (1, 197, 192), scale = qact1_scale

    x_blk = seq_final

    # ── Transformer Blocks ─────────────────────────────────────────────────────
    for blk in range(DEPTH):
        p = f"blk{blk}"
        b = f"blocks.{blk}"

        # Input scale to this block
        if blk == 0:
            blk_in_scale = qact1_scale
            blk_in_dtype = "int16"
        else:
            blk_in_scale = get_scale(sd, f"blocks.{blk-1}.qact4.act_scaling_factor")
            blk_in_dtype = "int32"

        # ── LayerNorm 1 ──────────────────────────────────────────────────────
        norm1_bias_arr, norm1_out_sc = fix_norm_bias_and_scale(sd, blk, "norm1")

        ln1_out = g.q_layernorm(x_blk, norm1_bias_arr, f"{p}_ln1", input_dtype=blk_in_dtype)
        # int32 at norm1_out_sc → int8 at qact1 scale
        qact1_blk_sc = get_scale(sd, f"{b}.qact1.act_scaling_factor")
        ln1_req = g.requantize_int32(ln1_out, norm1_out_sc, qact1_blk_sc, f"{p}_ln1_req")

        # ── QKV Projection ───────────────────────────────────────────────────
        qkv_w    = get_weight(sd, f"{b}.attn.qkv.weight_integer")  # (3*192, 192)
        qkv_b    = get_weight(sd, f"{b}.attn.qkv.bias_integer")    # (3*192,)
        qkv_ws   = get_scale(sd, f"{b}.attn.qkv.fc_scaling_factor")
        qkv_out_sc = get_scale(sd, f"{b}.attn.qact1.act_scaling_factor")

        # Weight is (out, in) in PyTorch → need (in, out) for x@W convention
        qkv_w_t = qkv_w.T  # (192, 576)

        qkv_out = g.qlinear_matmul_bias(
            ln1_req, qact1_blk_sc,
            qkv_w_t, qkv_ws,
            qkv_b,
            qkv_out_sc,
            "int8",
            f"{p}_qkv"
        )
        # qkv_out: int8 (1, 197, 576), scale = qkv_out_sc (= attn.qact1.act_scaling_factor)

        # Reshape: (1, 197, 576) → (1, 197, 3, 3, 64) → split Q,K,V
        # Actually reshape to (1, 197, 3, 64) per head group then permute
        # ONNX: reshape to (1, N, 3, H, d) then Transpose to (3, 1, H, N, d)
        qkv_shape5 = g.init_tensor(f"{p}_qkv_shape5",
            np.array([1, SEQ_LEN, 3, NUM_HEADS, HEAD_DIM], dtype=np.int64))
        qkv_5d = f"{p}_qkv_5d"
        g.add("Reshape", [qkv_out, qkv_shape5], [qkv_5d])
        # Transpose (1,N,3,H,d) → (3,1,H,N,d) = [2,0,3,1,4]
        qkv_t = f"{p}_qkv_transposed"
        g.add("Transpose", [qkv_5d], [qkv_t], perm=[2, 0, 3, 1, 4])
        # Gather Q, K, V
        idx0 = g.init_tensor(f"{p}_idx0", np.array(0, dtype=np.int64))
        idx1 = g.init_tensor(f"{p}_idx1", np.array(1, dtype=np.int64))
        idx2 = g.init_tensor(f"{p}_idx2", np.array(2, dtype=np.int64))
        q_mat = f"{p}_q"  # (1, H, N, d)
        k_mat = f"{p}_k"
        v_mat = f"{p}_v"
        g.add("Gather", [qkv_t, idx0], [q_mat], axis=0)
        g.add("Gather", [qkv_t, idx1], [k_mat], axis=0)
        g.add("Gather", [qkv_t, idx2], [v_mat], axis=0)

        # ── Attention Scores: Q @ K^T ─────────────────────────────────────────
        # K^T: (1, H, d, N) via Transpose
        kt_mat = f"{p}_kt"
        g.add("Transpose", [k_mat], [kt_mat], perm=[0, 1, 3, 2])

        # QLinearMatMul: (1,H,N,d) @ (1,H,d,N) → (1,H,N,N)
        # Q, K, V all share scale = blocks.{blk}.attn.qact1.act_scaling_factor
        # (TVM convert_model.py L97: input_scale for matmul_1 is attn.qact1)
        attn_raw_sc = get_scale(sd, f"{b}.attn.matmul_1.act_scaling_factor")
        attn_scores = g.gemmini_matmul_integer_inputs(q_mat, kt_mat, f"{p}_attn_mm")

        # Requantize attn scores to qact_attn1 scale
        qact_attn1_sc = get_scale(sd, f"{b}.attn.qact_attn1.act_scaling_factor")
        # In I-ViT PyTorch/TVM path, attention logits are scaled by qk_scale
        # (= head_dim^-0.5) before quantization to qact_attn1.
        attn_logits_sc = attn_raw_sc * (HEAD_DIM ** -0.5)
        attn_req = g.requantize_int32(attn_scores, attn_logits_sc, qact_attn1_sc, f"{p}_attn_req")

        # ── Shiftmax ───────────────────────────────────────────────────────────
        softmax_scale_ckpt = get_scale(sd, f"{b}.attn.int_softmax.act_scaling_factor")
        # TVM uses softmax_scale_int8 = softmax_scale * 256 (int8 domain correction)
        softmax_int8_sc = softmax_scale_ckpt * 256.0

        sm_out = g.shiftmax(attn_req, qact_attn1_sc, f"{p}_softmax")
        # sm_out: int8 (1, H, N, N), scale = softmax_int8_sc

        # ── Attn @ V ───────────────────────────────────────────────────────────
        # V scale = qkv_out_sc (attn.qact1); attn scale = softmax_int8_sc
        # TVM convert_model.py L109-111: matmul_2 input = softmax_int8_sc, qact1_scale
        # Checkpoint matmul_2 scale is based on the 16-bit IntSoftmax output.
        # The exported ORT/TVM path uses the int8-corrected Shiftmax scale (*256),
        # so the matmul natural scale must be widened accordingly.
        matmul2_sc = get_scale(sd, f"{b}.attn.matmul_2.act_scaling_factor") * 256.0
        attn_v = g.gemmini_matmul_integer_inputs(sm_out, v_mat, f"{p}_av_mm")
        # attn_v: int32 (1, H, N, d)

        # Transpose + Reshape: (1,H,N,d) → (1,N,H*d) = (1,N,192)
        qact2_attn_sc = get_scale(sd, f"{b}.attn.qact2.act_scaling_factor")
        attn_v_req    = g.requantize_int32(attn_v, matmul2_sc, qact2_attn_sc, f"{p}_attn_v_req")
        attn_perm     = f"{p}_attn_perm"
        g.add("Transpose", [attn_v_req], [attn_perm], perm=[0, 2, 1, 3])
        attn_flat_shape = g.init_tensor(f"{p}_attn_flat_shape",
            np.array([1, SEQ_LEN, EMBED_DIM], dtype=np.int64))
        attn_flat = f"{p}_attn_flat"
        g.add("Reshape", [attn_perm, attn_flat_shape], [attn_flat])

        # ── Output Projection ─────────────────────────────────────────────────
        proj_w  = get_weight(sd, f"{b}.attn.proj.weight_integer")  # (192, 192)
        proj_b  = get_weight(sd, f"{b}.attn.proj.bias_integer")
        proj_ws = get_scale(sd, f"{b}.attn.proj.fc_scaling_factor")
        qact3_sc = get_scale(sd, f"{b}.attn.qact3.act_scaling_factor")

        proj_out = g.qlinear_matmul_bias(
            attn_flat, qact2_attn_sc,
            proj_w.T, proj_ws,
            proj_b,
            qact3_sc,
            "int16",
            f"{p}_proj"
        )

        # ── Residual Add 1 ────────────────────────────────────────────────────
        qact2_blk_sc = get_scale(sd, f"{b}.qact2.act_scaling_factor")
        x_res1 = g.residual_add(
            x_blk, blk_in_scale,
            proj_out, qact3_sc,
            qact2_blk_sc,
            f"{p}_res1"
        )

        # ── LayerNorm 2 ───────────────────────────────────────────────────────
        norm2_bias_arr, norm2_out_sc = fix_norm_bias_and_scale(sd, blk, "norm2")
        ln2_out = g.q_layernorm(x_res1, norm2_bias_arr, f"{p}_ln2", input_dtype="int32")
        qact3_blk_sc = get_scale(sd, f"{b}.qact3.act_scaling_factor")
        ln2_req = g.requantize_int32(ln2_out, norm2_out_sc, qact3_blk_sc, f"{p}_ln2_req")

        # ── FC1 + ShiftGELU ───────────────────────────────────────────────────
        fc1_w   = get_weight(sd, f"{b}.mlp.fc1.weight_integer")  # (768, 192)
        fc1_b   = get_weight(sd, f"{b}.mlp.fc1.bias_integer")
        fc1_ws  = get_scale(sd, f"{b}.mlp.fc1.fc_scaling_factor")
        qact_gelu_sc = get_scale(sd, f"{b}.mlp.qact_gelu.act_scaling_factor")

        fc1_out = g.qlinear_matmul_bias(
            ln2_req, qact3_blk_sc,
            fc1_w.T, fc1_ws,
            fc1_b,
            qact_gelu_sc,
            "int8",
            f"{p}_fc1"
        )

        gelu_out = g.shift_gelu(fc1_out, qact_gelu_sc, f"{p}_gelu")
        # gelu_out: int32 — use QAT-calibrated output scale from checkpoint
        # (= blocks.{blk}.mlp.act.act_scaling_factor, matches TVM convert_model.py L132-134)
        gelu_out_sc = get_scale(sd, f"{b}.mlp.act.act_scaling_factor")
        qact_mlp1_sc = get_scale(sd, f"{b}.mlp.qact1.act_scaling_factor")
        gelu_req = g.requantize_int32(gelu_out, gelu_out_sc, qact_mlp1_sc, f"{p}_gelu_req")

        # ── FC2 + QLayerNorm ──────────────────────────────────────────────────
        fc2_w   = get_weight(sd, f"{b}.mlp.fc2.weight_integer")  # (192, 768)
        fc2_b   = get_weight(sd, f"{b}.mlp.fc2.bias_integer")
        fc2_ws  = get_scale(sd, f"{b}.mlp.fc2.fc_scaling_factor")
        qact_mlp2_sc = get_scale(sd, f"{b}.mlp.qact2.act_scaling_factor")

        fc2_out = g.qlinear_matmul_bias(
            gelu_req, qact_mlp1_sc,
            fc2_w.T, fc2_ws,
            fc2_b,
            qact_mlp2_sc,
            "int16",
            f"{p}_fc2"
        )

        qact4_sc = get_scale(sd, f"{b}.qact4.act_scaling_factor")

        # ── Residual Add 2 ────────────────────────────────────────────────────
        x_blk = g.residual_add(
            x_res1, qact2_blk_sc,
            fc2_out, qact_mlp2_sc,
            qact4_sc,
            f"{p}_res2"
        )

    # x_blk: int8 (1, 197, 192), scale = qact4_sc of last block

    # ── Extract CLS token ──────────────────────────────────────────────────────
    last_scale = get_scale(sd, f"blocks.{DEPTH-1}.qact4.act_scaling_factor")
    cls_idx = g.init_tensor("cls_idx", np.array([0], dtype=np.int64))
    cls_tok = "cls_tok"
    g.add("Gather", [x_blk, cls_idx], [cls_tok], axis=1)
    # cls_tok: int8 (1, 1, 192)

    # ── Final LayerNorm ────────────────────────────────────────────────────────
    final_norm_bias_arr, final_norm_out_sc = fix_final_norm_bias_and_scale(sd)
    final_ln = g.q_layernorm(cls_tok, final_norm_bias_arr, "final_ln", input_dtype="int32")
    qact2_final = get_scale(sd, "qact2.act_scaling_factor")
    final_ln_req = g.requantize_int32(final_ln, final_norm_out_sc, qact2_final,
                                      "final_ln_req")

    # Reshape: (1, 1, 192) → (1, 192)
    head_shape = g.init_tensor("head_shape",
                                np.array([1, EMBED_DIM], dtype=np.int64))
    cls_flat = "cls_flat"
    g.add("Reshape", [final_ln_req, head_shape], [cls_flat])

    # ── Classification Head ────────────────────────────────────────────────────
    head_w  = get_weight(sd, "head.weight_integer")   # (1000, 192)
    head_b  = get_weight(sd, "head.bias_integer")     # (1000,)
    head_ws = get_scale(sd, "head.fc_scaling_factor") # scalar or [1000]

    # TVM: output_scale = qact2 * head_ws  (convert_model.py L148-150)
    head_int32 = g.gemmini_matmul_integer(
        cls_flat,
        head_w.T.astype(np.int8),
        "head_mm",
        force_cpu=True,
    )
    head_b_name = g.init_tensor("head_bias", to_int32(head_b))
    g.add("Add", [head_int32, head_b_name], ["head_biased"])

    # Dequantize to float32: Cast(int32→f32) × (qact2_final × head_ws)
    g.add("Cast", ["head_biased"], ["head_cast_f32"], to=TensorProto.FLOAT)
    if isinstance(head_ws, np.ndarray) and head_ws.ndim > 0:
        # per-column scale: shape [1000] → broadcast over [1, 1000]
        head_out_scale_arr = (qact2_final * head_ws).astype(np.float32)
        head_dq_scale = g.init_tensor("head_dq_scale", head_out_scale_arr)
    else:
        head_dq_scale = g.scalar_f32("head_dq_scale",
                                      qact2_final * float(head_ws))
    logits = "logits"
    g.add("Mul", ["head_cast_f32", head_dq_scale], [logits])

    # ── Build ONNX graph ───────────────────────────────────────────────────────
    input_vi  = helper.make_tensor_value_info("image",  TensorProto.FLOAT,
                                               [1, IN_CHANNELS, IMG_SIZE, IMG_SIZE])
    output_vi = helper.make_tensor_value_info("logits", TensorProto.FLOAT,
                                               [1, NUM_CLASSES])

    graph = helper.make_graph(
        g.nodes,
        f"ivit_{model_name.replace('_patch16_224', '')}",
        [input_vi],
        [output_vi],
        initializer=g.initializers,
    )

    opset_imports = [
        helper.make_opsetid("", 13),        # default ONNX opset
        helper.make_opsetid("com.microsoft", 1),
        helper.make_opsetid("ivit", 1),     # custom domain for integer ops
    ]

    model = helper.make_model(graph, opset_imports=opset_imports)
    model.ir_version = 7
    return model


# ── State dict loading ────────────────────────────────────────────────────────

def load_state_dict(checkpoint_path: str) -> dict:
    """Load I-ViT checkpoint state_dict."""
    return load_checkpoint_state_dict(checkpoint_path)


def build_random_state_dict(model_name: str) -> dict:
    import torch

    sys.path.insert(0, str(REPO_ROOT / "I-ViT"))
    from models.model_utils import freeze_model

    model = create_model(model_name)
    model.eval()
    freeze_model(model)
    with torch.no_grad():
        model(torch.randn(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE))
    return model.state_dict()


# ── Host verification ─────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(path):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((int(w*s), int(h*s)), Image.BILINEAR)
    w2, h2 = img.size
    left = (w2 - 224) // 2
    top  = (h2 - 224) // 2
    img  = img.crop((left, top, left+224, top+224))
    arr  = np.array(img, dtype=np.float32) / 255.0
    arr  = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)[np.newaxis, ...]


def verify(model_path, image_path):
    import onnxruntime as ort
    print(f"\nHost verification (CPU, host custom ops):")
    print(f"  Model: {model_path}")
    print(f"  Image: {image_path}")
    print(f"  Custom ops: {DEFAULT_HOST_CUSTOM_OP_LIB}")

    sess_options = ort.SessionOptions()
    try:
        if not os.path.exists(DEFAULT_HOST_CUSTOM_OP_LIB):
            print("  NOTE: host custom op library not found.")
            print("  Build it with: make -C ONNXRuntime/ort_ivit_ops host")
            return
        sess_options.register_custom_ops_library(DEFAULT_HOST_CUSTOM_OP_LIB)
        sess = ort.InferenceSession(model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"  NOTE: failed to register or load host custom ops — {e}")
        return

    inp = preprocess_image(image_path)
    out = sess.run(None, {"image": inp})[0][0]
    top5 = np.argsort(out)[::-1][:5]
    print(f"  Top-5 class indices: {top5.tolist()}")
    print(f"  Top-5 scores:        {out[top5].tolist()}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build I-ViT DeiT INT8 ONNX graph for onnxruntime-riscv/Gemmini",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        default="deit_tiny_patch16_224",
        choices=SUPPORTED_MODEL_NAMES,
        help="I-ViT DeiT model variant",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT,
                        help="I-ViT QAT checkpoint (checkpoint_last.pth.tar)")
    parser.add_argument("--output",     default=OUTPUT_ONNX,
                        help="Output ONNX path")
    parser.add_argument("--allow-random-init", action="store_true",
                        help="Allow export without checkpoint using random weights")
    parser.add_argument("--verify",     action="store_true",
                        help="Run host onnxruntime sanity check after export")
    parser.add_argument("--image",      default=DEFAULT_IMAGE,
                        help="Test image for --verify")
    args = parser.parse_args()

    spec = configure_model(args.model_name)
    if args.checkpoint is None and not args.allow_random_init:
        print("ERROR: --checkpoint is required unless --allow-random-init is set")
        sys.exit(1)
    if args.checkpoint is not None and not os.path.exists(args.checkpoint) and not args.allow_random_init:
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        sd = load_state_dict(args.checkpoint)
    else:
        print(f"[WARN] Exporting {args.model_name} with random-initialized weights.")
        sd = build_random_state_dict(args.model_name)
    print(f"  Keys loaded: {len(sd)}")

    print(f"\nBuilding I-ViT DeiT-{spec['tag'].title()} ONNX graph ...")
    model = build_ivit_graph(sd, model_name=args.model_name)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    onnx.save(model, args.output)
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Saved: {args.output} ({size_mb:.1f} MB)")

    print("\nValidating ONNX model structure ...")
    try:
        onnx.checker.check_model(model)
        print("  Structure OK")
    except Exception as e:
        print(f"  Warning: {e}")
        print("  (Custom ops cause check failures — this is expected)")

    if args.verify:
        verify(args.output, args.image)


if __name__ == "__main__":
    main()
