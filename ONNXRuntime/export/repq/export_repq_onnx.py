#!/usr/bin/env python3
"""Host-side RepQ-ViT calibration/reparameterization and semantic ONNX export."""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import Parameter
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
ONNXRT_DIR = SCRIPT_DIR.parent.parent
REPO_ROOT = ONNXRT_DIR.parent
REPQ_CLASSIFICATION_ROOT = Path(
    os.environ.get(
        "REPQ_CLASSIFICATION_ROOT",
        str(REPO_ROOT / "RepQ-ViT" / "classification"),
    )
)
DEFAULT_OUTPUT = REPO_ROOT / "build" / "ort" / "repq_deit_tiny_w8a8_semantic.onnx"
DEFAULT_HOST_CUSTOM_OP_LIB = REPO_ROOT / "build" / "ort" / "ort_ivit_ops" / "libivit_ops_host.so"

if str(REPQ_CLASSIFICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(REPQ_CLASSIFICATION_ROOT))

from quant import quant_model, set_quant_state  # noqa: E402
from quant.quant_modules import QuantConv2d, QuantLinear, QuantMatMul  # noqa: E402
from utils import build_model, build_transform  # noqa: E402


MODEL_ZOO = {
    "vit_small": "vit_small_patch16_224",
    "vit_base": "vit_base_patch16_224",
    "deit_tiny": "deit_tiny_patch16_224",
    "deit_small": "deit_small_patch16_224",
    "deit_base": "deit_base_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "swin_small": "swin_small_patch4_window7_224",
}

EXPORT_OP_SUMMARY = (
    "Conv",
    "MatMul",
    "Gemm",
    "Softmax",
    "Round",
    "Clip",
    "Log",
    "Pow",
    "QLinearMatMul",
    "RepQUniformMatMul",
    "RepQLogMatMul",
    "QuantizeLinear",
    "DequantizeLinear",
    "Cast",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize and export RepQ-ViT to an ONNX graph on the host",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="deit_tiny",
        choices=sorted(MODEL_ZOO),
        help="RepQ-ViT model key",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="ImageNet root, or an ImageFolder root used for calibration/export",
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        default=None,
        help="Optional ImageFolder root used for calibration. Overrides --dataset for calibration.",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Optional ImageFolder root used to source export/example inputs. Overrides --dataset for export inputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output ONNX model path",
    )
    parser.add_argument(
        "--save-reparam-state",
        type=Path,
        default=None,
        help="Optional path to save the reparameterized quantized PyTorch state_dict",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional local timm-compatible checkpoint path for the FP32 source model",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional timm cache dir for pretrained weights",
    )
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Build the source model without pretrained weights",
    )
    parser.add_argument(
        "--allow-random-calibration",
        action="store_true",
        help="Use random tensors when ImageNet data is unavailable",
    )
    parser.add_argument(
        "--calib-batchsize",
        default=32,
        type=int,
        help="Calibration batch size",
    )
    parser.add_argument(
        "--calib-num-samples",
        default=32,
        type=int,
        help="Number of ImageNet samples used for calibration",
    )
    parser.add_argument(
        "--val-batchsize",
        default=16,
        type=int,
        help="Validation batch size when using ImageNet",
    )
    parser.add_argument(
        "--num-workers",
        default=8,
        type=int,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--w-bits",
        default=8,
        type=int,
        help="Weight bit-precision",
    )
    parser.add_argument(
        "--a-bits",
        default=8,
        type=int,
        help="Activation bit-precision",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Host device used for calibration/export",
    )
    parser.add_argument(
        "--opset",
        default=13,
        type=int,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--verify-ort",
        action="store_true",
        help="Run host ONNX Runtime on the exported graph and compare logits",
    )
    parser.add_argument(
        "--host-custom-op-library",
        type=Path,
        default=DEFAULT_HOST_CUSTOM_OP_LIB,
        help="Host ORT custom-op library used when verifying exported custom ops",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export a dynamic batch dimension on the ONNX input/output",
    )
    parser.add_argument(
        "--lower-qlinear-matmul",
        action="store_true",
        help="Rewrite eligible RepQ MatMul/Gemm nodes into QLinearMatMul + dequantize form",
    )
    parser.add_argument(
        "--repq-gemmini-kernel-mode",
        choices=("approx", "exact"),
        default="approx",
        help="Kernel mode used for lowered RepQ custom matmul ops",
    )
    parser.add_argument(
        "--export-batch-size",
        default=1,
        type=int,
        help="Batch size used for the ONNX example input",
    )
    parser.add_argument(
        "--skip-onnx-checker",
        action="store_true",
        help="Skip onnx.checker.check_model after export",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def get_model_preprocess(model_key: str) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    model_type = model_key.split("_")[0]
    if model_type == "deit":
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 0.875
    if model_type == "vit":
        return (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 0.9
    if model_type == "swin":
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 0.9
    raise NotImplementedError(f"Unsupported model family in {model_key}")


def resolve_imagefolder_root(root: Path, preferred_split: str | None) -> Path:
    if preferred_split is not None and (root / preferred_split).is_dir():
        return root / preferred_split
    return root


def load_dataset_samples(
    root: Path,
    transform,
    num_samples: int,
    seed: int,
    random_sample: bool,
) -> torch.Tensor:
    from torchvision import datasets

    dataset = datasets.ImageFolder(str(root), transform)
    if len(dataset) == 0:
        raise RuntimeError(f"No images found under: {root}")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")

    take = min(num_samples, len(dataset))
    if random_sample:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(dataset), size=take, replace=False).tolist()
    else:
        indices = list(range(take))

    batch = [dataset[index][0] for index in indices]
    return torch.stack(batch, dim=0)


def build_example_inputs(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    data_root = Path(args.dataset) if args.dataset else None
    calib_root = args.calib_dir or data_root
    eval_root = args.eval_dir or data_root

    if calib_root is not None:
        mean, std, crop_pct = get_model_preprocess(args.model)
        transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)
        calib_dir = resolve_imagefolder_root(Path(calib_root), "train")
        eval_dir = resolve_imagefolder_root(Path(eval_root or calib_root), "val")

        calib_data = load_dataset_samples(
            calib_dir,
            transform,
            num_samples=args.calib_num_samples,
            seed=args.seed,
            random_sample=True,
        )
        export_data = load_dataset_samples(
            eval_dir,
            transform,
            num_samples=args.export_batch_size,
            seed=args.seed,
            random_sample=False,
        )
        print(
            f"Loaded calibration/export images from {calib_dir} and {eval_dir} "
            f"(calibration samples={calib_data.shape[0]})"
        )
        return calib_data.to(device), export_data.to(device)

    if not args.allow_random_calibration:
        raise ValueError(
            "An ImageNet/ImageFolder path is required unless --allow-random-calibration is set."
        )

    calib_shape = (args.calib_batchsize, 3, 224, 224)
    export_shape = (args.export_batch_size, 3, 224, 224)
    calib_data = torch.randn(calib_shape, device=device)
    export_data = torch.randn(export_shape, device=device)
    print("[WARN] Using random calibration/export tensors; outputs are for structural validation only.")
    return calib_data, export_data


def build_fp_model(args: argparse.Namespace, model_name: str) -> torch.nn.Module:
    use_pretrained = not args.allow_random_init and args.checkpoint is None
    if use_pretrained:
        print(f"Building TIMM pretrained model: {model_name}")
    elif args.checkpoint is not None:
        print(f"Building model from local checkpoint: {args.checkpoint}")
    else:
        print(f"Building randomly initialized model: {model_name}")

    return build_model(
        model_name,
        pretrained=use_pretrained,
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        exportable=True,
    )


def apply_scale_reparameterization(q_model: torch.nn.Module, model_key: str) -> None:
    module_dict: dict[str, torch.nn.Module] = {}
    q_model_slice = q_model.layers if "swin" in model_key else q_model.blocks
    for name, module in q_model_slice.named_modules():
        module_dict[name] = module
        idx = name.rfind(".")
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name not in module_dict:
            raise RuntimeError(f"father module {father_name!r} not found")
        father_module = module_dict[father_name]

        if isinstance(module, nn.LayerNorm):
            if name.endswith("norm1"):
                next_module = father_module.attn.qkv
            elif name.endswith("norm2"):
                next_module = father_module.mlp.fc1
            elif name.endswith("norm"):
                next_module = father_module.reduction
            else:
                continue

            act_delta = next_module.input_quantizer.delta.reshape(-1)
            act_zero_point = next_module.input_quantizer.zero_point.reshape(-1)
            act_min = -act_zero_point * act_delta

            target_delta = torch.mean(act_delta)
            target_zero_point = torch.mean(act_zero_point)
            target_min = -target_zero_point * target_delta

            r = act_delta / target_delta
            b = act_min / r - target_min

            module.weight.data = module.weight.data / r
            module.bias.data = module.bias.data / r - b

            next_module.weight.data = next_module.weight.data * r
            if next_module.bias is not None:
                next_module.bias.data = next_module.bias.data + torch.mm(
                    next_module.weight.data, b.reshape(-1, 1)
                ).reshape(-1)
            else:
                next_module.bias = Parameter(torch.empty(next_module.out_features))
                next_module.bias.data = torch.mm(
                    next_module.weight.data, b.reshape(-1, 1)
                ).reshape(-1)

            next_module.input_quantizer.channel_wise = False
            next_module.input_quantizer.delta = target_delta
            next_module.input_quantizer.zero_point = target_zero_point
            next_module.weight_quantizer.inited = False


def prepare_quantized_model(
    args: argparse.Namespace,
    device: torch.device,
    calib_data: torch.Tensor,
) -> torch.nn.Module:
    model_name = MODEL_ZOO[args.model]
    model = build_fp_model(args, model_name)
    model.to(device)
    model.eval()

    wq_params = {"n_bits": args.w_bits, "channel_wise": True}
    aq_params = {"n_bits": args.a_bits, "channel_wise": False}
    q_model = quant_model(
        model,
        input_quant_params=aq_params,
        weight_quant_params=wq_params,
    )
    q_model.to(device)
    q_model.eval()

    def run_calibration_pass(tag: str) -> None:
        print(f"{tag}...")
        with torch.no_grad():
            for start in range(0, calib_data.shape[0], args.calib_batchsize):
                end = min(start + args.calib_batchsize, calib_data.shape[0])
                _ = q_model(calib_data[start:end])

    set_quant_state(q_model, input_quant=True, weight_quant=True)
    run_calibration_pass("Running initial calibration pass")

    print("Applying RepQ scale reparameterization...")
    with torch.no_grad():
        apply_scale_reparameterization(q_model, args.model)

    set_quant_state(q_model, input_quant=True, weight_quant=True)
    run_calibration_pass("Running post-reparameterization calibration pass")

    return q_model


def collect_qlinear_lowering_stats(
    model: torch.nn.Module,
    calib_data: torch.Tensor,
    batch_size: int,
) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    hooks = []

    def symmetric_scale(tensor: torch.Tensor) -> float:
        max_abs = float(tensor.detach().abs().max().cpu())
        return max(max_abs / 127.0, 1e-8)

    def register(module_name: str, module: torch.nn.Module) -> None:
        def hook(_module, inputs, output):
            observed = output.detach()
            if isinstance(module, (QuantLinear, QuantConv2d)) and module.bias is not None:
                bias = module.bias.detach().to(observed.device, observed.dtype)
                view_shape = [1] * observed.ndim
                if isinstance(module, QuantConv2d):
                    view_shape[1] = bias.numel()
                else:
                    view_shape[-1] = bias.numel()
                observed = observed - bias.view(*view_shape)
            record: dict[str, float] = {"y_scale": symmetric_scale(observed)}
            if isinstance(module, (QuantLinear, QuantConv2d)):
                record["a_scale"] = symmetric_scale(inputs[0])
            elif isinstance(module, QuantMatMul):
                record["a_scale"] = symmetric_scale(inputs[0])
                record["b_scale"] = symmetric_scale(inputs[1])
            stats[module_name] = record

        hooks.append(module.register_forward_hook(hook))

    for module_name, module in model.named_modules():
        if isinstance(module, (QuantLinear, QuantConv2d)):
            register(module_name, module)
        elif (
            isinstance(module, QuantMatMul)
            and module.quantizer_A.__class__.__name__ == "UniformQuantizer"
            and module.quantizer_B.__class__.__name__ == "UniformQuantizer"
        ):
            register(module_name, module)

    with torch.no_grad():
        for start in range(0, calib_data.shape[0], batch_size):
            end = min(start + batch_size, calib_data.shape[0])
            _ = model(calib_data[start:end])

    for hook in hooks:
        hook.remove()
    return stats


def lower_qlinear_matmul_nodes(
    output_path: Path,
    lowering_stats: dict[str, dict[str, float | int]],
    n_bits: int = 8,
    repq_gemmini_kernel_mode: str = "approx",
) -> dict[str, int]:
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(output_path))
    init_map = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
    const_map: dict[str, np.ndarray] = {}
    producer: dict[str, onnx.NodeProto] = {}

    for node in model.graph.node:
        for output in node.output:
            producer[output] = node
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    const_map[node.output[0]] = numpy_helper.to_array(attr.t)
                    break

    name_counter: dict[str, int] = {}
    profile_node_index = 0

    def unique_name(prefix: str) -> str:
        count = name_counter.get(prefix, 0)
        name_counter[prefix] = count + 1
        return prefix if count == 0 else f"{prefix}_{count}"

    def sanitize(prefix: str) -> str:
        return prefix.replace("/", "_").replace(".", "_").replace(":", "_")

    def make_profile_attrs(raw_name: str) -> dict[str, object]:
        nonlocal profile_node_index
        label = sanitize(raw_name).lstrip("_") or "unnamed"
        attrs = {
            "profile_label": label,
            "profile_node_index": int(profile_node_index),
        }
        profile_node_index += 1
        return attrs

    def get_attr(node, attr_name: str, default=None):
        for attr in node.attribute:
            if attr.name == attr_name:
                return helper.get_attribute_value(attr)
        return default

    def get_const(name: str) -> np.ndarray:
        if name in init_map:
            return init_map[name]
        if name in const_map:
            return const_map[name]
        raise KeyError(f"Constant tensor not found: {name}")

    def add_initializer(base_name: str, array: np.ndarray) -> str:
        name = unique_name(base_name)
        tensor = numpy_helper.from_array(array, name=name)
        model.graph.initializer.append(tensor)
        init_map[name] = array
        return name

    def scalar_f32(value: float) -> np.ndarray:
        return np.asarray(np.float32(max(value, 1e-8)), dtype=np.float32)

    def scalar_i8(value: int = 0) -> np.ndarray:
        return np.asarray(np.int8(value), dtype=np.int8)

    def symmetric_scale(values: np.ndarray) -> float:
        max_abs = float(np.max(np.abs(values.astype(np.float32))))
        return max(max_abs / 127.0, 1e-8)

    def parse_uniform_dequant(tensor_name: str):
        mul_node = producer.get(tensor_name)
        if mul_node is None or mul_node.op_type != "Mul":
            return None

        scale_input = None
        sub_input = None
        for input_name in mul_node.input:
            if input_name in init_map or input_name in const_map:
                scale_input = input_name
            else:
                sub_input = input_name
        if scale_input is None or sub_input is None:
            return None

        sub_node = producer.get(sub_input)
        if sub_node is None or sub_node.op_type != "Sub":
            return None
        clip_input = None
        zp_input = None
        for input_name in sub_node.input:
            if input_name in init_map or input_name in const_map:
                zp_input = input_name
            else:
                clip_input = input_name
        if clip_input is None or zp_input is None:
            return None

        clip_node = producer.get(clip_input)
        add_node = producer.get(clip_node.input[0]) if clip_node and clip_node.op_type == "Clip" else None
        round_node = producer.get(add_node.input[0]) if add_node and add_node.op_type == "Add" else None
        div_node = producer.get(round_node.input[0]) if round_node and round_node.op_type == "Round" else None
        if div_node is None or div_node.op_type != "Div":
            return None

        source_name = div_node.input[0]
        return {
            "source": source_name,
            "scale": scale_input,
            "zero_point": zp_input,
        }

    def quantize_int8(values: np.ndarray, scale: float) -> np.ndarray:
        quantized = np.rint(values.astype(np.float32) / np.float32(scale))
        return np.clip(quantized, -127, 127).astype(np.int8)

    def sym_scale_from_uniform_params(scale_name: str, zero_point_name: str) -> float:
        scales = get_const(scale_name).astype(np.float32).reshape(-1)
        zero_points = get_const(zero_point_name).astype(np.float32).reshape(-1)
        levels = float((1 << int(n_bits)) - 1)
        qmin = -zero_points
        qmax = levels - zero_points
        max_abs = max(
            float(np.max(np.abs(qmin * scales))),
            float(np.max(np.abs(qmax * scales))),
        )
        return max(max_abs / 127.0, 1e-8)

    def sym_scale_from_log_delta(delta_name: str) -> float:
        delta = float(np.asarray(get_const(delta_name), dtype=np.float32).reshape(-1)[0])
        return max(abs(delta) / 127.0, 1e-8)

    def parse_weight_input(tensor_name: str, force_transpose: bool = False):
        transpose_node = producer.get(tensor_name)
        transpose_perm = None
        weight_dequant_output = tensor_name

        if transpose_node is not None and transpose_node.op_type == "Transpose":
            weight_dequant_output = transpose_node.input[0]
            transpose_perm = get_attr(transpose_node, "perm", None)

        qinfo = parse_uniform_dequant(weight_dequant_output)
        if qinfo is None:
            return None
        source_name = qinfo["source"]
        if source_name not in init_map:
            return None

        weight_float = init_map[source_name].astype(np.float32)
        if transpose_perm is not None:
            weight_float = np.transpose(weight_float, axes=tuple(transpose_perm))
        elif force_transpose:
            weight_float = np.transpose(weight_float, axes=(1, 0))

        weight_scale = symmetric_scale(weight_float)
        weight_quant = quantize_int8(weight_float, weight_scale)
        return {
            "data": weight_quant.astype(np.int8),
            "scale": scalar_f32(weight_scale),
            "zero_point": scalar_i8(0),
        }

    def parse_exact_weight_input(tensor_name: str, force_transpose: bool = False):
        transpose_node = producer.get(tensor_name)
        transpose_perm = None
        weight_dequant_output = tensor_name

        if transpose_node is not None and transpose_node.op_type == "Transpose":
            weight_dequant_output = transpose_node.input[0]
            transpose_perm = get_attr(transpose_node, "perm", None)

        qinfo = parse_uniform_dequant(weight_dequant_output)
        if qinfo is None:
            return None
        source_name = qinfo["source"]
        if source_name not in init_map:
            return None

        weight_float = init_map[source_name].astype(np.float32)
        if transpose_perm is not None:
            weight_float = np.transpose(weight_float, axes=tuple(transpose_perm))
        elif force_transpose:
            weight_float = np.transpose(weight_float, axes=(1, 0))

        scale = get_const(qinfo["scale"]).astype(np.float32)
        zero_point = get_const(qinfo["zero_point"]).astype(np.float32)
        if scale.size == 1:
            exact_scale = scalar_f32(float(scale.reshape(-1)[0]))
        else:
            exact_scale = scale.reshape(-1).astype(np.float32)
        if zero_point.size == 1:
            exact_zero_point = np.asarray(np.float32(zero_point.reshape(-1)[0]), dtype=np.float32)
        else:
            exact_zero_point = zero_point.reshape(-1).astype(np.float32)

        source_init_name = add_initializer(f"{sanitize(tensor_name)}_b_source", weight_float.astype(np.float32))
        scale_name = add_initializer(f"{sanitize(tensor_name)}_b_scale_exact", exact_scale)
        zp_name = add_initializer(f"{sanitize(tensor_name)}_b_zp_exact", exact_zero_point)
        return {
            "source": source_init_name,
            "scale": scale_name,
            "zero_point": zp_name,
        }

    def node_to_module_key(node_name: str, output_name: str) -> str:
        raw_name = node_name or output_name
        parts = [part for part in raw_name.split("/") if part]
        body = parts[:-1]
        normalized = []
        for idx, part in enumerate(body):
            next_part = body[idx + 1] if idx + 1 < len(body) else None
            if next_part is not None and next_part.startswith(f"{part}."):
                continue
            normalized.append(part)
        return ".".join(normalized)

    def append_approx_gemmini_matmul(
        *,
        a_source: str,
        a_sym_scale: float,
        b_source: str | None,
        b_sym_scale: float,
        out_names: list[str],
        prefix: str,
        raw_name: str,
        b_const_int8: np.ndarray | None = None,
    ) -> None:
        profile_attrs = make_profile_attrs(raw_name)
        zero_point_name = add_initializer(f"{prefix}_sym_zp", scalar_i8(0))
        a_scale_name = add_initializer(f"{prefix}_a_sym_scale", scalar_f32(a_sym_scale))
        a_quant_name = unique_name(f"{prefix}_a_quant")
        rewritten.append(
            helper.make_node(
                "QuantizeLinear",
                [a_source, a_scale_name, zero_point_name],
                [a_quant_name],
                name=unique_name(f"{prefix}_AQuantizeLinear"),
            )
        )

        if b_const_int8 is None:
            if b_source is None:
                raise RuntimeError("Approx Gemmini matmul requires either b_source or b_const_int8")
            b_scale_name = add_initializer(f"{prefix}_b_sym_scale", scalar_f32(b_sym_scale))
            b_quant_name = unique_name(f"{prefix}_b_quant")
            rewritten.append(
                helper.make_node(
                    "QuantizeLinear",
                    [b_source, b_scale_name, zero_point_name],
                    [b_quant_name],
                    name=unique_name(f"{prefix}_BQuantizeLinear"),
                )
            )
        else:
            b_quant_name = add_initializer(f"{prefix}_b_sym_data", b_const_int8.astype(np.int8))

        matmul_output = unique_name(f"{prefix}_gemmini_mm")
        rewritten.append(
            helper.make_node(
                "GemminiMatMulInteger",
                [a_quant_name, b_quant_name, zero_point_name, zero_point_name],
                [matmul_output],
                domain="ivit",
                name=unique_name(f"{prefix}_GemminiMatMulInteger"),
                **profile_attrs,
            )
        )

        cast_output = unique_name(f"{prefix}_gemmini_mm_f32")
        rewritten.append(
            helper.make_node(
                "Cast",
                [matmul_output],
                [cast_output],
                name=unique_name(f"{prefix}_CastFloat"),
                to=onnx.TensorProto.FLOAT,
            )
        )
        mm_scale_name = add_initializer(f"{prefix}_gemmini_mm_scale", scalar_f32(a_sym_scale * b_sym_scale))
        rewritten.append(
            helper.make_node(
                "Mul",
                [cast_output, mm_scale_name],
                list(out_names),
                name=unique_name(f"{prefix}_DequantMul"),
            )
        )

    rewritten: list = []
    replace_output: dict[str, str] = {}
    lowered_counts = {
        "qlinear_conv": 0,
        "repq_uniform_matmul": 0,
        "repq_log_matmul": 0,
    }
    approximate_attr = 1 if repq_gemmini_kernel_mode == "approx" else 0

    for node in model.graph.node:
        original_inputs = list(node.input)
        rewritten_inputs = [replace_output.get(name, name) for name in original_inputs]

        if node.op_type == "MatMul":
            module_key = node_to_module_key(node.name, node.output[0])
            prefix = sanitize(node.name or node.output[0])

            repq_node = producer.get(original_inputs[0])
            if repq_node is not None and repq_node.op_type == "RepQLogQuant":
                b_info = parse_uniform_dequant(original_inputs[1])
                if b_info is not None:
                    if repq_gemmini_kernel_mode == "approx":
                        try:
                            attn_sym_scale = sym_scale_from_log_delta(repq_node.input[1])
                            value_sym_scale = sym_scale_from_uniform_params(b_info["scale"], b_info["zero_point"])
                        except KeyError:
                            attn_sym_scale = None
                            value_sym_scale = None
                        if attn_sym_scale is not None and value_sym_scale is not None:
                            append_approx_gemmini_matmul(
                                a_source=rewritten_inputs[0],
                                a_sym_scale=attn_sym_scale,
                                b_source=b_info["source"],
                                b_sym_scale=value_sym_scale,
                                out_names=list(node.output),
                                prefix=prefix,
                                raw_name=node.name or node.output[0],
                            )
                            lowered_counts["repq_log_matmul"] += 1
                            continue
                    profile_attrs = make_profile_attrs(node.name or node.output[0])
                    rewritten.append(
                        helper.make_node(
                            "RepQLogMatMul",
                            [
                                repq_node.input[0],
                                repq_node.input[1],
                                b_info["source"],
                                b_info["scale"],
                                b_info["zero_point"],
                            ],
                            list(node.output),
                            domain="ivit",
                            name=unique_name(f"{prefix}_RepQLogMatMul"),
                            n_bits=int(get_attr(repq_node, "n_bits", 8)),
                            approximate=approximate_attr,
                            **profile_attrs,
                        )
                    )
                    lowered_counts["repq_log_matmul"] += 1
                    continue

            linear_qinfo = parse_uniform_dequant(original_inputs[0])
            weight_info = parse_exact_weight_input(original_inputs[1])
            if (
                weight_info is not None
                and linear_qinfo is not None
            ):
                if repq_gemmini_kernel_mode == "approx":
                    try:
                        a_sym_scale = sym_scale_from_uniform_params(linear_qinfo["scale"], linear_qinfo["zero_point"])
                        b_sym_scale = sym_scale_from_uniform_params(weight_info["scale"], weight_info["zero_point"])
                        b_const = quantize_int8(init_map[weight_info["source"]].astype(np.float32), b_sym_scale)
                    except KeyError:
                        a_sym_scale = None
                        b_sym_scale = None
                        b_const = None
                    if a_sym_scale is not None and b_sym_scale is not None and b_const is not None:
                        append_approx_gemmini_matmul(
                            a_source=linear_qinfo["source"],
                            a_sym_scale=a_sym_scale,
                            b_source=None,
                            b_sym_scale=b_sym_scale,
                            out_names=list(node.output),
                            prefix=prefix,
                            raw_name=node.name or node.output[0],
                            b_const_int8=b_const,
                        )
                        lowered_counts["repq_uniform_matmul"] += 1
                        continue
                profile_attrs = make_profile_attrs(node.name or node.output[0])
                rewritten.append(
                    helper.make_node(
                        "RepQUniformMatMul",
                        [
                            linear_qinfo["source"],
                            linear_qinfo["scale"],
                            linear_qinfo["zero_point"],
                            weight_info["source"],
                            weight_info["scale"],
                            weight_info["zero_point"],
                        ],
                        list(node.output),
                        domain="ivit",
                        name=unique_name(f"{prefix}_RepQUniformMatMul"),
                        n_bits=int(n_bits),
                        approximate=approximate_attr,
                        **profile_attrs,
                    )
                )
                lowered_counts["repq_uniform_matmul"] += 1
                continue

            if module_key.endswith("matmul1"):
                a_info = parse_uniform_dequant(original_inputs[0])
                b_info = parse_uniform_dequant(original_inputs[1])
                if a_info is not None and b_info is not None:
                    if repq_gemmini_kernel_mode == "approx":
                        try:
                            a_sym_scale = sym_scale_from_uniform_params(a_info["scale"], a_info["zero_point"])
                            b_sym_scale = sym_scale_from_uniform_params(b_info["scale"], b_info["zero_point"])
                        except KeyError:
                            a_sym_scale = None
                            b_sym_scale = None
                        if a_sym_scale is not None and b_sym_scale is not None:
                            append_approx_gemmini_matmul(
                                a_source=a_info["source"],
                                a_sym_scale=a_sym_scale,
                                b_source=b_info["source"],
                                b_sym_scale=b_sym_scale,
                                out_names=list(node.output),
                                prefix=prefix,
                                raw_name=node.name or node.output[0],
                            )
                            lowered_counts["repq_uniform_matmul"] += 1
                            continue
                    profile_attrs = make_profile_attrs(node.name or node.output[0])
                    rewritten.append(
                        helper.make_node(
                            "RepQUniformMatMul",
                            [
                                a_info["source"],
                                a_info["scale"],
                                a_info["zero_point"],
                                b_info["source"],
                                b_info["scale"],
                                b_info["zero_point"],
                            ],
                            list(node.output),
                            domain="ivit",
                            name=unique_name(f"{prefix}_RepQUniformMatMul"),
                            n_bits=int(n_bits),
                            approximate=approximate_attr,
                            **profile_attrs,
                        )
                    )
                    lowered_counts["repq_uniform_matmul"] += 1
                    continue

        if node.op_type == "Conv":
            module_key = node_to_module_key(node.name, node.output[0])
            prefix = sanitize(node.name or node.output[0])
            x_info = parse_uniform_dequant(original_inputs[0])
            w_info = parse_weight_input(original_inputs[1])
            bias_name = original_inputs[2] if len(original_inputs) > 2 else None
            if (
                x_info is not None
                and w_info is not None
                and bias_name is not None
                and module_key in lowering_stats
            ):
                stats = lowering_stats[module_key]
                x_scale = float(stats["a_scale"])
                w_scale = float(np.asarray(w_info["scale"], dtype=np.float32).reshape(-1)[0])
                y_scale = float(stats["y_scale"])
                bias_float = init_map[bias_name].astype(np.float32)
                bias_scale = max(x_scale * w_scale, 1e-8)
                bias_int32 = np.rint(bias_float / bias_scale).astype(np.int32)

                x_scale_name = add_initializer(f"{prefix}_x_scale", scalar_f32(x_scale))
                x_zp_name = add_initializer(f"{prefix}_x_zp", scalar_i8(0))
                x_quant_name = unique_name(f"{prefix}_x_quant")
                rewritten.append(
                    helper.make_node(
                        "QuantizeLinear",
                        [x_info["source"], x_scale_name, x_zp_name],
                        [x_quant_name],
                        name=unique_name(f"{prefix}_QuantizeLinear"),
                    )
                )

                w_name = add_initializer(f"{prefix}_w_data", w_info["data"])
                w_scale_name = add_initializer(f"{prefix}_w_scale", scalar_f32(w_scale))
                w_zp_name = add_initializer(f"{prefix}_w_zp", scalar_i8(0))
                y_scale_name = add_initializer(f"{prefix}_y_scale", scalar_f32(y_scale))
                y_zp_name = add_initializer(f"{prefix}_y_zp", scalar_i8(0))
                bias_q_name = add_initializer(f"{prefix}_bias", bias_int32)

                conv_output = unique_name(f"{prefix}_qconv")
                conv_kwargs = {
                    attr.name: helper.get_attribute_value(attr)
                    for attr in node.attribute
                }
                rewritten.append(
                    helper.make_node(
                        "QLinearConv",
                        [
                            x_quant_name,
                            x_scale_name,
                            x_zp_name,
                            w_name,
                            w_scale_name,
                            w_zp_name,
                            y_scale_name,
                            y_zp_name,
                            bias_q_name,
                        ],
                        [conv_output],
                        name=unique_name(f"{prefix}_QLinearConv"),
                        **conv_kwargs,
                    )
                )
                dq_output = unique_name(f"{prefix}_dq")
                rewritten.append(
                    helper.make_node(
                        "DequantizeLinear",
                        [conv_output, y_scale_name, y_zp_name],
                        [dq_output],
                        name=unique_name(f"{prefix}_DequantizeLinear"),
                    )
                )
                replace_output[node.output[0]] = dq_output
                lowered_counts["qlinear_conv"] += 1
                continue

        if node.op_type == "Gemm":
            module_key = node_to_module_key(node.name, node.output[0])
            prefix = sanitize(node.name or node.output[0])
            attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
            if attrs.get("transB", 0) == 1:
                a_info = parse_uniform_dequant(original_inputs[0])
                w_info = parse_exact_weight_input(original_inputs[1], force_transpose=True)
                bias_name = original_inputs[2] if len(original_inputs) > 2 else None
                if a_info is not None and w_info is not None and bias_name is not None:
                    if repq_gemmini_kernel_mode == "approx":
                        try:
                            a_sym_scale = sym_scale_from_uniform_params(a_info["scale"], a_info["zero_point"])
                            b_sym_scale = sym_scale_from_uniform_params(w_info["scale"], w_info["zero_point"])
                            b_const = quantize_int8(init_map[w_info["source"]].astype(np.float32), b_sym_scale)
                        except KeyError:
                            a_sym_scale = None
                            b_sym_scale = None
                            b_const = None
                        if a_sym_scale is not None and b_sym_scale is not None and b_const is not None:
                            qmm_output = unique_name(f"{prefix}_approx_uniform")
                            append_approx_gemmini_matmul(
                                a_source=a_info["source"],
                                a_sym_scale=a_sym_scale,
                                b_source=None,
                                b_sym_scale=b_sym_scale,
                                out_names=[qmm_output],
                                prefix=prefix,
                                raw_name=node.name or node.output[0],
                                b_const_int8=b_const,
                            )
                            add_output = unique_name(f"{prefix}_bias_add")
                            rewritten.append(
                                helper.make_node(
                                    "Add",
                                    [qmm_output, bias_name],
                                    [add_output],
                                    name=unique_name(f"{prefix}_AddBias"),
                                )
                            )
                            replace_output[node.output[0]] = add_output
                            lowered_counts["repq_uniform_matmul"] += 1
                            continue
                    profile_attrs = make_profile_attrs(node.name or node.output[0])
                    qmm_output = unique_name(f"{prefix}_uniform")
                    rewritten.append(
                        helper.make_node(
                            "RepQUniformMatMul",
                            [
                                a_info["source"],
                                a_info["scale"],
                                a_info["zero_point"],
                                w_info["source"],
                                w_info["scale"],
                                w_info["zero_point"],
                            ],
                            [qmm_output],
                            domain="ivit",
                            name=unique_name(f"{prefix}_RepQUniformMatMul"),
                            n_bits=int(n_bits),
                            approximate=approximate_attr,
                            **profile_attrs,
                        )
                    )
                    add_output = unique_name(f"{prefix}_bias_add")
                    rewritten.append(
                        helper.make_node(
                            "Add",
                            [qmm_output, bias_name],
                            [add_output],
                            name=unique_name(f"{prefix}_AddBias"),
                        )
                    )
                    replace_output[node.output[0]] = add_output
                    lowered_counts["repq_uniform_matmul"] += 1
                    continue

        node_copy = copy.deepcopy(node)
        del node_copy.input[:]
        node_copy.input.extend(rewritten_inputs)
        rewritten.append(node_copy)

    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    for graph_output in model.graph.output:
        if graph_output.name in replace_output:
            graph_output.name = replace_output[graph_output.name]
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    return lowered_counts


def export_onnx(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    output_path: Path,
    opset: int,
    dynamic_batch: bool,
) -> None:
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            example_input,
            str(output_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            export_params=True,
            do_constant_folding=False,
            opset_version=opset,
            training=torch.onnx.TrainingMode.EVAL,
            dynamo=False,
        )

    # Keep the exported model compatible with the older ORT-RISC-V build used by
    # the Gemmini runner in this repository.
    import onnx

    onnx_model = onnx.load(str(output_path))
    onnx_model.ir_version = 7
    onnx.save(onnx_model, str(output_path))


def print_onnx_summary(output_path: Path) -> None:
    import onnx

    model = onnx.load(str(output_path))
    counts = Counter(node.op_type for node in model.graph.node)
    print(f"ONNX saved to: {output_path}")
    print(f"Total ONNX nodes: {len(model.graph.node)}")
    for op in EXPORT_OP_SUMMARY:
        print(f"  {op:16s}: {counts.get(op, 0)}")


def verify_with_ort(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    output_path: Path,
    custom_op_library: Path | None,
) -> None:
    import onnxruntime as ort

    with torch.no_grad():
        torch_logits = model(example_input).detach().cpu().numpy()

    sess_options = ort.SessionOptions()
    if custom_op_library is not None and custom_op_library.is_file():
        sess_options.register_custom_ops_library(str(custom_op_library))

    session = ort.InferenceSession(
        str(output_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    ort_logits = session.run(
        None, {"input": example_input.detach().cpu().numpy()}
    )[0]

    max_abs = float(np.max(np.abs(torch_logits - ort_logits)))
    mean_abs = float(np.mean(np.abs(torch_logits - ort_logits)))
    print("Host ORT verification:")
    print(f"  max_abs_diff : {max_abs:.6e}")
    print(f"  mean_abs_diff: {mean_abs:.6e}")


def maybe_check_onnx(output_path: Path) -> None:
    import onnx

    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)


def maybe_save_state(model: torch.nn.Module, output_path: Path, args: argparse.Namespace) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_key": args.model,
            "arch": MODEL_ZOO[args.model],
            "w_bits": args.w_bits,
            "a_bits": args.a_bits,
            "post_ln_quant": "layerwise",
            "post_softmax_quant": "log2",
        },
        output_path,
    )
    print(f"Saved reparameterized state_dict to: {output_path}")


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    if (
        not args.lower_qlinear_matmul
        and "lowered" in args.output.name
    ):
        print(
            "[WARN] Output filename suggests a lowered graph, but "
            "--lower-qlinear-matmul is disabled. "
            "The export will remain a semantic ONNX graph."
        )

    calib_data, export_data = build_example_inputs(args, device)
    q_model = prepare_quantized_model(args, device, calib_data)

    export_model = q_model.cpu()
    calib_input = calib_data.cpu()
    export_input = export_data.cpu()

    maybe_save_state(export_model, args.save_reparam_state, args)

    lowering_stats = None
    if args.lower_qlinear_matmul:
        print("Collecting output statistics for QLinearMatMul lowering...")
        lowering_stats = collect_qlinear_lowering_stats(
            export_model,
            calib_input,
            batch_size=args.calib_batchsize,
        )

    print("Exporting semantic ONNX graph...")
    export_onnx(
        export_model,
        export_input,
        args.output,
        opset=args.opset,
        dynamic_batch=args.dynamic_batch,
    )

    if args.lower_qlinear_matmul:
        lowered_counts = lower_qlinear_matmul_nodes(
            args.output,
            lowering_stats or {},
            n_bits=args.a_bits,
            repq_gemmini_kernel_mode=args.repq_gemmini_kernel_mode,
        )
        print(
            "Lowered Gemmini-friendly ops: "
            f"conv={lowered_counts['qlinear_conv']} "
            f"repq_uniform_matmul={lowered_counts['repq_uniform_matmul']} "
            f"repq_log_matmul={lowered_counts['repq_log_matmul']} "
            f"mode={args.repq_gemmini_kernel_mode}"
        )

    if not args.skip_onnx_checker:
        maybe_check_onnx(args.output)
        print("ONNX checker: OK")

    print_onnx_summary(args.output)

    if args.verify_ort:
        verify_with_ort(
            export_model,
            export_input,
            args.output,
            args.host_custom_op_library,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
