#!/usr/bin/env python3
"""Build calibrated RepQ models and extract TVM Relay params/metadata."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent
REPQ_CLASSIFICATION_ROOT = Path(
    os.environ.get(
        "REPQ_CLASSIFICATION_ROOT",
        str(REPO_ROOT / "RepQ-ViT" / "classification"),
    )
)

if str(REPQ_CLASSIFICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(REPQ_CLASSIFICATION_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from quant import quant_model, set_quant_state  # noqa: E402
from quant.quant_modules import QuantConv2d, QuantLinear, QuantMatMul  # noqa: E402
from utils import build_model, build_transform  # noqa: E402
from models.repq.layers import RepQModuleMeta  # noqa: E402


MODEL_ZOO = {
    "deit_tiny": "deit_tiny_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deit_tiny", choices=sorted(MODEL_ZOO))
    parser.add_argument("--dataset", default=None, help="ImageFolder root for calibration/export")
    parser.add_argument("--calib-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--allow-random-calibration", action="store_true")
    parser.add_argument("--calib-batchsize", default=32, type=int)
    parser.add_argument("--calib-num-samples", default=32, type=int)
    parser.add_argument("--w-bits", default=8, type=int)
    parser.add_argument("--a-bits", default=8, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-reparameterization", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def get_model_preprocess(model_key: str) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    if model_key == "deit_tiny":
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 0.875
    if model_key == "swin_tiny":
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 0.9
    raise RuntimeError(f"Unsupported model key: {model_key}")


def resolve_imagefolder_root(root: Path, preferred_split: str | None) -> Path:
    if preferred_split is not None and (root / preferred_split).is_dir():
        return root / preferred_split
    return root


def load_dataset_samples(root: Path, transform, num_samples: int, seed: int) -> torch.Tensor:
    from torchvision import datasets

    dataset = datasets.ImageFolder(str(root), transform)
    if len(dataset) == 0:
        raise RuntimeError(f"No images found under: {root}")
    take = min(num_samples, len(dataset))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=take, replace=False).tolist()
    batch = [dataset[index][0] for index in indices]
    return torch.stack(batch, dim=0)


def build_calibration_batch(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    data_root = Path(args.dataset) if args.dataset else None
    calib_root = args.calib_dir or data_root
    if calib_root is not None:
        mean, std, crop_pct = get_model_preprocess(args.model)
        transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)
        calib_dir = resolve_imagefolder_root(calib_root, "train")
        batch = load_dataset_samples(calib_dir, transform, args.calib_num_samples, args.seed)
        return batch.to(device)

    if not args.allow_random_calibration:
        raise RuntimeError("Calibration data is required unless --allow-random-calibration is set")
    return torch.randn((args.calib_batchsize, 3, 224, 224), device=device)


def build_fp_model(args: argparse.Namespace) -> torch.nn.Module:
    model_name = MODEL_ZOO[args.model]
    use_pretrained = not args.allow_random_init and args.checkpoint is None
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
        father_module = module_dict[father_name]

        if not isinstance(module, nn.LayerNorm):
            continue
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


def collapse_reparameterized_input_quantizers(q_model: torch.nn.Module, model_key: str) -> None:
    module_dict: dict[str, torch.nn.Module] = {}
    q_model_slice = q_model.layers if "swin" in model_key else q_model.blocks
    for name, module in q_model_slice.named_modules():
        module_dict[name] = module
        idx = name.rfind(".")
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        father_module = module_dict[father_name]

        if not isinstance(module, nn.LayerNorm):
            continue
        if name.endswith("norm1"):
            next_module = father_module.attn.qkv
        elif name.endswith("norm2"):
            next_module = father_module.mlp.fc1
        elif name.endswith("norm"):
            next_module = father_module.reduction
        else:
            continue

        next_module.input_quantizer.channel_wise = False
        next_module.input_quantizer.delta = torch.mean(next_module.input_quantizer.delta.reshape(-1))
        next_module.input_quantizer.zero_point = torch.mean(
            next_module.input_quantizer.zero_point.reshape(-1)
        )


def prepare_quantized_model(args: argparse.Namespace) -> torch.nn.Module:
    device = resolve_device(args.device)
    calib_batch = build_calibration_batch(args, device)
    model = build_fp_model(args).to(device).eval()

    q_model = quant_model(
        model,
        input_quant_params={"n_bits": args.a_bits, "channel_wise": False},
        weight_quant_params={"n_bits": args.w_bits, "channel_wise": True},
    )
    q_model.to(device).eval()

    def run_pass() -> None:
        with torch.no_grad():
            for start in range(0, calib_batch.shape[0], args.calib_batchsize):
                end = min(start + args.calib_batchsize, calib_batch.shape[0])
                _ = q_model(calib_batch[start:end])

    set_quant_state(q_model, input_quant=True, weight_quant=True)
    run_pass()
    if not args.skip_reparameterization:
        apply_scale_reparameterization(q_model, args.model)
    else:
        collapse_reparameterized_input_quantizers(q_model, args.model)
    set_quant_state(q_model, input_quant=True, weight_quant=True)
    run_pass()
    return q_model.cpu().eval()


def _tensor_to_numpy(tensor: torch.Tensor, dtype: str | None = None) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if dtype is not None:
        array = array.astype(dtype)
    return array


def _scalar_or_vector(tensor: torch.Tensor) -> np.ndarray | float:
    array = _tensor_to_numpy(tensor, "float32").reshape(-1)
    if array.size == 1:
        return float(array[0])
    return array


def _collapse_near_uniform(value: np.ndarray | float, *, atol: float) -> np.ndarray | float:
    array = np.asarray(value, dtype="float32").reshape(-1)
    if array.size == 1:
        return float(array[0])
    if float(np.max(array) - np.min(array)) <= atol:
        return float(np.mean(array))
    return array


def _module_meta(module: torch.nn.Module) -> RepQModuleMeta:
    if isinstance(module, (QuantConv2d, QuantLinear)):
        return RepQModuleMeta(
            n_bits=int(module.input_quantizer.n_bits),
            input_scale=_collapse_near_uniform(
                _scalar_or_vector(module.input_quantizer.delta),
                atol=1e-4,
            ),
            input_zero_point=_collapse_near_uniform(
                _scalar_or_vector(module.input_quantizer.zero_point),
                atol=1.0,
            ),
            weight_scale=_scalar_or_vector(module.weight_quantizer.delta),
            weight_zero_point=_scalar_or_vector(module.weight_quantizer.zero_point),
        )
    if isinstance(module, QuantMatMul):
        meta = RepQModuleMeta(
            n_bits=int(module.quantizer_B.n_bits),
            b_scale=_scalar_or_vector(module.quantizer_B.delta),
            b_zero_point=_scalar_or_vector(module.quantizer_B.zero_point),
        )
        if hasattr(module.quantizer_A, "zero_point") and module.quantizer_A.zero_point is not None:
            meta.input_scale = _scalar_or_vector(module.quantizer_A.delta)
            meta.input_zero_point = _scalar_or_vector(module.quantizer_A.zero_point)
        else:
            meta.log_scale = _scalar_or_vector(module.quantizer_A.delta)
        return meta
    raise TypeError(type(module))


def _extract_deit_params_and_meta(q_model: torch.nn.Module) -> tuple[dict[str, np.ndarray], dict[str, RepQModuleMeta]]:
    params: dict[str, np.ndarray] = {}
    meta: dict[str, RepQModuleMeta] = {}

    params["cls_token"] = _tensor_to_numpy(q_model.cls_token, "float32")
    params["pos_embed"] = _tensor_to_numpy(q_model.pos_embed, "float32")
    params["norm_weight"] = _tensor_to_numpy(q_model.norm.weight, "float32")
    params["norm_bias"] = _tensor_to_numpy(q_model.norm.bias, "float32")

    for name, module in q_model.named_modules():
        if isinstance(module, (QuantConv2d, QuantLinear, QuantMatMul)):
            meta[name] = _module_meta(module)

        prefix = name.replace(".", "_")
        if isinstance(module, QuantConv2d):
            params[prefix + "_weight"] = _tensor_to_numpy(module.weight, "float32")
            params[prefix + "_bias"] = _tensor_to_numpy(module.bias, "float32")
        elif isinstance(module, QuantLinear):
            params[prefix + "_weight"] = _tensor_to_numpy(module.weight, "float32")
            if module.bias is not None:
                params[prefix + "_bias"] = _tensor_to_numpy(module.bias, "float32")
        elif isinstance(module, nn.LayerNorm):
            params[prefix + "_weight"] = _tensor_to_numpy(module.weight, "float32")
            params[prefix + "_bias"] = _tensor_to_numpy(module.bias, "float32")

    return params, meta


def _extract_swin_params_and_meta(q_model: torch.nn.Module) -> tuple[dict[str, np.ndarray], dict[str, RepQModuleMeta]]:
    params: dict[str, np.ndarray] = {}
    meta: dict[str, RepQModuleMeta] = {}

    params["patch_embed_norm_weight"] = _tensor_to_numpy(q_model.patch_embed.norm.weight, "float32")
    params["patch_embed_norm_bias"] = _tensor_to_numpy(q_model.patch_embed.norm.bias, "float32")
    params["norm_weight"] = _tensor_to_numpy(q_model.norm.weight, "float32")
    params["norm_bias"] = _tensor_to_numpy(q_model.norm.bias, "float32")
    params["head_fc_weight"] = _tensor_to_numpy(q_model.head.fc.weight, "float32")
    params["head_fc_bias"] = _tensor_to_numpy(q_model.head.fc.bias, "float32")

    for name, module in q_model.named_modules():
        if isinstance(module, (QuantConv2d, QuantLinear, QuantMatMul)):
            meta[name] = _module_meta(module)

        prefix = name.replace(".", "_")
        if isinstance(module, QuantConv2d):
            params[prefix + "_weight"] = _tensor_to_numpy(module.weight, "float32")
            params[prefix + "_bias"] = _tensor_to_numpy(module.bias, "float32")
        elif isinstance(module, QuantLinear):
            params[prefix + "_weight"] = _tensor_to_numpy(module.weight, "float32")
            if module.bias is not None:
                params[prefix + "_bias"] = _tensor_to_numpy(module.bias, "float32")
        elif isinstance(module, nn.LayerNorm):
            params[prefix + "_weight"] = _tensor_to_numpy(module.weight, "float32")
            params[prefix + "_bias"] = _tensor_to_numpy(module.bias, "float32")
        elif name.endswith("attn"):
            table = getattr(module, "relative_position_bias_table", None)
            if table is not None:
                params[prefix + "_relative_position_bias_table"] = _tensor_to_numpy(table, "float32")

    return params, meta


def build_repq_tvm_artifacts(args: argparse.Namespace) -> tuple[str, dict[str, np.ndarray], dict[str, RepQModuleMeta]]:
    seed_everything(args.seed)
    q_model = prepare_quantized_model(args)
    model_name = MODEL_ZOO[args.model]
    if model_name == "deit_tiny_patch16_224":
        params, meta = _extract_deit_params_and_meta(q_model)
    elif model_name == "swin_tiny_patch4_window7_224":
        params, meta = _extract_swin_params_and_meta(q_model)
    else:
        raise RuntimeError(f"Unsupported model: {model_name}")
    return model_name, params, meta


def _meta_to_jsonable(meta: dict[str, RepQModuleMeta]) -> dict[str, dict[str, object]]:
    def _convert(value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value.tolist()
        if np.isscalar(value):
            return float(value)
        return np.asarray(value).tolist()

    return {
        name: {
            "n_bits": item.n_bits,
            "input_scale": _convert(item.input_scale),
            "input_zero_point": _convert(item.input_zero_point),
            "weight_scale": _convert(item.weight_scale),
            "weight_zero_point": _convert(item.weight_zero_point),
            "log_scale": _convert(item.log_scale),
            "b_scale": _convert(item.b_scale),
            "b_zero_point": _convert(item.b_zero_point),
        }
        for name, item in meta.items()
    }


def _jsonable_to_meta(meta_json: dict[str, dict[str, object]]) -> dict[str, RepQModuleMeta]:
    def _restore(value):
        if value is None:
            return None
        if isinstance(value, list):
            array = np.asarray(value, dtype="float32")
            if array.size == 1:
                return float(array.reshape(-1)[0])
            return array
        return float(value)

    restored: dict[str, RepQModuleMeta] = {}
    for name, item in meta_json.items():
        restored[name] = RepQModuleMeta(
            n_bits=int(item["n_bits"]),
            input_scale=_restore(item["input_scale"]),
            input_zero_point=_restore(item["input_zero_point"]),
            weight_scale=_restore(item["weight_scale"]),
            weight_zero_point=_restore(item["weight_zero_point"]),
            log_scale=_restore(item["log_scale"]),
            b_scale=_restore(item["b_scale"]),
            b_zero_point=_restore(item["b_zero_point"]),
        )
    return restored


def save_repq_tvm_artifacts(
    output_dir: Path,
    model_name: str,
    params: dict[str, np.ndarray],
    meta: dict[str, RepQModuleMeta],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "params.npz", **params)
    (output_dir / "meta.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "meta": _meta_to_jsonable(meta),
            },
            indent=2,
        )
        + "\n"
    )


def load_repq_tvm_artifacts(output_dir: Path) -> tuple[str, dict[str, np.ndarray], dict[str, RepQModuleMeta]]:
    meta_blob = json.loads((output_dir / "meta.json").read_text())
    params_npz = np.load(output_dir / "params.npz")
    params = {name: params_npz[name] for name in params_npz.files}
    meta = _jsonable_to_meta(meta_blob["meta"])
    return str(meta_blob["model_name"]), params, meta


def main() -> None:
    args = parse_args()
    model_name, params, meta = build_repq_tvm_artifacts(args)
    if args.output_dir is not None:
        save_repq_tvm_artifacts(args.output_dir, model_name, params, meta)
        print(f"Saved artifacts to {args.output_dir}")
    print(
        f"Prepared RepQ TVM artifacts for {model_name}: "
        f"{len(params)} params, {len(meta)} quantized modules"
    )
    print(f"Params: {len(params)} tensors")
    print(f"Quantized modules: {len(meta)}")


if __name__ == "__main__":
    main()
