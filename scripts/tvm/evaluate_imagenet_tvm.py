#!/usr/bin/env python3
"""Evaluate an I-ViT TVM INT8 model on an ImageNet-style validation folder."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent

for path in (REPO_ROOT, SCRIPTS_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

TVM_HOME = Path(os.environ.get("TVM_HOME", REPO_ROOT / "tvm-gemmini"))
TVM_PYTHON = TVM_HOME / "python"
if TVM_PYTHON.exists() and str(TVM_PYTHON) not in sys.path:
    sys.path.insert(0, str(TVM_PYTHON))

import numpy as np
import torch
import tvm
from PIL import Image
from tqdm import tqdm
from tvm import relay
from tvm.contrib import graph_executor

from models.ivit import builder as build_model
from models.ivit.layers import QuantizeContext
import pytorch_to_tvm_params


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"}
DEFAULT_CLASS_TO_LABEL = (
    TVM_HOME
    / "3rdparty"
    / "gemmini"
    / "software"
    / "onnxruntime-riscv"
    / "systolic_runner"
    / "imagenet_runner"
    / "tools"
    / "class_to_label.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate I-ViT TVM INT8 accuracy on ImageNet validation"
    )
    parser.add_argument("--checkpoint", required=True, help="PyTorch QAT checkpoint path")
    parser.add_argument(
        "--data-root",
        required=True,
        help="ImageNet validation root formatted as <root>/<wnid>/*.JPEG",
    )
    parser.add_argument(
        "--class-to-label",
        default=str(DEFAULT_CLASS_TO_LABEL),
        help="Path to ImageNet WNID-per-line class order file",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        choices=["deit_tiny_patch16_224", "swin_tiny_patch4_window7_224"],
        help="Model name. Omit to auto-detect from checkpoint keys.",
    )
    parser.add_argument(
        "--target",
        default="cuda",
        help="TVM target string. Examples: cuda, llvm, 'cuda -arch=sm_86'",
    )
    parser.add_argument(
        "--target-host",
        default="llvm",
        help="TVM host target for heterogeneous builds",
    )
    parser.add_argument("--device-id", type=int, default=0, help="Target device index")
    parser.add_argument(
        "--opt-level",
        type=int,
        default=3,
        help="TVM Relay build optimization level",
    )
    parser.add_argument(
        "--qnn-canonicalize",
        action="store_true",
        help="Run relay.qnn.transform.CanonicalizeOps() before build",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit number of validation samples",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index into the ImageFolder dataset",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Exclusive end index into the ImageFolder dataset",
    )
    parser.add_argument(
        "--save-params-dir",
        default=None,
        help="Optional directory to save generated params.npy",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Directory to save or reuse compiled TVM graph artifacts",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save evaluation results as JSON",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=500,
        help="Print running accuracy every N processed samples",
    )
    return parser.parse_args()


def scalar(value: object) -> float:
    arr = np.array(value)
    return float(arr.reshape(-1)[0])


def load_checkpoint(checkpoint_path: str):
    return torch.load(checkpoint_path, map_location=torch.device("cpu"))


def resolve_model_name(checkpoint, model_name: str | None) -> str:
    return pytorch_to_tvm_params.resolve_model_name(checkpoint, model_name=model_name)


def maybe_save_params(pretrained_params: dict[str, np.ndarray], save_dir: str | None) -> None:
    if not save_dir:
        return
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "params.npy", pretrained_params)


def load_wnid_mapping(class_to_label_path: str) -> dict[str, int]:
    with open(class_to_label_path, "r") as f:
        wnids = [line.strip() for line in f if line.strip()]
    return {wnid: idx for idx, wnid in enumerate(wnids)}


def build_eval_samples(data_root: str, class_to_label_path: str) -> list[tuple[Path, int]]:
    root = Path(data_root)
    wnid_to_idx = load_wnid_mapping(class_to_label_path)
    samples: list[tuple[Path, int]] = []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        label = wnid_to_idx.get(class_dir.name)
        if label is None:
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix in VALID_IMAGE_EXTS:
                samples.append((image_path, label))
    if not samples:
        raise RuntimeError(f"no evaluation samples found under {root}")
    return samples


def artifact_paths(artifact_dir: str):
    base = Path(artifact_dir)
    return {
        "dir": base,
        "graph": base / "graph.json",
        "lib": base / "mod.so",
        "params": base / "params.bin",
        "meta": base / "meta.json",
    }


def save_artifacts(lib, artifact_dir: str, meta: dict[str, object]) -> None:
    paths = artifact_paths(artifact_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    paths["graph"].write_text(lib.get_graph_json())
    paths["lib"].unlink(missing_ok=True)
    lib.get_lib().export_library(str(paths["lib"]))
    paths["params"].write_bytes(relay.save_param_dict(lib.get_params()))
    paths["meta"].write_text(json.dumps(meta, indent=2) + "\n")


def load_artifacts(artifact_dir: str, device):
    paths = artifact_paths(artifact_dir)
    loaded_lib = tvm.runtime.load_module(str(paths["lib"]))
    graph_json = paths["graph"].read_text()
    params_bytes = paths["params"].read_bytes()
    runtime = graph_executor.create(graph_json, loaded_lib, device)
    runtime.load_params(params_bytes)

    meta = {}
    if paths["meta"].exists():
        meta = json.loads(paths["meta"].read_text())
    return runtime, meta


def build_runtime(
    checkpoint,
    model_name: str,
    target: str,
    target_host: str,
    device_id: int,
    opt_level: int,
    qnn_canonicalize: bool,
    artifact_dir: str | None,
):
    pytorch_to_tvm_params.load_qconfig(checkpoint, model_name=model_name)
    pretrained_params = pytorch_to_tvm_params.build_param_dict(
        checkpoint, model_name=model_name
    )

    tvm_target = tvm.target.Target(target, host=target_host)
    dev = tvm.device(tvm_target.kind.name, device_id)
    if not dev.exist:
        raise RuntimeError(f"TVM device does not exist: {tvm_target.kind.name}({device_id})")

    # Match the Swin host-eval behavior used in the Spike flow: when opt_level=0,
    # canonicalize qnn ops up front so host LLVM lowering does not choke on qnn.conv2d.
    canonicalize_qnn = qnn_canonicalize or (
        model_name.startswith("swin_") and tvm_target.kind.name != "c" and opt_level == 0
    )

    if artifact_dir:
        paths = artifact_paths(artifact_dir)
        if paths["graph"].exists() and paths["lib"].exists() and paths["params"].exists():
            runtime, meta = load_artifacts(artifact_dir, dev)
            input_scale = float(meta.get("input_scale", scalar(QuantizeContext.qconfig_dict["qconfig_embed_conv"].input_scale)))
            return runtime, dev, input_scale, 0.0, pretrained_params

    module_or_func, _ = build_model.get_workload(
        name=model_name,
        batch_size=1,
        image_shape=(3, 224, 224),
        dtype="int8",
        data_layout="NCHW",
        kernel_layout="OIHW",
    )

    ir_mod = (
        module_or_func
        if isinstance(module_or_func, tvm.IRModule)
        else tvm.IRModule.from_expr(module_or_func)
    )
    if canonicalize_qnn:
        ir_mod = relay.transform.InferType()(ir_mod)
        ir_mod = relay.qnn.transform.CanonicalizeOps()(ir_mod)
        ir_mod = relay.transform.InferType()(ir_mod)

    build_start = time.time()
    with tvm.transform.PassContext(opt_level=opt_level):
        lib = relay.build(ir_mod, target=tvm_target, params=pretrained_params)
    build_seconds = time.time() - build_start

    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    input_scale = scalar(QuantizeContext.qconfig_dict["qconfig_embed_conv"].input_scale)

    if artifact_dir:
        save_artifacts(
            lib,
            artifact_dir,
            meta={
                "model_name": model_name,
                "target": str(tvm_target),
                "device_id": device_id,
                "input_scale": input_scale,
                "opt_level": opt_level,
                "qnn_canonicalize": canonicalize_qnn,
            },
        )

    return runtime, dev, input_scale, build_seconds, pretrained_params


def preprocess_image(image: Image.Image, input_scale: float) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")

    if image.width < image.height:
        new_width = 256
        new_height = int(round(image.height * 256 / image.width))
    else:
        new_height = 256
        new_width = int(round(image.width * 256 / image.height))

    image = image.resize((new_width, new_height), resample=Image.BICUBIC)
    left = (image.width - 224) // 2
    top = (image.height - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))

    img = np.asarray(image, dtype=np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    quantized = np.clip(np.round(img / input_scale), -128, 127).astype("int8")
    return quantized


def evaluate(
    runtime,
    dev,
    samples: Sequence[tuple[Path, int]],
    input_scale: float,
    start_index: int,
    end_index: int | None,
    num_samples: int | None,
    print_every: int,
):
    if start_index < 0:
        raise ValueError("start_index must be non-negative")

    dataset_size = len(samples)
    stop_index = dataset_size if end_index is None else min(end_index, dataset_size)
    if stop_index < start_index:
        raise ValueError("end_index must be greater than or equal to start_index")

    indices = list(range(start_index, stop_index))
    if num_samples is not None:
        indices = indices[:num_samples]
    total_samples = len(indices)
    if total_samples == 0:
        raise ValueError("no samples selected for evaluation")

    top1_correct = 0
    top5_correct = 0
    inference_seconds = 0.0

    for seen, dataset_idx in enumerate(
        tqdm(indices, total=total_samples, desc="ImageNet val"), start=1
    ):
        image_path, label = samples[dataset_idx]
        with Image.open(image_path) as image:
            input_data = preprocess_image(image.convert("RGB"), input_scale)

        runtime.set_input("data", tvm.nd.array(input_data, dev))
        start = time.time()
        runtime.run()
        inference_seconds += time.time() - start
        output = runtime.get_output(0).numpy()[0]

        top5 = np.argsort(output)[-5:][::-1]
        if int(top5[0]) == int(label):
            top1_correct += 1
        if int(label) in top5:
            top5_correct += 1

        if print_every > 0 and seen % print_every == 0:
            print(
                f"[{seen}/{total_samples}] "
                f"top1={100.0 * top1_correct / seen:.2f}% "
                f"top5={100.0 * top5_correct / seen:.2f}%"
            )

    return {
        "evaluated_samples": total_samples,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "top1_accuracy": 100.0 * top1_correct / total_samples,
        "top5_accuracy": 100.0 * top5_correct / total_samples,
        "inference_seconds": inference_seconds,
        "avg_latency_ms": 1000.0 * inference_seconds / total_samples,
        "throughput_fps": total_samples / inference_seconds if inference_seconds > 0 else 0.0,
    }


def main() -> int:
    args = parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    model_name = resolve_model_name(checkpoint, args.model_name)
    samples = build_eval_samples(args.data_root, args.class_to_label)

    runtime, dev, input_scale, build_seconds, pretrained_params = build_runtime(
        checkpoint=checkpoint,
        model_name=model_name,
        target=args.target,
        target_host=args.target_host,
        device_id=args.device_id,
        opt_level=args.opt_level,
        qnn_canonicalize=args.qnn_canonicalize,
        artifact_dir=args.artifact_dir,
    )
    maybe_save_params(pretrained_params, args.save_params_dir)

    results = evaluate(
        runtime=runtime,
        dev=dev,
        samples=samples,
        input_scale=input_scale,
        start_index=args.start_index,
        end_index=args.end_index,
        num_samples=args.num_samples,
        print_every=args.print_every,
    )

    summary = {
        "checkpoint": args.checkpoint,
        "data_root": args.data_root,
        "class_to_label": args.class_to_label,
        "model_name": model_name,
        "target": args.target,
        "target_host": args.target_host,
        "device_id": args.device_id,
        "opt_level": args.opt_level,
        "qnn_canonicalize": args.qnn_canonicalize,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "num_dataset_samples": len(samples),
        "build_seconds": build_seconds,
        "input_scale": input_scale,
        **results,
    }

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
