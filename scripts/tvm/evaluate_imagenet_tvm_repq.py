#!/usr/bin/env python3
"""Evaluate a direct-Relay RepQ model on ImageNet-style validation data."""

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
import tvm
from PIL import Image
from tqdm import tqdm
from tvm import relay
from tvm.contrib import graph_executor

from models.repq import builder as build_repq_model
from models.repq.layers import RepQContext
from repq_to_tvm_params import (
    MODEL_ZOO,
    build_repq_tvm_artifacts,
    load_repq_tvm_artifacts,
    save_repq_tvm_artifacts,
)


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deit_tiny", choices=sorted(MODEL_ZOO))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--class-to-label", default=str(DEFAULT_CLASS_TO_LABEL))
    parser.add_argument("--dataset", default=None, help="ImageFolder root for RepQ calibration")
    parser.add_argument("--calib-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--repq-artifact-dir", default=None, help="Optional saved RepQ artifact dir")
    parser.add_argument("--save-repq-artifact-dir", default=None, help="Optional output dir for saved RepQ artifacts")
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--allow-random-calibration", action="store_true")
    parser.add_argument("--calib-batchsize", default=32, type=int)
    parser.add_argument("--calib-num-samples", default=32, type=int)
    parser.add_argument("--w-bits", default=8, type=int)
    parser.add_argument("--a-bits", default=8, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--skip-reparameterization", action="store_true")
    parser.add_argument("--target", default="llvm")
    parser.add_argument("--target-host", default="llvm")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--opt-level", type=int, default=3)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument(
        "--disable-gemmini-ops",
        action="store_true",
        help="Build the exact plain-Relay path instead of the approximate Gemmini path",
    )
    return parser.parse_args()


def model_name_from_key(model_key: str) -> str:
    return MODEL_ZOO[model_key]


def crop_pct_for_model(model_name: str) -> float:
    if model_name == "swin_tiny_patch4_window7_224":
        return 0.9
    return 0.875


def build_eval_samples(data_root: str, class_to_label_path: str) -> list[tuple[Path, int]]:
    root = Path(data_root)
    with open(class_to_label_path, "r") as f:
        wnids = [line.strip() for line in f if line.strip()]
    wnid_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}

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


def save_compiled_artifacts(lib, artifact_dir: str, meta: dict[str, object]) -> None:
    paths = artifact_paths(artifact_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    paths["graph"].write_text(lib.get_graph_json())
    paths["lib"].unlink(missing_ok=True)
    lib.get_lib().export_library(str(paths["lib"]))
    paths["params"].write_bytes(relay.save_param_dict(lib.get_params()))
    paths["meta"].write_text(json.dumps(meta, indent=2) + "\n")


def load_compiled_artifacts(artifact_dir: str, device):
    paths = artifact_paths(artifact_dir)
    loaded_lib = tvm.runtime.load_module(str(paths["lib"]))
    graph_json = paths["graph"].read_text()
    params_bytes = paths["params"].read_bytes()
    runtime = graph_executor.create(graph_json, loaded_lib, device)
    runtime.load_params(params_bytes)
    meta = json.loads(paths["meta"].read_text()) if paths["meta"].exists() else {}
    return runtime, meta


def load_or_build_repq_artifacts(args: argparse.Namespace):
    if args.repq_artifact_dir:
        artifact_dir = Path(args.repq_artifact_dir)
        if (artifact_dir / "params.npz").exists() and (artifact_dir / "meta.json").exists():
            return load_repq_tvm_artifacts(artifact_dir)

    repq_args = argparse.Namespace(
        model=args.model,
        dataset=args.dataset,
        calib_dir=Path(args.calib_dir) if args.calib_dir else None,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        allow_random_init=args.allow_random_init,
        allow_random_calibration=args.allow_random_calibration,
        calib_batchsize=args.calib_batchsize,
        calib_num_samples=args.calib_num_samples,
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        device=args.device,
        seed=args.seed,
        skip_reparameterization=args.skip_reparameterization,
        output_dir=None,
    )
    model_name, params, meta = build_repq_tvm_artifacts(repq_args)
    if args.save_repq_artifact_dir:
        save_repq_tvm_artifacts(Path(args.save_repq_artifact_dir), model_name, params, meta)
    return model_name, params, meta


def build_runtime(args: argparse.Namespace, model_name: str, params, meta):
    tvm_target = tvm.target.Target(args.target, host=args.target_host)
    if tvm_target.kind.name == "c" and "gemmini" in str(tvm_target):
        raise RuntimeError(
            "evaluate_imagenet_tvm_repq.py is for host graph_executor evaluation. "
            "Use a separate AOT/Spike flow for c-device=gemmini targets."
        )
    dev = tvm.device(tvm_target.kind.name, args.device_id)
    if not dev.exist:
        raise RuntimeError(f"TVM device does not exist: {tvm_target.kind.name}({args.device_id})")

    # Host graph_executor evaluation is used here to validate the plain Relay
    # implementation. Keep Gemmini approximation ops for AOT/device flows only;
    # otherwise CPU smoke tests can report incorrect predictions even when the
    # exact quantized RepQ model is fine.
    use_gemmini_ops = tvm_target.kind.name == "c" and "gemmini" in str(tvm_target)
    if args.disable_gemmini_ops:
        use_gemmini_ops = False

    if args.artifact_dir:
        paths = artifact_paths(args.artifact_dir)
        if paths["graph"].exists() and paths["lib"].exists() and paths["params"].exists():
            runtime, cached_meta = load_compiled_artifacts(args.artifact_dir, dev)
            cached_use_gemmini_ops = cached_meta.get("use_gemmini_ops")
            if cached_use_gemmini_ops is None:
                cached_use_gemmini_ops = not bool(cached_meta.get("disable_gemmini_ops", False))
            if (
                cached_meta.get("model_name") == model_name
                and cached_meta.get("target") == str(tvm_target)
                and int(cached_meta.get("device_id", args.device_id)) == args.device_id
                and int(cached_meta.get("opt_level", args.opt_level)) == args.opt_level
                and bool(cached_use_gemmini_ops) == use_gemmini_ops
            ):
                return runtime, dev, float(cached_meta.get("build_seconds", 0.0))

    RepQContext.set_use_gemmini_ops(use_gemmini_ops)
    mod, tvm_params = build_repq_model.get_workload(
        name=model_name,
        params=params,
        meta=meta,
        batch_size=1,
        image_shape=(3, 224, 224),
    )

    build_start = time.time()
    with tvm.transform.PassContext(opt_level=args.opt_level):
        lib = relay.build(mod, target=tvm_target, params=tvm_params)
    build_seconds = time.time() - build_start
    runtime = graph_executor.GraphModule(lib["default"](dev))

    if args.artifact_dir:
        save_compiled_artifacts(
            lib,
            args.artifact_dir,
            meta={
                "model_name": model_name,
                "target": str(tvm_target),
                "device_id": args.device_id,
                "build_seconds": build_seconds,
                "opt_level": args.opt_level,
                "disable_gemmini_ops": not use_gemmini_ops,
                "use_gemmini_ops": use_gemmini_ops,
            },
        )

    return runtime, dev, build_seconds


def preprocess_image(image: Image.Image, model_name: str) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")

    crop_pct = crop_pct_for_model(model_name)
    resize_size = int(round(224 / crop_pct))

    if image.width < image.height:
        new_width = resize_size
        new_height = int(round(image.height * resize_size / image.width))
    else:
        new_height = resize_size
        new_width = int(round(image.width * resize_size / image.height))

    image = image.resize((new_width, new_height), resample=Image.BICUBIC)
    left = (image.width - 224) // 2
    top = (image.height - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))

    img = np.asarray(image, dtype=np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)


def evaluate(
    runtime,
    dev,
    samples: Sequence[tuple[Path, int]],
    model_name: str,
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
            input_data = preprocess_image(image, model_name)

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
    model_name, params, meta = load_or_build_repq_artifacts(args)
    samples = build_eval_samples(args.data_root, args.class_to_label)
    runtime, dev, build_seconds = build_runtime(args, model_name, params, meta)

    results = evaluate(
        runtime=runtime,
        dev=dev,
        samples=samples,
        model_name=model_name,
        start_index=args.start_index,
        end_index=args.end_index,
        num_samples=args.num_samples,
        print_every=args.print_every,
    )

    summary = {
        "model_key": args.model,
        "model_name": model_name,
        "data_root": args.data_root,
        "class_to_label": args.class_to_label,
        "target": args.target,
        "target_host": args.target_host,
        "device_id": args.device_id,
        "opt_level": args.opt_level,
        "build_seconds": build_seconds,
        "num_dataset_samples": len(samples),
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
