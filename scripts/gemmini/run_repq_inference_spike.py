#!/usr/bin/env python3
"""Run direct-Relay RepQ inference on Spike via TVM Gemmini AOT."""

from __future__ import annotations

import argparse
import os
import re
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).parent.absolute()
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent
TVM_SCRIPTS_DIR = SCRIPTS_DIR / "tvm"
sys.path.insert(0, str(TVM_SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

TVM_HOME = Path(os.environ.get("TVM_HOME", REPO_ROOT.parent / "tvm-gemmini"))
TVM_PYTHON = TVM_HOME / "python"
if str(TVM_PYTHON) not in sys.path:
    sys.path.insert(0, str(TVM_PYTHON))

import tvm
import tvm.contrib.gemmini as gemmini

from models.repq import builder as build_repq_model
from models.repq.layers import RepQContext
from repq_to_tvm_params import MODEL_ZOO, build_repq_tvm_artifacts, load_repq_tvm_artifacts
from scripts.gemmini.run_inference_spike import (
    _extract_main_input_spec,
    _generate_synthetic_input,
    _find_tvm_main_source,
    compile_for_spike,
    extract_spike_intrakernel_cycle_rows,
    extract_spike_main_total_cycles,
    extract_spike_semantic_cycle_rows,
    fix_generated_code,
    preprocess_for_gemmini,
    run_spike,
    run_verilator,
    write_spike_intrakernel_cycle_reports,
    write_spike_semantic_cycle_reports,
)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_IMAGE = SCRIPT_DIR / "test_cat.jpg"
DEFAULT_LABELS = SCRIPT_DIR / "imagenet_classes.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deit_tiny", choices=sorted(MODEL_ZOO))
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--calib-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--repq-artifact-dir", default=None)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--allow-random-calibration", action="store_true")
    parser.add_argument("--calib-batchsize", default=32, type=int)
    parser.add_argument("--calib-num-samples", default=32, type=int)
    parser.add_argument("--w-bits", default=8, type=int)
    parser.add_argument("--a-bits", default=8, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--skip-reparameterization", action="store_true")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "build" / "repq_tvm_spike"))
    parser.add_argument("--opt-level", type=int, default=None, choices=[0, 1, 2, 3])
    parser.add_argument("--usmp-alg", default=None)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--debug-unit",
        type=str,
        default=None,
        help=(
            "Relay debug cut point / standalone unit "
            "(e.g. only_embed, only_block0, only_stage0_block0, only_head, "
            "post_stage0_block0, pre_head)"
        ),
    )
    parser.add_argument(
        "--simulator",
        choices=["spike", "verilator"],
        default="spike",
        help="Simulator backend",
    )
    parser.add_argument(
        "--chipyard-dir",
        default=os.environ.get("CHIPYARD_DIR"),
        help="Chipyard root path (used for Verilator)",
    )
    parser.add_argument(
        "--verilator-config",
        default="BigRocketSaturnGemminiConfig",
        help="Chipyard config name for Verilator simulator",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=20000000000,
        help="Verilator +max-cycles limit (0 to disable)",
    )
    parser.add_argument(
        "--no-dramsim",
        action="store_true",
        help="Disable +dramsim when running Verilator",
    )
    parser.add_argument(
        "--verilator-verbose",
        action="store_true",
        help="Enable +verbose on Verilator runs",
    )
    parser.add_argument(
        "--verilator-save-logs",
        action="store_true",
        help="Save Verilator stdout/stderr logs into output-dir",
    )
    parser.add_argument(
        "--verilator-log-tail-lines",
        type=int,
        default=0,
        help="If saving Verilator logs, keep only last N lines (0 keeps full logs)",
    )
    parser.add_argument("--disable-gemmini-ops", action="store_true")
    parser.add_argument("--profile-semantic", action="store_true")
    parser.add_argument(
        "--profile-semantic-style",
        default="semantic",
        choices=["semantic", "aligned", "aligned_split"],
    )
    parser.add_argument(
        "--profile-intrakernel-requant",
        action="store_true",
        help=(
            "Emit rdcycle totals for generated fixed_point_multiply/requant/post-op kernels. "
            "Safe to combine with --profile-semantic."
        ),
    )
    parser.add_argument(
        "--uart-mode",
        type=str,
        default="full",
        choices=["full", "minimal"],
        help="UART verbosity for the generated baremetal harness",
    )
    return parser.parse_args()


def crop_pct_for_model(model_name: str) -> float:
    if model_name == "swin_tiny_patch4_window7_224":
        return 0.9
    return 0.875


def preprocess_image_float(image_path: Path, model_name: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
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
    return np.expand_dims(img, axis=0).astype("float32")


def prepare_repq_input_data(image_path: Path, model_name: str, input_spec: dict[str, object]):
    shape = tuple(input_spec["shape"])
    dtype = str(input_spec["dtype"])
    if shape == (1, 3, 224, 224) and dtype == "float32":
        return preprocess_image_float(image_path, model_name), "real_image"
    return _generate_synthetic_input(shape, dtype), "synthetic"


def load_labels(path: Path) -> list[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def parse_top5(stdout: str) -> list[int]:
    classes = []
    for line in stdout.splitlines():
        match = re.search(r"Class\s+(\d+)", line)
        if match:
            classes.append(int(match.group(1)))
    return classes


def parse_cycles(stdout: str) -> int | None:
    match = re.search(r"Cycles:\s+(\d+)", stdout)
    if match:
        return int(match.group(1))
    return None


TVM_MAIN_TOTAL_CYCLES_PREFIX = "[TVM_MAIN_TOTAL_CYCLES],"
TVM_MAIN_INNER_CYCLES_PREFIX = "[TVM_MAIN_INNER_CYCLES],"


def instrument_tvm_main_total_cycles(output_dir: Path) -> None:
    tvm_main_path = _find_tvm_main_source(output_dir)
    content = tvm_main_path.read_text()

    if TVM_MAIN_INNER_CYCLES_PREFIX in content:
        return

    support_code = """
extern volatile uint64_t tohost;
extern volatile uint64_t fromhost;

static void tvm_profile_print_char(char c) {
  volatile uint64_t magic_mem[8] __attribute__((aligned(64)));
  magic_mem[0] = 64;
  magic_mem[1] = 1;
  magic_mem[2] = (uintptr_t)&c;
  magic_mem[3] = 1;
  __sync_synchronize();
  tohost = (uintptr_t)magic_mem;
  while (fromhost == 0);
  fromhost = 0;
  __sync_synchronize();
}

static void tvm_profile_print_str(const char* s) {
  while (*s) tvm_profile_print_char(*s++);
}

static void tvm_profile_print_dec(uint64_t val) {
  char buf[21];
  int i = 20;
  buf[i] = 0;
  do {
    buf[--i] = '0' + (val % 10);
    val /= 10;
  } while (val && i > 0);
  tvm_profile_print_str(&buf[i]);
}

static inline void tvm_profile_cycle_barrier(void) {
  asm volatile ("" ::: "memory");
}

static inline uint64_t tvm_profile_read_cycles(void) {
  uint64_t cycles;
  tvm_profile_cycle_barrier();
  asm volatile ("rdcycle %0" : "=r" (cycles) : : "memory");
  tvm_profile_cycle_barrier();
  return cycles;
}

static uint64_t tvm_profile_main_cycles = 0;

static void tvm_profile_dump_main_cycles(void) {
  tvm_profile_print_str("[TVM_MAIN_INNER_CYCLES],");
  tvm_profile_print_dec(tvm_profile_main_cycles);
  tvm_profile_print_str("\\n");
}

"""
    include_token = '#include "tvm/runtime/c_runtime_api.h"\n'
    if include_token not in content:
        raise RuntimeError(f"Could not find TVM runtime include in {tvm_main_path}")
    updated = content.replace(include_token, include_token + support_code, 1)

    main_fn_re = re.compile(
        r"(int32_t tvmgen_default___tvm_main__\([^)]*\)\s*\{.*?)(\n\s*return 0;\n\})",
        re.DOTALL,
    )

    def _inject_dump(match: re.Match[str]) -> str:
        body, tail = match.groups()
        body = body.replace(
            "{",
            "{\n  uint64_t __tvm_main_profile_start = tvm_profile_read_cycles();",
            1,
        )
        return (
            body
            + "\n  tvm_profile_main_cycles = "
            "tvm_profile_read_cycles() - __tvm_main_profile_start;"
            "\n  tvm_profile_dump_main_cycles();"
            + tail
        )

    updated, replaced = main_fn_re.subn(_inject_dump, updated, count=1)
    if replaced != 1:
        raise RuntimeError("Failed to inject TVM main total cycle dump into tvm main")

    tvm_main_path.write_text(updated)
    print("[Fix] Instrumented TVM main total rdcycle dump")


def create_repq_harness(
    output_dir: Path,
    model_name: str,
    input_data: np.ndarray,
    classification_output: bool = True,
    debug_unit: str | None = None,
    uart_mode: str = "full",
) -> None:
    input_bytes = np.ascontiguousarray(input_data).view(np.uint8)
    input_c_array = ", ".join(str(int(x)) for x in input_bytes.flatten())
    input_size_bytes = input_bytes.size
    run_mode = "classification" if classification_output else "debug"
    debug_label = debug_unit if debug_unit is not None else "full_model"
    prologue_code = ""
    pre_run_code = ""
    post_run_code = ""
    status_error_code = ""
    epilogue_code = ""
    result_code = ""

    if uart_mode == "minimal":
        status_error_code = f"""
    uint64_t checksum = 0;
    for (int i = 0; i < TVMGEN_DEFAULT_OUTPUT_SIZE; ++i) {{
        checksum = checksum * 131 + output_data[i];
    }}
    print_str("[TVM_RUN_RESULT],");
    print_str("{debug_label}");
    print_str(",");
    print_dec((uint64_t)(status == 0 ? 0 : 1));
    print_str(",");
    print_dec(cycles);
    print_str(",");
    print_dec(checksum);
    print_str("\\n");
    if (status != 0) {{
        spike_exit(1);
    }}
"""
    elif classification_output:
        prologue_code = f"""
    print_str("\\n========================================\\n");
    print_str("RepQ TVM Gemmini Inference\\n");
    print_str("Model: {model_name}\\n");
    print_str("Run mode: {run_mode}\\n");
    print_str("Debug unit: {debug_label}\\n");
    print_str("========================================\\n\\n");
"""
        post_run_code = """
    print_str("Cycles: ");
    print_dec(cycles);
    print_str("\\n\\n");
"""
        status_error_code = """
    if (status != 0) {
        print_str("TVM run failed with status: ");
        print_dec((uint64_t)status);
        print_str("\\n");
        spike_exit(1);
    }
"""
        result_code = """
    print_str("Top-5 Predictions:\\n");
    float* output_logits = (float*)output_data;
    for (int rank = 0; rank < 5; rank++) {
        int max_idx = 0;
        float max_val = output_logits[0];
        for (int j = 1; j < 1000; j++) {
            if (output_logits[j] > max_val) {
                max_val = output_logits[j];
                max_idx = j;
            }
        }
        print_str("  ");
        print_dec(rank + 1);
        print_str(". Class ");
        print_dec(max_idx);
        print_str("\\n");
        output_logits[max_idx] = -1e9f;
    }
"""
        epilogue_code = '    print_str("\\nDone!\\n");'
    else:
        prologue_code = f"""
    print_str("\\n========================================\\n");
    print_str("RepQ TVM Gemmini Inference\\n");
    print_str("Model: {model_name}\\n");
    print_str("Run mode: {run_mode}\\n");
    print_str("Debug unit: {debug_label}\\n");
    print_str("========================================\\n\\n");
"""
        post_run_code = """
    print_str("Cycles: ");
    print_dec(cycles);
    print_str("\\n");
"""
        status_error_code = """
    if (status != 0) {
        print_str("TVM run failed with status: ");
        print_dec((uint64_t)status);
        print_str("\\n");
        spike_exit(1);
    }
"""
        result_code = """
    uint64_t checksum = 0;
    for (int i = 0; i < TVMGEN_DEFAULT_OUTPUT_SIZE; ++i) {
        checksum = checksum * 131 + output_data[i];
    }
    print_str("Checksum: ");
    print_dec(checksum);
    print_str("\\n");
"""
        epilogue_code = '    print_str("\\nDone!\\n");'

    harness_code = f"""
#include <stdint.h>
#include <stddef.h>
#include "gemmini.h"
#include "tvmgen_default.h"

extern volatile uint64_t tohost;
extern volatile uint64_t fromhost;

void print_char(char c) {{
    volatile uint64_t magic_mem[8] __attribute__((aligned(64)));
    magic_mem[0] = 64;
    magic_mem[1] = 1;
    magic_mem[2] = (uintptr_t)&c;
    magic_mem[3] = 1;
    __sync_synchronize();
    tohost = (uintptr_t)magic_mem;
    while (fromhost == 0);
    fromhost = 0;
    __sync_synchronize();
}}

void print_str(const char* s) {{
    while (*s) print_char(*s++);
}}

void print_dec(uint64_t val) {{
    char buf[21];
    int i = 20;
    buf[i] = 0;
    do {{
        buf[--i] = '0' + (val % 10);
        val /= 10;
    }} while (val && i > 0);
    print_str(&buf[i]);
}}

uint64_t read_cycles(void) {{
    uint64_t cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
}}

static inline void spike_exit(int code) {{
    tohost = (code << 1) | 1;
    while (1);
}}

#define INPUT_SIZE_BYTES ({input_size_bytes})

static const uint8_t input_data[INPUT_SIZE_BYTES] __attribute__((aligned(16))) = {{
    {input_c_array}
}};

static uint8_t output_data[TVMGEN_DEFAULT_OUTPUT_SIZE] __attribute__((aligned(16)));

int main() {{
{prologue_code}

    struct tvmgen_default_inputs inputs;
    inputs.data = (void*)input_data;

    struct tvmgen_default_outputs outputs;
    outputs.output = output_data;

{pre_run_code}
    gemmini_flush(0);
    uint64_t start = read_cycles();
    int32_t status = tvmgen_default_run(&inputs, &outputs);
    gemmini_fence();
    uint64_t end = read_cycles();
    uint64_t cycles = end - start;

{post_run_code}
{status_error_code}
{result_code}

{epilogue_code}
    spike_exit(0);
    return 0;
}}
"""
    with open(output_dir / "main.c", "w") as f:
        f.write(harness_code)


def load_or_build_artifacts(args: argparse.Namespace):
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
    return build_repq_tvm_artifacts(repq_args)


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    labels_path = Path(args.labels)
    output_dir = Path(args.output_dir).resolve()

    model_name, params, meta = load_or_build_artifacts(args)
    labels = load_labels(labels_path)
    print("=" * 60)
    print("RepQ TVM Gemmini Inference")
    print("=" * 60)
    print(f"Image: {image_path}")
    if args.debug_unit:
        print(f"Debug unit: {args.debug_unit}")

    gemmini.Environment.init_overwrite(
        dim=16,
        acc_rows=1024,
        bank_rows=4096,
        inp_dtype="int8",
        wgt_dtype="int8",
        acc_dtype="int32",
    )

    RepQContext.set_use_gemmini_ops(not args.disable_gemmini_ops)
    mod, tvm_params = build_repq_model.get_workload(
        name=model_name,
        params=params,
        meta=meta,
        batch_size=1,
        image_shape=(3, 224, 224),
        debug_unit=args.debug_unit,
    )
    input_spec = _extract_main_input_spec(mod)
    input_data, input_mode = prepare_repq_input_data(image_path, model_name, input_spec)
    print(f"Input mode: {input_mode}")
    print(f"Input tensor shape: {input_data.shape}, dtype: {input_data.dtype}")
    print(f"Value range: [{input_data.min()}, {input_data.max()}]")
    if RepQContext.use_gemmini_ops:
        mod = preprocess_for_gemmini(mod, model_name, canonicalize_qnn=False)
        mod = tvm.relay.transform.InferType()(mod)

    runtime = tvm.relay.backend.Runtime("crt", {"system-lib": False})
    executor = tvm.relay.backend.Executor("aot", options={"interface-api": "c", "unpacked-api": 1})
    target = tvm.target.Target({"kind": "c", "device": "gemmini"})
    usmp_alg = args.usmp_alg
    if usmp_alg is None:
        usmp_alg = "greedy_by_size" if model_name.startswith("swin_") else "hill_climb"
    if usmp_alg == "none":
        usmp_alg = ""
    opt_level = args.opt_level
    if opt_level is None:
        opt_level = 2 if model_name.startswith("swin_") else 3

    build_start = time.time()
    disabled_passes = ["AlterOpLayout"]
    print(
        f"[Build] relay.build 시작 (usmp_alg={usmp_alg}, opt_level={opt_level})"
    )
    if disabled_passes:
        print(f"[Build] disabled_pass={disabled_passes}")
    with gemmini.build_config(
        usmp_alg=usmp_alg,
        opt_level=opt_level,
        disabled_pass=disabled_passes,
    ):
        module = tvm.relay.build(
            mod,
            executor=executor,
            runtime=runtime,
            target=target,
            params=tvm_params,
        )
    print(f"[Build] relay.build finished in {time.time() - build_start:.1f}s")

    if output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mlf_path = output_dir / "model.tar"
    tvm.micro.export_model_library_format(module, mlf_path)
    with tarfile.open(mlf_path, "r:") as tar:
        tar.extractall(output_dir)

    if args.profile_semantic:
        try:
            fix_generated_code(
                output_dir,
                model_name,
                segment_profile_style=args.profile_semantic_style,
                instrument_requant_intrakernel=args.profile_intrakernel_requant,
                debug_unit=args.debug_unit,
            )
        except RuntimeError as exc:
            print(f"[WARN] Semantic profiling unavailable for RepQ TVM: {exc}")
            fix_generated_code(
                output_dir,
                model_name,
                instrument_requant_intrakernel=args.profile_intrakernel_requant,
                debug_unit=args.debug_unit,
            )
    else:
        fix_generated_code(
            output_dir,
            model_name,
            instrument_requant_intrakernel=args.profile_intrakernel_requant,
            debug_unit=args.debug_unit,
        )
    instrument_tvm_main_total_cycles(output_dir)
    classification_output = args.debug_unit is None
    create_repq_harness(
        output_dir,
        model_name,
        input_data,
        classification_output=classification_output,
        debug_unit=args.debug_unit,
        uart_mode=args.uart_mode,
    )
    binary = compile_for_spike(output_dir, f"{args.model}_repq")
    if binary is None:
        return 1

    ver_stdout_path = None
    ver_stderr_path = None
    if args.simulator == "spike":
        stdout, stderr = run_spike(binary, timeout=args.timeout)
        if stdout is None:
            return 1
    else:
        stdout, stderr, ver_stdout_path, ver_stderr_path = run_verilator(
            binary,
            timeout=args.timeout,
            chipyard_dir=args.chipyard_dir,
            verilator_config=args.verilator_config,
            max_cycles=args.max_cycles,
            dramsim=not args.no_dramsim,
            verbose=args.verilator_verbose,
            log_dir=output_dir if args.verilator_save_logs else None,
            log_tail_lines=args.verilator_log_tail_lines,
        )
        if stdout is None and ver_stdout_path is None:
            return 1
        if ver_stdout_path is not None:
            stdout = Path(ver_stdout_path).read_text()
        if ver_stderr_path is not None:
            stderr = Path(ver_stderr_path).read_text()
        elif stderr is None:
            stderr = ""

    if args.simulator == "spike":
        (output_dir / "spike_stdout.log").write_text(stdout)
        if stderr:
            (output_dir / "spike_stderr.log").write_text(stderr)
    else:
        if ver_stdout_path is None:
            (output_dir / "verilator_stdout.log").write_text(stdout)
        if stderr and ver_stderr_path is None:
            (output_dir / "verilator_stderr.log").write_text(stderr)

    print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)

    top5 = parse_top5(stdout) if classification_output else []
    cycles = parse_cycles(stdout)
    main_total_cycles = extract_spike_main_total_cycles(stdout)
    if top5:
        print("\nParsed Top-5:")
        for rank, cls_idx in enumerate(top5[:5], start=1):
            label = labels[cls_idx] if 0 <= cls_idx < len(labels) else "<unknown>"
            print(f"{rank}. {cls_idx} {label}")
    if cycles is not None:
        print(f"\nCycles: {cycles}")
    if main_total_cycles is not None:
        print(f"TVM_MAIN_TOTAL_CYCLES: {main_total_cycles}")
        (output_dir / "spike_total_main_cycles.log").write_text(
            f"Inference took {main_total_cycles} cycles\n"
        )
    if args.profile_intrakernel_requant:
        intrakernel_rows = extract_spike_intrakernel_cycle_rows(stdout)
        if intrakernel_rows:
            intrakernel_csv_path, intrakernel_txt_path = write_spike_intrakernel_cycle_reports(
                output_dir, intrakernel_rows
            )
            print(f"[Info] TVM intrakernel cycle CSV: {intrakernel_csv_path}")
            print(f"[Info] TVM intrakernel cycle report: {intrakernel_txt_path}")
        else:
            print("[WARN] No [TVM_INTRAKERNEL_CYCLES] rows found in simulation output")
    if args.profile_semantic:
        semantic_rows = extract_spike_semantic_cycle_rows(stdout)
        if semantic_rows:
            (
                segment_csv_path,
                segment_txt_path,
                layer_csv_path,
                layer_txt_path,
                component_csv_path,
                component_txt_path,
                layer_component_csv_path,
                layer_component_txt_path,
            ) = write_spike_semantic_cycle_reports(
                output_dir, semantic_rows, main_total_cycles=main_total_cycles
            )
            print(f"[Info] TVM semantic segment CSV: {segment_csv_path}")
            print(f"[Info] TVM semantic segment report: {segment_txt_path}")
            print(f"[Info] TVM layer cycle CSV: {layer_csv_path}")
            print(f"[Info] TVM layer cycle report: {layer_txt_path}")
            print(f"[Info] TVM component cycle CSV: {component_csv_path}")
            print(f"[Info] TVM component cycle report: {component_txt_path}")
            print(f"[Info] TVM layer-component cycle CSV: {layer_component_csv_path}")
            print(f"[Info] TVM layer-component cycle report: {layer_component_txt_path}")
        else:
            print("[WARN] No [TVM_SEMANTIC_CYCLES] rows found in Spike output")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
