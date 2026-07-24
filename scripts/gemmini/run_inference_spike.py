#!/usr/bin/env python3
"""
Run I-ViT inference on a REAL IMAGE using Gemmini.

This script:
1. Loads a real image (JPEG/PNG)
2. Applies ImageNet preprocessing + quantization
3. Embeds the preprocessed image into the test harness
4. Runs on Spike or Verilator and reports the predicted class

Usage:
    python run_inference_spike.py --image /path/to/image.jpg --checkpoint /path/to/checkpoint.pth.tar
"""

import os
import sys
import argparse
import subprocess
import pathlib
import shutil
import tarfile
import bisect
import re
import time
import threading
import resource
from collections import defaultdict, deque
import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
SCRIPTS_DIR = SCRIPT_DIR.parent  # scripts/
REPO_ROOT = SCRIPTS_DIR.parent  # I-ViT-Gemmini/
TVM_SCRIPTS_DIR = SCRIPTS_DIR / "tvm"
sys.path.insert(0, str(TVM_SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image
import tvm
from tvm import relay
import tvm.contrib.gemmini as gemmini
from tvm.contrib.gemmini.legalize import LegalizeGemmini
from tvm.relay.op.contrib.gemmini_byoc import (
    enabled as gemmini_byoc_enabled,
    llvm_riscv_target,
    partition_for_gemmini,
)
from tvm.ir.memory_pools import ConstantPoolInfo
from tvm.relay.build_module import bind_params_by_name

from models.ivit.builder import get_workload
import pytorch_to_tvm_params as convert_model

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SATURN_SPIKE_ISA = "rv64gcv_zvl512b_zicsr_zifencei_zicntr_zihpm"
SATURN_LINK_MARCH = "rv64gcv_zvl512b_zicntr_zicsr"


def riscv_march_enables_v(march: str) -> bool:
    """True if march enables the V (or Zve*) vector extension.

    Note: do not use ``\"v\" in march`` — that matches the ``v`` in ``rv64gc``.
    """
    m = (march or "").lower().strip()
    if not m:
        return False
    if "_zve" in m or "+zve" in m or m.startswith("zve"):
        return True
    # LLVM feature lists sometimes use +v
    if re.search(r"(^|[,+])v([,+]|$)", m):
        return True
    mo = re.match(r"rv(?:32|64|128)?([a-z0-9]*)", m)
    if mo:
        return "v" in mo.group(1)
    return False
IMAGENET_CLASSES = None

MODEL_SPECS = {
    "deit_tiny_patch16_224": {"embed_dim": 192, "depth": 12},
    "deit_small_patch16_224": {"embed_dim": 384, "depth": 12},
    "fq_deit_tiny_patch16_224": {"embed_dim": 192, "depth": 12},
    "ptq4_deit_tiny_patch16_224": {"embed_dim": 192, "depth": 12},
    "ptq4_deit_small_patch16_224": {"embed_dim": 384, "depth": 12},
    "swin_tiny_patch4_window7_224": {"embed_dim": 768, "depth": 12},
    "swin_small_patch4_window7_224": {"embed_dim": 768, "depth": 24},
}


def _random_state_model_name(model_name):
    if model_name == "fq_deit_tiny_patch16_224":
        raise RuntimeError("fq_deit_tiny_patch16_224 requires flexi e2e_model.pt checkpoint")
    if model_name.startswith("ptq4_deit_"):
        raise RuntimeError(f"{model_name}: use flexi e2e_model.pt or get_workload random scaffold")
    return model_name


def build_random_qat_state_dict(model_name):
    ivit_root = REPO_ROOT / "I-ViT"
    state_model_name = _random_state_model_name(model_name)

    # This runner imports the local `models.ivit` package before this point. The
    # vendored I-ViT tree also uses the top-level package name `models`, so build
    # the throwaway random QAT checkpoint in a clean subprocess.
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    script = r"""
import sys
import torch

ivit_root, model_name, output_path = sys.argv[1:4]
sys.path.insert(0, ivit_root)

from models.model_utils import freeze_model
from models.swin_quant import swin_small_patch4_window7_224, swin_tiny_patch4_window7_224
from models.vit_quant import deit_small_patch16_224, deit_tiny_patch16_224

builders = {
    "deit_tiny_patch16_224": deit_tiny_patch16_224,
    "deit_small_patch16_224": deit_small_patch16_224,
    "swin_tiny_patch4_window7_224": swin_tiny_patch4_window7_224,
    "swin_small_patch4_window7_224": swin_small_patch4_window7_224,
}
model = builders[model_name](pretrained=False).eval()
freeze_model(model)
with torch.no_grad():
    model(torch.randn(1, 3, 224, 224))
torch.save(model.state_dict(), output_path)
"""
    try:
        subprocess.run(
            [sys.executable, "-c", script, str(ivit_root), state_model_name, tmp_path],
            check=True,
        )
        return torch.load(tmp_path, map_location="cpu")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def maybe_raise_stack_limit_for_build(model_name, opt_level):
    """Avoid stack-overflow crashes on very large unoptimized Swin builds."""
    if not model_name.startswith("swin_") or opt_level != 0:
        return None

    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    target = hard
    if target == resource.RLIM_INFINITY:
        target = resource.RLIM_INFINITY
    elif soft >= hard:
        return None

    try:
        resource.setrlimit(resource.RLIMIT_STACK, (target, hard))
    except (OSError, ValueError):
        return None

    new_soft, _ = resource.getrlimit(resource.RLIMIT_STACK)
    return soft, new_soft


def rewrite_erf_to_fast_erf(mod):
    """Rewrite Relay ``erf`` → ``fast_erf`` only (GELU), without full FastMath.

    Softmax stays ``nn.softmax``; TOPI may still pick ``fast_softmax`` compute via
    ``TVM_TOPI_VECTORIZABLE_ELEMWISE_MATH``. Avoids scalar ``erff`` without
    ``relay.transform.FastMath()``.
    """
    from tvm.relay.dataflow_pattern import (
        DFPatternCallback,
        is_op,
        rewrite,
        wildcard,
    )

    class ErfToFastErf(DFPatternCallback):
        def __init__(self):
            super().__init__()
            self.x = wildcard()
            self.pattern = is_op("erf")(self.x)

        def callback(self, pre, post, node_map):  # noqa: ARG002
            return relay.Call(relay.op.get("fast_erf"), [node_map[self.x][0]])

    new_main = rewrite(ErfToFastErf(), mod["main"])
    mod.update_func(mod.get_global_var("main"), new_main)
    return mod


def preprocess_for_heterogeneous_gemmini(
    mod, model_name, canonicalize_qnn=False, *, enable_fast_math=False
):
    """Partition Gemmini matmuls for BYOC; leave CPU epilogues on LLVM host.

    When ``enable_fast_math`` is set, rewrite ``nn.softmax`` / ``exp`` / ``erf``
    / ``tanh`` to polynomial ``nn.fast_softmax`` / ``fast_exp`` / ``fast_erf`` /
    ``fast_tanh`` via ``relay.transform.FastMath()``.

    Prefer the lighter path for RVV fair builds: set
    ``TVM_TOPI_VECTORIZABLE_ELEMWISE_MATH=1`` and leave FastMath off. Then TOPI
    uses ``fast_softmax`` for ``nn.softmax``, and we only rewrite ``erf``→
    ``fast_erf`` for GELU — stock Softmax op in the graph, vectorizable math.
    """
    if model_name.startswith("swin_"):
        pattern = relay.op.contrib.get_pattern_table("gemmini")
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.ConvertLayout({"qnn.conv2d": ["NHWC", "HWIO"]})(mod)
        mod = relay.transform.FoldConstant()(mod)
        mod = relay.transform.MergeComposite(pattern)(mod)
        mod = relay.transform.InferType()(mod)
        mod = LegalizeGemmini()(mod)
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.AnnotateTarget("gemmini")(mod)
        mod = relay.transform.MergeCompilerRegions()(mod)
        mod = relay.transform.PartitionGraph()(mod)
        mod = relay.transform.InferType()(mod)
        if canonicalize_qnn:
            mod = relay.qnn.transform.CanonicalizeOps()(mod)
            mod = relay.transform.FoldConstant()(mod)
            mod = relay.transform.InferType()(mod)
    else:
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.ConvertLayout({"qnn.conv2d": ["NHWC", "HWIO"]})(mod)
        mod = relay.transform.SimplifyExpr()(mod)
        mod = partition_for_gemmini(mod)

    if enable_fast_math:
        mod = relay.transform.FastMath()(mod)
        mod = relay.transform.InferType()(mod)
    elif os.environ.get("TVM_TOPI_VECTORIZABLE_ELEMWISE_MATH", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        # GELU: erf → fast_erf. Softmax: kept as nn.softmax; TOPI strategy picks
        # fast_softmax compute when the same env is set.
        mod = rewrite_erf_to_fast_erf(mod)
        mod = relay.transform.InferType()(mod)
    return mod

def preprocess_for_gemmini(mod, model_name, canonicalize_qnn=False):
    """Apply Gemmini preprocess with a lighter pipeline for large Swin graphs."""
    if model_name.startswith("swin_"):
        pattern = relay.op.contrib.get_pattern_table("gemmini")
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.ConvertLayout({"qnn.conv2d": ["NHWC", "HWIO"]})(mod)
        mod = relay.transform.FoldConstant()(mod)
        mod = relay.transform.MergeComposite(pattern)(mod)
        mod = relay.transform.InferType()(mod)
        mod = LegalizeGemmini()(mod)
        mod = relay.transform.InferType()(mod)
        if canonicalize_qnn:
            # Lower standalone qnn ops up front while constants are still direct.
            mod = relay.qnn.transform.CanonicalizeOps()(mod)
            mod = relay.transform.FoldConstant()(mod)
            mod = relay.transform.InferType()(mod)
        return mod

    mod = gemmini.preprocess_pass(mod)
    if canonicalize_qnn:
        # At opt_level=0, some standalone qnn ops survive long enough to hit
        # unsupported generic codegen paths; canonicalize them explicitly.
        mod = relay.qnn.transform.CanonicalizeOps()(mod)
        mod = relay.transform.FoldConstant()(mod)
        mod = relay.transform.InferType()(mod)
    return mod


def load_imagenet_classes():
    global IMAGENET_CLASSES
    if IMAGENET_CLASSES is not None:
        return IMAGENET_CLASSES

    classes_url = (
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    )
    classes_file = SCRIPT_DIR / "imagenet_classes.txt"

    if not classes_file.exists():
        import urllib.request

        print(f"Downloading ImageNet classes...")
        urllib.request.urlretrieve(classes_url, classes_file)

    with open(classes_file, "r") as f:
        IMAGENET_CLASSES = [line.strip() for line in f.readlines()]

    return IMAGENET_CLASSES


def preprocess_image_float(image_path):
    """Preprocess image to float32 NCHW for FQ-DeiT (flexi FQViT prologue)."""
    img = Image.open(image_path).convert("RGB")

    width, height = img.size
    scale = 256 / min(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    img = img.resize((new_width, new_height), Image.BILINEAR)

    left = (new_width - 224) // 2
    top = (new_height - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))

    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    img_np = img_np.transpose(2, 0, 1)
    return np.expand_dims(img_np, axis=0)


def preprocess_image(image_path, input_scale):
    """
    Preprocess image for I-ViT inference.

    Steps:
    1. Resize to 256, center crop to 224x224
    2. Normalize with ImageNet mean/std
    3. Quantize to int8 using input_scale from checkpoint
    """
    img = Image.open(image_path).convert("RGB")

    width, height = img.size
    scale = 256 / min(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    img = img.resize((new_width, new_height), Image.BILINEAR)

    left = (new_width - 224) // 2
    top = (new_height - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))

    img_np = np.array(img, dtype=np.float32) / 255.0

    img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD

    img_np = img_np.transpose(2, 0, 1)
    img_np = np.expand_dims(img_np, axis=0)

    img_int8 = np.clip(np.round(img_np / input_scale), -128, 127).astype(np.int8)

    return img_int8


def _extract_main_input_spec(mod):
    main = mod["main"]
    data_param = None
    for param in main.params:
        if param.name_hint in ("data", "image"):
            data_param = param
            break
    if data_param is None:
        if not main.params:
            raise RuntimeError("Relay main has no parameters")
        data_param = main.params[0]

    tensor_type = getattr(data_param, "checked_type", None) or data_param.type_annotation
    if not isinstance(tensor_type, relay.TensorType):
        raise RuntimeError(f"Unsupported main input type: {tensor_type}")

    try:
        shape = tuple(int(dim) for dim in tensor_type.shape)
    except TypeError as exc:
        raise RuntimeError(f"Unsupported dynamic input shape: {tensor_type.shape}") from exc

    return {
        "name": data_param.name_hint,
        "shape": shape,
        "dtype": tensor_type.dtype,
    }


def _is_real_image_input(shape, dtype):
    return tuple(shape) == (1, 3, 224, 224) and dtype == "int8"


def _generate_synthetic_input(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    if dtype == "int8":
        return rng.integers(-8, 9, size=shape, dtype=np.int8)
    if dtype == "uint8":
        return rng.integers(0, 17, size=shape, dtype=np.uint8)
    if dtype == "int32":
        return rng.integers(-64, 65, size=shape, dtype=np.int32)
    if dtype == "int64":
        return rng.integers(-64, 65, size=shape, dtype=np.int64)
    if dtype == "float32":
        return (rng.standard_normal(size=shape).astype(np.float32) * 0.25).astype(np.float32)
    if dtype == "float64":
        return (rng.standard_normal(size=shape).astype(np.float64) * 0.25).astype(np.float64)
    raise RuntimeError(f"Unsupported standalone input dtype: {dtype}")


def prepare_input_data(
    image_path,
    input_scale,
    input_spec,
    *,
    force_synthetic=False,
    synthetic_seed=0,
    model_name=None,
):
    shape = input_spec["shape"]
    dtype = input_spec["dtype"]
    if force_synthetic:
        return _generate_synthetic_input(shape, dtype, seed=synthetic_seed), "synthetic"
    if model_name and (
        model_name.startswith("fq_deit_") or model_name.startswith("ptq4_deit_")
    ) and dtype == "float32":
        if image_path.exists():
            return preprocess_image_float(image_path), "real_image_float"
        return _generate_synthetic_input(shape, dtype, seed=synthetic_seed), "synthetic"
    if _is_real_image_input(shape, dtype):
        return preprocess_image(image_path, input_scale), "real_image"
    return _generate_synthetic_input(shape, dtype, seed=synthetic_seed), "synthetic"


def create_real_image_harness(
    output_dir,
    model_name,
    embed_dim,
    input_data,
    classification_output=True,
    debug_unit=None,
    uart_mode="full",
    enable_spike_rvv=False,
):
    """Create test harness with embedded input bytes."""

    input_bytes = np.ascontiguousarray(input_data).view(np.uint8)
    input_c_array = ", ".join(str(int(x)) for x in input_bytes.flatten())
    input_size_bytes = input_bytes.size
    run_mode = "classification" if classification_output else "debug"
    debug_label = debug_unit if debug_unit is not None else "full_model"
    spike_rvv_prologue = ""
    if enable_spike_rvv:
        # Stock Spike traps on vsetvli unless mstatus.VS != OFF (crt.S only sets FS|XS).
        # Also clear vtype.vill (set at reset): Saturn traps whole-register vl2r.v/vs2r.v
        # while vill=1, and LLVM emits them on paths with no preceding vsetvli.
        spike_rvv_prologue = (
            '    asm volatile("li t0, 0x600\\n csrs mstatus, t0\\n'
            ' vsetvli t0, zero, e8, m1, ta, ma" ::: "t0");\n'
        )
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
    g_checksum = checksum;
    print_str("[TVM_RUN_RESULT],");
    print_str("{debug_label}");
    print_str(",");
    print_dec((uint64_t)(status == 0 ? 0 : 1));
    print_str(",");
    print_dec(cycles);
    print_str(",");
    print_dec(checksum);
    print_str("\\n");
    print_str("[TVM_MAIN_TOTAL_CYCLES],");
    print_dec(cycles);
    print_str("\\n");
    if (status != 0) {{
        spike_exit(1);
    }}
"""
    elif classification_output:
        prologue_code = f"""
    print_str("\\n========================================\\n");
    print_str("I-ViT Real Image Inference\\n");
    print_str("Model: {model_name}\\n");
    print_str("Run mode: {run_mode}\\n");
    print_str("Debug unit: {debug_label}\\n");
    print_str("========================================\\n\\n");
"""
        pre_run_code = '    print_str("Running inference...\\n");'
        post_run_code = """
    print_str("\\n========================================\\n");
    print_str("Results\\n");
    print_str("========================================\\n");
    print_str("[TVM_MAIN_INNER_CYCLES],");
    print_dec(tvm_profile_main_cycles);
    print_str("\\n");
    print_str("Cycles: ");
    print_dec(cycles);
    print_str("\\n");
    print_str("[TVM_MAIN_TOTAL_CYCLES],");
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
        epilogue_code = '    print_str("\\nDone!\\n");'
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
    else:
        prologue_code = f"""
    print_str("\\n========================================\\n");
    print_str("I-ViT Real Image Inference\\n");
    print_str("Model: {model_name}\\n");
    print_str("Run mode: {run_mode}\\n");
    print_str("Debug unit: {debug_label}\\n");
    print_str("========================================\\n\\n");
"""
        pre_run_code = '    print_str("Running inference...\\n");'
        post_run_code = """
    print_str("\\n========================================\\n");
    print_str("Results\\n");
    print_str("========================================\\n");
    print_str("[TVM_MAIN_INNER_CYCLES],");
    print_dec(tvm_profile_main_cycles);
    print_str("\\n");
    print_str("Cycles: ");
    print_dec(cycles);
    print_str("\\n");
    print_str("[TVM_MAIN_TOTAL_CYCLES],");
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
    g_checksum = checksum;
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
volatile uint64_t g_cycles = 0;
volatile uint64_t g_checksum = 0;
/* Weak: present when TVM main cycle instrumentation is linked in. */
uint64_t tvm_profile_main_cycles __attribute__((weak)) = 0;

int main() {{
{spike_rvv_prologue}{prologue_code}
    
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
    g_cycles = cycles;
    
{post_run_code}
{status_error_code}
{result_code}
    
{epilogue_code}
    spike_exit(0);
    return 0;
}}
"""

    harness_path = output_dir / "main.c"
    with open(harness_path, "w") as f:
        f.write(harness_code)

    return harness_path


def split_oversized_tvm_main_for_riscv_link(
    output_dir,
    *,
    kernel_chunk_size=120,
    setup_chunk_size=1200,
):
    """Split giant tvm_main into smaller helpers to avoid R_RISCV_JAL truncation."""
    tvm_main_path = _find_tvm_main_source(output_dir)
    content = tvm_main_path.read_text()
    if "tvmgen_default___tvm_main_part_" in content:
        return False

    match = re.search(
        r"(TVM_DLL int32_t tvmgen_default___tvm_main__\s*\([^)]*\)\s*\{)",
        content,
        re.DOTALL,
    )
    if not match:
        return False

    start = match.start(1)
    brace_open = content.find("{", match.end(1) - 1)
    depth = 0
    end = None
    for idx in range(brace_open, len(content)):
        ch = content[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break
    if end is None:
        raise RuntimeError(f"Could not parse tvm main body in {tvm_main_path}")

    header = content[:start]
    signature = match.group(1)
    body = content[brace_open + 1 : end]
    footer = content[end + 1 :]

    lines = body.splitlines(keepends=True)
    sid_names = []
    prelude_lines = []
    sid_lines = []
    kernel_lines = []
    tail_lines = []
    seen_sid = False
    seen_kernel = False

    sid_decl_re = re.compile(r"^\s*void\*\s+(sid_\d+_let)\s*=")
    kernel_re = re.compile(r"^\s*if \(tvmgen_default_fused_")

    for line in lines:
        sid_match = sid_decl_re.match(line)
        if sid_match and not seen_kernel:
            seen_sid = True
            sid_names.append(sid_match.group(1))
            sid_lines.append(re.sub(r"^\s*void\*\s+(sid_\d+_let)\s*=", r"  \1 =", line))
            continue
        if kernel_re.match(line):
            seen_kernel = True
            kernel_lines.append(line)
            continue
        if seen_kernel:
            tail_lines.append(line)
        elif seen_sid:
            sid_lines.append(line)
        else:
            prelude_lines.append(line)

    if len(kernel_lines) < kernel_chunk_size:
        return False

    param_list = (
        "int8_t* data_buffer_var, float* output_buffer_var, "
        "uint8_t* global_const_workspace_0_var, uint8_t* global_workspace_1_var"
    )
    static_decls = "".join(f"static void* {name};\n" for name in sid_names)

    helpers = []
    setup_count = 0
    for chunk_idx in range(0, len(sid_lines), setup_chunk_size):
        chunk = sid_lines[chunk_idx : chunk_idx + setup_chunk_size]
        if not any(line.strip() for line in chunk):
            continue
        helpers.append(
            f"static int32_t tvmgen_default___tvm_main_setup_{setup_count}({param_list}) {{\n"
            + "".join(chunk)
            + "  return 0;\n}\n\n"
        )
        setup_count += 1

    kernel_count = 0
    for chunk_idx in range(0, len(kernel_lines), kernel_chunk_size):
        chunk = kernel_lines[chunk_idx : chunk_idx + kernel_chunk_size]
        helpers.append(
            f"static int32_t tvmgen_default___tvm_main_kernels_{kernel_count}({param_list}) {{\n"
            + "".join(chunk)
            + "  return 0;\n}\n\n"
        )
        kernel_count += 1

    main_calls = list(prelude_lines)
    for part_id in range(setup_count):
        main_calls.append(
            f"  if (tvmgen_default___tvm_main_setup_{part_id}("
            f"data_buffer_var, output_buffer_var, global_const_workspace_0_var, "
            f"global_workspace_1_var) != 0) return -1;\n"
        )
    for part_id in range(kernel_count):
        main_calls.append(
            f"  if (tvmgen_default___tvm_main_kernels_{part_id}("
            f"data_buffer_var, output_buffer_var, global_const_workspace_0_var, "
            f"global_workspace_1_var) != 0) return -1;\n"
        )

    new_main = signature + "\n" + "".join(main_calls) + "".join(tail_lines) + "}\n"
    updated = header + static_decls + "\n" + "".join(helpers) + new_main + footer
    tvm_main_path.write_text(updated)
    print(
        f"[Fix] Split tvm_main into {setup_count} setup + {kernel_count} kernel helpers "
        f"({len(kernel_lines)} kernel calls)"
    )
    return True


def _find_tvm_main_source(output_dir):
    codegen_src_dir = output_dir / "codegen" / "host" / "src"
    for candidate in sorted(codegen_src_dir.glob("default_lib*.c")):
        with open(candidate, "r") as f:
            content = f.read()
            if re.search(
                r"TVM_DLL int32_t tvmgen_default___tvm_main__\s*\([^;]*\)\s*\{",
                content,
                re.DOTALL,
            ):
                return candidate
    raise RuntimeError("Could not find generated tvm main source")


TVM_MAIN_TOTAL_CYCLES_PREFIX = "[TVM_MAIN_TOTAL_CYCLES],"
TVM_MAIN_INNER_CYCLES_PREFIX = "[TVM_MAIN_INNER_CYCLES],"


def _tvm_profile_support_code(*, inner_cycles_prefix):
    return f"""
extern volatile uint64_t tohost;
extern volatile uint64_t fromhost;

static void tvm_profile_print_char(char c) {{
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

static void tvm_profile_print_str(const char* s) {{
  while (*s) tvm_profile_print_char(*s++);
}}

static void tvm_profile_print_dec(uint64_t val) {{
  char buf[21];
  int i = 20;
  buf[i] = 0;
  do {{
    buf[--i] = '0' + (val % 10);
    val /= 10;
  }} while (val && i > 0);
  tvm_profile_print_str(&buf[i]);
}}

static inline void tvm_profile_cycle_barrier(void) {{
  asm volatile ("" ::: "memory");
}}

static inline uint64_t tvm_profile_read_cycles(void) {{
  uint64_t cycles;
  tvm_profile_cycle_barrier();
  asm volatile ("rdcycle %0" : "=r" (cycles) : : "memory");
  tvm_profile_cycle_barrier();
  return cycles;
}}

/* Non-static so the baremetal harness can print AFTER the timed region.
 * Dumping UART inside tvmgen_default_run inflates FireSim TOTAL by ~10^8 cycles. */
uint64_t tvm_profile_main_cycles = 0;

static void tvm_profile_dump_main_cycles(void) {{
  tvm_profile_print_str("{inner_cycles_prefix}");
  tvm_profile_print_dec(tvm_profile_main_cycles);
  tvm_profile_print_str("\\n");
}}

"""


def instrument_tvm_main_total_cycles(output_dir):
    tvm_main_path = _find_tvm_main_source(output_dir)
    content = tvm_main_path.read_text()

    if TVM_MAIN_INNER_CYCLES_PREFIX in content:
        return

    support_code = _tvm_profile_support_code(inner_cycles_prefix=TVM_MAIN_INNER_CYCLES_PREFIX)
    include_token = '#include "tvm/runtime/c_runtime_api.h"\n'
    if include_token not in content:
        raise RuntimeError(f"Could not find TVM runtime include in {tvm_main_path}")
    updated = content.replace(include_token, include_token + support_code, 1)

    main_fn_re = re.compile(
        r"(int32_t tvmgen_default___tvm_main__\([^)]*\)\s*\{.*?)(\n\s*return 0;\n\})",
        re.DOTALL,
    )

    def _inject_dump(match):
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
            + tail
        )

    updated, replaced = main_fn_re.subn(_inject_dump, updated, count=1)
    if replaced != 1:
        raise RuntimeError("Failed to inject TVM main total cycle dump into tvm main")

    tvm_main_path.write_text(updated)
    print("[Fix] Instrumented TVM main inner rdcycle dump")


def instrument_llvm_aot_main_total_cycles(output_dir):
    """Instrument llvm-gemmini shim around tvm_main (LLVM object has no C tvm_main)."""
    shim_path = output_dir / "codegen" / "host" / "src" / "llvm_aot_shim.c"
    if not shim_path.exists():
        raise RuntimeError(f"Missing LLVM AOT shim: {shim_path}")

    content = shim_path.read_text()
    if TVM_MAIN_INNER_CYCLES_PREFIX in content:
        return

    include_token = '#include "tvmgen_default.h"\n'
    if include_token not in content:
        raise RuntimeError(f"Could not find tvmgen_default include in {shim_path}")

    support_code = _tvm_profile_support_code(inner_cycles_prefix=TVM_MAIN_INNER_CYCLES_PREFIX)
    updated = content.replace(include_token, include_token + support_code, 1)

    run_fn_re = re.compile(
        r"int32_t tvmgen_default_run\(\s*"
        r"struct tvmgen_default_inputs\* inputs,\s*"
        r"struct tvmgen_default_outputs\* outputs\)\s*\{\s*"
        r"return tvmgen_default___tvm_main__\(\s*"
        r"inputs->data,\s*outputs->output,\s*global_const_workspace,\s*global_workspace\);\s*\}",
        re.DOTALL,
    )
    replacement = """int32_t tvmgen_default_run(
    struct tvmgen_default_inputs* inputs, struct tvmgen_default_outputs* outputs) {
  uint64_t __tvm_main_profile_start = tvm_profile_read_cycles();
  int32_t __tvm_main_status = tvmgen_default___tvm_main__(
      inputs->data, outputs->output, global_const_workspace, global_workspace);
  tvm_profile_main_cycles = tvm_profile_read_cycles() - __tvm_main_profile_start;
  /* Defer UART dump to harness post_run so FireSim TOTAL excludes print. */
  return __tvm_main_status;
}"""
    updated, replaced = run_fn_re.subn(replacement, updated, count=1)
    if replaced != 1:
        raise RuntimeError("Failed to inject TVM main inner cycle dump into llvm_aot_shim.c")

    shim_path.write_text(updated)
    print("[Fix] Instrumented llvm-gemmini tvm_main inner rdcycle dump")


def _riscv_objdump_bin():
    riscv = os.environ.get("RISCV", "/root/flexi/chipyard/.conda-env/riscv-tools")
    return f"{riscv}/bin/riscv64-unknown-elf-objdump"


def _extract_llvm_tvm_main_call_names_from_object(obj_path):
    """Recover tvm_main call order from CALL relocs inside tvm_main only."""
    objdump = _riscv_objdump_bin()
    nm = objdump.replace("objdump", "nm")

    nm_result = subprocess.run(
        [nm, str(obj_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if nm_result.returncode != 0:
        raise RuntimeError(f"nm failed for {obj_path}: {nm_result.stderr}")

    main_start = None
    main_size = None
    for line in nm_result.stdout.splitlines():
        # e.g. 0000000000000000 T tvmgen_default___tvm_main__
        parts = line.split()
        if len(parts) >= 3 and parts[-1] == "tvmgen_default___tvm_main__":
            main_start = int(parts[0], 16)
            # Prefer size from `nm -S` when available.
            break

    nm_s = subprocess.run(
        [nm, "-S", str(obj_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if nm_s.returncode == 0:
        for line in nm_s.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[-1] == "tvmgen_default___tvm_main__":
                main_start = int(parts[0], 16)
                main_size = int(parts[1], 16)
                break

    if main_start is None:
        raise RuntimeError(f"tvm_main symbol not found in {obj_path}")
    if main_size is None:
        # Fallback: disassemble and estimate until next global symbol.
        main_size = 0x10000

    main_end = main_start + main_size
    reloc = subprocess.run(
        [objdump, "-r", str(obj_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if reloc.returncode != 0:
        raise RuntimeError(f"objdump -r failed for {obj_path}: {reloc.stderr}")

    calls = []
    for line in reloc.stdout.splitlines():
        if "R_RISCV_CALL" not in line:
            continue
        addr_match = re.match(r"^([0-9a-fA-F]+)\s+", line)
        sym_match = re.search(
            r"(tvmgen_default_(?:fused_|gemmini_main_)[A-Za-z0-9_]+)",
            line,
        )
        if not addr_match or not sym_match:
            continue
        addr = int(addr_match.group(1), 16)
        if main_start <= addr < main_end:
            calls.append(sym_match.group(1))

    if not calls:
        raise RuntimeError(f"No TVM kernel call relocations found in tvm_main of {obj_path}")
    return calls


def _extract_llvm_tvm_main_call_names_from_elf(elf_path):
    objdump = _riscv_objdump_bin()
    result = subprocess.run(
        [objdump, "-d", str(elf_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"objdump -d failed for {elf_path}: {result.stderr}")

    calls = []
    in_main = False
    for line in result.stdout.splitlines():
        if re.match(r"^[0-9a-f]+ <tvmgen_default___tvm_main__>:", line):
            in_main = True
            continue
        if in_main and re.match(r"^[0-9a-f]+ <", line):
            break
        if not in_main:
            continue
        match = re.search(
            r"\bj(?:al|alr)\b.*<(tvmgen_default_(?:fused_|gemmini_main_)[^>]+)>",
            line,
        )
        if match:
            calls.append(match.group(1))
    if not calls:
        raise RuntimeError(f"No TVM kernel calls found in {elf_path} tvm_main")
    return calls


def _find_llvm_host_object(output_dir):
    lib_dir = output_dir / "codegen" / "host" / "lib"
    candidates = sorted(lib_dir.glob("default_lib*.o"))
    for path in candidates:
        try:
            _extract_llvm_tvm_main_call_names_from_object(path)
            return path
        except RuntimeError:
            continue
    raise RuntimeError(f"Could not find LLVM host object with tvm_main under {lib_dir}")


def prepare_llvm_host_object_for_wraps(obj_path, symbol_names):
    """Rename defined fused_* bodies to __real_* and leave CALL relocs unresolved.

    GNU ``--wrap`` only intercepts *undefined* references. Host fused kernels live in
    the same LLVM object as ``tvm_main``, so their CALL_PLT relocs bind locally and
    bypass wraps. Renaming the definitions makes those CALL sites undefined again.
    """
    try:
        import lief
    except ImportError as exc:
        raise RuntimeError(
            "llvm-gemmini kernel profiling requires the 'lief' package "
            "(pip install lief) to rebind same-object CALL relocs for --wrap"
        ) from exc

    obj_path = pathlib.Path(obj_path)
    binary = lief.parse(str(obj_path))
    if binary is None:
        raise RuntimeError(f"lief failed to parse {obj_path}")

    renamed = []
    for name in symbol_names:
        sym = binary.get_symbol(name)
        if sym is None:
            continue
        if int(sym.shndx) == 0:
            continue  # already undefined (e.g. gemmini_main_* in default_lib0)
        if sym.type != lief.ELF.Symbol.TYPE.FUNC:
            continue
        sym.name = f"__real_{name}"
        undef = lief.ELF.Symbol()
        undef.name = name
        undef.type = lief.ELF.Symbol.TYPE.FUNC
        undef.binding = lief.ELF.Symbol.BINDING.GLOBAL
        undef.value = 0
        undef.size = 0
        binary.add_symtab_symbol(undef)
        renamed.append(name)

    by_name = {}
    for sym in binary.symbols:
        by_name.setdefault(sym.name, []).append(sym)

    retargeted = 0
    call_types = {
        lief.ELF.Relocation.TYPE.RISCV_CALL,
        lief.ELF.Relocation.TYPE.RISCV_CALL_PLT,
    }
    for reloc in binary.relocations:
        if not reloc.has_symbol or reloc.type not in call_types:
            continue
        sym_name = reloc.symbol.name
        if not sym_name.startswith("__real_"):
            continue
        orig = sym_name[len("__real_") :]
        undefs = [cand for cand in by_name.get(orig, []) if int(cand.shndx) == 0]
        if not undefs:
            continue
        reloc.symbol = undefs[0]
        retargeted += 1

    binary.write(str(obj_path))
    print(
        f"[Info] Prepared {obj_path.name} for --wrap "
        f"(renamed={len(renamed)}, retargeted_calls={retargeted})"
    )
    return renamed


def generate_llvm_kernel_profile_wraps(output_dir, model_name, *, debug_unit=None):
    """Emit interposer stubs that rdcycle each llvm-gemmini fused/gemmini_main call.

    Host fused kernels share an object with ``tvm_main``, so GNU ``--wrap`` cannot
    intercept them. We rename those bodies to ``__real_*`` (LIEF) and provide strong
    interposers under the original names.

    ``gemmini_main_*`` lives in a separate C object, so plain ``--wrap`` works there.
    """
    host_obj = _find_llvm_host_object(output_dir)
    call_names = _extract_llvm_tvm_main_call_names_from_object(host_obj)
    fused_symbols = sorted({n for n in call_names if n.startswith("tvmgen_default_fused_")})
    gemmini_symbols = sorted({n for n in call_names if n.startswith("tvmgen_default_gemmini_main_")})
    prepare_llvm_host_object_for_wraps(host_obj, fused_symbols)
    call_names = _extract_llvm_tvm_main_call_names_from_object(host_obj)
    call_order_path = output_dir / "llvm_tvm_main_call_order.txt"
    call_order_path.write_text(
        "\n".join(f"{i+1},{name}" for i, name in enumerate(call_names)) + "\n"
    )

    _, semantic_segments, _ = _build_tvm_semantic_segments_from_call_names(
        call_names, model_name, debug_unit=debug_unit
    )
    wrap_symbols = sorted(set(call_names))
    kernel_names_initializer = ",\n".join(f'  "{name}"' for name in call_names)
    segment_layer_initializer = ",\n".join(f'  "{segment["layer"]}"' for segment in semantic_segments)
    segment_component_initializer = ",\n".join(
        f'  "{segment["component"]}"' for segment in semantic_segments
    )
    segment_start_initializer = ", ".join(
        str(segment["start_call"]) for segment in semantic_segments
    )
    segment_end_initializer = ", ".join(str(segment["end_call"]) for segment in semantic_segments)

    wrap_decls = []
    wrap_defs = []
    for name in wrap_symbols:
        use_gnu_wrap = name.startswith("tvmgen_default_gemmini_main_")
        real_name = f"__real_{name}"
        stub_name = f"__wrap_{name}" if use_gnu_wrap else name
        wrap_decls.append(
            f"int32_t {real_name}(void *a0, void *a1, void *a2, void *a3, "
            f"void *a4, void *a5, void *a6, void *a7);"
        )
        wrap_defs.append(
            f"""
int32_t {stub_name}(void *a0, void *a1, void *a2, void *a3,
                    void *a4, void *a5, void *a6, void *a7) {{
  uint32_t __idx = tvm_profile_next_call_index++;
  uint64_t __t0 = tvm_profile_read_cycles();
  int32_t __status = {real_name}(a0, a1, a2, a3, a4, a5, a6, a7);
  uint64_t __t1 = tvm_profile_read_cycles();
  if (__idx < {len(call_names)}U) {{
    tvm_profile_kernel_cycles[__idx] = __t1 - __t0;
  }}
  return __status;
}}"""
        )
    wrap_decls = "\n".join(wrap_decls)

    segment_count = max(1, len(semantic_segments))
    source = f"""#include <stdint.h>

extern volatile uint64_t tohost;
extern volatile uint64_t fromhost;

static void tvm_profile_print_char(char c) {{
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

static void tvm_profile_print_str(const char* s) {{
  while (*s) tvm_profile_print_char(*s++);
}}

static void tvm_profile_print_dec(uint64_t val) {{
  char buf[21];
  int i = 20;
  buf[i] = 0;
  do {{
    buf[--i] = '0' + (val % 10);
    val /= 10;
  }} while (val && i > 0);
  tvm_profile_print_str(&buf[i]);
}}

static inline void tvm_profile_cycle_barrier(void) {{
  asm volatile ("" ::: "memory");
}}

static inline uint64_t tvm_profile_read_cycles(void) {{
  uint64_t cycles;
  tvm_profile_cycle_barrier();
  asm volatile ("rdcycle %0" : "=r" (cycles) : : "memory");
  tvm_profile_cycle_barrier();
  return cycles;
}}

static uint32_t tvm_profile_next_call_index = 0;
static uint64_t tvm_profile_kernel_cycles[{len(call_names)}] = {{0}};
static const char* tvm_profile_kernel_names[{len(call_names)}] = {{
{kernel_names_initializer}
}};
static uint64_t tvm_profile_segment_cycles[{segment_count}] = {{0}};
static const char* tvm_profile_segment_layers[{segment_count}] = {{
{segment_layer_initializer if semantic_segments else '  ""'}
}};
static const char* tvm_profile_segment_components[{segment_count}] = {{
{segment_component_initializer if semantic_segments else '  ""'}
}};
static const uint32_t tvm_profile_segment_start_calls[{segment_count}] = {{
  {segment_start_initializer if semantic_segments else '0'}
}};
static const uint32_t tvm_profile_segment_end_calls[{segment_count}] = {{
  {segment_end_initializer if semantic_segments else '0'}
}};
static const uint32_t tvm_profile_segment_count = {len(semantic_segments)};
static const uint32_t tvm_profile_kernel_count = {len(call_names)};

{wrap_decls}
{''.join(wrap_defs)}

void tvm_profile_dump_llvm_kernel_cycles(void) {{
  for (uint32_t call_index = 0; call_index < tvm_profile_kernel_count; ++call_index) {{
    tvm_profile_print_str("[TVM_KERNEL_CYCLES],");
    tvm_profile_print_dec(call_index + 1);
    tvm_profile_print_str(",");
    tvm_profile_print_str(tvm_profile_kernel_names[call_index]);
    tvm_profile_print_str(",");
    tvm_profile_print_dec(tvm_profile_kernel_cycles[call_index]);
    tvm_profile_print_str("\\n");
  }}

  for (uint32_t segment_index = 0; segment_index < tvm_profile_segment_count; ++segment_index) {{
    uint64_t cycles = 0;
    uint32_t start_call = tvm_profile_segment_start_calls[segment_index];
    uint32_t end_call = tvm_profile_segment_end_calls[segment_index];
    if (start_call >= 1 && end_call >= start_call && end_call <= tvm_profile_kernel_count) {{
      for (uint32_t call_index = start_call - 1; call_index < end_call; ++call_index) {{
        cycles += tvm_profile_kernel_cycles[call_index];
      }}
    }}
    tvm_profile_segment_cycles[segment_index] = cycles;
    tvm_profile_print_str("[TVM_SEMANTIC_CYCLES],");
    tvm_profile_print_dec(segment_index + 1);
    tvm_profile_print_str(",");
    tvm_profile_print_str(tvm_profile_segment_layers[segment_index]);
    tvm_profile_print_str(",");
    tvm_profile_print_str(tvm_profile_segment_components[segment_index]);
    tvm_profile_print_str(",");
    tvm_profile_print_dec(start_call);
    tvm_profile_print_str(",");
    tvm_profile_print_dec(end_call);
    tvm_profile_print_str(",");
    tvm_profile_print_dec(cycles);
    tvm_profile_print_str("\\n");
  }}
}}
"""
    wrap_path = output_dir / "codegen" / "host" / "src" / "llvm_kernel_profile_wraps.c"
    wrap_path.write_text(source)
    wrap_flags = [f"-Wl,--wrap={name}" for name in gemmini_symbols]
    print(
        f"[Info] Generated llvm-gemmini kernel profile wraps: {wrap_path} "
        f"({len(call_names)} calls, fused_interpose={len(fused_symbols)}, "
        f"gemmini_wrap={len(gemmini_symbols)})"
    )
    return wrap_path, wrap_flags, call_names


def instrument_llvm_aot_kernel_cycle_dump(output_dir):
    """Ask llvm shim to dump wrap-based kernel/semantic cycles after tvm_main."""
    shim_path = output_dir / "codegen" / "host" / "src" / "llvm_aot_shim.c"
    content = shim_path.read_text()
    if "tvm_profile_dump_llvm_kernel_cycles" in content:
        return

    if "tvm_profile_dump_main_cycles();" not in content:
        raise RuntimeError(
            "llvm_aot_shim.c missing main cycle dump; call instrument_llvm_aot_main_total_cycles first"
        )

    decl = "\nvoid tvm_profile_dump_llvm_kernel_cycles(void);\n"
    include_token = '#include "tvmgen_default.h"\n'
    if include_token not in content:
        raise RuntimeError(f"Could not find tvmgen_default include in {shim_path}")
    content = content.replace(include_token, include_token + decl, 1)
    content = content.replace(
        "tvm_profile_dump_main_cycles();",
        "tvm_profile_dump_llvm_kernel_cycles();\n  tvm_profile_dump_main_cycles();",
        1,
    )
    shim_path.write_text(content)
    print("[Fix] Instrumented llvm-gemmini kernel/semantic cycle dump in shim")


def _extract_tvm_main_call_names_from_content(content):
    main_pos = content.find("TVM_DLL int32_t tvmgen_default___tvm_main__(")
    if main_pos < 0:
        raise RuntimeError("Could not locate tvm main body in generated source")

    body = content[main_pos:]
    names = re.findall(r"if \((tvmgen_default_fused_[A-Za-z0-9_]+)\(", body)
    if names:
        return names

    names = re.findall(r"__tvm_status = (tvmgen_default_fused_[A-Za-z0-9_]+)\(", body)
    if names:
        return names

    raise RuntimeError("Could not extract TVM kernel call order from generated source")



def _build_standalone_layer_ranges(call_names, model_name, debug_unit):
    if not debug_unit or not call_names:
        return None

    if model_name.startswith("deit_"):
        if debug_unit in (
            "only_embed",
            "only_embedding",
            "post_patch_embed",
            "post_concat",
            "post_pos_quant",
            "post_addpos",
        ):
            return [{"layer": "patch_embed", "start_call": 1, "end_call": len(call_names)}]
        block_match = re.match(r"^only_block(\d+)$", debug_unit)
        if block_match:
            return [
                {
                    "layer": f"block_{int(block_match.group(1))}",
                    "start_call": 1,
                    "end_call": len(call_names),
                }
            ]
        if debug_unit in (
            "only_head",
            "only_classifier",
            "classifier_only",
            "pre_head",
            "pre_head_req",
            "head_int",
            "head_float",
        ):
            return [{"layer": "head", "start_call": 1, "end_call": len(call_names)}]

    if model_name.startswith("swin_"):
        if debug_unit in (
            "only_embed",
            "only_embedding",
            "post_patch_embed_proj",
            "post_patch_embed_norm",
            "post_patch_embed_out",
            "post_stem",
        ):
            return [{"layer": "patch_embed", "start_call": 1, "end_call": len(call_names)}]
        stage_block_match = re.match(r"^only_stage(\d+)_block(\d+)$", debug_unit)
        if stage_block_match:
            return [
                {
                    "layer": f"stage{int(stage_block_match.group(1))}_block{int(stage_block_match.group(2))}",
                    "start_call": 1,
                    "end_call": len(call_names),
                }
            ]
        downsample_match = re.match(r"^only_stage(\d+)_downsample$", debug_unit)
        if downsample_match:
            return [
                {
                    "layer": f"stage{int(downsample_match.group(1))}_downsample",
                    "start_call": 1,
                    "end_call": len(call_names),
                }
            ]
        global_block_match = re.match(r"^only_block(\d+)$", debug_unit)
        if global_block_match and model_name == "swin_tiny_patch4_window7_224":
            block_idx = int(global_block_match.group(1))
            depths = (2, 2, 6, 2)
            running = 0
            for stage_idx, stage_depth in enumerate(depths):
                if block_idx < running + stage_depth:
                    return [
                        {
                            "layer": f"stage{stage_idx}_block{block_idx - running}",
                            "start_call": 1,
                            "end_call": len(call_names),
                        }
                    ]
                running += stage_depth
        if debug_unit in (
            "only_head",
            "only_classifier",
            "classifier_only",
            "pre_head_float",
            "pre_head",
            "head_int",
            "head_float",
        ):
            return [{"layer": "head", "start_call": 1, "end_call": len(call_names)}]

    return None


def build_tvm_layer_ranges_from_call_names(call_names, model_name, debug_unit=None):
    standalone_ranges = _build_standalone_layer_ranges(call_names, model_name, debug_unit)
    if standalone_ranges is not None:
        return standalone_ranges
    if model_name.startswith("deit_"):
        return _build_deit_layer_ranges(call_names)
    if model_name.startswith("swin_"):
        return _build_swin_layer_ranges(call_names)
    raise RuntimeError(f"Unsupported TVM layer profiling model: {model_name}")


def _build_tvm_semantic_segments_from_call_names(call_names, model_name, debug_unit=None):
    return _build_tvm_segments_from_call_names(
        call_names, model_name, style="semantic", debug_unit=debug_unit
    )


def _build_tvm_aligned_segments_from_call_names(call_names, model_name, debug_unit=None):
    return _build_tvm_segments_from_call_names(
        call_names, model_name, style="aligned", debug_unit=debug_unit
    )


def _build_tvm_aligned_split_segments_from_call_names(call_names, model_name, debug_unit=None):
    return _build_tvm_segments_from_call_names(
        call_names, model_name, style="aligned_split", debug_unit=debug_unit
    )


def _build_tvm_segments_for_style(call_names, model_name, style, debug_unit=None):
    if style == "semantic":
        return _build_tvm_semantic_segments_from_call_names(
            call_names, model_name, debug_unit=debug_unit
        )
    if style == "aligned":
        return _build_tvm_aligned_segments_from_call_names(
            call_names, model_name, debug_unit=debug_unit
        )
    if style == "aligned_split":
        return _build_tvm_aligned_split_segments_from_call_names(
            call_names, model_name, debug_unit=debug_unit
        )
    raise RuntimeError(f"Unsupported TVM segment style: {style}")


def _c_string_array_initializer(items):
    if items:
        return ",\n".join(f'  "{item}"' for item in items)
    return '  ""'


def _build_tvm_segments_from_call_names(call_names, model_name, style, debug_unit=None):
    layer_ranges = build_tvm_layer_ranges_from_call_names(
        call_names, model_name, debug_unit=debug_unit
    )
    segments = []
    component_rows = []

    for layer in layer_ranges:
        layer_name = layer["layer"]
        start_call = layer["start_call"]
        end_call = layer["end_call"]
        local_call_names = call_names[start_call - 1 : end_call]

        if style in ("aligned", "aligned_split"):
            raw_segments = _parse_aligned_component_segments(
                layer_name,
                model_name,
                local_call_names,
                split_post_ops=(style == "aligned_split"),
            )
        else:
            if layer_name == "patch_embed":
                raw_segments = _parse_patch_embed_component_segments(local_call_names)
            elif layer_name.endswith("downsample"):
                raw_segments = _parse_downsample_component_segments(local_call_names)
            elif layer_name == "head":
                raw_segments = _parse_head_component_segments(local_call_names)
            else:
                raw_segments = _parse_transformer_block_component_segments(local_call_names)

        assigned = {}
        for segment in raw_segments:
            for local_idx in range(segment["start"], segment["end"] + 1):
                if local_idx in assigned:
                    raise RuntimeError(
                        f"Overlapping TVM component mapping in {layer_name} at local call {local_idx + 1}"
                    )
                assigned[local_idx] = segment["component"]

        current_component = None
        current_start_local = None
        for local_idx, kernel_name in enumerate(local_call_names):
            if style in ("aligned", "aligned_split"):
                fallback_component = _fallback_tvm_aligned_component_name(
                    layer_name, kernel_name
                )
            else:
                fallback_component = _fallback_tvm_component_name(
                    layer_name, kernel_name
                )
            component = assigned.get(local_idx, fallback_component)
            component_rows.append(
                {
                    "call_index": start_call + local_idx,
                    "layer": layer_name,
                    "component": component,
                    "kernel_name": kernel_name,
                }
            )
            if current_component is None:
                current_component = component
                current_start_local = local_idx
                continue
            if component == current_component:
                continue
            segments.append(
                {
                    "segment_index": len(segments) + 1,
                    "layer": layer_name,
                    "component": current_component,
                    "start_call": start_call + current_start_local,
                    "end_call": start_call + local_idx - 1,
                }
            )
            current_component = component
            current_start_local = local_idx

        if current_component is not None:
            segments.append(
                {
                    "segment_index": len(segments) + 1,
                    "layer": layer_name,
                    "component": current_component,
                    "start_call": start_call + current_start_local,
                    "end_call": end_call,
                }
            )

    covered_calls = {row["call_index"] for row in component_rows}
    expected_calls = set(range(1, len(call_names) + 1))
    if covered_calls != expected_calls:
        missing_calls = sorted(expected_calls - covered_calls)
        raise RuntimeError(
            f"TVM semantic mapping did not cover all calls, first missing calls: {missing_calls[:10]}"
        )

    return layer_ranges, segments, component_rows


def _instrument_tvm_segment_cycles(
    output_dir,
    model_name,
    style="semantic",
    layer_filters=None,
    component_filters=None,
    debug_unit=None,
):
    """Wrap generated TVM callsites with rdcycle segment profiling only."""
    tvm_main_path = _find_tvm_main_source(output_dir)
    with open(tvm_main_path, "r") as f:
        content = f.read()

    if "tvm_profile_segment_cycles[" in content and "tvm_profile_dump_semantic_cycles" in content:
        print("[Fix] TVM segment profiling already present")
        return

    original_call_names = _extract_tvm_main_call_names_from_content(content)
    _, profiled_segments, _ = _build_tvm_segments_for_style(
        original_call_names, model_name, style, debug_unit=debug_unit
    )
    segment_starts = defaultdict(list)
    segment_ends = defaultdict(list)
    for segment_idx, segment in enumerate(profiled_segments):
        segment_starts[segment["start_call"]].append(segment_idx)
        segment_ends[segment["end_call"]].append(segment_idx)

    call_re = re.compile(
        r"^(\s*)if \((tvmgen_default_fused_[A-Za-z0-9_]+)\((.*)\)\s*!=\s*0\s*\)\s*return -1;$",
        re.MULTILINE,
    )

    call_index = 0

    def _replace(match):
        nonlocal call_index
        call_index += 1
        indent, func_name, args = match.groups()
        start_ids = segment_starts.get(call_index, [])
        end_ids = segment_ends.get(call_index, [])

        before_code = f"{indent}  uint64_t __tvm_boundary_before = tvm_profile_read_cycles();\n"
        for segment_idx in start_ids:
            before_code += (
                f"{indent}  tvm_profile_segment_start[{segment_idx}] = __tvm_boundary_before;\n"
            )

        after_code = f"{indent}  uint64_t __tvm_boundary_after = tvm_profile_read_cycles();\n"
        for segment_idx in end_ids:
            after_code += (
                f"{indent}  tvm_profile_segment_cycles[{segment_idx}] += "
                f"__tvm_boundary_after - tvm_profile_segment_start[{segment_idx}];\n"
            )

        return (
            f"{indent}{{\n"
            f"{before_code}"
            f"{indent}  int32_t __tvm_status = {func_name}({args});\n"
            f"{after_code}"
            f"{indent}  if (__tvm_status != 0 ) return -1;\n"
            f"{indent}}}"
        )

    updated = call_re.sub(_replace, content)
    if call_index == 0:
        raise RuntimeError(f"No TVM kernel callsites found in {tvm_main_path}")

    layer_filters = list(layer_filters or [])
    component_filters = list(component_filters or [])
    layer_filter_count = len(layer_filters)
    component_filter_count = len(component_filters)
    layer_filter_array_len = max(1, layer_filter_count)
    component_filter_array_len = max(1, component_filter_count)

    segment_layer_initializer = ",\n".join(f'  "{segment["layer"]}"' for segment in profiled_segments)
    segment_component_initializer = ",\n".join(
        f'  "{segment["component"]}"' for segment in profiled_segments
    )
    segment_start_initializer = ", ".join(
        str(segment["start_call"]) for segment in profiled_segments
    )
    segment_end_initializer = ", ".join(
        str(segment["end_call"]) for segment in profiled_segments
    )

    support_code = """
extern volatile uint64_t tohost;
extern volatile uint64_t fromhost;

static void tvm_profile_print_char(char c) {{
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

static void tvm_profile_print_str(const char* s) {{
  while (*s) tvm_profile_print_char(*s++);
}}

static void tvm_profile_print_dec(uint64_t val) {{
  char buf[21];
  int i = 20;
  buf[i] = 0;
  do {{
    buf[--i] = '0' + (val % 10);
    val /= 10;
  }} while (val && i > 0);
  tvm_profile_print_str(&buf[i]);
}}

static inline void tvm_profile_cycle_barrier(void) {{
  asm volatile ("" ::: "memory");
}}

static inline uint64_t tvm_profile_read_cycles(void) {{
  uint64_t cycles;
  tvm_profile_cycle_barrier();
  asm volatile ("rdcycle %0" : "=r" (cycles) : : "memory");
  tvm_profile_cycle_barrier();
  return cycles;
}}

static int tvm_profile_str_eq(const char* lhs, const char* rhs) {{
  while (*lhs && *rhs) {{
    if (*lhs != *rhs) return 0;
    ++lhs;
    ++rhs;
  }}
  return (*lhs == 0) && (*rhs == 0);
}}

static uint64_t tvm_profile_segment_cycles[{segment_count}] = {{0}};
static uint64_t tvm_profile_segment_start[{segment_count}] = {{0}};
static const char* tvm_profile_segment_layers[{segment_count}] = {{
{segment_layer_initializer}
}};
static const char* tvm_profile_segment_components[{segment_count}] = {{
{segment_component_initializer}
}};
static const uint32_t tvm_profile_segment_start_calls[{segment_count}] = {{
  {segment_start_initializer}
}};
static const uint32_t tvm_profile_segment_end_calls[{segment_count}] = {{
  {segment_end_initializer}
}};
static const char* tvm_profile_layer_filters[{layer_filter_array_len}] = {{
{layer_filter_initializer}
}};
static const char* tvm_profile_component_filters[{component_filter_array_len}] = {{
{component_filter_initializer}
}};
static const uint32_t tvm_profile_layer_filter_count = {layer_filter_count};
static const uint32_t tvm_profile_component_filter_count = {component_filter_count};
static uint64_t tvm_profile_main_cycles = 0;

static int tvm_profile_layer_allowed(const char* layer) {{
  if (tvm_profile_layer_filter_count == 0) return 1;
  for (uint32_t i = 0; i < tvm_profile_layer_filter_count; ++i) {{
    if (tvm_profile_str_eq(layer, tvm_profile_layer_filters[i])) return 1;
  }}
  return 0;
}}

static int tvm_profile_component_allowed(const char* component) {{
  if (tvm_profile_component_filter_count == 0) return 1;
  for (uint32_t i = 0; i < tvm_profile_component_filter_count; ++i) {{
    if (tvm_profile_str_eq(component, tvm_profile_component_filters[i])) return 1;
  }}
  return 0;
}}

static void tvm_profile_dump_semantic_cycles(void) {{
  for (uint64_t segment_index = 0; segment_index < {segment_count}ULL; ++segment_index) {{
    if (!tvm_profile_layer_allowed(tvm_profile_segment_layers[segment_index])) continue;
    if (!tvm_profile_component_allowed(tvm_profile_segment_components[segment_index])) continue;
    tvm_profile_print_str("[TVM_SEMANTIC_CYCLES],");
    tvm_profile_print_dec(segment_index + 1);
    tvm_profile_print_str(",");
    tvm_profile_print_str(tvm_profile_segment_layers[segment_index]);
    tvm_profile_print_str(",");
    tvm_profile_print_str(tvm_profile_segment_components[segment_index]);
    tvm_profile_print_str(",");
    tvm_profile_print_dec(tvm_profile_segment_start_calls[segment_index]);
    tvm_profile_print_str(",");
    tvm_profile_print_dec(tvm_profile_segment_end_calls[segment_index]);
    tvm_profile_print_str(",");
    tvm_profile_print_dec(tvm_profile_segment_cycles[segment_index]);
    tvm_profile_print_str("\\n");
  }}
}}

static void tvm_profile_dump_main_cycles(void) {{
  tvm_profile_print_str("{inner_cycles_prefix}");
  tvm_profile_print_dec(tvm_profile_main_cycles);
  tvm_profile_print_str("\\n");
}}

"""
    support_code = support_code.format(
        segment_count=len(profiled_segments),
        segment_layer_initializer=segment_layer_initializer,
        segment_component_initializer=segment_component_initializer,
        segment_start_initializer=segment_start_initializer,
        segment_end_initializer=segment_end_initializer,
        layer_filter_array_len=layer_filter_array_len,
        component_filter_array_len=component_filter_array_len,
        layer_filter_initializer=_c_string_array_initializer(layer_filters),
        component_filter_initializer=_c_string_array_initializer(component_filters),
        layer_filter_count=layer_filter_count,
        component_filter_count=component_filter_count,
        inner_cycles_prefix=TVM_MAIN_INNER_CYCLES_PREFIX,
    )
    updated = updated.replace(
        '#include "tvm/runtime/c_runtime_api.h"\n',
        '#include "tvm/runtime/c_runtime_api.h"\n' + support_code,
        1,
    )

    main_fn_re = re.compile(
        r"(int32_t tvmgen_default___tvm_main__\([^)]*\)\s*\{.*?)(\n\s*return 0;\n\})",
        re.DOTALL,
    )

    def _inject_dump(match):
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
            "\n  tvm_profile_dump_semantic_cycles();"
            "\n  tvm_profile_dump_main_cycles();"
            + tail
        )

    updated, replaced = main_fn_re.subn(_inject_dump, updated, count=1)
    if replaced != 1:
        raise RuntimeError("Failed to inject TVM segment dump into tvm main")

    with open(tvm_main_path, "w") as f:
        f.write(updated)
    print(
        f"[Fix] Instrumented {len(profiled_segments)} TVM {style} segments "
        f"(layer filters={layer_filters or ['*']}, component filters={component_filters or ['*']})"
    )


def _instrument_spike_kernel_cycles(output_dir, model_name, debug_unit=None):
    """Wrap generated TVM callsites with kernel and semantic rdcycle profiling."""
    tvm_main_path = _find_tvm_main_source(output_dir)
    with open(tvm_main_path, "r") as f:
        content = f.read()

    if "tvm_profile_kernel_cycles[" in content and "tvm_profile_dump_semantic_cycles" in content:
        print("[Fix] TVM Spike profiling already uses semantic rdcycle buffering")
        return
    if "tvm_profile_emit_kernel_cycles" in content:
        raise RuntimeError(
            "Legacy inline TVM Spike profiling is present. Re-extract model.tar before reinstrumenting."
        )

    original_call_names = _extract_tvm_main_call_names_from_content(content)
    _, semantic_segments, _ = _build_tvm_semantic_segments_from_call_names(
        original_call_names, model_name, debug_unit=debug_unit
    )
    segment_starts = defaultdict(list)
    segment_ends = defaultdict(list)
    for segment_idx, segment in enumerate(semantic_segments):
        segment_starts[segment["start_call"]].append(segment_idx)
        segment_ends[segment["end_call"]].append(segment_idx)

    call_re = re.compile(
        r"^(\s*)if \((tvmgen_default_fused_[A-Za-z0-9_]+)\((.*)\)\s*!=\s*0\s*\)\s*return -1;$",
        re.MULTILINE,
    )
    call_index = 0
    call_names = []

    def _replace(match):
        nonlocal call_index
        call_index += 1
        indent, func_name, args = match.groups()
        call_names.append(func_name)
        start_ids = segment_starts.get(call_index, [])
        end_ids = segment_ends.get(call_index, [])
        before_code = ""
        if start_ids:
            before_code += f"{indent}  uint64_t __tvm_boundary_before = tvm_profile_read_cycles();\n"
            for segment_idx in start_ids:
                before_code += (
                    f"{indent}  tvm_profile_segment_start[{segment_idx}] = __tvm_boundary_before;\n"
                )
        else:
            before_code += f"{indent}  uint64_t __tvm_boundary_before = tvm_profile_read_cycles();\n"

        after_code = f"{indent}  uint64_t __tvm_boundary_after = tvm_profile_read_cycles();\n"
        for segment_idx in end_ids:
            after_code += (
                f"{indent}  tvm_profile_segment_cycles[{segment_idx}] += "
                f"__tvm_boundary_after - tvm_profile_segment_start[{segment_idx}];\n"
            )
        return (
            f"{indent}{{\n"
            f"{before_code}"
            f"{indent}  int32_t __tvm_status = {func_name}({args});\n"
            f"{after_code}"
            f"{indent}  tvm_profile_kernel_cycles[{call_index - 1}] = "
            f"__tvm_boundary_after - __tvm_boundary_before;\n"
            f"{indent}  if (__tvm_status != 0 ) return -1;\n"
            f"{indent}}}"
        )

    updated = call_re.sub(_replace, content)
    if call_index == 0:
        raise RuntimeError(f"No TVM kernel callsites found in {tvm_main_path}")

    kernel_names_initializer = ",\n".join(f'  "{name}"' for name in call_names)
    segment_layer_initializer = ",\n".join(f'  "{segment["layer"]}"' for segment in semantic_segments)
    segment_component_initializer = ",\n".join(
        f'  "{segment["component"]}"' for segment in semantic_segments
    )
    segment_start_initializer = ", ".join(
        str(segment["start_call"]) for segment in semantic_segments
    )
    segment_end_initializer = ", ".join(str(segment["end_call"]) for segment in semantic_segments)
    support_code = """
extern volatile uint64_t tohost;
extern volatile uint64_t fromhost;

static void tvm_profile_print_char(char c) {{
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

static void tvm_profile_print_str(const char* s) {{
  while (*s) tvm_profile_print_char(*s++);
}}

static void tvm_profile_print_dec(uint64_t val) {{
  char buf[21];
  int i = 20;
  buf[i] = 0;
  do {{
    buf[--i] = '0' + (val % 10);
    val /= 10;
  }} while (val && i > 0);
  tvm_profile_print_str(&buf[i]);
}}

static inline void tvm_profile_cycle_barrier(void) {{
  asm volatile ("" ::: "memory");
}}

static inline uint64_t tvm_profile_read_cycles(void) {{
  uint64_t cycles;
  tvm_profile_cycle_barrier();
  asm volatile ("rdcycle %0" : "=r" (cycles) : : "memory");
  tvm_profile_cycle_barrier();
  return cycles;
}}

static uint64_t tvm_profile_kernel_cycles[{call_index}] = {{0}};
static const char* tvm_profile_kernel_names[{call_index}] = {{
{kernel_names_initializer}
}};
static uint64_t tvm_profile_segment_cycles[{segment_count}] = {{0}};
static uint64_t tvm_profile_segment_start[{segment_count}] = {{0}};
static const char* tvm_profile_segment_layers[{segment_count}] = {{
{segment_layer_initializer}
}};
static const char* tvm_profile_segment_components[{segment_count}] = {{
{segment_component_initializer}
}};
static const uint32_t tvm_profile_segment_start_calls[{segment_count}] = {{
  {segment_start_initializer}
}};
static const uint32_t tvm_profile_segment_end_calls[{segment_count}] = {{
  {segment_end_initializer}
}};
static uint64_t tvm_profile_main_cycles = 0;

static void tvm_profile_dump_kernel_cycles(void) {{
  for (uint64_t call_index = 0; call_index < {call_index}ULL; ++call_index) {{
    tvm_profile_print_str("[TVM_KERNEL_CYCLES],");
    tvm_profile_print_dec(call_index + 1);
    tvm_profile_print_str(",");
    tvm_profile_print_str(tvm_profile_kernel_names[call_index]);
    tvm_profile_print_str(",");
    tvm_profile_print_dec(tvm_profile_kernel_cycles[call_index]);
    tvm_profile_print_str("\\n");
  }}
}}

static void tvm_profile_dump_semantic_cycles(void) {{
  for (uint64_t segment_index = 0; segment_index < {segment_count}ULL; ++segment_index) {{
    tvm_profile_print_str("[TVM_SEMANTIC_CYCLES],");
    tvm_profile_print_dec(segment_index + 1);
    tvm_profile_print_str(",");
    tvm_profile_print_str(tvm_profile_segment_layers[segment_index]);
    tvm_profile_print_str(",");
    tvm_profile_print_str(tvm_profile_segment_components[segment_index]);
    tvm_profile_print_str(",");
    tvm_profile_print_dec(tvm_profile_segment_start_calls[segment_index]);
    tvm_profile_print_str(",");
    tvm_profile_print_dec(tvm_profile_segment_end_calls[segment_index]);
    tvm_profile_print_str(",");
    tvm_profile_print_dec(tvm_profile_segment_cycles[segment_index]);
    tvm_profile_print_str("\\n");
  }}
}}

static void tvm_profile_dump_main_cycles(void) {{
  tvm_profile_print_str("{inner_cycles_prefix}");
  tvm_profile_print_dec(tvm_profile_main_cycles);
  tvm_profile_print_str("\\n");
}}

"""
    support_code = support_code.format(
        call_index=call_index,
        kernel_names_initializer=kernel_names_initializer,
        segment_count=len(semantic_segments),
        segment_layer_initializer=segment_layer_initializer,
        segment_component_initializer=segment_component_initializer,
        segment_start_initializer=segment_start_initializer,
        segment_end_initializer=segment_end_initializer,
        inner_cycles_prefix=TVM_MAIN_INNER_CYCLES_PREFIX,
    )
    updated = updated.replace(
        '#include "tvm/runtime/c_runtime_api.h"\n',
        '#include "tvm/runtime/c_runtime_api.h"\n' + support_code,
        1,
    )

    main_fn_re = re.compile(
        r"(int32_t tvmgen_default___tvm_main__\([^)]*\)\s*\{.*?)(\n\s*return 0;\n\})",
        re.DOTALL,
    )

    def _inject_dump(match):
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
            "\n  tvm_profile_dump_kernel_cycles();"
            "\n  tvm_profile_dump_semantic_cycles();"
            "\n  tvm_profile_dump_main_cycles();"
            + tail
        )

    updated, replaced = main_fn_re.subn(_inject_dump, updated, count=1)
    if replaced != 1:
        raise RuntimeError("Failed to inject TVM Spike kernel dump into tvm main")

    with open(tvm_main_path, "w") as f:
        f.write(updated)
    print(
        f"[Fix] Instrumented {call_index} TVM callsites and "
        f"{len(semantic_segments)} semantic segments for Spike rdcycle profiling"
    )


def _instrument_spike_intrakernel_head(output_dir):
    tvm_main_path = _find_tvm_main_source(output_dir)
    with open(tvm_main_path, "r") as f:
        content = f.read()

    if "[TVM_INTRAKERNEL_CYCLES],head_pack," in content:
        print("[Fix] TVM intrakernel head profiling already present")
        return

    function_names = [
        "tvmgen_default_fused_nn_dense_add_cast_multiply",
        "tvmgen_default_fused_nn_dense_nn_bias_add_cast_multiply",
    ]
    updated = content
    patched = 0

    for function_name in function_names:
        fn_pos = updated.find(f"TVM_DLL int32_t {function_name}(")
        if fn_pos < 0:
            continue
        next_fn_pos = updated.find("\nTVM_DLL int32_t ", fn_pos + 1)
        if next_fn_pos < 0:
            next_fn_pos = len(updated)
        fn_text = updated[fn_pos:next_fn_pos]
        original_fn_text = fn_text

        pack_start = "  for (int32_t z = 0; z < 125; ++z) {\n"
        pack_end = (
            "\n  }\n  for (int32_t ax1_outer_ax0_outer_fused = 0; "
            "ax1_outer_ax0_outer_fused < 125; ++ax1_outer_ax0_outer_fused) {\n"
        )
        if pack_start not in fn_text or pack_end not in fn_text:
            raise RuntimeError(f"Failed to locate head pack loop in {function_name}")
        fn_text = fn_text.replace(
            pack_start,
            "  uint64_t __tvm_head_pack_start = tvm_profile_read_cycles();\n" + pack_start,
            1,
        )
        fn_text = fn_text.replace(
            pack_end,
            "\n  }\n"
            "  uint64_t __tvm_head_pack_cycles = tvm_profile_read_cycles() - __tvm_head_pack_start;\n"
            "  uint64_t __tvm_head_mm_cycles = 0;\n"
            "  uint64_t __tvm_head_post_cycles = 0;\n"
            "  for (int32_t ax1_outer_ax0_outer_fused = 0; ax1_outer_ax0_outer_fused < 125; ++ax1_outer_ax0_outer_fused) {\n",
            1,
        )

        compute_start = "    void* compute_global_let = ("
        post_loop = "    for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 8; ++ax1_inner_inner) {\n"
        fn_end = "    }\n  }\n  return 0;\n"
        if compute_start not in fn_text or post_loop not in fn_text or fn_end not in fn_text:
            raise RuntimeError(f"Failed to locate head compute loops in {function_name}")
        fn_text = fn_text.replace(
            compute_start,
            "    uint64_t __tvm_head_mm_start = tvm_profile_read_cycles();\n" + compute_start,
            1,
        )
        fn_text = fn_text.replace(
            post_loop,
            "    __tvm_head_mm_cycles += tvm_profile_read_cycles() - __tvm_head_mm_start;\n"
            "    uint64_t __tvm_head_post_start = tvm_profile_read_cycles();\n"
            + post_loop,
            1,
        )
        fn_text = fn_text.replace(
            fn_end,
            "    }\n"
            "    __tvm_head_post_cycles += tvm_profile_read_cycles() - __tvm_head_post_start;\n"
            "  }\n"
            '  tvm_profile_print_str("[TVM_INTRAKERNEL_CYCLES],head_pack,");\n'
            "  tvm_profile_print_dec(__tvm_head_pack_cycles);\n"
            '  tvm_profile_print_char(\'\\n\');\n'
            '  tvm_profile_print_str("[TVM_INTRAKERNEL_CYCLES],head_mm,");\n'
            "  tvm_profile_print_dec(__tvm_head_mm_cycles);\n"
            '  tvm_profile_print_char(\'\\n\');\n'
            '  tvm_profile_print_str("[TVM_INTRAKERNEL_CYCLES],head_post,");\n'
            "  tvm_profile_print_dec(__tvm_head_post_cycles);\n"
            '  tvm_profile_print_char(\'\\n\');\n'
            "  return 0;\n",
            1,
        )

        if fn_text == original_fn_text:
            continue
        updated = updated[:fn_pos] + fn_text + updated[next_fn_pos:]
        patched += 1

    if patched == 0:
        print("[Fix] No generic dense head kernel found for intrakernel profiling")
        return

    with open(tvm_main_path, "w") as f:
        f.write(updated)
    print(f"[Fix] Instrumented {patched} generic head kernel(s) for intrakernel profiling")


def _instrument_intrakernel_requant_kernels(output_dir):
    tvm_main_path = _find_tvm_main_source(output_dir)
    with open(tvm_main_path, "r") as f:
        content = f.read()

    call_names = _extract_tvm_main_call_names_from_content(content)
    profiled_kernels = []
    seen = set()
    for function_name in call_names:
        component = _classify_intrakernel_quant_kernel(function_name)
        if component is None:
            continue
        if function_name in seen:
            continue
        fn_pos = content.find(f"TVM_DLL int32_t {function_name}(")
        if fn_pos < 0:
            continue
        next_fn_pos = content.find("\nTVM_DLL int32_t ", fn_pos + 1)
        if next_fn_pos < 0:
            next_fn_pos = len(content)
        fn_text = content[fn_pos:next_fn_pos]
        if "gemmini_" in fn_text:
            continue
        seen.add(function_name)
        profiled_kernels.append((component, function_name))

    if not profiled_kernels:
        print("[Fix] No CPU quant/post-op kernels found for intrakernel profiling")
        return

    updated = content
    registry_marker = "static uint64_t tvm_profile_intrakernel_cycles["
    if registry_marker not in updated:
        kernel_components_initializer = ",\n".join(
            f'  "{component}"' for component, _ in profiled_kernels
        )
        kernel_names_initializer = ",\n".join(
            f'  "{name}"' for _, name in profiled_kernels
        )
        support_code = f"""
static void tvm_profile_print_char(char c);
static void tvm_profile_print_str(const char* s);
static void tvm_profile_print_dec(uint64_t val);
static inline uint64_t tvm_profile_read_cycles(void);

static uint64_t tvm_profile_intrakernel_cycles[{len(profiled_kernels)}] = {{0}};
static const char* tvm_profile_intrakernel_kinds[{len(profiled_kernels)}] = {{
{kernel_components_initializer}
}};
static const char* tvm_profile_intrakernel_labels[{len(profiled_kernels)}] = {{
{kernel_names_initializer}
}};

static void tvm_profile_dump_intrakernel_cycles(void) {{
  for (uint64_t intrakernel_index = 0; intrakernel_index < {len(profiled_kernels)}ULL; ++intrakernel_index) {{
    tvm_profile_print_str("[TVM_INTRAKERNEL_CYCLES],");
    tvm_profile_print_str(tvm_profile_intrakernel_kinds[intrakernel_index]);
    tvm_profile_print_str(",");
    tvm_profile_print_str(tvm_profile_intrakernel_labels[intrakernel_index]);
    tvm_profile_print_str(",");
    tvm_profile_print_dec(tvm_profile_intrakernel_cycles[intrakernel_index]);
    tvm_profile_print_char('\\n');
  }}
}}

"""
        updated = updated.replace(
            '#include "tvm/runtime/c_runtime_api.h"\n',
            '#include "tvm/runtime/c_runtime_api.h"\n' + support_code,
            1,
        )

    patched = 0

    for kernel_index, (_, function_name) in enumerate(profiled_kernels):
        fn_pos = updated.find(f"TVM_DLL int32_t {function_name}(")
        if fn_pos < 0:
            continue
        next_fn_pos = updated.find("\nTVM_DLL int32_t ", fn_pos + 1)
        if next_fn_pos < 0:
            next_fn_pos = len(updated)
        fn_text = updated[fn_pos:next_fn_pos]
        original_fn_text = fn_text

        body_open = fn_text.find("{")
        if body_open < 0:
            raise RuntimeError(f"Failed to locate function body for {function_name}")
        if "__tvm_requant_kernel_start" in fn_text:
            continue
        fn_text = (
            fn_text[: body_open + 1]
            + "\n  uint64_t __tvm_requant_kernel_start = tvm_profile_read_cycles();"
            + fn_text[body_open + 1 :]
        )

        return_stmt = "\n  return 0;\n"
        ret_pos = fn_text.rfind(return_stmt)
        if ret_pos < 0:
            raise RuntimeError(f"Failed to locate return statement for {function_name}")
        emit_code = (
            "\n  uint64_t __tvm_requant_kernel_cycles = "
            "tvm_profile_read_cycles() - __tvm_requant_kernel_start;\n"
            f"  tvm_profile_intrakernel_cycles[{kernel_index}] += __tvm_requant_kernel_cycles;\n"
            "  return 0;\n"
        )
        fn_text = fn_text[:ret_pos] + emit_code + fn_text[ret_pos + len(return_stmt) :]

        if fn_text == original_fn_text:
            continue
        updated = updated[:fn_pos] + fn_text + updated[next_fn_pos:]
        patched += 1

    if patched == 0:
        print("[Fix] Requant intrakernel profiling already present")
        return

    main_fn_re = re.compile(
        r"(int32_t tvmgen_default___tvm_main__\([^)]*\)\s*\{.*?)(\n\s*return 0;\n\})",
        re.DOTALL,
    )

    def _inject_intrakernel_dump(match):
        body, tail = match.groups()
        if "tvm_profile_dump_intrakernel_cycles();" in body:
            return match.group(0)
        return body + "\n  tvm_profile_dump_intrakernel_cycles();" + tail

    updated, replaced = main_fn_re.subn(_inject_intrakernel_dump, updated, count=1)
    if replaced != 1:
        raise RuntimeError("Failed to inject TVM intrakernel dump into tvm main")

    with open(tvm_main_path, "w") as f:
        f.write(updated)
    print(f"[Fix] Instrumented {patched} CPU quant/post-op kernel(s) for intrakernel profiling")


def _restore_codegen_from_model_tar(output_dir):
    """Restore generated code from the exported MLF before reapplying instrumentation."""
    mlf_path = output_dir / "model.tar"
    if not mlf_path.exists():
        return
    with tarfile.open(mlf_path, "r:") as tar:
        tar.extractall(output_dir)


def fix_generated_code(
    output_dir,
    model_name,
    instrument_kernel_cycles=False,
    segment_profile_style=None,
    segment_profile_layer_filters=None,
    segment_profile_component_filters=None,
    instrument_requant_intrakernel=False,
    debug_unit=None,
    skip_if_no_scalar_c=False,
    llvm_gemmini=False,
):
    """Apply generated TVM source fixes and optional TVM rdcycle instrumentation."""
    lib0_path = output_dir / "codegen" / "host" / "src" / "default_lib0.c"
    if skip_if_no_scalar_c and not lib0_path.exists():
        print("[Info] llvm-gemmini backend: skipping scalar C codegen fixes")
        return

    if instrument_kernel_cycles or segment_profile_style is not None or instrument_requant_intrakernel:
        _restore_codegen_from_model_tar(output_dir)

    with open(lib0_path, "r") as f:
        content = f.read()

    old_pattern = "&global_const_workspace,&global_workspace)"
    new_pattern = "(uint8_t*)&global_const_workspace,global_workspace)"

    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        with open(lib0_path, "w") as f:
            f.write(content)
        print("[Fix] Applied pointer type fix to generated code")

    if model_name.startswith("swin_") and not llvm_gemmini:
        # c-gemmini emits a giant C tvm_main that can overflow R_RISCV_JAL; split it.
        # llvm-gemmini emits tvm_main into an LLVM object (default_lib1.o), so no split.
        split_oversized_tvm_main_for_riscv_link(output_dir)

    if instrument_kernel_cycles and not llvm_gemmini:
        _instrument_spike_kernel_cycles(output_dir, model_name, debug_unit=debug_unit)
        _instrument_spike_intrakernel_head(output_dir)
    elif segment_profile_style is not None and not llvm_gemmini:
        _instrument_tvm_segment_cycles(
            output_dir,
            model_name,
            style=segment_profile_style,
            layer_filters=segment_profile_layer_filters,
            component_filters=segment_profile_component_filters,
            debug_unit=debug_unit,
        )
    if instrument_requant_intrakernel and not llvm_gemmini:
        _instrument_intrakernel_requant_kernels(output_dir)


def extract_spike_kernel_cycle_rows(stdout):
    rows = []
    for line in stdout.splitlines():
        if not line.startswith("[TVM_KERNEL_CYCLES],"):
            continue
        parts = line.strip().split(",", 3)
        if len(parts) != 4:
            continue
        _, call_index, kernel_name, cycles = parts
        try:
            rows.append(
                {
                    "call_index": int(call_index),
                    "kernel_name": kernel_name,
                    "cycles": int(cycles),
                }
            )
        except ValueError:
            continue
    return rows


def extract_spike_semantic_cycle_rows(stdout):
    rows = []
    for line in stdout.splitlines():
        if not line.startswith("[TVM_SEMANTIC_CYCLES],"):
            continue
        parts = line.strip().split(",", 6)
        if len(parts) != 7:
            continue
        _, segment_index, layer, component, start_call, end_call, cycles = parts
        try:
            rows.append(
                {
                    "segment_index": int(segment_index),
                    "layer": layer,
                    "component": component,
                    "start_call": int(start_call),
                    "end_call": int(end_call),
                    "cycles": int(cycles),
                }
            )
        except ValueError:
            continue
    return rows


def extract_spike_main_total_cycles(stdout):
    total_cycles = None
    for line in stdout.splitlines():
        if not line.startswith(TVM_MAIN_TOTAL_CYCLES_PREFIX):
            continue
        parts = line.strip().split(",", 1)
        if len(parts) != 2:
            continue
        try:
            total_cycles = int(parts[1])
        except ValueError:
            return None
    if total_cycles is not None:
        return total_cycles

    for line in stdout.splitlines():
        if not line.startswith("[TVM_RUN_RESULT],"):
            continue
        parts = line.strip().split(",")
        if len(parts) < 5:
            continue
        try:
            return int(parts[3])
        except ValueError:
            return None
    return None


def extract_spike_main_inner_cycles(stdout):
    inner_cycles = None
    for line in stdout.splitlines():
        if not line.startswith(TVM_MAIN_INNER_CYCLES_PREFIX):
            continue
        parts = line.strip().split(",", 1)
        if len(parts) != 2:
            continue
        try:
            inner_cycles = int(parts[1])
        except ValueError:
            return None
    return inner_cycles


def extract_spike_intrakernel_cycle_rows(stdout):
    rows = []
    for line in stdout.splitlines():
        if not line.startswith("[TVM_INTRAKERNEL_CYCLES],"):
            continue
        parts = line.strip().split(",", 3)
        try:
            if len(parts) == 3:
                _, component, cycles = parts
                rows.append({"component": component, "label": "", "cycles": int(cycles)})
            elif len(parts) == 4:
                _, component, label, cycles = parts
                rows.append({"component": component, "label": label, "cycles": int(cycles)})
        except ValueError:
            continue
    return rows


def _load_tvm_main_call_names(output_dir):
    call_order_path = pathlib.Path(output_dir) / "llvm_tvm_main_call_order.txt"
    if call_order_path.exists():
        names = []
        for line in call_order_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            if "," in line:
                names.append(line.split(",", 1)[1].strip())
            else:
                names.append(line)
        if names:
            return names

    try:
        tvm_main_path = _find_tvm_main_source(output_dir)
        with open(tvm_main_path, "r") as f:
            content = f.read()
        return _extract_tvm_main_call_names_from_content(content)
    except RuntimeError:
        host_obj = _find_llvm_host_object(output_dir)
        return _extract_llvm_tvm_main_call_names_from_object(host_obj)


def _matches_generated_name(name, base_name):
    return re.fullmatch(rf"{re.escape(base_name)}(?:_\d+)?", name) is not None


def _is_repq_layernorm_tail_kernel(name):
    return name.startswith("tvmgen_default_fused_add_rsqrt_multiply_multiply_add")


def _is_repq_graph_call_names(call_names):
    return any("tvmgen_default_fused_multiply_mean" in name for name in call_names)


def _layernorm_sequence_length(call_names, start_index):
    if start_index + 3 < len(call_names) and (
        _matches_generated_name(call_names[start_index], "tvmgen_default_fused_mean")
        and _matches_generated_name(
            call_names[start_index + 1], "tvmgen_default_fused_subtract"
        )
        and _matches_generated_name(
            call_names[start_index + 2], "tvmgen_default_fused_multiply_cast_sum"
        )
        and _is_divide_norm_kernel(call_names[start_index + 3])
    ):
        return 4

    if start_index + 3 < len(call_names) and (
        (
            _matches_generated_name(call_names[start_index], "tvmgen_default_fused_cast_mean")
            or _matches_generated_name(
                call_names[start_index], "tvmgen_default_fused_cast_cast_mean"
            )
        )
        and (
            _matches_generated_name(
                call_names[start_index + 1], "tvmgen_default_fused_round_cast_subtract"
            )
            or _matches_generated_name(
                call_names[start_index + 1], "tvmgen_default_fused_cast_round_cast_subtract"
            )
        )
        and _matches_generated_name(
            call_names[start_index + 2], "tvmgen_default_fused_multiply_sum"
        )
        and _is_divide_norm_kernel(call_names[start_index + 3])
    ):
        return 4

    if start_index + 4 < len(call_names) and (
        _matches_generated_name(call_names[start_index], "tvmgen_default_fused_cast")
        and (
            _matches_generated_name(
                call_names[start_index + 1], "tvmgen_default_fused_cast_mean"
            )
            or _matches_generated_name(
                call_names[start_index + 1], "tvmgen_default_fused_cast_cast_mean"
            )
        )
        and (
            _matches_generated_name(
                call_names[start_index + 2], "tvmgen_default_fused_round_cast_subtract"
            )
            or _matches_generated_name(
                call_names[start_index + 2], "tvmgen_default_fused_cast_round_cast_subtract"
            )
        )
        and _matches_generated_name(
            call_names[start_index + 3], "tvmgen_default_fused_multiply_sum"
        )
        and _is_divide_norm_kernel(call_names[start_index + 4])
    ):
        return 5

    if start_index + 3 < len(call_names) and (
        _matches_generated_name(call_names[start_index], "tvmgen_default_fused_mean")
        and _matches_generated_name(
            call_names[start_index + 1], "tvmgen_default_fused_subtract"
        )
        and _matches_generated_name(
            call_names[start_index + 2], "tvmgen_default_fused_multiply_mean"
        )
        and _is_repq_layernorm_tail_kernel(call_names[start_index + 3])
    ):
        return 4

    return 0


def _is_divide_norm_kernel(name):
    return name.startswith(
        "tvmgen_default_fused_divide_add_divide_divide_add_divide_divide_add_divide_divide_add_divide_di_"
    )


def _classify_intrakernel_quant_kernel(name):
    if "fixed_point_multiply" in name:
        return "requant_kernel"

    repq_quant_markers = (
        "divide_round_maximum_minimum",
        "round_maximum_minimum_cast",
        "softmax_maximum_divide_log_divide_negative_multiply_round_greater_equal",
        "multiply_erf",
    )
    if any(marker in name for marker in repq_quant_markers):
        return "cpu_quant_kernel"

    repq_postop_markers = (
        "reshape_transpose_split",
        "_nn_bias_add_add",
        "_nn_bias_add_reshape_transpose_split",
    )
    if any(marker in name for marker in repq_postop_markers):
        return "cpu_postop_kernel"

    return None


def _is_requant_kernel(name):
    return _classify_intrakernel_quant_kernel(name) is not None


def _is_qkv_split_kernel(name):
    return "reshape_transpose_split" in name


def _is_dense_head_kernel(name):
    return _is_gemm_kernel(name) or name.startswith("tvmgen_default_fused_nn_dense")


def _find_head_start(call_names, search_from):
    search_start = max(0, search_from - 1)

    # Prefer the last layernorm+dense tail in the graph. This avoids treating
    # the final block's norm2+MLP as the classifier head in DeiT-like graphs.
    for idx in range(len(call_names) - 4, search_start - 1, -1):
        ln_len = _layernorm_sequence_length(call_names, idx)
        if ln_len == 0:
            continue

        dense_idx = None
        for probe in range(idx + ln_len, min(len(call_names), idx + ln_len + 16)):
            if _is_dense_head_kernel(call_names[probe]):
                dense_idx = probe
                break
        if dense_idx is None:
            continue
        if any(
            _is_qkv_split_kernel(call_names[probe])
            for probe in range(idx + ln_len, min(len(call_names), dense_idx + 8))
        ):
            continue
        if any(
            call_names[probe].startswith("tvmgen_default_fused_nn_softmax")
            for probe in range(idx + ln_len, dense_idx)
        ):
            continue
        if _find_next_layernorm_start(call_names, dense_idx + 1) is not None:
            continue
        return idx + 1

    for idx in range(search_start, len(call_names) - 3):
        ln_len = _layernorm_sequence_length(call_names, idx)
        if ln_len == 0:
            continue

        if any(
            _is_qkv_split_kernel(call_names[probe])
            for probe in range(idx + ln_len, min(len(call_names), idx + ln_len + 12))
        ):
            continue

        dense_idx = None
        for probe in range(idx + ln_len, min(len(call_names), idx + ln_len + 16)):
            if _is_dense_head_kernel(call_names[probe]):
                dense_idx = probe
                break
        if dense_idx is None:
            continue
        if any(
            call_names[probe].startswith("tvmgen_default_fused_nn_softmax")
            for probe in range(idx + ln_len, dense_idx)
        ):
            continue
        return idx + 1

    raise RuntimeError("Could not infer TVM head start from generated call order")


def _build_deit_layer_ranges(call_names):
    block_starts = []
    for idx in range(len(call_names) - 5):
        if not _is_layernorm_sequence(call_names, idx):
            continue
        qkv_gemm = _find_next_local(call_names, idx + 4, _is_gemm_kernel, end_index=idx + 8)
        if qkv_gemm is None:
            continue
        split_idx = _find_next_local(
            call_names,
            qkv_gemm + 1,
            _is_qkv_split_kernel,
            end_index=qkv_gemm + 5,
        )
        if split_idx is None:
            continue
        block_starts.append(idx + 1)

    if not block_starts:
        raise RuntimeError(
            f"Expected at least one DeiT block start, found {len(block_starts)}: {block_starts}"
        )

    head_start = None
    if len(block_starts) >= 12:
        try:
            head_start = _find_head_start(call_names, block_starts[-1])
        except RuntimeError:
            # Partial or structurally different graphs can end before the classifier head exists.
            head_start = None

    ranges = []
    if block_starts[0] > 1:
        ranges.append(
            {"layer": "patch_embed", "start_call": 1, "end_call": block_starts[0] - 1}
        )

    for block_idx, start_call in enumerate(block_starts):
        end_call = (
            block_starts[block_idx + 1] - 1
            if block_idx + 1 < len(block_starts)
            else (head_start - 1 if head_start is not None else len(call_names))
        )
        ranges.append(
            {
                "layer": f"block_{block_idx}",
                "start_call": start_call,
                "end_call": end_call,
            }
        )

    if head_start is not None:
        ranges.append({"layer": "head", "start_call": head_start, "end_call": len(call_names)})
    return ranges


def _build_swin_layer_ranges(call_names):
    split_positions = [
        idx + 1
        for idx, name in enumerate(call_names)
        if _is_qkv_split_kernel(name)
    ]
    if len(split_positions) != 12 and _is_repq_graph_call_names(call_names):
        return _build_repq_swin_layer_ranges(call_names)
    if len(split_positions) != 12:
        raise RuntimeError(
            f"Expected 12 Swin attention split markers, found {len(split_positions)}"
        )

    block_starts = []
    downsample_starts = []
    for split_call in split_positions:
        window_start = max(1, split_call - 13)
        ln_start_calls = [
            call_idx
            for call_idx in range(window_start, split_call)
            if _is_layernorm_sequence(call_names, call_idx - 1)
        ]
        downsample_merge_calls = [
            call_idx
            for call_idx in range(window_start, split_call)
            if call_names[call_idx - 1].startswith(
                "tvmgen_default_fused_reshape_strided_slice"
            )
        ]

        if not ln_start_calls:
            raise RuntimeError(
                f"Could not infer Swin block start for split call {split_call}"
            )

        if downsample_merge_calls and len(ln_start_calls) >= 2:
            block_starts.append(ln_start_calls[-1])
            downsample_starts.append(downsample_merge_calls[0])
        elif len(ln_start_calls) >= 2:
            block_starts.append(ln_start_calls[-1])
        else:
            block_starts.append(ln_start_calls[0])

    stage_depths = [2, 2, 6, 2]
    if len(block_starts) != sum(stage_depths):
        raise RuntimeError(
            f"Expected {sum(stage_depths)} Swin block starts, found {len(block_starts)}"
        )
    if len(downsample_starts) != len(stage_depths) - 1:
        raise RuntimeError(
            f"Expected {len(stage_depths) - 1} Swin downsample starts, found {len(downsample_starts)}"
        )

    head_start = _find_head_start(call_names, block_starts[-1])
    ranges = []
    if block_starts[0] > 1:
        ranges.append(
            {"layer": "patch_embed", "start_call": 1, "end_call": block_starts[0] - 1}
        )

    block_cursor = 0
    for stage_idx, depth in enumerate(stage_depths):
        for block_idx in range(depth):
            start_call = block_starts[block_cursor]
            is_last_block = block_idx == depth - 1
            is_last_stage = stage_idx == len(stage_depths) - 1

            if is_last_block and not is_last_stage:
                downsample_start = downsample_starts[stage_idx]
                end_call = downsample_start - 1
            elif block_cursor + 1 < len(block_starts):
                end_call = block_starts[block_cursor + 1] - 1
            else:
                end_call = head_start - 1

            ranges.append(
                {
                    "layer": f"stage{stage_idx}_block{block_idx}",
                    "start_call": start_call,
                    "end_call": end_call,
                }
            )

            if is_last_block and not is_last_stage:
                next_block_start = block_starts[block_cursor + 1]
                ranges.append(
                    {
                        "layer": f"stage{stage_idx}_downsample",
                        "start_call": downsample_starts[stage_idx],
                        "end_call": next_block_start - 1,
                    }
                )

            block_cursor += 1

    ranges.append({"layer": "head", "start_call": head_start, "end_call": len(call_names)})
    return ranges


def _is_repq_swin_downsample_merge_kernel(name):
    return name.startswith(
        "tvmgen_default_fused_reshape_strided_slice_strided_slice_strided_slice_strided_slice_concatenat_"
    )


def _build_repq_swin_layer_ranges(call_names):
    softmax_positions = [
        idx + 1
        for idx, name in enumerate(call_names)
        if name.startswith("tvmgen_default_fused_nn_softmax")
    ]
    if len(softmax_positions) != 12:
        raise RuntimeError(
            f"Expected 12 RepQ Swin softmax markers, found {len(softmax_positions)}"
        )

    block_starts = []
    for block_idx, softmax_call in enumerate(softmax_positions):
        search_start = max(1, softmax_call - 192)
        ln_candidates = [
            call_idx
            for call_idx in range(search_start, softmax_call)
            if _is_layernorm_sequence(call_names, call_idx - 1)
        ]
        if not ln_candidates:
            raise RuntimeError(
                f"Could not infer RepQ Swin block start for softmax call {softmax_call}"
            )
        if block_idx == 0:
            block_starts.append(ln_candidates[0])
        else:
            block_starts.append(ln_candidates[-1])

    downsample_starts = [
        idx + 1
        for idx, name in enumerate(call_names)
        if _is_repq_swin_downsample_merge_kernel(name)
    ]
    stage_depths = [2, 2, 6, 2]
    if len(block_starts) != sum(stage_depths):
        raise RuntimeError(
            f"Expected {sum(stage_depths)} RepQ Swin block starts, found {len(block_starts)}"
        )
    if len(downsample_starts) != len(stage_depths) - 1:
        raise RuntimeError(
            f"Expected {len(stage_depths) - 1} RepQ Swin downsample starts, found {len(downsample_starts)}"
        )

    head_start = _find_head_start(call_names, block_starts[-1])
    ranges = []
    if block_starts[0] > 1:
        ranges.append(
            {"layer": "patch_embed", "start_call": 1, "end_call": block_starts[0] - 1}
        )

    block_cursor = 0
    for stage_idx, depth in enumerate(stage_depths):
        for block_idx in range(depth):
            start_call = block_starts[block_cursor]
            is_last_block = block_idx == depth - 1
            is_last_stage = stage_idx == len(stage_depths) - 1

            if is_last_block and not is_last_stage:
                downsample_start = downsample_starts[stage_idx]
                end_call = downsample_start - 1
            elif block_cursor + 1 < len(block_starts):
                end_call = block_starts[block_cursor + 1] - 1
            else:
                end_call = head_start - 1

            ranges.append(
                {
                    "layer": f"stage{stage_idx}_block{block_idx}",
                    "start_call": start_call,
                    "end_call": end_call,
                }
            )

            if is_last_block and not is_last_stage:
                next_block_start = block_starts[block_cursor + 1]
                ranges.append(
                    {
                        "layer": f"stage{stage_idx}_downsample",
                        "start_call": downsample_starts[stage_idx],
                        "end_call": next_block_start - 1,
                    }
                )

            block_cursor += 1

    ranges.append({"layer": "head", "start_call": head_start, "end_call": len(call_names)})
    return ranges


def build_tvm_layer_ranges(output_dir, model_name, debug_unit=None):
    call_names = _load_tvm_main_call_names(output_dir)
    return call_names, build_tvm_layer_ranges_from_call_names(
        call_names, model_name, debug_unit=debug_unit
    )


def _is_layernorm_sequence(call_names, start_index):
    return _layernorm_sequence_length(call_names, start_index) > 0


def _is_gemm_kernel(name):
    return (
        name.startswith("tvmgen_default_fused_contrib_gemmini_gemm")
        or name.startswith("tvmgen_default_fused_nn_dense")
        or name.startswith("tvmgen_default_gemmini_main_")
    )


def _gemmini_main_index(name):
    match = re.fullmatch(r"tvmgen_default_gemmini_main_(\d+)", name)
    return int(match.group(1)) if match else None


def _is_llvm_gemmini_call_names(call_names):
    return any(name.startswith("tvmgen_default_gemmini_main_") for name in call_names)


def _classify_llvm_ivit_block_kernel(name):
    """Heuristic component label for llvm-gemmini I-ViT DeiT block kernels."""
    gemm_idx = _gemmini_main_index(name)
    if gemm_idx is not None:
        if gemm_idx == 0:
            return "qkv"
        if gemm_idx in (3, 6, 9):
            return "attn_scores"
        if gemm_idx in (12, 15, 18):
            return "attn_v"
        if gemm_idx == 21:
            return "proj"
        if gemm_idx == 24:
            return "fc1"
        if gemm_idx == 27:
            return "fc2"
        return "matmul"

    if _matches_generated_name(name, "tvmgen_default_fused_cast_cast_mean") or _matches_generated_name(
        name, "tvmgen_default_fused_cast_mean"
    ):
        return "layernorm"
    if _matches_generated_name(name, "tvmgen_default_fused_cast_round_cast_subtract") or _matches_generated_name(
        name, "tvmgen_default_fused_round_cast_subtract"
    ):
        return "layernorm"
    if _matches_generated_name(name, "tvmgen_default_fused_multiply_sum"):
        return "layernorm"
    if _is_divide_norm_kernel(name):
        return "layernorm"
    if name.startswith("tvmgen_default_fused_max"):
        return "softmax"
    if name.startswith(
        "tvmgen_default_fused_subtract_right_shift_add_right_shift_subtract_maximum_divide_multiply"
    ):
        return "softmax"
    if name.startswith(
        "tvmgen_default_fused_divide_multiply_add_right_shift_cast_reshape"
    ):
        return "softmax"
    if name.startswith(
        "tvmgen_default_fused_cast_subtract_right_shift_add_right_shift_subtract_maximum_divide_multiply"
    ):
        return "gelu"
    if "fixed_point_multiply_per_axis_clip_cast_reshape_cast_multiply" in name:
        return "resadd"
    if "fixed_point_multiply_per_axis" in name:
        return "requant"
    if any(
        token in name
        for token in ("reshape", "transpose", "squeeze", "expand_dims", "layout_transform", "concatenate")
    ):
        return "layout"
    if "cast" in name:
        return "requant"
    return "misc"


def _parse_llvm_ivit_transformer_block_component_segments(call_names):
    """Component segments for llvm-gemmini I-ViT graphs (partially inlined LN)."""
    segments = []
    if not call_names:
        return segments

    current = _classify_llvm_ivit_block_kernel(call_names[0])
    start = 0
    for idx in range(1, len(call_names)):
        label = _classify_llvm_ivit_block_kernel(call_names[idx])
        if label == current:
            continue
        _append_component_segment(segments, start, idx - 1, current)
        start = idx
        current = label
    _append_component_segment(segments, start, len(call_names) - 1, current)
    return segments


def _is_layernorm_kernel_name(name):
    return (
        _matches_generated_name(name, "tvmgen_default_fused_mean")
        or _matches_generated_name(name, "tvmgen_default_fused_subtract")
        or _matches_generated_name(name, "tvmgen_default_fused_multiply_cast_sum")
        or _matches_generated_name(name, "tvmgen_default_fused_multiply_mean")
        or _matches_generated_name(name, "tvmgen_default_fused_cast_mean")
        or _matches_generated_name(name, "tvmgen_default_fused_cast_cast_mean")
        or _matches_generated_name(name, "tvmgen_default_fused_multiply_sum")
        or _is_repq_layernorm_tail_kernel(name)
        or _is_divide_norm_kernel(name)
    )


def _find_next_local(call_names, start_index, predicate, end_index=None):
    limit = len(call_names) if end_index is None else min(end_index, len(call_names))
    for idx in range(max(0, start_index), limit):
        if predicate(call_names[idx]):
            return idx
    return None


def _find_next_layernorm_start(call_names, start_index):
    for idx in range(max(0, start_index), len(call_names) - 3):
        if _is_layernorm_sequence(call_names, idx):
            return idx
    return None


def _append_component_segment(segments, start_index, end_index, component):
    if start_index is None or end_index is None or end_index < start_index:
        return
    segments.append({"start": start_index, "end": end_index, "component": component})


def _append_split_gemm_post_segments(
    segments, call_names, start_index, end_index, gemm_component, post_component
):
    if start_index is None or end_index is None or end_index < start_index:
        return

    current_component = None
    current_start = None
    for idx in range(start_index, end_index + 1):
        component = gemm_component if _is_gemm_kernel(call_names[idx]) else post_component
        if current_component is None:
            current_component = component
            current_start = idx
            continue
        if component == current_component:
            continue
        _append_component_segment(segments, current_start, idx - 1, current_component)
        current_component = component
        current_start = idx

    if current_component is not None:
        _append_component_segment(segments, current_start, end_index, current_component)


def _parse_patch_embed_component_segments(call_names):
    return [{"start": 0, "end": len(call_names) - 1, "component": "patch_embed"}]


def _parse_head_component_segments(call_names):
    segments = []
    idx = 0
    if _is_layernorm_sequence(call_names, idx):
        ln_len = _layernorm_sequence_length(call_names, idx)
        _append_component_segment(segments, idx, idx + ln_len - 1, "layernorm")
        idx += ln_len
    last_softmax = _find_next_local(
        call_names, idx, lambda name: name == "tvmgen_default_fused_nn_softmax"
    )
    if last_softmax is None:
        _append_component_segment(segments, idx, len(call_names) - 1, "head")
        return segments
    _append_component_segment(segments, idx, last_softmax - 1, "head")
    _append_component_segment(segments, last_softmax, last_softmax, "softmax")
    return segments


def _parse_downsample_component_segments(call_names):
    segments = []
    ln_start = _find_next_layernorm_start(call_names, 0)
    if ln_start is None:
        return [{"start": 0, "end": len(call_names) - 1, "component": "downsample"}]
    _append_component_segment(segments, 0, ln_start - 1, "downsample_merge")
    ln_len = _layernorm_sequence_length(call_names, ln_start)
    _append_component_segment(segments, ln_start, ln_start + ln_len - 1, "layernorm")
    _append_component_segment(
        segments, ln_start + ln_len, len(call_names) - 1, "downsample_reduction"
    )
    return segments


def _parse_repq_transformer_block_component_segments(call_names):
    segments = []
    idx = 0
    n = len(call_names)

    if idx < n and _is_layernorm_sequence(call_names, idx):
        ln_len = _layernorm_sequence_length(call_names, idx)
        _append_component_segment(segments, idx, idx + ln_len - 1, "layernorm")
        idx += ln_len

    qkv_start = _find_next_local(call_names, idx, _is_gemm_kernel)
    if qkv_start is None:
        return segments
    norm2_start = _find_next_layernorm_start(call_names, qkv_start + 1)
    if norm2_start is None:
        _append_component_segment(segments, qkv_start, n - 1, "misc")
        return segments

    softmax_start = _find_next_local(
        call_names,
        qkv_start + 1,
        lambda name: name.startswith("tvmgen_default_fused_nn_softmax"),
        norm2_start,
    )
    softmax_end = None
    if softmax_start is None:
        _append_component_segment(segments, qkv_start, norm2_start - 1, "attn")
    elif any("reshape_transpose_split" in name for name in call_names[qkv_start:softmax_start]):
        qkv_end = qkv_start
        for probe_idx in range(qkv_start + 1, softmax_start):
            if "reshape_transpose_split" in call_names[probe_idx]:
                qkv_end = probe_idx
                break
        _append_component_segment(segments, qkv_start, qkv_end, "qkv")
        _append_component_segment(segments, qkv_end + 1, softmax_start - 1, "attn_scores")

        softmax_end = softmax_start
        _append_component_segment(segments, softmax_start, softmax_end, "softmax")

        post_softmax_gemms = [
            probe_idx
            for probe_idx in range(softmax_end + 1, norm2_start)
            if _is_gemm_kernel(call_names[probe_idx])
        ]
        if post_softmax_gemms:
            proj_start = post_softmax_gemms[-1]
            _append_component_segment(segments, softmax_end + 1, proj_start - 1, "attn_v")
            _append_component_segment(segments, proj_start, proj_start, "proj")
            _append_component_segment(segments, proj_start + 1, norm2_start - 1, "resadd")
        else:
            _append_component_segment(segments, softmax_end + 1, norm2_start - 1, "attn_v")
    else:
        qkv_end = max(qkv_start, softmax_start - 5)
        attn_scores_start = max(qkv_end + 1, softmax_start - 4)
        _append_component_segment(segments, qkv_start, qkv_end, "qkv")
        _append_component_segment(segments, attn_scores_start, softmax_start - 1, "attn_scores")

        softmax_end = softmax_start
        if softmax_start + 1 < norm2_start and "maximum_divide_log_divide_negative" in call_names[softmax_start + 1]:
            softmax_end = softmax_start + 1
        _append_component_segment(segments, softmax_start, softmax_end, "softmax")

        post_softmax_gemms = [
            probe_idx
            for probe_idx in range(softmax_end + 1, norm2_start)
            if _is_gemm_kernel(call_names[probe_idx])
        ]
        if post_softmax_gemms:
            proj_start = post_softmax_gemms[-1]
            _append_component_segment(segments, softmax_end + 1, proj_start - 1, "attn_v")
            _append_component_segment(segments, proj_start, proj_start, "proj")
            _append_component_segment(segments, proj_start + 1, norm2_start - 1, "resadd")
        else:
            _append_component_segment(segments, softmax_end + 1, norm2_start - 1, "attn_v")

    norm2_len = _layernorm_sequence_length(call_names, norm2_start)
    _append_component_segment(segments, norm2_start, norm2_start + norm2_len - 1, "layernorm")

    tail_start = norm2_start + norm2_len
    tail_gemms = [probe_idx for probe_idx in range(tail_start, n) if _is_gemm_kernel(call_names[probe_idx])]
    if len(tail_gemms) >= 2:
        fc1_start = tail_gemms[0]
        fc2_start = tail_gemms[-1]
        gelu_start = _find_next_local(
            call_names,
            fc1_start + 1,
            lambda name: "multiply_erf" in name or "gelu" in name or "maximum" in name,
            fc2_start,
        )
        if gelu_start is None:
            _append_component_segment(segments, fc1_start, fc2_start - 1, "fc1")
        else:
            _append_component_segment(segments, fc1_start, gelu_start - 1, "fc1")
            _append_component_segment(segments, gelu_start, fc2_start - 1, "gelu")
        _append_component_segment(segments, fc2_start, n - 1, "fc2")
    elif tail_gemms:
        _append_component_segment(segments, tail_gemms[0], n - 1, "fc2")

    return segments


def _parse_transformer_block_component_segments(call_names):
    if _is_repq_graph_call_names(call_names):
        return _parse_repq_transformer_block_component_segments(call_names)
    if _is_llvm_gemmini_call_names(call_names):
        return _parse_llvm_ivit_transformer_block_component_segments(call_names)

    segments = []
    idx = 0
    n = len(call_names)

    while idx < n:
        if _is_layernorm_sequence(call_names, idx):
            ln_len = _layernorm_sequence_length(call_names, idx)
            _append_component_segment(segments, idx, idx + ln_len - 1, "layernorm")
            idx += ln_len
            continue
        if _matches_generated_name(call_names[idx], "tvmgen_default_fused_cast"):
            _append_component_segment(segments, idx, idx, "requant")
            idx += 1
            continue
        break

    qkv_start = _find_next_local(call_names, idx, _is_gemm_kernel)
    if qkv_start is None:
        return segments
    _append_component_segment(segments, idx, qkv_start - 1, "layout")
    norm2_start = _find_next_layernorm_start(call_names, qkv_start + 1)
    if norm2_start is None:
        _append_component_segment(segments, qkv_start, n - 1, "other")
        return segments

    pre_softmax_gemms = [
        probe_idx
        for probe_idx in range(qkv_start, norm2_start)
        if _is_gemm_kernel(call_names[probe_idx])
    ]
    qkv_end = qkv_start
    if len(pre_softmax_gemms) >= 2:
        qkv_end = pre_softmax_gemms[1] - 1
    elif qkv_end + 1 < n and "reshape_transpose_split" in call_names[qkv_end + 1]:
        qkv_end += 1
    _append_component_segment(segments, qkv_start, qkv_end, "qkv")

    softmax_start = _find_next_local(
        call_names, qkv_end + 1, lambda name: name.startswith("tvmgen_default_fused_max"), norm2_start
    )
    softmax_end = _find_next_local(
        call_names,
        qkv_end + 1,
        lambda name: name.startswith(
            "tvmgen_default_fused_divide_multiply_add_right_shift_cast_reshape"
        ),
        norm2_start,
    )
    if softmax_start is None or softmax_end is None or softmax_end < softmax_start:
        _append_component_segment(segments, qkv_end + 1, norm2_start - 1, "attn")
    else:
        attn_scores_start = pre_softmax_gemms[1] if len(pre_softmax_gemms) >= 2 else qkv_end + 1
        _append_component_segment(segments, attn_scores_start, softmax_start - 1, "attn_scores")
        _append_component_segment(segments, softmax_start, softmax_end, "softmax")
        post_softmax_gemms = [
            probe_idx
            for probe_idx in range(softmax_end + 1, norm2_start)
            if _is_gemm_kernel(call_names[probe_idx])
        ]
        if post_softmax_gemms:
            proj_start = post_softmax_gemms[-1]
            _append_component_segment(segments, softmax_end + 1, proj_start - 1, "attn_v")
            _append_component_segment(segments, proj_start, norm2_start - 1, "proj")
        else:
            _append_component_segment(segments, softmax_end + 1, norm2_start - 1, "attn_v")

    norm2_len = _layernorm_sequence_length(call_names, norm2_start)
    _append_component_segment(
        segments, norm2_start, norm2_start + norm2_len - 1, "layernorm"
    )

    tail_start = norm2_start + norm2_len
    tail_gemms = [idx for idx in range(tail_start, n) if _is_gemm_kernel(call_names[idx])]
    if len(tail_gemms) >= 2:
        fc1_start = tail_gemms[0]
        fc2_start = tail_gemms[-1]
        _append_component_segment(segments, tail_start, fc1_start - 1, "layout")
        gelu_start = _find_next_local(
            call_names,
            fc1_start + 1,
            lambda name: name.startswith("tvmgen_default_fused_max")
            or "gelu" in name
            or "maximum" in name,
            fc2_start,
        )
        if gelu_start is None:
            _append_component_segment(segments, fc1_start, fc2_start - 1, "fc1")
        else:
            _append_component_segment(segments, fc1_start, gelu_start - 1, "fc1")
            _append_component_segment(segments, gelu_start, fc2_start - 1, "gelu")

        trailing_resadd_start = None
        if n - 1 > fc2_start and call_names[n - 1].startswith("tvmgen_default_fused_cast"):
            trailing_resadd_start = n - 1
        if trailing_resadd_start is None:
            _append_component_segment(segments, fc2_start, n - 1, "fc2")
        else:
            _append_component_segment(segments, fc2_start, trailing_resadd_start - 1, "fc2")
            _append_component_segment(segments, trailing_resadd_start, n - 1, "resadd")
    elif tail_gemms:
        _append_component_segment(segments, tail_start, tail_gemms[0] - 1, "layout")
        _append_component_segment(segments, tail_gemms[0], tail_gemms[0], "fc")
        _append_component_segment(segments, tail_gemms[0] + 1, n - 1, "other")
    else:
        _append_component_segment(segments, tail_start, n - 1, "other")

    return segments


def _parse_aligned_component_segments(layer_name, model_name, call_names, split_post_ops=False):
    if layer_name == "patch_embed":
        return _parse_aligned_patch_embed_component_segments(
            model_name, call_names, split_post_ops=split_post_ops
        )
    if layer_name.endswith("downsample"):
        segments = []
        ln_start = _find_next_layernorm_start(call_names, 0)
        if ln_start is None:
            return [{"start": 0, "end": len(call_names) - 1, "component": "downsample_merge"}]
        _append_component_segment(segments, 0, ln_start - 1, "downsample_merge")
        _append_component_segment(segments, ln_start, ln_start + 3, "downsample_norm")
        _append_component_segment(
            segments, ln_start + 4, len(call_names) - 1, "downsample_reduction"
        )
        return segments
    if layer_name == "head":
        segments = []
        idx = 0
        if _is_layernorm_sequence(call_names, idx):
            ln_len = _layernorm_sequence_length(call_names, idx)
            _append_component_segment(segments, idx, idx + ln_len - 1, "head_norm")
            idx += ln_len
        _append_component_segment(segments, idx, len(call_names) - 1, "head_linear")
        return segments
    if _is_repq_graph_call_names(call_names):
        return _parse_repq_aligned_transformer_block_component_segments(
            layer_name, model_name, call_names
        )
    if _is_llvm_gemmini_call_names(call_names):
        return _parse_llvm_ivit_transformer_block_component_segments(call_names)
    return _parse_aligned_transformer_block_component_segments(
        layer_name, model_name, call_names, split_post_ops=split_post_ops
    )


def _parse_aligned_patch_embed_component_segments(model_name, call_names, split_post_ops=False):
    segments = []

    if model_name.startswith("swin_"):
        ln_start = _find_next_layernorm_start(call_names, 0)
        if ln_start is None:
            return [{"start": 0, "end": len(call_names) - 1, "component": "patch_embed"}]

        if split_post_ops:
            patch_pre_idx = ln_start - 1 if ln_start > 0 else None
            patch_embed_end = patch_pre_idx - 1 if patch_pre_idx is not None else ln_start - 1
            _append_component_segment(segments, 0, patch_embed_end, "patch_embed")
            _append_component_segment(segments, patch_pre_idx, patch_pre_idx, "patch_pre")
            ln_len = _layernorm_sequence_length(call_names, ln_start)
            _append_component_segment(segments, ln_start, ln_start + ln_len - 1, "patch_norm")
            _append_component_segment(
                segments, ln_start + ln_len, len(call_names) - 1, "patch_post"
            )
            return segments

        ln_len = _layernorm_sequence_length(call_names, ln_start)
        _append_component_segment(segments, 0, ln_start - 1, "patch_embed")
        _append_component_segment(segments, ln_start, len(call_names) - 1, "patch_norm")
        return segments

    if split_post_ops:
        conv_idx = _find_next_local(
            call_names,
            0,
            lambda name: name.startswith("tvmgen_default_fused_contrib_gemmini_conv2d"),
        )
        if conv_idx is None:
            return [{"start": 0, "end": len(call_names) - 1, "component": "patch_embed"}]
        _append_component_segment(segments, 0, conv_idx, "patch_embed")
        _append_component_segment(segments, conv_idx + 1, len(call_names) - 1, "patch_post")
        return segments

    return [{"start": 0, "end": len(call_names) - 1, "component": "patch_embed"}]


def _parse_repq_aligned_transformer_block_component_segments(layer_name, model_name, call_names):
    segments = []
    idx = 0
    n = len(call_names)

    if idx < n and _is_layernorm_sequence(call_names, idx):
        ln_len = _layernorm_sequence_length(call_names, idx)
        _append_component_segment(segments, idx, idx + ln_len - 1, "norm1")
        idx += ln_len

    qkv_start = _find_next_local(call_names, idx, _is_gemm_kernel)
    if qkv_start is None:
        return segments

    norm2_start = _find_next_layernorm_start(call_names, qkv_start + 1)
    if norm2_start is None:
        _append_component_segment(segments, qkv_start, n - 1, "misc")
        return segments

    softmax_start = _find_next_local(
        call_names,
        qkv_start + 1,
        lambda name: name.startswith("tvmgen_default_fused_nn_softmax"),
        norm2_start,
    )
    softmax_end = None
    if softmax_start is None:
        _append_component_segment(segments, qkv_start, norm2_start - 1, "attn_scores")
    elif any("reshape_transpose_split" in name for name in call_names[qkv_start:softmax_start]):
        qkv_end = qkv_start
        for probe_idx in range(qkv_start + 1, softmax_start):
            if "reshape_transpose_split" in call_names[probe_idx]:
                qkv_end = probe_idx
                break
        _append_component_segment(segments, qkv_start, qkv_end, "qkv")
        _append_component_segment(segments, qkv_end + 1, softmax_start - 1, "attn_scores")
        softmax_end = softmax_start
        _append_component_segment(segments, softmax_start, softmax_end, "softmax")
    else:
        qkv_end = max(qkv_start, softmax_start - 5)
        attn_scores_start = max(qkv_end + 1, softmax_start - 4)
        _append_component_segment(segments, qkv_start, qkv_end, "qkv")
        _append_component_segment(segments, attn_scores_start, softmax_start - 1, "attn_scores")
        softmax_end = softmax_start
        if softmax_start + 1 < norm2_start and "maximum_divide_log_divide_negative" in call_names[softmax_start + 1]:
            softmax_end = softmax_start + 1
        _append_component_segment(segments, softmax_start, softmax_end, "softmax")

    if softmax_end is None:
        _append_component_segment(segments, qkv_start, norm2_start - 1, "attn_v")
    else:
        post_softmax_gemms = [
            probe_idx
            for probe_idx in range(softmax_end + 1, norm2_start)
            if _is_gemm_kernel(call_names[probe_idx])
        ]
        if post_softmax_gemms:
            proj_start = post_softmax_gemms[-1]
            _append_component_segment(segments, softmax_end + 1, proj_start - 1, "attn_v")
            _append_component_segment(segments, proj_start, norm2_start - 1, "proj_res1")
        else:
            _append_component_segment(segments, softmax_end + 1, norm2_start - 1, "attn_v")

    norm2_len = _layernorm_sequence_length(call_names, norm2_start)
    _append_component_segment(segments, norm2_start, norm2_start + norm2_len - 1, "norm2")

    tail_start = norm2_start + norm2_len
    tail_gemms = [probe_idx for probe_idx in range(tail_start, n) if _is_gemm_kernel(call_names[probe_idx])]
    if len(tail_gemms) >= 2:
        fc1_start = tail_gemms[0]
        fc2_start = tail_gemms[-1]
        gelu_start = _find_next_local(
            call_names,
            fc1_start + 1,
            lambda name: "multiply_erf" in name or "gelu" in name or "maximum" in name,
            fc2_start,
        )
        if gelu_start is None:
            _append_component_segment(segments, fc1_start, fc2_start - 1, "fc1")
        else:
            _append_component_segment(segments, fc1_start, gelu_start - 1, "fc1")
            _append_component_segment(segments, gelu_start, fc2_start - 1, "gelu")
        _append_component_segment(segments, fc2_start, n - 1, "fc2_res2")
    elif tail_gemms:
        _append_component_segment(segments, tail_gemms[0], n - 1, "fc2_res2")

    return segments


def _parse_aligned_transformer_block_component_segments(
    layer_name, model_name, call_names, split_post_ops=False
):
    segments = []
    idx = 0
    n = len(call_names)

    if idx < n and _is_layernorm_sequence(call_names, idx):
        ln_len = _layernorm_sequence_length(call_names, idx)
        _append_component_segment(segments, idx, idx + ln_len - 1, "norm1")
        idx += ln_len

    qkv_start = _find_next_local(call_names, idx, _is_gemm_kernel)
    if qkv_start is None:
        return segments
    norm2_start = _find_next_layernorm_start(call_names, qkv_start + 1)
    if norm2_start is None:
        _append_component_segment(segments, qkv_start, n - 1, "misc")
        return segments

    pre_softmax_gemms = [
        probe_idx
        for probe_idx in range(qkv_start, norm2_start)
        if _is_gemm_kernel(call_names[probe_idx])
    ]
    qkv_end = qkv_start
    if len(pre_softmax_gemms) >= 2:
        qkv_end = pre_softmax_gemms[1] - 1
    elif qkv_end + 1 < n and "reshape_transpose_split" in call_names[qkv_end + 1]:
        qkv_end += 1
    if split_post_ops and qkv_start <= qkv_end:
        _append_component_segment(segments, qkv_start, qkv_start, "qkv_gemm")
        _append_component_segment(segments, qkv_start + 1, qkv_end, "qkv_post")
    else:
        _append_component_segment(segments, qkv_start, qkv_end, "qkv")

    softmax_start = _find_next_local(
        call_names, qkv_end + 1, lambda name: name.startswith("tvmgen_default_fused_max"), norm2_start
    )
    softmax_end = _find_next_local(
        call_names,
        qkv_end + 1,
        lambda name: name.startswith(
            "tvmgen_default_fused_divide_multiply_add_right_shift_cast_reshape"
        ),
        norm2_start,
    )
    if softmax_start is None or softmax_end is None or softmax_end < softmax_start:
        _append_component_segment(segments, qkv_end + 1, norm2_start - 1, "attn_scores")
    else:
        attn_scores_start = pre_softmax_gemms[1] if len(pre_softmax_gemms) >= 2 else qkv_end + 1
        if split_post_ops:
            _append_split_gemm_post_segments(
                segments,
                call_names,
                attn_scores_start,
                softmax_start - 1,
                "attn_scores_gemm",
                "attn_scores_post",
            )
        else:
            _append_component_segment(segments, attn_scores_start, softmax_start - 1, "attn_scores")
        _append_component_segment(segments, softmax_start, softmax_end, "softmax")
        post_softmax_gemms = [
            probe_idx
            for probe_idx in range(softmax_end + 1, norm2_start)
            if _is_gemm_kernel(call_names[probe_idx])
        ]
        if post_softmax_gemms:
            proj_start = post_softmax_gemms[-1]
            if split_post_ops:
                _append_split_gemm_post_segments(
                    segments,
                    call_names,
                    softmax_end + 1,
                    proj_start - 1,
                    "attn_v_gemm",
                    "attn_v_post",
                )
                _append_component_segment(segments, proj_start, proj_start, "proj_gemm")
                _append_component_segment(
                    segments, proj_start + 1, norm2_start - 1, "res1_postop"
                )
            else:
                _append_component_segment(segments, softmax_end + 1, proj_start - 1, "attn_v")
                _append_component_segment(
                    segments, proj_start, norm2_start - 1, "proj_res1"
                )
        else:
            _append_component_segment(segments, softmax_end + 1, norm2_start - 1, "attn_v")

    norm2_len = _layernorm_sequence_length(call_names, norm2_start)
    _append_component_segment(segments, norm2_start, norm2_start + norm2_len - 1, "norm2")

    tail_start = norm2_start + norm2_len
    tail_gemms = [probe_idx for probe_idx in range(tail_start, n) if _is_gemm_kernel(call_names[probe_idx])]
    if len(tail_gemms) >= 2:
        fc1_start = tail_gemms[0]
        fc2_start = tail_gemms[-1]
        gelu_start = _find_next_local(
            call_names,
            fc1_start + 1,
            lambda name: name.startswith("tvmgen_default_fused_max")
            or "gelu" in name
            or "maximum" in name,
            fc2_start,
        )
        if gelu_start is None:
            if split_post_ops:
                _append_component_segment(segments, fc1_start, fc1_start, "fc1_gemm")
                _append_component_segment(segments, fc1_start + 1, fc2_start - 1, "fc1_requant")
            else:
                _append_component_segment(segments, fc1_start, fc2_start - 1, "fc1")
        else:
            if split_post_ops:
                _append_component_segment(segments, fc1_start, fc1_start, "fc1_gemm")
                _append_component_segment(segments, fc1_start + 1, gelu_start - 1, "fc1_requant")
            else:
                _append_component_segment(segments, fc1_start, gelu_start - 1, "fc1")
            _append_component_segment(segments, gelu_start, fc2_start - 1, "gelu")
        if split_post_ops:
            _append_component_segment(segments, fc2_start, fc2_start, "fc2_gemm")
            _append_component_segment(segments, fc2_start + 1, n - 1, "res2_postop")
        else:
            _append_component_segment(segments, fc2_start, n - 1, "fc2_res2")
    elif tail_gemms:
        if split_post_ops:
            _append_component_segment(segments, tail_gemms[0], tail_gemms[0], "fc2_gemm")
            _append_component_segment(segments, tail_gemms[0] + 1, n - 1, "res2_postop")
        else:
            _append_component_segment(segments, tail_gemms[0], n - 1, "fc2_res2")

    return segments


def _fallback_tvm_component_name(layer_name, kernel_name):
    if layer_name == "patch_embed":
        return "patch_embed"
    if layer_name.endswith("downsample"):
        if _is_layernorm_kernel_name(kernel_name):
            return "layernorm"
        if "nn_dense" in kernel_name or _is_gemm_kernel(kernel_name):
            return "downsample_reduction"
        if kernel_name.startswith(
            "tvmgen_default_fused_reshape_cast_fixed_point_multiply_cast_fixed_point_multiply_add_clip_cast"
        ):
            return "downsample_reduction"
        if _matches_generated_name(kernel_name, "tvmgen_default_fused_cast_1") or _matches_generated_name(
            kernel_name, "tvmgen_default_fused_cast_2"
        ) or _matches_generated_name(kernel_name, "tvmgen_default_fused_cast_3"):
            return "downsample_reduction"
        return "downsample_merge"
    if layer_name == "head":
        if kernel_name == "tvmgen_default_fused_nn_softmax":
            return "softmax"
        if "nn_dense" in kernel_name:
            return "head"
        if _is_layernorm_kernel_name(kernel_name):
            return "layernorm"
        return "head"
    if _is_layernorm_kernel_name(kernel_name):
        return "layernorm"
    if kernel_name.startswith("tvmgen_default_fused_max"):
        return "softmax"
    if kernel_name.startswith(
        "tvmgen_default_fused_divide_multiply_add_right_shift_cast_reshape_split"
    ):
        return "softmax"
    if _is_gemm_kernel(kernel_name):
        return "matmul"
    if any(
        token in kernel_name
        for token in ("reshape", "transpose", "squeeze", "expand_dims", "layout_transform")
    ):
        return "layout"
    if "cast" in kernel_name:
        return "requant"
    return "other"


def _fallback_tvm_aligned_component_name(layer_name, kernel_name):
    if layer_name == "patch_embed":
        return "patch_embed"
    if layer_name.endswith("downsample"):
        if _is_layernorm_kernel_name(kernel_name):
            return "downsample_norm"
        if "nn_dense" in kernel_name or _is_gemm_kernel(kernel_name):
            return "downsample_reduction"
        if "cast" in kernel_name:
            return "downsample_reduction"
        return "downsample_merge"
    if layer_name == "head":
        if _is_layernorm_kernel_name(kernel_name):
            return "head_norm"
        return "head_linear"
    if kernel_name.startswith("tvmgen_default_fused_max"):
        return "softmax"
    if "gelu" in kernel_name or "maximum" in kernel_name:
        return "gelu"
    if any(
        token in kernel_name
        for token in ("reshape", "transpose", "squeeze", "expand_dims", "layout_transform")
    ):
        return "layout"
    if "cast" in kernel_name:
        return "requant"
    return "misc"


def build_tvm_component_rows(output_dir, model_name):
    call_names = _load_tvm_main_call_names(output_dir)
    _, _, component_rows = _build_tvm_semantic_segments_from_call_names(call_names, model_name)
    return call_names, component_rows


def write_spike_kernel_cycle_reports(output_dir, rows, topk=40):
    csv_path = output_dir / "spike_kernel_cycles.csv"
    txt_path = output_dir / "spike_kernel_cycles.txt"

    rows = sorted(rows, key=lambda row: row["call_index"])
    grouped = defaultdict(lambda: {"cycles": 0, "calls": 0})
    for row in rows:
        grouped[row["kernel_name"]]["cycles"] += row["cycles"]
        grouped[row["kernel_name"]]["calls"] += 1

    by_cycles = sorted(rows, key=lambda row: row["cycles"], reverse=True)
    grouped_rows = sorted(
        (
            {
                "kernel_name": kernel_name,
                "cycles": stats["cycles"],
                "calls": stats["calls"],
            }
            for kernel_name, stats in grouped.items()
        ),
        key=lambda row: row["cycles"],
        reverse=True,
    )

    with open(csv_path, "w") as f:
        f.write("call_index,kernel_name,cycles\n")
        for row in rows:
            f.write(f"{row['call_index']},{row['kernel_name']},{row['cycles']}\n")

    with open(txt_path, "w") as f:
        f.write("Spike rdcycle TVM kernel breakdown\n")
        f.write(f"Kernel calls: {len(rows)}\n")
        f.write(f"Profiled cycles: {sum(row['cycles'] for row in rows)}\n")
        f.write("\nTop kernel calls by cycles:\n")
        f.write("rank,call_index,cycles,kernel_name\n")
        for rank, row in enumerate(by_cycles[:topk], start=1):
            f.write(f"{rank},{row['call_index']},{row['cycles']},{row['kernel_name']}\n")
        f.write("\nGrouped by kernel function:\n")
        f.write("rank,cycles,calls,kernel_name\n")
        for rank, row in enumerate(grouped_rows[:topk], start=1):
            f.write(f"{rank},{row['cycles']},{row['calls']},{row['kernel_name']}\n")

    return csv_path, txt_path


def build_spike_segment_cycle_rows(
    output_dir, model_name, kernel_rows, style="semantic", debug_unit=None
):
    call_names = _load_tvm_main_call_names(output_dir)
    _, segments, _ = _build_tvm_segments_for_style(
        call_names, model_name, style, debug_unit=debug_unit
    )

    row_by_call = {row["call_index"]: row for row in kernel_rows}
    missing_calls = sorted(set(range(1, len(call_names) + 1)) - set(row_by_call))
    if missing_calls:
        raise RuntimeError(
            f"Kernel cycle rows missing call indices, first missing calls: {missing_calls[:10]}"
        )

    rows = []
    for segment in segments:
        rows.append(
            {
                "segment_index": segment["segment_index"],
                "layer": segment["layer"],
                "component": segment["component"],
                "start_call": segment["start_call"],
                "end_call": segment["end_call"],
                "cycles": sum(
                    row_by_call[call_index]["cycles"]
                    for call_index in range(segment["start_call"], segment["end_call"] + 1)
                ),
            }
        )
    return rows


def _write_spike_segment_cycle_reports_core(
    rows,
    segment_csv_path,
    segment_txt_path,
    layer_csv_path,
    layer_txt_path,
    component_csv_path,
    component_txt_path,
    layer_component_csv_path,
    layer_component_txt_path,
    report_label,
    main_total_cycles=None,
):
    rows = sorted(rows, key=lambda row: row["segment_index"])
    total_cycles = sum(row["cycles"] for row in rows)
    covered_calls = 0
    grouped_layers = defaultdict(lambda: {"cycles": 0, "calls": 0, "segments": 0, "start_call": None, "end_call": None})
    grouped_components = defaultdict(lambda: {"cycles": 0, "calls": 0, "segments": 0})
    grouped_layer_components = defaultdict(lambda: {"cycles": 0, "calls": 0, "segments": 0})

    with open(segment_csv_path, "w") as f:
        f.write("segment_index,layer,component,start_call,end_call,calls,cycles,percent\n")
        for row in rows:
            call_count = row["end_call"] - row["start_call"] + 1
            covered_calls += call_count
            percent = 100.0 * row["cycles"] / total_cycles if total_cycles else 0.0
            f.write(
                f"{row['segment_index']},{row['layer']},{row['component']},"
                f"{row['start_call']},{row['end_call']},{call_count},{row['cycles']},{percent:.6f}\n"
            )

            layer_bucket = grouped_layers[row["layer"]]
            layer_bucket["cycles"] += row["cycles"]
            layer_bucket["calls"] += call_count
            layer_bucket["segments"] += 1
            layer_bucket["start_call"] = (
                row["start_call"]
                if layer_bucket["start_call"] is None
                else min(layer_bucket["start_call"], row["start_call"])
            )
            layer_bucket["end_call"] = (
                row["end_call"]
                if layer_bucket["end_call"] is None
                else max(layer_bucket["end_call"], row["end_call"])
            )

            component_bucket = grouped_components[row["component"]]
            component_bucket["cycles"] += row["cycles"]
            component_bucket["calls"] += call_count
            component_bucket["segments"] += 1

            layer_component_bucket = grouped_layer_components[(row["layer"], row["component"])]
            layer_component_bucket["cycles"] += row["cycles"]
            layer_component_bucket["calls"] += call_count
            layer_component_bucket["segments"] += 1

    with open(segment_txt_path, "w") as f:
        f.write(f"Spike rdcycle {report_label} segment breakdown\n")
        f.write(f"Segments: {len(rows)}\n")
        f.write(f"Covered kernel calls: {covered_calls}\n")
        f.write(f"Profiled cycles: {total_cycles}\n")
        if main_total_cycles is not None:
            ratio = 100.0 * total_cycles / main_total_cycles if main_total_cycles else 0.0
            f.write(f"TVM main cycles: {main_total_cycles}\n")
            f.write(f"Profile/main ratio: {ratio:.3f}%\n")
        f.write("\nTop segments:\n")
        f.write("rank,layer,component,start_call,end_call,calls,cycles,percent\n")
        for rank, row in enumerate(sorted(rows, key=lambda item: item["cycles"], reverse=True)[:40], start=1):
            call_count = row["end_call"] - row["start_call"] + 1
            percent = 100.0 * row["cycles"] / total_cycles if total_cycles else 0.0
            f.write(
                f"{rank},{row['layer']},{row['component']},{row['start_call']},"
                f"{row['end_call']},{call_count},{row['cycles']},{percent:.3f}\n"
            )

    layer_rows = sorted(
        (
            {
                "layer": layer,
                "cycles": stats["cycles"],
                "calls": stats["calls"],
                "segments": stats["segments"],
                "start_call": stats["start_call"],
                "end_call": stats["end_call"],
            }
            for layer, stats in grouped_layers.items()
        ),
        key=lambda row: row["cycles"],
        reverse=True,
    )
    with open(layer_csv_path, "w") as f:
        f.write("layer,cycles,calls,segments,percent,start_call,end_call\n")
        for row in layer_rows:
            percent = 100.0 * row["cycles"] / total_cycles if total_cycles else 0.0
            f.write(
                f"{row['layer']},{row['cycles']},{row['calls']},{row['segments']},"
                f"{percent:.6f},{row['start_call']},{row['end_call']}\n"
            )
    with open(layer_txt_path, "w") as f:
        f.write(f"Spike rdcycle {report_label} layer breakdown\n")
        f.write(f"Profiled cycles: {total_cycles}\n")
        if main_total_cycles is not None:
            ratio = 100.0 * total_cycles / main_total_cycles if main_total_cycles else 0.0
            f.write(f"TVM main cycles: {main_total_cycles}\n")
            f.write(f"Profile/main ratio: {ratio:.3f}%\n")
        f.write("rank,layer,cycles,percent,calls,segments,start_call,end_call\n")
        for rank, row in enumerate(layer_rows, start=1):
            percent = 100.0 * row["cycles"] / total_cycles if total_cycles else 0.0
            f.write(
                f"{rank},{row['layer']},{row['cycles']},{percent:.3f},{row['calls']},"
                f"{row['segments']},{row['start_call']},{row['end_call']}\n"
            )

    component_rows = sorted(
        (
            {
                "component": component,
                "cycles": stats["cycles"],
                "calls": stats["calls"],
                "segments": stats["segments"],
            }
            for component, stats in grouped_components.items()
        ),
        key=lambda row: row["cycles"],
        reverse=True,
    )
    with open(component_csv_path, "w") as f:
        f.write("component,cycles,calls,segments,percent\n")
        for row in component_rows:
            percent = 100.0 * row["cycles"] / total_cycles if total_cycles else 0.0
            f.write(
                f"{row['component']},{row['cycles']},{row['calls']},{row['segments']},{percent:.6f}\n"
            )
    with open(component_txt_path, "w") as f:
        f.write(f"Spike rdcycle {report_label} component breakdown\n")
        f.write(f"Profiled cycles: {total_cycles}\n")
        f.write("rank,component,cycles,percent,calls,segments\n")
        for rank, row in enumerate(component_rows, start=1):
            percent = 100.0 * row["cycles"] / total_cycles if total_cycles else 0.0
            f.write(
                f"{rank},{row['component']},{row['cycles']},{percent:.3f},"
                f"{row['calls']},{row['segments']}\n"
            )

    layer_component_rows = sorted(
        (
            {
                "layer": layer,
                "component": component,
                "cycles": stats["cycles"],
                "calls": stats["calls"],
                "segments": stats["segments"],
            }
            for (layer, component), stats in grouped_layer_components.items()
        ),
        key=lambda row: row["cycles"],
        reverse=True,
    )
    with open(layer_component_csv_path, "w") as f:
        f.write("layer,component,cycles,calls,segments,percent\n")
        for row in layer_component_rows:
            percent = 100.0 * row["cycles"] / total_cycles if total_cycles else 0.0
            f.write(
                f"{row['layer']},{row['component']},{row['cycles']},{row['calls']},"
                f"{row['segments']},{percent:.6f}\n"
            )
    with open(layer_component_txt_path, "w") as f:
        f.write(f"Spike rdcycle {report_label} layer-component breakdown\n")
        f.write(f"Profiled cycles: {total_cycles}\n")
        f.write("rank,layer,component,cycles,percent,calls,segments\n")
        for rank, row in enumerate(layer_component_rows, start=1):
            percent = 100.0 * row["cycles"] / total_cycles if total_cycles else 0.0
            f.write(
                f"{rank},{row['layer']},{row['component']},{row['cycles']},{percent:.3f},"
                f"{row['calls']},{row['segments']}\n"
            )

    return (
        segment_csv_path,
        segment_txt_path,
        layer_csv_path,
        layer_txt_path,
        component_csv_path,
        component_txt_path,
        layer_component_csv_path,
        layer_component_txt_path,
    )


def write_spike_semantic_cycle_reports(output_dir, semantic_rows, main_total_cycles=None):
    return _write_spike_segment_cycle_reports_core(
        semantic_rows,
        output_dir / "spike_semantic_segment_cycles.csv",
        output_dir / "spike_semantic_segment_cycles.txt",
        output_dir / "spike_layer_cycles.csv",
        output_dir / "spike_layer_cycles.txt",
        output_dir / "spike_component_cycles.csv",
        output_dir / "spike_component_cycles.txt",
        output_dir / "spike_layer_component_cycles.csv",
        output_dir / "spike_layer_component_cycles.txt",
        report_label="TVM semantic",
        main_total_cycles=main_total_cycles,
    )


def write_spike_intrakernel_cycle_reports(output_dir, intrakernel_rows):
    csv_path = output_dir / "spike_intrakernel_cycles.csv"
    txt_path = output_dir / "spike_intrakernel_cycles.txt"

    grouped = defaultdict(int)
    total_cycles = 0
    for row in intrakernel_rows:
        grouped[(row["component"], row.get("label", ""))] += row["cycles"]
        total_cycles += row["cycles"]

    component_rows = sorted(
        (
            {
                "component": component,
                "label": label,
                "cycles": cycles,
            }
            for (component, label), cycles in grouped.items()
        ),
        key=lambda row: row["cycles"],
        reverse=True,
    )

    with open(csv_path, "w") as f:
        f.write("component,label,cycles,percent\n")
        for row in component_rows:
            percent = 100.0 * row["cycles"] / total_cycles if total_cycles else 0.0
            f.write(f"{row['component']},{row['label']},{row['cycles']},{percent:.6f}\n")

    with open(txt_path, "w") as f:
        f.write("Spike rdcycle TVM intrakernel breakdown\n")
        f.write(f"Profiled cycles: {total_cycles}\n")
        f.write("rank,component,label,cycles,percent\n")
        for rank, row in enumerate(component_rows, start=1):
            percent = 100.0 * row["cycles"] / total_cycles if total_cycles else 0.0
            f.write(
                f"{rank},{row['component']},{row['label']},{row['cycles']},{percent:.3f}\n"
            )

    return csv_path, txt_path


def write_spike_aligned_cycle_reports(output_dir, aligned_rows, main_total_cycles=None):
    return _write_spike_segment_cycle_reports_core(
        aligned_rows,
        output_dir / "spike_aligned_segment_cycles.csv",
        output_dir / "spike_aligned_segment_cycles.txt",
        output_dir / "spike_aligned_layer_cycles.csv",
        output_dir / "spike_aligned_layer_cycles.txt",
        output_dir / "spike_aligned_component_cycles.csv",
        output_dir / "spike_aligned_component_cycles.txt",
        output_dir / "spike_aligned_layer_component_cycles.csv",
        output_dir / "spike_aligned_layer_component_cycles.txt",
        report_label="TVM ORT-aligned",
        main_total_cycles=main_total_cycles,
    )


def write_spike_layer_cycle_reports(output_dir, model_name, rows):
    call_names, layer_ranges = build_tvm_layer_ranges(output_dir, model_name)
    row_by_call = {row["call_index"]: row for row in rows}

    summary_csv_path = output_dir / "spike_layer_cycles.csv"
    summary_txt_path = output_dir / "spike_layer_cycles.txt"
    call_map_csv_path = output_dir / "spike_call_layer_map.csv"

    layer_rows = []
    total_profiled_cycles = 0
    covered_calls = set()

    with open(call_map_csv_path, "w") as f:
        f.write("call_index,layer,kernel_name,cycles\n")
        for layer in layer_ranges:
            layer_cycles = 0
            layer_call_count = 0
            for call_index in range(layer["start_call"], layer["end_call"] + 1):
                row = row_by_call.get(call_index)
                if row is None:
                    continue
                layer_cycles += row["cycles"]
                layer_call_count += 1
                covered_calls.add(call_index)
                f.write(
                    f"{call_index},{layer['layer']},{row['kernel_name']},{row['cycles']}\n"
                )

            layer_rows.append(
                {
                    "layer": layer["layer"],
                    "start_call": layer["start_call"],
                    "end_call": layer["end_call"],
                    "calls": layer_call_count,
                    "cycles": layer_cycles,
                }
            )
            total_profiled_cycles += layer_cycles

    missing_calls = sorted(set(row_by_call) - covered_calls)
    if missing_calls:
        raise RuntimeError(
            f"TVM layer ranges did not cover all kernel calls, first missing calls: {missing_calls[:10]}"
        )

    with open(summary_csv_path, "w") as f:
        f.write("layer,start_call,end_call,calls,cycles\n")
        for row in layer_rows:
            f.write(
                f"{row['layer']},{row['start_call']},{row['end_call']},{row['calls']},{row['cycles']}\n"
            )

    ranked_rows = sorted(layer_rows, key=lambda row: row["cycles"], reverse=True)
    with open(summary_txt_path, "w") as f:
        f.write("Spike rdcycle TVM layer breakdown\n")
        f.write(f"Kernel calls in generated main: {len(call_names)}\n")
        f.write(f"Profiled cycles: {total_profiled_cycles}\n")
        f.write("\nLayers in execution order:\n")
        f.write("layer,start_call,end_call,calls,cycles\n")
        for row in layer_rows:
            f.write(
                f"{row['layer']},{row['start_call']},{row['end_call']},{row['calls']},{row['cycles']}\n"
            )
        f.write("\nLayers ranked by cycles:\n")
        f.write("rank,layer,cycles,percent,start_call,end_call,calls\n")
        for rank, row in enumerate(ranked_rows, start=1):
            pct = 100.0 * row["cycles"] / total_profiled_cycles if total_profiled_cycles else 0.0
            f.write(
                f"{rank},{row['layer']},{row['cycles']},{pct:.3f},{row['start_call']},{row['end_call']},{row['calls']}\n"
            )

    return summary_csv_path, summary_txt_path, call_map_csv_path


def write_spike_component_cycle_reports(output_dir, model_name, rows):
    _, component_rows = build_tvm_component_rows(output_dir, model_name)
    row_by_call = {row["call_index"]: row for row in rows}

    component_csv_path = output_dir / "spike_component_cycles.csv"
    component_txt_path = output_dir / "spike_component_cycles.txt"
    layer_component_csv_path = output_dir / "spike_layer_component_cycles.csv"
    layer_component_txt_path = output_dir / "spike_layer_component_cycles.txt"
    call_map_csv_path = output_dir / "spike_call_component_map.csv"

    grouped_components = defaultdict(lambda: {"cycles": 0, "calls": 0})
    grouped_layer_components = defaultdict(lambda: {"cycles": 0, "calls": 0})
    covered_calls = set()
    total_cycles = 0

    with open(call_map_csv_path, "w") as f:
        f.write("call_index,layer,component,kernel_name,cycles\n")
        for row in component_rows:
            prof_row = row_by_call.get(row["call_index"])
            if prof_row is None:
                continue
            cycles = prof_row["cycles"]
            covered_calls.add(row["call_index"])
            total_cycles += cycles
            grouped_components[row["component"]]["cycles"] += cycles
            grouped_components[row["component"]]["calls"] += 1
            layer_component_key = (row["layer"], row["component"])
            grouped_layer_components[layer_component_key]["cycles"] += cycles
            grouped_layer_components[layer_component_key]["calls"] += 1
            f.write(
                f"{row['call_index']},{row['layer']},{row['component']},"
                f"{row['kernel_name']},{cycles}\n"
            )

    missing_calls = sorted(set(row_by_call) - covered_calls)
    if missing_calls:
        raise RuntimeError(
            f"TVM component mapping did not cover all kernel calls, first missing calls: {missing_calls[:10]}"
        )

    component_rows_out = sorted(
        (
            {
                "component": component,
                "cycles": stats["cycles"],
                "calls": stats["calls"],
            }
            for component, stats in grouped_components.items()
        ),
        key=lambda row: row["cycles"],
        reverse=True,
    )
    with open(component_csv_path, "w") as f:
        f.write("component,cycles,calls,percent\n")
        for row in component_rows_out:
            pct = 100.0 * row["cycles"] / total_cycles if total_cycles else 0.0
            f.write(f"{row['component']},{row['cycles']},{row['calls']},{pct:.6f}\n")
    with open(component_txt_path, "w") as f:
        f.write("Spike rdcycle TVM component breakdown\n")
        f.write(f"Profiled cycles: {total_cycles}\n")
        f.write("rank,component,cycles,percent,calls\n")
        for rank, row in enumerate(component_rows_out, start=1):
            pct = 100.0 * row["cycles"] / total_cycles if total_cycles else 0.0
            f.write(f"{rank},{row['component']},{row['cycles']},{pct:.3f},{row['calls']}\n")

    layer_component_rows_out = sorted(
        (
            {
                "layer": layer,
                "component": component,
                "cycles": stats["cycles"],
                "calls": stats["calls"],
            }
            for (layer, component), stats in grouped_layer_components.items()
        ),
        key=lambda row: row["cycles"],
        reverse=True,
    )
    with open(layer_component_csv_path, "w") as f:
        f.write("layer,component,cycles,calls,percent\n")
        for row in layer_component_rows_out:
            pct = 100.0 * row["cycles"] / total_cycles if total_cycles else 0.0
            f.write(
                f"{row['layer']},{row['component']},{row['cycles']},{row['calls']},{pct:.6f}\n"
            )
    with open(layer_component_txt_path, "w") as f:
        f.write("Spike rdcycle TVM layer-component breakdown\n")
        f.write(f"Profiled cycles: {total_cycles}\n")
        f.write("rank,layer,component,cycles,percent,calls\n")
        for rank, row in enumerate(layer_component_rows_out, start=1):
            pct = 100.0 * row["cycles"] / total_cycles if total_cycles else 0.0
            f.write(
                f"{rank},{row['layer']},{row['component']},{row['cycles']},{pct:.3f},{row['calls']}\n"
            )

    return (
        component_csv_path,
        component_txt_path,
        layer_component_csv_path,
        layer_component_txt_path,
        call_map_csv_path,
    )


def create_errno_stub(output_dir):
    """Create syscalls.c with __errno support."""
    fixed_dir = output_dir / "fixed_syscalls"
    fixed_dir.mkdir(exist_ok=True)

    syscalls_content = """
#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>
#include <limits.h>

#define SYS_write 64

static int __errno_value = 0;
int* __errno(void) { return &__errno_value; }
int errno;

extern volatile uint64_t tohost;
extern volatile uint64_t fromhost;

static uintptr_t syscall(uintptr_t which, uint64_t arg0, uint64_t arg1, uint64_t arg2) {
    volatile uint64_t magic_mem[8] __attribute__((aligned(64)));
    magic_mem[0] = which;
    magic_mem[1] = arg0;
    magic_mem[2] = arg1;
    magic_mem[3] = arg2;
    __sync_synchronize();
    tohost = (uintptr_t)magic_mem;
    while (fromhost == 0);
    fromhost = 0;
    __sync_synchronize();
    return magic_mem[0];
}

void __attribute__((noreturn)) tohost_exit(uintptr_t code) {
    tohost = (code << 1) | 1;
    while (1);
}

void exit(int code) { tohost_exit(code); }
void abort() { exit(128 + 6); }

void printstr(const char* s) {
    const char* p = s;
    while (*p) p++;
    syscall(SYS_write, 1, (uintptr_t)s, p - s);
}

void __attribute__((weak)) thread_entry(int cid, int nc) {
    while (cid != 0);
}

int __attribute__((weak)) main(int argc, char** argv) {
    printstr("Implement main()!\\n");
    return -1;
}

void* memcpy(void* dest, const void* src, size_t len) {
    volatile char* d = (volatile char*)dest;
    volatile const char* s = (volatile const char*)src;
    while (len-- > 0) *d++ = *s++;
    return dest;
}

void* memset(void* dest, int byte, size_t len) {
    volatile char* d = (volatile char*)dest;
    while (len-- > 0) *d++ = (char)byte;
    return dest;
}

size_t strlen(const char *s) {
    const char *p = s;
    while (*p) p++;
    return p - s;
}

size_t strnlen(const char *s, size_t n) {
    const char *p = s;
    while (n-- && *p) p++;
    return p - s;
}

int strcmp(const char* s1, const char* s2) {
    unsigned char c1, c2;
    do { c1 = *s1++; c2 = *s2++; } while (c1 != 0 && c1 == c2);
    return c1 - c2;
}

char* strcpy(char* dest, const char* src) {
    char* d = dest;
    while ((*d++ = *src++));
    return dest;
}

static void init_tls() {}

void _init(int cid, int nc) {
    init_tls();
    thread_entry(cid, nc);
    int ret = main(0, 0);
    exit(ret);
}

#undef putchar
int putchar(int ch) {
    static char buf[64] __attribute__((aligned(64)));
    static int buflen = 0;
    buf[buflen++] = ch;
    if (ch == '\\n' || buflen == sizeof(buf)) {
        syscall(SYS_write, 1, (uintptr_t)buf, buflen);
        buflen = 0;
    }
    return 0;
}

int printf(const char* fmt, ...) { printstr(fmt); return 0; }
int sprintf(char* str, const char* fmt, ...) { strcpy(str, fmt); return strlen(str); }

static void print_hex64(uintptr_t v) {
    char buf[19];
    buf[0] = '0'; buf[1] = 'x';
    for (int i = 0; i < 16; i++) {
        int nib = (v >> ((15 - i) * 4)) & 0xf;
        buf[2 + i] = nib < 10 ? '0' + nib : 'a' + nib - 10;
    }
    buf[18] = 0;
    printstr(buf);
}

uintptr_t __attribute__((weak)) handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32]) {
    uintptr_t tval;
    asm volatile("csrr %0, mtval" : "=r"(tval));
    printstr("\\nTRAP mcause="); print_hex64(cause);
    printstr(" mepc="); print_hex64(epc);
    printstr(" mtval="); print_hex64(tval);
    printstr("\\n");
    tohost_exit(1337);
}
"""

    with open(fixed_dir / "syscalls.c", "w") as f:
        f.write(syscalls_content)

    return fixed_dir


def fix_gemmini_includes(output_dir):
    """Fix Gemmini header include paths."""
    import re

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    tvm_home = os.environ.get("TVM_HOME", str(repo_root / "tvm-gemmini"))
    gemmini_rocc_tests = f"{tvm_home}/3rdparty/gemmini/software/gemmini-rocc-tests"
    gemmini_include = pathlib.Path(gemmini_rocc_tests) / "include"

    fixed_include_dir = output_dir / "fixed_include"
    fixed_include_dir.mkdir(exist_ok=True)

    for header_file in gemmini_include.glob("*.h"):
        with open(header_file, "r") as f:
            content = f.read()
        content = re.sub(r'#include\s*"include/([^"]+)"', r'#include "\1"', content)
        content = re.sub(
            r'#include\s*"rocc-software/src/([^"]+)"', r'#include "\1"', content
        )
        with open(fixed_include_dir / header_file.name, "w") as f:
            f.write(content)

    rocc_src = pathlib.Path(gemmini_rocc_tests) / "rocc-software" / "src"
    if rocc_src.exists():
        for header_file in rocc_src.glob("*.h"):
            shutil.copy(header_file, fixed_include_dir / header_file.name)

    return fixed_include_dir


def create_tvm_stubs(output_dir):
    """Create TVM runtime stubs."""
    stub_dir = output_dir / "tvm_stubs" / "tvm" / "runtime"
    stub_dir.mkdir(parents=True, exist_ok=True)

    with open(stub_dir / "c_runtime_api.h", "w") as f:
        f.write("""
#ifndef TVM_RUNTIME_C_RUNTIME_API_H_
#define TVM_RUNTIME_C_RUNTIME_API_H_
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
#ifndef TVM_DLL
#define TVM_DLL
#endif
typedef int32_t tvm_index_t;
typedef void* TVMValue;
typedef int32_t TVMArrayHandle;
#define TVM_ASSERT(x) ((void)0)
#ifdef __cplusplus
}
#endif
#endif
""")

    with open(stub_dir / "c_backend_api.h", "w") as f:
        f.write("""
#ifndef TVM_RUNTIME_C_BACKEND_API_H_
#define TVM_RUNTIME_C_BACKEND_API_H_
#include <stdint.h>
#include <stddef.h>
#include "tvmgen_default.h"
#ifdef __cplusplus
extern "C" {
#endif
#ifndef TVM_DLL
#define TVM_DLL
#endif
#define TVM_BACKEND_WORKSPACE_SIZE \
    (TVMGEN_DEFAULT_WORKSPACE_SIZE + (64 * 1024 * 1024))
static char __tvm_workspace[TVM_BACKEND_WORKSPACE_SIZE];
static size_t __tvm_workspace_offset = 0;
static inline void* TVMBackendAllocWorkspace(int device_type, int device_id,
                                              uint64_t nbytes, int dtype_code_hint,
                                              int dtype_bits_hint) {
    void* ptr = &__tvm_workspace[__tvm_workspace_offset];
    __tvm_workspace_offset += ((nbytes + 15) / 16) * 16;
    return ptr;
}
static inline int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
    return 0;
}
#ifdef __cplusplus
}
#endif
#endif
""")

    return stub_dir


def _ndarray_byte_size(ndarr):
    dtype = ndarr.dtype
    if isinstance(dtype, str):
        match = re.match(r"^(?:float|int|uint)(?P<bits>\d+)$", dtype)
        if not match:
            raise RuntimeError(f"Unsupported ndarray dtype string: {dtype}")
        bits = int(match.group("bits"))
        lanes = 1
    else:
        bits = dtype.bits
        lanes = dtype.lanes
    elem_bytes = (bits * lanes + 7) // 8
    size = 1
    for dim in ndarr.shape:
        size *= int(dim)
    return size * elem_bytes


def _accumulate_constant_pool_bytes(pool_info):
    const_infos = sorted(pool_info.constant_info_array, key=lambda ci: int(ci.byte_offset))
    if not const_infos:
        return b""
    total = int(const_infos[-1].byte_offset) + _ndarray_byte_size(const_infos[-1].data)
    blob = bytearray(total)
    for ci in const_infos:
        off = int(ci.byte_offset)
        arr = ci.data.asnumpy().tobytes()
        blob[off : off + len(arr)] = arr
    return bytes(blob)


def _format_c_byte_array(data_bytes, indent="    ", line_width=16):
    if not data_bytes:
        return indent + "0x00"
    lines = []
    row = indent
    for i, byte_val in enumerate(data_bytes):
        token = f"0x{byte_val:02x}"
        if i % line_width == 0:
            if i > 0:
                lines.append(row.rstrip() + ",")
            row = indent + token
        else:
            row += ", " + token
    if row.strip():
        lines.append(row.rstrip())
    return "\n".join(lines)


def generate_llvm_aot_shim(output_dir, module):
    """Emit constants/workspace shim + tvmgen_default_run for LLVM AOT objects."""
    const_blob = b""
    workspace_size = None
    for allocated in dict(module.executor_codegen_metadata.pool_inputs).values():
        pool_info = allocated.pool_info
        if isinstance(pool_info, ConstantPoolInfo):
            const_blob = _accumulate_constant_pool_bytes(pool_info)
        elif pool_info.pool_name == "global_workspace":
            workspace_size = int(allocated.allocated_size)

    if workspace_size is None:
        header_path = output_dir / "codegen" / "host" / "include" / "tvmgen_default.h"
        match = re.search(
            r"TVMGEN_DEFAULT_WORKSPACE_SIZE\s+(\d+)", header_path.read_text()
        )
        if not match:
            raise RuntimeError(f"Could not determine workspace size from {header_path}")
        workspace_size = int(match.group(1))

    shim_dir = output_dir / "codegen" / "host" / "src"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim_path = shim_dir / "llvm_aot_shim.c"
    const_init = _format_c_byte_array(const_blob)
    shim_source = f"""#include <stdint.h>
#include "tvm/runtime/c_runtime_api.h"
#include "tvmgen_default.h"

#ifdef __cplusplus
extern "C" {{
#endif

__attribute__((section(".rodata.tvm"), aligned(16)))
static uint8_t global_const_workspace[{len(const_blob)}] = {{
{const_init}
}};

__attribute__((section(".bss.noinit.tvm"), aligned(16)))
static uint8_t global_workspace[{workspace_size}];

TVM_DLL int32_t tvmgen_default___tvm_main__(
    void* data, void* output0, uint8_t* global_const_workspace_0_var,
    uint8_t* global_workspace_1_var);

int32_t tvmgen_default_run(
    struct tvmgen_default_inputs* inputs, struct tvmgen_default_outputs* outputs) {{
  return tvmgen_default___tvm_main__(
      inputs->data, outputs->output, global_const_workspace, global_workspace);
}}

#ifdef __cplusplus
}}
#endif
"""
    shim_path.write_text(shim_source)
    print(
        f"[Info] Generated LLVM AOT shim: {shim_path} "
        f"(constants={len(const_blob)} bytes, workspace={workspace_size} bytes)"
    )
    return shim_path


def count_rvv_instructions(output_dir, *, object_name="default_lib1.o"):
    """Count RVV instructions in a precompiled LLVM object via objdump."""
    riscv = os.environ.get("RISCV", "/root/flexi/chipyard/.conda-env/riscv-tools")
    objdump = f"{riscv}/bin/riscv64-unknown-elf-objdump"
    obj_path = output_dir / "codegen" / "host" / "lib" / object_name
    if not obj_path.exists():
        print(f"[WARN] RVV check skipped; object not found: {obj_path}")
        return 0

    result = subprocess.run([objdump, "-d", str(obj_path)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[WARN] objdump failed for {obj_path}: {result.stderr}")
        return 0

    rvv_mnemonics = (
        "vsetvli",
        "vsetivli",
        "vle",
        "vse",
        "vlse",
        "vsse",
        "vluxei",
        "vsuxei",
        "vadd",
        "vsub",
        "vmul",
        "vdiv",
        "vfadd",
        "vfsub",
        "vfmul",
        "vfmacc",
        "vfnmacc",
        "vmacc",
        "vnmsac",
        "vslide",
        "vmv",
        "vmfeq",
        "vmfne",
        "vmfgt",
        "vmfge",
        "vmflt",
        "vmfle",
        "vmerge",
        "vmand",
        "vmor",
        "vmxor",
        "vmsbc",
        "vadc",
        "vcompress",
        "viota",
        "vid",
        "vrgather",
        "vred",
        "vwred",
        "vfred",
        "vfwred",
        "vcpop",
        "vfirst",
        "vmsbf",
        "vmsif",
        "vmsof",
        "viota.m",
        "vssrl",
        "vssra",
        "vnclip",
        "vnclipu",
    )
    pattern = re.compile(r"\b(" + "|".join(re.escape(m) for m in rvv_mnemonics) + r")\b")
    count = 0
    samples = []
    for line in result.stdout.splitlines():
        if pattern.search(line):
            count += 1
            if len(samples) < 8:
                samples.append(line.strip())
    print(f"[RVV] {object_name}: {count} vector instructions")
    vlenb_hits = [line.strip() for line in result.stdout.splitlines() if "vlenb" in line]
    if vlenb_hits:
        print(f"[WARN] {object_name}: {len(vlenb_hits)} csrr vlenb (Spike 1.1.x may trap)")
        for sample in vlenb_hits[:4]:
            print(f"       {sample}")
    else:
        print(f"[RVV] {object_name}: no csrr vlenb (fixed VLEN, Spike-safe)")
    for sample in samples:
        print(f"       {sample}")
    return count


def compile_for_spike_llvm_gemmini(
    output_dir,
    test_name="ivit_real",
    *,
    riscv_march="rv64gc",
    gcc_opt_level=2,
    extra_link_flags=None,
):
    """Link precompiled LLVM host objects + Gemmini C sources for Spike."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    tvm_home = os.environ.get("TVM_HOME", str(repo_root / "tvm-gemmini"))
    riscv = os.environ.get("RISCV", "/root/flexi/chipyard/.conda-env/riscv-tools")

    gemmini_rocc_tests = f"{tvm_home}/3rdparty/gemmini/software/gemmini-rocc-tests"
    riscv_tests = f"{gemmini_rocc_tests}/riscv-tests"
    bench_common = f"{riscv_tests}/benchmarks/common"
    cc = f"{riscv}/bin/riscv64-unknown-elf-gcc"

    codegen_dir = output_dir / "codegen" / "host"
    lib_dir = codegen_dir / "lib"
    llvm_objects = sorted(lib_dir.glob("*.o"))
    if not llvm_objects:
        print(f"[ERROR] No LLVM objects found under {lib_dir}")
        return None

    shim_path = codegen_dir / "src" / "llvm_aot_shim.c"
    if not shim_path.exists():
        print(f"[ERROR] Missing LLVM AOT shim: {shim_path}")
        return None

    fixed_include = fix_gemmini_includes(output_dir)
    create_tvm_stubs(output_dir)
    fixed_syscalls_dir = create_errno_stub(output_dir)

    cflags = [
        "-DPREALLOCATE=1",
        "-DMULTITHREAD=1",
        "-mcmodel=medany",
        "-std=gnu99",
        f"-O{gcc_opt_level}",
        "-ffast-math",
        "-fno-common",
        "-fno-builtin-printf",
        "-fno-builtin-memset",
        "-fno-builtin-memcpy",
        "-fno-tree-loop-distribute-patterns",
        f"-march={riscv_march}",
        "-mabi=lp64d",
        "-mrelax",
        "-nostdlib",
        "-nostartfiles",
        "-static",
        f"-T{bench_common}/test.ld",
        "-DBAREMETAL=1",
        "-DTVM_RUNTIME_ALLOC",
        f"-I{riscv_tests}",
        f"-I{riscv_tests}/env",
        f"-I{fixed_include}",
        f"-I{gemmini_rocc_tests}",
        f"-I{gemmini_rocc_tests}/include",
        f"-I{bench_common}",
        f"-I{codegen_dir}/src",
        f"-I{codegen_dir}/include",
        f"-I{output_dir}/tvm_stubs",
        "-DPRINT_TILE=0",
    ]

    source_files = [str(output_dir / "main.c"), str(shim_path)]
    source_files.append(str(fixed_syscalls_dir / "syscalls.c"))
    source_files += [str(f) for f in pathlib.Path(bench_common).glob("*.S")]
    source_files += [
        str(f)
        for f in (codegen_dir / "src").glob("*.c")
        if f.name != "llvm_aot_shim.c"
    ]

    build_dir = output_dir / "build"
    obj_dir = build_dir / "obj"
    build_dir.mkdir(exist_ok=True)
    obj_dir.mkdir(exist_ok=True)

    # Place small runtime objects (crt/syscalls/main/gemmini C) FIRST so their
    # .text (incl. _init) sits close to _start; the huge LLVM object goes LAST.
    # Otherwise a multi-MB default_lib1.o pushes _init out of crt's `j _init`
    # (R_RISCV_JAL, +-1MB) range -> "relocation truncated to fit".
    object_files = []
    for src in source_files:
        obj_path = obj_dir / (pathlib.Path(src).stem + ".o")
        compile_cmd = [cc] + cflags + ["-c", src, "-o", str(obj_path)]
        compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if compile_result.returncode != 0:
            print(f"[ERROR] Compilation failed for {src}:")
            print(compile_result.stderr)
            return None
        object_files.append(str(obj_path))
    object_files += [str(obj) for obj in llvm_objects]

    output_binary = build_dir / f"{test_name}-baremetal"
    cmd = (
        [cc]
        + cflags
        + object_files
        + ["-o", str(output_binary), "-lm", "-lgcc", "-Wl,--relax"]
        + list(extra_link_flags or [])
    )
    print(f"[Compile] Linking llvm-gemmini Spike binary ({len(llvm_objects)} LLVM objects)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR] llvm-gemmini link failed:")
        print(result.stderr)
        return None

    print(f"[OK] Binary: {output_binary}")
    return output_binary


def compile_for_spike(
    output_dir,
    test_name="ivit_real",
    *,
    riscv_march="rv64gc",
    gcc_opt_level=2,
    gcc_autovec=False,
):
    """Compile for Spike."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    tvm_home = os.environ.get("TVM_HOME", str(repo_root / "tvm-gemmini"))
    riscv = os.environ.get("RISCV", "/root/flexi/chipyard/.conda-env/riscv-tools")

    gemmini_rocc_tests = f"{tvm_home}/3rdparty/gemmini/software/gemmini-rocc-tests"
    riscv_tests = f"{gemmini_rocc_tests}/riscv-tests"
    bench_common = f"{riscv_tests}/benchmarks/common"

    cc = f"{riscv}/bin/riscv64-unknown-elf-gcc"

    codegen_dir = output_dir / "codegen" / "host"
    fixed_include = fix_gemmini_includes(output_dir)
    create_tvm_stubs(output_dir)
    fixed_syscalls_dir = create_errno_stub(output_dir)

    cflags = [
        "-DPREALLOCATE=1",
        "-DMULTITHREAD=1",
        "-mcmodel=medany",
        "-std=gnu99",
        f"-O{gcc_opt_level}",
        "-ffast-math",
        "-fno-common",
        "-fno-builtin-printf",
        "-fno-builtin-memset",
        "-fno-builtin-memcpy",
        "-fno-tree-loop-distribute-patterns",
        f"-march={riscv_march}",
        "-mrelax",
        "-nostdlib",
        "-nostartfiles",
        "-static",
        f"-T{bench_common}/test.ld",
        "-DBAREMETAL=1",
        "-DTVM_RUNTIME_ALLOC",
        f"-I{riscv_tests}",
        f"-I{riscv_tests}/env",
        f"-I{fixed_include}",
        f"-I{gemmini_rocc_tests}",
        f"-I{gemmini_rocc_tests}/include",
        f"-I{bench_common}",
        f"-I{codegen_dir}/src",
        f"-I{codegen_dir}/include",
        f"-I{output_dir}/tvm_stubs",
        "-DPRINT_TILE=0",
    ]
    if gcc_autovec:
        cflags.append("-ftree-vectorize")

    source_files = [str(output_dir / "main.c")]
    source_files += [str(fixed_syscalls_dir / "syscalls.c")]
    source_files += [str(f) for f in pathlib.Path(bench_common).glob("*.S")]
    source_files += [str(f) for f in (codegen_dir / "src").glob("*.c")]

    build_dir = output_dir / "build"
    obj_dir = build_dir / "obj"
    build_dir.mkdir(exist_ok=True)
    obj_dir.mkdir(exist_ok=True)
    output_binary = build_dir / f"{test_name}-baremetal"

    object_files = []
    for src in source_files:
        obj_path = obj_dir / (pathlib.Path(src).stem + ".o")
        compile_cmd = [cc] + cflags + ["-c", src, "-o", str(obj_path)]
        compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if compile_result.returncode != 0:
            print(f"[ERROR] Compilation failed for {src}:")
            print(compile_result.stderr)
            return None
        object_files.append(str(obj_path))

    cmd = (
        [cc]
        + cflags
        + object_files
        + ["-o", str(output_binary), "-lm", "-lgcc", "-Wl,--relax"]
    )

    print(f"[Compile] Building {test_name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] Compilation failed:")
        print(result.stderr)
        return None

    print(f"[OK] Binary: {output_binary}")
    return output_binary


def run_spike(binary_path, timeout=600, spike_isa=None):
    """Run on Spike."""
    riscv = os.environ.get("RISCV", "/root/flexi/chipyard/.conda-env/riscv-tools")
    spike = f"{riscv}/bin/spike"
    chipyard_lib = "/root/flexi/chipyard/.conda-env/lib"

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{chipyard_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    if "LD_PRELOAD" in env:
        del env["LD_PRELOAD"]

    cmd = [spike]
    if spike_isa:
        cmd.append(f"--isa={spike_isa}")
    cmd.extend(["--extension=gemmini", str(binary_path)])

    print(f"\n[Spike] Running inference...")
    timeout_arg = timeout if timeout and timeout > 0 else None

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=timeout_arg
        )
        if result.returncode != 0:
            print(f"[ERROR] Spike exited with code {result.returncode}")
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Exceeded {timeout}s")
        return None, None


def run_verilator(
    binary_path,
    timeout=600,
    chipyard_dir="/root/flexi/chipyard",
    verilator_config="BigRocketSaturnGemminiConfig",
    max_cycles=20000000000,
    dramsim=True,
    verbose=False,
    log_dir=None,
    log_tail_lines=20000,
):
    """Run on Chipyard Verilator simulator."""
    simulator = (
        pathlib.Path(chipyard_dir)
        / "sims"
        / "verilator"
        / f"simulator-chipyard.harness-{verilator_config}"
    )
    if not simulator.exists():
        print(f"[ERROR] Verilator simulator not found: {simulator}")
        return None, None, None, None

    cmd = [str(simulator), "+permissive"]
    if verbose:
        cmd.append("+verbose")

    if dramsim:
        dramsim_ini_dir = (
            pathlib.Path(chipyard_dir)
            / "generators"
            / "testchipip"
            / "src"
            / "main"
            / "resources"
            / "dramsim2_ini"
        )
        cmd += ["+dramsim", f"+dramsim_ini_dir={dramsim_ini_dir}"]

    if max_cycles and max_cycles > 0:
        cmd.append(f"+max-cycles={max_cycles}")
    else:
        print("[Info] Verilator +max-cycles is disabled")

    cmd += [
        f"+loadmem={binary_path}",
        "+permissive-off",
        str(binary_path),
    ]

    print(f"\n[Verilator] Running inference...")
    stdout_path = None
    stderr_path = None
    timeout_arg = timeout if timeout and timeout > 0 else None

    def _run_with_tailed_logs(cmd_args, timeout_sec, out_path, err_path, tail_lines):
        out_tail = deque(maxlen=tail_lines)
        err_tail = deque(maxlen=tail_lines)

        def _reader(pipe, sink):
            try:
                for line in iter(pipe.readline, ""):
                    sink.append(line)
            finally:
                pipe.close()

        proc = subprocess.Popen(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            errors="replace",
        )
        out_t = threading.Thread(target=_reader, args=(proc.stdout, out_tail), daemon=True)
        err_t = threading.Thread(target=_reader, args=(proc.stderr, err_tail), daemon=True)
        out_t.start()
        err_t.start()

        timed_out = False
        try:
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()

        out_t.join()
        err_t.join()

        with open(out_path, "w") as fout:
            fout.writelines(out_tail)
        with open(err_path, "w") as ferr:
            ferr.writelines(err_tail)
        return proc.returncode, timed_out

    try:
        if log_dir is not None:
            log_dir = pathlib.Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = log_dir / "verilator_stdout.log"
            stderr_path = log_dir / "verilator_stderr.log"

            if log_tail_lines and log_tail_lines > 0:
                print(
                    f"[Info] Saving only last {log_tail_lines} lines of Verilator logs "
                    "(set --verilator-log-tail-lines 0 for full logs)"
                )
                _, timed_out = _run_with_tailed_logs(
                    cmd, timeout_arg, stdout_path, stderr_path, log_tail_lines
                )
                if timed_out:
                    print(f"[TIMEOUT] Exceeded {timeout}s")
                    return None, None, stdout_path, stderr_path
                return "", "", stdout_path, stderr_path

            with open(stdout_path, "w") as fout, open(stderr_path, "w") as ferr:
                subprocess.run(cmd, stdout=fout, stderr=ferr, text=True, timeout=timeout_arg)
            return "", "", stdout_path, stderr_path

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_arg)
        return result.stdout, result.stderr, None, None
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Exceeded {timeout}s")
        return None, None, stdout_path, stderr_path


def _resolve_riscv_tool(tool_name):
    riscv = os.environ.get("RISCV", "/root/flexi/chipyard/.conda-env/riscv-tools")
    return pathlib.Path(riscv) / "bin" / tool_name


def decode_trace_with_spike_dasm(trace_path, decoded_path):
    """Decode verbose trace with spike-dasm for readability."""
    spike_dasm = _resolve_riscv_tool("spike-dasm")
    if not spike_dasm.exists():
        print(f"[WARN] spike-dasm not found: {spike_dasm}")
        return False
    with open(trace_path, "r") as fin, open(decoded_path, "w") as fout:
        result = subprocess.run([str(spike_dasm)], stdin=fin, stdout=fout, text=True)
    if result.returncode != 0:
        print("[WARN] spike-dasm decoding failed")
        return False
    print(f"[OK] Decoded trace: {decoded_path}")
    return True


def _load_function_ranges(binary_path):
    """Load text symbol ranges from ELF for PC->function lookup."""
    nm = _resolve_riscv_tool("riscv64-unknown-elf-nm")
    if not nm.exists():
        raise RuntimeError(f"nm not found: {nm}")
    result = subprocess.run(
        [str(nm), "-n", str(binary_path)], capture_output=True, text=True, check=True
    )
    symbols = []
    for line in result.stdout.splitlines():
        m = re.match(r"^([0-9a-fA-F]+)\s+([tT])\s+(\S+)$", line.strip())
        if not m:
            continue
        addr = int(m.group(1), 16)
        name = m.group(3)
        symbols.append((addr, name))
    if not symbols:
        raise RuntimeError("No text symbols found in binary")
    ranges = []
    for idx, (addr, name) in enumerate(symbols):
        end = symbols[idx + 1][0] if idx + 1 < len(symbols) else addr + 1
        if end > addr:
            ranges.append((addr, end, name))
    return ranges


def profile_kernels_from_trace(trace_path, binary_path, report_txt_path, report_csv_path, topk=40):
    """Attribute verbose trace cycles to kernels based on PC ranges."""
    ranges = _load_function_ranges(binary_path)
    starts = [r[0] for r in ranges]

    def find_func(pc):
        idx = bisect.bisect_right(starts, pc) - 1
        if idx < 0:
            return None
        start, end, name = ranges[idx]
        if start <= pc < end:
            return name
        return None

    cycle_re = re.compile(r"^C\d+:\s+(\d+)\s+\[\d+\]\s+pc=\[([0-9a-fA-F]+)\]")
    per_func_cycles = defaultdict(int)
    per_func_samples = defaultdict(int)

    prev_cycle = None
    prev_pc = None
    trace_points = 0
    with open(trace_path, "r") as f:
        for line in f:
            m = cycle_re.match(line)
            if not m:
                continue
            cycle = int(m.group(1))
            pc = int(m.group(2), 16)
            trace_points += 1
            if prev_cycle is not None:
                delta = cycle - prev_cycle
                if delta < 0:
                    delta = 0
                func = find_func(prev_pc)
                if func:
                    per_func_cycles[func] += delta
                    per_func_samples[func] += 1
            prev_cycle = cycle
            prev_pc = pc

    kernel_items = []
    other_cycles = 0
    for name, cycles in per_func_cycles.items():
        if name.startswith("tvmgen_default_fused_"):
            kernel_items.append((name, cycles, per_func_samples[name]))
        else:
            other_cycles += cycles
    kernel_items.sort(key=lambda x: x[1], reverse=True)
    total_kernel_cycles = sum(x[1] for x in kernel_items)
    total_cycles = total_kernel_cycles + other_cycles

    with open(report_txt_path, "w") as f:
        f.write("Kernel cycle profile from Verilator +verbose trace\n")
        f.write(f"Trace points: {trace_points}\n")
        f.write(f"Total attributed cycles: {total_cycles}\n")
        f.write(f"Kernel-attributed cycles: {total_kernel_cycles}\n")
        f.write(f"Non-kernel cycles: {other_cycles}\n")
        f.write("\nTop kernels by cycles:\n")
        f.write("rank,cycles,percent_of_kernels,samples,kernel\n")
        for rank, (name, cycles, samples) in enumerate(kernel_items[:topk], start=1):
            pct = (100.0 * cycles / total_kernel_cycles) if total_kernel_cycles else 0.0
            f.write(f"{rank},{cycles},{pct:.3f},{samples},{name}\n")

    with open(report_csv_path, "w") as f:
        f.write("kernel,cycles,samples\n")
        for name, cycles, samples in kernel_items:
            f.write(f"{name},{cycles},{samples}\n")

    return {
        "trace_points": trace_points,
        "total_cycles": total_cycles,
        "total_kernel_cycles": total_kernel_cycles,
        "other_cycles": other_cycles,
        "top": kernel_items[:topk],
    }


def main():
    parser = argparse.ArgumentParser(description="Run I-ViT on real image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--checkpoint", type=str, default="/root/checkpoint_last.pth.tar"
    )
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Use a random initialized QAT state_dict when --checkpoint is absent.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="auto",
        choices=[
            "auto",
            "deit_tiny_patch16_224",
            "deit_small_patch16_224",
            "fq_deit_tiny_patch16_224",
            "ptq4_deit_tiny_patch16_224",
            "ptq4_deit_small_patch16_224",
            "swin_tiny_patch4_window7_224",
            "swin_small_patch4_window7_224",
        ],
        help="Model name (auto detects from checkpoint keys)",
    )
    parser.add_argument("--output-dir", type=str, default="ivit_real_image_project")
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Host-side timeout in seconds (0 to disable)",
    )
    parser.add_argument(
        "--simulator",
        type=str,
        default="spike",
        choices=["spike", "verilator"],
        help="Simulator backend",
    )
    parser.add_argument(
        "--chipyard-dir",
        type=str,
        default="/root/flexi/chipyard",
        help="Chipyard root path (used for Verilator)",
    )
    parser.add_argument(
        "--verilator-config",
        type=str,
        default="BigRocketSaturnGemminiConfig",
        help="Chipyard config name for Verilator simulator",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Verilator +max-cycles limit (0 to disable simulator-side timeout)",
    )
    parser.add_argument(
        "--no-dramsim",
        action="store_true",
        help="Disable +dramsim when running Verilator",
    )
    parser.add_argument(
        "--debug-unit",
        type=str,
        default=None,
        help=(
            "Relay debug cut point (e.g. post_block0, block_0_pre_softmax, "
            "post_stage0_block0, only_stage0_block0, only_stage0_downsample, "
            "post_stem, pre_head, head_int)"
        ),
    )
    parser.add_argument(
        "--tvm-backend",
        type=str,
        default="c-gemmini",
        choices=["c-gemmini", "llvm-gemmini"],
        help=(
            "TVM codegen backend: c-gemmini (default scalar C) or "
            "llvm-gemmini (Path B heterogeneous LLVM CPU + Gemmini BYOC)"
        ),
    )
    parser.add_argument(
        "--llvm-tir-vectorize",
        action="store_true",
        help=(
            "With --tvm-backend llvm-gemmini: enable TIR VectorizeLoop "
            "(TE split+vectorize / Case A) instead of LLVM-only autovec (Case B)."
        ),
    )
    parser.add_argument(
        "--llvm-no-loop-vectorize",
        action="store_true",
        help=(
            "With llvm-gemmini: disable LLVM LoopVectorize/SLP "
            "(-vectorize-loops=false -vectorize-slp=false). Use with "
            "--llvm-tir-vectorize so RVV comes only from TIR VectorizeLoop lowering."
        ),
    )
    parser.add_argument(
        "--llvm-tir-max-vf",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap TE/TIR injective vectorize factor (sets TVM_TOPI_INJECTIVE_MAX_VF). "
            "For Swin e2e + --llvm-tir-vectorize, default becomes 16 if unset."
        ),
    )
    parser.add_argument(
        "--llvm-opt-level",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help=(
            "LLVM Target opt-level for llvm-gemmini host (default 3). "
            "For Swin e2e + --llvm-tir-vectorize, default becomes 2 if unset."
        ),
    )
    parser.add_argument(
        "--dump-tir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Dump final lowered TIR PrimFuncs (phase-3, pre-codegen) into DIR. "
            "Use with --llvm-tir-vectorize on/off to compare ramp/broadcast vs scalar."
        ),
    )
    parser.add_argument(
        "--llvm-rvv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable RVV (+v,+zvl512b) LLVM autovec on host epilogues (llvm-gemmini; Saturn/FireSim)",
    )
    parser.add_argument(
        "--llvm-fast-math",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "With llvm-gemmini: run Relay FastMath (softmax/exp/erf/tanh → polynomial "
            "fast_* ops) so TIR VectorizeLoop and LLVM RVV can vectorize float "
            "epilogues instead of scalar libm expf/erff. Default on; use "
            "--no-llvm-fast-math for libm ablation."
        ),
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Stop after exporting and compiling the baremetal ELF.",
    )
    parser.add_argument(
        "--riscv-march",
        type=str,
        default="rv64gc",
        help="RISC-V march for gcc link/compile (e.g. rv64gc, rv64gcv for Path A autovec)",
    )
    parser.add_argument(
        "--gcc-opt-level",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="gcc optimization level for baremetal link",
    )
    parser.add_argument(
        "--gcc-autovec",
        action="store_true",
        help="Enable gcc -ftree-vectorize (Path A: scalar C + RVV autovec)",
    )
    parser.add_argument(
        "--spike-isa",
        type=str,
        default=None,
        help="Spike --isa string (default: --riscv-march when it contains 'v')",
    )
    parser.add_argument(
        "--force-synthetic-input",
        action="store_true",
        help="Use deterministic synthetic input even for image-shaped model inputs.",
    )
    parser.add_argument(
        "--synthetic-seed",
        type=int,
        default=0,
        help="Seed for deterministic synthetic input generation.",
    )
    parser.add_argument(
        "--verilator-verbose",
        action="store_true",
        help="Pass +verbose to Verilator and save raw trace logs",
    )
    parser.add_argument(
        "--decode-dasm",
        action="store_true",
        help="Decode verbose trace via spike-dasm into a readable .dasm file",
    )
    parser.add_argument(
        "--profile-kernels",
        action="store_true",
        help="Profile TVM kernels: exact rdcycle on Spike, PC-attributed cycles on Verilator",
    )
    parser.add_argument(
        "--profile-semantic",
        action="store_true",
        help=(
            "Emit TVM semantic rdcycle segments from the generated full graph. "
            "Works on both Spike and Verilator and is better suited for component attribution."
        ),
    )
    parser.add_argument(
        "--profile-intrakernel-requant",
        action="store_true",
        help=(
            "Emit rdcycle totals for generated fixed_point_multiply/requant/post-op kernels. "
            "This is safe to combine with --profile-semantic and reports whole lowered CPU-side kernels."
        ),
    )
    parser.add_argument(
        "--profile-semantic-style",
        type=str,
        default="semantic",
        choices=["semantic", "aligned", "aligned_split"],
        help=(
            "Component naming style for --profile-semantic. "
            "'aligned' uses explicit names like norm1/norm2/proj_res1/fc2_res2. "
            "'aligned_split' further separates clean gemm vs requant/post-op regions "
            "such as fc1_gemm/fc1_requant."
        ),
    )
    parser.add_argument(
        "--profile-layer-filter",
        action="append",
        default=[],
        help="Only emit semantic rows for this exact layer name. Repeat to keep multiple layers.",
    )
    parser.add_argument(
        "--profile-component-filter",
        action="append",
        default=[],
        help="Only emit semantic rows for this exact component name. Repeat to keep multiple components.",
    )
    parser.add_argument(
        "--profile-topk",
        type=int,
        default=40,
        help="How many kernels to show in text report",
    )
    parser.add_argument(
        "--verilator-log-tail-lines",
        type=int,
        default=20000,
        help=(
            "When saving Verilator logs, keep only last N lines (default: 20000). "
            "Set 0 to save full logs."
        ),
    )
    parser.add_argument(
        "--verilator-save-logs",
        action="store_true",
        help="Always save Verilator stdout/stderr logs into --output-dir, even without +verbose",
    )
    parser.add_argument(
        "--uart-mode",
        type=str,
        default="full",
        choices=["full", "minimal"],
        help=(
            "UART verbosity for the generated baremetal harness. "
            "'minimal' prints a single [TVM_RUN_RESULT] line to reduce Verilator UART overhead."
        ),
    )
    parser.add_argument(
        "--usmp-alg",
        type=str,
        default=None,
        choices=["hill_climb", "greedy_by_size", "greedy_by_conflicts", "none"],
        help="USMP algorithm (default: auto by model)",
    )
    parser.add_argument(
        "--opt-level",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help="Relay build opt level (default: auto by model)",
    )
    parser.add_argument(
        "--disable-qnn-canonicalize",
        action="store_true",
        help=(
            "Skip relay.qnn.transform.CanonicalizeOps in Swin preprocess to keep "
            "ops like qnn.conv2d in Relay form."
        ),
    )
    parser.add_argument(
        "--enable-qnn-canonicalize",
        action="store_true",
        help=(
            "Enable relay.qnn.transform.CanonicalizeOps in Swin preprocess. "
            "Default is disabled to preserve qnn ops in Relay."
        ),
    )
    args = parser.parse_args()

    if args.profile_kernels and args.profile_semantic:
        print("[ERROR] --profile-kernels cannot be combined with --profile-semantic")
        return 1

    image_path = pathlib.Path(args.image)

    print("=" * 60)
    print("I-ViT Real Image Inference on Gemmini")
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

    print("\n[1/6] Loading checkpoint...")
    checkpoint_path = pathlib.Path(args.checkpoint).expanduser()
    requested_model_name = None if args.model_name == "auto" else args.model_name
    if (
        not checkpoint_path.exists()
        and not args.allow_random_init
        and not (requested_model_name and requested_model_name.startswith("ptq4_deit_"))
    ):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return 1

    # PTQ4 uses flexi e2e_model.pt (not I-ViT qconfig checkpoint).
    if requested_model_name and requested_model_name.startswith("ptq4_deit_"):
        model_name = requested_model_name
        ckpt = None
        print(f"       Model: {model_name} (PTQ4 flexi checkpoint via get_workload)")
        if model_name not in MODEL_SPECS:
            print(f"[ERROR] Unsupported model for this runner: {model_name}")
            return 1
        depth = MODEL_SPECS[model_name]["depth"]
        input_scale = None
        print("\n[2/6] Preparing model input...")
        mod, params = get_workload(model_name, batch_size=1, debug_unit=args.debug_unit)
        mod = relay.transform.InferType()(mod)
        input_spec = _extract_main_input_spec(mod)
        if (
            not args.force_synthetic_input
            and tuple(input_spec["shape"]) == (1, 3, 224, 224)
            and input_spec["dtype"] == "float32"
            and not image_path.exists()
            and not args.force_synthetic_input
        ):
            # Prefer flexi golden input when image missing
            from models.ivit.ptq4_checkpoint import DEFAULT_PTQ4_DEIT_T_INPUT
            if model_name.endswith("_tiny_patch16_224") and DEFAULT_PTQ4_DEIT_T_INPUT.is_file():
                input_data = np.fromfile(DEFAULT_PTQ4_DEIT_T_INPUT, dtype="<f4").reshape(1, 3, 224, 224)
                input_mode = "flexi_golden_f32"
            else:
                input_data, input_mode = prepare_input_data(
                    image_path,
                    input_scale,
                    input_spec,
                    force_synthetic=True,
                    model_name=model_name,
                )
        else:
            if (
                not args.force_synthetic_input
                and tuple(input_spec["shape"]) == (1, 3, 224, 224)
                and input_spec["dtype"] == "float32"
                and not image_path.exists()
            ):
                print(f"[ERROR] Image not found: {image_path}")
                return 1
            input_data, input_mode = prepare_input_data(
                image_path,
                input_scale,
                input_spec,
                force_synthetic=args.force_synthetic_input,
                model_name=model_name,
            )
        print(f"       Input mode: {input_mode}")
        print(f"       Input tensor shape: {input_data.shape}, dtype: {input_data.dtype}")
        # Jump into shared build path by setting flags used below — fall through via goto-like structure.
        # We set variables expected after the common input-prep block.
        skip_common_load = True
    else:
        skip_common_load = False

    if not skip_common_load:
      if checkpoint_path.exists():
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        model_name = convert_model.resolve_model_name(ckpt, requested_model_name)
      else:
        if requested_model_name is None:
            print("[ERROR] --model-name is required with --allow-random-init when checkpoint is absent")
            return 1
        model_name = requested_model_name
        print(f"       [WARN] Using random-initialized state for {model_name}")
        ckpt = build_random_qat_state_dict(model_name)
      if model_name not in MODEL_SPECS:
        print(f"[ERROR] Unsupported model for this runner: {model_name}")
        return 1

      depth = MODEL_SPECS[model_name]["depth"]
      convert_model.load_qconfig(ckpt, depth=depth, model_name=model_name)
      print(f"       Checkpoint: {checkpoint_path if checkpoint_path.exists() else '<random-init>'}")
      print(f"       Model: {model_name}")

      input_scale = None
      if model_name.startswith("fq_deit_"):
        input_scale = ckpt["qact_input.scale"].item()
        print(f"       FQ input scale (qact_input.scale): {input_scale}")
      else:
        input_scale = ckpt["qact_input.act_scaling_factor"].item()
        print(f"       Input quantization scale: {input_scale}")

      print("\n[2/6] Preparing model input...")
      mod, params = get_workload(model_name, batch_size=1, debug_unit=args.debug_unit)
      mod = relay.transform.InferType()(mod)
      input_spec = _extract_main_input_spec(mod)
      if (
        not args.force_synthetic_input
        and _is_real_image_input(input_spec["shape"], input_spec["dtype"])
        and not image_path.exists()
      ):
        print(f"[ERROR] Image not found: {image_path}")
        return 1
      input_data, input_mode = prepare_input_data(
        image_path,
        input_scale,
        input_spec,
        force_synthetic=args.force_synthetic_input,
        model_name=model_name,
      )
      print(f"       Input mode: {input_mode}")
      print(f"       Input tensor shape: {input_data.shape}, dtype: {input_data.dtype}")

    print(f"       Value range: [{input_data.min()}, {input_data.max()}]")

    print("\n[3/6] Building TVM model...")
    t_build_start = time.time()
    if model_name.startswith("ptq4_deit_"):
        # params already loaded from flexi e2e_model.pt via get_workload()
        pass
    else:
        params = convert_model.build_param_dict(ckpt, depth=depth, model_name=model_name)

    tvm_params = {k: tvm.nd.array(v) for k, v in params.items()}

    RUNTIME = tvm.relay.backend.Runtime("crt", {"system-lib": False})
    if args.tvm_backend == "llvm-gemmini":
        RUNTIME = tvm.relay.backend.Runtime("crt", {"system-lib": True})
    EXECUTOR = tvm.relay.backend.Executor(
        "aot", options={"interface-api": "c", "unpacked-api": 1}
    )

    usmp_alg = args.usmp_alg
    if usmp_alg is None:
        usmp_alg = "greedy_by_size" if model_name.startswith("swin_") else "hill_climb"
    if usmp_alg == "none":
        usmp_alg = ""
    opt_level = args.opt_level
    if opt_level is None:
        opt_level = 2 if model_name.startswith("swin_") else 3
    canonicalize_qnn = (
        (opt_level == 0) or args.enable_qnn_canonicalize
    ) and not args.disable_qnn_canonicalize
    disabled_passes = ["AlterOpLayout"]
    print(f"       Swin qnn canonicalize: {canonicalize_qnn}")
    use_llvm_gemmini = args.tvm_backend == "llvm-gemmini"
    if use_llvm_gemmini:
        if not gemmini_byoc_enabled():
            print(
                "[ERROR] --tvm-backend llvm-gemmini requires Gemmini BYOC target in TVM. "
                "Rebuild tvm-gemmini with USE_GEMMINI=ON (see docs/tvm_path_b_plan.md)."
            )
            return 1
        enable_fast_math = bool(args.llvm_fast_math)
        print(
            f"       TVM backend: llvm-gemmini "
            f"(RVV={args.llvm_rvv}, tir_vectorize={args.llvm_tir_vectorize}, "
            f"llvm_loop_vectorize={not args.llvm_no_loop_vectorize}, "
            f"fast_math={enable_fast_math})"
        )
        # Swin e2e + full VF=64 previously hung LLVM (~128GB RSS, 5h+). Cap VF / opt.
        llvm_opt_level = args.llvm_opt_level
        tir_max_vf = args.llvm_tir_max_vf
        if args.llvm_tir_vectorize and model_name.startswith("swin_"):
            if tir_max_vf is None:
                tir_max_vf = 16
            if llvm_opt_level is None:
                llvm_opt_level = 2
        if tir_max_vf is not None:
            os.environ["TVM_TOPI_INJECTIVE_MAX_VF"] = str(int(tir_max_vf))
            print(f"       TIR injective max VF cap: {tir_max_vf}")
        if llvm_opt_level is None:
            llvm_opt_level = 3
        # Fair autovec (match IREE ``--lmul 0`` + GenericVectorization style):
        # - no pinned LLVM LMUL; TE tile default LMUL=1 (VLEN/SEW)
        # - awkward axes (Softmax 197) stay contiguous → LLVM masked vp.reduce
        # - injective TE.vectorize ON: TVM analog of IREE GenericVectorization
        #   (structured auto-vec, not hand microkernels). Softmax/LN still force
        #   te_vectorize=False in their schedules.
        # FastMath (PTQ4): poly Softmax/GELU so float epilogues vectorize; does
        # not change the TE-VF policy above.
        if args.llvm_rvv:
            if "TVM_TOPI_FAIR_AUTOVEC" not in os.environ:
                os.environ["TVM_TOPI_FAIR_AUTOVEC"] = "1"
            te_lmul = os.environ.get("TVM_TOPI_INJECTIVE_LMUL", "1").strip() or "1"
            llvm_lmul = os.environ.get("TVM_LLVM_RVV_LMUL", "").strip() or "omit"
            print(
                f"       TIR/LLVM LMUL: TE={te_lmul}, "
                f"LLVM flag={llvm_lmul} (fair default: TE=1, LLVM omit)"
            )
        if args.llvm_tir_vectorize:
            if "TVM_TOPI_INJECTIVE_TE_VECTORIZE" not in os.environ:
                # Default ON: structured injective VF ≈ IREE GenericVectorization.
                os.environ["TVM_TOPI_INJECTIVE_TE_VECTORIZE"] = "1"
            te_vf = os.environ.get("TVM_TOPI_INJECTIVE_TE_VECTORIZE", "1")
            print(
                f"       Fair autovec: TE-VF(injective)={te_vf}, "
                "awkward Softmax/LN reduces → LLVM (IREE-like)"
            )
            if enable_fast_math:
                print("       FastMath: on (poly Softmax/GELU via Relay FastMath)")
            else:
                # Without Relay FastMath: still make Softmax/GELU vectorizable via
                # TOPI fast_softmax + erf→fast_erf (IREE-like math approx at TE).
                if "TVM_TOPI_VECTORIZABLE_ELEMWISE_MATH" not in os.environ:
                    os.environ["TVM_TOPI_VECTORIZABLE_ELEMWISE_MATH"] = "1"
                print(
                    "       FastMath: off; vectorizable elemwise math "
                    "(TOPI fast_softmax + erf→fast_erf, no Relay FastMath)"
                )
        mod = preprocess_for_heterogeneous_gemmini(
            mod,
            model_name,
            canonicalize_qnn=canonicalize_qnn,
            enable_fast_math=enable_fast_math,
        )
    else:
        print("       TVM backend: c-gemmini")
        mod = preprocess_for_gemmini(mod, model_name, canonicalize_qnn=canonicalize_qnn)
        llvm_opt_level = 3
    mod = relay.transform.InferType()(mod)
    if use_llvm_gemmini:
        llvm_target = llvm_riscv_target(
            enable_rvv=args.llvm_rvv,
            disable_ipo_inline=bool(args.profile_kernels),
            opt_level=int(llvm_opt_level),
            disable_loop_vectorize=bool(args.llvm_no_loop_vectorize),
        )
        print(f"       LLVM target opt-level: {llvm_opt_level}")
        if args.llvm_no_loop_vectorize:
            print(
                "       LLVM LoopVectorize/SLP: OFF "
                "(-vectorize-loops=false -vectorize-slp=false); "
                "TIR VectorizeLoop RVV lowering kept if --llvm-rvv"
            )
        if args.profile_kernels:
            print(
                "       llvm-gemmini profile: disable IPO inline "
                "(-inline-threshold=0) so host kernels stay callable"
            )
        gemmini_target = tvm.target.Target("gemmini", host=llvm_target)
        mod["main"] = bind_params_by_name(mod["main"], tvm_params)
        mod = relay.transform.InferType()(mod)
    print("       Preprocess pass done")
    stack_limit_change = maybe_raise_stack_limit_for_build(model_name, opt_level)
    if stack_limit_change is not None:
        old_soft, new_soft = stack_limit_change
        old_soft_str = "unlimited" if old_soft == resource.RLIM_INFINITY else str(old_soft)
        new_soft_str = "unlimited" if new_soft == resource.RLIM_INFINITY else str(new_soft)
        print(f"       Raised RLIMIT_STACK: {old_soft_str} -> {new_soft_str}")
    print(
        f"       relay.build 시작 (usmp_alg={usmp_alg}, opt_level={opt_level}) "
        f"- Swin은 수십 분 걸릴 수 있음"
    )
    if disabled_passes:
        print(f"       disabled_pass={disabled_passes}")

    dump_tir_extra_config = {}
    tir_dump_staging = None
    tir_dump_final = None
    tir_dump_counter = {"i": 0}
    if args.dump_tir:
        import tempfile

        tir_dump_final = pathlib.Path(args.dump_tir).resolve()
        # Dump to a staging dir first: export step does rmtree(output_dir), which
        # would delete dumps if --dump-tir points inside --output-dir.
        tir_dump_staging = pathlib.Path(
            tempfile.mkdtemp(prefix="tvm_tir_dump_")
        )
        tir_dump_dir = tir_dump_staging

        @tvm.tir.transform.prim_func_pass(opt_level=0)
        def _dump_tir_primfunc(f, mod, ctx):  # noqa: ARG001
            idx = tir_dump_counter["i"]
            tir_dump_counter["i"] = idx + 1
            sym = "func"
            if f.attrs is not None and "global_symbol" in f.attrs:
                sym = str(f.attrs["global_symbol"])
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", sym)[:120]
            path = tir_dump_dir / f"{idx:04d}_{safe}.tir"
            path.write_text(str(f))
            return f

        # phase 3 = after VectorizeLoop / Unroll / Simplify, immediately before codegen
        dump_tir_extra_config = {"tir.add_lower_pass": [[3, _dump_tir_primfunc]]}
        print(f"       TIR dump (staging): {tir_dump_staging}")
        print(f"       TIR dump (final):   {tir_dump_final}")

    if use_llvm_gemmini:
        build_ctx_kwargs = dict(
            usmp_alg=usmp_alg,
            opt_level=opt_level,
            disabled_pass=disabled_passes,
            enable_tir_vectorize=bool(args.llvm_tir_vectorize),
        )
        if dump_tir_extra_config:
            build_ctx_kwargs["config"] = dump_tir_extra_config
        build_ctx = gemmini.heterogeneous_build_config(**build_ctx_kwargs)
        build_kwargs = dict(
            executor=EXECUTOR,
            runtime=RUNTIME,
            target=[llvm_target, gemmini_target],
            params=tvm_params,
        )
    else:
        TARGET = tvm.target.target.Target({"kind": "c", "device": "gemmini"})
        build_ctx_kwargs = dict(
            usmp_alg=usmp_alg,
            opt_level=opt_level,
            disabled_pass=disabled_passes,
        )
        if dump_tir_extra_config:
            build_ctx_kwargs["config"] = dump_tir_extra_config
        build_ctx = gemmini.build_config(**build_ctx_kwargs)
        build_kwargs = dict(
            executor=EXECUTOR, runtime=RUNTIME, target=TARGET, params=tvm_params
        )

    with build_ctx:
        module = relay.build(mod, **build_kwargs)
    t_build_end = time.time()
    print(f"       relay.build 완료 ({t_build_end - t_build_start:.1f}s)")

    print("\n[4/6] Exporting C code...")
    output_dir = pathlib.Path(args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if tir_dump_staging is not None and tir_dump_final is not None:
        if tir_dump_final.exists():
            shutil.rmtree(tir_dump_final)
        tir_dump_final.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tir_dump_staging), str(tir_dump_final))
        print(
            f"       TIR dump complete: {tir_dump_counter['i']} PrimFuncs -> {tir_dump_final}"
        )

    mlf_path = output_dir / "model.tar"
    tvm.micro.export_model_library_format(module, mlf_path)
    with tarfile.open(mlf_path, "r:") as tar:
        tar.extractall(output_dir)

    fix_generated_code(
        output_dir,
        model_name,
        instrument_kernel_cycles=args.profile_kernels and args.simulator == "spike",
        segment_profile_style=args.profile_semantic_style if args.profile_semantic else None,
        segment_profile_layer_filters=args.profile_layer_filter,
        segment_profile_component_filters=args.profile_component_filter,
        instrument_requant_intrakernel=args.profile_intrakernel_requant,
        debug_unit=args.debug_unit,
        skip_if_no_scalar_c=use_llvm_gemmini,
        llvm_gemmini=use_llvm_gemmini,
    )
    if use_llvm_gemmini:
        generate_llvm_aot_shim(output_dir, module)
        instrument_llvm_aot_main_total_cycles(output_dir)
        llvm_profile_link_flags = []
        if args.profile_kernels and args.simulator == "spike":
            _, llvm_profile_link_flags, _ = generate_llvm_kernel_profile_wraps(
                output_dir,
                model_name,
                debug_unit=args.debug_unit,
            )
            instrument_llvm_aot_kernel_cycle_dump(output_dir)
        count_rvv_instructions(output_dir, object_name="default_lib1.o")
        # Continue to harness + link so --build-only still produces a FireSim ELF.

    if not use_llvm_gemmini:
        instrument_tvm_main_total_cycles(output_dir)
    classification_output = args.debug_unit is None
    create_real_image_harness(
        output_dir,
        model_name,
        MODEL_SPECS[model_name]["embed_dim"],
        input_data,
        classification_output=classification_output,
        debug_unit=args.debug_unit,
        uart_mode=args.uart_mode,
        enable_spike_rvv=(
            (use_llvm_gemmini and args.llvm_rvv)
            or bool(args.gcc_autovec)
            or riscv_march_enables_v(args.riscv_march or "")
        ),
    )

    print("\n[5/6] Compiling for Spike...")
    if use_llvm_gemmini:
        link_march = SATURN_LINK_MARCH if args.llvm_rvv else "rv64gc"
        compile_label = f"llvm-gemmini march={link_march} O{args.gcc_opt_level}"
        print(f"[Info] link flags: {compile_label}")
        binary = compile_for_spike_llvm_gemmini(
            output_dir,
            "ivit_real",
            riscv_march=link_march,
            gcc_opt_level=args.gcc_opt_level,
            extra_link_flags=llvm_profile_link_flags,
        )
    else:
        compile_label = (
            f"march={args.riscv_march} O{args.gcc_opt_level}"
            + (" autovec" if args.gcc_autovec else "")
        )
        print(f"[Info] gcc flags: {compile_label}")
        binary = compile_for_spike(
            output_dir,
            "ivit_real",
            riscv_march=args.riscv_march,
            gcc_opt_level=args.gcc_opt_level,
            gcc_autovec=args.gcc_autovec,
        )
    if binary is None:
        return 1
    if args.build_only:
        print(f"[Info] Build-only mode. Baremetal ELF: {binary}")
        return 0

    if args.simulator == "spike":
        spike_isa = args.spike_isa
        if spike_isa is None and use_llvm_gemmini and args.llvm_rvv:
            spike_isa = SATURN_SPIKE_ISA
        elif spike_isa is None and (
            args.gcc_autovec or riscv_march_enables_v(args.riscv_march or "")
        ):
            # Path A: prefer Saturn-complete ISA for local smoke; FireSim uses HW RVV.
            spike_isa = SATURN_SPIKE_ISA
        if spike_isa:
            print(f"[Info] Spike ISA: {spike_isa}")
        if use_llvm_gemmini and args.llvm_rvv:
            print(
                "[Info] llvm-gemmini RVV: fixed VLEN=512 LLVM cl-opt + mstatus.VS for Spike."
            )
        elif args.gcc_autovec or riscv_march_enables_v(args.riscv_march or ""):
            print(
                "[Info] Path A gcc-autovec: march has V; harness enables mstatus.VS + vill clear."
            )
        print("\n[6/6] Running on Spike...")
        stdout, stderr = run_spike(binary, timeout=args.timeout, spike_isa=spike_isa)
        ver_stdout_path = None
        ver_stderr_path = None
    else:
        verilator_verbose = args.verilator_verbose or args.profile_kernels
        save_verilator_logs = args.verilator_save_logs or verilator_verbose
        if args.profile_kernels and not args.verilator_verbose:
            print("[Info] --profile-kernels requested; enabling Verilator +verbose.")
        log_tail_lines = args.verilator_log_tail_lines
        if save_verilator_logs and log_tail_lines > 0:
            print(
                "[Info] --verilator-save-logs requested; "
                "saving full live Verilator stdout/stderr logs."
            )
            log_tail_lines = 0
        if (args.decode_dasm or args.profile_kernels) and log_tail_lines > 0:
            print(
                "[Info] --decode-dasm/--profile-kernels needs full trace; "
                "disabling log tail truncation for this run."
            )
            log_tail_lines = 0
        print("\n[6/6] Running on Verilator...")
        stdout, stderr, ver_stdout_path, ver_stderr_path = run_verilator(
            binary,
            timeout=args.timeout,
            chipyard_dir=args.chipyard_dir,
            verilator_config=args.verilator_config,
            max_cycles=args.max_cycles,
            dramsim=not args.no_dramsim,
            verbose=verilator_verbose,
            log_dir=output_dir if save_verilator_logs else None,
            log_tail_lines=log_tail_lines,
        )

    if stdout is None:
        return 1

    print("\n" + "=" * 60)
    print(f"Simulation Output ({args.simulator}):")
    print("=" * 60)
    if stdout:
        print(stdout)
    elif args.simulator == "verilator" and ver_stdout_path is not None:
        print(f"[Info] Verilator stdout saved to: {ver_stdout_path}")
        with open(ver_stdout_path, "r") as f:
            snippet = f.read(4000)
        if snippet:
            print(snippet)

    intrakernel_rows = (
        extract_spike_intrakernel_cycle_rows(stdout)
        if args.profile_kernels or args.profile_intrakernel_requant
        else []
    )

    if args.profile_kernels and args.simulator == "spike":
        kernel_rows = extract_spike_kernel_cycle_rows(stdout)
        semantic_rows = extract_spike_semantic_cycle_rows(stdout)
        main_total_cycles = extract_spike_main_total_cycles(stdout)
        if kernel_rows:
            csv_path, txt_path = write_spike_kernel_cycle_reports(
                output_dir, kernel_rows, topk=args.profile_topk
            )
            print(f"[Info] Spike kernel cycle CSV: {csv_path}")
            print(f"[Info] Spike kernel cycle report: {txt_path}")
            aligned_rows = build_spike_segment_cycle_rows(
                output_dir,
                model_name,
                kernel_rows,
                style="aligned",
                debug_unit=args.debug_unit,
            )
            (
                aligned_segment_csv_path,
                aligned_segment_txt_path,
                aligned_layer_csv_path,
                aligned_layer_txt_path,
                aligned_component_csv_path,
                aligned_component_txt_path,
                aligned_layer_component_csv_path,
                aligned_layer_component_txt_path,
            ) = write_spike_aligned_cycle_reports(
                output_dir, aligned_rows, main_total_cycles=main_total_cycles
            )
            print(f"[Info] Spike aligned segment CSV: {aligned_segment_csv_path}")
            print(f"[Info] Spike aligned segment report: {aligned_segment_txt_path}")
            print(f"[Info] Spike aligned layer cycle CSV: {aligned_layer_csv_path}")
            print(f"[Info] Spike aligned layer cycle report: {aligned_layer_txt_path}")
            print(f"[Info] Spike aligned component cycle CSV: {aligned_component_csv_path}")
            print(f"[Info] Spike aligned component cycle report: {aligned_component_txt_path}")
            print(
                "[Info] Spike aligned layer-component cycle CSV: "
                f"{aligned_layer_component_csv_path}"
            )
            print(
                "[Info] Spike aligned layer-component cycle report: "
                f"{aligned_layer_component_txt_path}"
            )
        else:
            print("[WARN] No [TVM_KERNEL_CYCLES] rows found in Spike output")
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
            print(f"[Info] Spike semantic segment CSV: {segment_csv_path}")
            print(f"[Info] Spike semantic segment report: {segment_txt_path}")
            print(f"[Info] Spike layer cycle CSV: {layer_csv_path}")
            print(f"[Info] Spike layer cycle report: {layer_txt_path}")
            print(f"[Info] Spike component cycle CSV: {component_csv_path}")
            print(f"[Info] Spike component cycle report: {component_txt_path}")
            print(f"[Info] Spike layer-component cycle CSV: {layer_component_csv_path}")
            print(f"[Info] Spike layer-component cycle report: {layer_component_txt_path}")
            if main_total_cycles is not None:
                print(f"[Info] TVM main total cycles: {main_total_cycles}")
        else:
            print("[WARN] No [TVM_SEMANTIC_CYCLES] rows found in Spike output")

    if args.profile_kernels or args.profile_intrakernel_requant:
        if intrakernel_rows:
            intrakernel_csv_path, intrakernel_txt_path = write_spike_intrakernel_cycle_reports(
                output_dir, intrakernel_rows
            )
            print(f"[Info] Spike intrakernel cycle CSV: {intrakernel_csv_path}")
            print(f"[Info] Spike intrakernel cycle report: {intrakernel_txt_path}")
        else:
            print("[WARN] No [TVM_INTRAKERNEL_CYCLES] rows found in simulation output")

    if args.profile_semantic:
        semantic_output_text = stdout
        if (not semantic_output_text) and args.simulator == "verilator" and ver_stdout_path is not None:
            with open(ver_stdout_path, "r") as f:
                semantic_output_text = f.read()
        semantic_rows = extract_spike_semantic_cycle_rows(semantic_output_text)
        main_total_cycles = extract_spike_main_total_cycles(semantic_output_text)
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
            if main_total_cycles is not None:
                print(f"[Info] TVM main total cycles: {main_total_cycles}")
        else:
            location = ver_stdout_path if args.simulator == "verilator" else "stdout"
            print(f"[WARN] No [TVM_SEMANTIC_CYCLES] rows found in {location}")

    if args.simulator == "verilator" and ver_stderr_path is not None:
        print(f"[Info] Verilator stderr trace saved to: {ver_stderr_path}")
        if args.decode_dasm:
            decoded_path = pathlib.Path(output_dir) / "verilator_stderr.dasm"
            decode_trace_with_spike_dasm(ver_stderr_path, decoded_path)
        if args.profile_kernels:
            report_txt = pathlib.Path(output_dir) / "kernel_profile.txt"
            report_csv = pathlib.Path(output_dir) / "kernel_profile.csv"
            profile = profile_kernels_from_trace(
                ver_stderr_path,
                binary,
                report_txt,
                report_csv,
                topk=args.profile_topk,
            )
            print("\nKernel Profile Summary (top kernels):")
            for idx, (name, cycles, samples) in enumerate(profile["top"][:10], start=1):
                pct = (
                    100.0 * cycles / profile["total_kernel_cycles"]
                    if profile["total_kernel_cycles"]
                    else 0.0
                )
                print(f"  {idx:2d}. {name}: {cycles} cycles ({pct:.2f}%), samples={samples}")
            print(f"[Info] Kernel profile report: {report_txt}")
            print(f"[Info] Kernel profile CSV: {report_csv}")

    if classification_output:
        classes = load_imagenet_classes()

        print("\n" + "=" * 60)
        print("Class Labels:")
        print("=" * 60)

        output_text = stdout
        if (not output_text) and args.simulator == "verilator" and ver_stdout_path is not None:
            with open(ver_stdout_path, "r") as f:
                output_text = f.read()

        for match in re.finditer(r"Class (\d+)", output_text):
            class_id = int(match.group(1))
            if class_id < len(classes):
                print(f"  Class {class_id}: {classes[class_id]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
