# ORT Repro

This is the minimal reproducible path for the ONNX Runtime flow. It assumes the
external Chipyard toolchain is installed and an I-ViT checkpoint is available.
If you are preparing RTL baselines, use this file first and treat the other
validation helpers as optional support tools.

## Primary Entry Points

- Build: `ONNXRuntime/build/build_ort_riscv.sh`
- Export: `ONNXRuntime/export/ivit/export_onnx.sh`
- Export: `ONNXRuntime/export/repq/export_repq_onnx.sh`
- Spike: `ONNXRuntime/run/run_ort_spike.sh`
- Verilator: `ONNXRuntime/run/run_ort_verilator.sh`

## Verified Baseline Outputs

The following fresh exports were generated under `build/ort/repro/`:

- `build/ort/repro/ivit_deit_tiny_int8.onnx`
- `build/ort/repro/ivit_swin_tiny_int8.onnx`
- `build/ort/repro/repq_deit_tiny_w8a8_lowered.onnx`
- `build/ort/repro/repq_swin_tiny_w8a8_lowered.onnx`

Fresh Spike logs were generated under `build/ort/repro/`:

- `build/ort/repro/ivit_deit_spike_x1.log`
- `build/ort/repro/ivit_swin_spike_x1.log`
- `build/ort/repro/repq_deit_spike_x1.log`
- `build/ort/repro/repq_swin_spike_x1.log`

These filenames record the validation run; generated models and logs are not
checked into Git. Recreate them with the steps below and supply the matching
external checkpoints.

## 1) Build

```bash
cd /path/to/flexi_baseline/ViT-Gemmini
export CHIPYARD_DIR=/path/to/chipyard
export RISCV="${CHIPYARD_DIR}/.conda-env/riscv-tools"
export IVIT_CHECKPOINT=/path/to/ivit-checkpoint.pth.tar
export SWIN_CHECKPOINT=/path/to/swin-checkpoint.pth.tar
source ./env.sh
ONNXRuntime/build/build_ort_riscv.sh
```

This builds:

- `onnxruntime-riscv`
- `ort_test`
- `build/ort/ort_ivit_ops/libivit_ops.a`
- `build/ort/ort_ivit_ops/libivit_ops_host.so`

## 2) Export

### I-ViT DeiT

```bash
ONNXRuntime/export/ivit/export_onnx.sh \
  --model-name deit_tiny_patch16_224 \
  --checkpoint "$IVIT_CHECKPOINT" \
  --output build/ort/repro/ivit_deit_tiny_int8.onnx
```

### I-ViT Swin

```bash
ONNXRuntime/export/ivit/export_onnx.sh \
  --model-name swin_tiny_patch4_window7_224 \
  --checkpoint "$SWIN_CHECKPOINT" \
  --output build/ort/repro/ivit_swin_tiny_int8.onnx
```

### RepQ DeiT

```bash
ONNXRuntime/export/repq/export_repq_onnx.sh \
  --model deit_tiny \
  --dataset /data/imagenet_val \
  --w-bits 8 \
  --a-bits 8 \
  --device cpu \
  --lower-qlinear-matmul \
  --verify-ort \
  --output build/ort/repro/repq_deit_tiny_w8a8_lowered.onnx \
  --save-reparam-state build/ort/repro/repq_deit_tiny_w8a8_reparam.pt
```

### RepQ Swin

```bash
ONNXRuntime/export/repq/export_repq_onnx.sh \
  --model swin_tiny \
  --dataset /data/imagenet_val \
  --w-bits 8 \
  --a-bits 8 \
  --device cpu \
  --lower-qlinear-matmul \
  --verify-ort \
  --output build/ort/repro/repq_swin_tiny_w8a8_lowered.onnx \
  --save-reparam-state build/ort/repro/repq_swin_tiny_w8a8_reparam.pt
```

Important:

- For RepQ, use lowered ONNX only.
- Do not use `semantic.onnx` as an RTL baseline candidate.

## 3) Structural Gate Before Simulation

Use these checks before Spike or Verilator:

- I-ViT ONNX should have `GemminiMatMulInteger` and no plain `MatMul/Gemm`.
- RepQ ONNX should have `RepQUniformMatMul` and `RepQLogMatMul` and no plain `MatMul/Gemm`.

Observed fresh export counts:

- `ivit_deit_tiny_int8.onnx`: `GemminiMatMulInteger=74`
- `ivit_swin_tiny_int8.onnx`: `GemminiMatMulInteger=77`
- `repq_deit_tiny_w8a8_lowered.onnx`: `RepQUniformMatMul=61`, `RepQLogMatMul=12`, `QLinearConv=1`
- `repq_swin_tiny_w8a8_lowered.onnx`: `RepQUniformMatMul=64`, `RepQLogMatMul=12`, `QLinearConv=1`

## 4) Spike

### I-ViT DeiT

```bash
ONNXRuntime/run/run_ort_spike.sh \
  scripts/gemmini/test_cat.jpg \
  1 \
  build/ort/repro/ivit_deit_tiny_int8.onnx \
  1 \
  --model-name deit_tiny_patch16_224 \
  --log-file build/ort/repro/ivit_deit_spike_x1.log
```

### I-ViT Swin

```bash
ONNXRuntime/run/run_ort_spike.sh \
  scripts/gemmini/test_cat.jpg \
  1 \
  build/ort/repro/ivit_swin_tiny_int8.onnx \
  1 \
  --model-name swin_tiny_patch4_window7_224 \
  --log-file build/ort/repro/ivit_swin_spike_x1.log
```

### RepQ DeiT

```bash
ONNXRuntime/run/run_ort_spike.sh \
  scripts/gemmini/test_cat.jpg \
  1 \
  build/ort/repro/repq_deit_tiny_w8a8_lowered.onnx \
  1 \
  --model-name deit_tiny_patch16_224 \
  --log-file build/ort/repro/repq_deit_spike_x1.log
```

### RepQ Swin

```bash
ONNXRuntime/run/run_ort_spike.sh \
  scripts/gemmini/test_cat.jpg \
  1 \
  build/ort/repro/repq_swin_tiny_w8a8_lowered.onnx \
  1 \
  --model-name swin_tiny_patch4_window7_224 \
  --log-file build/ort/repro/repq_swin_spike_x1.log
```

## 5) Assembly-Level Gemmini Gate

Do not rely only on log strings.

Check the linked runner:

```bash
nm -C tvm-gemmini/3rdparty/gemmini/software/onnxruntime-riscv/systolic_runner/imagenet_runner/ort_test | \
  rg 'GemminiMatMulInteger_Compute|RepQUniformMatMul_Compute|RepQLogMatMul_Compute|gemmini_matmul_int32'
```

Check that the custom-op compute paths call `gemmini_matmul_int32`:

```bash
"${RISCV}/bin/riscv64-unknown-linux-gnu-objdump" \
  -d --demangle \
  --disassemble='(anonymous namespace)::GemminiMatMulInteger_Compute(void*, OrtKernelContext*)' \
  tvm-gemmini/3rdparty/gemmini/software/onnxruntime-riscv/systolic_runner/imagenet_runner/ort_test
```

```bash
"${RISCV}/bin/riscv64-unknown-linux-gnu-objdump" \
  -d --demangle \
  --disassemble='(anonymous namespace)::gemmini_matmul_int32(signed char const*, signed char const*, int*, long, long, long, int) [clone .part.0]' \
  tvm-gemmini/3rdparty/gemmini/software/onnxruntime-riscv/systolic_runner/imagenet_runner/ort_test
```

The second disassembly should include Gemmini custom instructions emitted as
`.insn`.

## 6) Verilator

Dry-run first:

```bash
ONNXRuntime/run/run_ort_verilator.sh \
  scripts/gemmini/test_cat.jpg \
  1 \
  build/ort/repro/ivit_deit_tiny_int8.onnx \
  1 \
  --dry-run
```

Then run the real simulation:

```bash
ONNXRuntime/run/run_ort_verilator.sh \
  scripts/gemmini/test_cat.jpg \
  1 \
  build/ort/repro/ivit_deit_tiny_int8.onnx \
  1
```

RepQ uses the same runner:

```bash
ONNXRuntime/run/run_ort_verilator.sh \
  scripts/gemmini/test_cat.jpg \
  1 \
  build/ort/repro/repq_deit_tiny_w8a8_lowered.onnx \
  1
```

Optional caps:

```bash
TIMEOUT_SECS=1800 MAX_CYCLES=25000000000 \
ONNXRuntime/run/run_ort_verilator.sh \
  scripts/gemmini/test_cat.jpg \
  1 \
  build/ort/repro/ivit_deit_tiny_int8.onnx \
  1
```

## 7) Which Files Matter

Use these first:

- `ONNXRuntime/REPRO.md`
- `ONNXRuntime/build/build_ort_riscv.sh`
- `ONNXRuntime/export/ivit/export_onnx.sh`
- `ONNXRuntime/export/repq/export_repq_onnx.sh`
- `ONNXRuntime/run/run_ort_spike.sh`
- `ONNXRuntime/run/run_ort_verilator.sh`

Treat these as support tools, not the main baseline gate:

- `ONNXRuntime/tools/run_reference_inference.py`
- `ONNXRuntime/tools/run_repq_onnx_inference.py`
- `ONNXRuntime/tools/validate_custom_ops.py`
- `ONNXRuntime/tools/verify_gemmini_usage.py`
