# ViT-Gemmini

Minimal workspace for three Gemmini-targeted flows:

- `I-ViT DeiT-Tiny`
- `I-ViT Swin-Tiny`
- `RepQ-ViT DeiT-Tiny`

Supported export/runtime paths:

- `TVM -> Gemmini Spike`
- `ONNX Runtime -> Gemmini Spike`

## Layout

- `models/`: curated Relay model families for `ivit` and `repq`
- `I-ViT/`: vendored upstream I-ViT model code needed for checkpoint loading/export
- `RepQ-ViT/classification/`: vendored RepQ classification quantization code
- `ONNXRuntime/`: curated ORT export/build/run workspace
- `scripts/gemmini/`: TVM AOT + Spike runners
- `scripts/tvm/`: checkpoint/quantization parameter extraction helpers
- `tools/generate_random_ivit_checkpoint.py`: random smoke checkpoint generator
- `scripts/smoke_all.sh`: end-to-end smoke run for all three target models

## Prerequisites

Heavy infrastructure is kept external on purpose.

- `TVM_HOME`: Gemmini-enabled TVM checkout
- `CHIPYARD_DIR`: Chipyard checkout with Spike/pk/toolchain
- `RISCV`: optional; defaults to `${CHIPYARD_DIR}/.conda-env/riscv-tools`
- `ORT_RISCV_DIR`: optional; defaults to `${TVM_HOME}/3rdparty/gemmini/software/onnxruntime-riscv`

In this environment, the defaults in `env.sh` resolve to the existing `flexi` checkouts.
`env.sh` also pins the Chipyard `libstdc++` runtime so TVM Python bindings load
cleanly from the base conda environment.

## Setup

```bash
cd /root/ViT-Gemmini
source env.sh
```

## End-To-End Smoke

This generates random I-ViT checkpoints for smoke testing, exports:

- I-ViT DeiT -> ORT
- I-ViT Swin -> ORT
- RepQ DeiT -> ORT
- I-ViT DeiT -> TVM AOT Spike
- I-ViT Swin -> TVM AOT Spike
- RepQ DeiT -> TVM AOT Spike

and then runs Spike for all of them.

```bash
cd /root/ViT-Gemmini
source env.sh
bash scripts/smoke_all.sh
```

Validated in this environment on `2026-04-22`.

- `build/ort/ivit_deit_tiny_int8.onnx`
- `build/ort/ivit_swin_tiny_int8.onnx`
- `build/ort/repq_deit_tiny_w8a8_lowered.onnx`
- `build/tvm/deit/`
- `build/tvm/swin/`
- `build/tvm/repq_deit/`

Notes:

- `I-ViT Swin` TVM Relay build is slow in this environment and took about `963s`
  during validation.
- `RepQ DeiT` ORT export currently produces a semantic ONNX graph that still runs
  on Spike with the Gemmini-enabled ORT runner, but it is not lowered to the same
  custom-op style as the I-ViT ORT path unless `--lower-qlinear-matmul` is enabled.
- For fair TVM vs ORT performance comparison on RepQ, use
  `--repq-gemmini-kernel-mode approx`. That matches the TVM Gemmini path more
  closely. `exact` remains available for fidelity/debug comparisons.

## Manual Commands

Generate random I-ViT smoke checkpoints:

```bash
python3 tools/generate_random_ivit_checkpoint.py \
  --model-name deit_tiny_patch16_224 \
  --output build/checkpoints/ivit_deit_tiny_random.pth.tar

python3 tools/generate_random_ivit_checkpoint.py \
  --model-name swin_tiny_patch4_window7_224 \
  --output build/checkpoints/ivit_swin_tiny_random.pth.tar
```

Export ORT models:

```bash
python3 ONNXRuntime/export/ivit/export_onnx.py \
  --model-name deit_tiny_patch16_224 \
  --checkpoint build/checkpoints/ivit_deit_tiny_random.pth.tar \
  --output build/ort/ivit_deit_tiny_int8.onnx

python3 ONNXRuntime/export/ivit/export_onnx.py \
  --model-name swin_tiny_patch4_window7_224 \
  --checkpoint build/checkpoints/ivit_swin_tiny_random.pth.tar \
  --output build/ort/ivit_swin_tiny_int8.onnx

python3 ONNXRuntime/export/repq/export_repq_onnx.py \
  --model deit_tiny \
  --allow-random-init \
  --allow-random-calibration \
  --device cpu \
  --lower-qlinear-matmul \
  --repq-gemmini-kernel-mode approx \
  --output build/ort/repq_deit_tiny_w8a8_lowered.onnx
```

Run ORT on Spike:

```bash
ONNXRuntime/run/run_ort_spike.sh \
  scripts/gemmini/test_cat.jpg \
  1 \
  build/ort/ivit_deit_tiny_int8.onnx \
  1 \
  --model-name deit_tiny_patch16_224
```

Run TVM on Spike:

```bash
python3 scripts/gemmini/run_inference_spike.py \
  --image scripts/gemmini/test_cat.jpg \
  --checkpoint build/checkpoints/ivit_deit_tiny_random.pth.tar \
  --model-name deit_tiny_patch16_224 \
  --output-dir build/tvm/deit

python3 scripts/gemmini/run_inference_spike.py \
  --image scripts/gemmini/test_cat.jpg \
  --checkpoint build/checkpoints/ivit_swin_tiny_random.pth.tar \
  --model-name swin_tiny_patch4_window7_224 \
  --output-dir build/tvm/swin

python3 scripts/gemmini/run_repq_inference_spike.py \
  --model deit_tiny \
  --image scripts/gemmini/test_cat.jpg \
  --allow-random-init \
  --allow-random-calibration \
  --device cpu \
  --calib-batchsize 4 \
  --calib-num-samples 8 \
  --w-bits 8 \
  --a-bits 8 \
  --output-dir build/tvm/repq_deit
```
