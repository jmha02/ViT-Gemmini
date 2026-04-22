# ONNX Runtime

This is the canonical ONNX Runtime workspace for this repository.
It is organized by workflow so the top level stays small:

- `build/`: build entrypoints for `onnxruntime-riscv`, `ort_test`, and custom ops
- `export/`: ONNX export flows
- `run/`: Spike and Verilator runners
- `tools/`: host-side preprocessing and validation helpers
- `profiling/`: Verilator trace and cycle analysis helpers
- `ort_ivit_ops/`: custom op sources for I-ViT and RepQ lowering

If you want the shortest reproducible path for export -> Spike -> Verilator,
start with [REPRO.md](/root/flexi/third-party/I-ViT-Gemmini/ONNXRuntime/REPRO.md).

Compatibility note:

- `scripts/onnxrt` is a symlink to this directory for older paths.

## Entry Points

### Build

```bash
cd /root/flexi/third-party/I-ViT-Gemmini
ONNXRuntime/build/build_ort_riscv.sh
```

### Export

I-ViT exports:

```bash
ONNXRuntime/export/ivit/export_onnx.sh \
  --model-name deit_tiny_patch16_224 \
  --checkpoint /root/checkpoint_last.pth.tar \
  --output build/ort/ivit_tiny_int8.onnx
```

```bash
ONNXRuntime/export/ivit/export_onnx.sh \
  --model-name swin_tiny_patch4_window7_224 \
  --checkpoint /data/checkpoint.pth.tar \
  --output build/ort/swin_tiny_int8.onnx
```

Wrappers:

- `ONNXRuntime/export/ivit/export_ivit_onnx.sh`
- `ONNXRuntime/export/ivit/export_swin_ivit_onnx.sh`

RepQ export:

```bash
ONNXRuntime/export/repq/export_repq_onnx.sh \
  --model deit_tiny \
  --dataset /data/imagenet \
  --w-bits 8 \
  --a-bits 8 \
  --output build/ort/repq_deit_tiny_w8a8_lowered.onnx \
  --verify-ort
```

More export notes: [export/README.md](/root/flexi/third-party/I-ViT-Gemmini/ONNXRuntime/export/README.md)
Minimal baseline flow: [REPRO.md](/root/flexi/third-party/I-ViT-Gemmini/ONNXRuntime/REPRO.md)

### Run On Spike

```bash
ONNXRuntime/run/run_ort_spike.sh \
  scripts/gemmini/test_cat.jpg \
  1 \
  build/ort/ivit_tiny_int8.onnx \
  1 \
  --model-name deit_tiny_patch16_224
```

```bash
ONNXRuntime/run/run_ort_spike.sh \
  scripts/gemmini/test_cat.jpg \
  1 \
  build/ort/swin_tiny_int8.onnx \
  1 \
  --model-name swin_tiny_patch4_window7_224
```

`mode`:

- `0`: CPU fallback
- `1`: Gemmini output-stationary
- `2`: Gemmini weight-stationary

More run notes: [run/README.md](/root/flexi/third-party/I-ViT-Gemmini/ONNXRuntime/run/README.md)

### Run On Verilator

```bash
ONNXRuntime/run/run_ort_verilator.sh \
  scripts/gemmini/test_cat.jpg \
  1 \
  build/ort/ivit_tiny_int8.onnx \
  1
```

Block sweep:

```bash
ONNXRuntime/run/run_ort_verilator_blocks.sh 0 11 1
```

By default, the Verilator runner does not apply a host timeout or a simulator
cycle cap. Set `TIMEOUT_SECS` and `MAX_CYCLES` only when you want limits.

Detailed notes: [ORT_VERILATOR_PROFILING.md](/root/flexi/third-party/I-ViT-Gemmini/ONNXRuntime/profiling/ORT_VERILATOR_PROFILING.md)

## Validation Helpers

Gemmini usage:

```bash
python3 ONNXRuntime/tools/verify_gemmini_usage.py \
  --x1-log /path/to/ort_spike_x1.log \
  --x0-log /path/to/ort_spike_x0.log
```

Custom op correctness:

```bash
make -C ONNXRuntime/ort_ivit_ops host
python3 ONNXRuntime/tools/validate_custom_ops.py
```

PyTorch vs ORT:

```bash
python3 ONNXRuntime/tools/run_reference_inference.py --help
python3 ONNXRuntime/tools/run_repq_onnx_inference.py --help
```

Tool index: [tools/README.md](/root/flexi/third-party/I-ViT-Gemmini/ONNXRuntime/tools/README.md)

## Custom Op Sources

- `ONNXRuntime/ort_ivit_ops/ivit_ops.c`
- `ONNXRuntime/ort_ivit_ops/ivit_gemmini_ops.cc`

`GemminiMatMulInteger`, `RepQUniformMatMul`, and `RepQLogMatMul` are implemented
there and are the operators that dispatch matmul paths to Gemmini when Spike is
run with `mode=1` or `mode=2`.
