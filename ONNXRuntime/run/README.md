# Run Flows

For the minimal baseline path, see
`ONNXRuntime/REPRO.md`.

This directory contains runtime entrypoints after ONNX export.

- `run_ort_spike.sh`: run `ort_test` on Spike
- `run_ort_verilator.sh`: run `ort_test` on Verilator
- `run_ort_verilator_blocks.sh`: block-by-block Verilator sweep

## Spike

```bash
ONNXRuntime/run/run_ort_spike.sh \
  scripts/gemmini/test_cat.jpg \
  1 \
  build/ort/ivit_tiny_int8.onnx \
  1 \
  --model-name deit_tiny_patch16_224
```

`mode`:

- `0`: CPU fallback
- `1`: Gemmini output-stationary
- `2`: Gemmini weight-stationary

## Verilator

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

By default, `run_ort_verilator.sh` and `run_ort_verilator_blocks.sh` do not set
host timeout or simulator cycle caps. Set `TIMEOUT_SECS` and `MAX_CYCLES`
explicitly only when you want limits.

The detailed profiling workflow lives in
`ONNXRuntime/profiling/ORT_VERILATOR_PROFILING.md`.
