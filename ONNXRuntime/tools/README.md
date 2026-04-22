# Tools

This directory contains support helpers.
For baseline reproduction, start with `../REPRO.md` instead of browsing these
scripts one by one.

## Primary Support Tools

- `run_reference_inference.py`: host PyTorch vs host ORT for I-ViT
- `run_repq_onnx_inference.py`: host ORT top-k for RepQ
- `validate_custom_ops.py`: unit checks for custom op kernels

## Auxiliary Tools

- `preprocess_image_to_tensor.py`: image -> raw tensor for `ort_test`
- `ivit_input.py`: shared preprocessing helpers
- `ivit_model_io.py`: I-ViT checkpoint/model loading helpers
- `verify_gemmini_usage.py`: lightweight log parser only; not the primary RTL gate

## Recommended Order

1. Build with `ONNXRuntime/build/build_ort_riscv.sh`
2. Export with `ONNXRuntime/export/...`
3. Check the ONNX op mix
4. Run Spike with `ONNXRuntime/run/run_ort_spike.sh`
5. Inspect `ort_test` symbols/disassembly if Gemmini confirmation is needed
6. Run Verilator with `ONNXRuntime/run/run_ort_verilator.sh`
