# I-ViT ORT Custom Ops

This directory contains ONNX Runtime custom ops used by the integer-only I-ViT
graphs:

- `ivit.QLayernorm`
- `ivit.Shiftmax`
- `ivit.ShiftGELU`
- `ivit.GemminiMatMulInteger`
- `ivit.RequantizeInt32`

Build:

```bash
cd ONNXRuntime/ort_ivit_ops
make ORT_RISCV_DIR=/path/to/onnxruntime-riscv
make ORT_RISCV_DIR=/path/to/onnxruntime-riscv host
```

Outputs:

- `build/ort/ort_ivit_ops/libivit_ops.a`
- `build/ort/ort_ivit_ops/libivit_ops_host.so`

`ONNXRuntime/build/build_ort_riscv.sh` builds this archive automatically and links
it into `systolic_runner/imagenet_runner/ort_test`.
