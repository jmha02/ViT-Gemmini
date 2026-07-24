# I-ViT ORT Custom Ops

This directory contains ONNX Runtime custom ops used by the integer-only I-ViT
graphs:

- `ivit.QLayernorm`
- `ivit.Shiftmax`
- `ivit.ShiftGELU`
- `ivit.GemminiMatMulInteger`
- `ivit.FQPTFLayerNorm` — flexi FQ-DeiT PTF LayerNorm (CPU)
- `ivit.FQQKMatMul` — flexi FQ-DeiT Q@K^T reference kernel (CPU only; **not used** in `fq_deit_tiny_int8.onnx` — export uses `GemminiMatMulInteger` instead)
- `ivit.FQLISSoftmax` — flexi FQ-DeiT log-domain softmax (CPU)
- `ivit.FQAttnVMatMul` — flexi FQ-DeiT Attn@V shift reduction (CPU, not Gemmini)
- `ivit.RequantizeInt32`
- `ivit.TwinSoftmaxMatMul` — PTQ4ViT: fused float Softmax(+scale) + twin quant + **Gemmini** int MM (`tiled_matmul_auto`). Inputs: scores `[B,H,N,N]`, V int8 `[B,H,N,D]`, split, a_interval, bi`[H]`, attn_scale. **No RVV.**
- `ivit.TwinGeluLinear` — PTQ4ViT: fused Erf-GELU + twin quant + **Gemmini** int MM. Input0 is **pre-GELU** activations. **No RVV.**
- `ivit.SymQuantizeI8` — `cast_i8(clip(round(x/scale), -128, 127))` (replaces Div/Round/Clip/Cast chains)
- `ivit.FloatLayerNorm` — float LN `axis=-1` (gamma, beta, eps); same math as ReduceMean chain. **No RVV.**

FQ-DeiT Gemmini vs CPU map: see `docs/fq_deit_gemmini_map.md`.
PTQ4ViT sync / iree note: see `docs/ptq4_sync_note.md`.

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
