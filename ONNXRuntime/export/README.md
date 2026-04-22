# Export Flows

For the shortest baseline path, see
`ONNXRuntime/REPRO.md` first.

This directory is split by model family:

- `ivit/`: I-ViT DeiT and Swin exporters
- `repq/`: RepQ-ViT calibration, reparameterization, and ONNX export

## I-ViT

Canonical entrypoint:

```bash
ONNXRuntime/export/ivit/export_onnx.sh --help
```

Direct exporters:

- `ONNXRuntime/export/ivit/export_deit_ivit_onnx.py`
- `ONNXRuntime/export/ivit/export_swin_ivit_onnx.py`

Wrapper scripts:

- `ONNXRuntime/export/ivit/export_ivit_onnx.sh`
- `ONNXRuntime/export/ivit/export_swin_ivit_onnx.sh`

## RepQ

Canonical entrypoint:

```bash
ONNXRuntime/export/repq/export_repq_onnx.sh --help
```

This flow loads a timm DeiT/Swin model, applies RepQ quantization and
calibration, then exports the lowered ONNX model used by the ORT path.

## Outputs

Generated ONNX files are typically written under `build/ort/`.
