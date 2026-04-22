#!/usr/bin/env bash
#
# Export ONNX model(s) for ORT Spike flow.
# Supports both DeiT-Tiny and Swin-Tiny.
#
# Usage examples:
#   ONNXRuntime/export/ivit/export_onnx.sh --model-name deit_tiny_patch16_224 --checkpoint /root/checkpoint_last.pth.tar
#   ONNXRuntime/export/ivit/export_onnx.sh --model-name swin_tiny_patch4_window7_224 --checkpoint /root/swin_checkpoint.pth.tar
#
# Note:
#   Swin uses custom-op exporter by default (DeiT-style I-ViT graph).
#   Use --swin-export-style legacy-qop only for compatibility experiments.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PY_EXPORTER="${SCRIPT_DIR}/export_onnx.py"

if [ ! -f "$PY_EXPORTER" ]; then
    echo "ERROR: exporter not found: $PY_EXPORTER"
    exit 1
fi

python3 "$PY_EXPORTER" "$@"
