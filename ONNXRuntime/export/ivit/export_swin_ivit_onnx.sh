#!/usr/bin/env bash
#
# Convenience wrapper for Swin-Tiny I-ViT ONNX export (custom-op default path).
#
# Usage:
#   ONNXRuntime/export/ivit/export_swin_ivit_onnx.sh --checkpoint /path/to/swin_ckpt.pth.tar [--output /path/to/model.onnx]
#   ONNXRuntime/export/ivit/export_swin_ivit_onnx.sh --allow-random-init --output /path/to/model.onnx
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
EXPORT_SH="${SCRIPT_DIR}/export_onnx.sh"

if [ ! -x "$EXPORT_SH" ]; then
    echo "ERROR: exporter wrapper not found: $EXPORT_SH"
    exit 1
fi

"$EXPORT_SH" --model-name swin_tiny_patch4_window7_224 "$@"
