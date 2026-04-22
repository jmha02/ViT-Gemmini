#!/usr/bin/env bash
#
# Backward-compatible wrapper for DeiT-Tiny ONNX export.
#
# Usage:
#   ONNXRuntime/export/ivit/export_ivit_onnx.sh --checkpoint /path/to/checkpoint_last.pth.tar [--output /path/to/model.onnx]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
EXPORT_SH="${SCRIPT_DIR}/export_onnx.sh"

if [ ! -x "$EXPORT_SH" ]; then
    echo "ERROR: exporter wrapper not found: $EXPORT_SH"
    exit 1
fi

"$EXPORT_SH" --model-name deit_tiny_patch16_224 "$@"
