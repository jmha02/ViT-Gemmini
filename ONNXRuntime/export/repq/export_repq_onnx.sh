#!/usr/bin/env bash
#
# Convenience wrapper for host-side RepQ-ViT -> ONNX export.
#
# Examples:
#   ONNXRuntime/export/repq/export_repq_onnx.sh --model deit_tiny --dataset /data/imagenet --output build/ort/repq_deit_tiny_w8a8_semantic.onnx
#   ONNXRuntime/export/repq/export_repq_onnx.sh --model deit_tiny --allow-random-init --allow-random-calibration --verify-ort
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PY_EXPORTER="${SCRIPT_DIR}/export_repq_onnx.py"

if [[ ! -f "${PY_EXPORTER}" ]]; then
    echo "ERROR: exporter not found: ${PY_EXPORTER}"
    exit 1
fi

python3 "${PY_EXPORTER}" "$@"
