#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
source "${REPO_ROOT}/env.sh" >/dev/null
export PYTHONUNBUFFERED=1

BUILD_DIR="${REPO_ROOT}/build"
CKPT_DIR="${BUILD_DIR}/checkpoints"
ORT_DIR="${BUILD_DIR}/ort"
TVM_DIR="${BUILD_DIR}/tvm"
IMAGE="${REPO_ROOT}/scripts/gemmini/test_cat.jpg"

mkdir -p "${CKPT_DIR}" "${ORT_DIR}" "${TVM_DIR}"

DEIT_CKPT="${CKPT_DIR}/ivit_deit_tiny_random.pth.tar"
SWIN_CKPT="${CKPT_DIR}/ivit_swin_tiny_random.pth.tar"

if [[ ! -f "${DEIT_CKPT}" ]]; then
    python3 "${REPO_ROOT}/tools/generate_random_ivit_checkpoint.py" \
        --model-name deit_tiny_patch16_224 \
        --output "${DEIT_CKPT}"
fi

if [[ ! -f "${SWIN_CKPT}" ]]; then
    python3 "${REPO_ROOT}/tools/generate_random_ivit_checkpoint.py" \
        --model-name swin_tiny_patch4_window7_224 \
        --output "${SWIN_CKPT}"
fi

if [[ ! -x "${ORT_RISCV_DIR:-}/systolic_runner/imagenet_runner/ort_test" ]]; then
    "${REPO_ROOT}/ONNXRuntime/build/build_ort_riscv.sh"
fi

python3 "${REPO_ROOT}/ONNXRuntime/export/ivit/export_onnx.py" \
    --model-name deit_tiny_patch16_224 \
    --checkpoint "${DEIT_CKPT}" \
    --output "${ORT_DIR}/ivit_deit_tiny_int8.onnx"

python3 "${REPO_ROOT}/ONNXRuntime/export/ivit/export_onnx.py" \
    --model-name swin_tiny_patch4_window7_224 \
    --checkpoint "${SWIN_CKPT}" \
    --output "${ORT_DIR}/ivit_swin_tiny_int8.onnx"

python3 "${REPO_ROOT}/ONNXRuntime/export/repq/export_repq_onnx.py" \
    --model deit_tiny \
    --allow-random-init \
    --allow-random-calibration \
    --device cpu \
    --calib-batchsize 4 \
    --calib-num-samples 8 \
    --w-bits 8 \
    --a-bits 8 \
    --lower-qlinear-matmul \
    --repq-gemmini-kernel-mode approx \
    --output "${ORT_DIR}/repq_deit_tiny_w8a8_lowered.onnx"

"${REPO_ROOT}/ONNXRuntime/run/run_ort_spike.sh" \
    "${IMAGE}" 1 "${ORT_DIR}/ivit_deit_tiny_int8.onnx" 1 \
    --model-name deit_tiny_patch16_224 \
    --log-file "${ORT_DIR}/ivit_deit_spike_x1.log"

"${REPO_ROOT}/ONNXRuntime/run/run_ort_spike.sh" \
    "${IMAGE}" 1 "${ORT_DIR}/ivit_swin_tiny_int8.onnx" 1 \
    --model-name swin_tiny_patch4_window7_224 \
    --log-file "${ORT_DIR}/ivit_swin_spike_x1.log"

"${REPO_ROOT}/ONNXRuntime/run/run_ort_spike.sh" \
    "${IMAGE}" 1 "${ORT_DIR}/repq_deit_tiny_w8a8_lowered.onnx" 1 \
    --model-name deit_tiny_patch16_224 \
    --log-file "${ORT_DIR}/repq_deit_spike_x1.log"

python3 "${REPO_ROOT}/scripts/gemmini/run_inference_spike.py" \
    --image "${IMAGE}" \
    --checkpoint "${DEIT_CKPT}" \
    --model-name deit_tiny_patch16_224 \
    --output-dir "${TVM_DIR}/deit" \
    --timeout 900

python3 "${REPO_ROOT}/scripts/gemmini/run_inference_spike.py" \
    --image "${IMAGE}" \
    --checkpoint "${SWIN_CKPT}" \
    --model-name swin_tiny_patch4_window7_224 \
    --output-dir "${TVM_DIR}/swin" \
    --timeout 900

python3 "${REPO_ROOT}/scripts/gemmini/run_repq_inference_spike.py" \
    --model deit_tiny \
    --image "${IMAGE}" \
    --allow-random-init \
    --allow-random-calibration \
    --device cpu \
    --calib-batchsize 4 \
    --calib-num-samples 8 \
    --w-bits 8 \
    --a-bits 8 \
    --output-dir "${TVM_DIR}/repq_deit" \
    --timeout 900

echo
echo "Smoke run finished."
echo "  ORT outputs: ${ORT_DIR}"
echo "  TVM outputs: ${TVM_DIR}"
