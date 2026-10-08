#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
source "${REPO_ROOT}/env.sh" >/dev/null
cd "${REPO_ROOT}"

IMAGE="${IMAGE:-${SCRIPT_DIR}/test_cat.jpg}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/build/tvm/vfrec7_matrix}"
TIMEOUT="${TIMEOUT:-0}"
BUILD_ONLY="${BUILD_ONLY:-0}"

IVIT_DEIT_TINY_CHECKPOINT="${IVIT_DEIT_TINY_CHECKPOINT:-${IVIT_CHECKPOINT:-}}"
: "${IVIT_DEIT_TINY_CHECKPOINT:?Set IVIT_DEIT_TINY_CHECKPOINT to an I-ViT DeiT-Tiny QAT checkpoint}"
: "${IVIT_DEIT_SMALL_CHECKPOINT:?Set IVIT_DEIT_SMALL_CHECKPOINT to an I-ViT DeiT-Small QAT checkpoint}"
: "${IVIT_SWIN_TINY_CHECKPOINT:?Set IVIT_SWIN_TINY_CHECKPOINT to an I-ViT Swin-Tiny QAT checkpoint}"
: "${FLEXI_EVAL_ROOT:?Set FLEXI_EVAL_ROOT to the Flexi eval/ directory containing PTQ4ViT data}"

for checkpoint in \
  "${IVIT_DEIT_TINY_CHECKPOINT}" \
  "${IVIT_DEIT_SMALL_CHECKPOINT}" \
  "${IVIT_SWIN_TINY_CHECKPOINT}" \
  "${FLEXI_EVAL_ROOT}/data/ptq4_deit_t/e2e_model.pt"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Required checkpoint not found: ${checkpoint}" >&2
    exit 2
  fi
done
if [[ ! -f "${IMAGE}" ]]; then
  echo "Input image not found: ${IMAGE}" >&2
  exit 2
fi

run_pair() {
  local model_name="$1"
  local run_name="$2"
  local checkpoint="${3:-}"
  local -a common_args=(
    --image "${IMAGE}"
    --model-name "${model_name}"
    --tvm-backend llvm-gemmini
    --llvm-tir-vectorize
    --timeout "${TIMEOUT}"
  )
  if [[ "${BUILD_ONLY}" == "1" ]]; then
    common_args+=(--build-only)
  fi
  if [[ -n "${checkpoint}" ]]; then
    common_args+=(--checkpoint "${checkpoint}")
  fi

  echo "=== ${model_name}: vfrec7 OFF ==="
  env -u TVM_LLVM_VFREC7 python3 "${SCRIPT_DIR}/run_inference_spike.py" \
    "${common_args[@]}" \
    --output-dir "${OUTPUT_ROOT}/${run_name}/vfrec7_off"

  echo "=== ${model_name}: vfrec7 ON ==="
  TVM_LLVM_VFREC7=1 python3 "${SCRIPT_DIR}/run_inference_spike.py" \
    "${common_args[@]}" \
    --output-dir "${OUTPUT_ROOT}/${run_name}/vfrec7_on"
}

export FLEXI_EVAL_ROOT
run_pair deit_tiny_patch16_224 deit_tiny "${IVIT_DEIT_TINY_CHECKPOINT}"
run_pair deit_small_patch16_224 deit_small "${IVIT_DEIT_SMALL_CHECKPOINT}"
run_pair swin_tiny_patch4_window7_224 swin_tiny "${IVIT_SWIN_TINY_CHECKPOINT}"
run_pair ptq4_deit_tiny_patch16_224 ptq4_deit_tiny

echo
echo "TVM vfrec7 comparison completed. Artifacts: ${OUTPUT_ROOT}"
