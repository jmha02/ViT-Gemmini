#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"

# Resolve the sibling TVM checkout and the existing Chipyard cross-toolchain.
source "${REPO_ROOT}/env.sh" >/dev/null
unset LD_PRELOAD

if [[ -z "${ORT_RISCV_DIR:-}" || ! -d "${ORT_RISCV_DIR:-}" ]]; then
    echo "ERROR: onnxruntime-riscv submodule not found: ${ORT_RISCV_DIR:-<unset>}" >&2
    exit 1
fi
if [[ -z "${RISCV:-}" || ! -x "${RISCV:-}/bin/riscv64-unknown-linux-gnu-g++" ]]; then
    echo "ERROR: RISC-V GNU toolchain not found under RISCV=${RISCV:-<unset>}" >&2
    exit 1
fi

if [[ -n "${CHIPYARD_DIR:-}" && -d "${CHIPYARD_DIR}/.conda-env/bin" ]]; then
    PATH="${CHIPYARD_DIR}/.conda-env/bin:${PATH}"
fi
PATH="${RISCV}/bin:${PATH}"
export PATH

BUILD_JOBS="${BUILD_JOBS:-8}"
if [[ ! "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: BUILD_JOBS must be a positive integer (got ${BUILD_JOBS})." >&2
    exit 1
fi

export CC="${RISCV}/bin/riscv64-unknown-linux-gnu-gcc"
export CXX="${RISCV}/bin/riscv64-unknown-linux-gnu-g++"
export CFLAGS="-march=rv64imafdc -mabi=lp64d"
export CXXFLAGS="-march=rv64imafdc -mabi=lp64d"

PROTOC_DIR="${ORT_RISCV_DIR}/build/protoc"
if [[ ! -x "${PROTOC_DIR}/bin/protoc" ]]; then
    mkdir -p "${PROTOC_DIR}"
    curl -fsSL \
        "https://github.com/protocolbuffers/protobuf/releases/download/v3.16.0/protoc-3.16.0-linux-x86_64.zip" \
        -o "${PROTOC_DIR}/protoc.zip"
    unzip -oq "${PROTOC_DIR}/protoc.zip" -d "${PROTOC_DIR}"
fi

echo "Building ONNX Runtime for Gemmini Spike (${BUILD_JOBS} jobs)."
cd "${ORT_RISCV_DIR}"
python3 tools/ci_build/build.py \
    --riscv \
    --skip_submodule_sync \
    --update \
    --build \
    --build_dir="${ORT_RISCV_DIR}/build" \
    --config=Release \
    --parallel "${BUILD_JOBS}" \
    --cmake_extra_defines \
        onnxruntime_DEV_MODE=OFF \
        onnxruntime_SYSTOLIC_FP32=ON \
        onnxruntime_SYSTOLIC_INT8=OFF \
        "CMAKE_CXX_FLAGS=-march=rv64imafdc -mabi=lp64d -Wno-error=stringop-overflow"

OPS_DIR="${REPO_ROOT}/ONNXRuntime/ort_ivit_ops"
OPS_BUILD_DIR="${REPO_ROOT}/build/ort/ort_ivit_ops"
make -C "${OPS_DIR}" -j"${BUILD_JOBS}" \
    ORT_RISCV_DIR="${ORT_RISCV_DIR}" RISCV="${RISCV}"
make -C "${OPS_DIR}" -j"${BUILD_JOBS}" \
    ORT_RISCV_DIR="${ORT_RISCV_DIR}" RISCV="${RISCV}" host

RUNNER_DIR="${ORT_RISCV_DIR}/systolic_runner/imagenet_runner"
make -C "${RUNNER_DIR}" -j"${BUILD_JOBS}" ort_test \
    root_path="${ORT_RISCV_DIR}" \
    build_path="${ORT_RISCV_DIR}/build/Release" \
    extra_libs="${OPS_BUILD_DIR}/libivit_ops.a" \
    extra_defs=-DUSE_CUSTOM_OP_LIBRARY \
    CXX="${CXX}"

echo
echo "Built Gemmini ORT runner: ${RUNNER_DIR}/ort_test"
echo "Built RISC-V custom ops: ${OPS_BUILD_DIR}/libivit_ops.a"
echo "Built host custom ops: ${OPS_BUILD_DIR}/libivit_ops_host.so"
