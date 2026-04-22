#!/usr/bin/env bash
set -euo pipefail

_vit_this="$0"
if [[ -n "${BASH_VERSION:-}" ]]; then
    _vit_this="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
    _vit_this="${(%):-%N}"
fi

VIT_GEMMINI_ROOT="$(cd "$(dirname "${_vit_this}")" && pwd -P)"
unset _vit_this
export VIT_GEMMINI_ROOT

if [[ -z "${TVM_HOME:-}" ]]; then
    if [[ -d "${VIT_GEMMINI_ROOT}/tvm-gemmini" ]]; then
        export TVM_HOME="${VIT_GEMMINI_ROOT}/tvm-gemmini"
    elif [[ -d "/root/flexi/third-party/I-ViT-Gemmini/tvm-gemmini" ]]; then
        export TVM_HOME="/root/flexi/third-party/I-ViT-Gemmini/tvm-gemmini"
    fi
fi

if [[ -z "${CHIPYARD_DIR:-}" && -d "/root/flexi/chipyard" ]]; then
    export CHIPYARD_DIR="/root/flexi/chipyard"
fi

if [[ -z "${RISCV:-}" && -n "${CHIPYARD_DIR:-}" && -d "${CHIPYARD_DIR}/.conda-env/riscv-tools" ]]; then
    export RISCV="${CHIPYARD_DIR}/.conda-env/riscv-tools"
fi

if [[ -z "${ORT_RISCV_DIR:-}" && -n "${TVM_HOME:-}" && -d "${TVM_HOME}/3rdparty/gemmini/software/onnxruntime-riscv" ]]; then
    export ORT_RISCV_DIR="${TVM_HOME}/3rdparty/gemmini/software/onnxruntime-riscv"
fi

export REPQ_CLASSIFICATION_ROOT="${REPQ_CLASSIFICATION_ROOT:-${VIT_GEMMINI_ROOT}/RepQ-ViT/classification}"

if [[ -n "${TVM_HOME:-}" && -d "${TVM_HOME}/python" ]]; then
    case ":${PYTHONPATH:-}:" in
        *":${TVM_HOME}/python:"*) ;;
        *) export PYTHONPATH="${TVM_HOME}/python${PYTHONPATH:+:${PYTHONPATH}}" ;;
    esac
fi

if [[ -n "${TVM_HOME:-}" && -d "${TVM_HOME}/build" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *":${TVM_HOME}/build:"*) ;;
        *) export LD_LIBRARY_PATH="${TVM_HOME}/build${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
    esac
fi

if [[ -d "/usr/lib/x86_64-linux-gnu" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *":/usr/lib/x86_64-linux-gnu:"*) ;;
        *) export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
    esac
fi

if [[ -n "${CHIPYARD_DIR:-}" && -f "${CHIPYARD_DIR}/.conda-env/lib/libstdc++.so.6" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *":${CHIPYARD_DIR}/.conda-env/lib:"*) ;;
        *) export LD_LIBRARY_PATH="${CHIPYARD_DIR}/.conda-env/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
    esac
    case ":${LD_PRELOAD:-}:" in
        *":${CHIPYARD_DIR}/.conda-env/lib/libstdc++.so.6:"*) ;;
        *) export LD_PRELOAD="${CHIPYARD_DIR}/.conda-env/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}" ;;
    esac
fi

echo "VIT_GEMMINI_ROOT=${VIT_GEMMINI_ROOT}"
echo "TVM_HOME=${TVM_HOME:-}"
echo "CHIPYARD_DIR=${CHIPYARD_DIR:-}"
echo "RISCV=${RISCV:-}"
echo "ORT_RISCV_DIR=${ORT_RISCV_DIR:-}"
echo "REPQ_CLASSIFICATION_ROOT=${REPQ_CLASSIFICATION_ROOT}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
echo "LD_PRELOAD=${LD_PRELOAD:-}"
