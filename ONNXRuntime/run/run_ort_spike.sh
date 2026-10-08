#!/usr/bin/env bash
#
# Run I-ViT ONNX model on Spike + Gemmini via onnxruntime-riscv.
#
# This script runs exported ONNX models directly from I-ViT-Gemmini artifacts.
#
# Usage:
#   ONNXRuntime/run/run_ort_spike.sh [image] [mode] [model] [opt_level] [--model-name <name>] [--log-file <path>] [--trace-prefix <path>]
#
# Positional args:
#   image     : input image path (default: scripts/gemmini/test_cat.jpg)
#   mode      : 0=CPU, 1=Gemmini-OS, 2=Gemmini-WS (default: 1)
#   model     : ONNX model path (default: auto-select from model-name)
#   opt_level : ORT graph optimization level (default: 1)
#
# Environment:
#   USE_CUSTOM_OPS=1|0  (default: 1)
#   MODEL_NAME=deit_tiny_patch16_224|deit_small_patch16_224|fq_deit_tiny_patch16_224|ptq4_deit_tiny_patch16_224|swin_tiny_patch4_window7_224|swin_small_patch4_window7_224
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ONNXRT_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
REPO_ROOT="$(cd "$ONNXRT_DIR/.." && pwd -P)"
source "${REPO_ROOT}/env.sh" >/dev/null

ORT_RISCV_DIR="${ORT_RISCV_DIR:-${TVM_HOME:-${REPO_ROOT}/tvm-gemmini}/3rdparty/gemmini/software/onnxruntime-riscv}"
ORT_TEST_BIN="${ORT_RISCV_DIR}/systolic_runner/imagenet_runner/ort_test"
PK="${PK:-}"
if [[ -z "${PK}" && -n "${CHIPYARD_DIR:-}" ]]; then
    PK="${CHIPYARD_DIR}/toolchains/riscv-tools/riscv-pk/build/pk"
fi

IMAGE="${REPO_ROOT}/scripts/gemmini/test_cat.jpg"
MODE="1"
MODEL=""
OPT_LEVEL="1"
USE_CUSTOM_OPS="${USE_CUSTOM_OPS:-1}"
MODEL_NAME="${MODEL_NAME:-deit_tiny_patch16_224}"
LOG_FILE=""
INPUT_TENSOR=""
TRACE_PREFIX=""
TRACE_PATH_FOR_RUN=""
TENSOR_FILE=""
TEMP_TENSOR_CREATED=0

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-name)
            MODEL_NAME="${2:-}"
            shift 2
            ;;
        --log-file)
            LOG_FILE="${2:-}"
            shift 2
            ;;
        --trace-prefix)
            TRACE_PREFIX="${2:-}"
            shift 2
            ;;
        --tensor-file)
            TENSOR_FILE="${2:-}"
            shift 2
            ;;
        -h|--help)
            sed -n '1,40p' "$0"
            exit 0
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                POSITIONAL+=("$1")
                shift
            done
            ;;
        -*)
            echo "ERROR: unknown option: $1"
            exit 1
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if (( ${#POSITIONAL[@]} > 0 )); then
    IMAGE="${POSITIONAL[0]}"
fi
if (( ${#POSITIONAL[@]} > 1 )); then
    MODE="${POSITIONAL[1]}"
fi
if (( ${#POSITIONAL[@]} > 2 )); then
    MODEL="${POSITIONAL[2]}"
fi
if (( ${#POSITIONAL[@]} > 3 )); then
    OPT_LEVEL="${POSITIONAL[3]}"
fi

MODEL_DIR_LOCAL="${REPO_ROOT}/build/ort"

DEIT_MODEL_STATIC="${MODEL_DIR_LOCAL}/ivit_tiny_int8.onnx"
DEIT_MODEL_PARITY="${MODEL_DIR_LOCAL}/ivit_tiny_int8_parity.onnx"
DEIT_MODEL_DYNAMIC="${MODEL_DIR_LOCAL}/ivit_tiny_int8_dynamic.onnx"
DEIT_SMALL_MODEL="${MODEL_DIR_LOCAL}/ivit_deit_small_int8.onnx"
FQ_DEIT_MODEL="${MODEL_DIR_LOCAL}/fq_deit_tiny_int8.onnx"
PTQ4_DEIT_MODEL="${MODEL_DIR_LOCAL}/ptq4_deit_tiny_int8.onnx"
SWIN_MODEL_INT8="${MODEL_DIR_LOCAL}/swin_tiny_int8.onnx"
SWIN_MODEL_FP32="${MODEL_DIR_LOCAL}/swin_tiny_fp32.onnx"
SWIN_SMALL_MODEL_INT8="${MODEL_DIR_LOCAL}/swin_small_int8.onnx"

resolve_spike() {
    local preferred="${SPIKE_BIN:-}"
    local riscv_spike="${RISCV:-}/bin/spike"
    local chipyard_spike="${CHIPYARD_DIR:-}/.conda-env/riscv-tools/bin/spike"
    local legacy="${CHIPYARD_DIR:-}/toolchains/riscv-tools/riscv-isa-sim/build/spike"
    local legacy_real=""

    if [[ -n "$preferred" && -x "$preferred" ]]; then
        echo "$preferred"
        return 0
    fi

    if [[ -x "$riscv_spike" ]]; then
        echo "$riscv_spike"
        return 0
    fi

    if [[ -x "$chipyard_spike" ]]; then
        echo "$chipyard_spike"
        return 0
    fi

    if [ -x "$legacy" ]; then
        legacy_real="$(readlink -f "$legacy" || true)"
        if [[ "$(basename "$legacy_real")" == "xspike" ]]; then
            echo "[WARN] Using xspike-backed binary: $legacy" >&2
            echo "[WARN] If run hangs/fails, build/use chipyard/.conda-env spike." >&2
        fi
        echo "$legacy"
        return 0
    fi

    return 1
}

if [ -z "$MODEL" ]; then
    case "$MODEL_NAME" in
        deit_tiny_patch16_224)
            if [ -f "$DEIT_MODEL_STATIC" ]; then
                MODEL="$DEIT_MODEL_STATIC"
            elif [ -f "$DEIT_MODEL_PARITY" ]; then
                MODEL="$DEIT_MODEL_PARITY"
            else
                MODEL="$DEIT_MODEL_DYNAMIC"
            fi
            ;;
        swin_tiny_patch4_window7_224)
            if [ -f "$SWIN_MODEL_INT8" ]; then
                MODEL="$SWIN_MODEL_INT8"
            else
                MODEL="$SWIN_MODEL_FP32"
            fi
            ;;
        deit_small_patch16_224)
            MODEL="$DEIT_SMALL_MODEL"
            ;;
        fq_deit_tiny_patch16_224)
            MODEL="$FQ_DEIT_MODEL"
            ;;
        ptq4_deit_tiny_patch16_224)
            MODEL="$PTQ4_DEIT_MODEL"
            ;;
        swin_small_patch4_window7_224)
            MODEL="$SWIN_SMALL_MODEL_INT8"
            ;;
        *)
            echo "ERROR: unsupported MODEL_NAME: $MODEL_NAME"
            exit 1
            ;;
    esac
fi

SPIKE="$(resolve_spike || true)"

if [ -z "$SPIKE" ]; then
    echo "ERROR: spike binary not found."
    echo "  Tried:"
    echo "    - ${SPIKE_BIN:-<set SPIKE_BIN>}"
    echo "    - ${RISCV:-<set RISCV>}/bin/spike"
    echo "    - ${CHIPYARD_DIR:-<set CHIPYARD_DIR>}/.conda-env/riscv-tools/bin/spike"
    exit 1
fi

if [ ! -x "$PK" ]; then
    echo "ERROR: pk not found: ${PK:-<set PK or CHIPYARD_DIR>}"
    exit 1
fi

if [ ! -x "$ORT_TEST_BIN" ]; then
    echo "ERROR: ort_test not found: $ORT_TEST_BIN"
    echo "  Build first:"
    echo "    ${REPO_ROOT}/ONNXRuntime/build/build_ort_riscv.sh"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: ONNX model not found: $MODEL"
    echo "  Export first:"
    echo "    ${REPO_ROOT}/ONNXRuntime/export/ivit/export_onnx.sh --checkpoint <checkpoint>"
    exit 1
fi

if command -v python3 >/dev/null 2>&1; then
    ONNX_INT8_SUMMARY="$(python3 - "$MODEL" 2>/dev/null <<'PY'
import sys
from collections import Counter
try:
    import onnx
except Exception:
    print("")
    raise SystemExit(0)
model = onnx.load(sys.argv[1])
counts = Counter(node.op_type for node in model.graph.node)
print(
    f"{counts.get('MatMulInteger', 0)} "
    f"{counts.get('GemminiMatMulInteger', 0)} "
    f"{counts.get('QLinearMatMul', 0)} "
    f"{counts.get('QLinearConv', 0)} "
    f"{counts.get('RepQUniformMatMul', 0)} "
    f"{counts.get('RepQLogMatMul', 0)}"
)
PY
)"
    if [ -n "$ONNX_INT8_SUMMARY" ]; then
        MM_INT="$(echo "$ONNX_INT8_SUMMARY" | awk '{print $1}')"
        GMM_INT="$(echo "$ONNX_INT8_SUMMARY" | awk '{print $2}')"
        QMM="$(echo "$ONNX_INT8_SUMMARY" | awk '{print $3}')"
        QCONV="$(echo "$ONNX_INT8_SUMMARY" | awk '{print $4}')"
        RU_MM="$(echo "$ONNX_INT8_SUMMARY" | awk '{print $5}')"
        RL_MM="$(echo "$ONNX_INT8_SUMMARY" | awk '{print $6}')"
        if [ "${MM_INT:-0}" = "0" ] && [ "${GMM_INT:-0}" = "0" ] && [ "${QMM:-0}" = "0" ] && [ "${QCONV:-0}" = "0" ] && [ "${RU_MM:-0}" = "0" ] && [ "${RL_MM:-0}" = "0" ]; then
            echo "[WARN] ONNX graph has no Gemmini-backed MatMul/Conv custom ops."
            echo "[WARN] Gemmini acceleration may be limited for this model."
        fi
    fi
fi

if [ -n "$TENSOR_FILE" ]; then
    if [ ! -f "$TENSOR_FILE" ]; then
        echo "ERROR: tensor file not found: $TENSOR_FILE"
        exit 1
    fi
    INPUT_TENSOR="$TENSOR_FILE"
else
    if [ ! -f "$IMAGE" ]; then
        echo "ERROR: image not found: $IMAGE"
        exit 1
    fi

    mkdir -p "$MODEL_DIR_LOCAL"
    INPUT_TENSOR="$(mktemp "${MODEL_DIR_LOCAL}/ort_input_XXXXXX.bin")"
    TEMP_TENSOR_CREATED=1
    trap 'if [ "$TEMP_TENSOR_CREATED" = "1" ]; then rm -f "$INPUT_TENSOR"; fi' EXIT

    python3 "${ONNXRT_DIR}/tools/preprocess_image_to_tensor.py"         --image "$IMAGE"         --model-name "$MODEL_NAME"         --output "$INPUT_TENSOR" >/dev/null
fi

if [[ ! "$MODE" =~ ^[0-2]$ ]]; then
    echo "ERROR: mode must be 0, 1, or 2 (got: $MODE)"
    exit 1
fi

MODE_NAME=("CPU-fallback" "Gemmini-OS" "Gemmini-WS")

echo "=== I-ViT ONNX Runtime on Spike ==="
echo "Spike     : $SPIKE"
echo "PK        : $PK"
echo "Runner    : $ORT_TEST_BIN"
echo "ModelName : $MODEL_NAME"
echo "Model     : $MODEL"
if [ -n "$TENSOR_FILE" ]; then
    echo "Image     : <skipped>"
else
    echo "Image     : $IMAGE"
fi
echo "Tensor    : $INPUT_TENSOR"
echo "Mode (-x) : ${MODE_NAME[$MODE]} ($MODE)"
echo "Opt (-O)  : $OPT_LEVEL"
echo "CustomOps : $USE_CUSTOM_OPS"
if [ -n "$LOG_FILE" ]; then
    echo "Log file  : $LOG_FILE"
fi
if [ -n "$TRACE_PREFIX" ]; then
    TRACE_PATH_FOR_RUN="$TRACE_PREFIX"
elif [ -n "$LOG_FILE" ]; then
    TRACE_PATH_FOR_RUN="${LOG_FILE%.log}_profile"
fi

if [ -n "$TRACE_PREFIX" ]; then
    echo "Trace pref: $TRACE_PREFIX"
elif [ -n "$TRACE_PATH_FOR_RUN" ]; then
    echo "Trace pref: $TRACE_PATH_FOR_RUN (auto)"
fi
echo

CMD=(
    "$SPIKE" --extension=gemmini "$PK" "$ORT_TEST_BIN"
    -m "$MODEL"
    --tensor "$INPUT_TENSOR"
    -x "$MODE"
    -O "$OPT_LEVEL"
)

if [ "$USE_CUSTOM_OPS" = "1" ]; then
    CMD+=(-k 1)
fi

if [ -n "$TRACE_PATH_FOR_RUN" ]; then
    mkdir -p "$(dirname "$TRACE_PATH_FOR_RUN")"
    CMD+=(-t "$TRACE_PATH_FOR_RUN")
fi

printf "Command: "
printf "%q " "${CMD[@]}"
echo
echo

postprocess_rdcycle_log() {
    local log_path="$1"
    local out_prefix=""
    local node_csv=""
    if ! command -v python3 >/dev/null 2>&1; then
        echo "[WARN] python3 not found; skipping ORT raw rdcycle CSV generation."
        return 0
    fi
    if ! rg -q '^\[ORT_NODE_CYCLES\],' "$log_path"; then
        echo "[Info] No [ORT_NODE_CYCLES] rows found in $log_path; skipping ORT cycle CSV postprocess."
        return 0
    fi
    if [ -n "$TRACE_PREFIX" ]; then
        if [[ "$TRACE_PREFIX" == *_rdcycle ]]; then
            out_prefix="$TRACE_PREFIX"
        else
            out_prefix="${TRACE_PREFIX}_rdcycle"
        fi
    else
        out_prefix="${log_path%.log}_rdcycle"
    fi
    python3 "${ONNXRT_DIR}/profiling/ort_log_node_cycles.py" \
        --log "$log_path" \
        --out-prefix "$out_prefix" || {
        echo "[WARN] Failed to parse ORT raw rdcycle node log."
        return 0
    }
    echo "[Info] ORT raw rdcycle CSV prefix: ${out_prefix}"
    node_csv="${out_prefix}_node_cycles.csv"
    if [[ -f "$node_csv" ]]; then
        python3 "${ONNXRT_DIR}/profiling/ort_fine_grained_cycles.py" \
            --node-csv "$node_csv" \
            --out-prefix "$out_prefix" || {
            echo "[WARN] Failed to build ORT fine-grained cycle CSVs."
            return 0
        }
        echo "[Info] ORT fine-grained CSV prefix: ${out_prefix}"
    fi
}

if [ -n "$LOG_FILE" ]; then
    mkdir -p "$(dirname "$LOG_FILE")"
    set +e
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
    rc="${PIPESTATUS[0]}"
    set -e
    if [ "$rc" -eq 0 ]; then
        postprocess_rdcycle_log "$LOG_FILE"
    fi
    exit "$rc"
fi

"${CMD[@]}"
