#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"



NUM_WORKERS=4

TASK_NAME=""
EXP_NAME=""

CONFIG="config/default.yml"
MODEL_PATH=""

INPUT=""
OUTPUT_DIR=""
LOG_DIR=""

START_IDX=0
APPEND=1      

ENABLE_THINKING=0   

n_gpus_per_node=8
nnodes=4

EXTRA_PY_ARGS=()

usage() {
  cat <<EOF
Usage: $0 [options] [-- extra_python_args...]

Options:
  --task-name NAME            task name (default: ${TASK_NAME})
  --exp-name NAME             exp name (default: ${EXP_NAME})
  --num-workers N             number of workers (default: ${NUM_WORKERS})
  --config PATH               verifier config yaml (default: ${CONFIG})
  --model-path PATH           hf model path (default: ${MODEL_PATH})
  --input PATH                input jsonl (default: ${INPUT})
  --output-dir DIR            output directory (default: ${OUTPUT_DIR})
  --log-dir DIR               log directory (default: ${LOG_DIR})
  --start-idx N               start_idx (default: ${START_IDX})
  --no-append                 disable append mode for output
  --enable-thinking           use enable_thinking for qwen3
  -h, --help                  show help

EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nnodes)
      nnodes="$2"; shift 2;;
    --task-name)
      TASK_NAME="$2"; shift 2;;
    --exp-name)
      EXP_NAME="$2"; shift 2;;
    --num-workers)
      NUM_WORKERS="$2"; shift 2;;
    --config)
      CONFIG="$2"; shift 2;;
    --model-path)
      MODEL_PATH="$2"; shift 2;;
    --input)
      INPUT="$2"; shift 2;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2;;
    --log-dir)
      LOG_DIR="$2"; shift 2;;
    --start-idx)
      START_IDX="$2"; shift 2;;
    --no-append)
      APPEND=0; shift 1;;
    --enable-thinking)
      ENABLE_THINKING=1; shift 1;;
    -h|--help)
      usage; exit 0;;
    --)
      shift
      EXTRA_PY_ARGS=("$@")
      break;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1;;
  esac
done

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

OUTPUT="${OUTPUT_DIR}/${TASK_NAME}-${EXP_NAME}.jsonl"
LOG_FILE="${LOG_DIR}/${TASK_NAME}_${EXP_NAME}.log"

APPEND_FLAG=()
if [[ "${APPEND}" -eq 1 ]]; then
  APPEND_FLAG+=(--append)
fi

ENABLE_THINKING_ARG=()
if [[ "${ENABLE_THINKING}" -eq 1 ]]; then
  ENABLE_THINKING_ARG+=(--enable-thinking)
fi

echo "===> Running verify:"
echo "  TASK_NAME         = ${TASK_NAME}"
echo "  EXP_NAME          = ${EXP_NAME}"
echo "  NUM_WORKERS       = ${NUM_WORKERS}"
echo "  INPUT             = ${INPUT}"
echo "  OUTPUT            = ${OUTPUT}"
echo "  MODEL_PATH        = ${MODEL_PATH}"
echo "  CONFIG            = ${CONFIG}"
echo "  START_IDX         = ${START_IDX}"
echo "  APPEND            = ${APPEND}"
echo "  ENABLE_THINKING   = ${ENABLE_THINKING}  # 1=on, 0=off"
echo "  LOG_FILE          = ${LOG_FILE}"
echo "  EXTRA_PY_ARGS     = ${EXTRA_PY_ARGS[@]:-<none>}"
echo







NNODES=${nnodes}

NGPUS_PER_NODE=${n_gpus_per_node}

SHARED_ROOT=${SHARED_ROOT:-"/tmp"}

RAY_PORT=${RAY_PORT:-6380}

CLUSTER_DIR="${SHARED_ROOT}/${EXP_NAME}"
LOCK_DIR="${CLUSTER_DIR}/lock"
HEAD_IP_FILE="${CLUSTER_DIR}/head_ip"

mkdir -p "${CLUSTER_DIR}"

THIS_IP=${THIS_IP:-$(hostname -I | awk '{print $1}')}

echo "[NODE] Hostname: $(hostname), IP: ${THIS_IP}"
echo "[NODE] Shared cluster dir: ${CLUSTER_DIR}"

if mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "[HEAD] I AM HEAD NODE."

    echo "${THIS_IP}" > "${HEAD_IP_FILE}"
    echo "[HEAD] Wrote HEAD_IP: ${THIS_IP}"

    ray stop || true

    ray start --head \
        --node-ip-address="${THIS_IP}" \
        --port="${RAY_PORT}" \
        --num-gpus="${NGPUS_PER_NODE}"

    echo "[HEAD] Ray head started at ${THIS_IP}:${RAY_PORT}"
    export RAY_ADDRESS="${THIS_IP}:${RAY_PORT}"
    echo "[HEAD] Export RAY_ADDRESS=${RAY_ADDRESS}"

    echo "[HEAD] CWD: $(pwd)"

    export DESIRED_GPUS=$((NNODES * NGPUS_PER_NODE))
    echo "[HEAD] Waiting for ${DESIRED_GPUS} GPUs in the cluster..."

    python - << 'PY'
import os, time, ray

addr = os.environ.get("RAY_ADDRESS", "auto")
desired = int(os.environ["DESIRED_GPUS"])

ray.init(address=addr)
for i in range(600):
    res = ray.cluster_resources()
    gpu = int(res.get("GPU", 0))
    print(f"[HEAD][WAIT] current GPUs = {gpu}, desired = {desired}")
    if gpu >= desired:
        print("[HEAD][WAIT] enough GPUs available, proceed.")
        break
    time.sleep(5)
else:
    print("[HEAD][WAIT] Timeout waiting for GPUs, proceed anyway (may crash in trainer).")
ray.shutdown()
PY
    rm -rf ${CLUSTER_DIR}
    echo "[HEAD] Launch training"
    python src/run_verify_multihead.py \
    --config "${CONFIG}" \
    --input  "${INPUT}" \
    --output "${OUTPUT}" \
    --model_path "${MODEL_PATH}" \
    --record-batch-size 1 \
    --include_full_meta \
    --start_idx "${START_IDX}" \
    --num-workers "${NUM_WORKERS}" \
    --max-inflight-batches "${NUM_WORKERS}" \
    "${APPEND_FLAG[@]}" \
    "${ENABLE_THINKING_ARG[@]}" \
    "${EXTRA_PY_ARGS[@]}" \
    2>&1 | tee -a "${LOG_FILE}"

else
    echo "[WORKER] I AM WORKER NODE. Waiting for HEAD_IP..."

    until [ -f "${HEAD_IP_FILE}" ]; do
        sleep 3
    done

    HEAD_IP=$(cat "${HEAD_IP_FILE}")
    echo "[WORKER] Found HEAD_IP = ${HEAD_IP}"

    ray stop || true

    ray start \
        --address="${HEAD_IP}:${RAY_PORT}" \
        --num-gpus="${NGPUS_PER_NODE}"

    echo "[WORKER] Joined Ray cluster. Keeping alive..."
    tail -f /dev/null
fi
