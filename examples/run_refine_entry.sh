#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-"false"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-"1"}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-"1"}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-"1"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

CANDIDATE_CONFIG=""
VERIFIER_CONFIG=""
INPUT_PATH=""
OUTPUT_PATH=""
ROUND_OUT_DIR=""
METRICS_OUT_DIR=""
EXP_NAME=""

VERIFIER_TYPE="forward"

CANDIDATE_MODEL_PATH=""
VERIFIER_MODEL_PATH=""

NUM_CANDIDATE_WORKERS=1
NUM_VERIFIER_WORKERS=1
BATCH_SIZE=16
MAX_REFINE_ROUNDS=3

CAND_TEMPERATURE=""
CAND_TOP_P=""
CAND_MAX_NEW=""
VER_TEMPERATURE=""
VER_TOP_P=""
VER_MAX_NEW=""

RAY_ADDRESS=""


THINKING_CANDIDATE=0
THINKING_VERIFIER=1


usage() {
  cat <<EOF
Usage: $0 \\
  --candidate-config path/to/candidate.yaml \\
  --verifier-config  path/to/verifier.yaml \\
  --input            path/to/input.jsonl \\
  --output           path/to/final_output.jsonl \\
  --round-output-dir   path/to/round_dir \\
  --metrics-output-dir path/to/metrics_dir \\
  --exp-name         exp_foo \\
  [--verifier-type forward|backward|vanilla] \\
  [--candidate-model-path path/to/cand/model] \\
  [--verifier-model-path  path/to/ver/model] \\
  [--num-candidate-workers N] \\
  [--num-verifier-workers N] \\
  [--batch-size N] \\
  [--max-refine-rounds N] \\
  [--cand-temperature T] [--cand-top-p P] [--cand-max-new-tokens N] \\
  [--ver-temperature T]  [--ver-top-p P]  [--ver-max-new-tokens N] \\
  [--thinking-candidate|--no-thinking-candidate] \\
  [--thinking-verifier|--no-thinking-verifier] \\
  [--ray-address addr]

EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --candidate-config)
      CANDIDATE_CONFIG="$2"; shift 2;;
    --verifier-config)
      VERIFIER_CONFIG="$2"; shift 2;;
    --input)
      INPUT_PATH="$2"; shift 2;;
    --output)
      OUTPUT_PATH="$2"; shift 2;;
    --round-output-dir)
      ROUND_OUT_DIR="$2"; shift 2;;
    --metrics-output-dir)
      METRICS_OUT_DIR="$2"; shift 2;;
    --exp-name)
      EXP_NAME="$2"; shift 2;;
    --verifier-type)
      VERIFIER_TYPE="$2"; shift 2;;
    --candidate-model-path)
      CANDIDATE_MODEL_PATH="$2"; shift 2;;
    --verifier-model-path)
      VERIFIER_MODEL_PATH="$2"; shift 2;;
    --num-candidate-workers)
      NUM_CANDIDATE_WORKERS="$2"; shift 2;;
    --num-verifier-workers)
      NUM_VERIFIER_WORKERS="$2"; shift 2;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2;;
    --max-refine-rounds)
      MAX_REFINE_ROUNDS="$2"; shift 2;;
    --cand-temperature)
      CAND_TEMPERATURE="$2"; shift 2;;
    --cand-top-p)
      CAND_TOP_P="$2"; shift 2;;
    --cand-max-new-tokens)
      CAND_MAX_NEW="$2"; shift 2;;
    --ver-temperature)
      VER_TEMPERATURE="$2"; shift 2;;
    --ver-top-p)
      VER_TOP_P="$2"; shift 2;;
    --ver-max-new-tokens)
      VER_MAX_NEW="$2"; shift 2;;
    --ray-address)
      RAY_ADDRESS="$2"; shift 2;;

    --thinking-candidate)
      THINKING_CANDIDATE=1; shift 1;;
    --no-thinking-candidate)
      THINKING_CANDIDATE=0; shift 1;;
    --thinking-verifier)
      THINKING_VERIFIER=1; shift 1;;
    --no-thinking-verifier)
      THINKING_VERIFIER=0; shift 1;;

    -h|--help)
      usage;;
    *)
      echo "Unknown arg: $1"; usage;;
  esac
done


if [[ -z "${CANDIDATE_CONFIG}" ]] || [[ -z "${VERIFIER_CONFIG}" ]] \
   || [[ -z "${INPUT_PATH}" ]] || [[ -z "${OUTPUT_PATH}" ]] \
   || [[ -z "${ROUND_OUT_DIR}" ]] || [[ -z "${METRICS_OUT_DIR}" ]] \
   || [[ -z "${EXP_NAME}" ]]; then
  echo "[ERROR] Missing required arguments."
  usage
fi


PY_ARGS=()
PY_ARGS+=(--candidate-config "${CANDIDATE_CONFIG}")
PY_ARGS+=(--verifier-config "${VERIFIER_CONFIG}")
PY_ARGS+=(--input "${INPUT_PATH}")
PY_ARGS+=(--output "${OUTPUT_PATH}")
PY_ARGS+=(--round-output-dir "${ROUND_OUT_DIR}")
PY_ARGS+=(--metrics-output-dir "${METRICS_OUT_DIR}")
PY_ARGS+=(--exp-name "${EXP_NAME}")

PY_ARGS+=(--verifier-type "${VERIFIER_TYPE}")

PY_ARGS+=(--num-candidate-workers "${NUM_CANDIDATE_WORKERS}")
PY_ARGS+=(--num-verifier-workers "${NUM_VERIFIER_WORKERS}")
PY_ARGS+=(--batch-size "${BATCH_SIZE}")
PY_ARGS+=(--max-refine-rounds "${MAX_REFINE_ROUNDS}")


if [[ "${THINKING_CANDIDATE}" == "1" ]]; then
  PY_ARGS+=(--enable-thinking-candidate)
fi

if [[ "${THINKING_VERIFIER}" == "1" ]]; then
  PY_ARGS+=(--enable-thinking-verifier)
fi

if [[ -n "${CANDIDATE_MODEL_PATH}" ]]; then
  PY_ARGS+=(--candidate-model-path "${CANDIDATE_MODEL_PATH}")
fi
if [[ -n "${VERIFIER_MODEL_PATH}" ]]; then
  PY_ARGS+=(--verifier-model-path "${VERIFIER_MODEL_PATH}")
fi

if [[ -n "${CAND_TEMPERATURE}" ]]; then
  PY_ARGS+=(--candidate-temperature "${CAND_TEMPERATURE}")
fi
if [[ -n "${CAND_TOP_P}" ]]; then
  PY_ARGS+=(--candidate-top-p "${CAND_TOP_P}")
fi
if [[ -n "${CAND_MAX_NEW}" ]]; then
  PY_ARGS+=(--candidate-max-new-tokens "${CAND_MAX_NEW}")
fi

if [[ -n "${VER_TEMPERATURE}" ]]; then
  PY_ARGS+=(--ver-temperature "${VER_TEMPERATURE}")
fi
if [[ -n "${VER_TOP_P}" ]]; then
  PY_ARGS+=(--ver-top-p "${VER_TOP_P}")
fi
if [[ -n "${VER_MAX_NEW}" ]]; then
  PY_ARGS+=(--ver-max-new-tokens "${VER_MAX_NEW}")
fi

if [[ -n "${RAY_ADDRESS}" ]]; then
  PY_ARGS+=(--ray-address "${RAY_ADDRESS}")
fi


echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] THINKING_CANDIDATE=${THINKING_CANDIDATE}, THINKING_VERIFIER=${THINKING_VERIFIER}"
echo "[INFO] Running refine_main.py with args:"
printf ' %q' "${PY_ARGS[@]}"
echo

python -m refine.main_refine "${PY_ARGS[@]}"
