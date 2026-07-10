#!/bin/bash
# Evaluate a Search-R1 model against:
#   1. an SGLang server you started separately, e.g.
#        MODEL_PATH=/path/to/hf TP_SIZE=2 PORT=30000 \
#            CUDA_VISIBLE_DEVICES=0,1 bash eval/search/sglang_serve.sh
#   2. a local retrieval server (Search-R1's retrieval_launch.sh) on $SEARCH_URL.
#
# Then in another shell:
#   MODEL_PATH=/path/to/hf PORT=30000 \
#       DATASET=/path/to/test.parquet \
#       SEARCH_URL=http://127.0.0.1:8000/retrieve,http://127.0.0.1:8001/retrieve \
#       bash eval/search/eval.sh

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

MODEL_PATH=${MODEL_PATH:?"set MODEL_PATH to an HF checkpoint directory"}
DATASET=${DATASET:?"set DATASET to the Search-R1 test parquet path"}

NUM_SAMPLES=${NUM_SAMPLES:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-8192}
MAX_TURNS=${MAX_TURNS:-4}
TOPK=${TOPK:-3}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
SEARCH_BACKEND=${SEARCH_BACKEND:-local}
SEARCH_URL=${SEARCH_URL:-http://127.0.0.1:8000/retrieve,http://127.0.0.1:8001/retrieve}

OUTPUT=${OUTPUT:-${MODEL_PATH}/eval_search_r1.jsonl}
SUMMARY_OUTPUT=${SUMMARY_OUTPUT:-${MODEL_PATH}/eval_search_r1_summary.json}
SAMPLE_TIMEOUT=${SAMPLE_TIMEOUT:-600}
SERVER_WAIT_TIMEOUT=${SERVER_WAIT_TIMEOUT:-60}
MAX_CONCURRENT=${MAX_CONCURRENT:-8}
LIMIT=${LIMIT:-}
DATA_SOURCE_FILTER=${DATA_SOURCE_FILTER:-}

export USER=${USER:-"user"}
export LOGNAME=${LOGNAME:-"$USER"}
export HOME=${HOME:-/tmp}

cd "${SCRIPT_DIR}"

EXTRA=()
if [ -n "${LIMIT}" ]; then
    EXTRA+=(--limit "${LIMIT}")
fi
if [ -n "${DATA_SOURCE_FILTER}" ]; then
    EXTRA+=(--data-source-filter "${DATA_SOURCE_FILTER}")
fi
if [ "${PRINT_TURNS:-0}" = "1" ]; then
    EXTRA+=(--print-turns)
fi
if [ "${DEBUG_TRACE:-0}" = "1" ]; then
    EXTRA+=(--debug-trace)
fi

python eval.py \
    --model-path "${MODEL_PATH}" \
    --dataset "${DATASET}" \
    --num-samples "${NUM_SAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-context-len "${MAX_CONTEXT_LEN}" \
    --max-turns "${MAX_TURNS}" \
    --topk "${TOPK}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --server-wait-timeout "${SERVER_WAIT_TIMEOUT}" \
    --max-concurrent "${MAX_CONCURRENT}" \
    --search-backend "${SEARCH_BACKEND}" \
    --search-url "${SEARCH_URL}" \
    --output "${OUTPUT}" \
    --summary-output "${SUMMARY_OUTPUT}" \
    --sample-timeout "${SAMPLE_TIMEOUT}" \
    "${EXTRA[@]}"
