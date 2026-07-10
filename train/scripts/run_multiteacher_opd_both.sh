#!/bin/bash

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

unset NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN

export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_cache}
export USER="${USER:-user}"
export LOGNAME="${LOGNAME:-$USER}"
export HOME="${HOME:-/tmp}"

export no_proxy="localhost,127.0.0.1,0.0.0.0,${no_proxy:-}"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,${NO_PROXY:-}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELEASE_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${RELEASE_DIR}/.." &>/dev/null && pwd)"
SRC_DIR="${RELEASE_DIR}/src"
SLIME_DIR="${REPO_DIR}/third_party/slime"
MEGATRON_ROOT="${MEGATRON_ROOT:-/opt/Megatron-LM}"

RETOOL_MODEL_DIR=${RETOOL_MODEL_DIR:-/root/experiments/00_single_env/retool}
SEARCH_R1_MODEL_DIR=${SEARCH_R1_MODEL_DIR:-/root/experiments/00_single_env/search_r1}
BOTH_MODEL_DIR=${BOTH_MODEL_DIR:-/root/experiments/00_single_env/both}

HF_CHECKPOINT=${HF_CHECKPOINT:-${BOTH_MODEL_DIR}/Qwen3-4B-Instruct-2507-SFT/iter_0000185/hf}
REF_LOAD=${REF_LOAD:-${BOTH_MODEL_DIR}/Qwen3-4B-Instruct-2507-SFT/iter_0000185/torch_dist}
SAVE_DIR=${SAVE_DIR:-${BOTH_MODEL_DIR}/Qwen3-4B-Instruct-2507-SFT-RL-to-Qwen3-4B-Instruct-2507-SFT-mixed-two-file-retool1searchr11}
ROLLOUT_DEBUG_DIR=${SAVE_DIR}/rollout_debug

RETOOL_PROMPT_DATA=${RETOOL_PROMPT_DATA:-${RETOOL_MODEL_DIR}/data/dapo-math-6.4k/dapo-math-6.4k.jsonl}
SEARCH_R1_PROMPT_DATA=${SEARCH_R1_PROMPT_DATA:-${SEARCH_R1_MODEL_DIR}/data/nq_hotpotqa_train/train_reformat_6.5k.parquet}

RETOOL_INPUT_KEY=${RETOOL_INPUT_KEY:-prompt}
RETOOL_LABEL_KEY=${RETOOL_LABEL_KEY:-label}
RETOOL_METADATA_KEY=${RETOOL_METADATA_KEY:-metadata}
SEARCH_R1_INPUT_KEY=${SEARCH_R1_INPUT_KEY:-prompt}
SEARCH_R1_LABEL_KEY=${SEARCH_R1_LABEL_KEY:-reward_model}
SEARCH_R1_METADATA_KEY=${SEARCH_R1_METADATA_KEY:-metadata}

# Repeating a task changes the within-batch ratio:
#   retool,search-r1             -> roughly 1:1
#   retool,search-r1,search-r1   -> roughly 1:2
MIXED_TASK_ORDER=${MIXED_TASK_ORDER:-retool,search-r1}

rm -rf "${SAVE_DIR}"
mkdir -p "${ROLLOUT_DEBUG_DIR}"

launch_teacher() {
   local name="$1"
   local model_path="$2"
   local gpus="$3"
   local port="$4"
   local tp="$5"
   local dp="$6"
   local mem_fraction="$7"
   local max_running_requests="$8"
   local schedule_conservativeness="$9"
   local max_total_tokens="${10}"
   local extra_args="${11:-}"
   local log_path="${SAVE_DIR}/sglang_${name}_teacher_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"

   # Only cap the KV-cache token budget when explicitly requested. Leaving
   # --max-total-tokens unset lets SGLang auto-size from --mem-fraction-static
   # (this matches retool/retool_opd_4b24b.sh). A small cap (e.g. 20000) starves
   # the teacher's KV cache: each teacher logprob request prefills the FULL
   # prompt+response (up to ~10k tokens) so only ~2 sequences fit at once,
   # which makes the 64 in-flight requests queue and time out -> retries.
   local max_total_tokens_arg=""
   if [ -n "${max_total_tokens}" ]; then
      max_total_tokens_arg="--max-total-tokens '${max_total_tokens}'"
   fi

   CUDA_VISIBLE_DEVICES=${gpus} \
      bash -c "cd /tmp && exec python3 -m sglang.launch_server \
      --model-path '${model_path}' \
      --host 0.0.0.0 \
      --port '${port}' \
      --tp '${tp}' \
      --dp '${dp}' \
      --chunked-prefill-size '${TEACHER_CHUNKED_PREFILL_SIZE:-2048}' \
      --mem-fraction-static '${mem_fraction}' \
      --max-running-requests '${max_running_requests}' \
      ${max_total_tokens_arg} \
      --schedule-conservativeness '${schedule_conservativeness}' \
      --disable-radix-cache \
      ${extra_args}" \
      > "${log_path}" 2>&1 &

   local pid=$!
   echo "${pid}:${log_path}"
}

TEACHER_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
TEACHER_IP="${TEACHER_IP:-127.0.0.1}"

START_RETOOL_TEACHER=${START_RETOOL_TEACHER:-1}
START_SEARCH_R1_TEACHER=${START_SEARCH_R1_TEACHER:-1}

RETOOL_TEACHER_MODEL_PATH=${RETOOL_TEACHER_MODEL_PATH:-${RETOOL_MODEL_DIR}/Qwen3-4B-Instruct-2507-SFT-RL/iter_0000199/hf}
RETOOL_TEACHER_GPUS=${RETOOL_TEACHER_GPUS:-4,5,6}
RETOOL_TEACHER_PORT=${RETOOL_TEACHER_PORT:-13141}
RETOOL_TEACHER_TP=${RETOOL_TEACHER_TP:-1}
RETOOL_TEACHER_DP=${RETOOL_TEACHER_DP:-3}

SEARCH_R1_TEACHER_MODEL_PATH=${SEARCH_R1_TEACHER_MODEL_PATH:-${SEARCH_R1_MODEL_DIR}/Qwen3-4B-Instruct-2507-SFT-RL/iter_0000199/hf}
SEARCH_R1_TEACHER_GPUS=${SEARCH_R1_TEACHER_GPUS:-7}
SEARCH_R1_TEACHER_PORT=${SEARCH_R1_TEACHER_PORT:-13142}
SEARCH_R1_TEACHER_TP=${SEARCH_R1_TEACHER_TP:-1}
SEARCH_R1_TEACHER_DP=${SEARCH_R1_TEACHER_DP:-1}

RETOOL_TEACHER_PID=""
SEARCH_R1_TEACHER_PID=""

cleanup_teachers() {
   if [ -n "${RETOOL_TEACHER_PID:-}" ]; then
      kill "${RETOOL_TEACHER_PID}" 2>/dev/null || true
   fi
   if [ -n "${SEARCH_R1_TEACHER_PID:-}" ]; then
      kill "${SEARCH_R1_TEACHER_PID}" 2>/dev/null || true
   fi
}
trap cleanup_teachers EXIT

if [ "${START_RETOOL_TEACHER}" = "1" ]; then
   RETOOL_LAUNCH_INFO=$(
      launch_teacher \
         retool \
         "${RETOOL_TEACHER_MODEL_PATH}" \
         "${RETOOL_TEACHER_GPUS}" \
         "${RETOOL_TEACHER_PORT}" \
         "${RETOOL_TEACHER_TP}" \
         "${RETOOL_TEACHER_DP}" \
         "${RETOOL_TEACHER_MEM_FRACTION_STATIC:-0.85}" \
         "${RETOOL_TEACHER_MAX_RUNNING_REQUESTS:-128}" \
         "${RETOOL_TEACHER_SCHEDULE_CONSERVATIVENESS:-0.3}" \
         "${RETOOL_TEACHER_MAX_TOTAL_TOKENS:-}" \
         "${RETOOL_TEACHER_EXTRA_ARGS:-}"
   )
   RETOOL_TEACHER_PID="${RETOOL_LAUNCH_INFO%%:*}"
   RETOOL_TEACHER_LOG="${RETOOL_LAUNCH_INFO#*:}"
   RETOOL_TEACHER_URL="http://${TEACHER_IP}:${RETOOL_TEACHER_PORT}/generate"
else
   RETOOL_TEACHER_URL=${RETOOL_TEACHER_URL:-http://127.0.0.1:13141/generate}
fi

if [ "${START_SEARCH_R1_TEACHER}" = "1" ]; then
   SEARCH_R1_LAUNCH_INFO=$(
      launch_teacher \
         search_r1 \
         "${SEARCH_R1_TEACHER_MODEL_PATH}" \
         "${SEARCH_R1_TEACHER_GPUS}" \
         "${SEARCH_R1_TEACHER_PORT}" \
         "${SEARCH_R1_TEACHER_TP}" \
         "${SEARCH_R1_TEACHER_DP}" \
         "${SEARCH_R1_TEACHER_MEM_FRACTION_STATIC:-0.85}" \
         "${SEARCH_R1_TEACHER_MAX_RUNNING_REQUESTS:-128}" \
         "${SEARCH_R1_TEACHER_SCHEDULE_CONSERVATIVENESS:-0.3}" \
         "${SEARCH_R1_TEACHER_MAX_TOTAL_TOKENS:-}" \
         "${SEARCH_R1_TEACHER_EXTRA_ARGS:-}"
   )
   SEARCH_R1_TEACHER_PID="${SEARCH_R1_LAUNCH_INFO%%:*}"
   SEARCH_R1_TEACHER_LOG="${SEARCH_R1_LAUNCH_INFO#*:}"
   SEARCH_R1_TEACHER_URL="http://${TEACHER_IP}:${SEARCH_R1_TEACHER_PORT}/generate"
else
   SEARCH_R1_TEACHER_URL=${SEARCH_R1_TEACHER_URL:-http://127.0.0.1:13142/generate}
fi

if [ "${START_RETOOL_TEACHER}" = "1" ]; then
   echo "Starting Retool teacher model server..."
   until curl -sf --noproxy '*' "http://${TEACHER_IP}:${RETOOL_TEACHER_PORT}/health_generate" > /dev/null; do
      echo "Waiting for Retool teacher server..."
      tail -n 10 "${RETOOL_TEACHER_LOG}"
      sleep 5
   done
   curl --noproxy '*' "http://${TEACHER_IP}:${RETOOL_TEACHER_PORT}/get_model_info"
   echo "Retool teacher server is up at ${RETOOL_TEACHER_URL}."
fi

if [ "${START_SEARCH_R1_TEACHER}" = "1" ]; then
   echo "Starting Search-R1 teacher model server..."
   until curl -sf --noproxy '*' "http://${TEACHER_IP}:${SEARCH_R1_TEACHER_PORT}/health_generate" > /dev/null; do
      echo "Waiting for Search-R1 teacher server..."
      tail -n 10 "${SEARCH_R1_TEACHER_LOG}"
      sleep 5
   done
   curl --noproxy '*' "http://${TEACHER_IP}:${SEARCH_R1_TEACHER_PORT}/get_model_info"
   echo "Search-R1 teacher server is up at ${SEARCH_R1_TEACHER_URL}."
fi

sleep 10

MODEL_CONFIG=${MODEL_CONFIG:-"${SLIME_DIR}/scripts/models/qwen3-4B.sh"}
source "${MODEL_CONFIG}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_DIR}"
   --load "${SAVE_DIR}"
   --save-interval "${SAVE_INTERVAL:-20}"
   --rotary-base "${ROTARY_BASE:-5000000}"
)

ROLLOUT_ARGS=(
   --data-source-path mixed_data_source.MixedTaskDataSource
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT:-200}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-256}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-1}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE:-1}"
   --global-batch-size "${GLOBAL_BATCH_SIZE:-256}"
   --balance-data
   --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"
   --use-dynamic-global-batch-size
   --rollout-stop "\\</tool_call>"
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE:-2}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16400}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef "${OPD_KL_COEF:-1.0}"
   --use-kl-loss
   --kl-loss-coef "${KL_LOSS_COEF:-0.0}"
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-6}"
   --lr-decay-style constant
   --weight-decay "${WEIGHT_DECAY:-0.1}"
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.5}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_multiteacher_opd.generate
   --custom-rm-path generate_with_multiteacher_opd.reward_func
   --custom-reward-post-process-path generate_with_multiteacher_opd.post_process_rewards
)

USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-agentic-opd}
WANDB_GROUP=${WANDB_GROUP:-Both_Qwen3-4B-Instruct-2507-SFT-RL-to-Qwen3-4B-Instruct-2507-SFT-mixed-two-file-retool1searchr11}
DISABLE_WANDB_RANDOM_SUFFIX=${DISABLE_WANDB_RANDOM_SUFFIX:-1}
export WANDB_DIR=${WANDB_DIR:-${SAVE_DIR}}

WANDB_ARGS=()
if [[ "${USE_WANDB}" == "1" ]]; then
   WANDB_ARGS+=(--use-wandb)
fi
if [[ -n "${WANDB_PROJECT}" ]]; then
   WANDB_ARGS+=(--wandb-project "${WANDB_PROJECT}")
fi
if [[ -n "${WANDB_GROUP}" ]]; then
   WANDB_ARGS+=(--wandb-group "${WANDB_GROUP}")
fi
if [[ -n "${WANDB_KEY:-}" ]]; then
   WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
fi
if [[ "${DISABLE_WANDB_RANDOM_SUFFIX}" == "1" ]]; then
   WANDB_ARGS+=(--disable-wandb-random-suffix)
fi

export CUDA_VISIBLE_DEVICES=${STUDENT_GPUS:-0,1,2,3}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export TOOL_SANDBOX_BACKEND=${TOOL_SANDBOX_BACKEND:-subprocess}
export TOOL_SANDBOX_CONCURRENCY=${TOOL_SANDBOX_CONCURRENCY:-64}
export TOOL_SANDBOX_MAX_TURNS=${TOOL_SANDBOX_MAX_TURNS:-16}
export TOOL_SANDBOX_MAX_TOOL_CALLS=${TOOL_SANDBOX_MAX_TOOL_CALLS:-16}
export TOOL_SANDBOX_JUPYTER_TIMEOUT=${TOOL_SANDBOX_JUPYTER_TIMEOUT:-300}
export SEARCH_URL=${SEARCH_URL:-"http://127.0.0.1:8000/retrieve"} #,http://127.0.0.1:8001/retrieve"}
export SEARCH_R1_MAX_TURNS=${SEARCH_R1_MAX_TURNS:-4}

# Set to 0 for pure OPD with task rewards logged only.
export MULTITEACHER_OPD_USE_TASK_REWARD=${MULTITEACHER_OPD_USE_TASK_REWARD:-0}

export MULTITEACHER_OPD_TEACHER_MAX_INFLIGHT=${MULTITEACHER_OPD_TEACHER_MAX_INFLIGHT:-64}
# Per-teacher in-flight caps. Both teachers run at once here, so a single shared
# cap lets heavy retool prefills (long math+code traces) flood the retool
# teacher (only 3 GPUs in this script) and overrun its KV cache, which queues
# requests past the sock_read timeout -> the "retry timeout from the retool
# teacher" you see. Bound each teacher to what its KV can hold concurrently.
export MULTITEACHER_OPD_RETOOL_MAX_INFLIGHT=${MULTITEACHER_OPD_RETOOL_MAX_INFLIGHT:-32}
export MULTITEACHER_OPD_SEARCH_R1_MAX_INFLIGHT=${MULTITEACHER_OPD_SEARCH_R1_MAX_INFLIGHT:-16}
export MULTITEACHER_OPD_TEACHER_REQUEST_TIMEOUT=${MULTITEACHER_OPD_TEACHER_REQUEST_TIMEOUT:-600}
export MULTITEACHER_OPD_TEACHER_CONNECT_TIMEOUT=${MULTITEACHER_OPD_TEACHER_CONNECT_TIMEOUT:-60}
export MULTITEACHER_OPD_TEACHER_SOCK_READ_TIMEOUT=${MULTITEACHER_OPD_TEACHER_SOCK_READ_TIMEOUT:-300}
export MULTITEACHER_OPD_TEACHER_TOTAL_BUDGET=${MULTITEACHER_OPD_TEACHER_TOTAL_BUDGET:-900}
export MULTITEACHER_OPD_TEACHER_MAX_RETRIES=${MULTITEACHER_OPD_TEACHER_MAX_RETRIES:-2}

RAY_NUM_GPUS=${RAY_NUM_GPUS:-4}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-4}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-4}

ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || true
ulimit -u 65536 2>/dev/null || true
echo "[opd] ulimit -n=$(ulimit -n)  ulimit -u=$(ulimit -u)"

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${RAY_NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_ROOT}:${SRC_DIR}:${RELEASE_DIR}:${REPO_DIR}:${SLIME_DIR}\",
    \"PYTHONUNBUFFERED\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"HF_HUB_OFFLINE\": \"1\",
    \"TRANSFORMERS_OFFLINE\": \"1\",
    \"TOOL_SANDBOX_BACKEND\": \"${TOOL_SANDBOX_BACKEND}\",
    \"TOOL_SANDBOX_CONCURRENCY\": \"${TOOL_SANDBOX_CONCURRENCY}\",
    \"TOOL_SANDBOX_MAX_TURNS\": \"${TOOL_SANDBOX_MAX_TURNS}\",
    \"TOOL_SANDBOX_MAX_TOOL_CALLS\": \"${TOOL_SANDBOX_MAX_TOOL_CALLS}\",
    \"TOOL_SANDBOX_JUPYTER_TIMEOUT\": \"${TOOL_SANDBOX_JUPYTER_TIMEOUT}\",
    \"SEARCH_URL\": \"${SEARCH_URL}\",
    \"SEARCH_R1_MAX_TURNS\": \"${SEARCH_R1_MAX_TURNS}\",
    \"RETOOL_TEACHER_URL\": \"${RETOOL_TEACHER_URL}\",
    \"SEARCH_R1_TEACHER_URL\": \"${SEARCH_R1_TEACHER_URL}\",
    \"RETOOL_PROMPT_DATA\": \"${RETOOL_PROMPT_DATA}\",
    \"RETOOL_INPUT_KEY\": \"${RETOOL_INPUT_KEY}\",
    \"RETOOL_LABEL_KEY\": \"${RETOOL_LABEL_KEY}\",
    \"RETOOL_METADATA_KEY\": \"${RETOOL_METADATA_KEY}\",
    \"SEARCH_R1_PROMPT_DATA\": \"${SEARCH_R1_PROMPT_DATA}\",
    \"SEARCH_R1_INPUT_KEY\": \"${SEARCH_R1_INPUT_KEY}\",
    \"SEARCH_R1_LABEL_KEY\": \"${SEARCH_R1_LABEL_KEY}\",
    \"SEARCH_R1_METADATA_KEY\": \"${SEARCH_R1_METADATA_KEY}\",
    \"MIXED_TASK_ORDER\": \"${MIXED_TASK_ORDER}\",
    \"MULTITEACHER_OPD_USE_TASK_REWARD\": \"${MULTITEACHER_OPD_USE_TASK_REWARD}\",
    \"TORCHINDUCTOR_CACHE_DIR\": \"${TORCHINDUCTOR_CACHE_DIR}\",
    \"WANDB_DIR\": \"${WANDB_DIR}\",
    \"USER\": \"${USER}\",
    \"LOGNAME\": \"${LOGNAME}\",
    \"HOME\": \"${HOME}\",
    \"no_proxy\": \"${no_proxy:-}\",
    \"NO_PROXY\": \"${NO_PROXY:-}\",
    \"MULTITEACHER_OPD_TEACHER_MAX_INFLIGHT\": \"${MULTITEACHER_OPD_TEACHER_MAX_INFLIGHT}\",
    \"MULTITEACHER_OPD_RETOOL_MAX_INFLIGHT\": \"${MULTITEACHER_OPD_RETOOL_MAX_INFLIGHT}\",
    \"MULTITEACHER_OPD_SEARCH_R1_MAX_INFLIGHT\": \"${MULTITEACHER_OPD_SEARCH_R1_MAX_INFLIGHT}\",
    \"MULTITEACHER_OPD_TEACHER_REQUEST_TIMEOUT\": \"${MULTITEACHER_OPD_TEACHER_REQUEST_TIMEOUT}\",
    \"MULTITEACHER_OPD_TEACHER_CONNECT_TIMEOUT\": \"${MULTITEACHER_OPD_TEACHER_CONNECT_TIMEOUT}\",
    \"MULTITEACHER_OPD_TEACHER_SOCK_READ_TIMEOUT\": \"${MULTITEACHER_OPD_TEACHER_SOCK_READ_TIMEOUT}\",
    \"MULTITEACHER_OPD_TEACHER_TOTAL_BUDGET\": \"${MULTITEACHER_OPD_TEACHER_TOTAL_BUDGET}\",
    \"MULTITEACHER_OPD_TEACHER_MAX_RETRIES\": \"${MULTITEACHER_OPD_TEACHER_MAX_RETRIES}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --working-dir="${SLIME_DIR}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}"
