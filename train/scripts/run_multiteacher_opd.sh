#!/bin/bash

set -ex

# Example usage:
#   export PROMPT_DATA=/root/data/retool_search_mixed.jsonl
#   export RETOOL_TEACHER_URL=http://127.0.0.1:13141/generate
#   export SEARCH_R1_TEACHER_URL=http://127.0.0.1:13142/generate
#   bash retool_and_search-r1/scripts/run_multiteacher_opd.sh
#
# Instead of external URLs, you may provide an --sglang-config via SGLANG_CONFIG
# with frozen models named retool_teacher and search_r1_teacher.

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELEASE_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${RELEASE_DIR}/.." &>/dev/null && pwd)"
SRC_DIR="${RELEASE_DIR}/src"
SLIME_DIR="${REPO_DIR}/third_party/slime"

MODEL_CONFIG=${MODEL_CONFIG:-"${SLIME_DIR}/scripts/models/qwen3-4B.sh"}
source "${MODEL_CONFIG}"

HF_CHECKPOINT=${HF_CHECKPOINT:-/root/Qwen3-4B}
REF_LOAD=${REF_LOAD:-/root/Qwen3-4B_torch_dist}
SAVE_DIR=${SAVE_DIR:-/root/Qwen3-4B_retool_search_opd}
PROMPT_DATA=${PROMPT_DATA:-}

# Data input modes:
# 1. Pre-combined file: set PROMPT_DATA. Each row must carry metadata.task.
# 2. Automatic mixed batches: set RETOOL_PROMPT_DATA and SEARCH_R1_PROMPT_DATA.
#    MIXED_TASK_ORDER controls the within-batch ratio; repeating a task changes
#    the ratio, e.g. retool,search-r1,search-r1 gives roughly 1:2.
RETOOL_PROMPT_DATA=${RETOOL_PROMPT_DATA:-}
SEARCH_R1_PROMPT_DATA=${SEARCH_R1_PROMPT_DATA:-}
RETOOL_INPUT_KEY=${RETOOL_INPUT_KEY:-prompt}
RETOOL_LABEL_KEY=${RETOOL_LABEL_KEY:-label}
RETOOL_METADATA_KEY=${RETOOL_METADATA_KEY:-metadata}
SEARCH_R1_INPUT_KEY=${SEARCH_R1_INPUT_KEY:-prompt}
SEARCH_R1_LABEL_KEY=${SEARCH_R1_LABEL_KEY:-reward_model}
SEARCH_R1_METADATA_KEY=${SEARCH_R1_METADATA_KEY:-metadata}
MIXED_TASK_ORDER=${MIXED_TASK_ORDER:-retool,search-r1}

if [[ -n "${RETOOL_PROMPT_DATA}" || -n "${SEARCH_R1_PROMPT_DATA}" ]]; then
   : "${RETOOL_PROMPT_DATA:?Set RETOOL_PROMPT_DATA or use PROMPT_DATA for a pre-combined file.}"
   : "${SEARCH_R1_PROMPT_DATA:?Set SEARCH_R1_PROMPT_DATA or use PROMPT_DATA for a pre-combined file.}"
   DATA_SOURCE_ARGS=(--data-source-path mixed_data_source.MixedTaskDataSource)
else
   PROMPT_DATA=${PROMPT_DATA:-/root/data/retool_search_mixed.jsonl}
   DATA_SOURCE_ARGS=(--prompt-data "${PROMPT_DATA}")
fi

ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-4}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-4}
RAY_NUM_GPUS=${RAY_NUM_GPUS:-8}

if [[ -n "${SGLANG_CONFIG:-}" ]]; then
   SGLANG_CONFIG_ARGS=(--sglang-config "${SGLANG_CONFIG}")
else
   SGLANG_CONFIG_ARGS=()
fi

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_DIR}"
   --save-interval 20
)

ROLLOUT_ARGS=(
   "${DATA_SOURCE_ARGS[@]}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 256
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
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
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.7}"
   "${SGLANG_CONFIG_ARGS[@]}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_multiteacher_opd.generate
   --custom-rm-path generate_with_multiteacher_opd.reward_func
   --custom-reward-post-process-path generate_with_multiteacher_opd.post_process_rewards
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group retool-search-r1-multiteacher-opd
   # --wandb-key ${WANDB_KEY}
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${RAY_NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SRC_DIR}:${RELEASE_DIR}:${REPO_DIR}:${SLIME_DIR}\",
    \"PYTHONUNBUFFERED\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"RETOOL_TEACHER_URL\": \"${RETOOL_TEACHER_URL:-}\",
    \"SEARCH_R1_TEACHER_URL\": \"${SEARCH_R1_TEACHER_URL:-}\",
    \"RETOOL_TEACHER_MODEL_NAME\": \"${RETOOL_TEACHER_MODEL_NAME:-retool_teacher}\",
    \"SEARCH_R1_TEACHER_MODEL_NAME\": \"${SEARCH_R1_TEACHER_MODEL_NAME:-search_r1_teacher}\",
    \"MULTITEACHER_OPD_TEACHER_URLS\": \"${MULTITEACHER_OPD_TEACHER_URLS:-}\",
    \"MULTITEACHER_OPD_USE_TASK_REWARD\": \"${MULTITEACHER_OPD_USE_TASK_REWARD:-1}\",
    \"RETOOL_PROMPT_DATA\": \"${RETOOL_PROMPT_DATA}\",
    \"RETOOL_INPUT_KEY\": \"${RETOOL_INPUT_KEY}\",
    \"RETOOL_LABEL_KEY\": \"${RETOOL_LABEL_KEY}\",
    \"RETOOL_METADATA_KEY\": \"${RETOOL_METADATA_KEY}\",
    \"SEARCH_R1_PROMPT_DATA\": \"${SEARCH_R1_PROMPT_DATA}\",
    \"SEARCH_R1_INPUT_KEY\": \"${SEARCH_R1_INPUT_KEY}\",
    \"SEARCH_R1_LABEL_KEY\": \"${SEARCH_R1_LABEL_KEY}\",
    \"SEARCH_R1_METADATA_KEY\": \"${SEARCH_R1_METADATA_KEY}\",
    \"MIXED_TASK_ORDER\": \"${MIXED_TASK_ORDER}\",
    \"SEARCH_URL\": \"${SEARCH_URL:-}\",
    \"SEARCH_R1_MAX_TURNS\": \"${SEARCH_R1_MAX_TURNS:-4}\"
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
