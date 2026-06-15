#!/usr/bin/env bash
set -euo pipefail

# Smoke launcher for v1.3.1 trajectory-level GRPO-RLVR.
# This script prepares a tiny run configuration; it does not get invoked by the
# data builder. Run it manually after verifying DATA_DIR and SFT_ADAPTER.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_VLM_ROOT="$(cd "${DEFAULT_PROJECT_DIR}/.." && pwd)"
VLM_ROOT="${VLM_ROOT:-${DEFAULT_VLM_ROOT}}"
PROJECT_DIR="${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}"
if [[ -f "${VLM_ROOT}/activate_vlm_env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${VLM_ROOT}/activate_vlm_env.sh"
fi

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29631}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export EVIDENCE_STEPWISE_VERL_TMP="${EVIDENCE_STEPWISE_VERL_TMP:-${VLM_ROOT}/tmp/evidence_grounded_v1_3_1_rlvr}"
unset PYTORCH_CUDA_ALLOC_CONF

if [[ -d /root/models ]]; then
  DEFAULT_MODEL_ROOT="/root/models"
else
  DEFAULT_MODEL_ROOT="${VLM_ROOT}/models"
fi
if [[ -d /root/datasets ]]; then
  DEFAULT_DATASETS_ROOT="/root/datasets"
else
  DEFAULT_DATASETS_ROOT="${VLM_ROOT}/datasets"
fi
if [[ -d "${PROJECT_DIR}/outputs/v1_3_1_continued_from_v13best_sft_qwen25vl3b_full_save250_20260614_1652/adapter" ]]; then
  DEFAULT_OUTPUT_ROOT="${PROJECT_DIR}/outputs"
else
  DEFAULT_OUTPUT_ROOT="${VLM_ROOT}/outputs"
fi

MODEL_ROOT="${MODEL_ROOT:-${DEFAULT_MODEL_ROOT}}"
DATASETS_ROOT="${DATASETS_ROOT:-${DEFAULT_DATASETS_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
DATA_ROOT="${DATA_ROOT:-${DATASETS_ROOT}/evidence_grounded_vlm_agentrl/rlvr_v1_3_1_trajectory_level_latest}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}/verl}"
BASE_MODEL="${BASE_MODEL:-${MODEL_ROOT}/Qwen2.5-VL-3B-Instruct}"
SFT_ADAPTER="${SFT_ADAPTER:-${OUTPUT_ROOT}/v1_3_1_continued_from_v13best_sft_qwen25vl3b_full_save250_20260614_1652/adapter}"
OUT_DIR="${OUT_DIR:-${OUTPUT_ROOT}/v1_3_1_trajectory_grpo_smoke_$(date +%Y%m%d_%H%M)}"
AGENT_LOOP_CONFIG="${AGENT_LOOP_CONFIG:-${PROJECT_DIR}/configs/verl/evidence_stepwise_agent_loop_v1_3_1_rlvr.yaml}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
NNODES="${NNODES:-1}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-4}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-2}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
SAVE_FREQ="${SAVE_FREQ:-1}"
TEST_FREQ="${TEST_FREQ:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
ROLLOUT_N="${ROLLOUT_N:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-11264}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-11264}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-11264}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
LR="${LR:-2e-7}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.003}"
CLIP_RATIO="${CLIP_RATIO:-0.2}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:--1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.50}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-4}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-1}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-true}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-true}"
ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-bfloat16}"

if [[ ! -f "${DATA_DIR}/train.parquet" ]]; then
  echo "DATA_DIR missing train.parquet: ${DATA_DIR}" >&2
  exit 1
fi
if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "BASE_MODEL not found: ${BASE_MODEL}" >&2
  exit 1
fi
if [[ ! -d "${SFT_ADAPTER}" ]]; then
  echo "SFT_ADAPTER not found: ${SFT_ADAPTER}" >&2
  exit 1
fi
if [[ ! -f "${AGENT_LOOP_CONFIG}" ]]; then
  echo "AGENT_LOOP_CONFIG not found: ${AGENT_LOOP_CONFIG}" >&2
  exit 1
fi
if [[ -e "${OUT_DIR}" && "${OVERWRITE:-0}" != "1" ]]; then
  echo "OUT_DIR already exists: ${OUT_DIR}" >&2
  exit 1
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  rm -rf "${OUT_DIR}"
fi

python -m verl.trainer.main_ppo \
  actor_rollout_ref.model.path="${BASE_MODEL}" \
  actor_rollout_ref.model.trust_remote_code=true \
  +actor_rollout_ref.model.override_config.attn_implementation=eager \
  actor_rollout_ref.model.lora_rank=8 \
  actor_rollout_ref.model.lora_alpha=16 \
  actor_rollout_ref.model.lora_adapter_path="${SFT_ADAPTER}" \
  'actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.actor.optim.weight_decay=0.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.clip_ratio="${CLIP_RATIO}" \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.fsdp_config.model_dtype="${ACTOR_MODEL_DTYPE}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
  actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.enable_prefix_caching="${ENABLE_PREFIX_CACHING}" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb="${MM_PROCESSOR_CACHE_GB}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
  actor_rollout_ref.rollout.top_p="${TOP_P}" \
  actor_rollout_ref.rollout.top_k="${TOP_K}" \
  actor_rollout_ref.rollout.do_sample=true \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.load_format=auto \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=evidence_stepwise_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.ref.use_torch_compile=false \
  critic.enable=false \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=true \
  algorithm.use_kl_in_reward=false \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/val.parquet" \
  data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
  data.val_max_samples="${VAL_MAX_SAMPLES}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.truncation=right \
  data.filter_overlong_prompts=false \
  data.return_multi_modal_inputs=true \
  data.trust_remote_code=true \
  data.dataloader_num_workers=0 \
  reward.num_workers=0 \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.logger='[console]' \
  trainer.project_name=evidence-grounded-v1-3-1-rlvr \
  trainer.experiment_name=trajectory-grpo-smoke-v1-3-1 \
  trainer.val_before_train=false \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.resume_mode=disable \
  trainer.rollout_data_dir="${OUT_DIR}/rollout_data" \
  trainer.validation_data_dir="${OUT_DIR}/validation_data" \
  trainer.default_local_dir="${OUT_DIR}"
