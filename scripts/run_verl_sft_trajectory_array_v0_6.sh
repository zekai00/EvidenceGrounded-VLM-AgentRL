#!/usr/bin/env bash
set -euo pipefail

# verl SFT for v0.6 one-shot trajectory-array data.
# The default LoRA target is language attention only, avoiding visual-tower
# LoRA keys that vLLM rollout currently ignores during GRPO.

source /opt/conda/etc/profile.d/conda.sh
conda activate verl_test

cd /root/Workspace/VLM/EvidenceGrounded-VLM-AgentRL

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29583}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

DATA_DIR="${DATA_DIR:-/root/datasets/evidence_grounded_vlm_agentrl/trajectory_array_sft_v0_6_20260605_0350}"
OUT_DIR="${OUT_DIR:-/root/models/evidence_grounded_vlm_agentrl/verl_sft_v0_6_trajectory_array_langonly_$(date +%Y%m%d_%H%M)}"
BASE_MODEL="${BASE_MODEL:-/root/models/Qwen2.5-VL-3B-Instruct}"
SFT_ADAPTER="${SFT_ADAPTER:-}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-32}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
SAVE_FREQ="${SAVE_FREQ:-${TOTAL_TRAINING_STEPS}}"
TEST_FREQ="${TEST_FREQ:-${TOTAL_TRAINING_STEPS}}"
LR="${LR:-1e-5}"
MAX_LENGTH="${MAX_LENGTH:-6144}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-6144}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-131072}"
TARGET_MODULES="${TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj]}"

rm -rf "${OUT_DIR}"

MODEL_ARGS=(
  model.path="${BASE_MODEL}"
  model.trust_remote_code=true
  +model.override_config.attn_implementation=eager
  model.lora_rank=8
  model.lora_alpha=16
  "model.target_modules=${TARGET_MODULES}"
  model.enable_gradient_checkpointing=true
  model.use_remove_padding=false
)

if [[ -n "${SFT_ADAPTER}" ]]; then
  MODEL_ARGS+=(model.lora_adapter_path="${SFT_ADAPTER}")
fi

DATA_ARGS=(
  data.train_files="${DATA_DIR}/train.parquet"
  data.val_files="${DATA_DIR}/val.parquet"
  data.train_max_samples="${TRAIN_MAX_SAMPLES}"
  data.val_max_samples="${VAL_MAX_SAMPLES}"
  data.train_batch_size=1
  data.micro_batch_size_per_gpu=1
  data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}"
  data.use_dynamic_bsz=false
  data.max_length="${MAX_LENGTH}"
  data.truncation=right
  data.pad_mode=no_padding
  data.ignore_input_ids_mismatch=true
  data.num_workers=0
  +data.image_max_pixels="${IMAGE_MAX_PIXELS}"
)

python -m verl.trainer.sft_trainer \
  "${MODEL_ARGS[@]}" \
  engine.model_dtype=bf16 \
  engine.dtype=bfloat16 \
  engine.use_torch_compile=false \
  "${DATA_ARGS[@]}" \
  optim.lr="${LR}" \
  optim.weight_decay=0.0 \
  optim.lr_warmup_steps_ratio=0.0 \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.logger='[console]' \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.resume_mode=disable \
  trainer.project_name=evidence-grounded-v0-6-trajectory-array-sft \
  trainer.experiment_name=verl-sft-v0-6-trajectory-array-langonly \
  trainer.default_local_dir="${OUT_DIR}"
