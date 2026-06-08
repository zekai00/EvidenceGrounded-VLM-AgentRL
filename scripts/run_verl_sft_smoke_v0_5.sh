#!/usr/bin/env bash
set -euo pipefail

# Reproducible configurable verl SFT smoke for EvidenceGrounded v0.5.
# Run from anywhere:
#   bash /root/Workspace/VLM/EvidenceGrounded-VLM-AgentRL/scripts/run_verl_sft_smoke_v0_5.sh

source /opt/conda/etc/profile.d/conda.sh
conda activate verl_test

cd /root/Workspace/VLM/EvidenceGrounded-VLM-AgentRL

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29582}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

DATA_DIR="${DATA_DIR:-/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_5_evidence_selection_sft_verl_maxpix131k_20260602_2143}"
OUT_DIR="${OUT_DIR:-/root/models/evidence_grounded_vlm_agentrl/verl_smoke_v0_5_20260602_2108}"
BASE_MODEL="${BASE_MODEL:-/root/models/Qwen2.5-VL-3B-Instruct}"
SFT_ADAPTER="${SFT_ADAPTER:-}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-32}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-8}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
SAVE_FREQ="${SAVE_FREQ:-${TOTAL_TRAINING_STEPS}}"
TEST_FREQ="${TEST_FREQ:-${TOTAL_TRAINING_STEPS}}"
LR="${LR:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-6144}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-6144}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-}"

rm -rf "${OUT_DIR}"

MODEL_ARGS=(
  model.path="${BASE_MODEL}"
  model.trust_remote_code=true
  +model.override_config.attn_implementation=eager
  model.lora_rank=8
  model.lora_alpha=16
  'model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'
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
)

if [[ -n "${IMAGE_MAX_PIXELS}" ]]; then
  DATA_ARGS+=(data.image_max_pixels="${IMAGE_MAX_PIXELS}")
fi

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
  trainer.project_name=evidence-grounded-v0-5-sft \
  trainer.experiment_name=verl-smoke-v0-5 \
  trainer.default_local_dir="${OUT_DIR}"
