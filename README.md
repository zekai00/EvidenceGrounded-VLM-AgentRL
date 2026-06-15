# EvidenceGrounded-VLM-AgentRL

EvidenceGrounded-VLM-AgentRL trains a vision-language tool-call agent for evidence-grounded multimodal document understanding.

The current task is Chinese landscape-painting literature: given a PDF page, the agent must locate the target artwork image, inspect local caption and page evidence, retrieve/open supporting fragments when needed, write structured fields only when evidence supports them, and abstain when evidence is insufficient.

The project is now centered on the `v1.3.1` SFT -> RLVR/GRPO route.

## Current Route

The current route has three stages:

1. **Remote-VLM data construction**
   - Use raw PDF pages with visible images.
   - Run page-level annotation with `qwen3.7-max-2026-06-08`.
   - Produce `FigureTarget`, `EvidenceFragment`, and `FieldSupportLabel`.

2. **SFT for the page/caption/evidence protocol**
   - Train Qwen2.5-VL-3B LoRA adapters to follow the action schema.
   - Compare fresh SFT from the base model with continued SFT from the previous best adapter.
   - Track loss and GPU memory curves for every training run.

3. **Trajectory-level RLVR / GRPO**
   - Build executable v1.3.1 tasks from SFT artifacts.
   - Roll out the agent in an environment with legal action sets and verifier rewards.
   - Use compact state updates so multi-step rollouts do not exceed response length.
   - First optimize trajectory-level reward before adding more complex state-level sampling.

## Main Data Objects

### FigureTarget

A single target figure on a PDF page. It contains the page image, source PDF, page number, target bbox, caption bbox, trajectory type, and gold field values.

### EvidenceFragment

A fragment that can be opened, retrieved, or cited by the agent. Common fragment types include:

- `local_caption_visual`
- `same_page_body`
- `visual`
- `wrong_target_caption`

### FieldSupportLabel

A `(target, field, fragment)` label indicating whether a fragment supports a field value:

- `support`
- `no_support`
- `ambiguous`
- `wrong_target`

These labels are used for SFT data generation, field-support evaluation, and verifier rewards.

## Current v1.3.1 Dataset

The current main SFT dataset is:

```text
/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v1_3_1_remote_vlm_evidence_sft_20260614_1335
```

Summary:

| Item | Count |
|---|---:|
| page records | 1950 |
| accepted targets | 1804 |
| evidence fragments | 6044 |
| field support labels | 16416 |
| SFT rows | 29993 |

Target split:

| split | targets |
|---|---:|
| train | 1352 |
| val | 181 |
| test | 271 |

Trajectory type distribution:

| type | train | val | test | total |
|---|---:|---:|---:|---:|
| `caption_only` | 402 | 54 | 81 | 537 |
| `retrieve_needed` | 184 | 25 | 37 | 246 |
| `abstain_needed` | 631 | 84 | 126 | 841 |
| `wrong_target_negative` | 135 | 18 | 27 | 180 |

The current RLVR dataset is generated from the same v1.3.1 artifacts:

```text
/root/datasets/evidence_grounded_vlm_agentrl/rlvr_v1_3_1_trajectory_level_latest
```

## Target Fields

The current field protocol is BaseLocate4 + Metadata5.

BaseLocate4:

- `caption_text`
- `depicted_work_title`
- `image_scope`
- `object_type`

Metadata5:

- `creator_or_attribution`
- `creation_period_or_dynasty`
- `collection_institution`
- `dimensions`
- `medium_material`

Caption, page vision, page body text, and retrieved fragments are treated as different evidence sources. Metadata fields are retrieved only when local caption evidence is insufficient.

## Action Schema

The current executable RLVR environment uses this compact action set:

- `inspect_page`: inspect page image, candidate regions, and visible evidence ids.
- `crop_target`: crop the target figure or a candidate target region.
- `retrieve_evidence`: search same-page, same-document, or corpus evidence.
- `open_evidence`: open a visible or retrieved evidence fragment.
- `write_claim`: write one supported field value with evidence ids.
- `abstain_claim`: abstain one field when evidence is insufficient.
- `write_claims_chunk`: write or abstain a small chunk of fields.
- `finish`: end only after required fields are written or abstained.

The model must output exactly one JSON action per step.

## Current RLVR/GRPO Setup

The v1.3.1 GRPO smoke path uses:

- Base model: Qwen2.5-VL-3B-Instruct
- Initial adapter: v1.3.1 B-continued SFT adapter
- Reward level: trajectory-level
- Actor strategy: FSDP1 across 4 GPUs
- Rollout: vLLM async, `tensor_model_parallel_size=1`
- Compact state update: enabled

Latest 4GPU 5-step smoke:

```text
outputs/v1_3_1_trajectory_grpo_smoke_compact_footerfix_repair_4gpu_5step_20260615_1940
```

Observed smoke result:

- 40 train rollouts
- prompt/response clip ratio: `0.0`
- actor allocated memory: about `10.55GB`
- actor reserved memory: about `10.83GB`
- checkpoint saved at `global_step_5/actor`

This smoke verifies that the 4GPU RLVR/GRPO path runs end to end. It is not yet a convergence result.

## Repository Layout

```text
configs/verl/                         Verl AgentLoop configs
scripts/build_v1_3_remote_vlm_evidence_sft.py
scripts/build_v1_3_1_rlvr_trajectory_level.py
scripts/run_verl_v1_3_1_trajectory_grpo_smoke.sh
scripts/eval_trajectory_sft_actions.py
scripts/train_trajectory_sft_lora.py
src/evidence_agent_env/               Executable agent environment
src/evidence_agent_env/verl_stepwise_agent_loop.py
```

Large datasets, model weights, outputs, and internal reports are intentionally kept outside the public repository or ignored by Git.

## Build RLVR Data

```bash
python scripts/build_v1_3_1_rlvr_trajectory_level.py \
  --dataset-dir /root/datasets/evidence_grounded_vlm_agentrl/agentbench_v1_3_1_remote_vlm_evidence_sft_20260614_1335 \
  --output-root /root/datasets/evidence_grounded_vlm_agentrl \
  --latest-link rlvr_v1_3_1_trajectory_level_latest
```

This writes:

```text
/root/datasets/evidence_grounded_vlm_agentrl/rlvr_v1_3_1_trajectory_level_YYYYMMDD_HHMM
/root/datasets/evidence_grounded_vlm_agentrl/rlvr_v1_3_1_trajectory_level_latest
```

## Run a 4GPU GRPO Smoke

Example for the current local machine:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
N_GPUS_PER_NODE=4 \
TRAIN_MAX_SAMPLES=10 \
VAL_MAX_SAMPLES=4 \
TOTAL_TRAINING_STEPS=5 \
SAVE_FREQ=5 \
TEST_FREQ=5 \
TRAIN_BATCH_SIZE=2 \
ROLLOUT_N=4 \
PPO_MINI_BATCH_SIZE=2 \
PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1 \
MAX_RESPONSE_LENGTH=4096 \
MAX_MODEL_LEN=12288 \
MAX_NUM_BATCHED_TOKENS=12288 \
MAX_NUM_SEQS=4 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.65 \
MM_PROCESSOR_CACHE_GB=6 \
AGENT_NUM_WORKERS=4 \
bash scripts/run_verl_v1_3_1_trajectory_grpo_smoke.sh
```

The launcher can also run on a mirrored remote layout by overriding:

```bash
VLM_ROOT=/root/lzk/vlm \
PROJECT_DIR=/root/lzk/vlm/code/EvidenceGrounded-VLM-AgentRL \
MODEL_ROOT=/root/lzk/vlm/models \
DATASETS_ROOT=/root/lzk/vlm/datasets \
OUTPUT_ROOT=/root/lzk/vlm/outputs \
bash scripts/run_verl_v1_3_1_trajectory_grpo_smoke.sh
```

## Metrics

The main evaluation metrics are:

- `trajectory_final_field_f1`: final field set correctness after the full rollout.
- `support_correct_rate`: written fields supported by cited evidence.
- `abstain_f1`: evidence-insufficient fields abstained correctly.
- `target_bbox_iou_ge_05`: target localization quality.
- `wrong_target_rate`: wrong-neighbor target or wrong evidence usage.
- `finish_rate`: whether the rollout reaches a valid finish.
- `invalid_json_rate` / `valid_action_rate`: executable action quality.
- `schema_repair_rate`: how often the environment had to patch minor argument slips.

Strict next-action metrics such as exact JSON action match are useful diagnostics, but they are not the primary measure of final agent quality.

## Safety and Git Hygiene

Do not commit:

- `.env` or API keys
- remote server passwords
- model weights or adapters
- raw PDFs or page images
- large `outputs/`
- private internal reports unless intentionally force-added

The public README describes the project route and code entry points; internal experiment reports live under local `docs/` and are ignored by default.
