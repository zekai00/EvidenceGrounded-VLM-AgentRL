# EvidenceGrounded-VLM-AgentRL

EvidenceGrounded-VLM-AgentRL is a VLM agentic RL project for evidence-grounded multimodal document understanding.

The current main task is to train and evaluate a vision-language tool-call agent that reads Chinese landscape-painting PDF literature, locates the target artwork image, retrieves/open evidence, and writes structured claims only when the evidence supports them.

## Current Status

Current version: `v1.0.4`

Current best configuration:

- Base model: `/root/models/Qwen2.5-VL-3B-Instruct`
- Best adapter: `/root/models/evidence_grounded_vlm_agentrl/qwen25vl3b_v1_0_2b_sft3000_from_phase8_20260608_0316/adapter`
- Runtime: v1.0.4 overlay verifier + `retrieve_evidence.scope` repair + no-select tool schema + phase-aware mask
- Evidence index: `/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v1_0_4_llm_overlay_20260611_0222`
- Trusted evaluation set: `/root/datasets/evidence_grounded_vlm_agentrl/gold_eval_v1_0_4_caption_corrected_20260611_1830`

This repository is no longer at the early next-action SFT stage. The current bottleneck is not crop quality. On GoldEval, target crop/region selection is already stable. The main bottleneck is field-level claim grounding:

- whether each non-abstain claim cites valid `evidence_ids`;
- whether cited evidence can actually support the claim field;
- whether the model abstains only when evidence is insufficient;
- whether final claims remain supported after retrieve/open/write tool use.

## Agent Workflow

The no-select runtime currently follows this high-level workflow:

```text
inspect_page
-> crop_target
-> open_evidence(local_caption)
-> retrieve_evidence, if more evidence is needed
-> open_evidence(retrieved evidence)
-> write_claims_chunk / abstain
-> finish
```

This is not a rigid requirement to call every tool exactly once. The key rules are:

- inspect and crop before writing claims;
- use local caption when it is sufficient;
- retrieve/open external evidence when non-caption fields need support;
- every non-abstain claim must include evidence IDs;
- if evidence is insufficient, abstain instead of guessing;
- finish only after all target fields are written or abstained.

Current GoldEval target fields are Core5:

```text
caption_text
image_scope
depicted_work_title
displayed_region
object_type
```

## Key Results

### Caption-Corrected GoldEval Baseline

Report:

`docs/03_实验报告/v1.0.4CaptionCorrectedGoldEvalVal50Test100评测与Guard负结果报告_20260611_2301.md`

Baseline val50:

| Metric | Value |
|---|---:|
| trajectory_success_rate | 0.960 |
| finish_rate | 0.980 |
| premature_finish_task_rate | 0.020 |
| crop_success_rate | 1.000 |
| mean_final_reward | 0.898754 |
| mean_claim_supported_rate | 0.633667 |
| mean_evidence_recall | 0.493334 |
| no_retrieve_task_rate | 0.140 |
| mean_negative_write_claim_count | 1.400 |

Baseline test100:

| Metric | Value |
|---|---:|
| trajectory_success_rate | 0.850 |
| finish_rate | 0.970 |
| premature_finish_task_rate | 0.030 |
| crop_success_rate | 1.000 |
| mean_final_reward | 0.894403 |
| mean_claim_supported_rate | 0.619000 |
| mean_evidence_recall | 0.578889 |
| no_retrieve_task_rate | 0.050 |
| mean_negative_write_claim_count | 1.910 |

### Behavior Repair SFT Negative Result

Report:

`docs/03_实验报告/v1.0.4BehaviorRepairSFT训练评测负结果报告_20260612_0159.md`

Two small LoRA continuation branches were trained from the current best adapter:

- A: patch:replay = 30:70, 20 optimizer steps.
- B: patch:replay = 40:60, 40 optimizer steps.

Both were evaluated only on `val_gold_50`; `test_gold_100` was not used for selection.

| Metric | Baseline | A replay70 20step | B replay60 40step |
|---|---:|---:|---:|
| trajectory_success_rate | 0.960 | 0.900 | 0.860 |
| finish_rate | 0.980 | 1.000 | 1.000 |
| premature_finish_task_rate | 0.020 | 0.000 | 0.000 |
| mean_final_reward | 0.898754 | 0.824033 | 0.819815 |
| mean_claim_supported_rate | 0.633667 | 0.351333 | 0.351333 |
| mean_evidence_recall | 0.493334 | 0.544445 | 0.486667 |
| mean_negative_write_claim_count | 1.400 | 2.180 | 2.140 |

Decision:

- Do not adopt A or B.
- Do not run test100 for A/B.
- Keep the v1.0.2b adapter as the current best.
- Do not continue scaling the same behavior-repair recipe.

The lesson is important: rule-like repair data can reduce premature finish and no-retrieve behavior, but it can also make the model worse at writing supported claims. The next patch must preserve positive claim-support behavior first, with hard negative/abstain samples kept small.

## Reproducing Main Evaluation

Baseline val50:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/collect_rollouts.py \
  --tasks /root/datasets/evidence_grounded_vlm_agentrl/gold_eval_v1_0_4_caption_corrected_20260611_1830/val_gold_50.jsonl \
  --evidence-index /root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v1_0_4_llm_overlay_20260611_0222 \
  --output-dir outputs/v1_0_4_gold_eval_caption_corrected_val50_bf16 \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --adapter /root/models/evidence_grounded_vlm_agentrl/qwen25vl3b_v1_0_2b_sft3000_from_phase8_20260608_0316/adapter \
  --max-steps 14 \
  --tool-schema no_select --phase-aware-mask --enforce-tool-mask \
  --strict-claim-phase-hint --dynamic-tool-schema \
  --target-claim-fields-from-gold \
  --no-load-in-4bit --torch-dtype bf16 \
  --image-max-pixels 262144 --max-seq-length 14336 --max-new-tokens 256 \
  --temperature 0.0 --no-print-steps
```

Build behavior repair data:

```bash
python scripts/build_v1_0_4_behavior_repair_sft.py \
  --dataset-suffix A_replay70 \
  --replay-ratio 0.70 \
  --max-patch-train 480 \
  --max-patch-val 128 \
  --seed 42
```

Train a continuation LoRA branch:

```bash
python scripts/train_trajectory_sft_lora.py \
  --train-jsonl /root/datasets/evidence_grounded_vlm_agentrl/agentbench_v1_0_4_behavior_repair_sft_A_replay70_20260612_0006/sft/train.jsonl \
  --val-jsonl /root/datasets/evidence_grounded_vlm_agentrl/agentbench_v1_0_4_behavior_repair_sft_A_replay70_20260612_0006/sft/val.jsonl \
  --output-dir /root/models/evidence_grounded_vlm_agentrl/qwen25vl3b_v1_0_4_behavior_repair_A_replay70_20step \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --adapter /root/models/evidence_grounded_vlm_agentrl/qwen25vl3b_v1_0_2b_sft3000_from_phase8_20260608_0316/adapter \
  --epochs 0.1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-5 \
  --prompt-mode compact \
  --load-in-4bit --torch-dtype bf16 \
  --training-record
```

Training automatically records:

- `train_log.jsonl`
- `gpu_memory_monitor.jsonl`
- `训练记录.md`
- `training_assets/loss_curve.png`
- `training_assets/gpu_memory_curve.png`

## Documentation

Important docs:

- `docs/codex-worklog.md`
- `docs/01_规划与路线/`
- `docs/02_指标与数据/`
- `docs/03_实验报告/`
- `docs/04_阶段性总结/`
- `docs/05_相关资料/`

Recent related-paper digest:

`docs/05_相关资料/v1.0.4相关论文与实现导读_20260611.md`

Note: `docs/` is treated as a local/internal report directory in `.gitignore`. The public GitHub repository tracks the runnable code and README; local reports and downloaded PDFs stay in the working machine unless explicitly force-added.

## Repository Layout

```text
configs/     Experiment configuration
docs/        Chinese reports, metrics notes, related papers, and worklog
scripts/     Data construction, training, and evaluation scripts
src/         Environment, tools, verifier, prompting, and agent modules
```

Large datasets, model weights, and rollout outputs are kept outside git unless explicitly documented. The local dataset/model roots used by this project are:

```text
/root/datasets/evidence_grounded_vlm_agentrl
/root/models/evidence_grounded_vlm_agentrl
```

## Next Work

Recommended next steps:

1. Split evidence metrics into `retrieved_recall`, `opened_recall`, and `cited_recall`.
2. Build a smaller behavior patch with 85%-90% replay and positive claim-support rows as the majority.
3. Patch remaining GoldEval caption-boundary candidates such as `001631`, `001602`, and `001887`.
4. Run a small page-image retrieval probe using ColPali/VisRAG-style retrieval before changing the main evidence index.
5. Consider GRPO only after `claim_supported_rate` and cited evidence do not regress on `val_gold_50`.
