# EvidenceGrounded-VLM-AgentRL

> 🇨🇳 面向中文山水画文献的证据约束多模态 Agent：先定位目标图像，再打开/检索证据，只在证据支持时写结构化字段，证据不足时显式 abstain。
>
> 🇬🇧 An evidence-grounded multimodal tool-call agent for Chinese landscape-painting literature: locate the target artwork, inspect/retrieve evidence, write supported structured claims, and explicitly abstain when evidence is insufficient.

## 📌 Latest Status / 最新进展

The current route is `v1.3.1 SFT -> RLVR/GRPO`. The latest full validation compares the base model, continued-B SFT LoRA, and Stage A5.3 GRPO checkpoint on the full `val181` trajectory-level split.

当前主线是 `v1.3.1 SFT -> RLVR/GRPO`。最新完整评测在 `val181` trajectory-level validation split 上对比了 base、continued-B SFT LoRA 和 Stage A5.3 GRPO checkpoint。

| Model / Run | Reward mean | Finish | Traj success | Claim F1 | Abstain F1 | Unsupported | Invalid steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen2.5-VL-3B-Instruct | 0.8101 | 0.8729 | 0.8177 | 0.7626 | 0.6911 | 0.0718 | 0.3591 |
| Continued-B SFT LoRA | 0.6721 | 0.8122 | 0.7956 | 0.9154 | 0.7101 | 0.1160 | 0.2320 |
| Stage A5.3 GRPO 160-step | **0.9650** | **0.9890** | **0.9779** | **0.9872** | **0.7158** | 0.1215 | **0.0276** |

Key takeaways:

- ✅ GRPO improves the main validation reward, finish rate, trajectory success, claim-support F1, and invalid-step rate.
- ⚠️ Unsupported claims remain a residual issue: the RL checkpoint is slightly worse than SFT/base on mean `unsupported_claim_count`.
- 📉 SFT alone is worse than the base model on full `val181`, which suggests imitation learned the schema but did not robustly learn the evidence/abstain policy.
- 📄 Full local report: `/root/EvidenceGrounded-VLM-AgentRL-Outputs/docs/03_实验报告/20260616_val181_base_sft_rl对照评测报告.md`

## 🧭 Project Goal / 项目目标

Given a PDF page from Chinese landscape-painting literature, the agent must:

1. Inspect the page and candidate regions.
2. Crop the correct target artwork region.
3. Open local caption / visual / page-body evidence.
4. Retrieve additional evidence only when needed.
5. Write field-level claims with explicit `evidence_ids`.
6. Use `abstain_claim` when no evidence supports a field.
7. Call `finish` only after all required fields are written or abstained.

项目核心不是“生成看起来合理的描述”，而是训练一个可执行、可验证的 VLM Agent：每个字段都必须能追溯到证据；没有证据的字段必须显式放弃。

## 🧱 System Design / 系统设计

### 1. Remote-VLM Data Construction / 远程 VLM 数据构造

Raw PDF pages are processed into executable supervision objects:

- `FigureTarget`: target artwork on a page, including page image, target bbox, caption bbox, source PDF, split, and gold fields.
- `EvidenceFragment`: local caption, visual crop, same-page body text, retrieved text, or wrong-target distractor evidence.
- `FieldSupportLabel`: `(target, field, fragment)` support label used by SFT construction, reward assignment, and evaluation.

远程 VLM 标注使用 page-level 文档图像、目标图裁剪、caption 区域和正文上下文构造字段支持关系。核心产物是 target、evidence、field-support 三类结构化对象。

### 2. SFT / 监督微调

SFT trains Qwen2.5-VL-3B LoRA adapters to follow the tool-call protocol:

- legal JSON action format;
- page/crop/open/retrieve/write/abstain/finish workflow;
- evidence-id citation discipline;
- basic abstain behavior.

SFT 的作用是让模型“会说环境能执行的话”。但完整 val181 结果显示，SFT 不足以稳定学会“何时不该写 claim”。

### 3. RLVR / GRPO

RLVR turns each task into an executable trajectory. The agent receives compact state updates and is rewarded by a verifier, not by reference-action exact match.

RLVR 把任务变成可执行交互轨迹，用 verifier reward 直接优化最终行为质量，而不是只模仿某条参考动作。

Stage A5.3 includes the current stabilizing constraints:

- `r_*` IDs are region IDs and can only be used by `crop_target`.
- `v13_t_*` IDs are evidence IDs and can be used by `open_evidence` / `write_claim`.
- repeated completed fields are not reopened in the claim phase.
- fields without available supporting evidence expose `abstain_claim` rather than `write_claim`.
- `finish` is exposed only after `missing` fields are empty.
- `retrieve_evidence.scope` is repaired/constrained to legal enum values.

These constraints reduce invalid exploration while keeping the core policy choice: which evidence to open/retrieve, which claim to write, and when to abstain.

## 🗂️ Current Dataset / 当前数据集

Main v1.3.1 SFT dataset:

```text
/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v1_3_1_remote_vlm_evidence_sft_20260614_1335
```

RLVR dataset:

```text
/root/datasets/evidence_grounded_vlm_agentrl/rlvr_v1_3_1_trajectory_level_latest
```

Summary:

| Item | Count |
|---|---:|
| page records | 1950 |
| accepted targets | 1804 |
| evidence fragments | 6044 |
| field support labels | 16416 |
| SFT rows | 29993 |

Split:

| split | targets |
|---|---:|
| train | 1352 |
| val | 181 |
| test | 271 |

Trajectory types:

| type | train | val | test | total |
|---|---:|---:|---:|---:|
| `caption_only` | 402 | 54 | 81 | 537 |
| `retrieve_needed` | 184 | 25 | 37 | 246 |
| `abstain_needed` | 631 | 84 | 126 | 841 |
| `wrong_target_negative` | 135 | 18 | 27 | 180 |

## 🧾 Target Fields / 字段协议

The current field protocol is **BaseLocate4 + Metadata5**.

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

Only evidence-supported fields should be written with `write_claim`. Missing or unsupported fields should be completed with `abstain_claim`.

## 🛠️ Action Schema / 动作空间

The executable RLVR environment uses a compact action set:

| Action | Purpose |
|---|---|
| `inspect_page` | inspect page image, regions, and visible evidence ids |
| `crop_target` | crop one legal `r_*` region |
| `open_evidence` | open one legal `v13_t_*` evidence fragment |
| `retrieve_evidence` | retrieve same-page, same-document, or corpus evidence |
| `write_claim` | write one supported field value with evidence ids |
| `abstain_claim` | explicitly abstain one unsupported field |
| `write_claims_chunk` | compact multi-field write/abstain helper where enabled |
| `finish` | terminate after all required fields are complete |

The model must output exactly one JSON action per step.

## 🏆 Reward / 评价信号

The verifier reward is trajectory-level and combines:

- target localization quality;
- evidence hit / open / citation quality;
- field support correctness;
- abstain correctness;
- invalid action penalties;
- premature finish penalties;
- final trajectory success.

Important metrics:

| Metric | Meaning |
|---|---|
| `reward/mean@1` | main validation score averaged over tasks |
| `finish_rate` | whether a valid final `finish` is reached |
| `trajectory_success` | verifier-level complete trajectory success |
| `claim_support_f1` | F1 for supported field claims and cited evidence |
| `abstain_f1` | F1 for correctly abstained unsupported fields |
| `unsupported_claim_count` | average unsupported written claims, lower is better |
| `invalid_steps` | average illegal / parse-failed steps, lower is better |
| `response_clip_ratio` | generation truncated by response length limit |

## 🚀 Runbooks / 运行方式

### Build RLVR Data

```bash
python scripts/build_v1_3_1_rlvr_trajectory_level.py \
  --dataset-dir /root/datasets/evidence_grounded_vlm_agentrl/agentbench_v1_3_1_remote_vlm_evidence_sft_20260614_1335 \
  --output-root /root/datasets/evidence_grounded_vlm_agentrl \
  --latest-link rlvr_v1_3_1_trajectory_level_latest
```

### Run 4GPU GRPO

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
MAX_RESPONSE_LENGTH=6144 \
MAX_MODEL_LEN=16384 \
MAX_NUM_BATCHED_TOKENS=16384 \
MAX_NUM_SEQS=1 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.50 \
MM_PROCESSOR_CACHE_GB=4 \
AGENT_NUM_WORKERS=4 \
bash scripts/run_verl_v1_3_1_trajectory_grpo_smoke.sh
```

### Run val181 Evaluation

```bash
MODEL_KIND=rl \
VAL_MAX_SAMPLES=181 \
VAL_BATCH_SIZE=16 \
TRAIN_MAX_SAMPLES=4 \
TRAIN_BATCH_SIZE=4 \
AGENT_NUM_WORKERS=4 \
N_GPUS_PER_NODE=4 \
MAX_RESPONSE_LENGTH=6144 \
MAX_MODEL_LEN=16384 \
OUT_DIR=/root/EvidenceGrounded-VLM-AgentRL-Outputs/outputs/val181_eval_rl \
EVIDENCE_STEPWISE_VERL_TMP=/root/Workspace/VLM/tmp/evidence_grounded_v1_3_1_val181_rl \
bash scripts/run_verl_v1_3_1_trajectory_val_eval.sh
```

`MODEL_KIND` can be `base`, `sft`, or `rl`.

### Analyze val181 Runs

```bash
/opt/conda/envs/verl_test/bin/python scripts/analyze_val181_eval_runs.py
```

This writes metrics, charts, sample images, and a Markdown report under:

```text
/root/EvidenceGrounded-VLM-AgentRL-Outputs/docs/03_实验报告
```

## 📁 Repository Layout / 仓库结构

```text
configs/verl/                         verl AgentLoop configs
scripts/build_v1_3_remote_vlm_evidence_sft.py
scripts/build_v1_3_1_rlvr_trajectory_level.py
scripts/run_verl_v1_3_1_trajectory_grpo_smoke.sh
scripts/run_verl_v1_3_1_trajectory_val_eval.sh
scripts/analyze_val181_eval_runs.py
scripts/eval_trajectory_sft_actions.py
scripts/train_trajectory_sft_lora.py
src/evidence_agent_env/               executable agent environment
src/evidence_agent_env/verl_stepwise_agent_loop.py
```

Large datasets, model weights, outputs, and internal reports are intentionally kept outside the public repository or ignored by Git.

## 🔒 Safety and Git Hygiene / 安全与提交规范

Do not commit:

- `.env` or API keys;
- server passwords or private credentials;
- model weights, LoRA adapters, or checkpoints;
- raw PDFs, page images, and large generated outputs;
- private internal reports unless intentionally approved.

Public code and README describe the method and reproducible entry points. Large experiment outputs live under:

```text
/root/EvidenceGrounded-VLM-AgentRL-Outputs
```
