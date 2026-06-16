<h1 align="center">EvidenceGrounded-VLM-AgentRL</h1>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-bfloat16-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black">
  <img alt="Qwen" src="https://img.shields.io/badge/Qwen2.5--VL-3B-6B5BFF">
  <img alt="vLLM" src="https://img.shields.io/badge/vLLM-async%20rollout-00A3E0">
  <img alt="Ray" src="https://img.shields.io/badge/Ray-distributed-028CF0?logo=ray&logoColor=white">
  <img alt="verl" src="https://img.shields.io/badge/verl-GRPO-111827">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-SFT%20%2B%20RLVR-2E7D32">
  <img alt="Status" src="https://img.shields.io/badge/status-Stage%20A5.3-green">
</p>

<p align="center">
  <a href="#zh">中文</a> · <a href="#en">English</a>
</p>

<a id="zh"></a>

<details open>
<summary><strong>中文说明（默认展开）</strong></summary>

## 项目概览

`EvidenceGrounded-VLM-AgentRL` 训练一个面向中文山水画文献的多模态工具调用 Agent。给定 PDF 页面后，模型需要先定位目标图像，再打开或检索证据，只在证据支持时写结构化字段；证据不足时必须显式调用 `abstain_claim`，而不是编造字段。

项目当前主线是：

```text
v1.3.1 数据构造 -> SFT -> RLVR / GRPO
```

核心目标不是生成自然语言描述，而是训练一个可执行、可验证、可追溯证据的 VLM Agent。

## 最新进展

当前最重要的完整评测是 `val181` trajectory-level validation，对比了 base、continued-B SFT LoRA 和 Stage A5.3 GRPO checkpoint。

| 方案 | 奖励均值 | Finish率 | 轨迹成功率 | Claim F1 | Abstain F1 | Unsupported均值 | 非法步数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen2.5-VL-3B-Instruct | 0.8101 | 0.8729 | 0.8177 | 0.7626 | 0.6911 | 0.0718 | 0.3591 |
| Continued-B SFT LoRA | 0.6721 | 0.8122 | 0.7956 | 0.9154 | 0.7101 | 0.1160 | 0.2320 |
| Stage A5.3 GRPO 160-step | **0.9650** | **0.9890** | **0.9779** | **0.9872** | **0.7158** | 0.1215 | **0.0276** |

结论：

- GRPO 明显提升主奖励、Finish率、轨迹成功率、Claim F1，并显著降低非法步数。
- SFT 单独在完整 `val181` 上低于 base，说明模仿学习学到了动作格式，但没有稳定学会何时 abstain、何时避免 unsupported claim。
- `unsupported_claim_count` 仍是后续优化重点；RL 在该指标上略高于 SFT/base。
- 完整实验报告、图表和样例图片应放在仓库外部的 `OUTPUT_ROOT/docs/` 下，避免把大文件或本地绝对路径提交进仓库。

## 任务定义

给定一个 PDF 页面，Agent 需要完成：

1. `inspect_page`：查看页面、候选区域和可见 evidence id。
2. `crop_target`：裁剪正确的目标图像区域。
3. `open_evidence`：打开本地 caption、视觉证据或正文 evidence。
4. `retrieve_evidence`：在需要时检索同页、同文档或语料库证据。
5. `write_claim`：只对证据支持的字段写值，并给出 `evidence_ids`。
6. `abstain_claim`：证据不足时显式放弃字段。
7. `finish`：所有必需字段写完或 abstain 后才能结束。

## 数据对象

### FigureTarget

单个 PDF 页面上的目标图像，包含页面图、目标框、caption 框、来源 PDF、split、目标字段和 gold 字段。

### EvidenceFragment

可被打开、检索或引用的证据片段。常见类型包括：

- `local_caption_visual`
- `local_visual`
- `same_page_body`
- `retrieved_text`
- `wrong_target_caption`

### FieldSupportLabel

`(target, field, fragment)` 级别的支持关系标签，用于 SFT 构造、reward 分配和评测。

支持关系包括：

- `support`
- `no_support`
- `ambiguous`
- `wrong_target`

## 当前数据集

当前 v1.3.1 数据集统计：

| 项目 | 数量 |
|---|---:|
| page records | 1950 |
| accepted targets | 1804 |
| evidence fragments | 6044 |
| field support labels | 16416 |
| SFT rows | 29993 |

数据切分：

| split | targets |
|---|---:|
| train | 1352 |
| val | 181 |
| test | 271 |

轨迹类型：

| type | train | val | test | total |
|---|---:|---:|---:|---:|
| `caption_only` | 402 | 54 | 81 | 537 |
| `retrieve_needed` | 184 | 25 | 37 | 246 |
| `abstain_needed` | 631 | 84 | 126 | 841 |
| `wrong_target_negative` | 135 | 18 | 27 | 180 |

建议通过环境变量指定本地路径：

```bash
export MODEL_ROOT=/path/to/models
export DATASETS_ROOT=/path/to/datasets
export OUTPUT_ROOT=/path/to/outputs
```

不要在 README 或公开代码里写机器相关的绝对路径。

## 字段协议

当前字段协议是 **BaseLocate4 + Metadata5**。

BaseLocate4：

- `caption_text`
- `depicted_work_title`
- `image_scope`
- `object_type`

Metadata5：

- `creator_or_attribution`
- `creation_period_or_dynasty`
- `collection_institution`
- `dimensions`
- `medium_material`

原则：只有证据支持时才 `write_claim`；证据不足必须 `abstain_claim`。

## 动作空间

| Action | 含义 |
|---|---|
| `inspect_page` | 查看页面、候选区域和可见 evidence id |
| `crop_target` | 裁剪一个合法 `r_*` region |
| `open_evidence` | 打开一个合法 `v13_t_*` evidence fragment |
| `retrieve_evidence` | 检索同页、同文档或语料库证据 |
| `write_claim` | 写一个有证据支撑的字段 |
| `abstain_claim` | 对无证据字段显式 abstain |
| `write_claims_chunk` | 小块字段写入或 abstain 辅助动作 |
| `finish` | 所有字段完成后结束 |

模型每一步只能输出一个 JSON action。

## Stage A5.3 约束

当前 GRPO 稳定性修复包括：

- `r_*` 只作为 region id，只能用于 `crop_target`。
- `v13_t_*` 只作为 evidence id，可用于 `open_evidence` 和 `write_claim`。
- claim 阶段不再开放已完成字段，减少重复写字段。
- 当前字段没有可用 supporting evidence 时，只开放 `abstain_claim`。
- `missing` 字段为空后才开放 `finish`。
- `retrieve_evidence.scope` 约束或修复为合法枚举。

这些约束不是替模型直接写答案，而是减少非法 action space；模型仍然要决定看什么证据、写哪些 claim、何时 abstain。

## Reward 与关键指标

trajectory-level verifier reward 综合考虑：

- 目标定位质量。
- evidence 命中、打开和引用质量。
- 字段 claim 是否被证据支持。
- 无证据字段是否正确 abstain。
- 非法 action 惩罚。
- 过早 finish 惩罚。
- 最终轨迹是否成功。

常用指标：

| 指标 | 含义 |
|---|---|
| `reward/mean@1` | validation 主分数均值 |
| `finish_rate` | 是否合法到达 `finish` |
| `trajectory_success` | verifier 侧完整轨迹成功率 |
| `claim_support_f1` | 支持性 claim 和 evidence citation 的 F1 |
| `abstain_f1` | 无证据字段 abstain 的 F1 |
| `unsupported_claim_count` | unsupported claim 均值，越低越好 |
| `invalid_steps` | 非法或解析失败步数，越低越好 |
| `response_clip_ratio` | 生成是否被 response length 截断 |

## 运行方式

### 构建 RLVR 数据

```bash
python scripts/build_v1_3_1_rlvr_trajectory_level.py \
  --dataset-dir "${DATASETS_ROOT}/evidence_grounded_vlm_agentrl/agentbench_v1_3_1_remote_vlm_evidence_sft" \
  --output-root "${DATASETS_ROOT}/evidence_grounded_vlm_agentrl" \
  --latest-link rlvr_v1_3_1_trajectory_level_latest
```

### 运行 4GPU GRPO smoke

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_ROOT="${MODEL_ROOT}" \
DATASETS_ROOT="${DATASETS_ROOT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
N_GPUS_PER_NODE=4 \
TRAIN_MAX_SAMPLES=10 \
VAL_MAX_SAMPLES=4 \
TOTAL_TRAINING_STEPS=5 \
SAVE_FREQ=5 \
TEST_FREQ=5 \
TRAIN_BATCH_SIZE=2 \
ROLLOUT_N=4 \
MAX_RESPONSE_LENGTH=6144 \
MAX_MODEL_LEN=16384 \
AGENT_NUM_WORKERS=4 \
bash scripts/run_verl_v1_3_1_trajectory_grpo_smoke.sh
```

### 运行 val181 评测

```bash
MODEL_KIND=rl \
MODEL_ROOT="${MODEL_ROOT}" \
DATASETS_ROOT="${DATASETS_ROOT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
VAL_MAX_SAMPLES=181 \
VAL_BATCH_SIZE=16 \
TRAIN_MAX_SAMPLES=4 \
TRAIN_BATCH_SIZE=4 \
AGENT_NUM_WORKERS=4 \
N_GPUS_PER_NODE=4 \
MAX_RESPONSE_LENGTH=6144 \
MAX_MODEL_LEN=16384 \
OUT_DIR="${OUTPUT_ROOT}/val181_eval_rl" \
EVIDENCE_STEPWISE_VERL_TMP="${OUTPUT_ROOT}/tmp/val181_eval_rl" \
bash scripts/run_verl_v1_3_1_trajectory_val_eval.sh
```

`MODEL_KIND` 可取 `base`、`sft` 或 `rl`。

### 分析 val181 结果

```bash
python scripts/analyze_val181_eval_runs.py \
  --report-dir "${OUTPUT_ROOT}/docs/03_实验报告"
```

## 仓库结构

```text
configs/verl/                         verl AgentLoop 配置
scripts/build_v1_3_remote_vlm_evidence_sft.py
scripts/build_v1_3_1_rlvr_trajectory_level.py
scripts/run_verl_v1_3_1_trajectory_grpo_smoke.sh
scripts/run_verl_v1_3_1_trajectory_val_eval.sh
scripts/analyze_val181_eval_runs.py
scripts/analyze_verl_grpo_run.py
src/evidence_agent_env/               可执行 Agent 环境
src/evidence_agent_env/verl_stepwise_agent_loop.py
```

## 提交规范

不要提交：

- `.env`、API key 或服务器密码。
- 模型权重、LoRA adapter 或 checkpoint。
- 原始 PDF、页面图、裁剪图和大规模 outputs。
- 机器相关的绝对路径。
- 未确认可公开的内部实验报告。

</details>

<a id="en"></a>

<details>
<summary><strong>English</strong></summary>

## Overview

`EvidenceGrounded-VLM-AgentRL` trains a multimodal tool-call agent for Chinese landscape-painting literature. Given a PDF page, the agent must locate the target artwork, inspect or retrieve evidence, write structured fields only when supported, and explicitly abstain when evidence is insufficient.

The current route is:

```text
v1.3.1 data construction -> SFT -> RLVR / GRPO
```

The goal is not free-form captioning. The goal is an executable and verifiable VLM agent whose field-level claims are grounded in evidence.

## Latest Status

The latest full evaluation compares the base model, continued-B SFT LoRA, and Stage A5.3 GRPO checkpoint on the `val181` trajectory-level validation split.

| Run | Reward mean | Finish | Traj success | Claim F1 | Abstain F1 | Unsupported | Invalid steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen2.5-VL-3B-Instruct | 0.8101 | 0.8729 | 0.8177 | 0.7626 | 0.6911 | 0.0718 | 0.3591 |
| Continued-B SFT LoRA | 0.6721 | 0.8122 | 0.7956 | 0.9154 | 0.7101 | 0.1160 | 0.2320 |
| Stage A5.3 GRPO 160-step | **0.9650** | **0.9890** | **0.9779** | **0.9872** | **0.7158** | 0.1215 | **0.0276** |

Takeaways:

- GRPO improves the main reward, finish rate, trajectory success, claim-support F1, and invalid-step rate.
- SFT alone underperforms the base model on full `val181`, which suggests imitation learns the action format but not the abstain/evidence policy robustly enough.
- Unsupported claims remain a residual issue and should be optimized next.
- Full experiment reports, figures, and sampled images should live outside the repository under `OUTPUT_ROOT/docs/`.

## Task

For each PDF page, the agent should:

1. Inspect the page and candidate regions.
2. Crop the correct target region.
3. Open local caption, visual, or page-body evidence.
4. Retrieve additional evidence only when necessary.
5. Write field-level claims with explicit `evidence_ids`.
6. Use `abstain_claim` for unsupported fields.
7. Call `finish` only after all required fields are written or abstained.

## Data Objects

### FigureTarget

A target artwork on a PDF page, including page image, target bbox, caption bbox, source PDF, split, target fields, and gold fields.

### EvidenceFragment

An evidence fragment that can be opened, retrieved, or cited. Common types include:

- `local_caption_visual`
- `local_visual`
- `same_page_body`
- `retrieved_text`
- `wrong_target_caption`

### FieldSupportLabel

A `(target, field, fragment)` support label used for SFT construction, reward assignment, and evaluation.

## Dataset

Current v1.3.1 dataset summary:

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

Use environment variables instead of machine-specific absolute paths:

```bash
export MODEL_ROOT=/path/to/models
export DATASETS_ROOT=/path/to/datasets
export OUTPUT_ROOT=/path/to/outputs
```

## Field Protocol

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

Only evidence-supported fields should be written with `write_claim`; unsupported fields should be completed with `abstain_claim`.

## Action Schema

| Action | Purpose |
|---|---|
| `inspect_page` | inspect page, candidate regions, and visible evidence ids |
| `crop_target` | crop one legal `r_*` region |
| `open_evidence` | open one legal `v13_t_*` evidence fragment |
| `retrieve_evidence` | retrieve same-page, same-document, or corpus evidence |
| `write_claim` | write one supported field value |
| `abstain_claim` | explicitly abstain one unsupported field |
| `write_claims_chunk` | compact multi-field write/abstain helper |
| `finish` | terminate after all fields are complete |

The model must output exactly one JSON action per step.

## Stage A5.3 Stabilization

The current GRPO path includes:

- `r_*` ids are region ids and can only be used by `crop_target`.
- `v13_t_*` ids are evidence ids and can be used by `open_evidence` / `write_claim`.
- completed fields are not reopened in the claim phase.
- unsupported fields expose `abstain_claim` rather than `write_claim`.
- `finish` is exposed only after `missing` fields are empty.
- `retrieve_evidence.scope` is constrained or repaired to a legal enum.

These constraints reduce invalid exploration while preserving policy choices about evidence use, claim writing, and abstention.

## Reward and Metrics

The trajectory-level verifier reward combines:

- target localization;
- evidence hit/open/citation quality;
- field support correctness;
- abstain correctness;
- invalid action penalties;
- premature finish penalties;
- final trajectory success.

Important metrics:

| Metric | Meaning |
|---|---|
| `reward/mean@1` | main validation score |
| `finish_rate` | valid final `finish` rate |
| `trajectory_success` | verifier-level trajectory success |
| `claim_support_f1` | supported field claim and citation F1 |
| `abstain_f1` | unsupported-field abstention F1 |
| `unsupported_claim_count` | average unsupported claims, lower is better |
| `invalid_steps` | illegal or parse-failed steps, lower is better |
| `response_clip_ratio` | generation truncation rate |

## Runbooks

### Build RLVR Data

```bash
python scripts/build_v1_3_1_rlvr_trajectory_level.py \
  --dataset-dir "${DATASETS_ROOT}/evidence_grounded_vlm_agentrl/agentbench_v1_3_1_remote_vlm_evidence_sft" \
  --output-root "${DATASETS_ROOT}/evidence_grounded_vlm_agentrl" \
  --latest-link rlvr_v1_3_1_trajectory_level_latest
```

### Run 4-GPU GRPO Smoke

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_ROOT="${MODEL_ROOT}" \
DATASETS_ROOT="${DATASETS_ROOT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
N_GPUS_PER_NODE=4 \
TRAIN_MAX_SAMPLES=10 \
VAL_MAX_SAMPLES=4 \
TOTAL_TRAINING_STEPS=5 \
SAVE_FREQ=5 \
TEST_FREQ=5 \
TRAIN_BATCH_SIZE=2 \
ROLLOUT_N=4 \
MAX_RESPONSE_LENGTH=6144 \
MAX_MODEL_LEN=16384 \
AGENT_NUM_WORKERS=4 \
bash scripts/run_verl_v1_3_1_trajectory_grpo_smoke.sh
```

### Run val181 Evaluation

```bash
MODEL_KIND=rl \
MODEL_ROOT="${MODEL_ROOT}" \
DATASETS_ROOT="${DATASETS_ROOT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
VAL_MAX_SAMPLES=181 \
VAL_BATCH_SIZE=16 \
TRAIN_MAX_SAMPLES=4 \
TRAIN_BATCH_SIZE=4 \
AGENT_NUM_WORKERS=4 \
N_GPUS_PER_NODE=4 \
MAX_RESPONSE_LENGTH=6144 \
MAX_MODEL_LEN=16384 \
OUT_DIR="${OUTPUT_ROOT}/val181_eval_rl" \
EVIDENCE_STEPWISE_VERL_TMP="${OUTPUT_ROOT}/tmp/val181_eval_rl" \
bash scripts/run_verl_v1_3_1_trajectory_val_eval.sh
```

`MODEL_KIND` can be `base`, `sft`, or `rl`.

### Analyze val181 Runs

```bash
python scripts/analyze_val181_eval_runs.py \
  --report-dir "${OUTPUT_ROOT}/docs/03_实验报告"
```

## Repository Layout

```text
configs/verl/                         verl AgentLoop configs
scripts/build_v1_3_remote_vlm_evidence_sft.py
scripts/build_v1_3_1_rlvr_trajectory_level.py
scripts/run_verl_v1_3_1_trajectory_grpo_smoke.sh
scripts/run_verl_v1_3_1_trajectory_val_eval.sh
scripts/analyze_val181_eval_runs.py
scripts/analyze_verl_grpo_run.py
src/evidence_agent_env/               executable agent environment
src/evidence_agent_env/verl_stepwise_agent_loop.py
```

## Git Hygiene

Do not commit:

- `.env`, API keys, or server passwords.
- model weights, LoRA adapters, or checkpoints.
- raw PDFs, page images, crops, or large generated outputs.
- machine-specific absolute paths.
- private internal reports unless explicitly approved.

</details>
