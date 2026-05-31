# EvidenceGrounded-VLM-AgentRL

<p align="center">
  <a href="#中文"><kbd>中文</kbd></a>
  <a href="#english"><kbd>English</kbd></a>
</p>

## 中文

EvidenceGrounded-VLM-AgentRL 是一个面向多模态文档理解的 VLM 工具调用强化学习项目。

项目目标是训练一个视觉语言模型，让它在阅读复杂 PDF 页面、图像区域和文本证据时，不是一次性给出答案，而是通过多步工具调用主动取证、引用证据、写入结构化结论，并在证据不足时拒答。

核心任务定义：

```text
Claim-Level Evidence-Seeking VLM Agent for multimodal document understanding
```

也就是：给定一个多模态文档任务，模型需要围绕每个 claim 主动寻找证据，并输出可追溯、可验证的结果。

### 为什么需要 Agent

很多多模态文档任务不是单张图片问答。模型需要先定位相关区域，再检索上下文证据，再判断证据是否足够支持某个结构化字段。这个过程天然是多步的：

```text
observe page -> inspect/crop region -> retrieve evidence -> open evidence -> write or abstain -> finish
```

因此，本项目重点研究：

- VLM tool-call trajectory learning；
- evidence-grounded multimodal reasoning；
- verifier-guided reinforcement learning；
- unsupported claim 和 hallucination 控制；
- 可执行、可复现的多模态 agent environment。

### 当前进展

已完成第一版离线数据与 SFT 验证：

- 构建了证据索引和多步工具调用轨迹数据；
- 引入了低文本页面的 VLM 辅助转写流程；
- 训练了 Qwen2.5-VL-3B LoRA trajectory SFT adapter；
- 完成了生成式 next-action 评测；
- 明确了下一阶段从 `crop_image(bbox)` 迁移到 `propose_regions -> crop_region(region_id)` 的可执行环境方案。

当前 SFT adapter 在 held-out next-action 评测中的核心结果：

| split | valid action | action type | evidence overlap | retrieval scope |
|---|---:|---:|---:|---:|
| validation | 1.000 | 0.825 | 0.817 | 0.750 |
| test | 0.992 | 0.858 | 0.858 | 0.900 |

这些指标说明模型已经能较稳定地产生合法工具调用，并初步学会在证据检索和 claim 写入之间建立联系。

### 方法概览

当前训练链路分为四步：

1. Evidence indexing：从多模态文档中构建可检索证据块。
2. Trajectory SFT：把专家轨迹拆成“当前观察 + 历史 + 工具返回 -> 下一步 action”的监督样本。
3. Executable environment：实现 `reset/step` 风格的工具调用环境，使模型输出可以被真实执行。
4. On-policy RL：使用 verifier 对完整 trajectory 打分，对比 step-wise verifier-guided GRPO 和 trajectory-level clipped GRPO。

### 主要研究问题

- 小参数 VLM 是否能学会稳定的多步工具调用？
- 候选区域选择是否比直接预测像素级 bbox 更适合文档 agent？
- verifier-guided reward 能否提升证据命中率和拒答能力？
- clipped ratio、reference KL、SFT replay 对 VLM agentic RL 的稳定性有什么影响？

### 代码结构

```text
configs/     Experiment configuration
scripts/     Data construction, training, and evaluation scripts
src/         Environment, tools, verifier, and agent modules
```

大型数据、模型权重、实验输出和内部报告不会提交到本仓库。

### 下一步

下一阶段将实现可执行工具调用环境：

```text
reset
-> propose_regions
-> crop_region
-> retrieve_evidence
-> open_evidence
-> write_claim / abstain_claim
-> finish
```

然后在同一 verifier 下对比：

- SFT only；
- step-wise verifier-guided GRPO；
- trajectory-level GRPO；
- trajectory-level GRPO with clipped ratio；
- trajectory-level GRPO with clipped ratio and reference KL。

## English

EvidenceGrounded-VLM-AgentRL is a VLM tool-call reinforcement learning project for multimodal document understanding.

The goal is to train a vision-language model to solve evidence-seeking document tasks through multi-step tool calls. Instead of producing a one-shot answer, the model must inspect visual regions, retrieve evidence, cite supporting snippets, write structured claims, and abstain when evidence is insufficient.

Core task:

```text
Claim-Level Evidence-Seeking VLM Agent for multimodal document understanding
```

In this task, the model actively gathers evidence for each claim and produces traceable, verifiable outputs.

### Why Agentic Modeling

Many multimodal document tasks are not single-image question answering problems. The model often needs to locate a relevant region, retrieve surrounding textual evidence, verify whether the evidence supports a structured field, and decide whether to answer or abstain.

The natural workflow is multi-step:

```text
observe page -> inspect/crop region -> retrieve evidence -> open evidence -> write or abstain -> finish
```

This project focuses on:

- VLM tool-call trajectory learning;
- evidence-grounded multimodal reasoning;
- verifier-guided reinforcement learning;
- unsupported-claim and hallucination control;
- executable and reproducible multimodal agent environments.

### Current Progress

The first offline data and SFT stage has been completed:

- built an evidence index and multi-step tool-call trajectories;
- added VLM-assisted transcription for low-text pages;
- trained a Qwen2.5-VL-3B LoRA trajectory SFT adapter;
- evaluated generative next-action prediction;
- planned the migration from `crop_image(bbox)` to `propose_regions -> crop_region(region_id)` for the executable environment.

Current SFT adapter results on held-out next-action evaluation:

| split | valid action | action type | evidence overlap | retrieval scope |
|---|---:|---:|---:|---:|
| validation | 1.000 | 0.825 | 0.817 | 0.750 |
| test | 0.992 | 0.858 | 0.858 | 0.900 |

These results show that the model can reliably emit valid tool calls and has learned an initial connection between evidence retrieval and claim writing.

### Method Overview

The current training pipeline has four stages:

1. Evidence indexing: build retrievable evidence chunks from multimodal documents.
2. Trajectory SFT: convert expert trajectories into supervised samples of “current observation + history + tool results -> next action”.
3. Executable environment: implement a `reset/step` tool-call environment where model actions are actually executed.
4. On-policy RL: use a verifier to score complete trajectories and compare step-wise verifier-guided GRPO with trajectory-level clipped GRPO.

### Research Questions

- Can a compact VLM learn stable multi-step tool use?
- Is region-candidate selection more robust than direct pixel-level bbox prediction for document agents?
- Can verifier-guided rewards improve evidence hit rate and abstention behavior?
- How do clipped ratio, reference KL, and SFT replay affect the stability of VLM agentic RL?

### Repository Layout

```text
configs/     Experiment configuration
scripts/     Data construction, training, and evaluation scripts
src/         Environment, tools, verifier, and agent modules
```

Large datasets, model weights, experiment outputs, and internal reports are intentionally excluded from this repository.

### Next Step

The next milestone is an executable tool-call environment:

```text
reset
-> propose_regions
-> crop_region
-> retrieve_evidence
-> open_evidence
-> write_claim / abstain_claim
-> finish
```

Then we will compare the following methods under the same verifier:

- SFT only;
- step-wise verifier-guided GRPO;
- trajectory-level GRPO;
- trajectory-level GRPO with clipped ratio;
- trajectory-level GRPO with clipped ratio and reference KL.
