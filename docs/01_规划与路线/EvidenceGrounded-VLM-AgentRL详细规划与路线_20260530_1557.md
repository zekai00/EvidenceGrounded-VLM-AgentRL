# EvidenceGrounded-VLM-AgentRL 详细规划与路线

- 时间：2026-05-30 15:57 CST
- 新项目目录：`/root/Workspace/VLM/EvidenceGrounded-VLM-AgentRL`
- 旧项目目录：`/root/Workspace/VLM/ChineseLandscape-AgentRL`
- 数据根目录：`/root/datasets/evidence_grounded_vlm_agentrl`

## 1. 重新定义任务

旧说法“山水画 PDF 资料整理 agent”偏应用，面试时说服力不够。新任务定义为：

```text
Claim-Level Evidence-Seeking VLM Agent
```

中文表述：

```text
面向复杂多模态文档的 claim-level 主动取证 VLM Agent。
```

给定 PDF 页面、图像、OCR 文本、本地 evidence store 和有限工具调用预算，模型需要主动选择工具调用，为每个结构化 claim 找到文本证据和视觉证据，并决定哪些字段可以回答、哪些字段应该 abstain。

山水画是第一阶段领域，不是核心卖点。核心卖点是：

- 多模态文档理解；
- 主动取证；
- 工具调用策略学习；
- claim-level provenance；
- 证据约束输出；
- hallucination 控制；
- verifier-guided RL。

## 2. 为什么这个任务需要 Agent

普通 pipeline 适合做：

- PDF 批量解析；
- OCR；
- 图片 block 抽取；
- chunk 建库；
- 向量检索。

Agent 必要的部分是动态决策：

- 下一步看整页还是裁剪图像？
- 图注不清楚时是否 OCR 指定区域？
- 字段缺失时应该搜索什么 query？
- 多个 chunk 冲突时引用哪个？
- 证据不足时应该继续搜索还是 abstain？
- 什么时候 finish，避免无效工具调用？

如果任务只是“一次性抽字段”，不需要 agent。只有把任务设为有限预算下的主动取证，agent 和 RL 才有必要性。

## 3. MDP 形式化

状态：

```text
s_t = {
  page_image,
  optional_crop_images,
  current_claims,
  draft_card,
  history_actions,
  tool_results,
  remaining_budget
}
```

动作：

```text
a_t in {
  crop_image(bbox),
  ocr_region(bbox),
  search_evidence(query, filters),
  open_chunk(chunk_id),
  write_claim(field, value, evidence_ids, confidence),
  abstain_claim(field, reason),
  write_card(card),
  finish
}
```

终止：

```text
finish 或达到 max_tool_steps
```

总 reward：

```text
R = α FieldF1
  + β EvidenceHit@k
  + γ VisualIoU
  + δ CitationSupport
  + η AbstainCalibration
  - λ UnsupportedClaim
  - μ InvalidAction
  - ν RedundantToolCall
```

第一版建议：

| reward 项 | 权重 |
| --- | ---: |
| FieldF1 | 0.30 |
| EvidenceHit@3 | 0.25 |
| VisualIoU | 0.15 |
| CitationSupport | 0.15 |
| AbstainCalibration | 0.10 |
| 工具合法性/效率项 | 0.05 |

## 4. 数据基础

当前已有 canonical evidence store：

```text
/root/Workspace/ChineseLandscape/data/processed/documents
```

核心文件：

| 文件 | 数量 | 用途 |
| --- | ---: | --- |
| `chunks.jsonl` | 4703 | chunk-level 文本证据 |
| `documents.jsonl` | 53 | 文献级索引 |
| `pages.jsonl` | 332 | 部分页码/图片来源记录 |
| `images.jsonl` | 708 | 图片引用 |
| `source_aliases.json` | 53 aliases | 文献标题别名 |

限制：

- 多数文本 chunk 缺少可靠 `page_start/page_end`。
- 第一阶段只做 chunk-level citation。
- 不把 legacy chunks 声称为权威博物馆资料。

数据快照目标：

```text
/root/datasets/evidence_grounded_vlm_agentrl/evidence_store_legacy_milvus_20260530/
```

## 5. Benchmark 版本

### v0.1-local-evidence

目标：用本地 PDF + legacy Milvus chunks 跑通主动取证链路。

任务规模：

```text
train 400
val 100
test 100
```

如果短期数据不足，先以已有 158 条为 seed，扩展到 300-500 条。

任务输入：

```text
PDF page image + task goal + evidence store access
```

任务输出：

```json
{
  "claims": [
    {
      "field": "artist",
      "value": "吴冠中",
      "evidence_ids": ["chunk_xxx"],
      "confidence": 0.88
    }
  ],
  "visual_grounding": {
    "image_bbox": [394, 629, 656, 860],
    "caption_bbox": [394, 866, 601, 895]
  },
  "abstained_fields": [
    {
      "field": "dynasty",
      "reason": "local evidence not found"
    }
  ]
}
```

### v0.2-cross-document

目标：同一作品、技法、构图或视觉元素跨多篇 PDF 查证。

新增难点：

- source_file 不再限定为当前 PDF；
- 同一 claim 可能有多个支持 chunk；
- 需要判断 chunk 是否真的支持该字段。

### v0.3-authority-grounded

目标：引入博物馆/权威网页缓存。

数据源：

- museum collection record；
- official article；
- cached authority webpage；
- 本地 PDF 研究文本。

这一步才做 authority-level 对比和冲突检测。

## 6. 工具设计

### `crop_image`

```json
{"action": "crop_image", "bbox": [x1, y1, x2, y2]}
```

返回裁剪图路径。用于定位图像区域和放大观察。

### `ocr_region`

```json
{"action": "ocr_region", "bbox": [x1, y1, x2, y2]}
```

返回 OCR 文本。用于读取图注和局部正文。

### `search_evidence`

```json
{
  "action": "search_evidence",
  "query": "吴冠中 水墨 泼彩 平远 留白",
  "filters": {"source_file": "中国传统山水画的审美价值.pdf"}
}
```

返回 top-k chunk：

```json
{
  "chunk_id": "chunk_xxx",
  "doc_id": "doc_xxx",
  "source_file": "...pdf",
  "title": "...",
  "retrieval_text": "...",
  "score": 12.3
}
```

### `open_chunk`

```json
{"action": "open_chunk", "chunk_id": "chunk_xxx"}
```

返回完整 `raw_chunk_text`、`contextual_prefix`、metadata 和 quality。

### `write_claim`

```json
{
  "action": "write_claim",
  "field": "composition",
  "value": ["留白", "平远"],
  "evidence_ids": ["chunk_xxx"],
  "visual_bbox": [394, 629, 656, 860],
  "confidence": 0.78
}
```

### `abstain_claim`

```json
{
  "action": "abstain_claim",
  "field": "dynasty",
  "reason": "no supporting evidence in current page or evidence store"
}
```

### `finish`

```json
{"action": "finish"}
```

结束，由 verifier 判分。

## 7. 具体对比方法

不能只写“single-shot VLM”。必须做可复现、可解释的 baseline 矩阵。

### A. 规则与检索管线 Baselines

| 名称 | 方法 | 作用 |
| --- | --- | --- |
| `ImageBlock+CaptionRegex` | 用现有 PDF image block bbox + caption 邻近规则 + 正则抽字段 | 证明简单规则能做到什么 |
| `BM25ChunkOnly` | 不看图，只用 caption/query 搜 chunks，再用规则选字段 | 测文本证据库上限 |
| `OracleImageBox+OCR+BM25` | 给 gold image bbox/caption bbox，只比较证据检索和字段抽取 | 分离定位错误与取证错误 |
| `GoldEvidenceUpperBound` | 给 gold evidence chunk ids，让模型只写 claims | 估计 claim writing 上界 |

### B. 直接生成 VLM Baselines

| 名称 | 模型 | 输入 | 输出 |
| --- | --- | --- | --- |
| `Qwen25VL-3B-PageOnly-JSON` | Qwen2.5-VL-3B-Instruct | page image | card JSON |
| `Qwen25VL-7B-PageOnly-JSON` | Qwen2.5-VL-7B-Instruct | page image | card JSON |
| `Qwen3VL-4B-PageOnly-JSON` | 本地 Qwen3-VL-4B | page image | card JSON |
| `Qwen25VL-7B-Page+Crop-JSON` | Qwen2.5-VL-7B-Instruct | page image + gold/heuristic crop | card JSON |

这里不是 agent，没有工具历史。用于回答“直接 VLM 抽取够不够”。

### C. RAG + VLM Baselines

| 名称 | 检索 | 模型 | 特点 |
| --- | --- | --- | --- |
| `BM25Top5+Qwen25VL-7B` | BM25 top5 chunks | Qwen2.5-VL-7B | 检索结果直接塞 prompt |
| `HybridTop5+Qwen25VL-7B` | BM25 + embedding top5 | Qwen2.5-VL-7B | 更强文本检索 |
| `SourceFilteredTop5+Qwen25VL-7B` | 优先同 PDF source_file | Qwen2.5-VL-7B | 测 source prior 价值 |
| `Qwen37Max-RAG-Teacher` | top5 chunks | qwen3.7-max | teacher / gold 修正，不作为本地可部署主结果 |

这些 baseline 没有多步工具决策，只能一次性用给定检索结果回答。

### D. Prompted Tool-use Agent Baselines

| 名称 | 方法 | 作用 |
| --- | --- | --- |
| `ReAct-Qwen25VL-7B-NoTrain` | 手写 ReAct prompt，模型可调用工具，不训练 | 测 prompting 能否解决 |
| `ReAct-Qwen3VL-4B-NoTrain` | 同上，换本地 Qwen3-VL-4B | 测更强底模零样本工具能力 |
| `HeuristicPolicy` | 固定 `crop -> ocr -> search -> write -> finish` | 检查环境和 verifier |

### E. 训练方法

| 名称 | 训练 | 目的 |
| --- | --- | --- |
| `TrajectorySFT-Qwen25VL-7B` | history-aware next-action SFT | 学合法工具调用和基本策略 |
| `TrajectorySFT-Qwen3VL-4B` | 同上 | 测本地 Qwen3-VL 底模 |
| `SFT+VerifierGRPO` | on-policy GRPO | 优化有限预算取证策略 |
| `SFT+GRPO-NoAbstainReward` | 消融 | 测 abstain reward 价值 |
| `SFT+GRPO-NoEvidencePenalty` | 消融 | 测 unsupported claim 惩罚价值 |
| `SFT+GRPO-NoHistory` | 消融 | 测 history-aware 输入价值 |

## 8. 指标体系

### 工具调用指标

| 指标 | 含义 |
| --- | --- |
| `valid_json_rate` | 输出 JSON 是否可解析 |
| `valid_action_rate` | action 是否属于工具空间且参数合法 |
| `tool_success_rate` | 工具是否成功执行 |
| `avg_tool_calls` | 平均工具调用步数 |
| `redundant_tool_call_rate` | 重复搜索、重复打开同一 chunk 等 |

### 视觉定位指标

| 指标 | 含义 |
| --- | --- |
| `image_bbox_mIoU` | 预测图像 bbox 与 gold bbox 的 mean IoU |
| `image_bbox_IoU@0.5` | 图像 bbox IoU 超过 0.5 的比例 |
| `caption_bbox_mIoU` | 图注 bbox IoU |
| `caption_ocr_char_f1` | 图注 OCR 字符 F1 |

### Claim 指标

| 指标 | 含义 |
| --- | --- |
| `claim_micro_f1` | 所有字段整体 F1 |
| `claim_macro_f1` | 字段均衡 F1 |
| `title_em` | 作品名 exact/normalized match |
| `artist_em` | 画家 exact/normalized match |
| `dynasty_em` | 朝代 exact/normalized match |
| `visual_elements_f1` | 多标签视觉元素 F1 |
| `technique_f1` | 技法多标签 F1 |
| `composition_f1` | 构图多标签 F1 |

### Evidence 指标

| 指标 | 含义 |
| --- | --- |
| `evidence_hit@1` | top1 引用是否命中 gold evidence ids |
| `evidence_hit@3` | top3 是否命中 |
| `evidence_mrr` | evidence ranking MRR |
| `citation_support_precision` | 引用 chunk 是否真的支持 claim |
| `unsupported_claim_rate` | 没有证据却写字段的比例 |
| `wrong_citation_rate` | 引用不支持该字段的 chunk 比例 |

### Abstain 与可靠性指标

| 指标 | 含义 |
| --- | --- |
| `abstain_precision` | abstain 的字段是否确实缺证据 |
| `abstain_recall` | 应该 abstain 的字段是否被 abstain |
| `hallucination_rate` | 无证据或错误证据支撑的生成 |
| `confidence_ece` | 置信度校准误差 |

### 端到端指标

```text
CardScore = 0.20 * VisualGrounding
          + 0.30 * ClaimF1
          + 0.30 * EvidenceScore
          + 0.10 * AbstainScore
          + 0.10 * ToolEfficiency
```

同时报告：

- `end_to_end_success_rate`；
- paired win/loss vs baseline；
- family/source/topic breakdown。

## 9. 训练路线

### Phase 0：项目重命名与数据接入

输出：

- 新项目目录；
- evidence store snapshot；
- `search_evidence` / `open_chunk` 工具；
- 158 条 seed task 回填 `evidence_chunk_ids`。

验收：

- 能对任意 task 搜到 chunk-level evidence；
- verifier 能计算 evidence hit@k；
- 旧 `search_literature` 不再使用临时 PDF 文本搜索。

### Phase 1：v0.1-local-evidence benchmark

输出：

- 300-500 条 task；
- val/test 至少 100 条人工确认；
- schema：claim-level gold + evidence ids + visual bbox。

验收：

- `gold.evidence_chunk_ids` 覆盖率 > 80%；
- `needs_review=false` 的 val/test 样本至少 100 条；
- baseline 可完整运行。

### Phase 2：Baseline 矩阵

必须跑：

1. `ImageBlock+CaptionRegex`
2. `BM25Top5+Qwen25VL-7B`
3. `SourceFilteredTop5+Qwen25VL-7B`
4. `Qwen25VL-7B-PageOnly-JSON`
5. `Qwen25VL-7B-Page+Crop-JSON`
6. `ReAct-Qwen25VL-7B-NoTrain`
7. `Qwen37Max-RAG-Teacher`

验收：

- 每个 baseline 输出同一 schema；
- 同一 verifier 评测；
- 至少报告 claim/evidence/visual/tool 四组指标。

### Phase 3：Trajectory SFT

构造 history-aware next-action 数据：

```text
obs + history + tool_results + draft_claims -> next tool-call JSON
```

推荐轨迹：

```text
crop_image
ocr_region
search_evidence
open_chunk
write_claim(title/artist/dynasty/...)
abstain_claim(unknown fields)
write_card
finish
```

验收：

- local policy `valid_json_rate > 0.95`；
- `valid_action_rate > 0.90`；
- 超过 ReAct no-train tool agent。

### Phase 4：On-policy Rollout

每个 train task 采样 4-8 条轨迹。

记录：

- action history；
- tool result；
- final card；
- reward components；
- trainable group rate。

验收：

- trainable group rate > 0.40；
- all-zero group rate < 0.40；
- unsupported claim 有可观测惩罚信号。

### Phase 5：Verifier-guided GRPO

从 SFT adapter 启动，做小规模 GRPO。

目标：

- evidence hit@3 提升；
- unsupported claim rate 下降；
- avg tool calls 不明显上升；
- paired win/loss 优于 SFT。

关键对比：

```text
SFT vs SFT+GRPO
SFT+GRPO vs SFT+GRPO-NoAbstainReward
SFT+GRPO vs SFT+GRPO-NoEvidencePenalty
SFT+GRPO vs SFT+GRPO-NoHistory
```

## 10. 简历表述

中文：

```text
构建证据约束的多模态主动取证 VLM Agent RL 框架：将复杂 PDF/图像资料理解建模为有限步工具调用 MDP，设计 crop/OCR/retrieve/open/write/abstain 工具空间和 claim-level verifier reward，通过 trajectory SFT + on-policy GRPO 优化模型的证据检索、视觉定位、结构化 claim 生成和幻觉控制能力。
```

英文：

```text
Built an evidence-grounded multimodal tool-use RL framework for claim-level document understanding. Formulated VLM evidence seeking as a finite-budget MDP with visual cropping, OCR, retrieval, citation linking, claim writing, abstention and verifier-guided GRPO, improving grounded claim generation and reducing unsupported hallucinations.
```

## 11. 当前最重要的下一步

不要立刻训练。先完成：

1. 将 canonical evidence store 快照到 `/root/datasets/evidence_grounded_vlm_agentrl/`。
2. 实现 `search_evidence` 和 `open_chunk`。
3. 给现有 158 条 task 回填 `evidence_chunk_ids`。
4. 重建 trajectory SFT，使轨迹包含 `search_evidence/open_chunk/write_claim/abstain_claim`。
5. 跑 baseline 矩阵中的前 4 个，确认任务难度和指标稳定。

