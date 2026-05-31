# EvidenceGrounded-VLM-AgentRL 优化版规划与路线

- 时间：2026-05-30 18:42 CST
- 项目目录：`/root/Workspace/VLM/EvidenceGrounded-VLM-AgentRL`
- 数据目录：`/root/datasets/evidence_grounded_vlm_agentrl`
- 当前数据版本：`agentbench_v0_1_vlm_audited_flash_full_20260530_1641`

## 1. 核心结论

当前项目不要继续讲成“山水画 PDF 资料整理 agent”，也不要把重点放在“找 caption candidate”。更合理的项目定义是：

```text
Evidence-Grounded Figure Understanding with VLM Tool-Call RL
```

中文表述：

```text
证据约束的图文混排文档图像理解 VLM Agent RL
```

任务本体是：给定 PDF 页面中的目标图像，模型需要在有限工具预算内，结合视觉观察和可追溯文本证据，输出结构化 claim，并对缺证据字段拒答。

山水画只是第一阶段领域数据，不是项目卖点。项目卖点是：

- 图文混排 PDF 中的图像-文本证据对齐；
- VLM 幻觉控制；
- claim-level citation；
- 多步工具调用策略学习；
- verifier-guided RL。

## 2. 这个任务为什么有意义

普通 VLM 可以看图，但容易凭常识回答，不能保证证据闭环。普通 RAG 可以检索文字，但不知道目标图像到底是哪一张，也不知道文字是否真的对应当前图像。普通 OCR/规则可以抽图旁文字，但 PDF 排版里图注、正文、图号经常跨页或错位。

因此有价值的问题不是“把 PDF 解析出来”，而是：

```text
给定一个目标图像，模型是否能主动决定：
1. 先看图还是先找文字证据；
2. 在当前页、邻近页、同文档还是全库找证据；
3. 哪些字段可以写，哪些字段必须 abstain；
4. 输出的每个 claim 是否有证据支持。
```

这比“单步图像问答”更接近高可信多模态 RAG，也比“网页 GUI 操作”更贴合 VLM 文档理解场景。

面试/简历表述建议：

```text
构建了一个面向图文混排 PDF 的 evidence-grounded VLM agent benchmark，
训练模型在有限工具预算下完成图像识别、证据检索、claim 写作与拒答决策，
并通过 verifier-guided RL 降低 unsupported claim 和 hallucination。
```

## 3. 要避免的方向

### 不把 caption candidate 当主线

`caption` 是图注，例如：

```text
图2-3 南宋马远《踏歌图》 绢本 北京故宫博物院藏
```

图注很重要，但不应该单独做成项目主线。如果主线变成 `find_caption_candidates`，后续会陷入反复优化图注候选规则，削弱 agent RL 的意义。

优化后的做法：图注只是证据的一种，统一由 `retrieve_evidence` 返回。

### 不开放任意翻页全文读取

允许模型直接 `open_page(17)`、`open_page(18)` 甚至任意读整篇 PDF，会让任务变成“到处翻资料找答案”，边界不干净，也不利于 verifier 控制。

优化后的做法：允许通过检索访问前后页或全文片段，但不让模型任意读取整页全文。

### v0.2 不引入实时互联网

互联网搜索会带来不可复现、版权、时效和噪声问题。当前阶段先用本地 PDF、同文档 text span、canonical evidence store。权威网页可以作为 v0.3 的离线缓存数据源，而不是在线 web tool。

## 4. 优化后的任务定义

### 输入

每个 task 包含：

```json
{
  "task_id": "egva_xxx",
  "document_id": "doc_xxx",
  "source_file": "...pdf",
  "page": 17,
  "page_image": ".../page.png",
  "target_bbox": [x1, y1, x2, y2],
  "evidence_indexes": {
    "page_text": "当前 PDF 的 text layer / OCR span index",
    "document_text": "同一 PDF 的 chunk/span index",
    "corpus": "本地 canonical evidence store"
  }
}
```

### 输出

输出是一张 evidence card：

```json
{
  "is_relevant_figure": true,
  "claims": [
    {
      "field": "title",
      "value": "踏歌图",
      "evidence_ids": ["ev_same_doc_001"],
      "confidence": 0.86
    },
    {
      "field": "composition",
      "value": ["留白"],
      "evidence_ids": ["ev_corpus_010"],
      "visual_bbox": [120, 300, 850, 760],
      "confidence": 0.78
    }
  ],
  "abstained_fields": [
    {
      "field": "technique",
      "reason": "no direct supporting evidence found"
    }
  ]
}
```

字段分三类：

| 类型 | 字段 | 证据要求 |
| --- | --- | --- |
| 身份字段 | `title`, `artist`, `dynasty` | 优先当前页/邻近页/同文档证据，不建议只靠全库背景知识 |
| 视觉字段 | `visual_elements` | 可由 crop 图像支持，可附 `visual_bbox` |
| 解释字段 | `technique`, `composition` | 需要文本证据或图文共同证据支持，证据不足要 abstain |

## 5. 证据范围设计

统一使用 `retrieve_evidence(query, scope, anchor, top_k)`。不要再拆成 `find_caption_candidates`、`search_pdf_text`、`search_evidence` 多个概念。

### 证据范围

| scope | 查找位置 | 用途 | 约束 |
| --- | --- | --- | --- |
| `current_page` | 当前页 text layer / OCR spans | 图注、图号、图旁说明 | 最高优先级 |
| `nearby_pages` | 当前页 ±1 页 text spans | 跨页图注或上/下文说明 | 只返回 top-k span，不开放整页 |
| `same_document` | 同一 PDF 全文 chunk/span | 图号、作品名、作者在正文其他位置解释 | 只返回检索结果 |
| `corpus` | 所有本地文献 chunks | 背景性解释、技法/构图概念 | 不能单独强支撑身份字段 |
| `authority_cache` | 离线权威网页/博物馆记录 | v0.3 引入，用于权威校验 | 不用实时互联网 |

### 是否能前后翻页

可以，但不是让模型自由翻页。

正确形式：

```json
{
  "action": "retrieve_evidence",
  "query": "图2-3 踏歌图 马远",
  "scope": "nearby_pages",
  "anchor": {"page": 17, "bbox": [120, 300, 850, 760]},
  "top_k": 5
}
```

含义：环境在当前页、上一页、下一页中检索候选证据 span，只返回 top-k。模型不能直接读取整页，也不能无限翻页。

## 6. 工具协议 v0.2

保留少量高层工具，避免工具过碎。

### `crop_image`

```json
{"action":"crop_image","bbox":[120,300,850,760]}
```

用途：

- 确认目标图像是否相关；
- 提取视觉元素；
- 为视觉字段提供 grounding。

### `retrieve_evidence`

```json
{
  "action":"retrieve_evidence",
  "query":"马远 踏歌图 南宋 留白",
  "scope":"same_document",
  "anchor":{"page":17,"bbox":[120,300,850,760]},
  "top_k":5
}
```

返回：

```json
[
  {
    "evidence_id":"ev_same_doc_001",
    "scope":"same_document",
    "source_file":"...pdf",
    "page":17,
    "text":"图2-3 南宋马远《踏歌图》...",
    "score":0.91
  }
]
```

用途：

- 当前页找图注；
- 前后页找跨页说明；
- 同 PDF 找正文解释；
- corpus 找背景证据。

### `open_evidence`

```json
{"action":"open_evidence","evidence_id":"ev_same_doc_001"}
```

用途：

- 打开检索返回的 evidence span/chunk；
- 读取完整文本、source、page、quality metadata。

约束：

- 只能打开本轮或历史 `retrieve_evidence` 返回过的 `evidence_id`。
- 不能任意 open 某页或某 chunk。

### `write_claim`

```json
{
  "action":"write_claim",
  "field":"artist",
  "value":"马远",
  "evidence_ids":["ev_same_doc_001"],
  "confidence":0.86
}
```

### `abstain_claim`

```json
{
  "action":"abstain_claim",
  "field":"technique",
  "reason":"current page, nearby pages, and same-document search do not provide direct support"
}
```

### `finish`

```json
{"action":"finish"}
```

建议 action budget：

```text
max_steps = 6 到 8
```

典型轨迹：

```text
crop_image
retrieve_evidence(scope=current_page)
open_evidence
retrieve_evidence(scope=same_document 或 corpus)
open_evidence
write_claim / abstain_claim
finish
```

## 7. Verifier 与 Reward

### Verifier 检查项

| 检查项 | 含义 |
| --- | --- |
| JSON/action 合法性 | action 是否可解析、参数是否合法 |
| 工具权限 | 是否只打开 retrieve 返回过的 evidence |
| scope 合理性 | 是否在信息不足时逐步扩大范围 |
| claim 正确性 | 字段值是否匹配 gold |
| citation support | evidence 是否真的支持 claim |
| abstain 合理性 | 缺证据字段是否拒答 |
| 工具效率 | 是否无意义重复检索/打开 |

### Reward 建议

```text
R = 0.25 * ClaimF1
  + 0.25 * EvidenceSupport
  + 0.15 * VisualGrounding
  + 0.15 * AbstainCalibration
  + 0.10 * ScopePolicy
  + 0.10 * ToolValidity
  - 0.30 * UnsupportedClaim
  - 0.10 * RedundantToolCall
```

`ScopePolicy` 用于鼓励合理证据路径：

- 身份字段优先 `current_page / nearby_pages / same_document`；
- 解释字段可使用 `corpus`；
- 如果当前页已有强证据，不奖励继续全库搜索；
- 如果没有证据仍写 claim，强惩罚。

## 8. 当前 v0.1 状态

已经完成：

- 规则候选池：500 tasks
- VLM 审核后 clean tasks：416
- split：train 282 / val 62 / test 72
- SFT rows：train 3664 / val 806 / test 936
- claim chunk evidence coverage：0.9081

当前数据目录：

```text
/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_1_vlm_audited_flash_full_20260530_1641
```

当前 v0.1 的问题：

- 工具协议仍偏旧：`search_evidence/open_chunk`，需要迁移到 `retrieve_evidence/open_evidence`。
- evidence 主要来自 legacy Milvus chunk，page-level 证据不完整。
- `vlm_input_mode_counts` 中有 13 条 `dashscope_text_only`，建议复审或剔除出 val/test。
- val/test 还没有人工 spot-check。
- task 还没有真正可执行的 v0.2 环境和 verifier。

## 9. v0.2 数据重构计划

### Step 1：构建统一 evidence index

输出：

```text
evidence_index/
  page_spans.jsonl
  document_spans.jsonl
  corpus_chunks.jsonl
  evidence_manifest.json
```

每条 evidence 统一 schema：

```json
{
  "evidence_id":"ev_xxx",
  "scope":"current_page|nearby_pages|same_document|corpus",
  "doc_id":"doc_xxx",
  "source_file":"...pdf",
  "page":17,
  "bbox":null,
  "text":"...",
  "source_quality":"pdf_text_layer|ocr|legacy_milvus|manual",
  "citation_level":"page|chunk|span"
}
```

验收：

- 每个 task 至少能从 `current_page/nearby_pages/same_document` 检索到若干 span；
- corpus chunk 仍保留；
- 检索结果能统一用 `open_evidence` 打开。

### Step 2：把 v0.1 tasks 升级到 v0.2 schema

输出：

```text
agentbench_v0_2_retrieval_scope/
  tasks_all.jsonl
  train_tasks.jsonl
  val_tasks.jsonl
  test_tasks.jsonl
  sft/train.jsonl
  sft/val.jsonl
  sft/test.jsonl
```

升级内容：

- `gold.evidence_chunk_ids` 改为 `gold.evidence_ids`；
- 记录每个 evidence 的 `scope`；
- 增加 `retrieval_gold`：每个字段建议优先在哪个 scope 找；
- 移除 text-only 审核样本或放入 train-only。

验收：

- final tasks 目标 350-450；
- val/test 中 text-only audit 样本为 0；
- val/test 每个样本至少 3 个字段有 evidence support 或合理 abstain。

### Step 3：实现 v0.2 tool-call 环境

核心类：

```text
envs/evidence_agent/
  environment.py
  tools.py
  verifier.py
  schema.py
```

环境支持：

- `reset(task_id)`；
- `step(action_json)`；
- tool results；
- final card；
- reward components；
- rollout trace 保存。

验收：

- Heuristic policy 可以跑完整个 val；
- invalid action 有明确错误；
- reward components 可解释。

### Step 4：Baseline

必须先跑这些，避免直接训练后无法解释：

| baseline | 目的 |
| --- | --- |
| `PageOnly-VLM` | 测单步看页能力 |
| `Page+Crop-VLM` | 测裁剪图像帮助 |
| `RetrieveTopK+VLM` | 测非 agent RAG 能力 |
| `HeuristicPolicy` | 测固定工具路径上限 |
| `ReAct-NoTrain` | 测 prompt tool-use 能力 |
| `OracleEvidence` | 测 gold evidence 上限 |

验收：

- 所有 baseline 输出同一 card schema；
- 使用同一 verifier；
- 报告 claim/evidence/abstain/tool 四组指标。

### Step 5：Trajectory SFT

训练目标：

```text
obs + history + tool_results + draft_claims -> next_action_json
```

第一版使用 Qwen2.5-VL-7B 或 Qwen3-VL-4B/8B LoRA。

验收：

- valid_json_rate > 0.95；
- valid_action_rate > 0.90；
- evidence-supported claim 高于 ReAct-NoTrain；
- unsupported_claim_rate 低于 PageOnly-VLM。

### Step 6：Verifier-guided On-policy RL

先做小实验，不直接大规模训练。

配置建议：

```text
rollouts_per_task = 4
max_steps = 6 或 8
train_tasks = 100 到 200
eval_interval = 每 50 到 100 update
```

目标不是追求所有字段都答，而是：

- unsupported claim 下降；
- evidence support precision 上升；
- 合理 abstain 上升；
- 工具调用更有效率。

## 10. 指标

### End-to-end

```text
CardScore = 0.25 * ClaimF1
          + 0.30 * EvidenceSupport
          + 0.15 * VisualGrounding
          + 0.15 * AbstainScore
          + 0.15 * ToolPolicy
```

### 关键指标

| 指标 | 含义 |
| --- | --- |
| `claim_micro_f1` | 字段整体正确性 |
| `claim_macro_f1` | 字段均衡正确性 |
| `evidence_support_precision` | 引用证据是否支持 claim |
| `unsupported_claim_rate` | 无证据 claim 比例 |
| `abstain_precision/recall` | 拒答是否合理 |
| `scope_accuracy` | 是否选择合理证据范围 |
| `valid_action_rate` | 工具调用是否合法 |
| `avg_tool_calls` | 工具效率 |

### 必须做的 breakdown

- 按字段：title / artist / dynasty / visual_elements / technique / composition；
- 按 scope：current_page / nearby_pages / same_document / corpus；
- 按来源：不同 PDF source；
- 按模型：SFT / RL / baseline；
- 按错误类型：错字段、错证据、无证据瞎写、过度 abstain、无效工具。

## 11. 项目下一步

当前最优先顺序：

1. **重构工具协议文档和 schema**
   - 用 `retrieve_evidence/open_evidence` 替代旧 `search_evidence/open_chunk`。
   - 明确 scope 和 evidence_id。

2. **构建 v0.2 evidence index**
   - 从 PDF text layer/OCR 构建 page/document span。
   - 接入 legacy corpus chunks。

3. **升级 v0.1 数据为 v0.2**
   - 保留 416 clean tasks；
   - 剔除或复审 text-only 样本；
   - 为每条 claim 标注 evidence scope。

4. **实现 v0.2 environment + verifier**
   - 先让 heuristic policy 能跑通；
   - 再跑 baseline。

5. **训练 trajectory SFT**
   - 先不要直接 RL；
   - 先保证工具调用合法性和 evidence card schema 稳定。

6. **小规模 on-policy RL**
   - 以 unsupported claim 和 evidence support 为主 reward；
   - 做消融：无 abstain reward、无 evidence penalty、无 scope reward。

## 12. 当前需要特别注意

- 不要把任务做成 PDF parsing 工程；PDF parsing 只是环境工具，不是研究贡献。
- 不要让模型任意读整篇 PDF；所有文本访问都应通过检索返回 evidence。
- 不要把 corpus evidence 当成身份字段的强证据；身份字段应优先同页/邻页/同文档。
- 不要让 VLM teacher 的错误进入 val/test；val/test 要人工抽查。
- 不要过早做 RL；没有稳定 verifier 和 baseline，RL 结果无法解释。
- 不要只报告最终分数；必须报告 unsupported claim、evidence support、abstain calibration。
- 如果未来引入互联网，必须先缓存成离线 authority evidence store，再纳入 benchmark。
