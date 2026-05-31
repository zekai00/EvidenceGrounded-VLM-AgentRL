# EvidenceGrounded-VLM-AgentRL 工作说明

## 项目主线

本项目主线是“证据约束的多模态主动取证 VLM Agent RL”，不是单纯的山水画 PDF 抽取系统。

第一阶段用中国山水画资料作为验证场景，但简历和面试中的任务定义应是：

```text
Claim-Level Evidence-Seeking VLM Agent for multimodal document understanding.
```

模型需要在有限工具调用预算下主动决定：

- 看页面还是裁剪图像；
- 用哪个 retrieval scope 检索证据；
- 打开哪个 evidence；
- 为哪个 claim 写入证据；
- 哪些字段证据不足应该 abstain；
- 何时 finish。

## 关键术语

- VLM：Vision-Language Model，视觉语言模型，例如 Qwen2.5-VL / Qwen3-VL。
- claim：结构化声明，例如 `artist=吴冠中`、`composition=留白`。
- evidence：支持 claim 的文本 chunk、图像区域或权威记录。
- evidence index：证据索引。v0.2 使用 `/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_2_20260530_1914`，核心文件是 `corpus_chunks.jsonl`。
- legacy Milvus chunks：旧 Milvus 证据块，共 4703 个，已经通过 `legacy_chunk_map.json` 映射到新的 `ev_xxx` evidence ID。
- retrieval scope：检索范围。v0.2 支持 `current_page`、`nearby_pages`、`same_document`、`corpus`，分别表示当前页、前后页、同一篇文献、全语料。
- chunk-level citation：引用到证据块级别。旧 Milvus chunks 很多没有可靠页码，因此仍不能伪装成严格 page/bbox citation。
- tool-call trajectory：模型多步调用 `crop_image`、`retrieve_evidence`、`open_evidence`、`write_claim`、`abstain_claim`、`finish` 的完整轨迹。
- trajectory SFT：把高质量轨迹拆成“当前观察 + 历史 + 工具结果 -> 下一步 action”的监督微调数据。
- verifier-guided GRPO：用自动 verifier 的 reward，对同一任务多条 on-policy 轨迹做相对优势优化。

## v0.2 工具定义

- `crop_image(bbox)`：裁剪页面中的目标图像区域。
- `retrieve_evidence(query, scope, anchor, top_k)`：按范围检索证据。`scope` 只能使用 `current_page`、`nearby_pages`、`same_document`、`corpus`。
- `open_evidence(evidence_id)`：打开已经由 `retrieve_evidence` 返回的证据片段。
- `write_claim(field, value, evidence_ids, visual_bbox, confidence)`：写入有证据支撑的字段。
- `abstain_claim(field, reason)`：证据不足时放弃字段。
- `finish`：结束任务。

不要把 v0.1 的 `search_evidence/open_chunk` 继续当成主线工具；它们只作为历史数据迁移来源。

## 数据规范

- 数据集统一放 `/root/datasets/evidence_grounded_vlm_agentrl/`。
- 不把大型 PDF、截图、图片、rollout 原始产物提交到代码仓库。
- 每个数据版本必须有 `manifest.json`、`summary.json` 和中文报告。
- 当前可用数据：
  - v0.2 evidence index：`/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_2_20260530_1914`
  - v0.2 retrieval-scope SFT：`/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_2_retrieval_scope_20260530_1922`
  - v0.3 evidence index：`/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_20260530_2149`
  - v0.3 sentence-snippet SFT：`/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_3_sentence_snippet_20260530_2150`
  - v0.3.1 low-text VLM fallback smoke index：`/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_1_low_text_vlm_smoke_20260530_2308`
- v0.3 起，`retrieve_evidence` 的工具返回应使用 `display_snippet` 和 `evidence_summary`，不要再使用 v0.2 的机械 `snippet=text[:260]`。
- v0.3 起，PDF/text source 的 corpus chunks 应使用 `sentence_paragraph_aware` 分片；legacy Milvus chunks 只能标记为 `legacy_milvus_preserved`。
- v0.3.1 起，扫描页 VLM 转写证据必须标记为 `source_quality=vlm_ocr_fallback`、`citation_level=page_image_transcription`、`quality.silver_text=true`。这类证据只能视为 page-level 弱证据，不能伪装成 bbox-level citation 或人工校勘文本。
- 低文本页 VLM fallback 当前优先使用 `qwen3.6-flash-2026-04-16` + `image_url`，`qwen3.6-plus-2026-04-02` 兜底；本轮验证中 `qwen3.7-max*` 没有稳定接收到页面图像。
- 必须区分：
  - `legacy_milvus_pdf`：旧 Milvus 迁移证据；
  - `local_research_pdf`：本地研究 PDF；
  - `museum_authority`：博物馆/权威机构记录；
  - `web_cached`：缓存网页证据。

## 实验规范

报告中不能只写 overall success。至少包含：

- tool-call valid rate；
- claim micro/macro F1；
- evidence hit@1 / hit@3 / MRR；
- unsupported claim rate；
- abstain accuracy；
- visual grounding IoU；
- avg tool calls；
- paired win/loss against baseline。

## 第一阶段不做

- 不训练图像生成模型。
- 不把任务定义成“点网页按钮”。
- 不把外部网页实时搜索作为不可复现 verifier。
- 不把 legacy chunks 伪装成最终权威数据。
- 不把 v0.2 当前的 416 条迁移任务说成最终完整 benchmark；它只是第一版可训练 seed。
