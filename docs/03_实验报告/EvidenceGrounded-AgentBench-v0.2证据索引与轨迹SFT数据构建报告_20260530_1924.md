# EvidenceGrounded-AgentBench v0.2 证据索引与轨迹 SFT 数据构建报告

- 生成时间：2026-05-30 19:24 CST
- 项目目录：`/root/Workspace/VLM/EvidenceGrounded-VLM-AgentRL`
- 数据根目录：`/root/datasets/evidence_grounded_vlm_agentrl`
- 新权威语料目录：`/root/datasets/chinese_landscape_authority_corpus`

## 本阶段完成了什么

本阶段完成了 v0.2 数据构建的第一版闭环：

1. 从新的权威 PDF 语料和旧 Milvus canonical evidence store 构建离线 evidence index。
2. 将 v0.1 已经 VLM 审核过的 416 条任务迁移到 v0.2 tool schema。
3. 把旧 `chunk_xxx` 证据 ID 映射成新的稳定 `ev_xxx` evidence ID。
4. 将旧动作 `search_evidence/open_chunk` 改成新的 `retrieve_evidence/open_evidence`。
5. 生成 v0.2 oracle episodes 与 history-aware trajectory SFT 数据。
6. 对 JSONL 可读性、证据映射、claim 覆盖率、检索命中率做了质量统计。

## Evidence Index v0.2

输出目录：

`/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_2_20260530_1914`

核心文件：

- `authority_sources.jsonl`：权威来源索引。
- `page_spans.jsonl`：PDF text layer 抽取出的页内文本块。
- `document_spans.jsonl`：文档级文本段。
- `corpus_chunks.jsonl`：统一证据块，训练和检索主要使用这个文件。
- `legacy_chunk_map.json`：旧 `chunk_xxx` 到新 `ev_xxx` 的映射。
- `low_text_pages.jsonl`：PDF text layer 不足、后续需要 OCR/VLM fallback 的页面。
- `manifest.json` / `构建报告.md`：构建记录和统计。

规模统计：

- authority sources：167
- parsed PDFs：165
- PDF pages：3974
- page spans：69727
- document spans：6003
- corpus chunks：10706
- imported legacy Milvus chunks：4703
- low-text pages：826
- parse errors：0

重要说明：

- 现在是离线 JSONL evidence index，还不是新的 Milvus collection。
- PDF text layer 能抽取的内容已经进入 index；低文本页已记录，后续再做 OCR 或 VLM fallback。
- 旧 Milvus chunks 被保留为兼容层，所以旧 v0.1 gold 不会丢失。

## AgentBench v0.2 数据集

输出目录：

`/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_2_retrieval_scope_20260530_1922`

核心文件：

- `tasks_all.jsonl`：416 条任务。
- `train_tasks.jsonl` / `val_tasks.jsonl` / `test_tasks.jsonl`：任务级 split。
- `episodes/oracle_episodes.jsonl`：每条任务的 oracle tool-call 轨迹。
- `sft/train.jsonl` / `sft/val.jsonl` / `sft/test.jsonl`：逐步 next-action SFT 数据。
- `claim_gold.jsonl`：claim 级 gold。
- `evidence_links.jsonl`：claim 与 evidence 的链接。
- `quality_report.json`：质量评测统计。
- `review/review.html`：人工抽查页面。

数据规模：

- tasks：416
- split：train 282 / val 62 / test 72
- claims：2912
- non-abstain claims：2101
- claims with evidence：1908
- claim evidence coverage：0.9081
- SFT rows：train 4178 / val 917 / test 1063
- episodes：416
- avg actions per episode：14.80

## v0.2 Tool Schema

v0.2 的动作空间如下：

- `crop_image(bbox)`：裁剪目标图像区域。
- `retrieve_evidence(query, scope, anchor, top_k)`：按范围检索证据。
- `open_evidence(evidence_id)`：打开检索返回的证据片段。
- `write_claim(field, value, evidence_ids, visual_bbox, confidence)`：写入有证据支撑的字段。
- `abstain_claim(field, reason)`：证据不足时放弃字段。
- `finish`：结束任务。

`retrieve_evidence` 的 scope：

- `current_page`：只查当前页。
- `nearby_pages`：查当前页前后一页。
- `same_document`：查同一篇文献。
- `corpus`：查全语料。

这比 v0.1 的 `search_evidence(query, filters)` 更接近真实 agent 任务：模型需要决定何时先看当前页，何时扩大到前后页、同文档或全语料。

## 质量评测

JSONL 完整性检查通过：

- `tasks_all.jsonl`：416
- `episodes/oracle_episodes.jsonl`：416
- `claim_gold.jsonl`：2912
- `evidence_links.jsonl`：2101
- `sft/train.jsonl`：4178
- `sft/val.jsonl`：917
- `sft/test.jsonl`：1063

证据 ID 映射：

- gold evidence ID mentions：3010
- mapped evidence IDs found in index：3010
- mapped evidence ID presence：1.0000
- unmapped legacy gold evidence IDs：0

检索轨迹质量：

- retrieve calls：1664
- current_page nonempty rate：0.9904
- nearby_pages nonempty rate：0.9952
- same_document nonempty rate：1.0000
- corpus nonempty rate：1.0000
- current_page gold hit rate：0.3389
- nearby_pages gold hit rate：0.3798
- same_document gold hit rate：0.9952
- corpus gold hit rate：0.9928

解释：

- `current_page/nearby_pages` 的 gold hit 低，说明很多旧 gold evidence 来自旧 chunk 级文档证据，而不是严格页级证据。
- `same_document/corpus` 的 gold hit 接近 1，说明迁移后的 evidence ID 和检索轨迹可以支撑 v0.2 SFT。
- 这也说明下一步真正要补的是低文本页 OCR/VLM fallback 和更严格的 page-level citation，而不是继续只优化旧 chunk 映射。

## 当前限制

- v0.2 任务仍主要迁移自 v0.1 的 VLM-audited gold，还不是完全从新权威 corpus 重新生成的 image candidates。
- 826 个 low-text pages 还没有做 OCR/VLM fallback。
- 旧 Milvus chunks 仍是 chunk-level citation，很多没有可靠 page_start/page_end。
- 当前是 supervised oracle trajectory 数据，不是 online RL rollout 数据。

## 下一步

1. 针对 `low_text_pages.jsonl` 做 OCR/VLM fallback，优先补 museum catalog 和作品级馆藏条目。
2. 从新权威 corpus 重新生成 image candidate tasks，不再只依赖 v0.1 的旧 416 条。
3. 建立 v0.2 tool-call environment：让 `retrieve_evidence/open_evidence/write_claim` 真正可 step、可 verifier。
4. 用 v0.2 SFT 数据训练一个 trajectory SFT adapter。
5. 在 verifier 可用后，再做小规模 on-policy GRPO。
