# EvidenceGrounded-AgentBench v0.3 句子分片与 Snippet 修复重建报告

- 生成时间：2026-05-30 21:53 CST
- 项目目录：`/root/Workspace/VLM/EvidenceGrounded-VLM-AgentRL`
- 数据根目录：`/root/datasets/evidence_grounded_vlm_agentrl`

## 本次目标

v0.2 的两个主要问题是：

1. `corpus_chunks` 对 PDF/text source 使用字符窗口分片，默认 `900 chars + 160 chars overlap`，可能切断句子或段落。
2. `retrieve_evidence` 返回的 `snippet` 是 `text[:260]`，经常不是完整句子。

v0.3 的目标是修复这两个问题：

- chunk 分片改成 `sentence_paragraph_aware`。
- evidence row 增加 `raw_text / clean_text / display_snippet / evidence_summary`。
- `retrieve_evidence` 工具返回改用 `display_snippet` 和 `evidence_summary`。
- 不覆盖 v0.2，保留旧版本可复现。

## 修改文件

- `scripts/build_evidence_index_v0_3.py`
- `scripts/migrate_agentbench_v0_1_to_v0_3.py`
- `AGENTS.md`
- `docs/codex-worklog.md`

## v0.3 Evidence Index

输出目录：

`/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_20260530_2149`

核心改动：

- PDF 页内 block 仍来自 PyMuPDF `page.get_text("blocks")`，保留为 `page_spans`。
- 文档级 chunk 不再用纯字符窗口，而是先按段落、句子和长句边界切成 units，再累计成 chunk。
- chunk overlap 不再复制任意尾部字符，而是复制尾部 1-2 个句段 units。
- `display_snippet` 先按句子边界生成；如果首段较长，会继续放宽到约 2 倍长度以避免停在半句。
- `evidence_summary` 当前是抽取式摘要：优先使用完整句段压缩，不新增事实。
- legacy Milvus chunks 保留原分片，标记为 `legacy_milvus_preserved`。

规模统计：

- authority sources：167
- parsed PDFs：165
- PDF pages：3974
- page spans：69727
- document spans：6591
- corpus chunks：11294
- legacy chunks imported：4703
- low-text pages：826
- parse errors：0
- avg corpus chunk chars：613.59

分布：

- source quality：`{'pdf_text_layer': 82323, 'metadata_text_source': 586, 'legacy_milvus': 4703}`
- corpus segmentation：`{'sentence_paragraph_aware': 6591, 'legacy_milvus_preserved': 4703}`
- corpus citation level：`{'page_range_chunk': 6298, 'chunk': 4996}`
- display snippet rows：11294 / 11294
- evidence summary rows：11294 / 11294

## v0.3 AgentBench / SFT

输出目录：

`/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_3_sentence_snippet_20260530_2150`

核心文件：

- `tasks_all.jsonl`
- `train_tasks.jsonl` / `val_tasks.jsonl` / `test_tasks.jsonl`
- `episodes/oracle_episodes.jsonl`
- `sft/train.jsonl` / `sft/val.jsonl` / `sft/test.jsonl`
- `claim_gold.jsonl`
- `evidence_links.jsonl`
- `quality_report.json`
- `review/review.html`

规模：

- tasks：416
- split：train 282 / val 62 / test 72
- claims：2912
- non-abstain claims：2101
- claims with evidence：1908
- claim evidence coverage：0.9081
- SFT rows：train 4178 / val 917 / test 1063
- episodes：416
- avg actions per episode：14.80

检索质量：

- retrieve calls：1664
- current_page gold hit rate：0.3389
- nearby_pages gold hit rate：0.3798
- same_document gold hit rate：0.9928
- corpus gold hit rate：0.9928

Snippet 质量：

- display_snippet coverage：1.0000
- evidence_summary coverage：1.0000
- sentence boundary ok rows：9551 / 11294
- sentence boundary ok rate：0.8457
- avg snippet len：419.80
- max snippet len：1401

解释：

- v0.2 迁移初版的 sentence boundary ok rate 约 0.54。
- v0.3 最终版提升到 0.8457。
- 剩余不通过的主要原因是 PDF text layer 本身存在行断裂、古籍/目录式文本无完整句号、旧 legacy chunk 本身 citation 和段落结构不完整。

## 验证

已完成：

- `python3 -m py_compile scripts/build_evidence_index_v0_3.py scripts/migrate_agentbench_v0_1_to_v0_3.py`
- 全量 JSONL 可读性检查。
- SFT 行数检查：
  - train：4178
  - val：917
  - test：1063
- `retrieve_evidence` 工具返回检查：
  - 返回字段包含 `display_snippet`、`evidence_summary`、`source_quality`、`citation_level`。
  - 不再使用旧字段 `snippet`。

## 仍然存在的问题

- v0.3 仍是从 v0.1 VLM-audited gold 迁移过来的 seed 数据，不是从新权威 corpus 完全重新生成的新 benchmark。
- legacy Milvus chunks 仍然大量存在，很多 `page_start/page_end=null`，只能做 chunk-level citation。
- 826 个 low-text pages 仍未做 OCR/VLM fallback。
- 当前 `evidence_summary` 是抽取式摘要，还不是强模型生成的忠实摘要。
- 当前检索模拟仍带 oracle bonus，用于 SFT trace 构建可以，不能作为真实检索评测或 RL 环境。

## 下一步

1. 针对 `low_text_pages.jsonl` 做 OCR/VLM fallback。
2. 对 v0.3 `evidence_summary` 使用强模型做忠实压缩校正，保留 `raw_text` 可追溯。
3. 从新权威 corpus 重新生成更多 image candidate tasks，目标 1000-2000 条。
4. 实现真实 v0.3 tool-call environment：检索不能使用 gold/candidate bonus。
5. 用 v0.3 SFT 训练 trajectory adapter，再做小规模 on-policy GRPO。
