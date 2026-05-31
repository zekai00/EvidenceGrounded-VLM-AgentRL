# Low-Text Pages VLM Fallback 全量与 v0.3.1 SFT 重建报告

时间：2026-05-31 02:51 CST

## 完成内容

本轮完成了低文本页补证据链路：

1. 对 v0.3 evidence index 中 826 个 `low_text_pages` 做 VLM fallback 转写。
2. 将可用 VLM 转写结果合并为 v0.3.1 full evidence index。
3. 基于 v0.3.1 full evidence index 重建 AgentBench / trajectory SFT 数据。

## VLM Fallback

输出目录：

```text
/root/datasets/evidence_grounded_vlm_agentrl/low_text_vlm_fallback_v0_3_1_full_qwen36flash_20260530_2312
```

统计：

| 指标 | 数值 |
|---|---:|
| low_text_pages_total | 826 |
| processed_pages | 826 |
| ok_pages | 810 |
| readable_pages | 651 |
| pages_with_40plus_chars | 457 |
| avg_chars_after | 244.6 |

主模型 `qwen3.6-flash-2026-04-16`，兜底模型 `qwen3.6-plus-2026-04-02`。

## v0.3.1 Evidence Index

输出目录：

```text
/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_1_low_text_vlm_full_20260531_0140
```

合并策略：

- 只合并 `ok=true`、`is_readable=true`、清洗后文本长度不低于 40 字的页面。
- 标记为 `source_quality=vlm_ocr_fallback`。
- 标记为 `citation_level=page_image_transcription`。
- 标记为 `quality.silver_text=true` 和 `quality.requires_human_spot_check=true`。

合并结果：

| 指标 | 数值 |
|---|---:|
| usable_fallback_rows | 457 |
| added_page_spans | 457 |
| added_document_spans | 457 |
| added_corpus_chunks | 457 |
| qwen3.6-flash rows | 417 |
| qwen3.6-plus rows | 40 |

## v0.3.1 SFT 数据

输出目录：

```text
/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_3_1_low_text_vlm_full_sft_20260531_0248
```

任务规模：

| split | tasks | SFT rows |
|---|---:|---:|
| train | 282 | 4178 |
| val | 62 | 917 |
| test | 72 | 1063 |
| total | 416 | 6158 |

关键质量：

| 指标 | 数值 |
|---|---:|
| claims | 2912 |
| non_abstain_claims | 2101 |
| claims_with_evidence | 1908 |
| claim_evidence_coverage | 0.9081 |
| mapped_evidence_id_presence | 1.0 |
| avg_actions_per_episode | 14.80 |
| display_snippet_coverage | 1.0 |
| sentence_boundary_ok_rate | 0.8478 |

检索：

| scope | nonempty_rate | gold_hit_rate |
|---|---:|---:|
| current_page | 0.9976 | 0.3389 |
| nearby_pages | 1.0000 | 0.3798 |
| same_document | 1.0000 | 0.9928 |
| corpus | 1.0000 | 0.9928 |

## 结论

- 低文本页已有 457 页以低置信度 silver evidence 形式进入 page/document/corpus 三层索引。
- `current_page` retrieval 基本不再空缺，后续 trajectory SFT 可以在扫描页、图版页附近拿到页面级证据。
- 这批 VLM fallback 仍然不是人工校勘文本，不能当作 bbox-level citation，只能作为 page-level transcription evidence。
