# LowTextPages VLM Fallback 与 v0.3.1 合并试跑报告

- 生成时间：2026-05-30 23:08 CST
- 项目：`/root/Workspace/VLM/EvidenceGrounded-VLM-AgentRL`
- 原始 evidence index：`/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_20260530_2149`
- VLM fallback 输出：`/root/datasets/evidence_grounded_vlm_agentrl/low_text_vlm_fallback_v0_2_qwen36flash_smoke_20260530_2257`
- v0.3.1 smoke index：`/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_1_low_text_vlm_smoke_20260530_2308`

## 本阶段目的

v0.3 已经修复了 snippet 和 chunk 分片，但仍有 826 个 `low_text_pages`。这些页面通常是扫描页、影印古籍页、图版页或空白页，PDF text layer 几乎没有文字。如果不处理，agent 在 `current_page/nearby_pages` 范围检索时会缺少本页证据。

本阶段试跑目标是验证：

1. 能否把低文本 PDF 页渲染成图片。
2. 能否用 VLM 对整页图片做忠实转写。
3. 能否把转写结果作为低置信度 `vlm_ocr_fallback` 证据合并回 evidence index。

## 模型与接口结论

- `qwen3.7-max-2026-05-20`：对本任务图像请求响应慢，且本轮没有稳定得到有效图像理解结果。
- `qwen3.7-max-2026-05-17`：接口返回 JSON，但模型多次表示“未接收到页面图像”，说明当前 DashScope compatible 调用格式不适合作为本阶段主力。
- `qwen3.6-plus-2026-04-02`：`image_url` 格式可正常接收图像，能转写竖排古籍扫描页。
- `qwen3.6-flash-2026-04-16`：本轮采用为主模型，12 页中 11 页由 flash 成功完成；1 页超时后由 plus fallback 完成。

因此后续全量 fallback 建议使用：

```bash
python3 scripts/build_low_text_vlm_fallback.py \
  --model qwen3.6-flash-2026-04-16 \
  --fallback-models qwen3.6-plus-2026-04-02,qwen3.7-max-2026-05-17,qwen3.7-max \
  --dashscope-image-format image_url
```

## 小批量试跑结果

- low_text_pages_total：826
- selected_pages：12
- processed_pages：12
- ok_pages：12
- readable_pages：12
- pages_with_40plus_chars：12
- avg_chars_after：273.75
- page_type_counts：`{"scan_text": 12}`
- fallback_model_counts：`{"qwen3.6-flash-2026-04-16": 11, "qwen3.6-plus-2026-04-02": 1}`

输入页来自：

- PDF：`A01_古代画论_歷代名畫記十卷_明刻本影印.pdf`
- 页码：21-32
- 类型：竖排繁体中文古籍影印扫描页

## 抽样例子

页面图：

![](/root/datasets/evidence_grounded_vlm_agentrl/low_text_vlm_fallback_v0_2_qwen36flash_smoke_20260530_2257/rendered_pages/A01_古代画论_歷代名畫記十卷_明刻本影印_p0021.jpg)

VLM 转写片段：

```text
歷代名畫記
卷五
臆有背亡遺失尺度此其難也曹不興能之長康又
曾於瓦棺寺北小殿畫維摩詰畫訖光彩耀目數日
京師寺記云興寧中瓦棺寺初置僧眾設會請朝賢
鳴刹注䟽其時士大夫莫有過十萬者既至長康自
打刹注百萬長康素貧衆以爲大言後寺眾請勾䟽
長康曰宜備一壁遂閉户徃来一月餘日所畫維摩
詰一軀工畢將欲㸃眸子乃謂寺僧曰第一日觀者
請施十萬第二日可五萬第三日可任例責施及開
户光照一寺施者填咽俄而得百萬錢劉義慶世說
云桓大司馬每請長康與羊欣論書畫竟夕忘疲孫
暢之述畫記云畫冠冕而亡面貌勝於戴逵謝赫云
```

这类文本适合做检索召回和弱证据，不适合直接当作人工校勘版引用。原因是古籍 OCR/VLM 转写可能存在异体字、漏字、错字和断句问题。

## v0.3.1 Smoke Index 合并结果

合并脚本：

- `scripts/merge_low_text_fallback_into_index_v0_3_1.py`

合并方式：

- 复制 v0.3 index 到新目录，不修改原始 v0.3。
- 对每个可用 fallback 页面追加三类 evidence rows：
  - `page_spans.jsonl`：`vlm_ocr_page_transcription`
  - `document_spans.jsonl`：`vlm_ocr_page_chunk`
  - `corpus_chunks.jsonl`：`vlm_ocr_page_chunk`

新增行数：

- added_page_spans：12
- added_document_spans：12
- added_corpus_chunks：12

合并后行数对比：

- `page_spans.jsonl`：69727 -> 69739
- `corpus_chunks.jsonl`：11294 -> 11306

新增证据标记：

- `source_quality="vlm_ocr_fallback"`
- `citation_level="page_image_transcription"`
- `quality.silver_text=true`
- `quality.requires_human_spot_check=true`

## 质量判断

本轮链路成立：

- 页面渲染正常。
- `qwen3.6-flash-2026-04-16` 可以通过 `image_url` 接收整页图像。
- 12/12 页产生可读转写。
- fallback 文本已能进入 page/document/corpus 三层 evidence index。

仍需注意：

- 这不是最终校勘文本，不能标为高置信权威原文。
- 当前没有 bbox 级 OCR 坐标，因此只能做到 page-level citation。
- 全量 826 页按本轮速度串行会很慢，建议后续用 offset 分片、2-4 worker 小并发、失败可 resume 的方式跑。
- 空白页、封面、图版页应先用本地图像规则或 VLM page_type 过滤，避免浪费额度。

## 下一步建议

1. 全量处理 826 个 low_text_pages，主模型用 `qwen3.6-flash-2026-04-16`，plus 兜底。
2. 对全量 fallback 输出做质量分层：`scan_text` 可合并，`blank/image_plate/unknown` 暂不合并或只保留元数据。
3. 生成正式 `evidence_index_v0_3_1_low_text_vlm_full`。
4. 基于 v0.3.1 full index 重新构建 SFT 数据，让 trajectory 在低文本扫描页上也能检索到本页证据。
