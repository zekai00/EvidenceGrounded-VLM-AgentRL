# Codex Worklog

## 2026-05-30 15:57 CST 初始化

- 新建项目目录：`/root/Workspace/VLM/EvidenceGrounded-VLM-AgentRL`。
- 旧 `ChineseLandscape-AgentRL` 保留为历史目录。
- 新项目定位为 `Claim-Level Evidence-Seeking VLM Agent`，用山水画资料作为第一阶段验证场景。
- 新增主规划：
  - `docs/01_规划与路线/EvidenceGrounded-VLM-AgentRL详细规划与路线_20260530_1557.md`
- 新增配置：
  - `configs/evidence_agent_v0_1_local.yaml`

## 2026-05-30 18:12 CST EvidenceGrounded-AgentBench v0.1 VLM 审核

- 新增/更新脚本：
  - `scripts/build_local_evidence_agentbench.py`
  - `scripts/audit_agentbench_with_vlm.py`
- 构建规则候选池：
  - `/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_1_local_evidence_20260530_1625`
  - 输入 55 个 raw PDFs，抽取 500 个 PDF image-block candidate tasks。
- 增加 VLM 审核层：
  - `qwen3.7-max*` 使用 DashScope `{"type":"image","image":...}` 图片格式。
  - `qwen3.6/Qwen-VL` 风格模型使用 `{"type":"image_url","image_url":...}`。
  - JSON 解析失败也会触发 fallback。
- 最终 clean 数据：
  - `/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_1_vlm_audited_flash_full_20260530_1641`
  - 500 输入 tasks，VLM 移除 84 个无关图像，保留 416 个 clean tasks。
  - split：train 282 / val 62 / test 72。
  - SFT rows：train 3664 / val 806 / test 936。
  - claim chunk evidence coverage：0.9081。
- 报告：
  - `docs/03_实验报告/EvidenceGrounded-AgentBench-v0.1-VLM审核构建报告_20260530_1812.md`

## 2026-05-30 18:42 CST 优化版项目路线

- 新增规划：
  - `docs/01_规划与路线/EvidenceGrounded-VLM-AgentRL优化版规划与路线_20260530_1842.md`
- 关键调整：
  - 不再把 `find_caption_candidates` 作为核心工具，改为统一 `retrieve_evidence(query, scope, anchor, top_k)`。
  - 不开放任意翻页全文读取，只允许通过 `current_page / nearby_pages / same_document / corpus` scope 检索 evidence。
  - v0.2 先不接实时互联网；如需权威网页，后续做离线 `authority_cache`。
  - 下一阶段优先做 v0.2 evidence index、tool-call environment、verifier 和 baseline，再做 trajectory SFT / RL。

## 2026-05-30 19:24 CST v0.2 evidence index 与 retrieval-scope SFT 数据

- 新增脚本：
  - `scripts/build_evidence_index_v0_2.py`
  - `scripts/migrate_agentbench_v0_1_to_v0_2.py`
- 更新说明：
  - `AGENTS.md`：同步 v0.2 主线、工具定义、数据路径和不再使用的 v0.1 工具。
- 新建 evidence index：
  - `/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_2_20260530_1914`
  - 输入新权威语料：`/root/datasets/chinese_landscape_authority_corpus`
  - 解析 165 个 PDF、3974 页、生成 10706 个 corpus chunks。
  - 导入旧 Milvus chunks 4703 个，生成 `legacy_chunk_map.json`。
  - 记录 low-text pages 826 个，后续需要 OCR/VLM fallback。
- 新建 v0.2 dataset：
  - `/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_2_retrieval_scope_20260530_1922`
  - 416 tasks，split：train 282 / val 62 / test 72。
  - SFT rows：train 4178 / val 917 / test 1063。
  - 轨迹动作从 `search_evidence/open_chunk` 改为 `retrieve_evidence/open_evidence`。
  - `retrieve_evidence` scope 覆盖：`current_page / nearby_pages / same_document / corpus`。
- 质量评测：
  - JSONL 完整性检查通过。
  - old `chunk_xxx` 到 new `ev_xxx` gold evidence 映射：3010 / 3010，presence 1.0000。
  - claim evidence coverage：0.9081。
  - retrieval gold hit rate：current_page 0.3389，nearby_pages 0.3798，same_document 0.9952，corpus 0.9928。
- 报告：
  - `docs/03_实验报告/EvidenceGrounded-AgentBench-v0.2证据索引与轨迹SFT数据构建报告_20260530_1924.md`
  - `/root/Workspace/VLM/项目文档/03_实验与训练报告/EvidenceGrounded-AgentBench-v0.2证据索引与轨迹SFT数据构建报告_20260530_1924.md`
- 未解决问题：
  - v0.2 任务仍主要迁移自 v0.1 gold，还不是完全从新权威 corpus 重新生成。
  - 低文本页尚未 OCR/VLM fallback。
  - 旧 Milvus chunks 很多仍缺 page-level citation。
- 下一步计划：
  - 做 `low_text_pages.jsonl` 的 OCR/VLM fallback。
  - 从新权威 corpus 重新生成更多 image candidate tasks。
  - 实现 v0.2 tool-call environment 和 verifier。
  - 训练 trajectory SFT adapter，再做小规模 on-policy GRPO。

## 2026-05-30 20:45 CST v0.2 单条 SFT 轨迹样例

- 新增样例文档：
  - `docs/03_实验报告/v0.2单条SFT轨迹样例_20260530_2045.md`
  - `/root/Workspace/VLM/项目文档/03_实验与训练报告/v0.2单条SFT轨迹样例_20260530_2045.md`
- 样例任务：
  - `task_id=egva_v0_2_scope_000000`
  - PDF：`/root/Workspace/ChineseLandscape/data/raw_pdfs/中国传统山水画的审美价值.pdf`
  - 页码：第 5 页。
  - 轨迹长度：14 步。
- 文档内容：
  - 展示 PDF 页面图、带框 overlay、裁剪图。
  - 展示 gold claims。
  - 逐步展示 `crop_image -> retrieve_evidence(current_page/nearby_pages/same_document/corpus) -> open_evidence -> write_claim/abstain_claim -> finish`。
  - 解释一条完整 trajectory 会被拆成多条 step-level SFT rows。

## 2026-05-30 21:05 CST v0.2 样例改为真实 messages 输入输出

- 更新同一份样例文档：
  - 换用 `task_id=egva_v0_2_scope_000005`。
  - 图片 assets 同步替换为 `唐、五代、宋山水画比较：从景胜到意胜.pdf` 第 2 页。
  - 每一步改为展示真实 `messages[0]` 输入和真实 `messages[1]` 监督输出。
  - 补充说明 `snippet`、`chunk` 分片、`authority_level`、`citation_level`、`score` 的来源和当前缺陷。
- 结论：
  - 当前 v0.2 seed 可用于理解 trajectory SFT 形态。
  - snippet 机械截断、chunk 字符窗口分片、legacy chunk 页码缺失仍是 v0.3 前必须修正的问题。

## 2026-05-30 21:18 CST v0.2 样例补充逐步输入图像

- 更新同一份样例文档：
  - 每个 step 增加“真实输入图像”。
  - step 0 显示页面图。
  - step 1-14 显示页面图 + 裁剪图。
- 验证：
  - 15 个 step 均有“真实输入图像”段落。
  - 页面图引用 15 次，裁剪图引用 14 次。

## 2026-05-30 21:53 CST v0.3 句子分片与 snippet 修复重建

- 新增脚本：
  - `scripts/build_evidence_index_v0_3.py`
  - `scripts/migrate_agentbench_v0_1_to_v0_3.py`
- 更新：
  - `AGENTS.md` 增加 v0.3 数据路径和 snippet/tool 返回规范。
- 新建 evidence index：
  - `/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_20260530_2149`
  - 165 个 PDF，3974 页，11294 个 corpus chunks。
  - `sentence_paragraph_aware` chunks：6591。
  - `legacy_milvus_preserved` chunks：4703。
  - display snippet rows：11294 / 11294。
  - evidence summary rows：11294 / 11294。
- 新建 v0.3 dataset：
  - `/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_3_sentence_snippet_20260530_2150`
  - 416 tasks，split：train 282 / val 62 / test 72。
  - SFT rows：train 4178 / val 917 / test 1063。
- 质量评测：
  - JSONL 完整性检查通过。
  - evidence ID 映射：3010 / 3010，未映射 0。
  - snippet coverage：1.0000。
  - sentence boundary ok rate：0.8457。
  - `retrieve_evidence` 返回已改为 `display_snippet/evidence_summary`，不再使用旧 `snippet=text[:260]`。
- 报告：
  - `docs/03_实验报告/EvidenceGrounded-AgentBench-v0.3句子分片与Snippet修复重建报告_20260530_2153.md`
  - `/root/Workspace/VLM/项目文档/03_实验与训练报告/EvidenceGrounded-AgentBench-v0.3句子分片与Snippet修复重建报告_20260530_2153.md`

## 2026-05-30 23:08 CST LowTextPages VLM Fallback 小批量试跑

- 新增脚本：
  - `scripts/build_low_text_vlm_fallback.py`
  - `scripts/merge_low_text_fallback_into_index_v0_3_1.py`
- 背景：
  - v0.3 index 仍有 826 个 `low_text_pages`，主要是扫描页、影印古籍页、图版页或空白页。
  - 这些页没有可靠 PDF text layer，会影响 agent 在 `current_page/nearby_pages` 范围内取证。
- 模型接口结论：
  - `qwen3.7-max-2026-05-20` 响应慢，本轮未作为主力。
  - `qwen3.7-max-2026-05-17` 返回 JSON，但多次表示未接收到图像。
  - `qwen3.6-flash-2026-04-16` 使用 `image_url` 格式可正常接收页面图像并转写。
  - `qwen3.6-plus-2026-04-02` 可作为 fallback。
- 小批量 fallback 输出：
  - `/root/datasets/evidence_grounded_vlm_agentrl/low_text_vlm_fallback_v0_2_qwen36flash_smoke_20260530_2257`
  - selected_pages：12，processed_pages：12，readable_pages：12。
  - avg_chars_after：273.75。
  - fallback_model_counts：`qwen3.6-flash-2026-04-16` 11 页，`qwen3.6-plus-2026-04-02` 1 页。
- v0.3.1 smoke index：
  - `/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_1_low_text_vlm_smoke_20260530_2308`
  - 追加 `vlm_ocr_fallback` page/document/corpus rows 各 12 条。
  - 新增 citation level：`page_image_transcription`，表示整页图像转写证据，不提供 bbox 级引用。
- 验证：
  - `python3 -m py_compile scripts/build_low_text_vlm_fallback.py scripts/merge_low_text_fallback_into_index_v0_3_1.py`
  - 合并后 `page_spans.jsonl`：69727 -> 69739。
  - 合并后 `corpus_chunks.jsonl`：11294 -> 11306。
- 报告：
  - `docs/03_实验报告/LowTextPages-VLMFallback与v0.3.1合并试跑报告_20260530_2308.md`
  - `/root/Workspace/VLM/项目文档/03_实验与训练报告/LowTextPages-VLMFallback与v0.3.1合并试跑报告_20260530_2308.md`
- 下一步：
  - 用 `qwen3.6-flash-2026-04-16` 主跑、`qwen3.6-plus-2026-04-02` 兜底，全量处理 826 个 low_text_pages。
  - 对 `scan_text / blank / image_plate / unknown` 做质量分层，只把可读扫描文本合并为正式 v0.3.1 full index。
  - 基于 v0.3.1 full index 重建 SFT 数据。

## 2026-05-31 02:51 CST LowTextPages 全量 VLM Fallback 与 v0.3.1 SFT 重建

- 全量 fallback：
  - `/root/datasets/evidence_grounded_vlm_agentrl/low_text_vlm_fallback_v0_3_1_full_qwen36flash_20260530_2312`
  - 826 processed，810 ok，651 readable，457 pages_with_40plus_chars。
  - 主模型 `qwen3.6-flash-2026-04-16`，兜底 `qwen3.6-plus-2026-04-02`。
- v0.3.1 full evidence index：
  - `/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_1_low_text_vlm_full_20260531_0140`
  - 新增 page/document/corpus evidence rows 各 457 条。
  - 标记为 `source_quality=vlm_ocr_fallback`、`citation_level=page_image_transcription`、`quality.silver_text=true`。
- v0.3.1 SFT 数据：
  - `/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_3_1_low_text_vlm_full_sft_20260531_0248`
  - 416 tasks，train/val/test tasks = 282/62/72。
  - SFT rows：train 4178，val 917，test 1063。
  - retrieval nonempty：current_page 0.9976，nearby/same_document/corpus 均为 1.0。
- 报告：
  - `docs/03_实验报告/LowTextPages-VLMFallback全量与v0.3.1SFT重建报告_20260531_0251.md`
  - `/root/Workspace/VLM/项目文档/03_实验与训练报告/LowTextPages-VLMFallback全量与v0.3.1SFT重建报告_20260531_0251.md`

## 2026-05-31 06:45 CST EvidenceGrounded trajectory SFT 第一轮训练链路完成

### 完成内容

- 新增训练脚本：
  - `scripts/train_trajectory_sft_lora.py`
- 新增生成评测脚本：
  - `scripts/eval_trajectory_sft_actions.py`
- 新增 highlighted SFT 构建脚本：
  - `scripts/build_highlighted_sft_dataset.py`
- 构建 template-highlighted SFT 数据：
  - `/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_3_3_template_highlighted_sft_20260531_0504`
  - 416 个 task 全部完成 crop-to-page 模板定位。
  - template score：min 0.779，mean 0.968，max 0.999。
- 输出完整报告：
  - `docs/03_实验报告/EvidenceGrounded-SFT训练链路报告_20260531_0645.md`
  - `/root/Workspace/VLM/项目文档/03_实验与训练报告/EvidenceGrounded-SFT训练链路报告_20260531_0645.md`

### SFT 训练记录

| 阶段 | 输出 | 训练数据 | 结论 |
|---|---|---|---|
| base eval | `outputs/evidence_sft_base_eval_val12_smoke_20260531_0315` | 无训练 | 原始 Qwen2.5-VL-3B 不会稳定输出本项目 tool schema |
| smoke12 | `outputs/evidence_sft_qwen25vl3b_lora_compact_smoke12_20260531_0343/adapter` | 12 compact rows | 学到 JSON/action 格式，但 action type 只有 0.25 |
| v1 compact600 | `outputs/evidence_sft_qwen25vl3b_lora_compact_v1_600_20260531_0346/adapter` | v0.3.1 compact 600 rows | val120 action_type_acc 0.792 |
| v2 highlight360 | `outputs/evidence_sft_qwen25vl3b_lora_compact_v2_highlight360_20260531_0510/adapter` | v0.3.3 highlighted 360 rows | 当前保留版本 |
| v3 cropfix | `outputs/evidence_sft_qwen25vl3b_lora_compact_v3_cropfix_20260531_0553/adapter` | crop-only 282 rows | 不升级，未修好 bbox 且拉低部分非 crop 指标 |

### 当前保留 adapter

```text
/root/Workspace/VLM/EvidenceGrounded-VLM-AgentRL/outputs/evidence_sft_qwen25vl3b_lora_compact_v2_highlight360_20260531_0510/adapter
```

### 关键评测

v2 val120：

- valid_action_rate：1.000
- action_type_acc：0.825
- evidence_overlap_rate：0.817
- scope_acc_on_retrieve：0.750

v2 test120：

- valid_action_rate：0.992
- action_type_acc：0.858
- evidence_overlap_rate：0.858
- scope_acc_on_retrieve：0.900

### 未解决问题

- `crop_image` 类型判断已经稳定，但像素 bbox 仍不可靠，`bbox_iou@0.5` 基本为 0。
- 继续用 Qwen2.5-VL-3B 做 crop-only SFT 没有解决该问题。
- 后续应把 crop 改成红框检测/模板定位工具，或使用更强 VLM 专门做 grounding 对照。

### 下一步

- 固定 v2 adapter 作为 EvidenceGrounded 当前 SFT 起点。
- 实现可执行 tool-call environment 和 verifier。
- 用 v2 adapter 采集 on-policy rollouts，再做 verifier-guided GRPO。

## 2026-05-31 06:53 CST v0.3.3 完整 SFT 轨迹真实样例

### 完成内容

- 随机选取真实 val 任务：
  - `task_id=egva_v0_3_scope_000238`
  - PDF：`北宋山水画点景建筑布局分析与应用研究.pdf`
  - 页码：第 36 页
  - 轨迹长度：14 步
- 生成完整样例报告：
  - `docs/03_实验报告/v0.3.3完整SFT轨迹真实样例_20260531_0653.md`
  - `/root/Workspace/VLM/项目文档/03_实验与训练报告/v0.3.3完整SFT轨迹真实样例_20260531_0653.md`

### 报告内容

- 展示 v0.3.3 如何从 v0.3.1 原 bbox `[181, 468, 827, 614]` 经 crop-to-page 模板匹配修正为 page bbox `[224, 822, 1025, 1076]`。
- 展示带红框 page image 和 step 0 后的 crop image。
- 对 14 个 step 逐步展示：
  - 真实输入图像；
  - 当前 compact prompt；
  - 监督目标 JSON action。
