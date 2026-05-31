# EvidenceGrounded-AgentBench v0.1 VLM 审核构建报告

- 生成时间：2026-05-30 18:12 CST
- 数据输出目录：`/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_1_vlm_audited_flash_full_20260530_1641`
- 规则候选输入：`/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_1_local_evidence_20260530_1625/tasks_all.jsonl`
- 审核脚本：`scripts/audit_agentbench_with_vlm.py`

## 本次修正

- 当前构建过程已经使用 VLM，不再只是规则抽取。
- 脚本已支持 `qwen3.7-max-2026-05-20`、`qwen3.7-max-2026-05-17`、`qwen3.7-max`。
- 关键适配点：`qwen3.7-max*` 在 DashScope OpenAI-compatible 入口下不接受 `{"type":"image_url"}`，但接受 `{"type":"image","image":"data:image/jpeg;base64,..."}`；脚本的 `--dashscope-image-format auto` 会自动为 qwen3.7 使用该格式。
- JSON 解析失败现在也会触发 fallback，不再只在 API 报错时 fallback。

## 模型使用情况

最终 416 条 clean task 中的模型分布：

- `qwen3.7-max-2026-05-17`：133
- `qwen3.7-max-2026-05-20`：93
- `qwen3.6-flash-2026-04-16`：73
- `qwen3.6-35b-a3b`：78
- `kimi-k2.6`：32
- `qwen3.6-plus-2026-04-02`：4
- `qwen3.7-max`：2
- `/root/models/Qwen3-VL-4B-Instruct`：1

其中 `qwen3.7-max-2026-05-17` 和 `qwen3.7-max` 都已被实际触发，说明“05-20 额度/格式/输出不稳定时切到 05-17，再切到 max”的链路可用。

## 数据规模

- 输入任务：500
- VLM 标记无关并移除：84
- API/JSON 失败并移除：0
- 最终 clean tasks：416
- split：train 282 / val 62 / test 72
- unique sources：19

## Claim 与证据质量

- claims：2912
- non-abstain claims：2101
- claims_with_chunk_evidence：1908
- claim_chunk_evidence_coverage：0.9081
- `vlm_visual_or_ocr_support`：193
- `vlm_selected_chunk_support`：1908

字段覆盖：

- caption_text：399
- title：329
- artist：301
- dynasty：309
- visual_elements：416
- technique：265
- composition：197

## 轨迹 SFT 输出

- train SFT rows：3664
- val SFT rows：806
- test SFT rows：936
- review HTML：`/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_1_vlm_audited_flash_full_20260530_1641/review/review.html`

## 当前限制

- 仍是自动 VLM audit，不是人工金标；val/test 进入正式 benchmark 前建议抽样复查。
- citation 仍是 chunk-level，因为 legacy evidence store 中很多 `page_start/page_end` 为空。
- 13 条最终样本来自 `dashscope_text_only` fallback，建议后续用 vision-capable 模型复审这部分。
