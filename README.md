# EvidenceGrounded-VLM-AgentRL

证据约束的多模态主动取证 VLM Agent RL 项目。

本项目的核心任务是：

```text
Claim-Level Evidence-Seeking VLM Agent for multimodal document understanding
```

给定 PDF 页面图像、局部裁剪图、OCR/PDF text、retrieval evidence store，模型需要在有限工具调用预算内主动取证，为结构化 claim 找到可追溯证据，并判断哪些字段应该回答、哪些字段应该 abstain。

中国山水画资料是第一阶段验证场景。项目重点不是“山水画资料整理工具”，而是：

- 多模态文档理解；
- VLM tool-call trajectory SFT；
- verifier-guided on-policy RL；
- claim-level evidence grounding；
- hallucination / unsupported claim 控制；
- 可复现 evidence store 与可评测 agent environment。

## 当前状态

当前已完成：

- v0.3.1 evidence index：低文本页通过 VLM fallback 补充弱证据。
- v0.3.3 highlighted SFT：通过 crop-to-page 模板匹配修正目标框，并在 page image 上绘制红框。
- Qwen2.5-VL-3B LoRA trajectory SFT：得到第一版可用 tool-call adapter。

当前保留的 SFT adapter：

```text
outputs/evidence_sft_qwen25vl3b_lora_compact_v2_highlight360_20260531_0510/adapter
```

注意：`outputs/` 是本地实验产物，不提交 GitHub。需要复现实验时按报告中的命令重新生成，或单独发布到模型仓库。

## 关键结果

v2 SFT adapter 在 v0.3.3 compact prompt 下的生成式 next-action 评测：

| split | rows | valid_action_rate | action_type_acc | evidence_overlap_rate | scope_acc_on_retrieve |
|---|---:|---:|---:|---:|---:|
| val | 120 | 1.000 | 0.825 | 0.817 | 0.750 |
| test | 120 | 0.992 | 0.858 | 0.858 | 0.900 |

当前主要未解决问题：

- `crop_image` 的动作类型能学会，但 Qwen2.5-VL-3B 直接输出精确 page 像素 bbox 仍不可靠。
- 后续应把 crop 改成红框检测/模板定位工具，或使用更强 VLM 做 grounding 对照。

## 数据位置

所有大数据、PDF、页面图、训练集和实验生成物放在：

```text
/root/datasets/evidence_grounded_vlm_agentrl/
```

当前主要数据版本：

```text
/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_1_low_text_vlm_full_20260531_0140
/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_3_3_template_highlighted_sft_20260531_0504
```

当前可接入的 legacy evidence store 来自：

```text
/root/Workspace/ChineseLandscape/data/processed/documents
```

其中 `chunks.jsonl` 含 4703 个 legacy Milvus 迁移证据块。

## 代码结构

```text
configs/                  配置文件
docs/01_规划与路线/        项目规划
docs/02_指标与数据/        指标和数据说明
docs/03_实验报告/          实验报告与样例
scripts/                  数据构建、SFT 训练、评测脚本
src/                      后续 environment / verifier / agent 代码
```

## 主要脚本

```text
scripts/build_evidence_index_v0_3.py
scripts/build_low_text_vlm_fallback.py
scripts/merge_low_text_fallback_into_index_v0_3_1.py
scripts/build_highlighted_sft_dataset.py
scripts/train_trajectory_sft_lora.py
scripts/eval_trajectory_sft_actions.py
```

## 重要报告

```text
docs/03_实验报告/EvidenceGrounded-SFT训练链路报告_20260531_0645.md
docs/03_实验报告/v0.3.3完整SFT轨迹真实样例_20260531_0653.md
docs/03_实验报告/LowTextPages-VLMFallback全量与v0.3.1SFT重建报告_20260531_0251.md
```

## 旧项目关系

旧目录：

```text
/root/Workspace/VLM/ChineseLandscape-AgentRL
```

保留为山水画 v0.1 seed 和历史实验目录。新目录承接算法叙事和后续开发。

`EviTool-VL` 是另一个独立项目，主线是 Browser/GUI tool-call RL；本项目不应和 `EviTool-VL` 共用同一个 GitHub 仓库。
