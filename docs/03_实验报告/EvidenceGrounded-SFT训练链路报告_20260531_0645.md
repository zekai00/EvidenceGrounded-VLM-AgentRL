# EvidenceGrounded SFT 训练链路报告

生成时间：2026-05-31 06:45 CST

## 结论先行

当前保留的 EvidenceGrounded 第一版 SFT adapter：

```text
/root/Workspace/VLM/EvidenceGrounded-VLM-AgentRL/outputs/evidence_sft_qwen25vl3b_lora_compact_v2_highlight360_20260531_0510/adapter
```

它不是最终完整 agent，但已经达到“可用的 tool-call trajectory SFT 起点”：

| 指标 | val120 | test120 |
|---|---:|---:|
| valid_json_rate | 1.000 | 0.992 |
| valid_action_rate | 1.000 | 0.992 |
| action_type_acc | 0.825 | 0.858 |
| exact_action_acc | 0.325 | 0.283 |
| field_acc | 0.592 | 0.583 |
| evidence_overlap_rate | 0.817 | 0.858 |
| scope_acc_on_retrieve | 0.750 | 0.900 |

未解决问题：`crop_image` 的动作类型能学会，但 page-level 精确 bbox 仍不合格，`bbox_iou@0.5` 基本为 0。后续不建议继续用 3B 单纯 SFT 硬学像素框，应该改成检测/模板工具辅助，或用更强 VLM 专门做 grounding。

## 数据链路

原始 SFT 数据：

```text
/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_3_1_low_text_vlm_full_sft_20260531_0248
```

规模：

| split | tasks | SFT rows |
|---|---:|---:|
| train | 282 | 4178 |
| val | 62 | 917 |
| test | 72 | 1063 |

训练中发现 v0.3.1 有两个实际问题：

1. 原始 prompt 太长，后半段 step 可到 12k token 以上，直接 14k 训练会 OOM。
2. `crop_image.bbox` 不稳定等同于 page PNG 像素坐标；直接画框会框到正文上。

因此新增 compact prompt 和 template-highlighted 数据：

```text
/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_3_3_template_highlighted_sft_20260531_0504
```

v0.3.3 做了：

- 用 crop 图在 page PNG 上模板匹配，反推真实页面像素 bbox。
- 在 page image 上绘制红色矩形。
- 同步修正 `crop_image.bbox` 监督。
- 416 个 task 全部匹配成功；template score：min 0.779，mean 0.968，max 0.999。

中间废弃版本：

```text
/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_3_2_highlighted_sft_20260531_0448
```

该版本只按原 action bbox 画框，没有做 template correction；抽查发现红框落到正文区域，因此不用于训练结论。

## 脚本改动

新增：

```text
scripts/train_trajectory_sft_lora.py
scripts/eval_trajectory_sft_actions.py
scripts/build_highlighted_sft_dataset.py
```

关键能力：

- compact prompt：从结构化 `history/tool_results/draft_claims/images/action` 重建短 prompt，避免把长 OCR/检索返回完整塞入每一步。
- LoRA SFT：Qwen2.5-VL-3B-Instruct，4bit 加载，LoRA r=16，alpha=32。
- 生成评测：统计 JSON 合法率、action 合法率、动作类型准确率、字段准确率、证据 overlap、retrieve scope、crop IoU。
- highlighted SFT 构建：用模板匹配修正 page bbox 并绘制红框。

## 训练与评测记录

### Base 评测

模型：

```text
/root/models/Qwen2.5-VL-3B-Instruct
```

输出：

```text
outputs/evidence_sft_base_eval_val12_smoke_20260531_0315
```

结果：12 条 val smoke 上 `valid_action_rate=0.0`、`action_type_acc=0.0`。原始 instruct 模型倾向输出普通 claim JSON 或自然语言，不会遵循本项目工具 schema。

### Smoke SFT 12

训练输出：

```text
outputs/evidence_sft_qwen25vl3b_lora_compact_smoke12_20260531_0343/adapter
```

结果：val12 上 `valid_action_rate=1.0`，但 `action_type_acc=0.25`，明显只学到格式，未学到状态决策。

### v1 Compact 600

训练：

```text
outputs/evidence_sft_qwen25vl3b_lora_compact_v1_600_20260531_0346/adapter
```

配置：

- train rows：600，按 action 均衡采样。
- optimizer steps：75。
- final val loss：0.4867。

val120：

| 指标 | 数值 |
|---|---:|
| valid_action_rate | 0.992 |
| action_type_acc | 0.792 |
| evidence_overlap_rate | 0.783 |
| scope_acc_on_retrieve | 0.200 |

结论：可用性明显提升，但 retrieve scope 判断弱，crop bbox 仍不行。

### v2 Highlight 360

训练：

```text
outputs/evidence_sft_qwen25vl3b_lora_compact_v2_highlight360_20260531_0510/adapter
```

起点：

```text
outputs/evidence_sft_qwen25vl3b_lora_compact_v1_600_20260531_0346/adapter
```

数据：

```text
/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_3_3_template_highlighted_sft_20260531_0504
```

配置：

- train rows：360，按 action 均衡采样。
- optimizer steps：45。
- final val loss：0.4799。

val120：

| 指标 | 数值 |
|---|---:|
| valid_action_rate | 1.000 |
| action_type_acc | 0.825 |
| exact_action_acc | 0.325 |
| field_acc | 0.592 |
| evidence_overlap_rate | 0.817 |
| scope_acc_on_retrieve | 0.750 |

test120：

| 指标 | 数值 |
|---|---:|
| valid_action_rate | 0.992 |
| action_type_acc | 0.858 |
| exact_action_acc | 0.283 |
| field_acc | 0.583 |
| evidence_overlap_rate | 0.858 |
| scope_acc_on_retrieve | 0.900 |

结论：v2 是当前最佳 SFT adapter，保留为 current SFT。

### v3 Crop Fix

训练：

```text
outputs/evidence_sft_qwen25vl3b_lora_compact_v3_cropfix_20260531_0553/adapter
```

配置：

- 起点：v2 adapter。
- 只训练 `crop_image` rows，共 282 条。
- optimizer steps：36。
- final val loss：0.4768。

val120：

| 指标 | 数值 |
|---|---:|
| valid_action_rate | 1.000 |
| action_type_acc | 0.800 |
| field_acc | 0.542 |
| evidence_overlap_rate | 0.800 |
| scope_acc_on_retrieve | 0.550 |
| bbox_iou@0.5 on crop | 0.000 |

结论：v3 不升级。它没有修好 crop bbox，并且拉低了 `write_claim` 与 `retrieve_evidence`。

## 当前模型能力

当前 v2 SFT adapter 能做到：

- 稳定输出合法 JSON 工具调用。
- 大体判断下一步该调用哪个工具。
- `finish`、`open_evidence` 较稳。
- `retrieve_evidence` 的 scope 在 test120 上达到 0.90。
- `write_claim` 可写对一部分字段和 evidence，但仍会在证据不足/可写之间摇摆。

当前不能可靠做到：

- 精确输出 page PNG 像素 bbox。
- 对 `visual_elements/technique/composition` 这类需要综合视觉和文本的 claim 做高准确结构化生成。
- 替代完整 on-policy RL；它只是 RL 前的 trajectory SFT 起点。

## 当前合格判定

我把当前 v2 视为“合格的 SFT 起点”，原因是：

- schema 层已稳定：val/test `valid_action_rate` 约 0.99-1.00。
- step-level action type 已超过 0.80 左右：val 0.825，test 0.858。
- 检索决策有明显可学性：test `scope_acc_on_retrieve=0.90`。
- v3 证明继续用 3B SFT 强行修 crop bbox 收益不佳。

但它不应被描述为最终 agent。下一步应先补环境执行与 verifier，再做 on-policy RL 或工具辅助 grounding。

## 下一步

1. 固定 v2 为 EvidenceGrounded 当前 SFT adapter。
2. 实现可执行 tool-call environment：
   - `crop_image`
   - `retrieve_evidence`
   - `open_evidence`
   - `write_claim`
   - `abstain_claim`
   - `finish`
3. 对 crop 不再让 3B 直接猜 bbox：
   - 优先用红框检测/模板定位工具输出候选 bbox；
   - 或用 Qwen2.5-VL-7B / Qwen3-VL-8B 做 grounding 对照。
4. 用 verifier 评估完整 trajectory：
   - tool success
   - claim F1
   - evidence hit / MRR
   - unsupported claim rate
   - abstain accuracy
5. 通过 SFT adapter 采集小规模 on-policy rollouts，再进入 verifier-guided GRPO。
