# 可执行 Tool-Call Environment、Crop 方案与 GRPO 对比规划

生成时间：2026-05-31 12:35

本文回答三个问题：

1. 如何把 EvidenceGrounded-VLM-AgentRL 做成可执行 tool-call environment。
2. 不再让 3B VLM 直接猜 `crop_image` 的 bbox 后，应该怎样做 crop。
3. 如何用 verifier 评估完整 trajectory，并对比 clipped-ratio GRPO 和 verifier-guided GRPO。

## 1. 可执行 Tool-Call Environment 是什么

Tool-call environment 指一个可交互环境。模型不是一次性回答最终答案，而是每一步输出一个 JSON action，环境执行工具，再把新的观察结果返回给模型。

最小接口应该是：

```python
obs = env.reset(task_id)
obs, reward, terminated, info = env.step(action_json)
```

其中：

- `reset`：重置任务，加载某个 PDF 页面、目标任务、证据索引、当前草稿。
- `step`：接收模型输出的工具调用 JSON，执行工具，并返回下一步观察。
- `obs`：observation，模型在当前步能看到的输入，包括图像、任务描述、历史动作、工具返回、当前草稿 claim。
- `reward`：verifier 给出的当前奖励或最终奖励。
- `terminated`：任务是否结束。
- `info`：调试信息，例如 action 是否合法、命中的 evidence id、verifier 明细。

### 1.1 模型每一步的输入

每一步输入应包含四类内容：

```json
{
  "goal": "请为当前页面中的目标图像建立可追溯证据卡。",
  "images": [
    {"role": "page_image", "path": ".../pages/doc_p036.png"},
    {"role": "last_crop", "path": ".../crops/region_03.png"}
  ],
  "state_text": {
    "history": [
      {"action": "propose_regions", "args": {"source": "all", "top_k": 8}},
      {"action": "crop_region", "args": {"region_id": "r03"}}
    ],
    "tool_results": [
      {"tool": "propose_regions", "regions": [{"region_id": "r03", "bbox": [181, 468, 827, 614]}]},
      {"tool": "crop_region", "crop_path": ".../crops/region_03.png"}
    ],
    "draft_claims": [
      {"field": "subject", "value": null, "evidence_ids": []}
    ]
  }
}
```

注意：最终可执行环境不应依赖“预先画好红框”的图片。红框只能作为某些 SFT 阶段的辅助数据，不能作为真实 benchmark 的默认输入。

### 1.2 模型每一步的输出

模型每一步只输出一个工具调用 JSON。例如：

```json
{"action": "retrieve_evidence", "query": "北宋 山水画 点景建筑 布局", "scope": "same_document", "top_k": 5}
```

或者：

```json
{"action": "write_claim", "field": "composition", "value": "山水画中的建筑常作为点景元素组织空间层次", "evidence_ids": ["ev_000123", "ev_000987"], "confidence": 0.78}
```

### 1.3 第一版工具集合

建议把工具分成四组。

视觉定位工具：

- `propose_regions(source, top_k)`：从当前 PDF 页面自动提出候选区域，返回 `region_id + bbox + thumbnail + reason`。
- `crop_region(region_id)`：裁剪某个候选区域。模型选择 `region_id`，不直接猜像素坐标。
- `ocr_region(region_id)`：对候选区域做 OCR。
- `inspect_page()`：返回页面尺寸、PDF text layer、OCR 概要、已有图像区域。

证据检索工具：

- `retrieve_evidence(query, scope, anchor, top_k)`：检索证据。`scope` 可取 `current_page`、`nearby_pages`、`same_document`、`corpus`。
- `open_evidence(evidence_id)`：打开完整证据块。

结构化写入工具：

- `write_claim(field, value, evidence_ids, visual_region_id, confidence)`：写入有证据支撑的 claim。
- `abstain_claim(field, reason)`：证据不足时主动放弃该字段。

控制工具：

- `finish()`：结束任务。

### 1.4 可执行环境目录建议

建议新增：

```text
src/evidence_agent_env/
  env.py                 # reset/step 主环境
  actions.py             # action schema 和 JSON 校验
  tools/
    region_proposal.py   # propose_regions
    crop.py              # crop_region
    ocr.py               # ocr_region
    retrieval.py         # retrieve_evidence/open_evidence
  verifier.py            # trajectory verifier
  rollout.py             # 批量采样 trajectory
```

对应脚本：

```text
scripts/run_env_smoke.py
scripts/collect_rollouts.py
scripts/eval_trajectory_verifier.py
scripts/train_grpo_trajectory.py
```

## 2. Crop 不再让 3B 直接猜 bbox，该怎么做

当前 v0.3.3 的做法是：已知目标 crop 图，把 crop 图通过模板匹配定位回 page image，然后在 page image 上画红色矩形，并修正 `crop_image.bbox` 监督。

这不是最终方案。它只是为了解决 v0.3.1 里第一步 `crop_image` 欠约束的问题：原始页面上没有任何提示，3B VLM 很难直接知道该裁剪哪一块。

### 2.1 当前是不是把所有红框都画在图片上

不是“把所有候选框都画上去”。当前脚本对每个 task 生成一张高亮页面，在该 task 对应的目标区域上画一个红框。

这意味着：

- 对同一 PDF 页的不同 task，可能会生成不同的高亮页面副本。
- 红框是 SFT 辅助信号，不是最终环境里的真实输入。
- 如果真实没见过的文献没有红框，最终模型不能依赖红框完成任务。

因此，v0.3.3 highlighted SFT 可以保留为过渡训练集，但不能作为最终 benchmark 主设定。

### 2.2 最终 crop 应改成“候选区域工具 + 模型选择”

最终不应该要求 3B VLM 直接输出：

```json
{"action": "crop_image", "bbox": [181, 468, 827, 614]}
```

而应该改成两步：

第一步，环境自动产生候选区域：

```json
{"action": "propose_regions", "source": "all", "top_k": 8}
```

工具返回：

```json
{
  "tool": "propose_regions",
  "regions": [
    {
      "region_id": "r01",
      "bbox": [82, 112, 909, 356],
      "type": "text_block",
      "reason": "large OCR text block"
    },
    {
      "region_id": "r02",
      "bbox": [181, 468, 827, 614],
      "type": "figure_candidate",
      "reason": "embedded image-like region"
    }
  ]
}
```

第二步，模型选择候选区域：

```json
{"action": "crop_region", "region_id": "r02"}
```

这样 3B VLM 学的是“在候选区域中选择和任务目标最相关的区域”，而不是学习不稳定的像素级 bbox 回归。

### 2.3 候选区域从哪里来

候选区域可以由本地工具在真实未见文献上即时生成，不需要预先红框。

候选来源包括：

1. PDF embedded image extraction  
   用 PyMuPDF 等工具读取 PDF 页面里的图片对象和所在位置。适合可解析 PDF。

2. Layout segmentation  
   对页面图像做版面分析，找出图片区、标题区、正文区、图注区、表格区。可以先用 OpenCV 连通域、投影、边缘密度等规则实现第一版，再接入 layout detector。

3. OCR text anchors  
   OCR 识别 `图`、`图版`、`Fig.`、`Plate`、作品名、作者名等文本锚点，生成附近区域候选。

4. VLM proposal audit  
   用 Qwen3.7-Max 或本地 Qwen3-VL 对困难页面生成弱候选或审核候选，但它应作为数据构建和质检工具，不应成为训练后模型运行时的必要依赖。

### 2.4 为什么这比直接 bbox 更合理

原因有三点：

1. 真实工具调用 agent 更常见的是选择对象 ID、证据 ID、区域 ID，而不是裸猜像素。
2. 3B VLM 的空间精度有限，直接 bbox 会把主要误差浪费在坐标回归上。
3. 候选区域工具可以在真实未见文献上运行，因此不会依赖训练集红框。

### 2.5 v0.4 数据集应如何调整

v0.4 SFT trajectory 不再以 `crop_image(bbox)` 开头，而改成：

```text
Step 1: propose_regions(source="all", top_k=8)
Step 2: crop_region(region_id="r02")
Step 3: retrieve_evidence(...)
Step 4: open_evidence(...)
Step 5: write_claim(...)
Step 6: finish()
```

训练样本里保留候选区域列表，让模型学会基于页面图像、缩略图、OCR、任务目标选择正确 `region_id`。

## 3. 用 verifier 评估完整 trajectory

Verifier 是自动评估器。它不训练模型，而是给模型输出的完整轨迹打分。

完整 trajectory 包含：

```text
obs_0, action_0, tool_result_0,
obs_1, action_1, tool_result_1,
...
obs_T, action_T
```

评估时不只看最终答案，还看：

- 工具调用是否合法；
- 是否找到了正确 evidence；
- claim 是否被 evidence 支撑；
- crop/candidate 是否命中目标区域；
- 是否在证据不足时 abstain；
- 是否用过多无效工具调用；
- 是否提前 finish 或超步数。

### 3.1 推荐 reward 设计

完整轨迹奖励：

```text
R = w_s * R_success
  + w_c * R_claim
  + w_e * R_evidence
  + w_g * R_grounding
  + w_a * R_abstain
  - w_i * P_invalid
  - w_l * P_length
```

各项含义：

- `R_success`：最终任务是否成功。
- `R_claim`：结构化 claim 的 micro/macro F1。
- `R_evidence`：引用 evidence 的 hit@k、MRR、overlap。
- `R_grounding`：视觉区域是否命中目标区域，例如 region hit 或 IoU。
- `R_abstain`：证据不足字段是否正确 abstain。
- `P_invalid`：非法 JSON、非法 action、引用不存在 evidence、重复无意义工具调用的惩罚。
- `P_length`：过长轨迹惩罚。

第一版可以使用：

```text
R = 1.0 * success
  + 0.4 * claim_f1
  + 0.4 * evidence_f1
  + 0.2 * grounding_hit
  + 0.2 * abstain_acc
  - 0.1 * invalid_count
  - 0.02 * max(0, tool_calls - 6)
```

### 3.2 Verifier 不是“拍脑袋大模型裁判”

第一版 verifier 应尽量规则化：

- claim 字段和 gold claim 做规范化匹配；
- evidence id 与 gold evidence id / equivalent evidence set 做匹配；
- region id 与 gold region id 做匹配；
- bbox 只作为工具内部信息，评估时可转换成 IoU；
- unsupported claim 用引用证据是否覆盖来判断。

强模型可以用于数据构建和人工质检辅助，但最终 benchmark 的主 verifier 应可复现、可离线运行。

## 4. 对比真正 GRPO 和 verifier-guided GRPO

需要对比。否则很难证明我们的方法选择是合理的。

这里有两个概念：

- GRPO：Group Relative Policy Optimization，同一任务采样多条输出，用组内 reward 标准化得到 advantage，再更新策略。
- clipped ratio：PPO/GRPO 常用的比例裁剪，限制新策略相对旧策略一步变化过大。
- reference KL：让当前策略不要偏离 reference model 太远，reference 通常是 SFT 模型。

### 4.1 方案 A：Verifier-Guided Step-wise GRPO

这是更工程化的方案。

做法：

1. 在某个状态 `s_t` 下采样多个 action。
2. 环境执行 action，verifier 给局部奖励或短 horizon 奖励。
3. 对同一组 action 的 reward 做组内标准化：

```text
A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
```

4. 用 advantage 加权 action logprob：

```text
L = - A_i * log pi_theta(a_i | s_t)
```

优点：

- 样本效率高；
- 不容易因为整条轨迹全失败而没有学习信号；
- 适合早期工具环境还不稳定时使用。

缺点：

- 更像“verifier 引导的局部策略改进”；
- 对长程 credit assignment 支持不够；
- 和标准 trajectory-level RL 有差距。

### 4.2 方案 B：Trajectory-Level GRPO with Clipped Ratio + Reference KL

这是更接近真正 on-policy RL 的方案。

每个 task 采样 `G` 条完整轨迹，例如 `G=4` 或 `G=8`：

```text
tau_i = (s_0, a_0, s_1, a_1, ..., s_T, a_T)
R_i = verifier(tau_i)
```

组内 advantage：

```text
A_i = (R_i - mean(R_1...R_G)) / (std(R_1...R_G) + eps)
```

old-policy ratio：

```text
r_t(theta) = exp(log pi_theta(a_t | s_t) - log pi_old(a_t | s_t))
```

clipped GRPO loss：

```text
L_clip = - mean_t min(
  r_t(theta) * A_i,
  clip(r_t(theta), 1 - epsilon, 1 + epsilon) * A_i
)
```

reference KL：

```text
L_KL = beta * KL(pi_theta(. | s_t) || pi_ref(. | s_t))
```

总损失：

```text
L_total = L_clip + L_KL + lambda_sft * L_sft_replay
```

建议参数：

```text
epsilon = 0.2
beta = 0.01 起步；如果过保守可降到 0.001 或 0
G = 4 起步；稳定后再试 G = 8
max_steps = 8
lambda_sft = 0.05 到 0.2
```

其中：

- `pi_theta`：当前训练中的策略模型。
- `pi_old`：采样轨迹时冻结的旧策略。
- `pi_ref`：reference model，建议用 trajectory SFT adapter。
- `L_sft_replay`：混入少量 SFT 样本的监督损失，防止 RL 把 JSON 格式和基础工具能力冲坏。

### 4.3 应该如何做公平对比

建议固定：

- 同一底模；
- 同一 SFT 起点；
- 同一 train/val/test split；
- 同一最大步数；
- 同一采样预算；
- 同一 verifier；
- 同一随机种子集合。

对比组：

| 组别 | 说明 |
|---|---|
| SFT only | 只用 trajectory SFT，不做 RL |
| Step-wise verifier-guided GRPO | 当前更稳的局部 verifier 引导方案 |
| Trajectory GRPO no clip/no KL | 只有组内 advantage 和 logprob 加权 |
| Trajectory GRPO clipped | 加 old-policy ratio clip |
| Trajectory GRPO clipped + KL | 加 reference KL |
| Trajectory GRPO clipped + KL + SFT replay | 加 SFT replay 保守项 |

主指标：

- trajectory success rate；
- claim F1；
- evidence hit@1 / hit@3 / MRR；
- unsupported claim rate；
- abstain accuracy；
- region hit / visual IoU；
- valid action rate；
- average tool calls；
- timeout rate；
- paired win/loss against SFT baseline。

### 4.4 预期结论

如果环境和 verifier 已经稳定，trajectory-level clipped GRPO 更有说服力，因为它优化的是整条工具调用链。

如果早期 full trajectory 大量失败，step-wise verifier-guided GRPO 更实用，因为它能提供更密集的学习信号。

因此推荐路线是：

1. 先实现可执行环境和离线 verifier。
2. 用 SFT adapter 跑 rollout，得到 SFT baseline。
3. 先做小规模 step-wise verifier-guided GRPO，确认 reward 能推动局部动作变好。
4. 再做 trajectory-level clipped GRPO，对比是否带来完整任务成功率提升。
5. 最终报告中必须同时报告 SFT、step-wise GRPO、trajectory GRPO clipped、trajectory GRPO clipped+KL 的结果。

## 5. 下一步落地顺序

第一步，实现 environment smoke test：

```text
reset -> propose_regions -> crop_region -> retrieve_evidence -> open_evidence -> write_claim -> finish
```

第二步，重建 v0.4 数据：

```text
crop_image(bbox) -> propose_regions + crop_region(region_id)
```

第三步，训练 v0.4 trajectory SFT。

第四步，用 verifier 评估完整 trajectory，得到 SFT baseline。

第五步，做 GRPO 对比实验：

```text
SFT only
Step-wise verifier-guided GRPO
Trajectory GRPO clipped
Trajectory GRPO clipped + reference KL
```

第六步，输出完整报告，重点回答：

- 候选区域工具是否解决了 3B 直接 bbox 不准的问题；
- full trajectory success 是否提升；
- clipped ratio 和 reference KL 是否让训练更稳定；
- verifier-guided 局部奖励是否比纯 terminal reward 更高效。
