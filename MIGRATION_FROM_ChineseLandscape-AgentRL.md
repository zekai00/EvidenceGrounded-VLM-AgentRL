# 从 ChineseLandscape-AgentRL 到 EvidenceGrounded-VLM-AgentRL

## 为什么改名

`ChineseLandscape-AgentRL` 容易被理解为垂直应用项目：山水画 PDF 资料整理。

新名称 `EvidenceGrounded-VLM-AgentRL` 强调算法问题：

```text
多模态文档中的主动取证、claim-level grounding、工具调用策略学习和 verifier-guided RL。
```

山水画仍是第一阶段验证场景，但不是核心卖点。

## 保留旧目录的原因

旧目录已有 v0.1 seed、tool-call 环境、baseline 和 smoke 报告。直接移动会破坏历史路径，因此保留为 archive / seed implementation。

## 新目录承接内容

- 新规划和命名；
- evidence store 接入；
- claim-level benchmark；
- baseline 矩阵；
- trajectory SFT / on-policy GRPO 正式路线。

