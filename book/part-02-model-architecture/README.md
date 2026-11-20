# Part II — Model Architecture and Computation

**模型计算篇 · 约 65 页**

目标：理解模型结构如何决定计算模式、访存模式和并行方式。

## 本篇章节

| 章 | 标题 | 级别 | 状态 |
| --- | --- | --- | --- |
| 7 | [Transformer Architecture in Practice](ch07-transformer-architecture-in-practice.md) | TBD | draft |
| 8 | [Attention Mechanisms](ch08-attention-mechanisms.md) | TBD | draft |
| 9 | [Modern Attention Architectures](ch09-modern-attention-architectures.md) | TBD | draft |
| 10 | [Mixture of Experts](ch10-mixture-of-experts.md) | TBD | outline |
| 11 | [Alternative and Hybrid Architectures](ch11-alternative-and-hybrid-architectures.md) | TBD | outline |

## 本篇导言

同一个 attention 公式，换一种 KV 组织方式，KV Cache 可以缩小一个数量级；同一个 FFN，换成 MoE，每个 token 只激活八分之一的参数。模型结构不是论文里的细节，它直接决定显存占用、访存模式和并行策略。

本篇从一份真实的 `config.json` 出发，把 Transformer 拆成张量形状和算子，再逐层深入：注意力的各种变体、KV Cache 的结构性优化、MoE 的路由与通信，以及线性复杂度的替代架构。

## 学习目标

- 能从 `config.json` 推导每层的权重形状、参数量和显存占用。
- 能实现 MHA、MQA、GQA 和 MLA 的 KV 布局，并解释它们的取舍。
- 能分析 attention 在 prefill 和 decode 下的计算量与访存量。
- 能解释 MoE 的专家路由、负载均衡、容量因子和通信开销。
- 能判断稠密、MoE、混合架构分别适合什么场景。

## 阅读建议

- Chapter 7 是实践基线，建议对照一个真实 checkpoint 阅读，边读边打印张量形状。
- Chapter 8 讲注意力的一般机制，Chapter 9 讲现代变体，两章连续阅读。
- Chapter 10 可以独立阅读；如果只关心推理系统，重点看路由和 KV Cache 部分。
- Chapter 11 是选读，适合需要评估新架构的读者。
