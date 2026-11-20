# Part X — Advanced Topics and Research Directions

**前沿篇 · 约 35 页**

目标：介绍目前值得关注、但尚不适合成为前面核心主线的方向。

## 本篇章节

| 章 | 标题 | 级别 | 状态 |
| --- | --- | --- | --- |
| 48 | [Advanced Inference Techniques](ch48-advanced-inference-techniques.md) | TBD | draft |
| 49 | [Specialized Workloads](ch49-specialized-workloads.md) | TBD | draft |
| 50 | [Future Directions and Open Problems](ch50-future-directions-and-open-problems.md) | TBD | draft |

## 本篇导言

前面九篇建立的是“把一次生成做快”的知识体系：kernel、显存、调度、并行、训练、后训练、编译、服务。这一篇换两个角度看同一套系统。

一个角度是**改变生成方式**：多采样几次、按难度换模型、让长上下文不再线性增长、为检索和 agent 这类特殊负载重新设计调度。这些技术不改变单个 kernel 的效率，而是改变请求在时间、模型和计算量上的分配。

另一个角度是**往前看**：硬件、数值格式、模型结构和部署形态都在变化，今天的最优解未必是明年的最优解。这一篇不追新，而是把趋势背后的物理约束讲清楚——算力涨得比带宽快，decode 就长期访存受限；内存层级有价格差，KV 就必然走向池化。

## 学习目标

- 能判断 test-time scaling、投机解码、级联在给定 SLO 和 batch 规模下是否划算。
- 能分析长上下文的计算与显存成本，选择环形 attention、KV 压缩或分层存储。
- 能刻画 RAG、agent、结构化输出、多模态和语音负载的资源画像，找到各自的瓶颈。
- 能解释新硬件、新数值格式和新模型结构对系统栈的影响，判断哪些优化会持续增值。

## 阅读建议

- Chapter 48 承接 Chapter 28 的投机解码和 Chapter 29 的调度，建议对照阅读。
- Chapter 49 的五类负载可以按需选读，每节都是独立的资源画像分析。
- Chapter 50 是全书的收尾，建议在读完前面章节后再读，它对全书的结论给出了时间维度上的修正。
