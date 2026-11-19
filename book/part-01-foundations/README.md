# Part I — Foundations: Understanding LLM Computation

**基础篇 · 约 75 页**

目标：建立理解后续所有优化技术所需的数学、计算和硬件模型。

## 本篇章节

| 章 | 标题 | 级别 | 状态 |
| --- | --- | --- | --- |
| 1 | [Introduction to LLM Systems](ch01-introduction-to-llm-systems.md) | TBD | draft |
| 2 | [Transformer Computation Model](ch02-transformer-computation-model.md) | TBD | draft |
| 3 | [Computational Complexity and Cost Modeling](ch03-computational-complexity-and-cost-modeling.md) | Core | draft |
| 4 | [Modern Computer Architecture for LLMs](ch04-modern-computer-architecture-for-llms.md) | TBD | outline |
| 5 | [Performance Modeling and Analysis](ch05-performance-modeling-and-analysis.md) | TBD | outline |
| 6 | [Profiling and Debugging](ch06-profiling-and-debugging.md) | TBD | outline |

## 本篇导言

后面每一个优化技术，最终都要回答三个问题：它减少了多少计算、减少了多少访存、增加了多少通信。要回答这三个问题，先得有一套共同的度量语言。

本篇建立这套语言：从 Transformer 的计算图推导 FLOPs、显存和通信量，用 roofline 判断一个算子是算力受限还是带宽受限，再用 profiler 验证推理是否成立。

## 学习目标

- 能从模型结构直接估算参数量、FLOPs、激活显存和 KV Cache。
- 能区分 prefill 与 decode 的计算特征，并解释为什么 decode 通常受显存带宽限制。
- 能读懂 roofline 图，判断优化空间在算力侧还是带宽侧。
- 能用 Nsight Systems / Nsight Compute 定位 kernel 级和系统级瓶颈。

## 阅读建议

- Chapter 1 是全书地图，先读一遍，不必记住所有细节。
- Chapter 2 和 Chapter 3 是后续所有性能分析的基础，建议完整推导一遍。
- Chapter 4 可以在读到 GPU Kernel 部分时回头查阅。
- Chapter 6 建议边读边在真实 GPU 上运行，不要只看截图。
