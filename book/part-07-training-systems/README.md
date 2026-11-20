# Part VII — LLM Training Systems

**训练系统篇 · 约 75 页**

目标：理解训练过程的资源消耗和分布式执行方式，不把本书变成完整的训练算法教材。

## 本篇章节

| 章 | 标题 | 级别 | 状态 |
| --- | --- | --- | --- |
| 37 | [Training Execution Pipeline](ch37-training-execution-pipeline.md) | TBD | draft |
| 38 | [Training Memory Optimization](ch38-training-memory-optimization.md) | TBD | draft |
| 39 | [Distributed Training](ch39-distributed-training.md) | TBD | draft |
| 40 | [Training Infrastructure](ch40-training-infrastructure.md) | TBD | draft |

## 本篇导言

训练和推理用的几乎是同一批硬件，但资源画像完全不同。推理是“读一遍权重、算一次”，训练是“读一遍权重、算三次、再同步一遍梯度、更新一遍优化器状态”；70B 模型的一步训练要 6N FLOPs/token，显存账单是 112 GB 起步，而千卡规模的梯度同步在跨节点时一次就是 5.6 秒。

本篇不打算讲清楚 transformer 怎么训练、损失函数怎么设计，那是另一类书的内容。它回答的是系统问题：一次训练迭代的时间花在哪里，显存被什么吃掉，多卡之后通信怎么摊薄，集群怎么调度、故障怎么恢复、成本怎么核算。

## 学习目标

- 能拆解一次训练迭代的计算、显存和通信开销，估算给定硬件上的 tokens/s 和 MFU。
- 能推导参数、梯度、优化器状态和激活的显存公式，选择合适的重计算和分片策略。
- 能解释 DDP、ZeRO、FSDP、TP、PP、SP/CP 各自的通信量和适用边界，给出千卡规模的并行组合。
- 能设计 checkpoint、故障恢复和调度方案，估算 MTBF 和恢复时间。
- 能核算一次训练的 GPU 小时和单位 token 成本，并判断瓶颈在算力、通信还是运维。

## 阅读建议

- Chapter 37 建立迭代的时间模型，MFU 和 6N 的公式在后面反复用到。
- Chapter 38 讲显存，建议对照 Chapter 34 的并行策略一起读。
- Chapter 39 是核心，ZeRO 和 TP/PP 的通信量要自己算一遍。
- Chapter 40 面向运维与集群工程，和 Chapter 36 的可靠性讨论互补。
