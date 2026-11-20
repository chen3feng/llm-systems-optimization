# Part VI — Distributed LLM Systems

**核心篇 · 约 100 页**

目标：理解多 GPU、多节点环境下的计算分割、通信和同步。

## 本篇章节

| 章 | 标题 | 级别 | 状态 |
| --- | --- | --- | --- |
| 32 | [Distributed Computing Foundations](ch32-distributed-computing-foundations.md) | TBD | draft |
| 33 | [Communication Primitives](ch33-communication-primitives.md) | TBD | draft |
| 34 | [Model Parallelism](ch34-model-parallelism.md) | Core | draft |
| 35 | [Distributed Inference](ch35-distributed-inference.md) | TBD | draft |
| 36 | [Distributed Runtime and Reliability](ch36-distributed-runtime-and-reliability.md) | TBD | draft |

## 本篇导言

70B 模型的 FP16 权重是 140 GB，一张 80 GB 的 H100 放不下；即使用 FP8 压到 70 GB，KV Cache 和激活也没有余量。多卡不是“性能扩展”的可选项，而是部署大模型的必要条件。

但多一张卡就多一次通信。H100 的 NVLink 是 900 GB/s，跨节点 InfiniBand 400 Gb/s 只有 50 GB/s，相差 18 倍；而一次 tensor parallel 的 all-reduce 要搬运的数据量和层大小同阶。分布式 LLM 系统的核心问题，是让计算切分带来的通信量小于切分带来的收益，并让剩下的通信藏在计算后面。

本篇从集群的硬件拓扑讲起，经过集合通信、张量/流水/专家并行，到分布式推理的部署形态和容错。它不追求覆盖所有并行框架，而是把每种并行策略的通信量、显存收益和失效模式算清楚。

## 学习目标

- 能描述 NVLink、PCIe、InfiniBand 的带宽和拓扑，估算跨卡、跨节点的通信时间。
- 能推导 all-reduce、all-gather、reduce-scatter 的通信量，解释 ring 和 tree 的取舍。
- 能计算 tensor parallel、pipeline parallel、expert parallel 的显存收益与通信开销。
- 能判断一个模型该用 TP、PP、EP 还是它们的组合，并给出并行度。
- 能设计分布式推理的请求路由、KV 传输和故障恢复策略。

## 阅读建议

- Chapter 32 建立硬件和故障模型，是后面所有分析的基础。
- Chapter 33 讲通信原语，建议结合 NCCL 的文档阅读。
- Chapter 34 是核心，TP/PP/EP 的公式要自己推一遍。
- Chapter 35 面向部署，重点看并行度选择、KV 传输和负载均衡。
- Chapter 36 面向运维，适合需要设计生产集群的读者。
