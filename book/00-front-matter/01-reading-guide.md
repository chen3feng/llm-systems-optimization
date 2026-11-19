# 阅读指南

## 章节分级

为控制全书篇幅，章节分为三级：

| 级别 | 含义 | 写作要求 |
| --- | --- | --- |
| Core | 必须深入 | 包含数学推导、C++/CUDA 实现和 Benchmark |
| Supporting | 解释原理与工程影响 | 讲清机制和典型实现，不做全面源码分析 |
| Survey | 介绍方向与演进 | 说明开放问题，不展开实现细节 |

## 推荐阅读顺序

目录顺序不等于严格的学习顺序。对有 C++ 和系统背景的读者，建议：

1. Part I：完整学习。
2. Part II：重点掌握 Transformer、Attention、MoE。
3. Part IV：深入 CUDA 和 Kernel。
4. Part V：实现 Inference Engine。
5. Part VI：学习 Distributed Systems。
6. Part III：结合实际 Kernel 学习量化和压缩。
7. Part VII：理解训练系统。
8. Part IX：深入 Compiler 和 Runtime。
9. Part VIII：理解 Post-training 的质量与训练机制。
10. Part X：研究前沿。

先能执行和分析模型，再研究训练和模型质量，对系统工程师更容易形成直觉。

## 实验环境分级

| 环境 | 能完成的实验 |
| --- | --- |
| CPU-only | Tensor、Transformer Reference、量化算法、调度模拟 |
| 单 NVIDIA GPU | CUDA Kernel、Attention、KV Cache、单卡推理 |
| 多 NVIDIA GPU | TP、PP、NCCL、分布式推理 |
| 多节点集群 | RDMA、通信拓扑、分布式训练和 Serving |

## 章节结构

每章统一采用 11 段式模板，见
[templates/chapter-template.md](../../templates/chapter-template.md)。
