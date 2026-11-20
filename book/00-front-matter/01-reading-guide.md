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

## 如何使用书中的估算

本书的每个结论都尽量配一个可验算的数字。读的时候建议随手准备一个计算器，按下面的顺序检验：

1. 先算计算量（FLOPs）和访存量（bytes），求算术强度；
2. 对照硬件的峰值算力和带宽，判断是计算受限还是访存受限；
3. 把受限的那一侧除以硬件能力，得到理论下界；
4. 和书中给出的实测或估算对比，差距就是需要解释的优化空间。

这套方法在 Chapter 3、5 建立，后面每一章都会用到。如果某章的数字和自己的场景不符，先检查假设：精度、batch、序列长度、并行度，通常是这四个参数之一不同。

## 附录

| 附录 | 什么时候查 |
| --- | --- |
| A. 线性代数 | 忘了形状和 FLOPs 的系数 |
| B. 浮点格式 | 调精度、算量化的数值范围 |
| C. CUDA | 写 kernel 时查 API 和架构特性 |
| D. 分布式通信 | 估算通信量、调 NCCL 参数 |
| E. 显存计算器 | 规划部署、估算并发 |
| F. Benchmark 方法 | 做性能对比实验 |
| G. C++/CUDA 工程 | 搭构建、调试和 CI |
| H. 术语表 | 遇到不熟的中英文对照 |
| I. 论文清单 | 想继续深入某个方向 |

## 章节结构

每章统一采用 11 段式模板，见
[templates/chapter-template.md](../../templates/chapter-template.md)。
