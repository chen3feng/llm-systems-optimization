# Part IV — GPU Computing and Kernel Optimization

**核心篇 · 约 125 页**

目标：从 CUDA 编程模型深入到实际 Kernel 的设计和性能调优。

## 本篇章节

| 章 | 标题 | 级别 | 状态 |
| --- | --- | --- | --- |
| 17 | [CUDA Programming Model](ch17-cuda-programming-model.md) | TBD | draft |
| 18 | [GPU Memory System](ch18-gpu-memory-system.md) | TBD | draft |
| 19 | [GPU Execution and Optimization](ch19-gpu-execution-and-optimization.md) | TBD | draft |
| 20 | [Matrix Multiplication Optimization](ch20-matrix-multiplication-optimization.md) | Core | draft |
| 21 | [Reduction and Elementwise Kernels](ch21-reduction-and-elementwise-kernels.md) | TBD | draft |
| 22 | [Attention Kernel Implementation](ch22-attention-kernel-implementation.md) | Core | draft |
| 23 | [GPU Kernel Development Frameworks](ch23-gpu-kernel-development-frameworks.md) | TBD | draft |
| 24 | [Kernel Optimization Methodology](ch24-kernel-optimization-methodology.md) | TBD | draft |

## 本篇导言

前三个 Part 讲的是“要算什么”：模型结构决定了张量形状，数值格式决定了字节数，性能模型决定了受限类型。到了 GPU 上，这些问题全部变成具体的执行细节：一个 block 有几个 warp、K/V tile 放在哪里、GEMM 的累加器占多少寄存器、softmax 的同步能不能省掉。

本篇从 CUDA 的执行语义讲起，经过内存层次、执行优化、GEMM 和 attention kernel，最后落到 kernel 开发框架和优化方法论。它不追求覆盖 CUDA 的全部 API，而是围绕 LLM 的真实 kernel 展开：decode 的 GEMV 为什么受带宽限制、prefill 的 attention 为什么要分块、Tensor Core 的指令怎么组织、什么时候该用 Triton 而不是手写 CUDA。

## 学习目标

- 能解释 kernel launch、线程层次、stream 和 event 的执行语义，写出正确的异步代码。
- 能分析一个 kernel 的内存层次、占用率和延迟隐藏需求，判断瓶颈在算力、带宽还是启动开销。
- 能实现并优化 GEMM：从 shared memory tiling 到 register tiling、Tensor Core 和流水线。
- 能实现归约、softmax、RMSNorm、SiLU·Mul 等 LLM 基础 kernel，并把它们融合。
- 能读懂 FlashAttention 的分块和 online softmax，理解 GQA、causal mask 和 FlashDecoding 的 kernel 设计。
- 能在 CUDA、Triton、CUTLASS 之间做出选择，并为不同架构和形状维护可移植的实现。

## 阅读建议

- Chapter 17–19 是基础，建议按顺序阅读；如果已经熟悉 CUDA，可以只看 18.6–18.7 和 19.5–19.6。
- Chapter 20 和 22 是核心，代码较多，建议对照 CUTLASS 或 FlashAttention 的源码阅读。
- Chapter 21 可以独立阅读，重点是融合和数值稳定性。
- Chapter 23 适合需要选型的读者；Chapter 24 是方法论，可以在做完一个 kernel 后回头读。
- 所有性能数字都是特定硬件和形状下的量级，换环境要重新测量。
