# Part V — LLM Inference Systems

**核心篇 · 约 125 页**

目标：理解一个 LLM 推理引擎的完整生命周期，并具备实现简化 Runtime 的能力。

## 本篇章节

| 章 | 标题 | 级别 | 状态 |
| --- | --- | --- | --- |
| 25 | [LLM Inference Execution](ch25-llm-inference-execution.md) | TBD | draft |
| 26 | [KV Cache Architecture](ch26-kv-cache-architecture.md) | Core | draft |
| 27 | [Advanced KV Cache Management](ch27-advanced-kv-cache-management.md) | TBD | draft |
| 28 | [Decoding Optimization](ch28-decoding-optimization.md) | TBD | draft |
| 29 | [Batching and Scheduling](ch29-batching-and-scheduling.md) | TBD | draft |
| 30 | [Inference Runtime Architecture](ch30-inference-runtime-architecture.md) | TBD | draft |
| 31 | [LLM Inference Engine Implementation](ch31-llm-inference-engine-implementation.md) | TBD | draft |

## 本篇导言

一个 70B 模型在 H100 上 decode 一个 token，理论上要 42 毫秒——不是算不出来，是 140 GB 的权重必须从 HBM 读一遍。而同一张卡上，prefill 8K 上下文只需要不到 200 毫秒，因为矩阵乘的算术强度高得多。推理系统的全部设计，都围绕这个不对称展开：如何让 decode 的权重读取被尽可能多的请求摊薄，如何让 prefill 的长请求不阻塞已经在 decode 的请求，如何在有限显存里放下更多轮对话的 KV。

本篇从一次请求的执行流程讲起，深入 KV Cache 的分配与共享、decode 的采样与投机、连续批处理与调度，最后落到一个可实现的 Runtime 结构。它回答的是工程问题：vLLM 为什么要 PagedAttention、SGLang 的 radix tree 解决什么、chunked prefill 在什么情况下反而变慢、speculative decoding 的加速比为什么在 batch 增大时消失。

## 学习目标

- 能拆解 TTFT、TPOT 和端到端延迟，判断一次请求的时间花在 prefill、decode 还是排队。
- 能推导 KV Cache 的字节占用，解释 PagedAttention 如何消除外部碎片并支持前缀共享。
- 能设计 prefix cache 的索引、引用计数、copy-on-write 和淘汰策略。
- 能实现 top-k/top-p 采样和 speculative decoding，并分析接受率、draft 成本与 batch 对加速比的影响。
- 能解释 continuous batching、chunked prefill、抢占和 token budget 调度的取舍。
- 能画出一个推理引擎的组件边界和请求状态机，并实现一个简化版本。

## 阅读建议

- Chapter 25 建立全局图景，建议先读；它定义的所有延迟指标贯穿本篇。
- Chapter 26 是核心，KV Cache 的分配方式决定了后面所有优化能不能落地。
- Chapter 27 和 28 可以按需阅读：长上下文服务重点看 27，低延迟交互重点看 28。
- Chapter 29 和 30 连续阅读，前者讲策略，后者讲这些策略在 runtime 里怎么落地。
- Chapter 31 是实现篇，建议对照 vLLM、SGLang 或 llama.cpp 的源码阅读。
