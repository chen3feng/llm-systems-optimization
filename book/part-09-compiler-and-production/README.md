# Part IX — Compiler, Runtime and Production Serving

**核心篇 · 约 75 页**

目标：理解模型计算图如何经过编译和 Runtime，最终转化为硬件执行与生产服务。

## 本篇章节

| 章 | 标题 | 级别 | 状态 |
| --- | --- | --- | --- |
| 45 | [Compiler and Graph Optimization](ch45-compiler-and-graph-optimization.md) | TBD | draft |
| 46 | [Runtime and Memory Management](ch46-runtime-and-memory-management.md) | TBD | draft |
| 47 | [Production LLM Serving](ch47-production-llm-serving.md) | TBD | draft |

## 本篇导言

前八篇拆开讲了 kernel、KV Cache、调度、并行和训练。这一篇把它们重新拼回一个完整的系统：从一张计算图开始，经过编译、下沉、代码生成，落到 GPU 上执行；执行时由 Runtime 管显存、管 stream、管 launch；最后作为线上服务接住真实流量。

这一篇关心的是“胶水层”的成本。单看每一项都不起眼：一次 kernel launch 3 µs，一次显存分配 2 µs，一次同步几十微秒。但 decode 一步只有 10–15 ms，而这些开销要重复上千次。一个 GPU 利用率 40% 的服务，问题往往不在 kernel 写得不好，而在于图没融合、launch 没合并、显存碎片化了、CPU 跟不上 GPU。

## 学习目标

- 能解释计算图的表示与变换，判断一个模型在 `torch.compile`、XLA 或 Triton 下能获得多少融合收益。
- 能估算 kernel launch、显存分配和同步的开销，判断服务是 GPU 绑定还是 CPU 绑定。
- 能设计 CUDA Graph 的接入方案，处理地址固定、形状分桶和通信捕获的约束。
- 能划分推理服务的显存池，规划 KV Cache 容量与并发上限。
- 能设计接入、路由、扩缩容和多租户方案，给出 TTFT/TPOT 的 SLO 与容量规划。
- 能核算单位 token 成本，并判断瓶颈在吞吐、利用率还是缓存命中率。

## 阅读建议

- Chapter 45 承接 Part IV 的 kernel 知识和 Part V 的推理引擎，建议先读 Chapter 19 和 Chapter 24。
- Chapter 46 的显存部分要和 Chapter 27 的 KV Cache 管理对照，两者讲的是同一块显存的两个视角。
- Chapter 47 是全书最贴近线上的一章，建议在读完 Part V、Part VI 后再读，它对前面所有优化给出经济上的结算。
- 三章连起来读，可以回答一个完整的问题：一次推理请求从进入系统到返回 token，每一微秒花在哪里。
