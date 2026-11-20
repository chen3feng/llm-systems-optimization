---
chapter: 46
part: 9
title: Runtime and Memory Management
level: tbd
status: draft
prerequisites: [45, 27]
lab:
---

# Chapter 46. Runtime and Memory Management

编译好的 kernel 不会自己跑起来。从 HTTP 请求进来到 token 流式返回，中间要经过调度、分配显存、启动 kernel、同步结果、处理网络。这些工作单看每一项都只要几微秒，但它们串在每一步解码的路径上，而一步解码只有 10–15 毫秒。

运行时是最容易被忽视、也最容易拖垮整体的地方。一个 GPU 利用率 40% 的服务，瓶颈往往不在 kernel，而在 CPU 侧的启动开销、频繁的同步点，或者显存分配器的一次碎片整理。这一章讲运行时的架构、内存分配、CUDA Graph 和异步执行，重点在“哪些开销本可以不存在”。

## 46.1 Runtime Architecture

推理运行时从上到下大致分五层：

```text
接入层       HTTP / gRPC / WebSocket，鉴权、限流
调度层       请求排队、continuous batching、抢占
执行层       图执行、kernel launch、同步
内存层       权重、KV Cache、激活、临时 buffer
驱动层       CUDA Runtime / Driver，stream、event
```

每一层都有状态，也都可能成为瓶颈：

- **接入层**：tokenize 在 CPU 上做，一条 1 KB 的 prompt 约 0.2 ms；看起来不多，但单核每秒只能处理几千条，并发高时 CPU 先饱和；
- **调度层**：每个解码步要重新组 batch，决定哪些请求加入、哪些退出；这一步是纯 CPU 的，通常是几十微秒；
- **执行层**：每个 kernel 一次 launch，一次 3–5 µs；decode 一步上千次 launch，就是几毫秒；
- **内存层**：KV Cache 分配、prefix 命中查找、临时 buffer 复用；
- **驱动层**：CUDA API 调用本身有开销，`cudaLaunchKernel` 在驱动里约 2 µs。

把这些数字加起来，就明白为什么 decode 的实际 TPOT 是 10–15 ms，而纯计算只占一半：另一半是运行时开销。优化的方向不是让 kernel 更快，而是让这些开销不进入关键路径。

## 46.2 Memory Allocation Strategies

`cudaMalloc` 是一次昂贵的内核态操作：通常几百微秒，物理内存紧张时可能到秒级。更糟糕的是它会**同步整个设备**——分配期间所有 stream 都得等。

所以没有哪个框架会为每个张量调 `cudaMalloc`。主流的方案是**池化 + 缓存**：

| 策略 | 分配开销 | 碎片 | 适用 |
| --- | --- | --- | --- |
| 直接 `cudaMalloc` | 100 µs–1 s | 由驱动管理 | 初始化、大块常驻 |
| 缓存池（caching allocator） | 1–2 µs | 可能膨胀 | 通用推理/训练 |
| 固定大小 slab | 亚微秒 | 无内部碎片 | KV block、固定形状 |
| `cudaMallocAsync` | 几微秒 | 流序，可回收 | 新架构 |

**缓存池**是 PyTorch 的做法：向驱动申请大段（segment，通常 2 MB 或 20 MB 对齐），按 512 字节粒度切块给张量；释放时块回到池里，不还给驱动。这样反复申请的同一形状的张量几乎零开销，代价是池会一直占着显存不释放。

**固定大小 slab** 更适合推理引擎。KV Cache 的块大小是固定的（16 或 32 token），用 slab 分配器，每个块的位置可以用一个索引表示，分配就是从一个空闲队列里弹出一个 ID，开销在纳秒级。vLLM 的 block table 就是这样工作的。

**`cudaMallocAsync`** 是 CUDA 11.2 引入的流序分配器：分配和释放在 stream 上排队，由驱动管理物理页。它的好处是不需要框架自己维护池，释放的内存能真正还给系统；坏处是分配仍然要走驱动，且跨 stream 复用的语义微妙。

选择的标准很直接：**分配频率高、大小固定，用 slab；大小多变、生命周期不一，用缓存池；需要在多个进程间共享或动态伸缩，才考虑 `cudaMallocAsync`**。

## 46.3 GPU Memory Pools

推理服务的显存要划分成几块互不干扰的区域，否则一块的膨胀会拖垮另一块。

一个典型的单机 8×H100（640 GB）部署 70B 模型的划分：

| 区域 | 大小 | 说明 |
| --- | --- | --- |
| 权重 | 140 GB | BF16，TP=8 每卡 17.5 GB |
| KV Cache | 400 GB | 每卡 50 GB，按 token 算可容纳约 16 万 token |
| 激活与临时 buffer | 40 GB | prefill 的大 batch 峰值 |
| CUDA context 与库 workspace | 20 GB | cuBLAS、cuDNN、NCCL |
| 预留 | 40 GB | 碎片和峰值余量 |

这个划分决定了服务的并发上限。KV Cache 400 GB 是硬上限：按每 token 320 KiB 算，能同时容纳约 128 万个 token；一个 8K 上下文的请求占 2.6 GB，也就是约 150 个并发请求。超过这个数，调度器必须排队或抢占。

几个工程细节：

- **启动时一次性预分配**：KV Cache 在服务启动时按可用显存的比例（通常 0.85–0.9）一次性申请，避免运行中因碎片导致分配失败；
- **预留 headroom**：给临时 buffer 留 5%–10%，否则 prefill 的大激活会 OOM；
- **多进程共享**：同一张卡上跑多个模型实例时，显存池不能互相侵占，通常用 MPS 或各自的显存上限隔离；
- **碎片监控**：PyTorch 的 `max_memory_allocated` 和 `memory_reserved` 的差值持续增长，说明碎片在积累。

碎片是推理服务最常见的“慢性病”：刚启动时能跑 200 并发，跑了一天只能跑 150。原因是不同长度的请求交替分配释放，池里留下了大量小块，虽然总量够，但拼不出连续的大块。缓解手段包括固定块大小（paged KV 天然规避）、定期重启、以及选择 `expandable_segments` 这类允许物理页不连续的分配模式。

## 46.4 Memory Lifetime Management

显存泄漏在推理服务里很少表现为“越用越多”，更常见的是**生命周期设计错误**：

- **张量引用未释放**：一个异常分支里捕获了中间张量，整个请求的显存都不释放；
- **KV block 泄漏**：请求被抢占后没归还块，空闲块逐渐耗尽；
- **prefix cache 无界增长**：命中率很高，但缓存本身把 KV Cache 占满，最终挤掉正在运行的请求；
- **图捕获的静态 buffer 常驻**：CUDA Graph 捕获时分配的所有 buffer 都是常驻的，每个图一份。

prefix cache 的回收策略尤其需要设计。常见的是 LRU 加引用计数：块被正在运行的请求引用时不能回收；引用计数归零后进入 LRU 队列；队列超过阈值按最久未用淘汰。看起来很标准，但两个细节容易出错：

- **部分块**：一个请求只用了块的前 8 个槽位，剩下的槽位不能被其他请求复用，引用计数要按块而不是按 token 算；
- **淘汰与命中的竞争**：调度器刚查到某块能命中，另一个线程把它淘汰了，就会读到垃圾数据。需要一个“命中的块立即加引用”的原子步骤。

Python 的 GC 也会引入非确定性。大量小对象（请求状态、token 列表）会在 GC 触发时造成几十毫秒的停顿，这在 decode 循环里就是一次明显的抖动。生产服务通常调大 GC 阈值，或在关键路径上禁用自动 GC、改用显式回收。

## 46.5 CUDA Graph Integration

回到那个数字：decode 一步约 2,400 次 kernel launch，每次 3–5 µs，合计 7–12 ms，和计算本身相当。

CUDA Graph 把一串 launch 记录下来，之后用一次 `cudaGraphLaunch` 重放，CPU 侧开销降到几十微秒。对 LLM 推理，这通常是 decode 阶段最大的单项优化。

但 CUDA Graph 有三个硬约束：

- **地址固定**：捕获时 kernel 的参数地址是写死的。输入张量、输出张量、KV Cache 的块表都必须在固定地址上，每次执行前把数据拷进这些固定 buffer；
- **形状固定**：图里的 grid/block 尺寸是捕获时确定的，batch 变了就得换一张图；
- **控制流固定**：数据依赖的分支（比如按序列长度选不同 kernel）在捕获时就被固化。

工程上的做法是**按 batch 分桶 + 输入拷贝**：

```python
graph = graphs[bucket]                # bucket = next_pow2(batch)
graph.input_buf.copy_(input_ids)
graph.kv_blocks.copy_(block_table)    # 块表也要是固定地址
graph.replay()
```

代价是两次拷贝（输入和块表），每步几十微秒，远小于省下的 10 ms。块表拷贝的量取决于 batch 和上下文长度，128 个请求、每请求 64 个块，就是 8,192 个 int32，32 KB，拷贝可以忽略。

另一个坑是**多 stream 与通信**。TP 推理里有 all-reduce，CUDA Graph 可以捕获 NCCL 调用，但要求参与通信的 rank 用同样的图、同样的顺序；一旦某个 rank 的分支不同，捕获就会失败或死锁。

实践中，prefill 因为形状变化大、计算占比高，往往不做图捕获；decode 因为形状稳定、launch 开销占比高，几乎都会做。这也解释了为什么很多引擎的 TTFT 优化空间小、而 TPOT 优化空间大——两者的瓶颈性质不同。

## 46.6 CPU-GPU Scheduling

CPU 和 GPU 之间是异步的：CPU 把 kernel 塞进 stream 就返回，GPU 按顺序执行。理想情况下 CPU 跑在前面，GPU 永远不饿。

实际情况是 CPU 经常跟不上：

- **Python 开销**：每个算子经过 Python 层调度约 10–20 µs，比 kernel 本身还长；
- **同步点**：任何需要读回数值的操作都会阻塞，比如 `if logits.max() > threshold`、打印中间结果、`.item()`；
- **GIL**：多线程服务里，Python 的全局锁让 tokenizer 和调度器无法真正并行；
- **锁竞争**：调度器、块分配器、prefix cache 共享数据结构，高并发下锁竞争会把 CPU 时间放大。

一个典型的配置是把职责拆开：前端进程做 HTTP 和 tokenize（可以多进程），调度器单独一个进程，GPU 执行在同进程的多个 stream 上。这样 tokenize 的 CPU 开销和 GPU 执行可以重叠。

调度粒度也影响 CPU 效率。每步重新组 batch 的 continuous batching，在 128 并发的规模下，每步要做上百次决策，CPU 时间可能到几百微秒。优化手段是**增量调度**：只在有请求进出时更新 batch，其余步骤复用上一步的调度结果。

## 46.7 Async Execution and Synchronization

正确使用 stream 和 event 能省掉大量隐式同步。

几条规则：

- **默认 stream 是隐式有序的**：PyTorch 的默认 stream 上，张量操作按顺序执行，但跨 stream 需要显式 event；
- **拷贝要用 pinned memory**：H2D/D2H 用小页内存会退化成同步拷贝，带宽从 50 GB/s 掉到 6 GB/s 且阻塞；
- **避免 device-wide 同步**：`torch.cuda.synchronize()` 等所有 stream，`cudaDeviceSynchronize` 更贵；用 `cudaStreamSynchronize` 或 event 只等需要的部分；
- **不要在读回数值的分支上做调度决策**：需要根据 GPU 结果分支时，宁可多做一点计算，也不要把结果读回 CPU。

一个常被忽视的同步点是 **CUDA Graph 与动态形状的冲突**。捕获时如果一个 tensor 的 shape 依赖运行时数据，捕获会失败，框架回退到 eager 执行，性能悄悄掉一半。这类回退通常只打一行 warning，容易被日志淹没。

另一个是 **NCCL 的隐式同步**。集合通信本身是异步的，但如果不及时消费输出，后续计算会等；如果通信 stream 和计算 stream 之间没有正确的 event 依赖，则可能出现数据竞争——表现为偶发的错误结果，而不是崩溃，排查极其困难。

## 46.8 Runtime Instrumentation

运行时的问题大多不是“慢”，而是“偶发地慢”。定位它们需要分层的观测：

| 层次 | 工具 | 能看到什么 |
| --- | --- | --- |
| Kernel | Nsight Compute | 单 kernel 的占用率、访存效率、停顿原因 |
| 时间线 | Nsight Systems | kernel 间隙、同步点、CPU-GPU 重叠 |
| API | CUPTI | 每次 CUDA API 的耗时和调用栈 |
| 运行时 | 自建指标 | 编译缓存命中、块分配失败、队列长度 |
| 服务 | 分布式追踪 | 端到端延迟分解 |

时间线（timeline）是最有用的。一次典型的 decode 步骤在 Nsight Systems 里能看到：GPU 上有几十个时间间隙，每个间隙对应一次 CPU 侧的调度或一次同步。间隙总长除以步骤总长，就是“CPU 绑定的比例”。

自建指标里最值得盯的几个：

- **kernel 间隙占比**：GPU 空闲时间 / 总时间；
- **图重编译次数**：每分钟触发多少次形状特化；
- **块分配失败率**：KV 块申请不到导致抢占或排队；
- **CUDA API 耗时分布**：`cudaMalloc`、`cudaMemcpy` 的 P99；
- **CPU 单步耗时**：decode 每一步在 CPU 侧花多久。

这些指标不需要昂贵的工具，埋点成本很低，但能覆盖 80% 的运行时问题。剩下的 20%——比如某个 kernel 因为 bank conflict 慢了 3 倍——才需要 Nsight 级别的分析。

## 本章小结

- 推理运行时分接入、调度、执行、内存、驱动五层；decode 的 TPOT 里约一半是运行时开销，优化方向是让这些开销离开关键路径。
- `cudaMalloc` 要几百微秒且同步设备，框架必须池化；缓存池适合通用场景，固定大小 slab 适合 KV block，`cudaMallocAsync` 适合需要真正回收的场景。
- 70B 在 8×H100 上的显存划分里，KV Cache 通常占一半以上，它决定了并发上限；启动时一次性预分配并预留 5%–10% headroom。
- 显存问题多来自生命周期设计——块泄漏、prefix cache 无界、图静态 buffer 常驻；Python GC 停顿也会表现为延迟抖动。
- CUDA Graph 把 decode 的 2,400 次 launch 压成一次重放，省下 7–12 ms；代价是地址、形状、控制流都必须固定，靠 batch 分桶和输入拷贝解决。
- CPU 侧的开销来自 Python 调度、同步点、GIL 和锁竞争；拆分进程、增量调度、避免读回数值能显著改善。
- 跨 stream 要显式 event，拷贝要用 pinned memory；CUDA Graph 捕获失败和 NCCL 隐式同步是最难排查的两类问题。
- 运行时观测优先看 kernel 间隙占比、重编译次数、块分配失败率和 API 耗时分布，这些埋点能覆盖大多数问题。

## 延伸阅读

- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023.
- Yu et al., [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu), OSDI 2022.
- NVIDIA, [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).
- NVIDIA, [Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/), 2019.
- PyTorch, [CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management).
- Gregg and Hazelwood, [Where Is the Data? Why You Cannot Debate CPU vs. GPU Performance Without the Answer](https://ieeexplore.ieee.org/document/5762737), ISPASS 2011.
- vLLM Team, [vLLM: Easy, Fast, and Cheap LLM Serving](https://github.com/vllm-project/vllm).
