---
chapter: 30
part: 5
title: Inference Runtime Architecture
level: tbd
status: draft
prerequisites: [25, 26, 29]
lab:
---

# Chapter 30. Inference Runtime Architecture

线上 p99 TTFT 从 400 ms 涨到 3 秒，GPU 利用率却只有 38%。抓一次 trace 就能看到原因：每个 decode step 结束后，主线程都在等一次 `cudaStreamSynchronize`，把采样的 token 复制回 CPU，再决定下一步调度。GPU 在这段时间里空转，batch 越大，空转越久。

这类问题不是某个 kernel 慢，而是组件之间的协作方式错了。调度器、KV 管理器、model runner、worker 进程、流式输出，任何一处同步或拷贝放在错误的位置，都会把 GPU 的几十毫秒吃掉。推理 runtime 的架构，本质上是在安排这些组件谁在哪个进程、以什么顺序、在哪个时间点交互。

这一章按请求生命周期展开：先给出状态机和八步时序，再逐个拆解 model runner、scheduler、KV cache manager、worker 分离、异步执行、CPU-GPU 协调、内存池、流式响应和错误处理。

## 30.1 Request Lifecycle

一个请求从进入系统到连接关闭，经历六个状态：

```text
ARRIVED ──tokenize──▶ QUEUED ──admit──▶ PREFILL ──last chunk──▶ DECODE ──stop──▶ FINISHED
   │                    │                 │                     │
   └────error───────────┴────abort────────┴─────abort───────────┴──────────────▶ ABORTED
```

| 状态 | 进入条件 | KV 状态 | 退出 |
| --- | --- | --- | --- |
| ARRIVED | HTTP 请求已接收 | 未分配 | 进入 QUEUED 或 ABORTED |
| QUEUED | 等待准入或优先级 | 未分配，或被 swap 到 host | 进入 PREFILL 或 ABORTED |
| PREFILL | 获得 KV 块，开始算 prompt | 按 chunk 增长 | 进入 DECODE 或 ABORTED |
| DECODE | prompt 处理完，生成第一个 token | 每 token 增长 | 进入 FINISHED 或 ABORTED |
| FINISHED | 遇到 EOS、停止串或最大长度 | 已释放 | 终态 |
| ABORTED | 取消、超时或错误 | 已释放 | 终态 |

每个状态只有一个所有者：QUEUED 属于调度器，PREFILL 和 DECODE 属于 model runner，终态属于输出层。跨状态转移必须通过队列，不能直接改对方的数据结构，否则并发取消和抢占会产生悬空指针。

### 一次请求的八步时序

1. 前端接收 HTTP 请求，做 tokenize，分配 request id 和 deadline，放入等待队列。
2. 调度器在下一轮准入中选中它，KV manager 按 `ceil(prompt_len / 16)` 分配初始块。
3. Model runner 把 token id、position、slot mapping 和 block table 拼成扁平张量，执行第一个 prefill chunk。
4. GPU 完成前向，采样器在 GPU 上取出第一个 token，异步复制 4 个字节到 pinned buffer。
5. 输出层 detokenize，通过 SSE 发出首段文本，记录 TTFT。
6. 进入 decode 循环：每轮追加一个 token，跨块边界时向 KV manager 申请新块，采样并流式发送。
7. 遇到 EOS、停止串或最大长度，标记 FINISHED，释放全部 KV 块，冲刷剩余文本。
8. 记录 TTFT、TPOT、排队时间和 token 数，释放请求对象，关闭连接。

八步中只有第 4、6 步的采样在关键路径上，其余都应与 GPU 执行重叠，这是后面的主线。

## 30.2 Model Runner

Model runner 是每轮真正调用模型的部分。它不持有请求对象，也不做调度决策，只接收一个 batch 描述，返回采样结果。这种无状态设计让它能被 CUDA Graph 捕获，也方便在 worker 进程里独立运行。

输入是一组扁平张量，而不是每个序列一个张量：

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| `input_ids` | `[T]` | 本轮所有 token，按序列拼接 |
| `positions` | `[T]` | 每个 token 的绝对位置 |
| `seq_lens` | `[B]` | 每个序列当前总长度 |
| `query_start_loc` | `[B+1]` | 每个序列在 `input_ids` 中的起止 |
| `slot_mapping` | `[T]` | 每个 token 的 KV 写入槽位 |
| `block_tables` | `[B, max_blocks]` | 逻辑块到物理块的映射 |

`slot_mapping` 是连接 scheduler 和 attention kernel 的关键。它由 KV manager 在分配块时生成，把 `(seq, position)` 映射到一个全局槽位号；attention kernel 据此把新算出的 K、V 写到正确位置，不需要知道分页结构。`block_tables` 则供 attention 读取历史 KV。

扁平化让一次 launch 处理整个 batch。若每个序列单独调用 attention，batch 64 就是 64 次 launch，按每次 3–5 微秒算为 200–320 微秒，还没算 Python 开销。变长序列由 `query_start_loc` 区分，kernel 内部按段处理。

decode 形状每轮都变，CUDA Graph 按 batch size 分桶捕获：1、2、4、8、16、32、64、128。实际 batch 向上取整，空槽用 dummy 序列填充。block table 地址通过 graph exec 参数更新，避免每轮重捕获。

```python
class ModelRunner:
    def execute(self, batch):
        meta = self.prepare(batch)          # 扁平化、生成 slot_mapping
        graph = self.graphs.get(batch.bucket)
        if graph is not None:
            graph.replay(meta)              # 一次提交上千个 kernel
        else:
            self.forward(meta)
        return self.sampler.sample(meta.logits, batch.sampling_params)
```

Model runner 里不能有数据相关的 Python 分支。采样参数（temperature、top-p）要提前编译进 kernel 参数或查表，不能每个请求走一次 Python `if`。

## 30.3 Scheduler

Scheduler 是策略层，输入是运行队列、等待队列和 KV 水位，输出是一份执行计划。它不碰 GPU，也不直接改 KV 块，只描述“这一轮想做什么”。

| 字段 | 类型 | 用途 |
| --- | --- | --- |
| `decodes` | `[seq_id]` | 本轮各生成一个 token 的序列 |
| `prefills` | `[(seq_id, chunk_len)]` | 本轮要算的 prefill 及其长度 |
| `blocks_to_allocate` | `[(seq_id, n)]` | 需要新增的 KV 块 |
| `blocks_to_free` | `[block_id]` | 可以回收的块 |
| `preempted` | `[seq_id]` | 因显存不足被暂停的序列 |
| `swapped_in` | `[seq_id]` | 从 host 换回的序列 |

调度和执行分成两个阶段：`schedule()` 只读状态，生成计划；`update(result)` 在 worker 返回后写回 token、更新长度、处理完成和抢占。两阶段之间不修改 block table，这样计划可以安全地异步提交给 worker。

Scheduler 必须快。一轮的 GPU 时间是 20–50 ms，调度逻辑应控制在 1 ms 内。常见性能陷阱是用 Python 对象表示每个 token、在循环里做排序和哈希、或者每轮重新计算整个 block table。运行序列超过一千时，这些开销会接近 GPU 时间。

调度策略本身在 Chapter 29 讨论；这里只强调边界：scheduler 决定“谁上”，KV manager 决定“放哪”，model runner 决定“怎么算”。三者分离，才能在不改 kernel 的情况下替换调度算法。

## 30.4 KV Cache Manager

KV manager 管理固定大小的物理块。块大小通常是 16 个 token，对 70B 模型是 `16 × 320 KiB = 5 MiB`。固定块消除了外部碎片，代价是每个序列平均浪费半块。

接口很小，但每个方法都有并发约束：

| 方法 | 作用 | 约束 |
| --- | --- | --- |
| `can_allocate(seq, n)` | 查询是否有足够空闲块 | 只读，可在调度前调用 |
| `allocate(seq, n)` | 分配 n 个块并建 block table | 必须在 forward 前完成 |
| `append(seq)` | 为下一个 token 追加槽位 | 跨块边界时隐式分配 |
| `free(seq)` | 归还全部块 | 引用计数归零才真正回收 |
| `fork(parent, child)` | 共享 prefix 块 | 引用计数加一，写时复制 |
| `block_table(seq)` | 导出逻辑到物理映射 | forward 期间保持不变 |

显存能放多少块可以直接算：

$$
N_{\text{blocks}} = \left\lfloor \frac{M_{\text{kv}}}{16 \times 320\ \text{KiB}} \right\rfloor
$$

以 8×H100、TP=8 为例，每卡 80 GB，权重分片 17.5 GB，激活和 workspace 6 GB，预留 4 GB，剩下 52.5 GB 给 KV：

```text
N_blocks = 52.5 GB / 5 MiB = 10,752
```

对应 172,032 个 token。按平均 2K 上下文算，理论上能放 84 个序列；留 20% 余量后是 67 个，与 Chapter 29 的准入数字一致。

前缀共享也在这里：两个请求有相同 system prompt 时，`fork` 让它们指向同一组物理块，引用计数加一；写入时才复制该块。共享以块为粒度，只有块对齐的前缀能共享。块表本身也是内存，一千个序列、每序列 2,048 块时约 8 MB。

## 30.5 Worker and Engine Separation

单进程 runtime 在原型阶段够用，生产环境通常拆成 engine 和 worker 两层：

| 组件 | 进程 | 职责 | 通信方式 |
| --- | --- | --- | --- |
| Frontend | engine | HTTP、鉴权、限流 | 外部网络 |
| Tokenizer | engine 或独立池 | 文本转 token | 进程内队列 |
| Scheduler | engine | 组批、准入、抢占 | 进程内调用 |
| KV metadata | engine | block table、引用计数 | 共享内存 |
| Model worker | 每 GPU 一个 | 权重、forward、采样 | ZMQ / 共享内存 |
| TP/PP 通信 | worker 之间 | AllReduce、SendRecv | NCCL |
| Detokenizer | engine 或独立池 | token 转文本 | 异步队列 |

拆分的第一个理由是 Python GIL。调度、tokenize、detokenize、HTTP 都是 CPU 活，放在一个进程里会互抢 GIL；拆开后，worker 可以专心跑 GPU 循环。

第二个理由是 CUDA context。worker 崩溃不会带崩 scheduler；重启只需重新加载权重，engine 可以继续接受请求并排队。

第三个理由是数据量。engine 发给 worker 的元数据一轮几百 KB 到几 MB；worker 只回传采样后的 token id，`B × 4` 字节。若回传 logits，`B × 128K × 4` 字节，batch 64 就是 32 MB，PCIe 和序列化都吃不消。**在 GPU 上采样、只回传 token id**，是这套架构最重要的约定之一。

TP=8 时，8 个 worker 各自持有 1/8 的权重，每层做一次 AllReduce。只有 rank 0 负责和 engine 通信，其余 rank 由 rank 0 广播元数据。跨节点时，engine 在头节点，worker 通过 RDMA 或 TCP 连接，拓扑要避免跨 leaf switch 的 AllReduce。

## 30.6 Async Execution

Engine 主循环是单线程的，一轮只允许一个 batch 在 GPU 上执行；但“单线程”不等于“阻塞”，连接处理、tokenize、detokenize、流式发送都要并行。

```python
async def engine_loop(self):
    while self.running:
        plan = self.scheduler.schedule()
        fut = self.executor.submit(plan)       # 提交后立即返回
        self.streamer.bind(plan)               # 注册本轮的输出
        result = await fut                     # 只等 GPU，不阻塞其他协程
        self.scheduler.update(result)
        await self.streamer.publish(result.tokens)
```

`await fut` 会让出事件循环，其他连接的处理、已生成 token 的发送、健康检查都能继续跑。等 GPU 完成后，`update` 写回状态，下一轮再调度。

异步的边界要选对：

| 阶段 | 放在哪里 | 原因 |
| --- | --- | --- |
| Tokenize | 线程池 | 纯 CPU，可能几十毫秒 |
| Schedule | 主循环 | 需要最新状态，但必须小于 1 ms |
| Forward | worker，异步 future | GPU 执行 |
| Sample | worker 内部 | 避免回传 logits |
| Detokenize | 线程池 | 纯 CPU，可能触发 UTF-8 拼接 |
| Stream write | 协程 + 有界队列 | 网络慢时不能阻塞引擎 |

一个常见错误是在主循环里同步 detokenize。中文 token 可能半个字一个，需要缓存字节直到拼成 UTF-8；单次不贵，乘以每轮几千个 token 就会占掉调度时间。

另一个错误是每轮 `await` 所有 worker 的完整结果，包括日志和统计。统计应该走单独的通道，不能进关键路径。

## 30.7 CPU-GPU Coordination

CPU 和 GPU 的协调成本，主要来自同步和 launch。

一次 kernel launch 在 GPU 侧约 3–5 微秒，70B 模型一层十几个 kernel，80 层就是上千次。单卡 decode 一步约 42 ms（140 GB 权重 / 3.35 TB/s），launch 开销占比：

$$
T_{\text{launch}} = N \cdot t_{\text{launch}} = 1000 \times 4\ \mu s = 4\ \text{ms},
\quad \frac{T_{\text{launch}}}{T_{\text{step}}} = \frac{4}{42} \approx 9.5\%
$$

这还只是 launch。如果每步再加一次 `cudaStreamSynchronize`，GPU 流水线被排空，实际损失往往超过 10 ms。

| 操作 | 是否同步 | 典型开销 | 替代方案 |
| --- | --- | --- | --- |
| `cudaStreamSynchronize` | 是 | 5–20 μs + 流水线排空 | 用 event，且只在必须时等 |
| `.item()` / `.cpu()` | 是 | 10–50 μs | 异步复制到 pinned buffer |
| 逐序列复制 token | 是，B 次 | B × 10 μs | 一次复制整个 `[B]` 张量 |
| `cudaMalloc` | 可能同步 | 10–100 μs | 启动时预分配内存池 |
| Kernel launch | 否 | CPU 5–10 μs，GPU 3–5 μs | CUDA Graph、persistent kernel |
| Logits 回传 | 是 | 每序列 512 KB | GPU 上采样，只回传 4 字节 |

原则是：关键路径上只允许一次小的异步 D2H 拷贝。采样后的 token id 是 `B` 个 int32，batch 64 只有 256 字节；复制到 pinned buffer，等下一轮开始再消费，就能把同步延迟藏到 GPU 执行后面。

CUDA Graph 解决 launch 开销，但 decode 的 block table 每轮都变。做法是把 block table 放在固定地址的显存里，graph 捕获时只记录指针，重放前用 `cudaMemcpyAsync` 更新内容；或者使用 indirect addressing，让 kernel 从设备内存读 block table。前者简单，后者更适合大规模分页。

## 30.8 Memory Pool and Resource Management

Runtime 启动时一次性把显存划成几个池，运行期间不再向驱动申请：

| 池 | 分配时机 | 典型大小 | 复用方式 |
| --- | --- | --- | --- |
| 权重 | 加载时 | 17.5 GB/卡（70B TP=8） | 只读，常驻 |
| KV blocks | 启动时 | 52.5 GB | 固定块，空闲链表 |
| 激活 workspace | 启动时 | 4–8 GB | 按最大 batch 预留 |
| 采样缓冲 | 启动时 | 几 MB | 每轮覆盖 |
| Pinned host | 启动时 | 8–16 GB | swap 和 D2H 暂存 |
| Block table | 启动时 | 几 MB 到几十 MB | 按序列覆盖 |

KV 池的大小由预算决定：

$$
M_{\text{kv}} = M_{\text{total}} - M_{\text{weights}} - M_{\text{act}} - M_{\text{reserve}}
$$

`M_reserve` 不能省。CUDA 的通信 buffer、NCCL 的临时区、cuBLAS 的 workspace 都会在运行时申请；如果把显存算到 100%，第一次 AllReduce 就可能 OOM。经验值是留 5%–8%。

固定块池没有外部碎片，但有内部碎片：每个序列最后一个块平均浪费 8 个 token，即 2.5 MiB。100 个序列是 250 MB，占 52.5 GB 的 0.5%。真正要监控的是块利用率：已分配块数除以总块数；长期高于 95% 时抢占会频繁发生，应该扩容或降低准入。

OOM 的处理顺序是：先抢占低优先级序列，再拒绝新请求，最后才让 worker 抛错。让 CUDA 的 OOM 冒到顶层通常意味着 engine 已经不可恢复，因为异常发生时可能有半个 batch 的 kernel 已经入队。

## 30.9 Streaming Response

流式响应的目标是把首 token 尽快送到客户端，同时不因为网络慢拖住引擎。

Detokenize 要处理 UTF-8 边界。一个中文字符占 3 字节，tokenizer 可能把它拆成两个 token；前半段不是合法 UTF-8，不能直接发。输出层维护字节缓冲，拼成完整字符才发送。

SSE 的每一帧有固定开销：

$$
R = \frac{H + D}{D}
$$

`H` 是帧头、JSON 包装和换行，约 120 字节；`D` 是有效文本。每 token 发一次，两个汉字约 6 字节，`R ≈ 21`；每 8 个 token 合并一次，`D ≈ 24` 字节，`R ≈ 6`。帧越碎，协议开销越大。

| 策略 | 每帧文本 | 100 token 的帧数 | 额外字节 | 首字节延迟 |
| --- | --- | --- | --- | --- |
| 每 token | 3–6 B | 100 | 12 KB | 最低 |
| 每 8 token | 24–48 B | 13 | 1.6 KB | +8 × TPOT |
| 按标点/句子 | 50–200 B | 5–10 | 1 KB | 不确定 |
| 每 20 ms | 不定 | 由 TPOT 决定 | 低 | ≤20 ms |

实践是混合：首 token 立即发，之后按 10–20 ms 或 8 个 token 合并，遇到句号、换行强制冲刷。TTFT 不受影响，后续帧开销也可控。

背压必须有界。每个连接一个 64 帧的队列，队列满时合并旧帧或丢弃中间帧，不能阻塞 engine 主循环。客户端读得太慢，超过 deadline 就断开并释放请求；否则一个慢客户端会拖住整个 batch。

## 30.10 Error Handling and Cancellation

推理 runtime 的错误按影响范围分三级：请求级、engine 级、worker 级。处理方式完全不同。

| 错误 | 范围 | 处理 | 客户端看到 |
| --- | --- | --- | --- |
| Tokenize 失败 | 请求 | 立即拒绝 | 400 |
| 准入拒绝 | 请求 | 不入队 | 429 + `Retry-After` |
| KV 不足 | 请求或 engine | 先抢占，再拒绝 | 429 或 503 |
| 客户端断开 | 请求 | 标记 abort，释放 KV | 无 |
| 超过 deadline | 请求 | 中止并释放 | 504 |
| Detokenize 异常 | 请求 | 替换非法字节或中止 | 500 |
| Worker 崩溃 | worker | 中止该 worker 全部请求，重启 | 503 |
| NCCL 超时 | engine | 中止全部请求，重建通信组 | 503 |

取消可能发生在三个位置：队列里、prefill 中、已提交给 worker。队列里的直接移除；prefill 中的在下一个 chunk 边界停止；已提交的无法撤回，只能等这一步完成，丢弃输出并释放 KV。最后一种浪费一轮计算，但不破坏状态。

取消与完成的竞态由状态机解决：只有 PREFILL 或 DECODE 能转 ABORTED，且转移原子。KV manager 的 `free` 必须幂等，否则取消与完成同时发生会双重释放块。

超时要分层：排队超时、首 token 超时、总时长超时。只设总时长，请求可能在队列里等 29 秒、再跑 2 秒才失败。分层超时能尽早释放容量。

Worker 恢复时间取决于权重加载。单卡 17.5 GB，NVMe 冷读按 3 GB/s 约 6 秒，命中 page cache 接近 1 秒。重启期间 engine 可继续接受请求并排队，或返回 503 让负载均衡器切走。runtime 不保证 exactly-once，重放由客户端的幂等键决定。

## 本章小结

- 请求状态机有 ARRIVED、QUEUED、PREFILL、DECODE、FINISHED、ABORTED 六个状态，每个状态只有一个所有者，跨状态通过队列转移。
- 一次请求穿过八步：接收与 tokenize、准入与分块、准备 prefill、GPU 采样、首 token 流式返回、decode 循环、结束释放、指标与关闭。
- Model runner 是无状态的，输入是扁平张量和 slot mapping，输出是采样后的 token；变长 batch 一次 launch 处理，CUDA Graph 按 batch 分桶。
- Scheduler 只产出计划，KV manager 只管理块，model runner 只执行；三者边界清晰，才能独立替换和异步提交。
- KV 块固定 16 token、5 MiB，`N_blocks = M_kv / 5 MiB`；8×H100 上 52.5 GB 池约 10,752 块、172K token。
- Engine 与 worker 分进程，避免 GIL 和 CUDA context 相互影响；worker 只回传 token id，不回传 logits。
- 异步执行的关键是选对边界：GPU forward 用 future，tokenize/detokenize 放线程池，流式写用有界队列。
- 每步同步是性能杀手；单卡 42 ms 的步长里，1,000 次 launch 已占约 4 ms，再加 `cudaStreamSynchronize` 会排空流水线。
- 显存池启动时预分配，KV 预算要留 5%–8% 余量；OOM 先抢占、再拒绝、最后才抛错。
- 流式响应首 token 立即发，后续按时间或 token 数合并；每 token 一帧的协议开销可达 20 倍。
- 错误按请求、engine、worker 三级处理；取消要覆盖队列、prefill、已提交三种位置，`free` 必须幂等。

## 延伸阅读

- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023.
- Yu et al., [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu), OSDI 2022.
- NVIDIA, [TensorRT-LLM Architecture](https://nvidia.github.io/TensorRT-LLM/architecture/overview.html).
- Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104), 2023.
- Zhong et al., [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670), OSDI 2024.
- vLLM, [Engine and Worker source](https://github.com/vllm-project/vllm/tree/main/vllm/engine).
- NVIDIA, [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs).
- Triton Inference Server, [Architecture](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/architecture.html).
