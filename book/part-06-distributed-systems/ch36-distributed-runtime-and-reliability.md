---
chapter: 36
part: 6
title: Distributed Runtime and Reliability
level: tbd
status: draft
prerequisites: [33, 34, 35]
lab:
---

# Chapter 36. Distributed Runtime and Reliability

分布式推理里，单卡故障的概率被放大了：8 卡一组，单卡年故障率 2%，整组年故障率就接近 15%。更麻烦的是故障的传播方式——一张卡的 NCCL 挂起会让整个集合通信卡死，进程还在、端口还通，请求却永远不返回。

这一章讲运行时骨架：worker 怎么启动和发现彼此、进程组怎么组织、同步点在哪里、错误怎么传播、故障后怎么恢复，以及弹性扩缩容对并行策略意味着什么。

## 36.1 Worker Lifecycle

一个 worker 从进程启动到对外服务，要经过固定的阶段：

```text
init → rendezvous → NCCL init → load weights → warmup → serve → drain → shutdown
```

### Rendezvous

各 rank 先要知道彼此的存在：

```bash
MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
WORLD_SIZE=16 RANK=0 LOCAL_RANK=0 \
python -m engine.server
```

- `MASTER_ADDR/PORT`：所有 rank 连到同一个会合点；
- `WORLD_SIZE`：总进程数，必须一致；
- `RANK`：全局编号，决定分片和通信拓扑；
- `LOCAL_RANK`：节点内编号，决定 `cudaSetDevice`。

rendezvous 超时是启动期最常见的故障：某个 rank 没起来，其他 rank 在端口上等到超时，然后整组退出。排查时要先看哪个 rank 没到，而不是看通信代码。

### 初始化顺序

所有 rank 必须按相同顺序创建进程组、注册 buffer、发起集体通信。顺序不一致会导致集合通信错配——rank 0 在 all-reduce，rank 1 在 all-gather，两边互相等待，没有任何报错，直到超时。

```cpp
// 所有 rank 都按 TP → PP → DP 的顺序创建
ncclComm_t tp_comm, pp_comm, dp_comm;
ncclCommSplit(world, tp_color, tp_rank, &tp_comm, nullptr);
ncclCommSplit(world, pp_color, pp_rank, &pp_comm, nullptr);
```

### Warm-up

首次 kernel launch、首次 NCCL 通信、CUDA Graph 捕获都在 warm-up 完成。warm-up 的输入形状要覆盖生产中的 batch 和序列长度，否则第一次真实请求会触发 JIT 编译或 graph 重捕获，延迟尖刺。

warm-up 也要跑一次完整的 decode 循环，确认 KV Cache 分配、采样、输出路径都正常。

## 36.2 Process Groups

### 组的层级

```text
World（16 卡）
├── TP 组（8 卡 × 2）
│   ├── TP[0]: GPU 0–7
│   └── TP[1]: GPU 8–15
└── PP 组（跨节点的同一 stage）
    ├── PP[0]: GPU 0, 8
    └── PP[1]: GPU 1, 9
```

每个组对应一个 NCCL communicator。通信只在组内发生，组间通过显式的数据搬运连接。

### 创建与销毁

NCCL communicator 的创建是集合操作，所有成员必须参与。动态创建子组时，通常用 `ncclCommSplit`：

```cpp
// color 相同的 rank 进入同一个子组
ncclCommSplit(world_comm, color, key, &sub_comm, &config);
```

`color` 决定分组，`key` 决定组内顺序。两者在所有 rank 上必须自洽。

销毁 communicator 也要同步：一个 rank 提前销毁，其他 rank 的下一次通信就会失败。

### 组与故障域

组的边界就是故障域。TP 组内一张卡故障，整个组不可用；不同 TP 组之间互不影响。所以：

- 组越大，单点故障影响面越大；
- 组越小，并行度越低；
- 故障恢复的粒度是组，不是卡。

## 36.3 Synchronization and Barriers

### 隐式同步

集合通信本身就是同步点：all-reduce 返回时，所有 rank 都到达了。很多代码依赖这一点，不再额外加 barrier。

但要注意：`ncclAllReduce` 默认是异步的（相对于 host），返回只代表 kernel 已入队。要确认完成，需要 `cudaStreamSynchronize` 或事件。

### 显式 barrier

推理循环里有时需要全局对齐，比如所有 rank 都完成 KV 分配后再开始下一步：

```cpp
void barrier(ncclComm_t comm, cudaStream_t stream) {
  ncclAllReduce(nullptr, nullptr, 0, ncclInt, ncclSum, comm, stream);
  cudaStreamSynchronize(stream);
}
```

零字节 all-reduce 只传递到达信号。代价是一次往返延迟，高频调用会显著拖慢 TPOT。

### 超时

```bash
export NCCL_TIMEOUT=1800        # 秒
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
```

超时的作用不是“让通信变快”，而是“让挂起变成可观测的失败”。没有超时，一个挂起的 rank 会让整个组永远等待；有超时，进程退出，调度器重启。

超时值要大于最长合法停顿：大 checkpoint 保存、长 prefill、GC 暂停都可能造成几十秒的停顿。设得太短会误杀。

### 死锁的常见来源

- 条件分支里的集合通信：部分 rank 进入分支，部分没有；
- 不同 rank 的循环次数不同；
- 一个 rank 抛异常退出，其他 rank 还在等；
- 动态 batch 导致某个 rank 跳过了一次通信。

规则很简单：集合通信必须由组内所有 rank 以相同顺序、相同次数调用。

## 36.4 Error Propagation

### 错误类型

| 类型 | 表现 | 传播范围 |
| --- | --- | --- |
| CUDA 执行错误 | kernel 返回错误码 | 当前 rank |
| sticky error | context 损坏，后续调用全失败 | 当前进程 |
| NCCL 通信错误 | 返回 `ncclSystemError` | 整个 communicator |
| 节点失联 | TCP 断开、心跳丢失 | 整个组 |
| 静默错误 | 无报错，结果错误 | 难以检测 |

最危险的是最后一种：ECC 未纠正错误、网络丢包导致的静默数据损坏，输出会缓慢漂移而不是立刻失败。

### 异步错误

CUDA 的 kernel 错误是异步的：launch 返回成功，错误在下一次同步时暴露。所以错误检查点要放在同步之后：

```cpp
kernel<<<grid, block, 0, stream>>>(...);
cudaError_t err = cudaStreamSynchronize(stream);
if (err != cudaSuccess) {
  // 此时才是真正的失败点，栈信息已经丢失
}
```

分布式场景下，一个 rank 失败后，其他 rank 还在往 communicator 里发数据。NCCL 检测到对端消失需要时间，这期间整个组处于不确定状态。

### 快速失败

生产系统倾向于“快速失败”而不是“带病运行”：

- 任一 rank 检测到不可恢复错误，立即标记组失败；
- 通过 side channel（不是 NCCL）通知其他 rank；
- 所有 rank 在下一个检查点退出；
- 调度器摘除并重建整个组。

带病运行的代价是请求超时、结果错误和更难排查的现场。

## 36.5 GPU and Node Failure

### GPU 故障

- **Xid 错误**：驱动报告的硬件/驱动错误，Xid 79 表示 GPU 掉线；
- **ECC 错误**：可纠正错误计数上升是预警，不可纠正错误直接失败；
- **NVLink 降级**：链路从 900 GB/s 降到 450 GB/s，不报错但性能腰斩；
- **显存故障**：分配失败或数据损坏。

Xid 和 ECC 计数要纳入监控。单次可纠正 ECC 不是故障，但短时间内的密集出现往往预示硬件退化。

### 节点故障

- 电源、网卡、主板故障导致整节点掉线；
- Kubernetes 的 node NotReady；
- IB 交换机端口故障导致部分链路不可达。

节点故障时，该节点上的所有 rank 同时消失。TP 组如果是节点内布局，整组失效；如果是跨节点布局，故障影响多个组。

### 检测

```text
进程存活：supervisor 轮询 PID
心跳：每 1 秒通过 TCP 发送，连续 3 次丢失判定失联
NCCL：超时后返回错误
端到端：定期发探针请求，检查 TTFT/TPOT
```

心跳必须走独立通道。NCCL 挂起时，基于 NCCL 的心跳也会挂起，等于没有检测。

## 36.6 Recovery and Restart

### 恢复路径

```text
故障检测 → 摘除故障组 → 分配新资源 → 重新 rendezvous
→ 加载权重 → warmup → 加入服务 → 恢复流量
```

### 时间预算

以 70B、8 卡为例：

| 阶段 | 耗时 |
| --- | --- |
| 资源调度 | 10 s – 数分钟 |
| 容器启动 | 10–30 s |
| 权重加载（page cache 冷） | 20–60 s |
| NCCL 初始化 | 1–5 s |
| CUDA Graph 捕获 + warmup | 5–20 s |
| 合计 | 1–3 分钟 |

这段时间里容量下降，剩余副本要承接流量。如果副本数刚好满足峰值，故障就会引发雪崩。

### 恢复策略

- **整组重建**：最简单，故障域清晰；
- **热备**：预留空闲副本，故障时立即接管，成本是闲置算力；
- **降级**：减少 batch、降低并行度、缩短上下文，先保住可用性；
- **请求重放**：失败的请求重新排队，要求请求幂等或至少可重试。

### 状态丢失

重启后 prefix cache 清空，热点前缀要重新计算。如果流量有强前缀局部性，重启后的前几分钟 TTFT 会明显变差。可以用持久化的 KV 存储（把热点前缀的 KV 存到远端）缓解，但传输成本要算清楚。

## 36.7 Distributed Checkpointing

### 推理场景

推理只需要权重，不需要优化器状态。分片保存的权重按 rank 存储：

```text
checkpoint/
├── metadata.json        # 并行配置、张量映射
├── rank-0.safetensors
├── rank-1.safetensors
...
```

加载时按同样的并行配置读取。如果新实例的 TP 度不同，需要 reshard：

```text
TP=8 的 rank 0 持有 head 0–7
TP=4 的 rank 0 需要 head 0–15 → 从原 rank 0 和 rank 1 各取一半
```

### 训练场景

训练 checkpoint 包含权重、优化器状态、学习率调度器和 RNG 状态。7B、Adam、BF16 的完整状态约 112 GB，70B 约 1.1 TB。

保存时间：1.1 TB 写到 7 GB/s 的并行文件系统约 160 秒。如果每 100 步保存一次，开销不可忽略。常见优化：

- 异步保存：先写显存快照，后台线程落盘；
- 分层保存：近期存内存，定期存远端；
- 只保存增量。

### 原子性

checkpoint 写到一半崩溃会留下损坏的文件。标准做法是先写临时目录，完成后原子重命名：

```text
ckpt-1000.tmp/ → 写完 → rename → ckpt-1000/
```

元数据里记录分片数和校验和，加载时先校验再使用。

### 拓扑变化后的恢复

从 64 卡训练的 checkpoint 恢复到 32 卡，需要重新分片。工具链（如 `torch.distributed.checkpoint`）支持按新的 process group 重映射，但要求保存时记录了全局张量布局，而不是只存本地分片。

## 36.8 Elasticity and Scaling

### 推理的弹性

推理扩缩容的单位是副本（一个完整的 TP/PP 组），不是单卡：

```text
扩容：新副本加载 → warmup → 注册到 router → 接流量
缩容：从 router 摘除 → 等待在途请求完成 → 释放
```

因为不需要改变并行度，推理的弹性比训练简单得多。代价是扩容的粒度大（至少一个 TP 组）和冷启动慢。

### 训练的弹性

训练改变 world size 需要：

1. 所有 rank 在 checkpoint 处停住；
2. 重新分片权重和优化器状态；
3. 用新的拓扑重建 process group；
4. 从 checkpoint 继续。

这不是“无缝”的：正在进行的 step 要丢弃，通信拓扑要重建，数据加载器的分片要重算。多数框架只支持在 step 边界做弹性调整。

### 缩容的安全性

缩容最容易被忽略的是在途请求。直接从 router 摘除会让正在生成的请求失败。正确顺序是：

```text
标记 draining → 停止接新请求 → 等待在途请求完成或超时
→ 保存必要的状态 → 释放 GPU
```

draining 的时间要覆盖最长请求的生成时间。对 4K 输出的请求，可能是几十秒。

## 36.9 Fault-tolerant Serving

### 副本级容错

单副本故障时，router 把流量切到其他副本：

```text
health check 失败 → 标记 unhealthy → 从负载均衡池移除
→ 触发重建 → 健康后重新加入
```

关键是健康检查要能发现“进程活着但不能服务”的状态。探针请求要覆盖完整路径：tokenize → prefill → decode → detokenize。

### 请求级容错

- **重试**：连接失败、5xx 时重试其他副本；
- **幂等**：同一请求重试不能产生重复副作用（对生成类请求通常可接受）；
- **超时**：为 TTFT 和总时长分别设超时，避免请求永久占用连接；
- **熔断**：某副本连续失败时暂时不再路由过去。

流式响应中途失败时，客户端已经收到部分 token。重试会从头生成，用户看到重复内容。协议层要支持“从断点继续”或者明确告知重试。

### 容量规划

故障容错需要冗余。如果 N 个副本刚好承载峰值流量，坏一个就过载。常见配置是 `N+1` 或 `N+20%`，代价是闲置算力。

更精细的做法是让剩余副本降级运行：减小 batch、缩短最大输出、拒绝低优先级请求，用质量换可用性。

### 灰度与回滚

新版本先接少量流量，观察 TTFT、TPOT、错误率和输出质量，再逐步放量。回滚要能在分钟级完成，所以旧版本的权重和镜像要保留。

模型更新时，prefix cache 的键通常包含模型版本，避免新旧权重混用同一份 KV。

## 本章小结

- worker 的生命周期是 rendezvous → NCCL init → 加载 → warmup → 服务 → drain；启动期故障多来自会合和初始化顺序。
- 进程组的边界就是故障域；组越大，单点故障影响面越大。
- 集合通信必须所有 rank 同序同次数调用；超时的作用是让挂起变成可观测的失败。
- CUDA 错误是异步的；分布式系统倾向快速失败，而不是带病运行。
- GPU 故障看 Xid/ECC，节点故障看心跳；心跳必须走独立于 NCCL 的通道。
- 70B 副本的恢复时间在 1–3 分钟量级，容量规划要为此留冗余。
- checkpoint 要原子写、可重分片；推理只存权重，训练还要存优化器状态和 RNG。
- 推理弹性以副本为单位；缩容必须先 draining，再释放。
- 容错要覆盖副本级、请求级和容量级；流式响应中途失败需要协议层支持。

## 延伸阅读

- NVIDIA, [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/).
- PyTorch, [Distributed Checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html).
- NVIDIA, [Xid Errors](https://docs.nvidia.com/deploy/xid-errors/).
- Rajbhandari et al., [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857), SC 2021.
- Mohan et al., [CheckFreq: Frequent, Fine-Grained DNN Checkpointing](https://www.usenix.org/conference/fast21/presentation/mohan), FAST 2021.
