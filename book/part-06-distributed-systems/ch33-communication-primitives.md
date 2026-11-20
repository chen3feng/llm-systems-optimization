---
chapter: 33
part: 6
title: Communication Primitives
level: tbd
status: draft
prerequisites: [32]
lab:
---

# Chapter 33. Communication Primitives

一个 tensor parallel 的 all-reduce，数据量和模型的一层权重同阶；一个 MoE 的 all-to-all，每个 token 要去不同的专家。通信原语不是“调个库”，它的算法选择、消息大小和拓扑匹配，直接决定并行策略能不能兑现。

这一章讲点对点、broadcast、reduce、all-reduce、all-gather、reduce-scatter、all-to-all 的语义和算法，重点算清两件事：每种原语搬多少字节，以及延迟随 rank 数怎么增长。最后讲 NCCL 如何根据拓扑选择 ring 或 tree，以及怎么用 nccl-tests 做基准。

## 33.1 Point-to-point Communication

点对点是两个 rank 之间的消息传递：

```cpp
ncclSend(buf, count, ncclFloat16, peer, comm, stream);
ncclRecv(buf, count, ncclFloat16, peer, comm, stream);
```

语义上是异步的：调用把操作排进 stream，返回不代表数据已经到达。同步靠 stream 的顺序或 `ncclGroupEnd` / event。

### 死锁

两个 rank 同时先 send 再 recv，且消息很大时，双方的发送缓冲区都满，谁也收不了：

```text
rank 0: send(to 1) → 阻塞
rank 1: send(to 0) → 阻塞
```

解决：

- 用非阻塞的 `ncclSend`/`ncclRecv` 配合 `ncclGroupStart/End`；
- 或者让通信模式对称：偶数 rank 先 send，奇数 rank 先 recv；
- 或者用 `MPI_Isend/Irecv` 加 `Waitall`。

NCCL 的 send/recv 在同一个 communicator 上必须成对出现，且 group 内的操作要一起提交，否则会挂起。

### 带宽

点对点的有效带宽取决于路径：NVLink 900 GB/s，PCIe P2P 32–64 GB/s，InfiniBand 50 GB/s。小消息的延迟占主导，1 MB 以下很难跑满带宽。

## 33.2 Broadcast

一个 rank 把数据发给所有 rank。

### 线性 broadcast

root 依次发给每个 rank：`N-1` 次点对点，延迟 `(N-1)α`，root 的出带宽是瓶颈。

### 二叉树 broadcast

root 发给两个子节点，子节点再转发：

```text
step 0: 0 → 1, 2
step 1: 1 → 3, 4;  2 → 5, 6
step 2: 3 → 7 ...
```

延迟 `⌈log2 N⌉ · α`，每个非叶节点只发两份。`N = 8` 时 3 步，线性要 7 步。

大消息时，树形 broadcast 的总带宽需求仍然是 `S`，但每个节点的出带宽只需 `2S`（分给两个子节点），比线性的 `(N-1)S` 均衡得多。

## 33.3 Reduce and AllReduce

Reduce 把 `N` 份数据归约到一份；all-reduce 归约后把结果发给所有 rank。

### Ring all-reduce

`N` 个 rank 排成环，数据分成 `N` 块，分两个阶段：

**Reduce-scatter**：每块数据绕环一圈，经过 `N-1` 次累加后，某个 rank 持有该块的完整和。

**All-gather**：完整的块再绕环一圈，广播给所有 rank。

每个 rank 每步发送 `S/N`，共 `2(N-1)` 步，总通信量：

$$
\text{bytes per rank} = \frac{2(N-1)}{N} S
$$

`N = 8` 时是 `1.75S`；`N` 很大时趋近 `2S`。带宽不随 `N` 增长，这是 ring 的核心优势。

延迟是 `2(N-1)α`，随 `N` 线性增长。小消息时延迟主导，`N = 64` 的 ring 要 126 步，每步 1 μs 就是 126 μs，比计算还长。

### Tree all-reduce

树形归约加 broadcast：先沿树向上归约，再沿树向下广播。延迟 `O(log N)`，但中间节点的带宽压力大，总通信量高于 ring。

### 选择

- 大消息、节点内：ring，带宽最优；
- 小消息、跨节点：tree 或 recursive halving/doubling，延迟更低；
- 有 SHARP 交换机：在网内归约，流量减半。

NCCL 默认按消息大小和拓扑自动选，通常不需要手动指定。

### 计算示例

70B 模型 140 GB 的全参数 all-reduce，8 卡 NVLink：

$$
1.75 \times 140 \text{ GB} = 245 \text{ GB}, \quad
245 / 900 \text{ GB/s} \approx 0.27 \text{ s}
$$

跨节点 50 GB/s 时约 4.9 s。这就是为什么大模型的 TP 必须留在节点内。

## 33.4 AllGather

每个 rank 持有一个分片，all-gather 后每个 rank 都有全部分片。

### Ring all-gather

`N-1` 步，每步每个 rank 发送自己当前持有的一个分片，总通信量：

$$
\frac{N-1}{N} S
$$

`N = 8` 时是 `0.875S`。它常和 reduce-scatter 组合成 all-reduce，也用于 ZeRO 的参数收集。

### 递归倍增

第 `k` 步，每个 rank 和距离 `2^k` 的伙伴交换已收集的块：

```text
step 0: 交换 1 块
step 1: 交换 2 块
step 2: 交换 4 块
...
```

`log2 N` 步完成，延迟 `O(log N)`，但每步消息更大，总通信量略高于 ring。

## 33.5 ReduceScatter

每个 rank 持有一份完整数据，reduce-scatter 后每个 rank 得到一部分归约结果。

它是 all-reduce 的前半段，也是 ZeRO 的核心：梯度按参数分片，每个 rank 只负责自己那一片的归约，显存和通信同时下降。

通信量同样是：

$$
\frac{N-1}{N} S
$$

`N-1` 步 ring。ZeRO-3 把一个 step 拆成 reduce-scatter + all-gather，通信量和标准 all-reduce 相同，但每个 rank 只存 `1/N` 的优化器状态。

## 33.6 All-to-All

每个 rank 把自己的一份数据发给所有其他 rank，同时接收来自所有 rank 的数据。

通信量（每 rank）：

$$
\frac{N-1}{N} S
$$

延迟 `(N-1)α`，因为要和每个 peer 单独通信。

### MoE 的 dispatch 和 combine

MoE 里，每个 token 被路由到 `k` 个专家，专家分布在不同 rank：

```text
dispatch:  token → 按专家分组 → all-to-all → 各 rank 的专家
combine:   专家输出 → all-to-all → 回到原 rank
```

DeepSeek-V3 有 671B 参数、37B 激活，每个 token 的 all-to-all 数据量约 6.7 MB。按 50 GB/s 的跨节点带宽，一次约 134 μs；一个 decode step 有几十层，通信会占掉可观比例。

### 分层 all-to-all

跨节点 all-to-all 很贵，常用分层：

1. 节点内先按目标节点聚合（NVLink）；
2. 节点间交换（InfiniBand）；
3. 节点内再分发（NVLink）。

这把 `N²` 的跨节点消息对数降到 `节点数²`，是 MoE 推理的标准优化。

### 负载不均

all-to-all 的时间由最慢的 peer 决定。如果某个专家的 token 特别多，它所在 rank 的发送会拖慢所有人。MoE 需要 capacity factor、负载均衡 loss 或动态路由来避免这种长尾。

## 33.7 Ring and Tree Algorithms

| 维度 | Ring | Tree |
| --- | --- | --- |
| 带宽 | 最优，`2(N-1)/N S` | 略高 |
| 延迟 | `O(N)` | `O(log N)` |
| 适用消息 | 大 | 小 |
| 适用规模 | 节点内、中等 N | 跨节点、大 N |
| 实现 | 简单，易流水 | 需要树结构维护 |

### 双二叉树

NCCL 的 double binary tree 用两棵树分别承载前半和后半数据，避免根节点成为瓶颈，同时保持 `O(log N)` 延迟。它是跨节点 all-reduce 的常用算法。

### 递归折半/倍增

Recursive halving 做 reduce-scatter，recursive doubling 做 all-gather，延迟 `O(log N)`，通信量和 ring 同阶。适合中等消息。

### 选择不是静态的

NCCL 会根据消息大小切换算法：小消息用 tree，大消息用 ring；跨节点和节点内可能用不同的算法组合。`NCCL_ALGO` 可以强制，但通常让 NCCL 自己选更好。

## 33.8 NCCL Architecture

NCCL 是 NVIDIA 的集合通信库，也是 PyTorch `dist` 的默认后端。

### 拓扑探测

初始化时，NCCL 读取 PCI 拓扑、NVLink 连接、网卡位置，构建一张图。它据此决定：

- 哪些 GPU 之间走 NVLink；
- 哪些走 PCIe；
- 哪些走网络；
- 每个 communicator 用几条 ring/channel。

拓扑识别错误会导致性能静默下降。容器里没有正确的 `/sys` 或 `nvidia-smi` 权限时，NCCL 可能退化成 PCIe 路径。

### Channel

一个 communicator 可以建多个 channel（ring），每个 channel 是一组独立的 ring。多 channel 让多个 SM 同时参与通信，提高带宽利用率。

channel 数由 `NCCL_NCHANNELS` 或自动决定。太少跑不满带宽，太多会增加调度开销。

### Kernel 与 proxy

NCCL 的通信由 GPU kernel 和 CPU proxy 线程共同完成：

- 节点内：kernel 直接做 load/store 或 P2P；
- 跨节点：proxy 线程把数据从 GPU 搬到网卡，或使用 GPUDirect RDMA。

`NCCL_PROTO` 可以选择 LL（low latency）、LL128、Simple。LL 适合小消息，Simple 适合大消息。

### SHARP

支持 SHARP 的交换机可以在网内做归约，数据只上传一次、下行广播，流量减半。它需要网卡和交换机都支持，NCCL 通过 `NCCL_SHARP` 启用。

### 常用环境变量

| 变量 | 作用 |
| --- | --- |
| `NCCL_DEBUG=INFO` | 打印拓扑和算法选择 |
| `NCCL_ALGO` | 强制 Ring/Tree/CollNet |
| `NCCL_PROTO` | 选择 LL/LL128/Simple |
| `NCCL_NCHANNELS` | 通道数 |
| `NCCL_IB_HCA` | 指定网卡 |
| `NCCL_SOCKET_IFNAME` | 指定 socket 网卡 |

调试通信问题时，先看 `NCCL_DEBUG=INFO` 的拓扑和算法，再测带宽。

## 33.9 Communication Benchmarking

### nccl-tests

```bash
./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8
```

- `-b`：起始消息大小；
- `-e`：结束大小；
- `-f`：倍增因子；
- `-g`：每进程 GPU 数。

输出包括 algbw（算法带宽）和 busbw（总线带宽）。两者的关系：

$$
\text{busbw} = \text{algbw} \times \frac{2(N-1)}{N}
$$

busbw 衡量“每个 rank 实际搬运的字节”，更能反映硬件利用率。比较不同 `N` 的结果时，看 busbw，不要看 algbw。

### 测什么

- 小消息（8 B–1 MB）：看延迟，是否随 `N` 增长；
- 大消息（1 MB–1 GB）：看带宽，是否接近链路峰值；
- 跨节点：确认走的是 IB 而不是 TCP；
- 拓扑：`NCCL_DEBUG=INFO` 是否识别到 NVLink。

### 常见问题

- **没 warm-up**：第一次通信包含建链，慢一个数量级；
- **消息太小**：延迟主导，测不出带宽；
- **只测一个大小**：LLM 的 all-reduce 从几 MB 到几百 MB 都有；
- **忽略 busbw**：algbw 随 `N` 下降是正常的，不代表性能差；
- **容器拓扑错误**：NCCL 退化成 PCIe，带宽掉一个数量级。

### 与计算重叠

通信可以和计算重叠，但基准通常单独测通信。实际系统里，all-reduce 可能藏在 GEMM 后面，端到端看到的是 `max(计算, 通信)`，而不是两者之和。评估时要测端到端，不能只看 nccl-tests。

## 本章小结

- 点对点通信要避免死锁，大消息用非阻塞或对称模式；带宽取决于路径。
- Broadcast 用二叉树把延迟从 `O(N)` 降到 `O(log N)`。
- Ring all-reduce 每 rank 搬 `2(N-1)/N S`，带宽最优但延迟 `O(N)`；tree 延迟低但带宽略差。
- All-gather 和 reduce-scatter 各搬 `(N-1)/N S`，组合起来就是 all-reduce，也是 ZeRO 的基础。
- All-to-all 每 rank 搬 `(N-1)/N S`，延迟 `O(N)`；MoE 的 dispatch/combine 依赖它，分层 all-to-all 能减少跨节点流量。
- NCCL 根据拓扑和消息大小选择 ring/tree/collnet，channel 数决定带宽利用率。
- 基准看 busbw 而不是 algbw；注意 warm-up、消息大小和容器拓扑。
- 通信可以与计算重叠，端到端评估比单独测通信更有意义。

## 延伸阅读

- NVIDIA, [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/).
- NVIDIA, [nccl-tests](https://github.com/NVIDIA/nccl-tests).
- Thakur et al., [Optimization of Collective Communication Operations in MPICH](https://ieeexplore.ieee.org/document/1392756), 2005.
- Patarasuk & Yuan, [Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations](https://www.sciencedirect.com/science/article/pii/S0743731508001666), 2009.
- NVIDIA, [SHARP](https://docs.nvidia.com/networking/display/sharpv214).
- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), 2024.
