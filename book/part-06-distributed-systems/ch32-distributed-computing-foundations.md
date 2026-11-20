---
chapter: 32
part: 6
title: Distributed Computing Foundations
level: tbd
status: draft
prerequisites: [4, 5]
lab:
---

# Chapter 32. Distributed Computing Foundations

一张 H100 有 80 GB 显存和 3.35 TB/s 带宽。70B 模型的 FP16 权重是 140 GB，装不下；即使压到 FP8 的 70 GB，加上 KV Cache 和激活也没有余量。多卡不是“跑得更快”的优化，而是“能不能跑”的前提。

但多一张卡就多一次通信。H100 的 NVLink 是 900 GB/s，跨节点 InfiniBand 400 Gb/s 只有 50 GB/s，相差 18 倍。一次 all-reduce 要搬运的数据量和模型大小同阶，通信很快会吃掉并行带来的收益。

这一章讲分布式 LLM 系统的物理基础：进程怎么组织、GPU 怎么连、网络怎么连、消息多大时延迟开始占主导。后面的并行策略，全部建立在这些数字上。

## 32.1 Distributed Execution Model

单个 GPU 的限制来自三个方向：

- **显存**：权重、KV Cache、激活、优化器状态；
- **带宽**：decode 每 token 要读一遍权重；
- **算力**：prefill 的大 GEMM 需要更多 Tensor Core。

分布式的目标是把这三者分摊到多个设备，同时让通信开销小于收益。

### 两种基本切分

**数据并行**：每个设备持有完整模型，处理不同的数据分片，梯度做 all-reduce。它解决吞吐，不解决单卡放不下模型的问题。

**模型并行**：把模型本身切开，每个设备持有一部分权重。它解决显存，但引入了层内或层间的通信。张量并行、流水并行、专家并行都是模型并行的变体。

实际系统通常组合使用：张量并行放在节点内（NVLink），流水并行跨节点（InfiniBand），数据并行复制剩下的副本。

### SPMD

绝大多数框架采用 SPMD（single program, multiple data）：所有进程跑同一份代码，靠 rank 区分角色。

```python
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

if rank == 0:
    print("coordinator")
```

同一段代码里，`if rank == 0` 决定谁打印、谁保存 checkpoint、谁接收请求。控制流的分叉要尽量少，否则容易出现某些 rank 先到集合通信、另一些还在做别的事，导致挂起。

### 扩展效率

$$
\text{scaling efficiency} = \frac{\text{throughput}(N)}{\text{throughput}(1) \times N}
$$

线性扩展是 100%。实际系统在 `N` 增大时效率下降，原因包括：

- 通信占比上升；
- 每个设备的计算粒度变小，固定开销占比上升；
- 负载不均和 straggler；
- 全局资源（如 L2、网络）竞争。

对 LLM 推理，张量并行的扩展效率在节点内通常 70–90%，跨节点会明显下降。

## 32.2 Process and Rank

### Rank 与 world size

`rank` 是进程在通信组里的编号，`world_size` 是进程总数。一个进程通常绑定一张 GPU：

```bash
torchrun --nproc_per_node=8 --nnodes=2 \
  --node_rank=0 --master_addr=10.0.0.1 --master_port=29500 \
  train.py
```

- `nproc_per_node`：每个节点的进程数，通常等于 GPU 数；
- `nnodes`：节点数；
- `node_rank`：节点编号；
- `master_addr/port`：rendezvous 地址。

### local rank

`local_rank` 是进程在节点内的编号，用来绑定 GPU：

```python
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
```

全局 rank 决定通信中的位置，local rank 决定用哪张卡。两者不能混用：跨节点做 all-reduce 用全局 rank，`cudaSetDevice` 用 local rank。

### 进程组

一个作业可以有多个通信组：

```python
# 每个节点 8 卡，节点内 TP=8
tp_group = dist.new_group(ranks=list(range(node * 8, node * 8 + 8)))
# 全局 DP 组
dp_group = dist.new_group(ranks=list(range(world_size)))
```

进程组决定集合通信的范围。TP 组用 NVLink，DP 组跨节点，两者的通信量和频率完全不同，必须分开建组。

### 初始化

NCCL 初始化时会探测拓扑，建立 ring 或 tree。第一次集合通信有明显延迟（连接建立、内存注册），基准测试要跳过前几次。

```python
# warm-up
for _ in range(10):
    dist.all_reduce(tensor)
torch.cuda.synchronize()
```

## 32.3 GPU Topology

### 节点内

一个 8 卡 H100 节点通常是：

```text
GPU0 ─┐
GPU1 ─┼── NVSwitch ── GPU4
GPU2 ─┤                GPU5
GPU3 ─┘                GPU6, GPU7
```

所有 GPU 通过 NVSwitch 全互联，任意两卡之间的带宽相同，约 900 GB/s（H100 SXM）。这比 PCIe 拓扑好得多：没有“跨 CPU socket 更慢”的问题，TP 的通信模式可以认为是均质的。

PCIe 服务器则不同：GPU 挂在不同的 CPU root complex 下，跨 socket 的 P2P 要走 CPU 互联（UPI），带宽减半、延迟翻倍。

### 查看拓扑

```bash
nvidia-smi topo -m
```

输出用 `NV#`、`PIX`、`PHB`、`SYS` 标记连接类型：

| 标记 | 含义 | 相对带宽 |
| --- | --- | --- |
| `NV#` | NVLink，`#` 是链路数 | 最高 |
| `PIX` | 同一 PCIe switch | 高 |
| `PHB` | 同一 PCIe host bridge | 中 |
| `SYS` | 跨 CPU socket / NUMA | 低 |

调度器应该把通信密集的 rank 放在 `NV#` 或 `PIX` 上，避免 `SYS`。

### 多节点

跨节点只能走网络。一个节点内的 8 张卡是一个“island”，island 内用 NVLink，island 间用 InfiniBand。这决定了并行策略的层次：

- TP 放在 island 内；
- PP/DP 跨 island；
- 如果 TP 必须跨节点，通信会成为瓶颈。

## 32.4 PCIe and NVLink

### PCIe

| 代次 | 每 lane 带宽 | x16 单向 | 典型用途 |
| --- | --- | --- | --- |
| PCIe 4.0 | 16 GT/s | 32 GB/s | A100 服务器 |
| PCIe 5.0 | 32 GT/s | 64 GB/s | H100 PCIe |
| PCIe 6.0 | 64 GT/s | 128 GB/s | 新一代平台 |

PCIe 是双向的，表中是单向带宽，两个方向各有独立的 lane。GPU 通过 PCIe 访问 host 内存，延迟约 1–2 μs，带宽远低于 HBM。

### NVLink

| 代次 | 每 GPU 双向带宽 | 代表产品 |
| --- | --- | --- |
| NVLink 3 | 600 GB/s | A100 |
| NVLink 4 | 900 GB/s | H100 |
| NVLink 5 | 1.8 TB/s | B200 |

NVLink 是 GPU 之间的直连，支持 load/store 和原子操作，延迟约 1 μs。NVSwitch 把多个 GPU 连成全互联，避免拓扑不对称。

### P2P 与 GPUDirect

- **P2P（peer-to-peer）**：一张卡直接读写另一张卡的显存，不经过 host；
- **GPUDirect RDMA**：网卡直接读写 GPU 显存，不经过 host 内存，跨节点通信延迟更低。

两者都需要硬件和驱动支持。不支持的路径会退化成“GPU → host → GPU”，带宽减半、延迟翻倍。NCCL 会自动选择路径，但拓扑配置错误时会静默退化。

## 32.5 NUMA and CPU Affinity

### GPU 属于哪个 NUMA 节点

每张 GPU 通过 PCIe 挂在某个 CPU socket 下。跨 socket 访问 host 内存要经过 UPI，带宽更低、延迟更高。

```bash
nvidia-smi topo -m   # 看 GPU 和 NIC 的亲和性
numactl --hardware   # 看 CPU 和内存的 NUMA 布局
```

### 绑核

数据加载、tokenize、网络收发这些 CPU 工作，应该绑在和 GPU 同一个 NUMA 节点上：

```bash
numactl --cpunodebind=0 --membind=0 ./server
```

否则 CPU 线程跨 socket 读写内存，PCIe 传输的 host 端会成为瓶颈。对 decode 这种每步只有几毫秒的场景，一次跨 socket 的 memcpy 就能占掉可观比例。

### 网卡亲和性

多网卡机器上，每张网卡也属于某个 NUMA 节点。跨节点通信应该用和 GPU 同侧的网卡：

```text
GPU0-3 ── PCIe switch ── NIC0 (socket 0)
GPU4-7 ── PCIe switch ── NIC1 (socket 1)
```

如果 GPU0 的数据从 NIC1 出去，就要跨 socket。rail-optimized 网络就是为这种拓扑设计的：每个 GPU 对应一张同侧的网卡。

## 32.6 Network Topology

### Fat-tree

经典数据中心拓扑：

```text
        spine switches
       /      |      \
    leaf     leaf     leaf
   /  \      /  \     /  \
 host host host host host host
```

任意两台主机之间的带宽由 spine 层决定。无阻塞（non-blocking）fat-tree 的 bisection bandwidth 等于总接入带宽；实际部署常有 2:1 或 3:1 收敛，跨 pod 通信会争抢。

### Rail-optimized

LLM 集群常见的设计：每个 GPU 接一张同编号的网卡，所有编号相同的网卡连到同一个 leaf：

```text
GPU0..GPU7 → NIC0..NIC7
NIC0 of all nodes → leaf0
NIC1 of all nodes → leaf1
...
```

这样 TP/PP 的通信可以走固定 rail，避免跨 rail 的拥塞。坏处是单 rail 故障会影响所有节点。

### 拓扑感知调度

并行度要和拓扑匹配：

- 8 卡节点内做 TP=8；
- 两个节点做 PP=2，每个 stage 8 卡；
- 跨节点 DP 尽量放在同一 leaf 下。

Kubernetes 的 topology-aware scheduling、Slurm 的 `--gpus-per-node` 和 `--ntasks-per-node` 都在做这件事。

## 32.7 Communication Latency and Bandwidth

### α-β 模型

一次消息传递的时间近似为：

$$
T = \alpha + \frac{S}{B}
$$

- `α`：固定延迟（软件栈、协议、跳数）；
- `S`：消息字节数；
- `B`：有效带宽。

小消息由 `α` 主导，大消息由 `S/B` 主导。两者的交叉点决定了什么时候该合并消息。

| 路径 | 延迟 α | 带宽 B |
| --- | --- | --- |
| NVLink（节点内） | ~1 μs | 900 GB/s |
| PCIe P2P | ~2 μs | 32–64 GB/s |
| InfiniBand NDR | ~2–5 μs | 50 GB/s |
| TCP/IP | ~10–100 μs | 10–25 GB/s |

### 小消息的效率

一条 1 KB 的消息走 InfiniBand：`1 μs + 1 KB / 50 GB/s ≈ 1.02 μs`，有效带宽只有约 1 GB/s。延迟完全主导。

这解释了两件事：

- 集合通信要把小消息合并成大消息；
- 每层都做 all-reduce 的 TP，层越小，通信效率越低。

### 跨节点估算

70B 模型做一次全参数 all-reduce，数据量 140 GB。ring all-reduce 每个 rank 搬运 `2(N-1)/N × S`：

- 8 卡 NVLink：`1.75 × 140 GB = 245 GB`，按 900 GB/s 约 0.27 s；
- 跨节点 50 GB/s：约 4.9 s。

一次 all-reduce 就要几秒，训练一步的时间根本承受不了。所以大模型的并行策略必须把通信限制在节点内，或者用梯度累积、ZeRO 等手段减少通信量。

## 32.8 Collective Communication Fundamentals

### 常用集合操作

| 操作 | 语义 | 通信量（每 rank） |
| --- | --- | --- |
| Broadcast | 一个 rank 发给所有 | `S` |
| Reduce | 所有 rank 归约到一个 | `S` |
| All-reduce | 归约后广播给所有 | `2(N-1)/N × S`（ring） |
| All-gather | 收集所有 rank 的分片 | `(N-1)/N × S` |
| Reduce-scatter | 归约后分散 | `(N-1)/N × S` |
| All-to-all | 每对 rank 交换分片 | `(N-1)/N × S` |

`S` 是总数据量。all-reduce 可以拆成 reduce-scatter + all-gather，通信量相同，但可以流水线化。

### Ring all-reduce

`N` 个 rank 排成环，分两个阶段：

1. **reduce-scatter**：每个 rank 把数据分成 `N` 块，沿环传递并累加，`N-1` 步后每块有一个完整的和；
2. **all-gather**：把完整的块沿环广播，`N-1` 步后每个 rank 都有全部结果。

每步传 `S/N`，总通信量 `2(N-1)/N × S`。`N` 大时趋近 `2S`，与 `N` 无关——这是 ring 的优点：带宽可扩展。

缺点是步数随 `N` 增长，延迟 `2(N-1)α` 对小消息不友好。Tree 算法延迟 `O(log N)`，适合小消息和跨节点。

### NCCL 的选择

NCCL 会根据消息大小、拓扑和 `NCCL_ALGO` 选择 ring、tree 或 collnet。通常不需要手动干预，但在特定拓扑下调参可能带来 10–20% 的差异。

### 通信与计算重叠

集合通信可以和计算重叠：

```python
# 反向传播时，梯度一旦就绪就启动 all-reduce
dist.all_reduce(grad, async_op=True)
# ... 继续计算其他层 ...
handle.wait()
```

NCCL 的 kernel 在独立的 stream 上执行，SM 资源允许时可以和计算 kernel 并发。重叠得好，通信时间可以部分藏进计算。

## 本章小结

- 分布式解决显存、带宽、算力三个限制，代价是通信。
- SPMD 下所有进程跑同一份代码，rank 决定角色；local rank 绑定 GPU，全局 rank 决定通信位置。
- 节点内 NVLink/NVSwitch 提供 900 GB/s 全互联；跨节点 InfiniBand 400 Gb/s 只有 50 GB/s，相差 18 倍。
- PCIe 和 NUMA 亲和性决定 host 侧路径，绑核和同侧网卡能避免跨 socket。
- 网络拓扑（fat-tree、rail-optimized）决定 bisection bandwidth 和跨 pod 拥塞。
- 通信时间用 `α + S/B` 估算；小消息由延迟主导，大消息由带宽主导。
- ring all-reduce 每 rank 搬运 `2(N-1)/N × S`，70B 的 140 GB 在 8 卡 NVLink 上约 0.27 s，跨节点约 4.9 s。
- 集合通信可以和计算重叠，但需要独立的 stream 和足够的 SM 资源。

## 延伸阅读

- NVIDIA, [NVLink and NVSwitch](https://www.nvidia.com/en-us/data-center/nvlink/).
- NVIDIA, [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/).
- Thakur et al., [Optimization of Collective Communication Operations in MPICH](https://ieeexplore.ieee.org/document/1392756), 2005.
- Patarasuk & Yuan, [Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations](https://www.sciencedirect.com/science/article/pii/S0743731508001666), 2009.
- NVIDIA, [GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/).
- Hoefler et al., [Demystifying Parallel and Distributed Deep Learning: An In-Depth Concurrency Analysis](https://arxiv.org/abs/1802.09941), 2019.
