---
chapter: 4
part: 1
title: Modern Computer Architecture for LLMs
level: tbd
status: draft
prerequisites: [3]
lab:
---

# Chapter 4. Modern Computer Architecture for LLMs

一次 HBM 访问的延迟大约是 500 个时钟周期。

在这 500 个周期里，一个 SM 理论上可以发射上千条指令。如果它没有足够的 warp 可以切换，就只能空转。GPU 的整个架构——几万个线程、每个 SM 几十个 warp、上百 KB 的 shared memory——都是为了回答同一个问题：**怎么让这 500 个周期不白等**。

这一章讲硬件。不是芯片设计的细节，而是那些会直接出现在性能数字里的结构：SIMD 宽度、NUMA 距离、SM 的调度方式、寄存器与 shared memory 的容量、HBM 的带宽、卡与卡之间的链路。上一章的 roofline 公式里，峰值算力和带宽是两个抽象符号；这一章把它们拆开，看看它们从哪来、为什么跑不满。

## 4.1 CPU Architecture and SIMD

LLM 系统的大部分计算在 GPU 上，但 CPU 从来没有离开过关键路径。

模型加载、tokenization、请求调度、采样、结果后处理、日志和网络收发都在 CPU 上。CPU 推理也不是边缘场景：llama.cpp 在笔记本和服务器上跑量化模型，靠的就是 CPU 的向量指令和内存带宽。

### SIMD 的宽度

现代 CPU 用 SIMD（Single Instruction, Multiple Data）一次处理多个数据。x86 的演进基本就是不断加宽向量寄存器：

| 指令集 | 向量宽度 | FP32 通道数 | 引入时间 |
| --- | --- | --- | --- |
| SSE | 128 bit | 4 | 1999 |
| AVX | 256 bit | 8 | 2011 |
| AVX2 | 256 bit | 8 | 2013 |
| AVX-512 | 512 bit | 16 | 2017 |

ARM 这边，NEON 固定 128 bit；SVE 允许 128 到 2048 bit 的可变宽度，Apple M 系列和部分服务器芯片各有自己的实现。

每条 FMA（fused multiply-add）指令完成一次乘加，按 2 FLOPs 计。一个核心的 FP32 峰值是：

$$
\text{FLOPs/cycle} = \text{向量通道数} \times 2 \times \text{FMA 单元数}
$$

以 56 核的 Sapphire Rapids 为例，AVX-512 有 16 个 FP32 通道，每核两个 FMA 单元，全核约 2 GHz：

$$
56 \times 16 \times 2 \times 2 \times 2\text{e}9 \approx 7.2 \text{ TFLOPS}
$$

这是理论值。AVX-512 满载会让频率下降，实测通常在 3–5 TFLOPS 之间。**厂商规格表上的 CPU 峰值，和持续负载下的真实值往往差一倍。**

### AMX 与矩阵引擎

Sapphire Rapids 引入了 AMX（Advanced Matrix Extensions）。它不再是一条指令处理一个向量，而是一组 8 个 1 KB 的 tile 寄存器，一条指令完成 `16 × 16` 的输出块和长度为 32 的内积。BF16 下每周期约 1024 次乘加，单核算力比 AVX-512 高一个数量级。

这是 CPU 第一次拥有接近 GPU Tensor Core 的矩阵单元。但它的生态远不如 CUDA 成熟，实际用于 LLM 推理的代码仍然以量化和手写 kernel 为主。

### 向量化代码

一段标量点积：

```cpp
float dot(const float* a, const float* b, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}
```

编译器通常能自动向量化，但前提是它确定没有别名（alias）。加 `restrict` 和 `#pragma omp simd` 可以帮它：

```cpp
float dot(const float* __restrict a, const float* __restrict b, int n) {
  float s = 0.0f;
  #pragma omp simd reduction(+:s)
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}
```

手写 AVX2 版本用 `_mm256_fmadd_ps`，一次处理 8 个 float。要跑到峰值，循环还得展开、用多个累加器打破依赖链——单个累加器的加法延迟会让流水线停顿。

### CPU 的算术强度

CPU 的机器平衡点比 GPU 低得多。96 核的 EPYC 9654 有 12 通道 DDR5-4800，带宽约 460 GB/s；FP32 峰值按 10 TFLOPS 估算，脊点只有约 22 FLOP/byte。GPU 是 295。

这不是坏事，它说明 CPU 上几乎一切 LLM 计算都是带宽受限的。llama.cpp 选择 4-bit 量化，不只是为了省显存，更是为了把每 token 要读的字节数压到带宽能承受的范围。

一个 70B 模型量化到 4 bit 约 35 GB。单路 EPYC 的 460 GB/s 意味着：

$$
\frac{35 \text{ GB}}{460 \text{ GB/s}} \approx 76 \text{ ms/token}
$$

也就是每个 token 最多 13 token/s。双路机器做 NUMA interleave 可以把两条内存通道都用上，理论上接近 26 token/s，但跨 socket 的访问会打折。**在 CPU 上，模型能跑多快几乎完全由内存带宽决定，和核数关系不大。**

## 4.2 NUMA and Memory Locality

多路服务器里，每个 CPU 有自己的内存控制器。CPU 0 访问插在 CPU 0 上的内存，走本地通路；访问 CPU 1 的内存，要经过 UPI（Intel）或 Infinity Fabric（AMD）跨 socket。

| 访问类型 | 典型延迟 | 相对带宽 |
| --- | --- | --- |
| 本地 DRAM | 80–100 ns | 100% |
| 跨 socket DRAM | 140–200 ns | 50%–70% |
| 跨 socket 且拥塞 | 300 ns 以上 | 更低 |

Linux 的默认策略是 first-touch：内存页分配给第一个写它的线程所在的 NUMA 节点。这个策略对单线程初始化、多线程使用的程序很危险——如果主线程把 140 GB 权重全部读进本地节点，后续工作线程有一半在远程访问。

### 在 LLM 场景里的影响

模型加载是最典型的例子。用 `mmap` 读权重，页的归属取决于哪个线程先触碰。正确的做法是：

- 加载时按 NUMA 节点并行，每个节点用自己的线程读自己那份；
- 或者用 `numactl --interleave=all` 让页均匀分布，接受一部分远程访问；
- 推理时把线程绑定到对应的 CPU，`numactl --cpunodebind` 或 `taskset`。

GPU 服务器还有一层：GPU 插在某个 PCIe root complex 下，物理上属于某个 socket。主机线程如果跑在另一个 socket 上，每次 `cudaMemcpy` 的元数据和 doorbell 都要跨 UPI。8 卡机器通常 4 卡挂 socket 0、4 卡挂 socket 1，NUMA 绑定要和 GPU 拓扑对齐。

### 大页

140 GB 的权重，按 4 KB 页算是 3,500 万个页表项。TLB 装不下，每次访问都可能 miss。透明大页（THP）或显式 hugetlbfs 把页放大到 2 MB 或 1 GB，TLB 覆盖范围提高两三个数量级。

对顺序扫描的推理负载，大页通常能带来 5%–15% 的提升。对随机访问的 KV Cache offload 场景，提升更明显。

### 怎么观察

```bash
numastat -p $(pgrep -f vllm)      # 每个节点的本地/远程分配
lstopo-no-graphics                 # 拓扑图
perf stat -e node-load-misses,node-loads -p <pid>
```

如果 `numastat` 显示某个进程的 `Foreign` 列很大，说明它在大量跨节点访问。

## 4.3 GPU Architecture Overview

CPU 把芯片面积花在缓存、分支预测、乱序执行和少量强核上；GPU 把面积花在大量简单的执行单元和高带宽内存上。

| | CPU | GPU |
| --- | --- | --- |
| 优化目标 | 单线程延迟 | 吞吐 |
| 核心数 | 几十 | 上百个 SM，每个 SM 上百个 lane |
| 控制逻辑 | 复杂 | 极简 |
| 缓存 | 大，多级 | 小，靠软件管理 shared memory |
| 内存带宽 | 数百 GB/s | 数 TB/s |
| 适合 | 分支多、依赖链长的代码 | 数据并行、规则计算 |

### 一个具体的 GPU

H100 SXM 的关键规格：

| 项目 | 数值 |
| --- | --- |
| SM 数量 | 132 |
| FP32 CUDA core | 16,896 |
| Tensor Core | 每个 SM 4 个 |
| 寄存器文件 | 每个 SM 256 KB |
| Shared memory | 每个 SM 最多 228 KB |
| L2 cache | 50 MB |
| HBM3 容量 | 80 GB |
| HBM3 带宽 | 3.35 TB/s |
| FP16 dense | 989 TFLOPS |
| FP32 | 67 TFLOPS |

989 TFLOPS 是 Tensor Core 的数字。如果只用普通 FP32 CUDA core，峰值只有 67 TFLOPS，差 15 倍。**同一块芯片，用不用 Tensor Core，性能是两个时代。**

### 执行层次

CUDA 的抽象层次是 grid → block → warp → thread：

- **Thread**：最小的执行单位，有自己的寄存器和栈；
- **Warp**：32 个线程，硬件调度的基本单位，同一 warp 的线程执行同一条指令；
- **Block**：一组 warp，最多 1024 个线程，可以共享 shared memory，用 `__syncthreads()` 同步；
- **Grid**：一次 kernel launch 的全部 block。

Block 被分配到 SM 上执行，一个 SM 可以同时驻留多个 block。Block 内的 warp 可以跨 SM 的分区调度，但 block 不会拆到多个 SM 上。

### 为什么需要这么多线程

CPU 靠乱序执行和缓存把单线程延迟藏起来；GPU 没有这些，它靠**线程级并行**：一个 warp 因为等内存停住，调度器立刻切到另一个 ready 的 warp。

HBM 延迟约 500 周期。如果一个 SM 只有 4 个 warp，而它们都在等内存，SM 就空了。要填满这 500 周期，需要几十个 warp 轮转，并且每个 warp 都要有足够的独立指令。

这就是 occupancy（占用率）重要的原因，也是为什么“GPU 利用率 100%”可能只发挥了 10% 的算力。

## 4.4 SM, Warp, and Execution Units

一个 Hopper SM 分成 4 个处理块（processing block），每个块有：

- 1 个 warp scheduler；
- 1 个 dispatch unit；
- 32 个 FP32 lane；
- 16 个 INT32 lane（与 FP32 共享部分通路）；
- 1 个 Tensor Core；
- 一组 LD/ST 单元。

每个周期，每个 scheduler 可以发射 1 条 warp 指令。4 个分区合计每周期 4 条。一条 FP32 指令作用于 32 个 lane，所以一个 SM 每周期最多完成 `4 × 32 = 128` 次 FP32 运算；FMA 再乘 2，即 256 FLOPs。

132 个 SM、1.98 GHz：

$$
132 \times 128 \times 2 \times 1.98\text{e}9 \approx 67 \text{ TFLOPS}
$$

和规格表上的 FP32 峰值对上了。

### Warp 与分支发散

Warp 的 32 个线程共享程序计数器。如果代码里有分支，而 warp 内的线程走了不同路径，硬件会把两条路径串行执行，每条路径上只有一部分 lane 有效：

```cuda
if (threadIdx.x < 16) {
  y = f(x);      // 只有一半 lane 工作
} else {
  y = g(x);      // 另一半，串行执行
}
```

这叫分支发散。它不会让结果出错，但会让有效吞吐减半。LLM kernel 里常见的发散来源是 causal mask、变长序列和边界判断。写法上可以让整个 warp 走同一分支，或者用无分支的 `select` 替代。

### Tensor Core

Tensor Core 执行矩阵乘加，一条指令完成一个小矩阵块。Volta 的第一代做 `4×4×4`；Ampere 的 `m16n8k16` 是 warp 级指令；Hopper 的 `wgmma` 把操作数扩大到 `m64nNk16`，并且可以直接从 shared memory 取数。

Tensor Core 的吞吐比 CUDA core 高一个数量级，但它要求数据布局对齐、精度匹配。这也是为什么手写 kernel 的性能差距往往不在算法，而在“有没有用上正确的 MMA 指令”。

### Warp 级原语

很多 kernel 需要 warp 内通信。`__shfl_down_sync` 可以在寄存器之间交换数据，不经过 shared memory：

```cuda
__inline__ __device__ float warp_reduce_sum(float v) {
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}
```

这段代码 5 步完成 32 个线程的归约，比每个线程写 shared memory 再树形归约快得多。LayerNorm、softmax、top-k 采样都会用到类似的模式。

## 4.5 Registers, Shared Memory, and Caches

GPU 的存储层次是软件可见的。写得快不快，很大程度取决于你把数据放在哪。

| 存储 | 位置 | 容量 | 延迟（周期） | 谁来管理 |
| --- | --- | --- | --- | --- |
| 寄存器 | SM 内 | 256 KB/SM | 1 | 编译器 |
| Shared memory | SM 内 | 最多 228 KB/SM | 20–30 | 程序员 |
| L1 cache | SM 内 | 与 shared memory 共享 | 30–40 | 硬件 |
| L2 cache | 芯片内 | 50 MB | 200 左右 | 硬件 |
| HBM | 片外 | 80 GB | 400–600 | 硬件 |

### 寄存器与 occupancy

每个 SM 有 65,536 个 32-bit 寄存器。一个线程最多用 255 个。寄存器用量直接限制能同时驻留多少线程：

$$
\text{threads per SM} = \left\lfloor \frac{65536}{\text{regs per thread}} \right\rfloor
$$

255 个寄存器时，只能放 256 个线程，即 8 个 warp；H100 的 SM 最多支持 64 个 warp，占用率只有 12.5%。这种 kernel 通常无法隐藏内存延迟。

反过来，如果寄存器压到 32 个，理论上能放 2048 个线程，占满 64 个 warp。编译器可以用 `__launch_bounds__` 限制寄存器数量，代价是可能溢出到 local memory。

### Shared memory 与 bank conflict

Shared memory 被分成 32 个 bank，每个 bank 4 字节宽。一个 warp 的 32 个线程如果访问 32 个不同的 bank，一次完成；如果两个线程访问同一个 bank 的不同地址，就发生 bank conflict，访问被串行化。

```cuda
__shared__ float tile[32][32];    // 有冲突
__shared__ float tile[32][33];    // 填充一列，消除冲突
```

`tile[32][32]` 里，`tile[tid][j]` 的地址是 `tid * 32 + j`，所有线程的 `tid * 32` 对 32 取模都是 0，全部落到同一个 bank。补一列变成 33，地址错开，冲突消失。这是矩阵转置和 GEMM 里最常见的一个修复。

### 全局内存合并访问

HBM 以 32 字节 sector 和 128 字节 cache line 为单位传输。一个 warp 的 32 个线程如果访问连续的 128 字节，硬件合并成一次事务；如果访问间隔很大，就会产生 32 次独立事务，有效带宽降到 1/32。

```cuda
// 合并：连续 float
out[i] = in[i] * 2.0f;

// 合并：float4，每个线程 16 字节，一个 warp 覆盖 512 字节
float4 v = in4[i];
v.x *= 2; v.y *= 2; v.z *= 2; v.w *= 2;
out4[i] = v;

// 不合并：跨步访问
out[i * stride] = in[i * stride] * 2.0f;
```

AoS（array of structs）和 SoA（struct of arrays）的选择也在这里起作用。如果 kernel 只需要结构体里的一个字段，AoS 会把整个结构体读进来，浪费带宽。LLM 里的 KV Cache 布局、attention 的 head 排布，都要考虑这一点。

### 异步拷贝与流水线

Hopper 之前，从 HBM 到 shared memory 要经过寄存器，还要显式同步。`cp.async` 让拷贝和计算重叠；Hopper 的 TMA（Tensor Memory Accelerator）更进一步，用一条指令搬一整块多维张量，并自动处理边界。

FlashAttention 的性能来自这套组合：分块加载 Q/K/V 到 shared memory，在片上完成 softmax，避免把 `S × S` 矩阵写回 HBM。它不是减少了浮点运算，而是把访存从 `O(S^2)` 降到了 `O(S^2 / tile)` 的片上操作。

## 4.6 HBM and Memory Bandwidth

HBM 把 DRAM die 堆叠起来，用 TSV（through-silicon via）垂直连接，每个 stack 有很宽的接口。

H100 的 3.35 TB/s 可以这样算出来：5 个 HBM3 stack，每个 1024 bit 接口，每 pin 5.2 Gbps：

$$
5 \times 1024 \times 5.2\text{e}9 / 8 \approx 3.33 \text{ TB/s}
$$

| GPU | 显存 | 带宽 | FP16 dense | 脊点（FLOP/byte） |
| --- | --- | --- | --- | --- |
| V100 | 32 GB HBM2 | 900 GB/s | 125 TFLOPS | 139 |
| A100 80GB | 80 GB HBM2e | 2.0 TB/s | 312 TFLOPS | 156 |
| H100 SXM | 80 GB HBM3 | 3.35 TB/s | 989 TFLOPS | 295 |
| H200 | 141 GB HBM3e | 4.8 TB/s | 989 TFLOPS | 206 |
| B200 | 192 GB HBM3e | 8 TB/s | 约 2,250 TFLOPS | 281 |

H200 是一个有意思的反例：容量和带宽都涨了，但计算没变，脊点反而从 295 降到 206。**带宽增长慢于算力增长，意味着越来越多的算子会掉到带宽受限区。**

### 延迟与带宽的换算

带宽是“每秒能搬多少字节”，延迟是“一个字节要等多久”。两者由一个叫 Little's Law 的关系连起来：

$$
\text{in-flight bytes} = \text{bandwidth} \times \text{latency}
$$

H100 的 3.35 TB/s、500 周期、1.98 GHz，延迟约 250 ns：

$$
3.35\text{e}12 \times 250\text{e}-9 \approx 838 \text{ KB}
$$

整个 GPU 需要约 840 KB 的数据同时在路上，才能跑满带宽。分摊到 132 个 SM，每个 SM 约 6.4 KB。一个 warp 用 `float4` 一次读 512 字节，需要 12 个 warp 同时在飞行中。

这解释了为什么低占用率的 kernel 跑不满带宽：不是带宽不够，是请求发得不够多。

### 有效带宽

规格表上的带宽是理论峰值。实际能跑到的比例取决于访问模式：

| 访问模式 | 典型有效带宽 |
| --- | --- |
| 大块连续拷贝 | 85%–92% |
| 合并的只读扫描 | 80%–90% |
| 带 stride 的访问 | 30%–60% |
| 随机小粒度访问 | 10%–30% |

做 roofline 分析时，用有效带宽而不是峰值，否则会低估带宽受限的程度。

## 4.7 PCIe, NVLink, and GPU Interconnects

单卡解决不了 70B 以上的模型，卡间链路就成了系统设计的一部分。

| 链路 | 单向带宽 | 双向带宽 | 典型用途 |
| --- | --- | --- | --- |
| PCIe 4.0 x16 | 32 GB/s | 64 GB/s | GPU 与 CPU、网卡 |
| PCIe 5.0 x16 | 64 GB/s | 128 GB/s | 新一代服务器 |
| NVLink 3（A100） | 300 GB/s | 600 GB/s | 8 卡节点内 |
| NVLink 4（H100） | 450 GB/s | 900 GB/s | 8 卡节点内 |
| NVLink 5（B200） | 900 GB/s | 1.8 TB/s | 机柜级 |
| InfiniBand NDR | 50 GB/s | — | 跨节点 |

NVLink 4 的 900 GB/s 是 PCIe 5.0 的 7 倍。这个差距决定了哪些并行策略必须放在节点内。

### 拓扑决定并行策略

张量并行每层要做两次 AllReduce，decode 时每个 token 要做 `2L` 次。70B、TP=8、decode，每次消息约 57 KB，一层两次，80 层合计约 4.6 MB/token。

- 走 NVLink 900 GB/s：约 5 微秒；
- 走 PCIe 4.0 32 GB/s：约 144 微秒。

单看传输时间，PCIe 也还能接受。但每次集合通信还有几微秒的固定延迟，160 次累计就是几百微秒到 1 毫秒。当 decode 的单 token 时间只有 5–10 毫秒时，这是 10%–20% 的开销。

数据并行不同：它每一步只同步一次梯度，消息大但次数少，可以跨节点走 InfiniBand。流水线并行传的是激活，量小，但依赖链长，对延迟敏感。

**一个经验规则：TP 留在 NVLink 域内，PP 和 DP 可以跨节点。**

### NVSwitch 与机柜级互联

8 卡节点内如果只用 NVLink 点对点连接，拓扑是不对称的：GPU 0 到 GPU 7 可能要经过两跳。NVSwitch 提供全互联，任意两卡之间都是单跳、全带宽。

GB200 NVL72 把这种思路推到机柜级：72 个 GPU 通过 NVLink 5 组成一个域，任意两卡之间的带宽接近节点内。这让“一个模型横跨 72 卡做张量并行”变得可行，也对调度和容错提出了新问题。

### 主机链路

模型加载、KV Cache offload、checkpoint 读写都要经过 PCIe。140 GB 的权重，PCIe 5.0 x16 单向 64 GB/s，理论最短 2.2 秒；实际受存储和驱动影响，通常在 10–60 秒。

推理系统里，PCIe 还承担健康检查、日志和请求收发。这些流量不大，但如果和模型加载或 KV offload 抢同一条链路，尾延迟会明显变差。

## 4.8 GPU Architecture Evolution

把几代数据中心 GPU 放在一起看，趋势很清楚。

| 年份 | GPU | 显存 | 带宽 | FP16 dense | 新特性 |
| --- | --- | --- | --- | --- | --- |
| 2016 | P100 | 16 GB HBM2 | 732 GB/s | 21 TFLOPS | 无 Tensor Core |
| 2017 | V100 | 32 GB HBM2 | 900 GB/s | 125 TFLOPS | 第一代 Tensor Core |
| 2020 | A100 | 80 GB HBM2e | 2.0 TB/s | 312 TFLOPS | TF32、BF16、MIG |
| 2022 | H100 | 80 GB HBM3 | 3.35 TB/s | 989 TFLOPS | FP8、TMA、Transformer Engine |
| 2024 | H200 | 141 GB HBM3e | 4.8 TB/s | 989 TFLOPS | 大显存 |
| 2024 | B200 | 192 GB HBM3e | 8 TB/s | 约 2,250 TFLOPS | FP4、第五代 Tensor Core |

### 趋势一：精度越来越低

FP32 → TF32（10 bit 尾数）→ FP16/BF16 → FP8 → FP4。每降一级，同样的芯片面积和带宽能换来大约两倍的吞吐。

这对系统的影响是双向的：

- 计算变快，prefill 的算力上限提高；
- 权重和 KV Cache 的字节数减半，decode 的带宽压力减半。

但低精度的数值范围也变窄。FP16 的最大值是 65,504，FP8 E4M3 只有 448。attention 的分数、softmax 的指数、梯度的累积都可能溢出。量化和混合精度不是“换个 dtype”那么简单，它要求逐层的数值分析。

### 趋势二：算力增长快于带宽

从 A100 到 H100，FP16 算力涨了 3.2 倍，带宽只涨了 1.7 倍。脊点从 156 升到 295。

这意味着：**同一个 kernel，在两代 GPU 上的瓶颈可能不同。** 在 A100 上还能算得动的形状，到 H100 上可能已经掉到带宽受限区。优化代码时不能只看“算力够不够”，要看算术强度有没有跟着硬件一起涨。

### 趋势三：scale-up 域越来越大

NVLink 从 8 卡域走向 72 卡域。机柜变成一个新的部署单位，电力和散热在机柜层面设计，故障域也在机柜层面划分。

这让更大规模的张量并行成为可能，但也让“一个请求要用多少卡”变成一个调度问题：用 72 卡跑一个小 batch 的 decode，通信开销可能比计算还大。

### 趋势四：专用单元越来越多

TMA 负责搬数据，Transformer Engine 负责 FP8 的动态缩放，DPX 指令处理动态规划，Tensor Memory 在 Blackwell 上给 Tensor Core 提供专用操作数存储。

通用 CUDA core 的比例在下降。写 kernel 的人必须知道这些单元怎么用，否则只能拿到峰值的一小部分。

### 对软件的含义

硬件演进给软件提了三个要求：

1. **新指令要跟上。** 不用 FP8 和 TMA 的 kernel，在 H100 上可能只有峰值的三分之一。
2. **带宽优化更重要。** 算力涨得快，带宽涨得慢，融合、量化、减少中间张量这些手段的收益在变大。
3. **通信要能重叠。** 卡越多，通信占比越高，不能重叠的通信会直接吃掉扩展性。

下一章把这些硬件参数变成分析工具：给定一个 kernel 或一个模型，怎么判断它受什么限制、理论上能跑多快、实际差在哪里。

## 本章小结

- CPU 的 SIMD 宽度决定理论峰值，但实际性能受频率、NUMA 和内存带宽约束；CPU 上的 LLM 推理几乎总是带宽受限。
- NUMA 距离会显著改变延迟和带宽；模型加载和线程绑定要与拓扑对齐。
- GPU 用大量简单线程隐藏内存延迟，SM、warp 和 occupancy 是理解性能的基本单位。
- 寄存器、shared memory 和合并访问决定一个 kernel 能不能接近峰值；bank conflict 和跨步访问是最常见的两个坑。
- HBM 带宽和延迟由 Little's Law 联系；跑满带宽需要足够的并发请求。
- NVLink 与 PCIe 差一个数量级，拓扑决定哪种并行策略放在哪里。
- GPU 的算力增长快于带宽增长，低精度和专用单元是主要演进方向，软件必须跟着变。

## 延伸阅读

- NVIDIA, [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).
- NVIDIA, [NVIDIA H100 Tensor Core GPU Architecture](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c).
- NVIDIA, [NVIDIA A100 Tensor Core GPU Architecture](https://resources.nvidia.com/en-us-ampere-architecture/nvidia-a100-tensor-core-gpu-architecture-whitepaper).
- Hennessy & Patterson, [Computer Architecture: A Quantitative Approach](https://www.elsevier.com/books/computer-architecture/hennessy/978-0-12-811905-1), 6th edition, 2017.
- Jia et al., [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/abs/1804.06826), 2018.
- Luo et al., [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://arxiv.org/abs/2402.13499), 2024.
- Williams et al., [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://dl.acm.org/doi/10.1145/1498765.1498785), 2009.
- Little, [A Proof for the Queuing Formula: L = λW](https://doi.org/10.1287/opre.9.3.383), 1961.
