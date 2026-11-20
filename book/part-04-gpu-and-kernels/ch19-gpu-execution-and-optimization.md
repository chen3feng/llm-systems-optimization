---
chapter: 19
part: 4
title: GPU Execution and Optimization
level: tbd
status: draft
prerequisites: [4, 17, 18]
lab:
---

# Chapter 19. GPU Execution and Optimization

同一个 kernel，改一个 block 大小、加一行 `__launch_bounds__`、把 shared memory 换成 warp shuffle，性能可能差两倍。这些改动背后没有魔法，只有四个变量：占用率够不够、寄存器够不够、warp 有没有发散、指令和访存能不能填满流水线。

这一章讲 GPU 执行层面的优化。它不是“技巧合集”，而是从执行模型出发，解释每个手段在解决什么问题、什么时候无效。很多优化之所以失败，是因为瓶颈根本不在它针对的地方。

## 19.1 Occupancy

占用率是驻留 warp 数与硬件上限的比值：

$$
\text{occupancy} = \frac{\text{active warps per SM}}{\text{max warps per SM}}
$$

H100 每个 SM 最多 64 个 warp。占用率由寄存器、shared memory、block 大小共同决定。

### 理论占用率与实测占用率

`cudaOccupancyMaxActiveBlocksPerMultiprocessor` 给出理论值：资源允许时能驻留多少 block。实测占用率由 profiler 统计，可能更低，因为：

- block 的调度有粒度，尾部可能空转；
- 某些指令会占用额外资源；
- 驱动保留了少量资源。

两者差距大，说明问题不在资源限制，而在调度或负载均衡。

### 占用率什么时候重要

占用率的作用是隐藏延迟。当 warp 因为等待内存而停顿时，调度器切换到其他 warp。warp 越多，能掩盖的停顿越长。

它重要，当：

- kernel 是访存受限的；
- 每次访存延迟高（HBM）；
- 指令级并行不足。

它不重要，当：

- kernel 是计算密集的，指令流水线本来就满；
- 每个线程有大量 ILP，单个 warp 就能填满流水线；
- 瓶颈在 HBM 带宽，而不是延迟。

GEMM 的占用率常常只有 25%，但性能接近峰值；elementwise kernel 占用率接近 100%，仍然受带宽限制。占用率是手段，不是目标。

## 19.2 Register Pressure

寄存器压力是 occupancy 下降最常见的原因。

### 什么时候会 spill

- 单个线程的局部数组太大；
- 循环展开过度，同时活跃的变量太多；
- 编译器为了减少分支，把多个路径的变量都保留下来。

```bash
ptxas info: Used 255 registers, 128 bytes spill stores, 128 bytes spill loads
```

看到 spill 就要处理。spill 的数据在 local memory，实际落在 HBM，每次访问几百周期。

### 减少寄存器用量

- 缩小 tile：每个线程算更少的输出；
- 缩短 live range：让变量在最后一次使用后立即失效；
- 用 shared memory 替代大数组；
- 降低展开因子；
- 用 `__launch_bounds__` 给编译器上限，让它自己权衡。

### 增加寄存器用量

反过来，计算密集的 kernel 常常需要更多寄存器来保存累加器和操作数。这时候应该主动放宽限制：

```cuda
__global__ void __launch_bounds__(128, 1) gemm(...) {
  // 允许每个线程用更多寄存器，减少 spill
}
```

### 权衡

寄存器、占用率、ILP 三者互相制约：

- 寄存器多 → 占用率低，但每个线程的 ILP 高；
- 寄存器少 → 占用率高，但可能 spill 或 ILP 不足。

没有通用答案。判断依据是 profiler 里的 stall reason：如果 stall 在 `long_scoreboard`（等内存），可能需要更多 warp；如果 stall 在 `math_pipe_throttle`，说明计算已经饱和，再加占用率没用。

## 19.3 Warp Divergence

Warp 内的 32 个线程共享一个指令流。分支不一致时，硬件串行执行每条路径，只有对应 lane 有效。

```cuda
if (threadIdx.x < 16) {
  y = f(x);   // 16 个 lane 有效
} else {
  y = g(x);   // 另外 16 个 lane 有效
}
```

两条路径各执行一次，总时间接近两者之和，而不是并行。利用率 50%。

### 常见的发散来源

- 边界检查：最后一个 block 里只有部分线程有效；
- 数据相关的分支：不同 token 走不同路径；
- 短循环：每个线程的循环次数不同；
- attention 的 causal mask：不同 query 看到的 key 数不同。

### 减少发散

**调整数据布局**，让同一 warp 的线程走同一分支。比如按长度排序后再处理变长序列：

```text
排序前：[3, 100, 5, 100, 7, 100]  → warp 内频繁发散
排序后：[3, 5, 7, 100, 100, 100]  → warp 内基本一致
```

**用 select 代替分支**：

```cuda
// 有分支
float y = (x > 0) ? x : 0.0f;

// 无分支（编译器通常会自动生成）
float y = fmaxf(x, 0.0f);
```

**把边界检查提到 warp 粒度**：如果整个 warp 都越界，直接返回；只有部分越界时再逐线程判断。

### 发散的代价被高估了吗

不一定。现代 GPU 有独立的 branch unit，简单分支的开销很小。真正昂贵的是：

- 两条路径都很长；
- 发散导致访存不合并；
- 发散破坏了 warp 级原语的假设。

不要为了消除一个两行的分支把代码写得没法读。先测量。

## 19.4 Instruction Throughput

H100 的每个 SM 有 4 个 warp scheduler，每个每周期发射一条指令。也就是每周期最多 4 条 warp 指令、128 个线程操作。FP32 有 128 个 core，正好对应。

### 指令 mix

一个 kernel 的指令通常包括：

- 浮点运算（FMA、ADD、MUL）；
- 整数运算（地址计算、循环）；
- 访存（LDG、STG、LDS）；
- 控制流（BRA、EXIT）；
- 特殊函数（MUFU，如 exp、rsqrt）。

只有浮点运算计入峰值 FLOPS。地址计算和循环开销会稀释有效算力。

### 减少非计算指令

- 用指针递增代替 `i * stride` 的重复乘法；
- 把循环不变量提到循环外；
- 用 `#pragma unroll` 减少循环控制开销；
- 用向量化访存减少 LDG 数量。

```cuda
// 每次迭代都算地址
for (int i = 0; i < n; ++i) out[i] = in[i] * a;

// 指针递增
const float* p = in;
float* q = out;
for (int i = 0; i < n; ++i) *q++ = *p++ * a;
```

现代编译器通常能自动做这些优化，但复杂的下标表达式不一定。

### 特殊函数单元

`expf`、`logf`、`sinf` 走 MUFU，吞吐远低于 FMA。softmax 里的 `exp` 是热点之一。

- 用 `__expf()` 代替 `expf()`，精度略低但快得多；
- 用 `exp2f()` 配合换底，硬件对 2 的幂更友好；
- 能查表就查表。

FlashAttention 里把 softmax 的缩放换成 `exp2`，是常见的优化。

## 19.5 Latency Hiding

GPU 靠大量并发 warp 隐藏延迟。需要多少并发，可以用 Little's law 估算：

$$
\text{bytes in flight} = \text{bandwidth} \times \text{latency}
$$

每个 SM 分到约 25 GB/s。HBM 延迟约 600 周期，1.8 GHz 下是 333 ns：

$$
25 \times 10^9 \times 333 \times 10^{-9} \approx 8.3 \text{ KB}
$$

一个 warp 发一次 128 字节的合并 load，在途数据是 128 字节。要填满 8.3 KB，需要约 65 个 warp——正好是 H100 的满占用率。

这个计算说明：满占用率下，每个 warp 只需要保持一次未完成的 load，就能把 HBM 带宽跑满。如果每个 warp 能发多次 load（ILP），所需 warp 数还可以减半。

### 两种隐藏方式

- **TLP**（thread-level parallelism）：更多 warp，靠占用率；
- **ILP**（instruction-level parallelism）：每个线程发出多个独立访存，靠展开和重排。

```cuda
// ILP 低：一次 load，等结果
float a = in[i];
out[i] = a * 2;

// ILP 高：四次独立 load，可以重叠
float4 v = in4[i];
out4[i] = make_float4(v.x * 2, v.y * 2, v.z * 2, v.w * 2);
```

向量化访存同时提高了 ILP 和合并度，是 elementwise kernel 最有效的优化之一。

### 延迟隐藏的极限

当带宽已经饱和，再加 warp 不会更快，只会增加竞争。profiler 里表现为 `dram__throughput` 接近 100%，此时优化方向是减少字节数，而不是提高并发。

## 19.6 Memory-bound versus Compute-bound Kernels

Roofline 模型把 kernel 分成两类：

- **memory-bound**：算术强度低于脊点，性能由带宽决定；
- **compute-bound**：算术强度高于脊点，性能由算力决定。

H100 FP16 的脊点约 295 FLOP/Byte（989 TFLOPS / 3.35 TB/s）。LLM 里的典型 kernel：

| Kernel | 算术强度 | 类型 |
| --- | --- | --- |
| elementwise（读 1 写 1） | 0.25 | memory-bound |
| LayerNorm | ~1–2 | memory-bound |
| decode attention（GQA） | ~8 | memory-bound |
| prefill attention | `S/4`，长序列时很高 | 取决于序列长度 |
| GEMM（大 M/N/K） | 数百到数千 | compute-bound |
| decode GEMM（batch 1） | ~1 | memory-bound |

### 优化方向

**memory-bound**：

- 减少字节：量化、压缩、跳过不用的数据；
- 提高合并度：SoA、向量化、对齐；
- 融合：把多个 kernel 合成一个，避免中间结果写回 HBM；
- 提高 L2 命中：分块、持久化窗口。

**compute-bound**：

- 用 Tensor Core；
- 降低精度（FP16 → FP8）；
- 提高 tile 复用，减少数据搬运；
- 优化指令 mix，减少非计算指令。

方向搞反了，优化就没有效果。一个 elementwise kernel 再怎么调 tile，也不会超过带宽上限。

## 19.7 Warp-level Primitives

同一个 warp 的线程可以通过 shuffle 直接交换寄存器，不需要 shared memory。

```cuda
float v = __shfl_down_sync(0xffffffff, val, offset);
```

`0xffffffff` 是参与线程的掩码，`offset` 是源 lane 的偏移。

### Warp 归约

```cuda
__inline__ __device__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}
```

5 步完成 32 个值的归约，没有 shared memory，没有 `__syncthreads()`。

对比 shared memory 版本：

- 需要 5 轮循环；
- 每轮一次 `__syncthreads()`；
- 需要 32 个 float 的 shared memory。

Shuffle 版本的延迟低一个数量级，是 reduction、softmax、LayerNorm 的标准做法。

### 其他原语

| 原语 | 作用 |
| --- | --- |
| `__shfl_sync` | 从指定 lane 取值 |
| `__shfl_up_sync` / `__shfl_down_sync` | 向上/向下偏移取值 |
| `__shfl_xor_sync` | 按 XOR 交换，适合蝶形归约 |
| `__ballot_sync` | 收集谓词，返回 32 位掩码 |
| `__any_sync` / `__all_sync` | warp 内是否有/全有 |
| `__popc` | 统计掩码中 1 的个数 |

`__ballot_sync` 常用于实现 warp 内的 compaction：把满足条件的 lane 紧凑排列。

### 掩码必须正确

`__shfl_sync` 的掩码必须包含所有参与线程。在发散分支里调用，掩码和实际活跃 lane 不匹配，结果是未定义的。

## 19.8 Cooperative Groups

Cooperative groups 把线程组织成显式的组，提供集合操作。

```cuda
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void kernel(float* out, const float* in) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

  float v = in[threadIdx.x];
  float sum = cg::reduce(warp, v, cg::plus<float>());
  if (warp.thread_rank() == 0) out[blockIdx.x] = sum;
}
```

好处是：

- 代码表达意图，而不是手写下标；
- 不同的组大小（2、4、8、16、32）用同一套接口；
- `cg::reduce` 会自动选择 shuffle 或 shared memory。

### Grid 同步

```cuda
cg::grid_group grid = cg::this_grid();
grid.sync();
```

要求用 `cudaLaunchCooperativeKernel` 启动，且所有 block 能同时驻留。它适合需要多轮全局同步的算法，比如迭代求解。

代价是 grid 大小受占用率限制，不能随意扩大。

## 19.9 Persistent Kernels

普通 kernel 的 block 数由数据量决定，可能远超硬件容量，GPU 分波执行。每一波结束都有调度开销，最后一波还可能很空。

Persistent kernel 反过来：启动固定数量的 block（通常是 `SM 数 × 每 SM 驻留 block 数`），每个 block 循环领取任务。

```cuda
__global__ void persistent(float* out, const float* in, int n, int* counter) {
  __shared__ int tile_id;
  while (true) {
    if (threadIdx.x == 0) tile_id = atomicAdd(counter, 1);
    __syncthreads();
    if (tile_id >= num_tiles) break;

    int base = tile_id * TILE;
    for (int i = threadIdx.x; i < TILE; i += blockDim.x) {
      out[base + i] = in[base + i] * 2;
    }
    __syncthreads();
  }
}
```

好处：

- 消除波之间的调度间隙；
- 避免尾效应；
- 可以在 kernel 内做全局协调。

代价：

- 需要全局计数器，原子操作有开销；
- 负载不均时，先完成的 block 空转；
- 不能依赖 block 的执行顺序。

Persistent kernel 在 decode 这种每步工作量小、launch 频繁的场景里特别有价值。它把一个 kernel 变成常驻的服务循环，减少了 launch 和调度的固定开销。

## 19.10 CUDA Graphs

decode 的每一步包含上千个 kernel launch。每次 launch 的 CPU 开销约 5–10 微秒，GPU 侧约 3–5 微秒。70B 模型 decode 一步约 42 ms，1,000 次 launch 累计 3–5 ms，占 10% 左右。

CUDA Graphs 把这些 launch 捕获成一张有向无环图，之后一次提交、重复执行：

```cuda
cudaGraph_t graph;
cudaGraphExec_t exec;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
for (int layer = 0; layer < num_layers; ++layer) {
  rms_norm<<<grid, block, 0, stream>>>(...);
  qkv_gemm<<<grid, block, 0, stream>>>(...);
  attention<<<grid, block, 0, stream>>>(...);
  // ...
}
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

for (int step = 0; step < max_steps; ++step) {
  cudaGraphLaunch(exec, stream);
  cudaStreamSynchronize(stream);
}
```

Graph 的 launch 开销约 1–2 微秒，与图的大小基本无关。

### 适用条件

- 形状固定：batch、序列长度、并行配置不变；
- 没有 host 端同步；
- 没有数据相关的控制流。

变长 batch 需要重新捕获，或者用 `cudaGraphExecKernelNodeSetParams` 更新节点参数。前者开销大，后者要求参数布局一致。

### Graph 不等于更快

Graph 只减少 launch 开销，不改变 kernel 本身的执行时间。如果 kernel 是带宽受限的，Graph 带来的提升就是那 10% 的 launch 开销。

它也不能跨 stream 自动并行：捕获时是什么依赖关系，重放时就是什么依赖关系。

## 本章小结

- 占用率用于隐藏延迟，不是越高越好；计算密集的 kernel 可以低占用率运行。
- 寄存器压力会导致 spill 或占用率下降，`__launch_bounds__` 是最直接的控制手段。
- Warp 发散让分支串行执行；调整数据布局、去分支、warp 粒度边界检查都能缓解。
- 指令吞吐受非计算指令和特殊函数影响，减少地址计算、用 `exp2` 都能提升有效算力。
- Little's law 给出隐藏延迟所需的在途字节数；满占用率下每个 warp 一次 128 字节 load 约能填满 HBM。
- 先判断 memory-bound 还是 compute-bound，再选择优化方向；方向错了，调参没有意义。
- Warp shuffle 用寄存器交换替代 shared memory，是归约和 scan 的首选。
- Cooperative groups 提供显式的组和 grid 同步，代价是 grid 大小受限。
- Persistent kernel 消除波间调度和尾效应，适合 decode 这种小而频繁的任务。
- CUDA Graphs 把上千次 launch 压成一次提交，适合形状固定的推理循环。

## 延伸阅读

- NVIDIA, [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).
- NVIDIA, [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).
- Williams et al., [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://dl.acm.org/doi/10.1145/1498765.1498785), 2009.
- NVIDIA, [Cooperative Groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups).
- NVIDIA, [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs).
- Harris, [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/), 2018.
