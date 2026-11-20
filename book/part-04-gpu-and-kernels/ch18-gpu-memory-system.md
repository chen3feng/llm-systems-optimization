---
chapter: 18
part: 4
title: GPU Memory System
level: tbd
status: draft
prerequisites: [4, 17]
lab:
---

# Chapter 18. GPU Memory System

一个 H100 的 HBM 带宽是 3.35 TB/s，但单个 SM 只能分到约 25 GB/s。要让带宽跑满，需要一百多个 SM 同时工作；而每个 SM 内部，又要靠寄存器、shared memory 和 cache 把重复访问挡在 HBM 之前。

LLM 的 kernel 绝大多数是访存受限的。GEMM 的算术强度可以做得很高，但 attention、LayerNorm、softmax、elementwise 的瓶颈都在字节上。理解内存层次，就是理解这些 kernel 为什么慢、以及能快到哪里。

这一章按层次讲：寄存器、shared memory、L1/L2、HBM、host 内存，每一层讲容量、延迟、带宽和访问规则，最后落到合并访问、bank conflict、对齐和异步拷贝这些直接影响性能的细节。

## 18.1 GPU Memory Hierarchy

H100 的内存层次大致如下：

| 层次 | 容量 | 延迟（周期） | 带宽 | 作用域 |
| --- | --- | --- | --- | --- |
| 寄存器 | 256 KB/SM | ~1 | 极高 | 线程 |
| Shared memory | 228 KB/SM | ~20–30 | ~30 TB/s 聚合 | Block |
| L1 cache | 与 shared memory 共享 | ~30 | — | SM |
| L2 cache | 50 MB | ~200 | ~10 TB/s | 全 GPU |
| HBM | 80 GB | ~500–800 | 3.35 TB/s | 全 GPU |
| Host 内存 | TB 级 | ~10 μs（PCIe） | 64 GB/s（PCIe 5.0 x16） | 跨设备 |

几个数量级一目了然：寄存器和 HBM 的延迟差 500 倍以上，带宽差两个数量级。优化的核心就是把访问尽量留在上层。

### 每 SM 的带宽份额

3.35 TB/s 分给 132 个 SM，平均每个 SM 约 25 GB/s。一个 SM 同一时刻最多驻留 64 个 warp、2,048 个线程；如果每个线程每周期发一个 4 字节的 load，理论需求约 `2,048 × 4 × 1.8G ≈ 14.7 TB/s`，远超 HBM 能提供的量。

所以“访存受限”不是 SM 发不出请求，而是 HBM 供不上。这也解释了为什么增加占用率有时没用：瓶颈在 HBM，不在延迟隐藏。

### 缓存的作用

L1 和 shared memory 共享同一块物理存储（H100 上合计 256 KB/SM，其中 shared memory 最多 228 KB）。L2 是全局共享的，50 MB，所有 SM 的请求都经过它。

L2 的命中率对性能影响很大。LLM 的权重在 decode 时每个 token 都要读一遍，但模型是几十 GB，L2 装不下，命中率接近 0。这也是 decode 完全受 HBM 带宽限制的原因。

## 18.2 Registers and Register Allocation

寄存器是最快的存储，也是最稀缺的资源。

### 寄存器文件

每个 SM 有 65,536 个 32 位寄存器（256 KB）。线程的局部变量、地址计算、循环计数器都放在寄存器里。

每个线程最多使用 255 个寄存器。超过时会 spill 到 local memory——物理上在 HBM，延迟和带宽都差一个数量级。

### 分配粒度

寄存器按 warp 分配，每个线程的寄存器数会向上取整到分配粒度（通常是 8）。一个用 40 个寄存器的 kernel，实际按 40 算；用 33 个，可能按 40 算。

```bash
nvcc -Xptxas -v kernel.cu
```

编译输出会显示每个 kernel 的寄存器数、shared memory 和 spill 情况。这是优化前必看的信息。

### 占用率与寄存器

寄存器数直接限制每个 SM 能驻留多少 warp：

$$
\text{warps/SM} = \left\lfloor \frac{65{,}536}{\text{regs/thread} \times 32} \right\rfloor
$$

- 32 个寄存器/线程：64 个 warp，满占用；
- 64 个寄存器/线程：32 个 warp，50%；
- 128 个寄存器/线程：16 个 warp，25%。

GEMM 这类计算密集的 kernel 通常用 128–255 个寄存器，占用率低但每个线程做了大量计算，性能反而更好。

### 控制寄存器用量

```cuda
__global__ void __launch_bounds__(256, 2) kernel(...) {
  // 编译器会保证 256 线程/block 时至少 2 个 block 可驻留
}
```

`__launch_bounds__` 给编译器一个约束：每个线程最多能用多少寄存器。编译器会据此调整，必要时 spill。

也可以用 `-maxrregcount=N` 全局限制，但它是全局的，不如 `__launch_bounds__` 精确。

### 什么时候该 spill

几乎永远不该。spill 意味着寄存器不够，数据被赶到 local memory。如果编译器报告 spill，先看是不是循环里用了太大的数组，或者 `#pragma unroll` 展开过度。

## 18.3 Shared Memory

Shared memory 是 block 内线程共享的片上存储，延迟约 20–30 周期，带宽约 128 字节/周期/SM。

### 分配

```cuda
__shared__ float tile[32][33];        // 静态
extern __shared__ float dyn[];        // 动态，launch 时指定大小
```

动态 shared memory 的大小在 launch 的第三个参数里给：

```cuda
kernel<<<grid, block, 32 * 33 * sizeof(float)>>>(...);
```

每个 block 默认最多 48 KB，超过要用 `cudaFuncSetAttribute` 申请，H100 上最高 227 KB。

### 为什么需要 shared memory

Global memory 的访问要经过 L2 和 HBM，延迟几百周期。如果多个线程要读同一块数据，先搬到 shared memory 再读，能把重复访问变成一次 HBM 读取。

矩阵乘法的 tile、attention 的 K/V 块、归约的中间结果，都放在 shared memory 里。

### Bank 结构

Shared memory 被分成 32 个 bank，每个 bank 4 字节宽。一次访问中，如果 32 个线程访问 32 个不同的 bank，就是无冲突的；如果两个线程访问同一 bank 的不同地址，就要串行。

```cuda
__shared__ float s[32][32];
s[threadIdx.x][0] = ...;   // 所有线程访问同一列 → 32 路冲突
```

第 `i` 行第 `j` 列的元素落在 bank `(i * 32 + j) % 32 = j`。所以按列访问时，所有线程都落在 bank 0，冲突 32 次。

### Padding

把数组改成 `s[32][33]`：

```cuda
__shared__ float s[32][33];
s[threadIdx.x][0] = ...;   // bank = (i * 33) % 32 = i，无冲突
```

多出来的一列打破了 32 的整除关系。这是最常用的 padding 技巧。

### 广播

如果 32 个线程访问同一个地址，硬件会广播，不算冲突。读取同一个权重、同一个 scale 时不用担心。

## 18.4 Global Memory

Global memory 就是 HBM，容量最大、延迟最高。

### 访问粒度

Global memory 的访问以 32 字节 sector 为单位，以 128 字节 cache line 为单位。一次 warp 的访问会被拆成若干个 sector。

理想情况：32 个线程各读 4 字节，地址连续，正好 128 字节，4 个 sector，一次事务完成。

最坏情况：32 个线程各读 4 字节，地址相隔 128 字节，需要 32 个 sector，带宽利用率 12.5%。

### 合并访问

```cuda
// 好：连续地址
out[i] = in[i];

// 坏：跨步访问
out[i] = in[i * stride];
```

跨步访问是性能杀手。矩阵按行存储时，按列遍历会导致每个线程跨一整行，几乎每次访问都要新的 cache line。

### 向量化

一次读 16 字节（`float4`）比读 4 次 4 字节更高效：

```cuda
const float4* in4 = reinterpret_cast<const float4*>(in);
float4 v = in4[i];
```

要求地址 16 字节对齐，元素数是 4 的倍数。编译器有时能自动向量化，但显式写更可靠。

### L2 与持久化

L2 是所有 SM 共享的。默认情况下，数据用完后会被淘汰。Hopper 支持给一段内存设置持久化窗口：

```cuda
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr = ptr;
attr.accessPolicyWindow.num_bytes = bytes;
attr.accessPolicyWindow.hitRatio = 1.0;
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

它适合反复读取的小表，比如 MoE 的路由表或 embedding 的一部分。对几十 GB 的权重没有意义。

## 18.5 L1 and L2 Cache

### L1

L1 是每个 SM 私有的，和 shared memory 共用物理存储。H100 上合计 256 KB，可以配置成不同的 carveout。

L1 缓存 global load，但不缓存 global store（store 直接写到 L2）。所以读多写少的访问模式更容易受益。

`__ldg()` 或 `const __restrict__` 可以提示编译器走只读缓存路径，在旧架构上有用；现代架构上编译器通常能自己判断。

### L2

L2 是全局的，50 MB，分成两个 25 MB 的 partition（H100）。所有 HBM 访问都经过 L2。

L2 的带宽约 10 TB/s，是 HBM 的 3 倍。如果数据能命中 L2，有效带宽会显著提高。

### 缓存命中率

LLM 推理的缓存行为：

- **权重**：每个 token 读一遍，模型远大于 L2，命中率接近 0；
- **KV Cache**：每个请求的 KV 只读一次，命中率也低；
- **激活**：同一层内可能被多个 kernel 读写，有局部性，但 kernel 之间通常已经写回 HBM。

所以 decode 的瓶颈就是 HBM 带宽，cache 帮不上忙。这也是为什么权重压缩（量化）能直接提速：它减少的是 HBM 字节数。

## 18.6 Memory Coalescing

合并访问是 global memory 优化的第一原则。

### 例子：行优先遍历

```cuda
// 合并：线程 i 读 a[i]
__global__ void copy(float* dst, const float* src, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}
```

32 个线程的地址是 `base + 0..31`，连续 128 字节，4 个 sector，一次事务。

### 例子：跨步遍历

```cuda
// 不合并：线程 i 读 a[i * 4]
int i = blockIdx.x * blockDim.x + threadIdx.x;
dst[i] = src[i * 4];
```

32 个线程跨越 512 字节，需要 16 个 sector，但只用了其中 128 字节。带宽利用率 25%。

### 结构体数组 vs 数组结构体

```cuda
struct Particle { float x, y, z, w; };
Particle p[N];

// 每个线程读 p[i].x，跨步 16 字节，利用率 25%
```

改成 SoA：

```cuda
float x[N], y[N], z[N], w[N];
```

每个数组内部连续，访问 `x[i]` 就是合并的。GPU 代码里应优先用 SoA。

### 转置

矩阵转置是典型的非合并访问：读一行是连续的，写到目标矩阵的一列就变成跨步。

解决办法是用 shared memory 做中转：

```cuda
__global__ void transpose(const float* in, float* out, int n) {
  __shared__ float tile[32][33];
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  tile[threadIdx.y][threadIdx.x] = in[y * n + x];
  __syncthreads();
  x = blockIdx.y * 32 + threadIdx.x;
  y = blockIdx.x * 32 + threadIdx.y;
  out[y * n + x] = tile[threadIdx.x][threadIdx.y];
}
```

读和写都变成合并的，代价是一次 shared memory 往返。

## 18.7 Bank Conflicts

Bank conflict 只发生在 shared memory。

### 冲突的判定

32 个 bank，地址 `addr` 落在 bank `(addr / 4) % 32`。

同一个 warp 内，如果多个线程访问同一 bank 的不同地址，就冲突，串行执行。

```cuda
__shared__ float s[32][32];
float v = s[threadIdx.x][threadIdx.y];
```

当 `threadIdx.y` 固定、`threadIdx.x` 变化时，地址是 `x * 32 + y`，bank 是 `(x * 32 + y) % 32 = y`，所有线程都在同一个 bank——32 路冲突。

### 解决

- Padding：`s[32][33]`；
- 改变布局：按行存而不是按列；
- Swizzle：XOR 地址的低位，CUTLASS 常用。

```cuda
// XOR swizzle
int idx = (row * 32) ^ (col & 31);
```

Swizzle 不增加存储，但让访问模式打散到不同 bank，适合 tile 较大的 GEMM。

### 冲突的代价

n 路冲突意味着 n 个周期完成一次访问，而不是 1 个。32 路冲突直接把 shared memory 带宽降到 1/32。

但 shared memory 仍然比 HBM 快，所以有冲突不一定比走 global 慢；要结合整体判断。

## 18.8 Memory Alignment

### 对齐要求

- `float4` 要求 16 字节对齐；
- `float2` 要求 8 字节；
- `double` 要求 8 字节；
- Tensor Core 的操作数通常要求 128 字节对齐。

未对齐的访问可能被拆成多次事务，甚至触发硬件异常。

### 分配的对齐

`cudaMalloc` 返回的指针至少 256 字节对齐。自己管理内存池时，要保证每个子分配的起始地址满足最严格的对齐要求。

```cuda
size_t aligned = (size + 255) & ~size_t(255);
```

### 结构体对齐

```cuda
struct __align__(16) Vec4 { float x, y, z, w; };
```

如果结构体要按 16 字节读取，必须显式对齐，否则编译器可能按 4 字节布局。

### 边界 padding

矩阵的行长度不是 4 的倍数时，向量化读取会越界。解决办法：

- 把行长度 padding 到 4 或 8 的倍数；
- 或者对最后几个元素用标量处理。

LLM 的 hidden size（如 4096、8192）通常已经是 128 的倍数，但 batch 和序列长度不一定。

## 18.9 Pinned Host Memory

### 为什么需要

GPU 通过 DMA 访问 host 内存。普通（pageable）内存可能被操作系统换出，驱动必须先把它锁住、拷到内部缓冲区，DMA 才能进行。这个过程是同步的，host 会阻塞。

Pinned（page-locked）内存不会被换出，DMA 可以直接访问：

```cuda
float* h;
cudaMallocHost(&h, bytes);      // 分配 pinned
cudaHostRegister(h, bytes, 0);  // 或把已有内存锁住
```

只有 pinned 内存才能配合 `cudaMemcpyAsync` 实现真正的异步传输。

### 代价

- 分配慢；
- 占用物理内存，不能换出；
- 分配过多会拖慢整个系统。

通常只给传输缓冲区用，不要给所有 host 内存用。

### 写合并内存

```cuda
cudaHostAlloc(&h, bytes, cudaHostAllocWriteCombined);
```

Write-combined 内存适合“只写不读”的场景，比如把结果从 GPU 拷回 host。读它会非常慢。

## 18.10 Unified Memory

统一内存让 host 和 device 共用一套地址：

```cuda
float* x;
cudaMallocManaged(&x, bytes);
kernel<<<grid, block>>>(x);
```

驱动按页迁移数据：CPU 访问时迁到 host，GPU 访问时迁到 device。

### 性能陷阱

- 第一次访问触发 page fault，延迟可达毫秒级；
- 频繁跨设备访问会导致页抖动（thrashing）；
- 迁移是整页的，可能搬走大量暂时不用的数据。

### 预取与建议

```cuda
cudaMemPrefetchAsync(x, bytes, device_id, stream);
cudaMemAdvise(x, bytes, cudaMemAdviseSetReadMostly, device_id);
```

预取可以隐藏迁移延迟；`ReadMostly` 告诉驱动这份数据只读，可以复制到多个设备而不失效。

### 适用场景

统一内存适合：

- 数据结构复杂、指针多的场景；
- 原型开发；
- 显存小于数据量、需要 oversubscription 的情况。

不适合对延迟敏感的推理主路径。生产推理引擎通常自己管理显存池。

## 18.11 Asynchronous Memory Transfers

### 基本用法

```cuda
cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
```

配合 pinned 内存和多个 stream，可以让拷贝与计算重叠。

### 双缓冲

```cuda
for (int i = 0; i < n; ++i) {
  int cur = i % 2;
  cudaMemcpyAsync(d_in[cur], h_in + i * chunk, bytes,
                  cudaMemcpyHostToDevice, stream);
  kernel<<<grid, block, 0, stream>>>(d_in[cur], d_out[cur]);
  cudaMemcpyAsync(h_out + i * chunk, d_out[cur], bytes,
                  cudaMemcpyDeviceToHost, stream);
}
```

一个 buffer 在传输时，另一个在计算。理想情况下总时间由 `max(传输, 计算)` 决定，而不是两者之和。

### cp.async

Ampere 引入的 `cp.async` 允许把 global memory 直接拷到 shared memory，不经过寄存器：

```cuda
__pipeline_memcpy_async(&smem[tid], &gmem[tid], sizeof(float));
__pipeline_commit();
__pipeline_wait_prior(0);
```

它减少了寄存器压力和指令数，是 GEMM 和 attention kernel 的标准做法。

### TMA

Hopper 的 Tensor Memory Accelerator 进一步把多维拷贝卸载给专用硬件：

```cuda
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
```

TMA 支持多维 tile、自动处理边界，一个线程就能发起整块传输。它需要 tensor map 描述符，编程模型比 `cp.async` 复杂，但能把地址计算和拷贝指令从 SM 上彻底移走。

### 传输与计算的权衡

异步传输不是免费的：

- 需要额外的 buffer；
- 需要同步（event 或 barrier）；
- 过度拆分会让每次传输太小，DMA 效率下降。

通常按几十 KB 到几 MB 分块，具体取决于 kernel 的计算量和 HBM 延迟。

## 本章小结

- GPU 内存层次从寄存器到 HBM，延迟差 500 倍以上，带宽差两个数量级。
- 每 SM 平均只能分到约 25 GB/s 的 HBM 带宽，访存受限是常态。
- 寄存器按 warp 分配，超过 255 个/线程会 spill；`__launch_bounds__` 可以约束用量。
- Shared memory 有 32 个 bank，按列访问会产生 32 路冲突，padding 到 33 可以消除。
- Global memory 以 32 字节 sector 访问，连续地址才能合并；跨步访问会把带宽利用率降到 1/4 甚至更低。
- L1 与 shared memory 共享物理存储；L2 全局共享但装不下 LLM 权重，decode 的命中率接近 0。
- 对齐影响向量化和 Tensor Core 访问，`float4` 要求 16 字节对齐。
- Pinned 内存是异步传输的前提，但分配昂贵，只用于传输缓冲区。
- 统一内存简化编程，但 page fault 和迁移开销让它不适合延迟敏感的主路径。
- `cp.async` 和 TMA 把拷贝从寄存器路径上移走，是现代 GEMM/attention kernel 的基础。

## 延伸阅读

- NVIDIA, [CUDA C++ Programming Guide, Chapter 5: Performance Guidelines](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#performance-guidelines).
- Harris, [How to Access Global Memory Efficiently in CUDA C/C++ Kernels](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/), 2013.
- Harris, [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/), 2013.
- NVIDIA, [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).
- Jia et al., [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/abs/1804.06826), 2018.
- NVIDIA, [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/).
