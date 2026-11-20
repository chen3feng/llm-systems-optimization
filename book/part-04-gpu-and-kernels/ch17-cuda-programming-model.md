---
chapter: 17
part: 4
title: CUDA Programming Model
level: tbd
status: draft
prerequisites: [4, 6]
lab:
---

# Chapter 17. CUDA Programming Model

`kernel<<<grid, block>>>(args)` 看起来像一次函数调用。

它不是。CPU 把这条命令写进队列就返回了，GPU 可能几微秒后才开始执行；如果 kernel 越界，错误也可能在几毫秒后、完全不相干的一行代码上才报出来。

这一章讲 CUDA 的执行语义：host 和 device 怎么分工、launch 到底做了什么、线程层次怎么映射到硬件、同步和流如何控制顺序。kernel 内部的优化留到后面几章，这里先把“什么时候执行、在哪里执行、谁和谁同步”讲清楚。

## 17.1 Host and Device Execution

CUDA 程序由两部分组成：

- **host 代码**：在 CPU 上运行，负责分配内存、拷贝数据、启动 kernel；
- **device 代码**：在 GPU 上运行，以 kernel 为单位并行执行。

两者有独立的地址空间。host 不能直接解引用 device 指针，device 也不能访问 host 的栈变量。

### 一个完整的例子

```cuda
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  const int n = 1 << 20;
  const size_t bytes = n * sizeof(float);
  float *h_a, *h_b, *h_c;
  cudaMallocHost(&h_a, bytes);   // pinned host memory
  cudaMallocHost(&h_b, bytes);
  cudaMallocHost(&h_c, bytes);
  // ... 初始化 h_a, h_b ...

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  int block = 256;
  int grid = (n + block - 1) / block;
  vector_add<<<grid, block>>>(d_a, d_b, d_c, n);

  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);
}
```

`__global__` 标记 kernel，表示它由 host 调用、在 device 上执行。`__device__` 标记只能在 device 上调用的函数，`__host__ __device__` 表示两边都能用。

### 统一内存

`cudaMallocManaged` 分配的内存可以被 host 和 device 共同访问：

```cuda
float* x;
cudaMallocManaged(&x, bytes);
```

驱动会在访问时按页迁移数据。它简化了代码，但不是免费的：第一次访问会触发 page fault，迁移有延迟；频繁跨设备访问会让性能急剧下降。

统一内存适合原型和数据结构复杂的场景，不适合对延迟敏感的 kernel。

### 错误检查

CUDA 的错误是异步的。kernel launch 本身只返回“命令是否成功入队”，不返回“kernel 是否执行成功”。

```cpp
#define CUDA_CHECK(x) do { \
  cudaError_t e = (x); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); \
  } \
} while (0)
```

每次 API 调用后检查，每次 launch 后至少检查一次 `cudaGetLastError()`。调试时可以用 `cudaDeviceSynchronize()` 把异步错误变成同步错误。

## 17.2 Kernel Launch

launch 的语法是：

```cuda
kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args...);
```

四个参数分别是：

- `gridDim`：grid 的 block 数，可以是 `int` 或 `dim3`；
- `blockDim`：每个 block 的线程数，最大 1024；
- `sharedMemBytes`：动态 shared memory 大小，默认 0；
- `stream`：在哪个流上执行，默认是 legacy default stream。

### 异步语义

launch 是异步的：CPU 把命令写入队列后立即返回。这意味着：

```cuda
kernel<<<grid, block>>>(d_data);
cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
```

第二行的拷贝会等 kernel 完成，因为它们在同一个 stream 上，有顺序依赖。但如果 kernel 在另一个 stream 上，拷贝可能先执行。

### 限制

| 项目 | 限制 |
| --- | --- |
| 每 block 线程数 | 1024 |
| grid x 维度 | `2^31 - 1` |
| grid y/z 维度 | 65,535 |
| kernel 参数 | 传统 4 KB，CUDA 12.1+ 部分架构支持 32 KB |
| 每 SM 驻留 block 数 | 受寄存器、shared memory、warp 上限约束 |

参数是通过常量内存传递的，按值拷贝。传大结构体时要注意 4 KB 限制，超过时应该用指针。

### Launch 开销

一次 launch 在 GPU 侧约 3–5 微秒，CPU 侧提交约 5–10 微秒。听起来不多，但 decode 时每层有十几个 kernel，80 层就是上千次 launch，累计几毫秒——和单 token 的计算时间同一量级。

减少 launch 开销的手段：

- 融合 kernel，减少数量；
- 用 CUDA Graphs 一次性提交整串 launch；
- 用 persistent kernel，在 kernel 内部循环处理多个任务。

### Grid 大小

一个常见的写法：

```cuda
int block = 256;
int grid = (n + block - 1) / block;
```

当 `n` 很大时，grid 可能有几十万个 block。这没问题，GPU 会分批调度；但如果每个 block 的工作量很小，调度开销会占主导。

另一种写法是固定 grid，让每个线程处理多个元素：

```cuda
int grid = std::min((n + block - 1) / block, 4096);
```

具体用哪种，取决于任务和 GPU 规模。grid-stride loop 在小任务上更灵活。

## 17.3 Thread, Warp, Block and Grid

CUDA 的线程层次是 grid → block → warp → thread。

### 索引

```cuda
int tid = threadIdx.x;                    // block 内
int bid = blockIdx.x;                     // grid 内
int gid = blockIdx.x * blockDim.x + threadIdx.x;
```

二维和三维索引用 `threadIdx.x/y/z` 和 `blockIdx.x/y/z`。矩阵乘法通常用二维 block，图像处理用三维。

### Warp 与发散

32 个连续线程组成一个 warp。同一个 warp 的线程执行同一条指令。

如果 warp 内的线程走了不同的分支：

```cuda
if (threadIdx.x < 16) {
  y = f(x);
} else {
  y = g(x);
}
```

硬件会串行执行两条路径，每条路径上只有一半 lane 有效。这就是分支发散。

避免发散的方法：

- 让整个 warp 走同一分支；
- 用 `select` 或无分支写法；
- 把边界判断放在 warp 粒度，而不是线程粒度。

### Block 与 Shared Memory

一个 block 内的线程可以通过 shared memory 通信：

```cuda
__shared__ float tile[32][33];
tile[threadIdx.y][threadIdx.x] = x[...];
__syncthreads();
```

Block 是同步的单位。不同 block 之间没有顺序保证，也不能用 `__syncthreads()` 同步。

### Grid-stride loop

当数据量大于一次能启动的线程数时，用循环让每个线程处理多个元素：

```cuda
__global__ void scale(float* x, float a, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += blockDim.x * gridDim.x) {
    x[i] *= a;
  }
}
```

这种写法让 grid 大小与数据量解耦，适合需要固定占用率或 persistent kernel 的场景。

### Block 大小

- 128 或 256 是最常见的起点；
- 必须是 32 的倍数，否则最后一个 warp 有空闲 lane；
- 太小，block 调度开销占比高；太大，占用率和负载均衡变差。

最终要通过测量决定，不能只看理论占用率。

## 17.4 SM Scheduling

Block 被分配到 SM 上执行。一个 SM 可以同时驻留多个 block，具体数量由三个资源决定：

$$
\text{blocks/SM} = \min\left(
\left\lfloor \frac{\text{registers}}{\text{regs/thread} \times \text{threads/block}} \right\rfloor,
\left\lfloor \frac{\text{shared memory}}{\text{smem/block}} \right\rfloor,
\left\lfloor \frac{\text{max warps}}{\text{warps/block}} \right\rfloor
\right)
$$

H100 每个 SM 有 65,536 个寄存器、228 KB shared memory、最多 64 个 warp。

举一个具体例子。某 kernel 每线程用 40 个寄存器，block 大小 128，每个 block 申请 32 KB shared memory。三项分别给出：

- 寄存器：`65,536 / (40 × 128) = 12.8`，取 12；
- shared memory：`228 / 32 = 7.1`，取 7；
- warp：`64 / 4 = 16`。

最小值是 7，所以每个 SM 驻留 7 个 block、28 个 warp，占用率 `28 / 64 = 43.75%`。此时瓶颈不是寄存器，而是 shared memory：只要把每 block 的 shared memory 降到 24 KB，占用率就能提到 57%。

这个计算可以交给 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 在运行时完成，但知道瓶颈在哪个资源上，才知道该优化什么。

### 没有顺序保证

Block 的执行顺序是不确定的。不能假设 block 0 先于 block 1 完成，也不能在 block 之间做生产者-消费者同步。

如果需要跨 block 同步，只能用：

- 多个 kernel，利用 stream 的顺序；
- cooperative groups 的 grid sync；
- 原子操作加自旋（容易死锁，慎用）。

### 尾效应

假设 132 个 SM、每个 SM 能放 8 个 block，一波能跑 1,056 个 block。

如果 grid 有 1,100 个 block，就要跑两波：第一波 1,056 个，第二波 44 个。第二波只用了 4% 的 SM，但时间和满波差不多。

$$
\text{效率} = \frac{1{,}100}{2 \times 1{,}056} \approx 52\%
$$

调整 grid 大小、让每个 block 多做一点工作，或者用 persistent kernel，都能缓解尾效应。

### 占用率不是越高越好

占用率高意味着有更多 warp 可以隐藏延迟，但也意味着每个线程分到的寄存器更少。有些 kernel（如大 tile 的 GEMM）需要大量寄存器和 shared memory，占用率低但性能更好。

判断标准是实际吞吐，不是占用率数字。

## 17.5 Synchronization and Barriers

### __syncthreads()

`__syncthreads()` 是 block 级屏障：所有线程必须到达，才能继续。

它有两个作用：

- 同步执行；
- 保证屏障前的 shared memory 写入对屏障后的线程可见。

```cuda
__shared__ float s[256];
s[threadIdx.x] = x[threadIdx.x];
__syncthreads();
float left = s[(threadIdx.x + 255) % 256];
```

### 发散的屏障会死锁

```cuda
if (threadIdx.x < 16) {
  __syncthreads();   // 只有一半线程到达
}
```

如果 block 内的线程没有全部到达屏障，硬件会一直等。这是未定义行为，通常会挂起。

屏障必须放在所有线程都会执行到的位置，或者用 `__syncwarp()` 做 warp 级同步。

### 内存栅栏

`__syncthreads()` 只保证 block 内的顺序。跨 block 或跨设备时，需要内存栅栏：

| 函数 | 作用范围 |
| --- | --- |
| `__threadfence_block()` | block 内 |
| `__threadfence()` | device 内所有线程 |
| `__threadfence_system()` | 包括 host 和其他设备 |

栅栏保证写入的顺序，不保证其他线程已经看到。配合原子操作或 volatile 使用。

### 原子操作

```cuda
atomicAdd(&out[0], value);
atomicMax(&out[0], value);
atomicCAS(&lock, 0, 1);
```

原子操作保证读-改-写的原子性，但代价高。大量原子操作会成为瓶颈，通常先用 shared memory 做局部归约，再对全局做一次原子操作。

### Block 内归约

```cuda
__global__ void block_sum(const float* x, float* out, int n) {
  __shared__ float s[256];
  int tid = threadIdx.x;
  float v = 0.0f;
  for (int i = blockIdx.x * blockDim.x + tid; i < n;
       i += blockDim.x * gridDim.x) {
    v += x[i];
  }
  s[tid] = v;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(out, s[0]);
}
```

这段代码是很多 reduction kernel 的骨架。后面的章节会讲怎么用 warp shuffle 和向量化把它优化到接近带宽上限。

### Grid 同步

Cooperative groups 提供 grid 级同步：

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernel(...) {
  cg::grid_group grid = cg::this_grid();
  // ...
  grid.sync();
}
```

它要求用 `cudaLaunchCooperativeKernel` 启动，并且 grid 必须能全部驻留。不是所有场景都适用。

## 17.6 Streams and Events

### Stream 是什么

Stream 是一串有序操作的队列。同一个 stream 内的操作按顺序执行，不同 stream 之间可以并发。

```cuda
cudaStream_t s1, s2;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);

kernel_a<<<grid, block, 0, s1>>>();
kernel_b<<<grid, block, 0, s2>>>();
```

`kernel_a` 和 `kernel_b` 可以同时运行，前提是资源允许。

### 默认流

不指定 stream 时使用 legacy default stream。它有隐式同步语义：default stream 中的操作会等待其他 blocking stream 中的操作完成，反之亦然。

这会破坏并发。要获得真正的并发，可以：

- 显式使用非默认 stream；
- 编译时加 `--default-stream per-thread`，让每个 host 线程有自己的默认流。

### 异步拷贝

```cuda
cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream);
```

`cudaMemcpyAsync` 只有在 host 内存是 pinned 时才真正异步。pageable 内存会被驱动先拷到内部缓冲区，host 端仍然阻塞。

```cuda
cudaMallocHost(&h_a, bytes);   // pinned
```

Pinned memory 的代价是分配慢、占用物理内存，不能滥用。

### 多流流水线

```cpp
cudaStream_t stream[2];
for (int i = 0; i < 2; ++i) cudaStreamCreate(&stream[i]);

for (int i = 0; i < n; i += chunk) {
  int s = (i / chunk) % 2;
  cudaMemcpyAsync(d_in[s], h_in + i, bytes,
                  cudaMemcpyHostToDevice, stream[s]);
  kernel<<<grid, block, 0, stream[s]>>>(d_in[s], d_out[s], count);
  cudaMemcpyAsync(h_out + i, d_out[s], bytes,
                  cudaMemcpyDeviceToHost, stream[s]);
}
```

两条流交替，让一个 chunk 的拷贝和另一个 chunk 的计算重叠。这是最基础的流水线。

### Event

Event 用于计时和跨流同步：

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>();
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
```

跨流依赖：

```cuda
cudaEventRecord(event, s1);
cudaStreamWaitEvent(s2, event, 0);
```

这比 `cudaDeviceSynchronize()` 精确得多，只同步必要的部分。

### 优先级

```cuda
int priority_high, priority_low;
cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, priority_high);
```

高优先级 stream 在资源竞争时更容易被调度，但不能抢占已经在执行的 kernel。

## 17.7 Asynchronous Execution

CUDA 的异步性贯穿整个 API。理解哪些调用会阻塞，是写出正确代码的前提。

### 同步点

| 调用 | 行为 |
| --- | --- |
| `cudaDeviceSynchronize()` | 等所有 stream 完成 |
| `cudaStreamSynchronize(s)` | 等指定 stream 完成 |
| `cudaEventSynchronize(e)` | 等 event 之前的操作完成 |
| `cudaMemcpy()`（pageable） | host 阻塞到拷贝完成 |
| `cudaMemcpyAsync()`（pinned） | 立即返回 |
| `cudaMalloc()` | 可能阻塞，取决于分配器 |

### 错误传播

kernel 越界、非法地址等错误是异步的。launch 返回 `cudaSuccess` 不代表 kernel 成功。

```cuda
kernel<<<grid, block>>>();
cudaError_t err = cudaGetLastError();   // 只检查 launch 参数
cudaDeviceSynchronize();                // 这里才会暴露执行错误
```

调试时可以设 `CUDA_LAUNCH_BLOCKING=1`，让每次 launch 同步执行，错误会停在正确的调用上。代价是性能大幅下降，只用于调试。

### CUDA Graphs

decode 时每层十几个 kernel，80 层上千次 launch。每次 launch 的 CPU 开销累积起来很可观。

CUDA Graphs 把整串操作捕获成一张图，一次提交：

```cuda
cudaGraph_t graph;
cudaGraphExec_t exec;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
for (int i = 0; i < layers; ++i) {
  rms_norm<<<..., stream>>>(...);
  qkv_gemm<<<..., stream>>>(...);
  attention<<<..., stream>>>(...);
  // ...
}
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

for (int step = 0; step < max_steps; ++step) {
  cudaGraphLaunch(exec, stream);
}
```

Graph 的约束：

- 捕获期间不能有 host 同步；
- 所有指针和形状在捕获时固定，变长 batch 需要重新捕获或用 graph 的可更新参数；
- 动态控制流（如提前退出）不能直接捕获。

对固定形状的 decode，Graph 能把 launch 开销压到接近零。

## 17.8 CUDA Runtime and Driver APIs

CUDA 有两套 API：

| | Runtime API | Driver API |
| --- | --- | --- |
| 前缀 | `cuda` | `cu` |
| 初始化 | 隐式 | 显式 `cuInit` |
| Context | 隐式管理 | 显式创建 |
| 模块加载 | 编译进程序 | 运行时加载 cubin/PTX |
| 易用性 | 高 | 低 |
| 控制力 | 中 | 高 |

大多数程序用 runtime API。需要动态加载 kernel、管理多个 context、或减少 API 开销时，才用 driver API。

### Driver API 示例

```cpp
CUmodule module;
CUfunction func;
cuModuleLoad(&module, "kernel.cubin");
cuModuleGetFunction(&func, module, "vector_add");

void* args[] = {&d_a, &d_b, &d_c, &n};
cuLaunchKernel(func, grid, 1, 1, block, 1, 1,
               0, 0, args, nullptr);
```

### 编译与架构

```bash
nvcc -arch=sm_90 -O3 kernel.cu -o kernel
```

`sm_90` 是 H100 的 compute capability。编译产物可以是：

- **cubin**：特定架构的机器码，不能跨架构；
- **PTX**：虚拟指令集，运行时 JIT 编译到目标架构，向前兼容。

`__CUDA_ARCH__` 宏在 device 代码里区分架构：

```cuda
#if __CUDA_ARCH__ >= 900
  // Hopper 专用指令
#endif
```

### 多 GPU

```cuda
int count;
cudaGetDeviceCount(&count);
cudaSetDevice(1);

cudaDeviceEnablePeerAccess(0, 0);
cudaMemcpyPeer(dst, 1, src, 0, bytes);
```

每张卡有独立的 context 和地址空间。跨卡访问要走 NVLink 或 PCIe，带宽和延迟取决于拓扑。

### 什么时候用 Driver API

- 推理引擎需要在运行时加载不同模型的 kernel；
- 需要精细控制 context 和模块生命周期；
- 需要绕过 runtime 的隐式同步；
- 需要和图形 API（OpenGL/Vulkan）互操作。

对大多数 kernel 开发，runtime API 足够了。

## 本章小结

- CUDA 程序由 host 代码和 device kernel 组成，两者有独立地址空间。
- kernel launch 是异步的：CPU 提交后立即返回，错误可能延迟暴露。
- 线程层次是 grid → block → warp → thread；warp 是 32 个线程，分支发散会让路径串行。
- block 被分配到 SM，执行顺序不确定；占用率由寄存器、shared memory 和 warp 上限共同决定。
- `__syncthreads()` 是 block 级屏障，放在发散分支里会死锁；跨 block 同步需要多个 kernel 或 cooperative groups。
- stream 是有序队列，不同 stream 可以并发；异步拷贝需要 pinned memory。
- event 用于精确计时和跨流同步，比 `cudaDeviceSynchronize()` 更细粒度。
- CUDA Graphs 把整串 launch 捕获成一张图，适合固定形状的 decode。
- runtime API 易用，driver API 控制力强；编译时要选对 `sm_` 架构。

## 延伸阅读

- NVIDIA, [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).
- NVIDIA, [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).
- NVIDIA, [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/).
- NVIDIA, [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs).
- Harris, [How to Access Global Memory Efficiently in CUDA C/C++ Kernels](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/), 2013.
- Harris, [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/), 2013.
