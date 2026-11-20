# Appendix C. CUDA Reference

本附录是本书用到 CUDA 概念的速查表。具体的编程模型在 Part IV 展开。

## C.1 执行模型

| 概念 | 说明 | 典型规模 |
| --- | --- | --- |
| Thread | 最小的执行单位 | — |
| Warp | 32 个线程，同步执行 | 32 |
| Block | 一个 SM 上的线程组，可共享 shared memory | 128–1,024 线程 |
| Grid | 所有 block 的集合 | 最多 2³¹−1 个 block |
| SM | 流式多处理器 | H100 有 132 个 |

一个 block 内的线程可以通过 shared memory 和 `__syncthreads()` 协作；不同 block 之间只能通过 global memory 和原子操作通信。

```cuda
__global__ void kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * 2.0f;
}

kernel<<<grid, block>>>(x, n);
```

## C.2 内存层级

| 空间 | 作用域 | 延迟（约） | 容量 |
| --- | --- | --- | --- |
| Register | 线程 | 1 cycle | 每线程最多 255 个 |
| Shared memory | Block | ~30 cycle | 每 SM 最多 228 KB（H100） |
| L1 / L2 | SM / 全卡 | ~30 / ~200 cycle | L2 50 MB |
| Global (HBM) | 全局 | ~400–800 cycle | 80 GB |
| Constant | 全局只读 | 缓存命中 ~1 cycle | 64 KB |
| Texture | 全局 | 类似 L1 | — |

shared memory 是性能的关键：它把重复访问的数据从 HBM 搬到片上。矩阵乘的 tile、attention 的 K/V 块都放在这里。

## C.3 占用率

占用率 = 活跃 warp 数 / 每 SM 最大 warp 数。它受四个因素限制：

- **寄存器**：每线程用的寄存器越多，能并发的 warp 越少；
- **shared memory**：每 block 申请的 shared memory 越大，能并发的 block 越少；
- **线程数**：每 block 的线程数；
- **block 数**：每 SM 的最大 block 数（通常 32）。

经验值：计算密集的 kernel 需要 50% 以上的占用率才能藏住延迟；访存密集的 kernel 需要更高。但占用率不是越高越好——有些 kernel 用低占用率换更多寄存器，反而更快。

## C.4 同步原语

| 原语 | 范围 | 用途 |
| --- | --- | --- |
| `__syncthreads()` | Block | 等待所有线程到达 |
| `__syncwarp()` | Warp | warp 内同步（Volta 之后需要显式） |
| `__threadfence()` | 设备 | 保证写操作对其他 block 可见 |
| `cudaDeviceSynchronize()` | 设备 | 等待所有 stream 完成 |
| `cudaStreamSynchronize()` | Stream | 等待单个 stream |

warp 内的线程在 Volta 之后可以独立调度（independent thread scheduling），所以不能再假设“同一 warp 的线程自动同步”。

## C.5 原子操作

```cuda
atomicAdd(&counter, 1);
atomicCAS(&lock, 0, 1);
```

原子操作在 L2 上执行，同一地址的竞争会串行化。高竞争时性能急剧下降，常见缓解：

- **分片累加**：每个 block 累加到私有副本，最后合并；
- **warp 聚合**：先用 `__shfl_down_sync` 在 warp 内求和，再原子加一次；
- **避免在热点路径用全局原子**：用 block 内的归约替代。

## C.6 常用内建函数

| 函数 | 作用 |
| --- | --- |
| `__shfl_down_sync(mask, v, delta)` | warp 内向下移位，做归约 |
| `__ballot_sync(mask, pred)` | 收集 warp 内的布尔值 |
| `__popc(x)` | 统计 1 的个数 |
| `__ldg(ptr)` | 通过只读缓存加载 |
| `__fmaf_rn(a, b, c)` | 融合乘加，单次舍入 |
| `__expf(x)` / `__exp2f(x)` | 快速近似指数 |
| `__rsqrtf(x)` | 快速平方根倒数，用于 RMSNorm |

`__expf` 和 `__rsqrtf` 的精度低于标准库，但在 softmax 和归一化里通常足够，能快 2–5 倍。

## C.7 编译与架构

```bash
nvcc -O3 -arch=sm_90a -lineinfo kernel.cu -o kernel
```

| 架构 | 计算能力 | 特性 |
| --- | --- | --- |
| Ampere | sm_80 | TF32、异步拷贝 |
| Hopper | sm_90 / sm_90a | TMA、warpgroup、FP8 |
| Blackwell | sm_100 / sm_100a | FP4、更大 shared memory |

`sm_90a` 中的 `a` 表示架构专属特性（如 `wgmma`），用这些特性编译的二进制不能在前代 GPU 上运行。分发时通常编译多个架构，或者用 PTX 做 JIT。

## C.8 错误检查

```cuda
#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); \
    } } while (0)
```

异步 kernel 的错误不会在 launch 时返回，而是在下一次同步时暴露。调试时在关键位置插入 `cudaGetLastError()`，或用 `compute-sanitizer` 定位越界和竞态。
