---
chapter: 23
part: 4
title: GPU Kernel Development Frameworks
level: tbd
status: draft
prerequisites: [17, 19, 20, 22]
lab:
---

# Chapter 23. GPU Kernel Development Frameworks

写一个能跑的 CUDA kernel 需要几十行；写一个接近峰值的 kernel 需要几千行，还要针对每代架构重写。于是出现了两类工具：一类把 tile、流水线、MMA 封装成模板（CUTLASS），另一类把 kernel 写成 Python（Triton）。

这一章比较四种开发方式：CUDA C++、PTX/SASS、Triton、CUTLASS/模板，再加上编译器自动生成。重点不是“哪个最好”，而是每种方式把复杂度放在了哪里、什么时候值得用。

## 23.1 CUDA C++ Development

CUDA C++ 是最底层的通用选择。

### 工具链

```bash
nvcc -O3 -arch=sm_90 -lineinfo kernel.cu -o kernel
```

- `-arch=sm_90`：目标架构，Hopper；
- `-lineinfo`：保留行号信息，profiler 能把性能数据映射回源码；
- `-Xptxas -v`：输出寄存器、shared memory 和 spill 统计；
- `-G`：生成调试信息，关闭优化，只用于调试。

### 项目结构

一个可维护的 CUDA 项目通常分离：

```text
src/
  kernel.cu          # kernel 实现
  kernel.h           # host 侧接口
  allocator.cpp      # 显存池
tests/
  correctness.cpp    # 与参考实现对比
  benchmark.cpp      # 性能基准
```

kernel 的 host 接口应该隐藏 launch 配置，调用方只传张量和形状。这样调 tile 大小、换实现时不用改上层。

### 调试

- `compute-sanitizer`：检查越界、未初始化读、race；
- `cuda-gdb`：单步调试 device 代码；
- `printf`：简单但有效，注意它会串行化 warp；
- `CUDA_LAUNCH_BLOCKING=1`：让错误停在正确的 launch 上。

```bash
compute-sanitizer --tool memcheck ./kernel
```

sanitizer 会让程序慢几十倍，但能发现那些“结果偶尔不对”的问题。

### 测试

每个 kernel 都要有参考实现。最简单的做法是用 CPU 写一遍，或者调用 PyTorch 的对应算子：

```cpp
// 与 torch 的 attention 对比
auto ref = torch::nn::functional::scaled_dot_product_attention(q, k, v);
```

容差按数据类型定：FP32 用 `1e-5`，FP16 用 `1e-2`，FP8 用 `1e-1`。比较时要覆盖边界形状。

## 23.2 PTX and SASS

### PTX

PTX 是 NVIDIA 的虚拟指令集，介于 CUDA C++ 和机器码之间。`nvcc -ptx` 可以生成：

```bash
nvcc -ptx kernel.cu -o kernel.ptx
```

PTX 是稳定的、跨架构的，驱动在运行时把它 JIT 编译成目标 GPU 的机器码。`asm volatile` 可以直接嵌入 PTX：

```cuda
asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
             : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
```

Tensor Core、TMA、`cp.async` 这些指令，CUDA C++ 没有直接语法，只能通过 inline PTX 或库封装使用。

### SASS

SASS 是真正的机器码。`cuobjdump -sass` 或 `nvdisasm` 可以反汇编：

```bash
cuobjdump -sass kernel
```

看 SASS 能发现：

- 编译器有没有生成预期的指令（如 FFMA、HMMA）；
- 有没有多余的 MOV、分支；
- 寄存器有没有被浪费；
- 访存是不是合并的（LDG.E.128 vs LDG.E）。

SASS 与架构强相关，不同代的指令集不同。除非在做极致优化，否则不需要读 SASS。

### 什么时候下探到 PTX/SASS

- 需要新架构的指令，CUDA C++ 还不支持；
- 编译器生成的代码明显低效，需要手写指令序列；
- 做微架构研究，需要精确控制指令。

对绝大多数 kernel，CUDA C++ 加好的算法就够了。

## 23.3 Triton Programming Model

Triton 是 OpenAI 开发的 Python DSL，用 block 而不是 thread 作为编程单位。

### 一个向量加法

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
add_kernel[grid](x, y, out, n, BLOCK=1024)
```

概念对应：

| Triton | CUDA |
| --- | --- |
| `program_id` | `blockIdx` |
| program 实例 | block |
| `tl.arange` | 线程索引 + 向量 |
| `num_warps` | block 的 warp 数 |
| `num_stages` | 流水线级数 |

程序员不写 `threadIdx`，也不管理 warp。tile 内的线程分配、向量化、shared memory 由编译器决定。

### 为什么容易写快

- 编译器自动做向量化和合并访问；
- `tl.dot` 直接映射到 Tensor Core；
- `num_stages` 自动生成流水线；
- 同一份代码可以跑在不同架构上。

代价是控制力：fragment 布局、warp 专门化、非标准指令，Triton 不一定暴露。

## 23.4 Triton Kernel Design

### 矩阵乘法

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

要点：

- `BLOCK_M/N/K` 是编译期常量，决定 tile 和寄存器用量；
- `tl.dot` 要求输入是 2 的幂的块，且通常至少 16×16；
- mask 处理边界，`other=0.0` 保证不污染累加；
- 指针递增避免重复计算地址。

### 自动调参

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
                      num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
                      num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(...):
    ...
```

第一次遇到某个 `(M, N, K)` 时，Triton 会跑完所有配置，选最快的缓存起来。生产环境通常离线调好，避免运行时抖动。

### 限制

- 不支持任意 PTX，特殊指令要用 `tl.inline_asm_elementwise`；
- 对 shared memory 布局的控制有限；
- 复杂控制流（数据相关的循环）支持有限；
- 编译时间长，大型 kernel 可能几十秒。

## 23.5 CUTLASS and Template-based Kernels

CUTLASS 用 C++ 模板表达 GEMM 的分层结构：

```cpp
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm90,
    cutlass::gemm::GemmShape<128, 128, 32>,   // threadblock tile
    cutlass::gemm::GemmShape<64, 64, 32>,     // warp tile
    cutlass::gemm::GemmShape<16, 8, 16>,      // instruction tile
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;                                        // pipeline stages
```

模板参数决定了 tile、流水线、epilogue 和调度策略。CUTLASS 3.x 的 CuTe 用布局代数描述 tensor 的映射，把 tile 的划分从手写下标变成类型推导。

### 优缺点

优点：

- 接近峰值的性能；
- 覆盖 Hopper/Blackwell 的新特性（TMA、WGMMA、FP8）；
- epilogue 可融合。

缺点：

- 模板错误信息极长；
- 学习曲线陡；
- 编译时间长；
- 定制需要理解内部布局。

如果只是调一个标准 GEMM，用 cuBLAS；需要融合或特殊布局，用 CUTLASS；需要研究性修改，才读它的源码。

## 23.6 Kernel Fusion and Code Generation

手写 kernel 的很大一部分工作是把相邻算子融合。编译器可以自动做这件事。

### TorchInductor

`torch.compile` 把 Python 模型编译成 Triton kernel：

```python
model = torch.compile(model, mode="max-autotune")
```

它会把 elementwise、norm、激活融合成一个 kernel，GEMM 仍然调用 cuBLAS 或生成 Triton matmul。

优点：

- 不用手写；
- 能跟上模型结构的变化；
- 对 elementwise 链的融合效果好。

局限：

- 动态形状会触发重编译；
- attention 等复杂算子仍然依赖手写实现；
- 生成的 kernel 不一定最优。

### TVM / XLA

TVM 用 schedule 描述 tile 和并行，XLA 在计算图上做融合和布局优化。它们更偏向编译器研究或特定后端，LLM 推理里不如 Inductor 常见。

### 融合的边界

自动融合适合：

- 逐元素链；
- reduction + elementwise；
- 简单的 broadcast。

不适合：

- 需要跨 block 同步的归约；
- 需要特定 shared memory 布局的 attention；
- 有复杂 mask 和边界处理的操作。

## 23.7 Auto-tuning

同一个 kernel，tile 大小、warp 数、流水线级数的最优值随形状和架构变化。

### 搜索空间

| 参数 | 典型范围 |
| --- | --- |
| `BLOCK_M` | 32–256 |
| `BLOCK_N` | 32–256 |
| `BLOCK_K` | 16–128 |
| `num_warps` | 2–16 |
| `num_stages` | 2–5 |

组合数很容易上百。全量搜索在离线阶段做，运行时查表。

### 成本模型

没有 GPU 时，可以用解析模型估算：

$$
t \approx \max\left(\frac{\text{FLOPs}}{\text{peak}},\
\frac{\text{bytes}}{\text{bandwidth}}\right) + \text{overhead}
$$

但真实的 tile 效率、bank conflict、尾效应很难建模。实践上还是以实测为准。

### 形状特化

LLM 推理的形状是有限的：

- prefill：`M = batch × seq`，常见 512、1024、2048、4096；
- decode：`M = batch`，常见 1、8、32、128；
- `N`、`K` 由模型固定。

可以只对这些形状调参，而不是连续空间。vLLM、TensorRT-LLM 都维护这样的形状表和 kernel 选择逻辑。

## 23.8 Compiler-generated Kernels

### TensorRT-LLM

NVIDIA 的推理栈，把模型编译成高度融合的 engine：

- GEMM 用 cuBLASLt/CUTLASS；
- attention 用 FMHA；
- 层间融合（norm + residual、SiLU·Mul）；
- 针对具体 GPU 和 batch 配置调优。

代价是编译时间长、灵活性低，模型结构变化要重新编译。

### ONNX Runtime

跨平台的图执行引擎，支持多种后端。它对 LLM 的支持不如专用栈，但适合部署和兼容性要求高的场景。

### 什么时候手写

- 编译器不支持的新算子；
- 融合模式编译器识别不了；
- 需要极致性能的热点 kernel；
- 研究新算法。

一般策略是：先用编译器/库拿到基线，profile 找热点，只对热点手写。

## 23.9 CUDA and Triton Performance Comparison

以 H100、FP16、4096³ GEMM 为例：

| 实现 | 相对 cuBLAS | 开发成本 |
| --- | --- | --- |
| cuBLAS | 100% | 极低 |
| CUTLASS | 95–100% | 中 |
| Triton（调优后） | 80–95% | 低 |
| 手写 CUDA | 70–90% | 高 |

Attention 的差距更大一些：

| 实现 | 相对 FlashAttention-2 |
| --- | --- |
| Triton FlashAttention | 70–90% |
| 手写 CUDA（无 TMA） | 50–70% |
| 手写 CUDA（TMA + WGMMA） | 90–100% |

### 选择建议

- **原型和融合算子**：Triton，开发快，性能足够；
- **标准 GEMM**：cuBLAS；
- **需要 epilogue 融合或特殊布局**：CUTLASS；
- **新架构特性、极致性能**：CUDA C++ + PTX；
- **整模型部署**：TensorRT-LLM 或 vLLM 的融合 kernel。

Triton 的差距在缩小。Hopper 的 TMA、Blackwell 的 FP4，Triton 都在逐步支持。对多数团队，Triton 加少量手写热点，是性价比最高的组合。

## 本章小结

- CUDA C++ 提供完整控制，工具链成熟；调试用 compute-sanitizer，优化看 `-Xptxas -v`。
- PTX 是虚拟指令集，SASS 是机器码；只有需要特殊指令或微架构调优时才下探。
- Triton 以 block 为编程单位，自动处理向量化、流水线和 Tensor Core，开发效率高。
- Triton 的 `tl.dot`、`num_stages` 和 autotune 能覆盖多数 kernel，但对布局和特殊指令的控制有限。
- CUTLASS 用模板表达 tile 和流水线，性能接近峰值，学习曲线陡。
- TorchInductor 自动融合 elementwise 和 norm，复杂算子仍依赖手写。
- Auto-tuning 的搜索空间随 tile、warp、stage 增长，生产环境通常离线调好。
- TensorRT-LLM 等编译器能生成高度融合的 engine，代价是灵活性。
- Triton 在 GEMM 上能达到 cuBLAS 的 80–95%，attention 上 70–90%；极致性能仍需 CUDA/CUTLASS。

## 延伸阅读

- Tillet et al., [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf), 2019.
- NVIDIA, [CUTLASS Documentation](https://docs.nvidia.com/cutlass/).
- NVIDIA, [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).
- NVIDIA, [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/).
- Ansel et al., [PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation](https://dl.acm.org/doi/10.1145/3620665.3640366), 2024.
- Chen et al., [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen), 2018.
