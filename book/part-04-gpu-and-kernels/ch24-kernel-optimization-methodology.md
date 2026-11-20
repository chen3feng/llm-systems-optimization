---
chapter: 24
part: 4
title: Kernel Optimization Methodology
level: tbd
status: draft
prerequisites: [5, 6, 19, 20, 21, 22]
lab:
---

# Chapter 24. Kernel Optimization Methodology

优化 kernel 最容易犯的错误，是一上来就改代码。

看到循环就展开，看到 shared memory 就加 padding，看到分支就去掉——这些手段本身没错，但如果瓶颈不在这里，改完只会让代码更难维护，性能原地不动。真正的优化是一个受控实验：先建立正确性基线，再测量，再定位，再改一个变量，再测量。

这一章讲这个流程。它适用于单个 kernel，也适用于整个推理引擎的热点优化。

## 24.1 Establishing a Correctness Baseline

没有正确性基线，性能数字没有意义。

### 参考实现

每个 kernel 都要有一个可信任的参考：

- CPU 上的朴素实现；
- PyTorch 的对应算子；
- 数值稳定的高精度版本（FP64）；
- 已有的成熟库。

```python
ref = torch.nn.functional.layer_norm(x, shape, weight, bias, eps)
out = custom_kernel(x, weight, bias, eps)
err = (out - ref).abs().max().item()
```

参考实现要独立于被优化的代码。用同一个 kernel 的不同版本互相对比，只能发现回归，发现不了共同的错误。

### 容差

容差取决于数据类型和归约长度：

| 数据类型 | 绝对容差 | 相对容差 |
| --- | --- | --- |
| FP32 | `1e-5` | `1e-4` |
| TF32 | `1e-3` | `1e-2` |
| FP16/BF16 | `1e-2` | `1e-2` |
| FP8 | `1e-1` | `1e-1` |

这些只是起点。长序列 attention、大 `K` 的 GEMM，误差会随累加长度增长，容差要相应放宽。

### 边界用例

- 空输入、单元素输入；
- 长度不是 tile 大小的倍数；
- 长度为 1 的序列；
- 全零、全负、极大值、极小值；
- NaN/Inf 输入；
- batch 和 head 数为奇数。

边界用例往往能暴露越界读写和 mask 错误，而随机测试覆盖不到。

### 确定性

如果 kernel 用原子操作做归约，每次运行的结果可能不同。测试时要么固定归约顺序，要么用统计容差。

对训练，非确定性会影响复现；对推理，它会让输出在边界上抖动。是否可接受，要在设计阶段决定。

## 24.2 Selecting Optimization Targets

优化第一个原则：只优化热点。

### 先测量，再动手

```text
总时间 100%
├── GEMM            70%
├── attention       20%
├── norm/激活        8%
└── 其他             2%
```

把 norm 优化一倍，总时间只降 4%；把 GEMM 优化 10%，总时间降 7%。Amdahl 定律决定了收益上限。

### 热点可能不是你以为的那个

常见的误判：

- 以为 attention 最慢，实际是权重加载；
- 以为 GEMM 是算力瓶颈，实际是 `M=1` 的带宽瓶颈；
- 以为小 kernel 可以忽略，实际几十次 launch 累积成主要开销。

profile 之前不要下结论。

### 优化的层次

从收益和成本看，优先级通常是：

1. **算法**：换更少的计算或访存（FlashAttention、量化、融合）；
2. **数据布局**：合并访问、SoA、对齐；
3. **并行划分**：tile、block、warp 的分配；
4. **指令级**：向量化、ILP、快速数学；
5. **微调**：寄存器、bank、调度。

越靠前收益越大，也越难改。不要在第 5 层花几天，而第 1 层还有明显空间。

## 24.3 Profiling and Bottleneck Identification

Nsight Compute 给出每个 kernel 的“Speed of Light”指标：

- **Compute throughput**：SM 的算力利用率；
- **Memory throughput**：HBM/L2 的带宽利用率；
- **Duration**：kernel 时间；
- **Achieved occupancy**：实际驻留 warp 比例；
- **Stall reasons**：warp 为什么停。

### 判断瓶颈

| 现象 | 判断 | 下一步 |
| --- | --- | --- |
| Memory 接近 100%，Compute 很低 | 带宽受限 | 减少字节、融合、量化 |
| Compute 接近 100%，Memory 很低 | 算力受限 | Tensor Core、降精度、减少指令 |
| 两者都不高，stall 在 `long_scoreboard` | 延迟未被隐藏 | 提高占用率或 ILP |
| 两者都不高，stall 在 `barrier` | 同步过多 | 减少 `__syncthreads`、改 warp 级 |
| shared memory 冲突高 | bank conflict | padding、swizzle |
| 占用率低，寄存器接近上限 | 寄存器压力 | 缩小 tile、`__launch_bounds__` |
| kernel 时间很短但总时间长 | launch 开销 | 融合、CUDA Graphs |

### 例子：decode 的 GEMM

`M=1` 的 GEMM，Nsight 显示 Memory 90%、Compute 10%。结论是带宽受限，优化方向是：

- 权重量化，减少字节；
- 提高合并度；
- 用 GEMV 专用 kernel，而不是通用 GEMM。

这时候调 tile 大小没有意义，因为算力本来就没用满。

### 例子：prefill 的 attention

`S=8192` 的 causal attention，Compute 60%、Memory 40%。还有提升空间，但要看 stall：

- stall 在 `math_pipe_throttle`：算力接近饱和，换 Tensor Core 或降精度；
- stall 在 `long_scoreboard`：K/V 加载没跟上，加深流水线；
- stall 在 `barrier`：softmax 的同步太多，改 warp 级归约。

同一个 kernel，不同的 stall 对应完全不同的优化。

## 24.4 Parameter Tuning

参数包括 tile 大小、block 大小、展开因子、流水线级数、warp 数。

### 一次只改一个

同时改 tile 和 block，性能变了也不知道是哪个起的作用。每次只改一个变量，记录结果。

```text
baseline:        1.00x
BLOCK_K 32→64:   1.12x
num_stages 3→4:  1.05x
BLOCK_M 128→64:  0.93x  (回退)
```

### 测量噪声

GPU 的性能测量有噪声：

- 时钟波动；
- 其他进程干扰；
- 首次运行的初始化；
- 输入数据不同导致的 cache 行为差异。

做法：

- 先 warm-up 几次；
- 跑 20–50 次，取中位数；
- 锁定时钟（`nvidia-smi -lgc`）；
- 固定输入和随机种子；
- 报告方差，不只看平均值。

差异小于 3% 的结果，通常不值得当作改进。

### 调参的边界

- tile 受 shared memory 和寄存器限制；
- block 大小受 1024 和 warp 粒度限制；
- 流水线级数受 shared memory 容量限制；
- 有些参数在某个形状下最优，换个形状就变差。

生产环境通常按形状分别调参，维护一张表，而不是找一组“通用最优”。

## 24.5 Memory and Compute Optimization

### 访存优化

按优先级：

1. **减少字节**：量化、压缩、跳过零；
2. **融合**：避免中间结果落 HBM；
3. **合并访问**：连续地址、SoA、对齐；
4. **向量化**：`float4`/`half2`；
5. **缓存**：L2 持久化、shared memory 复用；
6. **异步**：`cp.async`、TMA、双缓冲。

一个 elementwise kernel 的优化顺序就是这个：先看能不能融合，再看访存是否合并，最后才调 block 大小。

### 计算优化

1. **Tensor Core**：矩阵乘法必须用；
2. **降低精度**：FP16 → FP8 → FP4，配合缩放；
3. **减少指令**：地址计算、循环控制、类型转换；
4. **提高 ILP**：展开、独立累加器；
5. **快速数学**：`__expf`、`rsqrtf`、`exp2`。

计算优化的前提是确认算力受限。带宽受限时降精度也有用，因为它同时减少了字节。

### 一个融合的例子

decode 时的 `residual + rmsnorm + scale`：

```text
版本 A（3 个 kernel）：
  add:      读 32 KB，写 16 KB
  rmsnorm:  读 16 KB，写 16 KB
  scale:    读 16 KB，写 16 KB
  合计约 112 KB，3 次 launch

版本 B（1 个 kernel）：
  读 48 KB，写 16 KB
  合计 64 KB，1 次 launch
```

流量降 40%，launch 从 3 次降到 1 次。每次 launch 3–5 µs，仅这一项就省 6–10 µs。对 decode 每层几百微秒的预算，这是可观的。

## 24.6 Numerical Validation

性能优化不能以正确性为代价。每次改动后都要重新验证。

### 误差指标

- **最大绝对误差**：`max|out - ref|`；
- **最大相对误差**：`max|out - ref| / (|ref| + eps)`；
- **均方根误差**：对整体分布更敏感；
- **余弦相似度**：对方向敏感，适合 embedding 和 hidden state。

不同指标暴露不同问题。最大绝对误差能发现个别离群点，RMSE 能发现系统性偏移。

### 误差会累积

单层误差 `1e-3`，80 层后可能放大到 `1e-1`。验证不能只看单层，还要看端到端：

- 输出的 perplexity；
- 生成文本的一致性；
- 下游任务的准确率。

一个 kernel 的单层误差在容差内，不代表模型质量不变。

### 常见数值问题

- FP16 累加溢出；
- softmax 没减最大值；
- `rsqrt(0)` 产生 inf；
- 除零没有保护；
- NaN 通过 `fmaxf` 传播；
- 快速数学改变了舍入行为。

测试里要显式构造这些输入。

## 24.7 Performance Portability

一个 kernel 在 H100 上调好，换到 A100 或 Blackwell 可能差很多。

### 差异来源

| 维度 | A100 | H100 | B200 |
| --- | --- | --- | --- |
| SM 数 | 108 | 132 | 148 |
| shared memory/SM | 164 KB | 228 KB | 228 KB |
| Tensor Core | 第三代 | 第四代 | 第五代 |
| 新特性 | `cp.async` | TMA、WGMMA | FP4、TMEM |
| HBM 带宽 | 2.0 TB/s | 3.35 TB/s | 8 TB/s |

tile 大小、流水线深度、指令选择都要重新调。

### 可移植的写法

- 用编译期分支选择架构相关代码：

```cuda
#if __CUDA_ARCH__ >= 900
  // TMA 路径
#else
  // cp.async 路径
#endif
```

- 运行时查询设备属性，选择 kernel 变体：

```cpp
cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
```

- 提供 fallback：新特性不可用时退回通用实现；
- 把架构相关的参数放在配置表里，而不是硬编码。

### 形状可移植性

同一张卡上，`M=1` 和 `M=4096` 的最优 kernel 可能完全不同。推理引擎通常维护多个 kernel：

- GEMV（小 M）；
- GEMM（大 M）；
- 分块的 attention（prefill）；
- FlashDecoding（decode）。

按形状分派，而不是一个 kernel 打天下。

## 24.8 Regression and Benchmark Automation

优化是一次性的，回归是长期的。没有自动化，性能会在后续改动中悄悄退化。

### 基准套件

覆盖：

- 真实形状：从生产流量采样，而不是只测方阵；
- 边界形状：1、奇数、非 tile 倍数；
- 不同 batch：1、8、32、128；
- 不同序列长度：128、2K、8K、32K；
- 多个 GPU 架构。

### 稳定的测量

- 固定时钟和功耗上限；
- 独占 GPU，避免其他进程干扰；
- warm-up 足够；
- 多次取中位数；
- 记录方差。

CI 机器通常不是独占的，噪声大。可以设一个较宽的阈值（如 10%），只捕获明显回归；精细对比在专用机器上做。

### 阈值与告警

```text
benchmark            baseline    current    ratio
gemm_4096            0.82 ms     0.84 ms    0.98x
attention_8k         1.05 ms     1.42 ms    0.74x  ← 告警
rmsnorm_decode       3.1 us      3.2 us     0.97x
```

回归超过阈值就阻断合并，或者至少让人看到。

### 版本追踪

把性能数据和时间、commit、驱动版本、GPU 型号关联起来。性能退化可能来自：

- 代码改动；
- 驱动升级；
- 编译器版本；
- 时钟策略变化。

没有历史数据，就无法定位。

### 正确性也要回归

性能基准旁边要有正确性测试。优化后的 kernel 可能更快但算错，只有数值测试能发现。

## 本章小结

- 先建立正确性基线：独立参考实现、合理容差、边界用例、确定性策略。
- 只优化热点；用 Amdahl 定律判断收益上限，profile 之前不要下结论。
- Nsight 的 SOL 指标和 stall reason 是定位瓶颈的主要依据；不同 stall 对应不同优化。
- 调参一次只改一个变量，注意 warm-up、时钟和测量噪声；小于 3% 的差异通常不可信。
- 访存优化优先减少字节和融合，其次是合并、向量化和缓存。
- 计算优化优先 Tensor Core 和低精度，其次是减少指令和提高 ILP。
- 数值验证要看单层误差和端到端质量，误差会随层数累积。
- 可移植性要求按架构和形状分派 kernel，提供 fallback。
- 基准和正确性测试都要自动化，覆盖真实形状和边界，跟踪历史趋势。

## 延伸阅读

- Williams et al., [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://dl.acm.org/doi/10.1145/1498765.1498785), 2009.
- NVIDIA, [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/).
- NVIDIA, [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).
- Gregg & Hazelwood, [Where is the Data? Why You Cannot Debate CPU vs. GPU Performance Without the Answer](https://ieeexplore.ieee.org/document/5762735), 2011.
- Hoefler & Belli, [Scientific Benchmarking of Parallel Computing Systems](https://dl.acm.org/doi/10.1145/2807591.2807644), 2015.
- Chen et al., [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen), 2018.
