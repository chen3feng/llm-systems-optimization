---
chapter: 21
part: 4
title: Reduction and Elementwise Kernels
level: tbd
status: draft
prerequisites: [12, 17, 18, 19]
lab:
---

# Chapter 21. Reduction and Elementwise Kernels

一个 70B 模型有 80 层，每层至少一次 RMSNorm、一次 SiLU·Mul、一次残差加法，decode 时还要对 128k 的词表做 softmax。这些 kernel 单个只花几微秒，加起来却和 GEMM 同一量级——不是因为它们算得多，而是因为每层都要把中间结果写回 HBM，再读出来。

这一章讲归约和逐元素 kernel。它们的算术强度极低，优化目标不是算力，而是减少字节、减少 kernel 数量、减少启动开销。很多推理引擎一半的工程量花在把这些小算子融合掉。

## 21.1 Reduction Patterns

归约是把一个数组压成一个值：求和、最大值、平方和、范数。

```text
sum([x0, x1, ..., xn-1]) → s
```

### 串行与树形

串行归约是 `n` 次加法，延迟 `O(n)`。树形归约两两合并，延迟 `O(log n)`：

```text
第 1 轮：n/2 次加法
第 2 轮：n/4 次加法
...
第 log2(n) 轮：1 次加法
```

8192 个元素，串行要 8191 步，树形只要 13 步。GPU 的并行归约就是树形。

### 结合律与浮点

浮点加法不满足结合律：`(a + b) + c` 和 `a + (b + c)` 的结果可能不同。树形归约改变了加法顺序，所以：

- 结果和串行版本有微小差异；
- 不同 block 划分会得到不同结果；
- 用原子操作做跨 block 归约，结果不可复现。

对训练，这种差异通常可以接受；对需要逐位复现的测试，必须固定归约顺序。

### 常见变体

| 归约 | 单位元 | 合并函数 |
| --- | --- | --- |
| Sum | 0 | `a + b` |
| Max | `-inf` | `fmaxf(a, b)` |
| Min | `+inf` | `fminf(a, b)` |
| Sum of squares | 0 | `a + b*b` |
| Product | 1 | `a * b` |
| LogSumExp | `-inf` | `max(a,b) + log1p(exp(min-max))` |

Softmax 的分母是 Sum，LayerNorm 的方差是 Sum of squares，attention 的数值稳定版本是 LogSumExp。

## 21.2 Parallel Reduction

一个 block 内的高效归约分三步：

1. 每个线程把多个元素累加到自己的局部变量；
2. warp 内用 shuffle 归约；
3. 每个 warp 的结果写到 shared memory，第一个 warp 再做一次归约。

```cuda
__inline__ __device__ float warp_reduce_sum(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

__global__ void block_reduce(const float* in, float* out, int n) {
  __shared__ float warp_sums[32];
  int tid = threadIdx.x;
  int lane = tid & 31;
  int wid = tid >> 5;

  float v = 0.0f;
  for (int i = blockIdx.x * blockDim.x + tid; i < n;
       i += blockDim.x * gridDim.x) {
    v += in[i];
  }

  v = warp_reduce_sum(v);
  if (lane == 0) warp_sums[wid] = v;
  __syncthreads();

  v = (tid < (blockDim.x >> 5)) ? warp_sums[tid] : 0.0f;
  if (wid == 0) v = warp_reduce_sum(v);
  if (tid == 0) out[blockIdx.x] = v;
}
```

### 为什么用 shuffle

shared memory 版本每轮都要 `__syncthreads()`，5 轮就是 5 次屏障。shuffle 版本在寄存器之间交换，延迟低一个数量级，也不需要同步整个 block。

只有在 warp 之间合并时才需要 shared memory，而且只需要一次屏障。

### 跨 block 归约

一个 block 的结果只是部分和。跨 block 归约有两种做法：

**原子操作**：

```cuda
if (tid == 0) atomicAdd(out, v);
```

简单，但结果不可复现，且大量 block 竞争同一个地址会成为瓶颈。

**两阶段 kernel**：

```text
kernel 1：每个 block 输出一个部分和到临时数组
kernel 2：对一个 block 数的数组再做一次归约
```

多一次 launch，但结果确定，也能处理任意大的输入。生产代码更常用这种。

### 性能

归约是纯带宽受限的。1 GB 的 FP32 数组，理想时间 `1 GB / 3.35 TB/s ≈ 0.3 ms`。优化到 90% 带宽就是好结果。

提高带宽的手段：

- 向量化读取（`float4`），每个线程一次读 16 字节；
- 每个线程处理多个元素，减少 block 数和调度开销；
- 用 `__ldg` 或只读缓存；
- 避免 shared memory bank conflict。

## 21.3 Softmax

Softmax 把一组 logits 变成概率：

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

直接算会溢出：`exp(1000)` 是 inf。标准做法是先减最大值：

$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j x_j
$$

减最大值不改变结果，但保证指数不超过 0。

### 三遍扫描

```cuda
__global__ void softmax_row(const float* x, float* y, int n) {
  // 假设一个 block 处理一行
  float m = -INFINITY;
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    m = fmaxf(m, x[i]);
  m = block_reduce_max(m);

  float sum = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    sum += __expf(x[i] - m);
  sum = block_reduce_sum(sum);

  float inv = 1.0f / sum;
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    y[i] = __expf(x[i] - m) * inv;
}
```

三遍读 `x`，一遍写 `y`。如果行能装进 L2，后两遍命中缓存；装不下就要重复读 HBM。

### 两遍与一遍

- 两遍：第一遍同时求 `max` 和 `sum` 需要 online 算法；或者先求 max，第二遍算 exp 和 sum 并写回未归一化的值，第三遍再缩放。
- 一遍：用 online softmax，边扫描边更新 `(m, sum)`，最后再归一化。FlashAttention 用的就是这个。

对词表 softmax（128k），一行 512 KB，放得进 L2，三遍扫描的实际开销远低于三次 HBM 访问。

### 小 kernel 的真相

128k 词表的 softmax，读写合计约 1 MB。按 3.35 TB/s 算，带宽时间约 0.3 微秒；而一次 kernel launch 要 3–5 微秒。

也就是说，这个 kernel 的时间几乎全是启动和调度开销，不是计算或带宽。优化它的正确方式是融合：把 softmax 和采样、top-k、logits 处理合成一个 kernel，而不是把 softmax 本身再调快 10%。

## 21.4 RMSNorm and LayerNorm

### LayerNorm

$$
\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta
$$

需要均值和方差，两遍扫描或一遍 Welford。

### RMSNorm

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_i x_i^2 + \epsilon}} \odot \gamma
$$

省掉了均值减法和 `β`，只算平方和的均值。计算更少，是现代 LLM 的主流选择。

```cuda
__global__ void rms_norm(const half* x, const half* w, half* y,
                         int n, float eps) {
  int row = blockIdx.x;
  const half* xr = x + (size_t)row * n;
  half* yr = y + (size_t)row * n;

  float ss = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = __half2float(xr[i]);
    ss += v * v;
  }
  ss = block_reduce_sum(ss);

  float inv = rsqrtf(ss / n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = __half2float(xr[i]) * inv * __half2float(w[i]);
    yr[i] = __float2half(v);
  }
}
```

### 单遍版本

hidden 是 8192，block 256 个线程，每个线程负责 32 个元素。这 32 个值可以留在寄存器里：

```text
第一遍：读入 32 个值，累加平方和
归约得到 inv
第二遍：直接用寄存器里的值乘 inv 和 weight，写出
```

省掉一次 HBM 读。代价是每个线程 32 个寄存器（FP16 可以打包成 16 个 32 位寄存器），occupancy 会下降。对 decode 这种只有一行的情况，单遍通常更快。

### 延迟视角

decode 时 RMSNorm 只处理一个 token：8192 个 FP16，16 KB 读、16 KB 写。一个 block 跑在一个 SM 上，SM 的带宽份额约 25 GB/s，时间约 `32 KB / 25 GB/s ≈ 1.3 μs`。

而一次 launch 3–5 μs。80 层就是 80 次 norm，累计几百微秒。融合掉这些 kernel 的收益，比把单个 norm 优化 20% 大得多。

## 21.5 Activation Kernels

### SiLU 与 SwiGLU

SiLU：

$$
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

SwiGLU 的 FFN：

$$
\text{FFN}(x) = W_{\text{down}}\big(\text{SiLU}(W_{\text{gate}} x) \odot W_{\text{up}} x\big)
$$

`gate` 和 `up` 两个 GEMM 的输出逐元素相乘，再进 `down` GEMM。

### GELU

$$
\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)
$$

tanh 版本比 erf 版本快，精度略低，是 BERT/GPT 时代的常见选择。现代 LLM 更多用 SwiGLU。

### 融合的 SiLU·Mul

```cuda
__global__ void silu_mul(const half* gate, const half* up, half* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float g = __half2float(gate[i]);
    float u = __half2float(up[i]);
    float s = g / (1.0f + __expf(-g));
    out[i] = __float2half(s * u);
  }
}
```

读两个数组、写一个数组。算术强度约 `3 FLOPs / 6 bytes ≈ 0.5`，完全受带宽限制。

### 向量化

```cuda
const half2* g2 = reinterpret_cast<const half2*>(gate);
const half2* u2 = reinterpret_cast<const half2*>(up);
half2* o2 = reinterpret_cast<half2*>(out);
```

`half2` 让一次 load 拿两个值，指令数减半。进一步用 `float4` 一次读 8 个 half，带宽利用率更高。

### 特殊函数

`sigmoid` 和 `exp` 走 MUFU，吞吐低于 FMA。对 8192 个元素，exp 的开销可能超过访存。

- 用 `__expf` 而不是 `expf`，误差约 2 ulp；
- 用 `__frcp_rn` 或近似除法；
- 确认 `-use_fast_math` 的副作用后再开。

## 21.6 Elementwise Fusion

逐元素 kernel 的问题是 HBM 往返。一个简单的链：

```text
y = x + residual
z = rmsnorm(y)
w = z * scale
```

三个 kernel 意味着：

- 写 `y`，读 `y`；
- 写 `z`，读 `z`；
- 写 `w`。

融合成一个 kernel，中间结果留在寄存器，只读写首尾：

```cuda
__global__ void fused_residual_norm_scale(
    const half* x, const half* residual, const half* weight,
    half* out, int n, float eps) {
  // 一个 block 处理一行
  // 1. y = x + residual，同时累加平方和
  // 2. inv = rsqrt(mean(y^2) + eps)
  // 3. out = y * inv * weight
}
```

### 融合的收益

以 8192 hidden 为例：

| 版本 | HBM 流量 | kernel 数 |
| --- | --- | --- |
| 三个独立 kernel | 约 7 × 16 KB ≈ 112 KB | 3 |
| 融合 | 约 4 × 16 KB = 64 KB | 1 |

流量减少约 40%，launch 从 3 次降到 1 次。decode 时后者往往快 2–3 倍。

### 融合的边界

- 中间结果必须能放进寄存器或 shared memory；
- 跨 block 的归约不能随便融合；
- 融合后的 kernel 寄存器压力更大，可能降低占用率；
- 调试和复用的难度上升。

实践中，norm + residual + scale、attention 的 softmax + PV、GEMM 的 bias + 激活，是最值得融合的组合。

## 21.7 Reduction Fusion

### Softmax 与 Attention

标准 attention 的 softmax 要为每个 query 行计算，中间矩阵是 `S × S`。FlashAttention 把 softmax 拆成 online 形式，和 `QK^T`、`PV` 融合在一个 kernel 里，避免把 `S × S` 写回 HBM。

这是“归约融合”最重要的例子：不是把 softmax 算得更快，而是让它不再产生中间矩阵。

### Norm 与残差

```text
h = h + attn_out
h = rmsnorm(h)
```

把加法和 norm 融合，省掉一次 `h` 的写回和读取。LLM 每层有两次这样的模式（attention 后、FFN 后），80 层就是 160 次。

### Split Reduction

当归约维度太长、单个 block 处理不完时，可以分两步：

```text
kernel 1：每个 block 处理一段，输出部分和 + 部分 max
kernel 2：合并部分结果
```

对 softmax，第二步要用 `max` 和 `sum` 一起合并：

$$
m = \max(m_1, m_2), \quad
s = s_1 e^{m_1 - m} + s_2 e^{m_2 - m}
$$

这就是 online softmax 的合并公式，也是 FlashAttention 跨 tile 累加的基础。

### 确定性

用原子操作做归约，执行顺序不定，结果每次可能不同。对推理服务，这意味着同样的输入可能产生细微不同的输出。

需要确定性时：

- 用固定顺序的两阶段归约；
- 每个 block 的输出写到固定位置；
- 避免 `atomicAdd` 做浮点累加。

代价是性能和灵活性，通常只在测试和回归中使用。

## 21.8 Numerical Stability

### 减最大值

softmax 不减最大值，`exp(x)` 在 `x > 88` 时溢出成 inf，`x < -88` 时下溢成 0。减掉最大值后，指数范围是 `[-inf, 0]`，不会溢出。

### 累加精度

平方和、方差、softmax 分母都要用 FP32 累加。FP16 的 10 位尾数在累加几千项后会严重失真。

```cuda
float ss = 0.0f;                  // 正确
half ss = 0;                      // 错误，很快溢出或丢精度
```

### 方差的算法

`E[x²] - E[x]²` 在均值很大、方差很小时会灾难性抵消。LayerNorm 的输入通常均值接近 0，问题不大；但如果激活有大的直流分量，应该用 Welford 或两遍算法。

### 快速数学的代价

`-use_fast_math` 会：

- 把除法换成近似倒数；
- 把 `expf` 换成 `__expf`；
- 开启 flush-to-zero；
- 允许重排浮点运算。

它能让 elementwise kernel 快 10–30%，但会改变结果。训练时慎用，推理时要验证精度。

### 特殊值

- `rsqrtf(0)` 返回 inf，要加 `eps`；
- `log(0)` 返回 -inf，LogSumExp 要处理全 `-inf` 的行；
- NaN 会通过 `fmaxf` 传播，调试时先检查输入。

## 本章小结

- 归约用树形结构，8192 个元素只需 13 步；浮点加法不满足结合律，归约顺序会影响结果。
- Block 内归约先用 warp shuffle，再用 shared memory 合并 warp 结果，只需一次屏障。
- 跨 block 归约用两阶段 kernel 或原子操作；前者确定，后者简单但不可复现。
- Softmax 必须减最大值；三遍扫描在行能进 L2 时开销可接受，online 版本适合融合。
- RMSNorm 比 LayerNorm 少算均值和 `β`；decode 时可以单遍完成，把输入留在寄存器。
- SiLU·Mul 是纯带宽 kernel，向量化和 `__expf` 是主要优化手段。
- 融合能同时减少 HBM 流量和 launch 次数，是 decode 优化的核心。
- Reduction fusion 的典型是 FlashAttention 的 online softmax。
- 累加用 FP32，方差避免灾难性抵消，`-use_fast_math` 要验证精度。
- 小 kernel 的时间常常由 launch 主导，融合比微调单 kernel 更有效。

## 延伸阅读

- Harris, [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf), NVIDIA, 2007.
- NVIDIA, [CUDA C++ Programming Guide, Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions).
- Milakov & Gimelshein, [Online Normalizer Calculation for Softmax](https://arxiv.org/abs/1805.02867), 2018.
- Zhang & Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467), 2019.
- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), 2022.
- Welford, [Note on a Method for Calculating Corrected Sums of Squares and Products](https://doi.org/10.1080/00401706.1962.10490022), 1962.
