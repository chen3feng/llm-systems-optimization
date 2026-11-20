---
chapter: 12
part: 3
title: Numerical Formats for LLMs
level: tbd
status: draft
prerequisites: [3, 4]
lab:
---

# Chapter 12. Numerical Formats for LLMs

同一个 70B 模型，FP32 权重是 280 GB，FP16 是 140 GB，FP8 是 70 GB，FP4 是 35 GB。

8 倍的差距，决定了它是需要 4 张 80 GB 的卡、2 张，还是 1 张。数值格式不是“存得省一点”的细节，它直接决定部署方案、带宽需求和吞吐上限。

这一章讲浮点数的位是怎么分的、为什么 FP16 会溢出而 BF16 不会、FP8 的缩放怎么做、INT4 的零点是什么，以及这些格式在 Tensor Core 上分别意味着多少算力。

## 12.1 Floating-point Representation

一个浮点数由三部分组成：

$$
\text{value} = (-1)^s \times 2^{e - \text{bias}} \times \left(1 + \frac{m}{2^M}\right)
$$

- `s`：符号位，1 bit；
- `e`：指数位，`E` bit；
- `m`：尾数位，`M` bit；
- `bias`：指数偏移，通常是 `2^(E-1) - 1`。

指数决定范围，尾数决定精度。这两个预算互相挤占：同样的总位数，指数多一位，范围翻倍，精度减半。

### 范围与精度

- **范围**：能表示的最大值和最小值；
- **精度**：相邻两个可表示数之间的间隔，通常用 ULP（unit in the last place）或机器 epsilon 衡量；
- **相对精度**：`2^-M`，尾数每多一位，精度翻倍。

FP16 有 5 位指数，最大 65,504；BF16 有 8 位指数，最大和 FP32 一样是 `3.4e38`，但尾数只有 7 位，精度差得多。

### 特殊值

IEEE 754 还定义了：

- `+0` 和 `-0`；
- `+inf` 和 `-inf`：溢出或除以零；
- `NaN`：`0/0`、`inf - inf` 等未定义运算；
- 次正规数（subnormal）：指数全 0 时，尾数不再加 1，用来表示比最小正规数更小的值，代价是精度骤降。

很多 GPU kernel 默认开启 flush-to-zero（FTZ），把次正规数当 0 处理，以换取速度。做数值敏感的推理时要注意这个开关。

## 12.2 IEEE 754 and Reduced Precision

| 格式 | 符号 | 指数 | 尾数 | 总位数 | 指数偏移 |
| --- | --- | --- | --- | --- | --- |
| FP32 | 1 | 8 | 23 | 32 | 127 |
| TF32 | 1 | 8 | 10 | 19（存储 32） | 127 |
| BF16 | 1 | 8 | 7 | 16 | 127 |
| FP16 | 1 | 5 | 10 | 16 | 15 |
| FP8 E4M3 | 1 | 4 | 3 | 8 | 7 |
| FP8 E5M2 | 1 | 5 | 2 | 8 | 15 |
| FP4 E2M1 | 1 | 2 | 1 | 4 | 1 |

这张表是后面所有讨论的基础。几个关键观察：

- BF16 和 FP32 的指数位相同，所以范围相同；尾数少 16 位，精度低但不会溢出。
- FP16 的尾数比 BF16 多 3 位，精度更高，但指数只有 5 位，范围窄得多。
- TF32 是 NVIDIA 的 19 位格式，存储时占 32 位，只在 Tensor Core 内部使用。
- FP8 有两种：E4M3 偏精度，E5M2 偏范围。

### 舍入

默认的舍入模式是 round-to-nearest, ties-to-even：舍入到最近的可表示值，正好在中间时取偶数尾数。它比“四舍五入”更公平，不会系统性地偏大。

舍入误差是不可避免的。一次矩阵乘法有 `K` 次累加，最坏情况下误差随 `K` 增长；实际中因为误差符号随机，通常按 `√K` 增长。这也是为什么长 `K` 的 GEMM 要用 FP32 累加。

## 12.3 FP32, FP16 and BF16

### FP32

8 位指数、23 位尾数，范围 `±3.4e38`，相对精度约 `1.2e-7`。

FP32 是训练和推理的“真值”基准。但它在 GPU 上的算力只有 FP16 的几十分之一：A100 的 FP32 是 19.5 TFLOPS，FP16 Tensor Core 是 312 TFLOPS。

### FP16

5 位指数、10 位尾数，最大值 65,504，最小正规数 `6.1e-5`，相对精度约 `1e-3`。

FP16 的问题是范围窄：

- 梯度小于 `6e-5` 会下溢成 0；
- 激活大于 65,504 会溢出成 inf；
- 累加 8,192 个 FP16 数，中间和可能溢出。

训练时通常用 loss scaling：把 loss 乘一个大的常数，反向传播后再除回来，把梯度推到 FP16 能表示的范围。

### BF16

8 位指数、7 位尾数，范围和 FP32 相同，相对精度约 `8e-3`。

BF16 牺牲精度换范围，好处是：

- 不需要 loss scaling；
- 和 FP32 之间的转换是截断，硬件实现简单；
- 训练稳定性接近 FP32。

现代 LLM 训练基本默认 BF16。FP16 更多出现在推理和对精度更敏感的旧模型上。

### TF32

TF32 保持 FP32 的 8 位指数，尾数截到 10 位。它只在矩阵乘法里使用，输入输出仍然是 FP32。

A100 上 TF32 的算力是 156 TFLOPS，是 FP32 的 8 倍。`torch.backends.cuda.matmul.allow_tf32 = True` 打开的就是这个模式。

代价是精度：10 位尾数的相对误差约 `1e-3`。对大多数训练任务够用，对需要高精度的科学计算不够。

## 12.4 FP8 Formats

FP8 有两种 widely used 的变体，都来自 OCP 的 FP8 规范：

| 格式 | 指数 | 尾数 | 最大值 | 最小正规数 | 用途 |
| --- | --- | --- | --- | --- | --- |
| E4M3 | 4 | 3 | 448 | 0.0156 | 权重、激活 |
| E5M2 | 5 | 2 | 57,344 | 6.1e-5 | 梯度 |

E4M3 精度更高，但范围窄；E5M2 范围大，但只有 2 位尾数，精度很差。训练时常见的组合是：前向用 E4M3，反向的梯度用 E5M2。

### 缩放

FP8 的范围装不下 LLM 的激活和权重，必须缩放：

$$
x_{\text{fp8}} = \text{cast}\left(\frac{x}{s}\right), \quad \hat{x} = x_{\text{fp8}} \times s
$$

`s` 是缩放因子。按粒度分：

| 粒度 | 说明 | 开销 |
| --- | --- | --- |
| Per-tensor | 整个张量一个 `s` | 最低，受离群值影响大 |
| Per-channel | 每个输出通道一个 `s` | 中等 |
| Per-block | 每 `128×128` 或 `1×128` 一块 | 高，精度最好 |

DeepSeek-V3 的 FP8 训练用细粒度块缩放：权重按 `128×128` 分块，激活按 `1×128` 分块，每个块一个缩放因子。这样离群值只影响自己所在的块，不会把整个张量压扁。

### Scaling 策略

- **Current scaling**：用当前 tensor 的 `max` 算 `s`，简单但每次都要归约；
- **Delayed scaling**：用上一步的 `max` 估计，和计算重叠，NVIDIA Transformer Engine 采用这种；
- **Dynamic scaling**：根据溢出情况动态调整，适合梯度。

### 精度

E4M3 的相对精度约 `6e-2`，比 FP16 差一个数量级。对权重来说，这个误差通常可以接受；对累加来说不行，必须用 FP32 累加器。

FP8 的 attention 还要额外小心：`QK^T` 的分数范围可能很大，softmax 前要减最大值，缩放因子要按块算。

## 12.5 FP4 and Low-bit Formats

FP4 把每个数压到 4 bit。E2M1 能表示的值只有：

```text
0, 0.5, 1, 1.5, 2, 3, 4, 6（以及它们的负数）
```

最大值 6，最小非零值 0.5。这个范围不可能直接表示 LLM 的权重，必须配合块缩放。

### MXFP4 与 NVFP4

| 格式 | 块大小 | 块缩放 | 额外缩放 |
| --- | --- | --- | --- |
| MXFP4 | 32 | E8M0（2 的幂） | 无 |
| NVFP4 | 16 | E4M3 | 每张量 FP32 |

MXFP4 是 OCP 的 microscaling 标准，32 个元素共享一个 2 的幂缩放因子。NVFP4 是 NVIDIA 的变体，块更小、缩放更精细，精度更好。

Blackwell 的 Tensor Core 原生支持 FP4，稠密算力是 FP8 的 2 倍。代价是：

- 需要校准或量化感知训练；
- 缩放因子的存储和计算有开销；
- 精度损失在长尾任务上更明显。

### 2-bit 及以下

2-bit 量化主要用于权重（W2A16），激活仍然是高精度。它能进一步压缩显存，但 kernel 复杂、精度风险大，目前更多在研究和小模型部署中使用。

## 12.6 Integer Quantization

整数量化用仿射映射：

$$
q = \text{round}\left(\frac{x}{s}\right) + z, \quad \hat{x} = s(q - z)
$$

- `s`：缩放因子（scale）；
- `z`：零点（zero point），保证实数 0 能被精确表示。

对称量化取 `z = 0`，范围是 `[-127, 127]` 或 `[-128, 127]`；非对称量化用 `z` 平移，适合 ReLU 之后的非负激活。

### 粒度

| 粒度 | 说明 | 精度 | 开销 |
| --- | --- | --- | --- |
| Per-tensor | 整个张量一组 `s, z` | 最低 | 最小 |
| Per-channel | 每个输出通道一组 | 中等 | 中等 |
| Per-group | 每 32/64/128 个元素一组 | 最高 | 最大 |

LLM 权重通常用 per-channel 或 group-wise（如 group size 128）；激活用 per-tensor 或 per-token。

### 常见组合

| 组合 | 权重 | 激活 | 场景 |
| --- | --- | --- | --- |
| W8A8 | INT8 | INT8 | 服务器推理，Tensor Core 支持 |
| W4A16 | INT4 | FP16/BF16 | 显存受限，权重只读 |
| W4A8 | INT4 | INT8 | 平衡显存和算力 |
| W4A4 | INT4 | INT4 | 极端压缩，精度风险大 |

W4A16 在 decode 阶段很有吸引力：权重读取减半，激活仍然是高精度，kernel 只需要在 GEMM 前反量化权重。

### 零点与打包

INT4 每两个数打包进一个字节。解包、查表、反量化都要在 kernel 里做。GPTQ、AWQ、GGUF 的 Q4_K 等格式各有自己的打包方式，kernel 必须按格式解释字节。

## 12.7 Rounding and Numerical Error

### 误差来源

- **表示误差**：数不能被格式精确表示；
- **舍入误差**：每次运算后的舍入；
- **累加误差**：大量小量相加时的精度损失；
- **抵消误差**：两个相近的大数相减，有效位数骤减。

### 累加

FP16 的尾数只有 10 位。累加 8,192 个数时，如果直接顺序相加，小的数会被大的数“吃掉”。

解决办法：

- 用 FP32 累加器；
- 分块求和（pairwise summation），把误差从 `O(n)` 降到 `O(log n)`；
- Tensor Core 内部已经是 FP32 累加，不要在外面又转回 FP16。

### 量化误差

均匀量化的信噪比大约是：

$$
\text{SNR} \approx 6.02b + 1.76 \text{ dB}
$$

`b` 是位数。每多 1 bit，SNR 提高约 6 dB。INT8 约 50 dB，INT4 约 26 dB。

但 LLM 的分布不是均匀的，权重近似高斯，激活有严重的离群值。实际的量化误差取决于：

- 离群值有多大；
- 缩放粒度有多细；
- 是否对离群通道做特殊处理。

### 离群值

LLM 的激活里存在“超级通道”：某些维度的值比其余维度大几十倍。per-tensor 量化会被这些通道主导，其他通道被压成 0。

对策：

- per-channel / per-token 缩放；
- 保留离群通道为高精度（混合精度）；
- SmoothQuant 把激活的缩放迁移到权重上，让两边都好量化。

## 12.8 Accumulation Precision

矩阵乘法 `C = A B` 的每个元素是 `K` 次乘加。累加器的精度决定了结果的精度。

### Tensor Core 的累加

| 输入 | 累加 | 说明 |
| --- | --- | --- |
| FP16 | FP32 | 标准 HMMA，A100/H100 都是 |
| BF16 | FP32 | 同上 |
| TF32 | FP32 | 同上 |
| FP8 | FP32 | Hopper 起支持 |
| FP4 | FP32 | Blackwell |

FP32 累加不是可选项。如果累加器也是 FP16，`K = 8192` 的 GEMM 会产生肉眼可见的误差。

### K 维分块

大 `K` 的 GEMM 会被切成多个 `K` 块，每块累加到寄存器，最后再加起来。分块的顺序影响误差：

- 先加相近数量级的项，误差小；
- 先加大的项，小的项容易被吞掉。

cuBLAS 和 CUTLASS 的 tile 顺序是经过调优的，自己写 kernel 时不要随意改。

### 归约的精度

LayerNorm、softmax、attention 的归约都要用 FP32：

```cuda
float sum = 0.0f;
for (int i = 0; i < n; ++i) {
  sum += __half2float(x[i]) * __half2float(x[i]);
}
```

如果 `sum` 用 FP16，8,192 个平方和只要数值稍大就会超过 65,504，变成 inf。这不是“可能不准”，是直接得到错误结果。

## 12.9 Hardware Support and Tensor Core Implications

| GPU | FP32 | TF32 | FP16/BF16 | FP8 | FP4 |
| --- | --- | --- | --- | --- | --- |
| A100 | 19.5 | 156 | 312 | — | — |
| H100 | 67 | 495 | 989 | 1,979 | — |
| B200 | — | — | 约 2,250 | 约 4,500 | 约 9,000 |

单位是 TFLOPS，dense。每一代低精度算力大约翻倍。

### 对 roofline 的影响

精度降低同时改变两边：

- 峰值算力提高（FP16 → FP8 是 2 倍）；
- 权重和激活的字节数减半。

算术强度翻倍，脊点也翻倍。所以一个 kernel 属于算力受限还是带宽受限，通常不会因为换精度而改变；但绝对时间会减半。

真正的变化来自 batch 和融合，而不是精度本身。

### 对 kernel 的要求

- **打包**：FP4/INT4 要在一个字节里放多个值，解包要快；
- **缩放**：FP8/FP4 的 scale 要融合进 GEMM，不能单独一次 kernel；
- **累加**：FP32 累加器要常驻寄存器；
- **转换**：BF16/FP16/FP8 之间的转换要避免额外的 HBM 往返；
- **对齐**：Tensor Core 要求操作数按 16 字节或 128 字节对齐，边界要 padding。

低精度不是“把 dtype 改一下”。它要求整条数据通路都按这个格式设计。

### 选择建议

- **训练**：BF16 权重和激活，FP32 累加；需要更快时用 FP8，配合细粒度缩放；
- **推理，质量优先**：BF16 或 FP16；
- **推理，显存受限**：FP8 权重，必要时 KV Cache 也量化；
- **推理，极致压缩**：INT4/FP4 权重 + FP16 激活，配合好的量化算法；
- **精度敏感**：保留 FP32 的 norm、softmax 和累加。

格式的选择最终是一个约束优化问题：显存、带宽、算力、精度，四个约束里总有一个最先撞墙。

## 本章小结

- 浮点数由符号、指数、尾数组成；指数决定范围，尾数决定精度。
- FP16 精度高但范围窄，BF16 范围大但精度低；现代 LLM 训练默认 BF16。
- TF32 保持 FP32 范围，尾数截到 10 位，只在 Tensor Core 内使用。
- FP8 分 E4M3 和 E5M2，必须配合缩放；细粒度块缩放能显著降低离群值的影响。
- FP4 需要块缩放（MXFP4/NVFP4），Blackwell 原生支持，精度风险更高。
- 整数量化用仿射映射，`s` 和 `z` 决定范围；W4A16 是显存受限场景的常见选择。
- 累加必须用 FP32；归约、softmax、LayerNorm 的中间结果也要用 FP32。
- 低精度同时提高算力和降低字节数，不改变算子的受限类型，但会改变绝对时间。
- 格式选择取决于显存、带宽、算力和精度的约束，没有普遍最优解。

## 延伸阅读

- IEEE, [IEEE Standard for Floating-Point Arithmetic (IEEE 754-2019)](https://ieeexplore.ieee.org/document/8766229).
- Goldberg, [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://dl.acm.org/doi/10.1145/103162.103163), 1991.
- Micikevicius et al., [Mixed Precision Training](https://arxiv.org/abs/1710.03740), 2017.
- Micikevicius et al., [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433), 2022.
- OCP, [OCP Microscaling Formats (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
- Dettmers et al., [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339), 2022.
- Xiao et al., [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438), 2022.
- NVIDIA, [Transformer Engine](https://github.com/NVIDIA/TransformerEngine).
