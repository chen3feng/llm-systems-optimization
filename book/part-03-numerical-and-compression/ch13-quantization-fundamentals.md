---
chapter: 13
part: 3
title: Quantization Fundamentals
level: tbd
status: draft
prerequisites: [12]
lab:
---

# Chapter 13. Quantization Fundamentals

70B 模型的 BF16 权重是 140 GB，一张 80 GB 的 H100 装不下。换成 INT4，权重降到 35 GB，一张卡放得下；但 8K 上下文的 KV Cache 还要 2.6 GB，32K 时超过 10 GB，余量并不宽裕。

每权重少一个 bit，70B 模型少 8.75 GB，代价是输出会变。工程问题不是“能不能量化”，而是“在给定精度损失下，把哪个张量、按什么粒度、压到几位”。

## 13.1 Quantization Objectives

量化把连续值映射到离散整数：

$$
q = \text{round}\left(\frac{x}{s}\right) + z, \qquad \hat{x} = s(q - z)
$$

`s` 是 scale，`z` 是 zero-point，`q` 限制在 `[q_min, q_max]`。一个方案同时影响四个指标。

### 显存

权重显存约 `P × b / 8` 字节，`P` 是参数量，`b` 是每权重位数：

| 精度 | 每权重位数 | 70B 权重 | 相对 BF16 |
| --- | --- | --- | --- |
| BF16 / FP16 | 16 | 140 GB | 1.00 |
| FP8 / INT8 | 8 | 70 GB | 0.50 |
| INT4 | 4 | 35 GB | 0.25 |
| INT4, group 128 | 4.125 | 36.1 GB | 0.258 |

最后一行是真实的 INT4：每 128 个权重共享一个 FP16 scale，折合 0.125 bit/权重。宣传的“4-bit”通常指 4.125 或 4.5。

权重只是一部分。Llama 3 70B 有 80 层、8 个 KV 头、head_dim 128，FP16 下每 token 的 KV Cache 是：

$$
2 \times 80 \times 8 \times 128 \times 2 = 327{,}680 \text{ bytes} \approx 320 \text{ KB}
$$

8K 上下文是 2.6 GB，128K 是 42 GB。W4A16 不压 KV Cache；长上下文场景里，KV Cache 可能比权重更先撞墙。

### 带宽

decode 每生成一个 token 要读全部权重。H100 SXM 的带宽是 3.35 TB/s，140 GB 的理论下限是 `140/3350 ≈ 41.8 ms`；INT4 的 35 GB 是 10.4 ms。实际 kernel 能到 60%–80% 有效带宽，端到端通常从约 50 ms 降到约 14 ms，不是精确的 4 倍。

同一时刻的计算量是 `2P` FLOPs/token，70B 约 140 GFLOPs，按 989 TFLOPS FP16 只要 0.14 ms。算术强度约 1 FLOP/byte，decode 彻底带宽受限。W4A16 减少的是字节，不是 FLOPs。

### 算力

prefill 是大的 GEMM，batch × sequence 足够大时变成算力受限。只有激活也量化，才能用上更快的整数 Tensor Core。H100 的 INT8 与 FP8 dense 算力都是 1,979 TOPS/TFLOPS，是 FP16 的 2 倍。W4A16 在进入 Tensor Core 前要反量化回 FP16，prefill 不会因为 4-bit 变快。

### 精度

量化误差表现为 logits 偏移、perplexity 上升、任务准确率下降。它随层数、激活分布和输入长度变化，必须端到端测量。

四个目标互相牵制：位宽降低省显存和带宽，但损失精度；激活量化换来算力，却更容易被离群值破坏。

## 13.2 Symmetric and Asymmetric Quantization

### 非对称量化

把 `[x_min, x_max]` 映射到 `[q_min, q_max]`，由端点相等解出：

$$
s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}, \qquad
z = \text{round}\left(q_{\min} - \frac{x_{\min}}{s}\right)
$$

`z` 让实数 0 落在整数上：`q(0) = z`，反量化回 0。对 padding 和 ReLU 后的激活很重要，否则 0 会变成非零值，污染结果。

例：`x ∈ [-2, 6]`，INT8 非对称，`q ∈ [0, 255]`。`s = 8/255 = 0.03137`，`z = round(0 + 2/0.03137) = 64`。区间完整使用。

### 对称量化

强制 `z = 0`：

$$
s = \frac{\max(|x|)}{q_{\max}}, \qquad
q = \text{clip}\left(\text{round}\left(\frac{x}{s}\right), -q_{\max}, q_{\max}\right)
$$

同样的 `x ∈ [-2, 6]`，`s = 6/127 = 0.04724`。负半轴只用到 `-42`，`[-127, -43]` 这 85 个码点浪费，步长是非对称版本的 1.5 倍。

### 为什么对称量化的零点是 0

除了权重近似对称，还有系统原因：整数 GEMM 的累加。对称量化下：

$$
\sum_i \hat{a}_i \hat{b}_i = s_a s_b \sum_i q_{a,i} q_{b,i}
$$

非对称量化展开后多出 `z_b Σq_a`、`z_a Σq_b` 和 `K z_a z_b` 三个修正项，需要额外的行和归约和寄存器。对称量化把这些全部消掉，所以权重侧几乎总是对称的。

### 表示范围

| 格式 | `q_min` | `q_max` | 可用码点 | 说明 |
| --- | --- | --- | --- | --- |
| INT8 对称 | -127 | 127 | 255 | `-128` 常留空，保持对称 |
| INT8 非对称 | 0 | 255 | 256 | 适合非负激活 |
| INT4 对称 | -7 | 7 | 15 | 比 16 个码点少 6% 分辨率 |
| INT4 非对称 | 0 | 15 | 16 | 权重和激活都可用 |

INT4 少一个码点就是 6% 的分辨率，所以 4-bit 经常用满 `[-8, 7]` 或非对称；INT8 少一个只损失 0.4%。

| 维度 | 对称 | 非对称 |
| --- | --- | --- |
| 参数 | 只有 `s` | `s` 和 `z` |
| 零值 | 精确为 0 | 精确为 0（`z` 对齐） |
| 范围利用 | 受绝对值最大端支配 | 贴合 `[min, max]` |
| 整数 GEMM | 直接累加 | 需要行和修正 |
| 典型用途 | 权重、Tensor Core | 激活、非负分布 |

## 13.3 Per-tensor、Per-channel and Group-wise Quantization

粒度决定多少个数共享一组 `s, z`。越细越能适应局部动态范围，元数据和 kernel 开销也越大。

- **Per-tensor**：整个张量一组参数。
- **Per-channel**：每个输出通道一组；对 `[out, in]` 权重就是每行一组。
- **Group-wise**：每个通道内按输入维切成长度 `g` 的组，每组一组参数。

对一个 `4096 × 4096` 矩阵（16.78M 权重）：

| 粒度 | scale 数量 | FP16 元数据 | 额外 bit/权重 | 4-bit 实际位宽 |
| --- | --- | --- | --- | --- |
| Per-tensor | 1 | 2 B | 约 0 | 4.000 |
| Per-channel | 4,096 | 8 KB | 0.0039 | 4.004 |
| Group 128 | 131,072 | 256 KB | 0.125 | 4.125 |
| Group 64 | 262,144 | 512 KB | 0.250 | 4.250 |
| Group 32 | 524,288 | 1 MB | 0.500 | 4.500 |

Group 32 的元数据是 group 128 的 4 倍。对 70B 模型，group 128 的 scale 约 1.1 GB，group 32 约 4.4 GB，在 35 GB 权重上不是零头。

### 误差

收益来自“局部最大值”更小。一个通道里若有 10 倍于其余值的离群点，per-tensor 的 scale 被它撑大，其余值挤在少数码点上；group 把离群点关进 128 个元素，只影响这一组。

在 `512 × 512`、iid `N(0,1)` 的矩阵上做对称量化，实测 SQNR：

| 位宽 | Per-tensor | Per-channel | Group 128 | Group 32 |
| --- | --- | --- | --- | --- |
| 8 | 39.7 dB | 42.6 dB | 43.8 dB | 45.4 dB |
| 4 | 14.6 dB | 17.4 dB | 18.6 dB | 20.3 dB |
| 3 | 7.2 dB | 10.1 dB | 11.2 dB | 12.9 dB |
| 2 | 0.3 dB | 1.5 dB | 2.3 dB | 3.6 dB |

这是无离群值的理想情况，粒度差距已有 4–6 dB。真实权重更重尾，端到端影响还会被后续层放大或平均，单矩阵 SQNR 不能直接外推。

### Group size 的取舍

| group size | 优点 | 缺点 | 典型使用 |
| --- | --- | --- | --- |
| 32 | 误差最小，贴合 32 线程 | 0.5 bit/权重，归约频繁 | GGUF sub-block、低 bit 实验 |
| 64 | 误差和开销折中 | 对齐不如 32/128 自然 | 部分 AWQ/GPTQ 配置 |
| 128 | 0.125 bit/权重，误差可接受 | 组内仍有局部离群风险 | GPTQ、AWQ 默认 |

`g = 1` 等于每个权重一个 scale，没有压缩收益；group 32 又比 group 128 多 4 倍缩放指令。实际选择通常由 kernel 和格式决定：GPTQ/AWQ 默认 128，GGUF 用 32 元素 sub-block。

### 激活的粒度

权重的 per-channel 沿输出维，scale 可以折进输出，几乎免费。激活不同：

- **Per-tensor**：最省，但被离群值支配。
- **Per-token**：每个 token 一个 scale，decode 时天然适合动态量化。
- **Per-channel（per hidden dim）**：能精确处理超级通道，但 scale 沿 GEMM 的 K 维变化，Tensor Core 不能直接接受。

这正是 SmoothQuant 要解决的：把激活的 per-channel scale 迁到权重，让激活回到 per-tensor，从而用上 W8A8 整数 GEMM。细节在 Chapter 14。

## 13.4 Scale and Zero-point

### 从 min/max 计算

非对称量化：

$$
s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}, \qquad
z = \text{round}\left(q_{\min} - \frac{x_{\min}}{s}\right)
$$

对称量化：

$$
s = \frac{\max(|x_{\min}|, |x_{\max}|)}{q_{\max}}, \qquad z = 0
$$

两个公式都假设 `x_min < x_max`。若整个 group 全是 0，`s = 0`，要特殊处理成极小值，否则除零。

### 一个完整的例子

INT8 对称，权重一行是 `[-0.031, 0.052, 0.007, -0.018]`：

$$
s = \frac{0.052}{127} = 4.094 \times 10^{-4}
$$

量化得到 `q = [-76, 127, 17, -44]`，反量化得到 `[-0.0311, 0.0520, 0.00696, -0.0180]`。最大误差是半个 step（`2.05e-4`）。均匀量化的误差方差是 `s²/12`。

### Clipping

min/max 对离群值没有抵抗力。激活最大值是 20、99.9 分位数是 4.0 时：

- 不裁剪：`s = 20/127 = 0.157`，绝大多数值只落在少数码点。
- 裁剪到 4.0：`s = 4/127 = 0.0315`，分辨率提高 5 倍，0.1% 的值被截断。

截断产生偏差，舍入产生方差，最优裁剪点让两者之和最小：

$$
t^* = \arg\min_t \; \mathbb{E}\left[(x - Q_t(x))^2\right]
$$

对 Gaussian 的低 bit 量化，`t` 常取 `2.5σ` 到 `3σ`，位宽越高越接近 `4σ`；对 LLM 激活，直接用 99.9% 或 99.99% 分位数更稳。权重分布紧凑，裁剪收益小；激活收益大，但要用端到端指标验证，不能只看 MSE。

### Zero-point 的细节

对称量化的 `z = 0` 是明确选择“实数 0 对应整数 0”，带来两个性质：反量化是纯乘法；点积不需要修正项。非对称的 `z` 在 INT8 是 `[-128, 127]`，INT4 是 `[-8, 7]`。`z` 也占存储，per-group 时和 scale 一样多，所以 4-bit 权重格式常只用对称量化。

## 13.5 Calibration and Activation Statistics

权重范围直接从权重算；激活范围依赖输入，必须用代表性样本跑前向统计，这叫校准。

| 方法 | 统计量 | 优点 | 缺点 |
| --- | --- | --- | --- |
| Min-max | `min`、`max` | 简单，覆盖所有值 | 一个离群点毁掉整个张量 |
| Percentile | 99.9% / 99.99% 分位 | 抗离群，分辨率高 | 依赖样本量，尾部被截断 |
| MSE | 最小化重建误差 | 直接优化目标 | 需要搜索裁剪点，慢 |
| KL 散度 | 量化前后分布差异 | 适合激活，TensorRT 使用 | 需要直方图，实现复杂 |

KL 校准最小化：

$$
D_{\text{KL}}(P \,\|\, Q) = \sum_i p_i \log \frac{p_i}{q_i}
$$

`P` 是 FP32 激活直方图，`Q` 是量化后反量化的直方图，在候选裁剪点里选 KL 最小的。

常见配置：GPTQ 用 128 条、每条 2048 token（C4）；AWQ 用 128 条（Pile）；SmoothQuant 用 512 条。128 条已覆盖约 26 万 token，足以估计每层统计量，再加收益递减而前向时间线性增长。对 70B，跑一遍校准要几十分钟到几小时。

更重要的是分布匹配。用英文维基校准、部署中文或代码，激活的离群通道和幅度可能不同。校准集应混入目标领域样本。W8A8 激活量化对此尤其敏感；W4A16 的 scale 主要来自权重，受影响较小。

### 动态量化

校准得到静态 scale。输入分布漂移时，动态量化在每次前向用当前 tensor 的 `max` 现算：per-token 动态适合 decode；per-tensor 动态需要全局归约，难和 GEMM 重叠；延迟缩放用上一步的 `max` 估计，NVIDIA Transformer Engine 的 FP8 采用这种。W8A8 整数 GEMM 通常要求静态 per-tensor scale，因为累加结果要在最后统一乘 `s_a s_b`。

## 13.6 Outlier Handling

Dettmers 等人在 LLM.int8() 里观察到：模型超过 6.7B 后，少数隐藏维度出现幅度高达其余维度 20 倍以上的“超级通道”，且在不同 token、样本上稳定出现。

这是结构性的。假设 4096 维里有 4 个通道是 20、其余是 1，per-tensor 的 `max = 20`，其余维度的有效位数只剩 `log2(127/20) ≈ 2.7` bit，INT8 退化成 3-bit。

| 方法 | 做法 | 代价 | 适用 |
| --- | --- | --- | --- |
| Per-channel 权重 | 每个输出通道一组 scale | 几乎为零 | 所有权重量化 |
| Per-token 激活 | 每个 token 一组 scale | 小，需动态归约 | W8A8 激活 |
| Per-channel 激活 | 每个隐藏维一组 scale | kernel 复杂，GEMM 不支持 | 离线分析、SmoothQuant |
| 混合精度 | 离群通道保留 FP16 | 分支和额外带宽 | LLM.int8() |
| 旋转/重参数化 | 把离群值摊到多个维度 | 额外矩阵乘 | 研究型方案 |

LLM.int8() 用阈值（论文中是 6.0）检测离群通道，这些通道用 FP16 算，其余用 INT8，最后相加。它证明少量通道承担了大部分误差，但 kernel 要在两种精度间切换，速度提升有限。

SmoothQuant 换思路：不保留离群通道，而是对 `Y = XW` 插入对角矩阵 `S`：

$$
Y = (X S^{-1})(S W)
$$

把激活的离群幅度压下去，同时放大权重的对应通道。激活变得好量化，权重仍然好量化。公式和 `α` 的选取在 Chapter 14。

per-token 量化对“不同 token 幅度差异大”有效，但对“同一通道在所有 token 上都大”的超级通道无效：每个 token 的 scale 都被同一个通道撑大。真正解决超级通道的是 per-channel 激活或 SmoothQuant，这个区别经常被混淆。

## 13.7 Weight-only and Weight-activation Quantization

| 组合 | 权重 | 激活 | 70B 权重 | 加速来源 | 典型算法 |
| --- | --- | --- | --- | --- | --- |
| W8A8 | INT8 | INT8 | 70 GB | 带宽减半 + INT8 算力 | SmoothQuant |
| W4A16 | INT4 | FP16/BF16 | 35 GB | 权重带宽降到 1/4 | GPTQ、AWQ |
| W4A8 | INT4 | INT8 | 35 GB | 权重带宽 + INT8 算力 | 专门 kernel |
| W4A4 | INT4 | INT4 | 35 GB | 两者都降 | 研究阶段 |

**W8A8** 是最“正统”的方案：权重和激活都是 INT8，GEMM 用 INT8 Tensor Core（IMMA），累加器是 INT32。权重带宽减半，prefill 算力翻倍。难点在激活：per-tensor 会被离群值摧毁，per-channel 又不受 GEMM 支持。SmoothQuant 把难度迁到权重，才让 W8A8 落地。

**W4A16** 只量化权重，激活保持 FP16/BF16。它不追求算力，只追求带宽：权重读取从 140 GB 降到 35 GB，反量化在 kernel 内完成。它不需要校准激活，流程简单，是单卡部署 70B、消费级 GPU 跑 7B/13B 的主流方案。GPTQ、AWQ 都主要面向 W4A16。缺点是 prefill 不变快，实际加速比通常低于 4 倍。

**W4A8** 想同时拿显存和算力。硬件没有原生的 INT4×INT8 MMA，kernel 通常先把 INT4 反量化成 INT8 再做 IMMA。反量化开销吃掉一部分收益，生态比 W4A16 窄。

**W4A4** 两边都是 4-bit。激活的 4-bit 误差很大，需要旋转、混合精度或 QAT 才能保住质量，目前主要用于研究和小模型。

选择：

- **显存是唯一瓶颈、decode 为主**：W4A16，成熟、风险低。
- **prefill 吞吐也重要、有 INT8 硬件**：W8A8，需要校准和 SmoothQuant。
- **两者都要、愿意维护专用 kernel**：W4A8。
- **极端压缩、能接受精度损失**：W4A4 或混合精度。

KV Cache 量化是独立的轴。W4A16 的模型仍然可以有 FP16 KV Cache，也可以额外压到 INT8/FP8，长上下文要单独评估。

## 13.8 Quantization Error Analysis

### 四个指标

MSE：

$$
\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \hat{x}_i)^2
$$

SQNR：

$$
\text{SQNR} = 10\log_{10}\frac{\sum_i x_i^2}{\sum_i (x_i - \hat{x}_i)^2}
$$

Cosine similarity 衡量方向一致性；perplexity `PPL = exp(-(1/N)Σ log p(x_i))` 把整个网络的误差压成一个数。

$$
\cos(\mathbf{x}, \hat{\mathbf{x}}) = \frac{\mathbf{x}\cdot\hat{\mathbf{x}}}{\|\mathbf{x}\|_2\|\hat{\mathbf{x}}\|_2}
$$

### 每 bit 约 6 dB

对满量程正弦，均匀量化的理论 SQNR 是：

$$
\text{SNR} \approx 6.02b + 1.76 \text{ dB}
$$

步长 `Δ = 2A/2^b`，误差方差 `Δ²/12`，信号功率 `A²/2`，相除取对数即得。每多 1 bit，步长减半，噪声功率降到 1/4，即 6.02 dB。

| 位宽 | 理论 SQNR（满量程正弦） | 实测 SQNR（Gaussian，per-tensor） | 实测 cosine |
| --- | --- | --- | --- |
| 8 | 49.9 dB | 39.7 dB | 0.99995 |
| 4 | 25.8 dB | 14.6 dB | 0.9829 |
| 3 | 19.8 dB | 7.2 dB | 0.9161 |
| 2 | 13.8 dB | 0.3 dB | 0.3963 |

实测来自 13.3 节的 `512 × 512` 模拟。绝对值低约 10 dB，因为 Gaussian 的峰值因子大于正弦，max 缩放浪费了范围；斜率仍是每 bit 约 6 dB：4-bit 到 8-bit 涨了 25.1 dB。

### SQNR 与 cosine

若量化误差与信号独立，cosine 可由 SQNR 近似：

$$
\cos \approx \frac{1}{\sqrt{1 + 10^{-\text{SQNR}/10}}}
$$

18.6 dB 对应 `10^-1.86 = 0.0138`，`cos ≈ 0.9932`，与 group 128、4-bit 的结果一致。看到 cosine 就能反推等效 SQNR。

### 从单层误差到端到端

单层 SQNR 不能直接预测 perplexity，原因有三：80 层误差会沿残差叠加，也会被归一化和注意力平均；随机零均值误差比裁剪偏差危害小；perplexity 对常见 token 不敏感，对长尾和推理链敏感。

经验上，W8A8 的 perplexity 变化在 0.01–0.1 量级，W4A16 group 128 在 0.1–0.5 量级，3-bit 开始明显恶化。但这些数字高度依赖模型、校准集和评估集，不能当常数。可靠结论只能来自自己的端到端对比：同一份校准数据、同一套 prompt、同一个 tokenizer。

还有一点：量化误差不均匀。绝对值大的权重绝对误差大、相对误差小；绝对值小的可能被量化到 0。小权重可能控制长尾行为，所以“平均 MSE 很小”不等于“模型行为不变”。

## 本章小结

- 量化在显存、带宽、算力、精度之间取舍；70B 模型 BF16 是 140 GB，INT8 是 70 GB，INT4 是 35 GB。
- 仿射量化由 `s`、`z` 定义；对称量化取 `z = 0`，让零值精确、整数 GEMM 无需修正项。
- INT8 有 255/256 个码点，INT4 只有 15/16 个；4-bit 少一个码点就是 6% 分辨率。
- per-channel 几乎免费，group-wise 有效但要付元数据：group 128 是 4.125 bit/权重，group 32 是 4.5 bit。
- 校准方法有 min-max、percentile、MSE、KL；校准集的分布匹配比数量更重要。
- LLM 激活存在超级通道，per-tensor 会被少数通道支配；per-token 解决不了通道级离群，SmoothQuant 通过迁移 scale 解决。
- W4A16 省带宽、不改算力，适合 decode；W8A8 同时省带宽和提算力，但依赖激活校准。
- 均匀量化的 SQNR 约每 bit 提高 6 dB；cosine 可由 SQNR 反推，端到端质量必须用 perplexity 和任务指标验证。

## 延伸阅读

- Jacob et al., [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877), CVPR 2018.
- Krishnamoorthi, [Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper](https://arxiv.org/abs/1806.08342), 2018.
- Nagel et al., [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295), 2021.
- Gholami et al., [A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630), 2021.
- Dettmers et al., [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339), 2022.
- Xiao et al., [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438), 2022.
- PyTorch, [Quantization documentation](https://pytorch.org/docs/stable/quantization.html).
