---
chapter: 14
part: 3
title: Practical LLM Quantization
level: tbd
status: draft
prerequisites: [12, 13]
lab:
---

# Chapter 14. Practical LLM Quantization

一个 70B 模型的 Q4_K_M GGUF 文件约 40 GB，可以在一台 64 GB 内存的机器上跑起来；同样的模型 FP16 是 140 GB。把权重压到 4-bit 的算法不止一种：GPTQ 用 Hessian 做逐列补偿，AWQ 保护约 0.1%–1% 的显著权重，SmoothQuant 把激活的离群值迁到权重上。

这些方法的目标相同——让量化后的层输出接近 FP32——但假设、成本和适用位宽不同。这一章讲它们的算法、关键参数、GGUF 的块结构，以及量化矩阵乘和基准测试的工程细节。

## 14.1 Post-training Quantization

PTQ（post-training quantization）不重新训练模型，只用量化算法和少量校准数据。流程通常是四步：

1. **确定目标**：位宽、group size、权重/激活组合、目标硬件。
2. **校准**：用 128–512 条代表性样本跑前向，收集每层输入激活、`max`、Hessian 或直方图。
3. **逐层重建**：对每一层单独最小化 `||WX - ŴX||²`，假设层间误差不耦合。
4. **导出与评估**：转成目标格式（GPTQ、AWQ、GGUF），测 perplexity、任务准确率和延迟。

逐层重建是 PTQ 的核心近似。它把整个网络的最优量化拆成每层独立的子问题，计算量从“训练”降到一次校准前向加每层一个小规模重建。层与层之间的误差会传播，但量化后的模型离 FP32 不远，逐层最优通常足够好。

| 方法 | 校准数据 | 优化方式 | 典型位宽 | 成本 |
| --- | --- | --- | --- | --- |
| RTN | 不需要 | 直接舍入 | 任意 | 秒级 |
| GPTQ | 128 × 2048 token | 逐列重建，Hessian 补偿 | 3–4 bit | 分钟到小时 |
| AWQ | 128 条 | 逐通道 scale 搜索 | 4 bit | 约半小时 |
| SmoothQuant | 512 条 | 激活到权重的 scale 迁移 | 8 bit | 约一小时 |
| QAT | 训练数据 | 反向传播，学习舍入 | ≤4 bit | GPU 天级 |

表中的成本以 7B 模型、单卡为参考。RTN（round-to-nearest）是基线：按 min/max 算 scale，直接舍入。它在 8-bit 上够用，4-bit 下明显差于 GPTQ/AWQ，因为舍入误差没有被补偿。

## 14.2 GPTQ

GPTQ 把逐层量化写成一个最小二乘问题。对一层的权重 `W`（`d_out × d_in`）和校准输入 `X`（`d_in × N`）：

$$
\min_{\hat{W}} \; \| WX - \hat{W}X \|_2^2
$$

它源自 OBQ（optimal brain quantization）：一次量化一个权重，然后调整其余权重来补偿这个误差。对第 `i` 个权重，补偿量是：

$$
\delta = -\frac{w_i - \hat{w}_i}{[H^{-1}]_{ii}} \, H^{-1}_{:,i}
$$

其中 `H = 2XX^T` 是 Hessian，`H^{-1}_{:,i}` 是逆矩阵的第 `i` 列。直观上，`H` 描述了哪些权重对输出影响大；`[H^{-1}]_{ii}` 小的方向，量化误差会被放大，所以补偿要多。

OBQ 每量化一个权重都要更新剩余权重并维护 Hessian 逆，整体复杂度是 `O(d³)`，大模型不可行。GPTQ 做了三个工程化改动：

1. **固定顺序**：所有行按同一列顺序量化，可以批量更新。
2. **Cholesky**：用 Cholesky 分解一次算出 `H^{-1}`，避免逐次求逆。
3. **块更新**：一次处理 128 列，把 rank-1 更新聚成矩阵乘。

伪代码：

```python
H = 2 * X @ X.T
H += damp * torch.eye(d_in)          # damp 通常取 diag 均值的 1%
Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))

for i in range(d_in):
    w = W[:, i]
    q = quantize(w, scale[:, i // group_size])
    err = (w - q) / Hinv[i, i]
    W[:, i] = q
    W[:, i+1:] -= err[:, None] * Hinv[i, i+1:][None, :]
```

`W[:, i+1:]` 的更新就是“用量化误差修正剩余列”。如果没有校准数据，`H = I`，`Hinv` 是对角阵，补偿项消失，GPTQ 退化成 RTN。

### 关键参数

- **group size**：默认 128。每组独立算 scale，组内仍按列补偿。128 是精度和元数据的折中；64 更准，但 scale 翻倍。
- **act-order（desc_act）**：按 `diag(H)` 从大到小重排列，先量化影响大的列，让后面的列有更多机会补偿。它通常能改善 3-bit 和 4-bit 的质量，但推理时权重列的顺序变了，kernel 要支持 permutation，或者导出时把顺序还原。
- **damp**：给 `H` 加一个小对角项，防止奇异。常用 `0.01 × mean(diag(H))`。
- **位宽**：4-bit 在多数模型上接近无损；3-bit 需要 group 128、act-order，且质量开始明显下降；2-bit 目前不实用。

### 为什么需要校准数据

`H` 完全由输入激活决定。权重的量化误差对输出的影响是 `δ^T H δ`，`H` 大的方向必须更小心。没有校准数据，就不知道哪些输入方向重要，只能假设各向同性，补偿无从谈起。校准集也不必很大：GPTQ 论文用 128 条 2048-token 序列，在 175B 模型上约 4 GPU 小时；7B 模型通常几十分钟。

## 14.3 AWQ

AWQ（activation-aware weight quantization）的观察是：权重不是同等重要的。按激活幅度排序，约 0.1%–1% 的权重承担了大部分输出能量；把这些“显著权重”保护好，其余权重可以压得更狠。

AWQ 不逐列重建，而是对每个输入通道乘一个 scale：

$$
Y = WX = (W \operatorname{diag}(s))(\operatorname{diag}(s)^{-1} X)
$$

量化的是 `W diag(s)`，激活侧除以 `s`。`s` 按通道选：

$$
s_j = \frac{(\max |X_j|)^\alpha}{(\max |W_j|)^{1-\alpha}}
$$

`α ∈ [0, 1]`，通常网格搜索若干点（AWQ 官方实现默认 20 个），选让层输出 MSE 最小的。`α = 0` 时 `s` 只由权重决定，`α = 1` 时只由激活决定。直观上，激活大的通道把权重放大，量化时的相对误差变小；激活侧除以同样的因子，数学上完全等价。

关键点：

- **不需要反向传播**。只用校准激活的 `max` 和少量前向，搜索 `s` 的指数。比 GPTQ 便宜，也好并行。
- **scale 可以折进前一层**。W4A16 下，`diag(s)^{-1}` 作用在激活上；如果前一个是 RMSNorm，它的 per-channel 权重可以直接吸收 `1/s`，推理时零开销。
- **粒度**：W4A16、group 128、非对称（带 zero-point）。显著权重的保护通过 per-channel scale 实现，不是混合精度。
- **效果**：4-bit 下通常优于 RTN，和 GPTQ 接近；3-bit 下两者的排序更依赖模型和 group size，需要实测。

AWQ 的“保护 0.1%–1%”不是把权重挑出来单独存 FP16，而是让这些通道的量化步长更小。真正的混合精度方案见 14.6 节。

## 14.4 SmoothQuant

SmoothQuant 处理的是 W8A8 的激活离群值。激活按通道有超级通道，权重相对好量化；直接把激活压到 INT8，per-tensor scale 会被少数通道撑大。

它插入一个对角矩阵 `S`：

$$
Y = XW = (X S^{-1})(S W)
$$

`S` 的第 `j` 个对角元是：

$$
s_j = \frac{\max |X_j|^\alpha}{\max |W_j|^{1-\alpha}}
$$

`α` 控制“把多少难度从激活迁到权重”。`α = 0.5` 是论文的默认值，常见范围 0.4–0.6。

| `α` | 激活 | 权重 | 结果 |
| --- | --- | --- | --- |
| 0 | 保持离群 | 不变 | 激活难量化，W8A8 失败 |
| 0.5 | 离群被压一半 | 对应通道放大 | 两边都在 INT8 范围内 |
| 1 | 完全平滑 | 承担全部离群 | 权重出现新离群，精度下降 |

SmoothQuant 是离线的，不需要训练，校准集用 512 条左右。处理后：

- 激活可以用 per-tensor 或 per-token 的 INT8；
- 权重用 per-channel 的 INT8；
- GEMM 走 INT8 Tensor Core，累加 INT32。

它只改变尺度，不改变数值，所以可以和权重量化叠加：先做 SmoothQuant 把激活变成 W8A8，再用 GPTQ/AWQ 把权重进一步压到 4-bit，得到 W4A8。叠加时要注意两次 scale 的先后顺序，否则会重复缩放。

## 14.5 Quantization-aware Training

QAT 在训练的前向里插入“伪量化”节点：

$$
\hat{x} = s \cdot \text{clamp}\left(\text{round}\left(\frac{x}{s}\right), -Q_n, Q_p\right)
$$

前向用的是量化后的 `x̂`，反向传播时 `round` 的导数为零，无法直接求梯度。STE（straight-through estimator）把 `round` 的导数近似为 1：

$$
\frac{\partial L}{\partial x} \approx \frac{\partial L}{\partial \hat{x}}
$$

这样梯度可以穿过量化节点，更新前面的权重。STE 是有偏的，但实践上足够稳定。

LSQ（learned step size quantization）进一步把 `s` 变成可学习参数。对范围内的值：

$$
\frac{\partial \hat{x}}{\partial s} = q - \frac{x}{s}, \qquad q = \text{round}\left(\frac{x}{s}\right)
$$

超出范围时梯度是 `-Q_n` 或 `Q_p`。这样 `s` 会根据任务损失自动调整，而不是固定用 min/max。

### 什么时候值得做

QAT 的成本是完整的训练流程：数据、GPU 小时、超参数、调参。它适合：

- PTQ 在 4-bit 以下失败，且精度不能妥协；
- 领域分布和校准集差异大，静态 scale 不可靠；
- 小模型（<1B）或极端压缩（2–3 bit），PTQ 的误差无法接受；
- 有现成的训练管线和标注数据。

对 70B 模型的一次性部署，QAT 通常不划算：GPTQ/AWQ 的 4-bit 已经够好，QAT 的训练成本很难回收。反过来，如果模型要长期服务、对质量极其敏感，QAT 的收益可以摊薄。

## 14.6 Mixed-precision Quantization

不同层对量化的敏感度不同。首尾层、embedding、lm_head 通常更敏感；部分研究报告 `down_proj` 和早期 attention 也容易受影响。敏感层保留 8-bit 或 FP16，其余压到 4-bit，可以在相同平均位宽下拿到更好的质量。

| 组件 | 典型位宽 | 理由 |
| --- | --- | --- |
| Embedding / lm_head | 8-bit 或 FP16 | 直接决定输入输出，误差不可恢复 |
| 前 1–2 层 | 8-bit | 靠近输入，误差会逐层放大 |
| 后 1–2 层 | 8-bit | 直接产生 logits |
| 中间 attention/MLP | 4-bit | 占参数大头，误差可被平均 |
| `down_proj` | 4-bit 或 8-bit | 部分模型对其敏感，需实测 |
| RMSNorm / softmax | FP16/FP32 | 归约和指数，不能低精度 |
| KV Cache | 8-bit 或 FP16 | 与权重量化独立评估 |

分配方法有两种：

1. **静态规则**：按上表，首尾层 8-bit，其余 4-bit。GGUF 的 Q4_K_M 就是这种思路——对一半的 `attention.wv` 和 `feed_forward.w2` 用 Q6_K，其余用 Q4_K。
2. **敏感度搜索**：逐层量化，测 perplexity 增量，按“每 bit 换来的误差”排序，在显存预算下贪心分配。它更准，但要对每层跑评估，成本高。

混合精度的平均位宽可能只比纯 4-bit 高 0.2–0.5 bit，但质量接近 8-bit。代价是 kernel 要处理多种格式，加载器要按张量选择反量化路径。

## 14.7 GGUF and Runtime-oriented Quantization

GGUF 是 llama.cpp 生态的文件格式：一个文件包含 metadata（模型结构、tokenizer、超参数）、张量信息和按块排列的量化数据。它支持 mmap、CPU/GPU 混合卸载，是本地推理最常用的格式。

### 块结构

GGUF 的量化以 block 为单位，每个 block 共享 scale（有的还有 min）：

| 类型 | 每 block 权重 | 每 block 字节 | 有效位宽 | 结构 |
| --- | --- | --- | --- | --- |
| Q4_0 | 32 | 18 | 4.5 | FP16 `d` + 16 B 的 4-bit 值 |
| Q4_K | 256 | 144 | 4.5 | `d`、`dmin`、12 B 的 6-bit scales/mins、128 B 的 4-bit 值 |
| Q5_K | 256 | 176 | 5.5 | Q4_K 结构 + 32 B 高位 |
| Q6_K | 256 | 210 | 6.5625 | 128 B 低位 + 64 B 高位 + 16 B 的 8-bit scales + FP16 `d` |
| Q8_0 | 32 | 34 | 8.5 | FP16 `d` + 32 B 的 INT8 值 |

Q4_0 是对称的：`x = d · q`，`q ∈ [-8, 7]`。Q8_0 也是：`x = d · q`。Q4_K 和 Q5_K 是非对称的：

$$
x = d \cdot s_c \cdot q - d_{\min} \cdot m_c
$$

`d`、`dmin` 是 super-block 级的 FP16 缩放，`s_c`、`m_c` 是每个 32 元素 sub-block 的 6-bit scale 和 min。

### k-quants 的 super-block

Q4_K/Q5_K/Q6_K 属于 k-quants。一个 super-block 是 256 个权重，分成 8 个 sub-block，每个 32 个。8 个 scale 和 8 个 min 各 6 bit，共 `8 × 6 × 2 = 96 bit = 12 B`，打包在 super-block 头里。

为什么是 256：32 个元素的 sub-block 保证局部动态范围，256 的 super-block 把 scale 的元数据摊薄到 0.375 bit/权重（12 B / 256），同时让一次 SIMD 读取覆盖足够多的数据。Q4_K 的 144 B 正好是 4.5 bit/权重，和 Q4_0 相同，但精度更好，因为 scale 更细。

`Q4_K_M` 是混合版本：一半的 `attention.wv` 和 `feed_forward.w2` 用 Q6_K，其余用 Q4_K，平均约 4.8 bit/权重。70B 模型因此约 42 GB。

### 为什么 CPU 推理常用 GGUF

- **反量化便宜**：每个 block 只做一次乘加，解包是位操作和查表，CPU 的 SIMD 能顺序处理。
- **不需要 Tensor Core**：CPU 没有矩阵乘单元，瓶颈是内存带宽；4-bit 直接把字节数减到 FP16 的 1/4。
- **内存布局连续**：block 按行排列，mmap 后可以按页读取，不需要把整个模型加载到 RAM。
- **单一文件**：metadata、tokenizer、权重在一起，分发和版本管理简单。
- **CPU/GPU 混合**：可以把部分层放到 GPU，其余留在 CPU，按显存自动分配。

代价是格式和 kernel 绑定。GGUF 的 block 布局是为 llama.cpp 的 kernel 设计的，换一个推理引擎不一定能直接高效使用。

## 14.8 Quantized Matrix Multiplication

W4A16 的 decode 是 GEMV：一个 token 的激活向量乘权重矩阵。以 GGUF 的 Q4_0 为例，权重每字节打包两个 4-bit 值，低 nibble 在前。kernel 在寄存器里反量化，再做 FP16 FMA，累加用 FP32。

```cpp
// W4A16 GEMV，group size = 32；q 为 4-bit 有符号，范围 [-8, 7]
float dot_w4a16(const uint8_t* w, const half* x,
                const half* scales, int K) {
  float acc = 0.f;
  for (int g = 0; g < K / 32; ++g) {
    float partial = 0.f;
    for (int i = 0; i < 32; ++i) {
      uint8_t byte = w[(g * 32 + i) >> 1];
      int q = (i & 1) ? (byte >> 4) : (byte & 0x0F);
      q = (q ^ 8) - 8;                     // 符号扩展
      partial += __half2float(x[g * 32 + i]) * (float)q;
    }
    acc += partial * __half2float(scales[g]);  // 每组乘一次 scale
  }
  return acc;
}
```

真实的 kernel 不会这样逐元素写。它会：

- 一次读 32 bit，解出 8 个 int4，用 `prmt`/`lop3` 做并行解包；
- 用 `half2` 做两路 FMA，把反量化和乘加重叠；
- 每个 warp 负责一行或几行，把 scale 放在寄存器；
- 对 group 32，先累加组内点积，再乘 scale，避免每个元素都乘。

### 指令

| 指令 | 用途 | 精度 |
| --- | --- | --- |
| `__dp4a` | 4 个 int8 点积累加到 int32 | W8A8 的标量路径 |
| IMMA | INT8 Tensor Core 矩阵乘 | W8A8 |
| HMMA | FP16/BF16 Tensor Core | W4A16 反量化后 |
| `prmt` / `lop3` | 解包 4-bit、重排字节 | W4A16 |

W4A16 在 Tensor Core 上的路径是：读 4-bit 权重 → 解包 → 转 FP16 → HMMA。反量化发生在寄存器或 shared memory，权重不会以 FP16 形式写回 HBM。这是它省带宽的关键：HBM 上永远只有 4-bit。

W8A8 用 IMMA，输入是 INT8，累加是 INT32。矩阵乘结束后统一乘 `s_a s_b`，再转回 FP16。per-channel 权重的 scale 可以在输出时逐行应用；per-group 权重则要在 K 维分组累加，每组乘一次。

反量化的开销决定了 W4A16 的实际加速比。decode、batch 1 时，反量化可以和内存读取重叠，效率较高；prefill、batch 大时，计算变密，反量化指令会挤占 FMA 的发射槽，W4A16 可能不如 W8A8。

## 14.9 Quantization Benchmarking

量化模型的评估要同时看质量和性能，两者都不能只看一个数。

| 指标 | 测什么 | 注意 |
| --- | --- | --- |
| Perplexity | 语言建模损失 | 数据集、tokenizer、上下文长度必须一致 |
| 任务准确率 | MMLU、HellaSwag、GSM8K 等 | few-shot 设置、prompt 模板会影响结果 |
| TTFT | 首 token 延迟 | 主要受 prefill 和权重加载影响 |
| TPOT | 每 token 输出时间 | decode 带宽受限，最能体现 W4A16 的收益 |
| 显存 | 权重 + KV Cache + 激活 | 量化权重不等于量化 KV Cache |
| 吞吐 | tokens/s | 要注明 batch 和并发数 |

### 常见误判

- **只比 perplexity**：4-bit 模型的 perplexity 可能只涨 0.2，但指令跟随、长上下文检索、代码生成的质量可能明显下降。要补任务评测和人工抽查。
- **不同 tokenizer 比 perplexity**：wikitext2 的 perplexity 依赖分词。Llama 和 GPT-2 的 tokenizer 不同，数值不可直接比较。
- **batch 和上下文不一致**：W4A16 在 batch 1 快，在 batch 32 可能被反量化拖慢；TTFT 和 TPOT 的排序可能相反。
- **忽略 KV Cache**：权重量化到 4-bit，KV Cache 仍是 FP16，长上下文下显存可能没省多少。
- **校准集泄漏**：校准样本和评估样本重叠，perplexity 会虚低。
- **benchmark 污染**：任务准确率受训练数据污染影响，量化前后的差异可能被噪声淹没。
- **没有 warmup**：第一次运行包含加载、编译、时钟爬升，必须预热后取稳定值。

可靠的对比方法是：同一硬件、同一 batch/context、同一 tokenizer、同一评测脚本，先跑 FP16 基线，再跑量化模型，记录多次运行的中位数和尾部延迟。质量上至少覆盖 perplexity、一个通用任务集和一个领域任务集；性能上分开报告 TTFT、TPOT 和显存峰值。

## 本章小结

- PTQ 用校准数据做逐层重建，把网络级量化拆成每层的最小二乘问题；RTN 是无校准的基线。
- GPTQ 用 Hessian 逆做逐列补偿，`H = 2XX^T`，group 128、act-order 是 3–4 bit 的关键参数；校准数据决定 `H`。
- AWQ 用 per-channel scale 保护 0.1%–1% 的显著权重，不需要反向传播，scale 可折进前一层。
- SmoothQuant 用 `s_j = max|X_j|^α / max|W_j|^(1-α)` 把激活离群迁到权重，`α = 0.5`，面向 W8A8。
- QAT 用 fake quant、STE 和 LSQ 学习舍入；只有在 PTQ 失败、位宽极低或领域差异大时才值得。
- 混合精度按敏感度分配位宽：首尾层和 embedding 保留 8-bit，中间层 4-bit，norm/softmax 保持高精度。
- GGUF 以 block 为单位量化，Q4_K 的 256 元素 super-block 把 8 个 32 元素 sub-block 的 6-bit scale/min 打包进 12 B，4.5 bit/权重。
- 量化 GEMM 的关键是寄存器内反量化、按组乘 scale、int4 打包和 dp4a/IMMA/HMMA 的选择。
- 评估要同时看 perplexity、任务准确率、TTFT/TPOT 和显存，避免 tokenizer、batch、上下文和校准集带来的误判。

## 延伸阅读

- Frantar et al., [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323), ICLR 2023.
- Lin et al., [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978), MLSys 2024.
- Xiao et al., [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438), ICML 2023.
- Dettmers et al., [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339), 2022.
- Esser et al., [Learned Step Size Quantization](https://arxiv.org/abs/1902.08295), ICLR 2020.
- ggml-org, [llama.cpp quantize tool documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md).
- ggml-org, [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
- NVIDIA, [CUTLASS: CUDA Templates for Linear Algebra Subroutines](https://github.com/NVIDIA/cutlass).
