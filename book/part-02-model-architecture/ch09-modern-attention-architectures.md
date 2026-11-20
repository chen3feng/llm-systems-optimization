---
chapter: 9
part: 2
title: Modern Attention Architectures
level: tbd
status: draft
prerequisites: [8]
lab:
---

# Chapter 9. Modern Attention Architectures

FlashAttention 这个名字容易让人以为它是一种近似算法。

它不是。它的输出和标准 attention 在数值上几乎一致，改变的是数据在 GPU 存储层次里的移动方式：把 `S × S` 的分数矩阵留在 shared memory 里，而不是写回 HBM 再读出来。

这一章讲两件事：FlashAttention 如何用分块和 online softmax 把 attention 变成 IO-aware 的算法，以及 MLA 如何用低秩表示把 KV Cache 再压缩一个量级。前者改变的是计算过程，后者改变的是数据结构。

## 9.1 FlashAttention Principles

标准 attention 的问题不在 FLOPs，在访存。

`S = 8192`、`n_h = 64`、batch 1、FP16 时，分数矩阵有：

$$
64 \times 8192 \times 8192 \times 2 \approx 8.6 \text{ GB}
$$

标准实现会把它写进 HBM，再读出来做 softmax，再写回概率，再读出来乘 V。一个 `S²` 的中间结果，至少产生四次 HBM 往返。H100 的 50 MB L2 装不下，只能反复走 3.35 TB/s 的 HBM。

FlashAttention 的观察是：GPU 的 shared memory 虽然只有 228 KB，但带宽比 HBM 高一个数量级。如果把 Q/K/V 切成能放进 shared memory 的块，在片上完成 attention，就能把 `S²` 的流量从 HBM 移到片上。

### 分块

设 Query 块大小为 `B_r`，Key/Value 块大小为 `B_c`。外层循环遍历 K/V 块，内层循环遍历 Q 块：

```text
for each K/V block j:
    load K_j, V_j into shared memory
    for each Q block i:
        load Q_i into shared memory
        compute S_ij = Q_i K_j^T / sqrt(d)
        update running softmax and output for block i
```

每个 `S_ij` 只在片上存在，从不写回 HBM。Q 块会被重复读取 `S / B_c` 次，但每次只是一小块。

### Online softmax

分块之后，softmax 不能直接算——它需要整行的最大值和归一化因子。FlashAttention 用的是 online softmax：维护三个运行量，逐块更新。

对每一行，维护：

- `m`：目前见过的最大值；
- `l`：目前见过的指数和；
- `o`：目前的未归一化输出。

处理一个新块时：

$$
\begin{aligned}
m_{\text{new}} &= \max(m, \max(s)) \\
l_{\text{new}} &= e^{m - m_{\text{new}}} l + \sum_j e^{s_j - m_{\text{new}}} \\
o_{\text{new}} &= e^{m - m_{\text{new}}} o + \sum_j e^{s_j - m_{\text{new}}} v_j
\end{aligned}
$$

最后 `o / l` 就是结果。这个递推是精确的：它只是把“先求全局最大值再归一化”换成了“边看边修正”。

### 伪代码

```python
def flash_attention(Q, K, V, Br, Bc):
    # Q: [S, d], K/V: [S, d]，单头
    O = zeros_like(Q)
    L = zeros(S)
    M = full(S, -inf)

    for j in range(0, S, Bc):
        Kj = K[j:j+Bc]                       # [Bc, d]
        Vj = V[j:j+Bc]
        for i in range(0, S, Br):
            Qi = Q[i:i+Br]                   # [Br, d]
            Sij = Qi @ Kj.T / sqrt(d)        # [Br, Bc]
            Sij = apply_mask(Sij, i, j)

            mi = rowmax(Sij)                 # [Br]
            m_new = maximum(M[i:i+Br], mi)
            alpha = exp(M[i:i+Br] - m_new)

            P = exp(Sij - m_new[:, None])    # [Br, Bc]
            L[i:i+Br] = L[i:i+Br] * alpha + rowsum(P)
            O[i:i+Br] = O[i:i+Br] * alpha[:, None] + P @ Vj
            M[i:i+Br] = m_new

    return O / L[:, None]
```

每个块只读写 `B_r × d` 和 `B_c × d` 的数据。分数矩阵 `S_ij` 是 `B_r × B_c`，留在寄存器或 shared memory 里。

### 反向传播

标准 attention 在反向时要用前向保存的 `S × S` 概率矩阵。FlashAttention 不保存，而是在反向时用 Q、K、V、O 和 `l` 重新计算：

1. 重算 `S` 和 `P`；
2. 用 `P` 计算对 Q/K/V 的梯度。

这多花约 30% 的前向 FLOPs，但省下了 `S²` 的显存。训练 128K 上下文的模型，这是能不能跑起来的分界线。

### 它没有改变复杂度

FlashAttention 的 FLOPs 仍然是 `O(S²d)`，HBM 流量是：

$$
\Theta\left(\frac{S^2 d^2}{M}\right)
$$

其中 `M` 是 shared memory 容量（以元素计）。相比标准实现的 `Θ(S²)`，它把常数降低了 `d²/M` 倍。`d = 128`、`M ≈ 100K` 个元素时，大约是 6–7 倍。

**它减少的是 HBM 访问，不是计算量。** 长上下文下 attention 仍然贵，只是不再被显存带宽卡死。

## 9.2 IO-aware Attention

“IO-aware” 的意思是：算法设计时把存储层次当成一等约束，而不是先写出数学上最优的算法，再指望编译器把数据搬好。

### 存储层次

| 层级 | 容量 | 带宽（量级） | 谁管理 |
| --- | --- | --- | --- |
| 寄存器 | 256 KB/SM | 最高 | 编译器 |
| Shared memory | 228 KB/SM | 约 20 TB/s | 程序员 |
| L2 | 50 MB | 约 5–10 TB/s | 硬件 |
| HBM | 80 GB | 3.35 TB/s | 硬件 |

从 HBM 到 shared memory 的带宽差了一个数量级，容量差了五个数量级。算法的性能取决于它把中间结果放在哪一层。

### 算术强度

标准 attention 如果物化 `S²`：

- FLOPs：`4 S² d`；
- 访存：写入 `S²`、读出做 softmax、写回、读出乘 V，至少 `4 S²` 个元素；
- 强度：约 `d/2` 到 `d`，即 64–128 FLOP/byte。

H100 的脊点是 295。这个区间低于脊点，所以标准 attention 在长序列上是带宽受限的。

FlashAttention 把 `S²` 的流量压到 `S²d/M`，强度提高到约 `d·M/d² = M/d`，也就是 800 左右，进入算力受限区。

### 分块大小的权衡

`B_r` 和 `B_c` 不能无限大，shared memory 放不下。以 `d = 128`、FP16 为例：

| 张量 | 128×128 tile |
| --- | --- |
| Q tile | 32 KB |
| K tile | 32 KB |
| V tile | 32 KB |
| S/P tile（FP32） | 64 KB |
| 合计 | 160 KB |

A100 每个 SM 有 164 KB shared memory，刚好放下；H100 有 228 KB，可以留出余量做双缓冲。

tile 越大，HBM 读取次数越少，但占用率越低、并行度越差。tile 越小，并行度越高，但重复读取 Q 的次数增加。实际 kernel 会针对 `d`、序列长度和 GPU 型号选一组配置，并在边界块上处理 mask。

### 因果 mask 的块跳过

causal attention 的上三角全是 `-inf`。如果 Q 块 `i` 和 K/V 块 `j` 满足 `j > i`，整块都不需要计算。

一个 `128×128` 的块，对角线以上的块直接跳过，对角线上的块单独处理 mask。这样有效计算量减半，而且跳过的是整块，没有分支发散。

### 和 roofline 的关系

FlashAttention 没有把 attention 变成“免费”。它做的是把 attention 从带宽受限推到算力受限，让 GPU 的 Tensor Core 真正忙起来。

在 A100 上，标准 PyTorch attention 大约只能跑到峰值的 20%–30%；FlashAttention-2 能到 50%–73%。剩下的差距来自 softmax 的逐元素运算、mask 和边界处理——这些不是矩阵乘法，Tensor Core 用不上。

## 9.3 FlashAttention Algorithm Evolution

三代 FlashAttention 的演进，本质上是不断把 GPU 的新硬件特性用满。

### FlashAttention-1

2022 年的第一版解决了三个问题：

- 分块，避免物化 `S²`；
- online softmax，保证分块后结果精确；
- 反向重算，避免保存 `S²`。

在 A100 上，它比当时的 PyTorch 实现快 2–4 倍，显存从 `O(S²)` 降到 `O(S)`。

第一版的并行方式是二维的：grid 覆盖 Q 块和 K/V 块两个维度，因此需要跨块归约部分结果（split-K）。这带来了额外的通信和 shared memory 开销。

### FlashAttention-2

2023 年的第二版主要改工作划分：

- **只沿 Q 块并行**，K/V 块在 kernel 内部循环，不需要跨块归约；
- **减少非矩阵乘 FLOPs**，softmax 的缩放次数降到最低；
- **改进反向并行**，按 K/V 块划分，减少原子操作和同步；
- **支持更大的 head dim 和 varlen**。

结果是在 A100 上比第一版快约 2 倍，FP16 达到峰值 50%–73%。

### FlashAttention-3

2024 年的第三版针对 Hopper：

- **TMA**：用 Tensor Memory Accelerator 异步搬运 Q/K/V 块，不占寄存器和指令发射槽；
- **WGMMA**：warpgroup 级的异步矩阵乘，`m64nNk16`，直接读 shared memory；
- **Warp specialization**：生产者 warp 负责 TMA，消费者 warp 负责 MMA 和 softmax；
- **Ping-pong 调度**：两组 warpgroup 交替执行，让 softmax 和 GEMM 重叠；
- **FP8 支持**：带 per-block scaling 的 FP8 attention；
- **head dim 256**：支持 Gemma 2 这类宽头模型。

在 H100 上，FP16 能达到约 75% 的峰值，比 FlashAttention-2 快 1.5–2 倍。FP8 版本还能再快一倍，但精度需要逐层验证。

Blackwell 上的实现继续沿着这个方向：更大的片上存储、双 SM 协作的 MMA、更激进的异步流水线。每一代的核心问题都一样——**让矩阵乘、softmax 和数据搬运尽可能重叠**。

### FlashDecoding

prefill 时 `S` 很大，Q 块很多，并行度足够。decode 时每个请求只有 1 个 Query，如果 batch 很小，grid 里只有几十个块，GPU 大部分 SM 空闲。

FlashDecoding 把 K/V 沿序列维度切块，每个块独立计算部分输出和部分 softmax 统计量，最后合并：

$$
O = \frac{\sum_k e^{m_k - M} l_k O_k}{\sum_k e^{m_k - M} l_k}
$$

这把 decode 的并行度从 `batch × n_h` 提高到 `batch × n_h × (S / chunk)`。128K 上下文、batch 1 时，这是能不能跑满 GPU 的关键。

### 其他实现

FlashAttention 不是唯一选择：

| 实现 | 特点 |
| --- | --- |
| FlashAttention（Dao-AILab） | 参考实现，CUDA，覆盖 FA1/2/3 |
| FlashInfer | 面向推理，支持 paged KV、GQA、RoPE 融合 |
| CUTLASS FMHA | NVIDIA 的模板库，可定制 |
| cuDNN attention | 闭源，集成在框架后端 |
| Triton attention | 易修改，性能接近手写 CUDA |
| xFormers memory-efficient | PyTorch 生态的老实现 |

PyTorch 的 `scaled_dot_product_attention` 会在这些后端之间自动 dispatch。选择哪个后端，取决于 dtype、head dim、mask 类型和 GPU 型号。

## 9.4 Multi-head Latent Attention

GQA 把 KV 头数从 64 降到 8，KV Cache 缩小 8 倍。还能再小吗？

MLA（Multi-head Latent Attention）的思路是：不存 K 和 V，存一个低秩的 latent 向量，用的时候再投影出来。

### 结构

对每个 token：

$$
c_t^{KV} = W^{DKV} h_t, \quad c_t^{KV} \in \mathbb{R}^{512}
$$

这个 512 维向量就是缓存的内容。K 和 V 由它上投影得到：

$$
k_t^C = W^{UK} c_t^{KV}, \quad v_t^C = W^{UV} c_t^{KV}
$$

Query 也做低秩压缩：

$$
c_t^Q = W^{DQ} h_t, \quad c_t^Q \in \mathbb{R}^{1536}
$$

### Decoupled RoPE

RoPE 是位置相关的，不能直接作用在压缩后的 latent 上——上投影之后，位置信息会被混进所有维度。

MLA 的解决办法是加一个独立的 RoPE 分支：

$$
k_t^R = \text{RoPE}(W^{KR} h_t), \quad k_t^R \in \mathbb{R}^{64}
$$

注意力分数由两部分组成：

$$
\text{score} = q_t^C \cdot k_t^C + q_t^R \cdot k_t^R
$$

内容部分走 latent，位置部分走单独的 64 维 key。这个 key 在所有头之间共享。

### 缓存大小

每个 token、每层缓存：

$$
512 + 64 = 576 \text{ 个元素}
$$

对比同样 128 个头的模型：

| 方案 | 每 token 每层元素数 | 相对 MHA |
| --- | --- | --- |
| MHA（128 头） | 32,768 | 1× |
| GQA（8 组） | 2,048 | 1/16 |
| MLA | 576 | 1/57 |

DeepSeek-V3 有 61 层，FP16 下每 token 的 KV Cache：

$$
61 \times 576 \times 2 \approx 70 \text{ KB}
$$

128K 上下文、单请求约 9.2 GB。同配置的 GQA（8 组）约 33 GB，MHA 约 524 GB。**MLA 让 128K 上下文的 KV Cache 从“一台机器”变成“一张卡的一部分”。**

### 矩阵吸收

MLA 的代价是推理时多了一次上投影。如果每步都把 512 维 latent 投影成 128 个头的 K/V，计算量不小。

实际实现用矩阵吸收（matrix absorption）把它折叠进已有的投影：

$$
q^C (k^C)^T = (W^{UQ} c^Q)(W^{UK} c^{KV})^T
= c^Q (W^{UQ\,T} W^{UK}) (c^{KV})^T
$$

`W^{UQ\,T} W^{UK}` 可以预计算成一个 `[1536, 512]` 的矩阵，推理时直接用它把 `c^Q` 映射到 key 空间。V 的投影也可以折进输出投影 `W^O`。

这样 decode 时不需要真的构造每头的 K/V，只做 latent 上的矩阵乘。代价是：

- 预计算的矩阵更大，参数略增；
- 数值顺序改变，和训练时的计算不完全一致；
- prefill 和 decode 可能走不同的代码路径。

### 为什么有效

MLA 本质上是把“缓存 K/V”换成了“缓存压缩表示 + 运行时重建”。

- 显存和带宽：缓存从 `2 n_h d` 降到 `d_c + d_r`；
- 计算：增加上投影或吸收矩阵的乘法；
- 质量：低秩压缩会损失信息，但 DeepSeek 的实验显示，在同等缓存预算下 MLA 优于 GQA。

在 decode 阶段，带宽是瓶颈，用计算换带宽是划算的。这正是 MLA 的立足点。

## 9.5 KV Compression and Latent Representation

MLA 是 KV 压缩的一种。把这一类方法放在一起看，能看清各自的取舍。

| 方法 | 缓存什么 | 压缩比 | 额外开销 | 代表 |
| --- | --- | --- | --- | --- |
| MQA | 1 组 K/V | `n_h` | 质量损失 | PaLM |
| GQA | `G` 组 K/V | `n_h/G` | 质量损失小 | Llama 3 |
| MLA | latent + RoPE key | 约 57（vs MHA） | 上投影/吸收 | DeepSeek-V3 |
| KV 量化 | FP8/INT8/INT4 的 K/V | 2–8 | 反量化、缩放 | KIVI |
| Token 淘汰 | 重要 token 的 K/V | 可变 | 评分、索引 | H2O |
| 跨层共享 | 一层 KV 给多层用 | `L` | 结构改动 | YOCO |

### 量化

KV Cache 量化的难点在 K 的分布。K 的某些通道有巨大的离群值，直接 per-tensor 量化会把其他通道压成 0。

常见做法：

- K 按通道（per-channel）量化，V 按 token（per-token）量化；
- 保留离群通道为高精度；
- FP8 E4M3 代替 INT8，动态范围更大。

KIVI 用 2-bit 量化 K/V，配合分组和残差，能把缓存压到 1/4 以下。代价是 kernel 复杂度和精度风险。

### Token 淘汰

不是所有 token 都同等重要。H2O 用累计注意力分数识别 heavy hitters，只保留分数高的 token 和最近的窗口。

风险是被淘汰的 token 后面可能又需要。淘汰策略通常在长对话、流式生成里有效，在需要精确召回的任务里要谨慎。

### 跨层共享

YOCO（You Only Cache Once）让所有层共享一份 KV，只缓存一层。这需要改模型结构（cross-layer attention），不是纯推理优化，但能把缓存再降 `L` 倍。

### 一个判断框架

选择压缩方法时，问三个问题：

1. **压缩的是显存还是带宽？** decode 阶段两者都重要，但带宽通常先成为瓶颈。
2. **额外计算能不能被隐藏？** 上投影、反量化、评分都要时间；如果 GPU 已经算力受限，压缩未必划算。
3. **质量损失能不能接受？** 量化和淘汰有精度风险，GQA/MLA 是结构级的，风险更可控。

## 9.6 Attention Variants and Hardware Trade-offs

把 MHA、GQA、MQA、MLA 放在同一张表里，以 64 个 Query 头、`d = 128`、FP16 为例：

| 变体 | 每 token 每层 KV | 相对 MHA | Decode 强度 | Kernel 复杂度 |
| --- | --- | --- | --- | --- |
| MHA | 32 KB | 1× | 1 | 低 |
| GQA 8:1 | 4 KB | 1/8 | 8 | 中 |
| MQA | 0.5 KB | 1/64 | 64 | 低 |
| MLA | 1.15 KB | 约 1/28 | 约 36 | 高 |

### 硬件约束

**head dim**。Tensor Core 的 MMA 形状偏好 64 或 128。`d = 64` 在 Hopper 的 WGMMA 上需要 padding，效率下降；`d = 256`（Gemma 2）需要更大的 tile，shared memory 压力大，FlashAttention-3 才原生支持。

**shared memory**。`d` 越大，一个 tile 占的空间越多，能放的块越少，占用率越低。这是 head dim 不能无限增大的工程原因。

**FP8**。attention 的 QK 可以用 FP8，但 softmax 必须在 FP32 里做；P 转回 FP8 做 PV 时，需要 per-block scaling。FP8 能把吞吐翻倍，但数值范围窄，长上下文下误差会累积。

**KV 布局**。GQA/MLA 的 K/V 不是连续的 `[B, S, H]`，kernel 要按组索引。PagedAttention 又把它切成不连续的块，attention kernel 必须支持 block table 间接寻址。FlashInfer 这类库的一大价值就是把这些布局组合都实现好。

### 质量与成本的取舍

| 选择 | 收益 | 代价 |
| --- | --- | --- |
| MHA → GQA | KV 降 8×，质量几乎不变 | 需要 uptrain 或从头训练 |
| GQA → MQA | KV 再降 8× | 质量下降，训练不稳定 |
| GQA → MLA | KV 降 3.5×，质量更好 | 实现复杂，需要改结构 |
| FP16 → FP8 KV | 带宽减半 | 精度风险，需要校准 |
| 全量 → 滑动窗口 | 长上下文计算线性化 | 远距离信息丢失 |

没有普遍最优的方案。模型结构、上下文长度、并发量、SLA 共同决定选择。

### 一个经验规则

- **短上下文、高质量要求**：GQA，成熟、kernel 支持好；
- **超长上下文、显存敏感**：MLA 或 MLA + FP8 KV；
- **极致吞吐、质量可妥协**：MQA + 量化；
- **长上下文但只需局部信息**：滑动窗口 + 全局 token。

这些规则会随硬件变化。带宽增长慢于算力，压缩 KV 的收益会继续变大；但低精度的数值问题也会更突出。

## 本章小结

- 标准 attention 的瓶颈是 `S²` 矩阵的 HBM 往返；FlashAttention 用分块把中间结果留在 shared memory。
- Online softmax 通过维护运行最大值、指数和和未归一化输出，让分块计算保持精确。
- FlashAttention 不改变 FLOPs 和 `O(S²)` 的渐近复杂度，减少的是 HBM 访问；显存从 `O(S²)` 降到 `O(S)`。
- FlashAttention-2 改进了工作划分，减少非矩阵乘开销；FlashAttention-3 利用 TMA、WGMMA、warp specialization 和 FP8，在 H100 上达到约 75% 峰值。
- FlashDecoding 沿 KV 维度切分，解决小 batch 长上下文 decode 的并行度问题。
- MLA 用低秩 latent 加解耦 RoPE 缓存 KV，把每 token 每层从 2,048 个元素降到 576 个。
- 矩阵吸收把上投影折进 Q/O 投影，用计算换带宽，适合 decode 场景。
- KV 压缩还有量化、token 淘汰、跨层共享等路线；选择取决于显存、带宽、算力和质量约束。

## 延伸阅读

- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), 2022.
- Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691), 2023.
- Shah et al., [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision](https://arxiv.org/abs/2407.08608), 2024.
- Dao et al., [FlashDecoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html), 2023.
- DeepSeek-AI, [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434), 2024.
- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), 2024.
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), 2023.
- Zhang et al., [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048), 2023.
- Liu et al., [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750), 2024.
- Sun et al., [You Only Cache Once: Decoder-Decoder Architectures for Language Models](https://arxiv.org/abs/2405.05254), 2024.
- Dao-AILab, [FlashAttention repository](https://github.com/Dao-AILab/flash-attention).
- FlashInfer, [FlashInfer: Kernel Library for LLM Serving](https://github.com/flashinfer-ai/flashinfer).
