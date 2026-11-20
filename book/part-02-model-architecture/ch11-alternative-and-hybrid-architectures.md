---
chapter: 11
part: 2
title: Alternative and Hybrid Architectures
level: tbd
status: draft
prerequisites: [5, 8]
lab:
---

# Chapter 11. Alternative and Hybrid Architectures

Transformer 的每一层都要看前面所有 token。序列翻倍，计算翻四倍，KV Cache 翻一倍。

如果换一种结构，每层只维护一个固定大小的状态，每来一个 token 更新一次，会怎样？计算变成线性，显存不再随上下文增长，推理时也不需要 KV Cache。

这就是状态空间模型（State Space Model，SSM）和线性注意力的出发点。它们不是 Transformer 的微调，而是另一条技术路线。这一章讲它们怎么工作、强在哪、弱在哪，以及为什么现在的主流是大模型里的混合结构。

## 11.1 State Space Models

SSM 来自控制论。连续时间下，一个线性时不变系统是：

$$
\begin{aligned}
h'(t) &= A h(t) + B x(t) \\
y(t) &= C h(t) + D x(t)
\end{aligned}
$$

`h(t)` 是状态，`x(t)` 是输入，`y(t)` 是输出。`A`、`B`、`C`、`D` 是系统矩阵。

### 离散化

计算机处理的是离散序列。用步长 `Δ` 做零阶保持（ZOH）离散化：

$$
\bar{A} = e^{\Delta A}, \quad
\bar{B} = (\Delta A)^{-1}(e^{\Delta A} - I) \Delta B
$$

得到递推：

$$
h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t + D x_t
$$

每一步只做一次矩阵乘加。状态大小固定，与序列长度无关。

### 两种计算视角

同一个递推有两种算法：

- **递归**：逐 token 更新，`O(1)` 时间、`O(N)` 状态，适合推理；
- **卷积**：把递推展开成卷积核 `K = (C\bar{B}, C\bar{A}\bar{B}, C\bar{A}^2\bar{B}, \ldots)`，用 FFT 做 `O(S log S)`，适合训练。

训练时并行、推理时递归，这是 SSM 相比 RNN 的关键优势。普通 RNN 的训练必须串行，梯度还会沿时间步衰减。

### S4

S4（Structured State Space）用 HiPPO 理论初始化 `A`，让状态能记住长窗口的历史。`A` 被约束成对角加低秩（DPLR）结构，使卷积核可以高效计算。

S4 在 Long Range Arena 上把 16K 序列的平均准确率从 Transformer 的约 50% 提到 86% 附近，训练和推理都是线性的。

但 S4 是线性时不变的：`A`、`B`、`C` 不随输入变化。它无法根据内容决定“记住什么、忘记什么”。

## 11.2 Selective State Space Models

Mamba 的关键改动是让参数依赖输入：

$$
\Delta_t = \text{softplus}(W_\Delta x_t), \quad
B_t = W_B x_t, \quad
C_t = W_C x_t
$$

`Δ` 像一个门：大的 `Δ` 让状态快速更新，小的 `Δ` 让状态保持不变。模型可以根据当前 token 决定记住还是忽略。

### 代价

输入依赖破坏了线性时不变性。卷积核不再固定，不能用 FFT。

Mamba 用并行扫描（parallel scan）计算递推。递推 `h_t = a_t h_{t-1} + b_t` 满足结合律，可以像前缀和一样并行：

$$
(a_2, b_2) \circ (a_1, b_1) = (a_2 a_1, a_2 b_1 + b_2)
$$

在 GPU 上，扫描的深度是 `O(log S)`，总工作量是 `O(S)`。Mamba 的 kernel 把状态放在 shared memory 里，避免每一步都写回 HBM。

### 选择性为什么重要

线性时不变的 SSM 对所有 token 一视同仁，无法做内容相关的过滤。选择性让模型可以：

- 在遇到分隔符时重置状态；
- 忽略无关的填充 token；
- 在关键信息出现时保持状态。

这篇论文的消融实验显示，去掉选择性后，语言建模的质量明显下降。

## 11.3 Mamba Architecture

Mamba 把 SSM 放进一个类似 Transformer block 的结构：

```text
x -> Linear -> Conv1d -> SiLU -> SSM -> * SiLU(gate) -> Linear -> out
```

具体来说：

1. 输入经过线性投影，维度扩展到 `2 × d_inner`；
2. 分成两路：一路进 conv1d 和 SSM，一路做门控；
3. SSM 输出与门控分支逐元素相乘；
4. 再投影回 `d_model`。

没有 attention，没有 KV Cache。每个 token 只和固定大小的状态交互。

### 规模

Mamba 论文训练了 130M 到 2.8B 的模型。在困惑度和下游任务上，同规模的 Mamba 与 Transformer 相当或更好，推理吞吐更高，长序列的显存优势明显。

### Mamba-2

Mamba-2 用状态空间对偶（State Space Duality，SSD）把 SSM 写成矩阵形式，发现它和带 mask 的线性注意力是同一类结构。这样一来：

- 计算可以用矩阵乘，走 Tensor Core；
- 状态可以更大；
- 训练和推理的 kernel 更简单。

SSD 的形式下，`A` 被约束成标量乘单位矩阵，递推变成可并行的矩阵运算。Mamba-2 比 Mamba-1 快 2–8 倍，质量相当或更好。

## 11.4 Linear Attention

线性注意力是另一条路径。标准 attention 是：

$$
y_t = \frac{\sum_{s \le t} \exp(q_t \cdot k_s) v_s}{\sum_{s \le t} \exp(q_t \cdot k_s)}
$$

把 `exp(q·k)` 换成特征映射的内积 `φ(q) · φ(k)`：

$$
y_t = \frac{\phi(q_t)^T \sum_{s \le t} \phi(k_s) v_s^T}{\phi(q_t)^T \sum_{s \le t} \phi(k_s)}
$$

分子里的求和可以增量维护：

$$
S_t = S_{t-1} + \phi(k_t) v_t^T, \quad y_t = \phi(q_t)^T S_t
$$

`S_t` 是一个 `d × d` 的状态矩阵。每步更新是 `O(d²)`，与序列长度无关。

### 与 SSM 的关系

线性注意力就是一个输入依赖的线性递推，状态是 `d × d` 矩阵。SSM 的状态通常是 `N × d`，`N` 是状态维度。两者在数学上高度重叠，Mamba-2 的 SSD 就是把它们统一起来的尝试。

### 变体

| 方法 | 特征映射/衰减 | 特点 |
| --- | --- | --- |
| Linear Transformer | `φ(x) = elu(x) + 1` | 最简单 |
| Performer | 随机特征近似 softmax | 有理论保证，方差大 |
| RetNet | 固定衰减 `γ` | 支持并行、递归、分块三种形式 |
| GLA | 门控衰减 | 输入依赖的遗忘 |
| DeltaNet | delta 规则更新 | 能修改已写入的状态 |

### 弱点

固定大小的状态装不下所有历史。需要精确回忆某个位置的细节时，线性注意力会丢信息。

这在 associative recall 任务上表现得很明显：给一串键值对，问其中一个键对应的值。Transformer 可以直接查到，线性注意力的状态是压缩的，容易混淆。

## 11.5 Hybrid Transformer-SSM Models

纯 SSM 在长程检索上不如 Transformer，纯 Transformer 在长上下文上太贵。混合结构试图两者兼得。

### 常见组合

| 模型 | 结构 | 规模 |
| --- | --- | --- |
| Jamba | 1 层 attention + 7 层 Mamba，交替 MoE | 52B 总 / 12B 激活，256K 上下文 |
| Griffin | 局部 attention + RG-LRU 循环 | 14B |
| Zamba | 共享的全局 attention + Mamba | 7B |
| Samba | 滑动窗口 attention + Mamba | 3.8B |
| Nemotron-H | attention 与 Mamba 混合 | 8B / 47B |

Jamba 的比例是 1:7：每 8 层里 1 层 attention、7 层 Mamba。attention 层负责精确检索，Mamba 层负责长程的高效传播。

### 为什么有效

- attention 提供“按内容精确查找”的能力；
- SSM 提供线性的长程记忆；
- 大部分层用 SSM，长上下文的成本大幅下降；
- 少量 attention 层保留关键能力，KV Cache 只在这些层增长。

Jamba 在 256K 上下文上的吞吐明显高于同规模 Transformer，质量接近。Griffin 的论文报告，14B 模型的质量与 Llama-2 13B 相当，推理吞吐高数倍。

### 设计变量

混合结构的设计空间很大：

- attention 和 SSM 的比例；
- attention 是全局还是滑动窗口；
- SSM 用 Mamba-1 还是 Mamba-2；
- 是否叠加 MoE；
- 状态维度多大。

这些选择目前没有统一结论，更多是经验。

## 11.6 Recurrent and State-based Inference

SSM 的推理和 Transformer 有本质区别。

### 没有 KV Cache

Transformer 每生成一个 token，都要读一遍历史 K/V。SSM 只维护一个固定状态：

```text
state = init
for token in sequence:
    state = A_bar * state + B_bar * token
    out = C * state
```

状态大小与序列长度无关。128K 上下文和 8K 上下文，显存占用一样。

### 状态大小

以 Mamba-2.8B 为例：64 层、`d_model = 2560`、扩展因子 2、状态维度 16。每层状态约 `16 × 5120 = 81,920` 个元素，FP32 下 320 KB；64 层约 20 MB/请求。

Transformer 2.8B 在 8K 上下文下的 KV Cache 约 2–3 GB。差两个数量级。

### Prefill 与 Decode

- **Prefill**：并行扫描，`O(S)` 工作量、`O(log S)` 深度；大 batch 时能跑满 GPU。
- **Decode**：每步一次递推，读状态和权重；和 Transformer 一样受带宽限制，但读的东西少得多。

### 批处理

每个请求有自己的状态，批处理就是把状态按 batch 维堆起来。不需要 paged KV，不需要块表，调度简单得多。

代价是状态是固定容量的。上下文超过状态能表示的范围，信息就会丢。这不是“缓存不够”，是结构性的遗忘。

## 11.7 Architecture Selection and Hardware Implications

| 维度 | Transformer | MoE | SSM / 线性注意力 | 混合 |
| --- | --- | --- | --- | --- |
| 训练并行 | 高 | 高 | 扫描，较高 | 高 |
| Prefill | `O(S²)` | `O(S²)` | `O(S)` | 混合 |
| Decode 显存 | KV 随 `S` 增长 | KV 随 `S` 增长 | 固定状态 | 少量层增长 |
| 长上下文 | 贵 | 贵 | 便宜 | 较便宜 |
| 精确检索 | 强 | 强 | 弱 | 较强 |
| Kernel 成熟度 | 高 | 中 | 中 | 低 |
| 生态支持 | 最广 | 广 | 较少 | 少 |

### 硬件层面

- **Tensor Core 利用率**：SSM 的逐元素递推不容易用上矩阵单元。Mamba-2 的 SSD 形式把它变成矩阵乘，才真正吃满 Tensor Core。
- **内存访问**：SSM 的瓶颈通常在状态读写和 scan 的中间结果；kernel 要把状态留在 shared memory。
- **批处理形状**：Transformer 的 KV 长度可变，SSM 的状态形状固定，后者对 kernel 更友好。
- **长上下文**：SSM 的优势在显存和计算，但 prefill 的并行 scan 效率取决于实现。

### 选择建议

- **通用对话、工具调用、代码**：Transformer + GQA，生态最成熟，检索能力最强。
- **超长文档、流式输入**：混合结构或 SSM，显存和吞吐优势明显。
- **需要精确召回**：不要用纯 SSM；加 attention 层或改用 RAG。
- **边缘设备**：SSM 的固定状态和小显存有吸引力，但要确认 kernel 支持。

架构选择不是“谁取代谁”。Transformer 在检索和生态上有巨大优势，SSM 在长上下文和固定显存上有结构性优势。混合结构是目前最现实的折中，代价是复杂度。

## 本章小结

- SSM 用固定大小的状态做线性递推，训练可以并行扫描，推理是 `O(1)` 每 token。
- S4 用结构化矩阵和卷积视角解决了长序列训练；Mamba 加入输入依赖的选择性，提升了语言建模质量。
- Mamba-2 的 SSD 把 SSM 写成矩阵形式，能用 Tensor Core，速度大幅提升。
- 线性注意力用特征映射替代 softmax，状态是 `d × d` 矩阵；和 SSM 在数学上高度重叠。
- 纯 SSM/线性注意力的弱点是精确检索；固定状态装不下所有历史。
- 混合结构（Jamba、Griffin、Zamba、Samba）用少量 attention 层补足检索，大部分层用 SSM 降低长上下文成本。
- SSM 推理不需要 KV Cache，显存与序列长度无关，但状态容量是硬上限。
- 架构选择取决于任务、上下文长度、硬件和生态，目前 Transformer 仍是默认，混合结构是长上下文场景的有力候选。

## 延伸阅读

- Gu et al., [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396), 2021.
- Gu & Dao, [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752), 2023.
- Dao & Gu, [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060), 2024.
- Katharopoulos et al., [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236), 2020.
- Choromanski et al., [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794), 2020.
- Sun et al., [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621), 2023.
- Peng et al., [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048), 2023.
- AI21 Labs, [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887), 2024.
- De et al., [Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models](https://arxiv.org/abs/2402.19427), 2024.
- Glorioso et al., [Zamba: A Compact 7B SSM Hybrid Model](https://arxiv.org/abs/2405.16712), 2024.
- Ren et al., [Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling](https://arxiv.org/abs/2406.07522), 2024.
