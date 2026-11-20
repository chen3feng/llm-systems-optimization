---
chapter: 8
part: 2
title: Attention Mechanisms
level: tbd
status: draft
prerequisites: [2, 7]
lab:
---

# Chapter 8. Attention Mechanisms

Attention 的核心公式只有一行：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

但把它变成能处理 batch、变长序列、KV Cache 和 128K 上下文的代码，会多出十几个分支：哪些位置要被 mask、K/V 有几个头、窗口多大、哪些块可以跳过、mask 用什么值表示。

每一个分支都同时改变两件事：**结果对不对**，和**要花多少计算与带宽**。这一章把这些分支讲清楚，并给出成本公式。FlashAttention 的实现细节留到下一章，这里只关心 attention 在算什么、为什么这样算。

## 8.1 Scaled Dot-Product Attention

先把形状定下来。设：

- `B`：batch size；
- `S`：Query 的序列长度；
- `S_kv`：Key/Value 的序列长度，自注意力里等于 `S`；
- `n_h`：Query 头数；
- `d`：每个头的维度。

单个头的计算是三步：

1. `QK^T`：`[S, d] × [d, S_kv] → [S, S_kv]`，得到注意力分数；
2. 除以 `√d`，加 mask，按行做 softmax；
3. 乘 `V`：`[S, S_kv] × [S_kv, d] → [S, d]`。

FLOPs 是 `4 S S_kv d`，其中 QK 和 AV 各 `2 S S_kv d`。causal mask 让有效区域减半，近似为 `2 S² d`。

### 为什么要除以 √d

假设 `q` 和 `k` 的每个分量独立、均值 0、方差 1。点积是 `d` 个乘积之和，方差为 `d`，标准差是 `√d`。

`d = 128` 时，未缩放分数的标准差约 11.3。softmax 的输入差十几个数量级，输出会变成接近 one-hot 的分布：最大值接近 1，其余接近 0。反向传播时，softmax 的梯度是 `p(1-p)`，大部分位置接近 0，梯度消失。

除以 `√d` 把标准差拉回 1，softmax 工作在梯度良好的区间。这不是启发式，是方差分析的直接结果。

### 数值稳定

softmax 的实现要先减去每行的最大值：

$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j x_j
$$

不减最大值，`exp(100)` 在 FP16 里直接溢出成 inf。减去之后，最大项是 `exp(0) = 1`，分母至少为 1，不会除零。

如果一行全部被 mask 成 `-inf`，`m` 也是 `-inf`，`x_i - m` 变成 `nan`。实现里要保证每行至少有一个有效位置，或者在检测到全 mask 时输出 0。

### 一个直接的实现

```python
def attention(q, k, v, mask=None):
    # q: [B, n_h, S, d]
    # k: [B, n_h, S_kv, d]
    # v: [B, n_h, S_kv, d]
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, n_h, S, S_kv]
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)
```

这段代码会物化完整的 `[B, n_h, S, S_kv]` 矩阵。`S = 8192`、`n_h = 64`、FP16 时，这个矩阵有 43 亿个元素，8.6 GB。这正是 FlashAttention 要解决的问题：不物化，分块计算。但无论怎么分块，数学公式和 FLOPs 不变。

## 8.2 Multi-Head Attention

一个头只能学一种“关注方式”。多头让模型在多个子空间里并行地做这件事。

### 结构

把 `H` 维的输入投影成 `n_h` 组 Q/K/V，每组维度 `d = H / n_h`：

$$
Q_i = XW_Q^i,\quad K_i = XW_K^i,\quad V_i = XW_V^i, \quad i = 1 \ldots n_h
$$

每个头独立做 attention，输出拼接后再投影：

$$
\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_{n_h}) W_O
$$

参数量是 `4H²`：Q、K、V、O 各一个 `[H, H]` 矩阵。

### 张量形状

| 张量 | 形状 |
| --- | --- |
| 输入 `X` | `[B, S, H]` |
| Q/K/V 投影后 | `[B, S, H]` 各一个 |
| reshape 成头 | `[B, S, n_h, d]` |
| transpose 后 | `[B, n_h, S, d]` |
| 分数 | `[B, n_h, S, S_kv]` |
| 每头输出 | `[B, n_h, S, d]` |
| 合并 | `[B, S, H]` |
| 输出投影 | `[B, S, H]` |

实现上通常把三个投影拼成一个 `[H, 3H]` 的 GEMM，再一次 reshape 成 Q/K/V。这样只需一次 kernel launch，也只需读一遍输入。

### 头在学什么

多头不是为了增加参数量——同样参数量下，一个大头和多个小头的 FLOPs 差不多。它的作用是让不同的头关注不同的关系。

可解释性研究发现了若干专门化的头：有的头关注上一个 token（previous-token head），有的头做归纳复制（induction head），有的头盯着句法边界。这些发现不代表每个头都有明确分工，但说明多头确实提供了多种归纳偏置。

### head_dim 的选择

`d` 通常是 64 或 128。`d` 太小，每个头的表达能力不足，头数多但每个头学不到复杂模式；`d` 太大，头数变少，并行度和多样性下降，attention 的 `S²d` 计算也更贵。

`d = 128` 还有一个工程上的好处：它是 16 的倍数，适合 Tensor Core 的 `m16n8k16` 形状；shared memory 的 tile 也容易对齐。

## 8.3 Multi-Query Attention

MHA 的 KV Cache 太大。每个头都要存自己的 K 和 V，`n_h` 个头就是 `n_h` 份。

MQA（Multi-Query Attention）的做法是：所有 Query 头共享一组 K/V。

$$
Q_i = XW_Q^i, \quad K = XW_K, \quad V = XW_V
$$

K/V 的形状从 `[B, S, n_h, d]` 变成 `[B, S, 1, d]`。KV Cache 直接缩小 `n_h` 倍。

### 收益

以 70B、80 层、`d = 128`、8K 上下文为例：

| 方案 | 每 token KV | 8K 上下文 |
| --- | --- | --- |
| MHA（64 个 KV 头） | 2.62 MB | 20 GiB |
| MQA（1 个 KV 头） | 41 KB | 0.31 GiB |

差 64 倍。这个差距直接决定了单卡能放多少并发请求。

decode 时每个 step 都要把 KV Cache 读一遍。MQA 把读取量降到 1/64，TPOT 的带宽部分也随之下降。

### 代价

MQA 的表达能力比 MHA 弱。K/V 只有一组，所有头看到的是同一个键值空间，头之间的多样性受损。训练也可能不稳定，需要更仔细的初始化和 warmup。

PaLM、Falcon、StarCoder 用过 MQA。它的质量损失在小模型上可以接受，在大模型上更明显。

### 算术强度的变化

decode 时，单层每 token 的 attention FLOPs 是 `4 n_h S d`，KV 读取是 `4 n_kv S d` 字节。强度是：

$$
I = \frac{n_h}{n_kv}
$$

MHA 是 1，MQA 是 `n_h`（例如 64），GQA 介于两者之间。这就是 MQA 在 decode 上更快的原因：它让 attention 从极度带宽受限，变成中等强度。

## 8.4 Grouped-Query Attention

GQA 是 MHA 和 MQA 的折中：把 Query 头分成 `G` 组，每组共享一组 K/V。

$$
n_{kv} = G, \quad g = \frac{n_h}{G}
$$

每个 KV 头服务 `g` 个 Query 头。`g = 1` 退化成 MHA，`g = n_h` 退化成 MQA。

### 主流配置

| 模型 | `n_h` | `n_kv` | `g` |
| --- | --- | --- | --- |
| Llama 3 8B | 32 | 8 | 4 |
| Llama 3 70B | 64 | 8 | 8 |
| Qwen2.5 7B | 28 | 4 | 7 |
| Mistral 7B | 32 | 8 | 4 |

Llama 3 70B 的 8:1 是一个常见的平衡点：KV Cache 是 MHA 的 1/8，质量下降很小。

### 从头训练与 uptraining

GQA 可以直接从头训练，也可以从已有的 MHA checkpoint 转换：

1. 把同一组内的 `g` 个 K 头取平均，得到一个 K 头；
2. V 头同样处理；
3. 微调少量 step 恢复质量。

这比从头训练便宜得多。Ainslie 等人的实验显示，uptrain 后的 GQA 在大部分任务上接近 MHA，推理成本却接近 MQA。

### 头映射

Query 头 `h` 使用哪个 KV 头？标准做法是整除：

$$
\text{kv\_head}(h) = \left\lfloor \frac{h}{g} \right\rfloor
$$

Hugging Face 的 `repeat_kv` 就是这么实现的：

```python
def repeat_kv(x, n_rep):
    # x: [B, n_kv, S, d]
    b, n_kv, s, d = x.shape
    x = x[:, :, None, :, :].expand(b, n_kv, n_rep, s, d)
    return x.reshape(b, n_kv * n_rep, s, d)
```

`expand` 不复制内存，`reshape` 之后才真正展开。如果顺序写错——比如用交错映射而不是分块映射——模型不会报错，只是输出变差。移植时一定要和参考实现对比 logits。

### 显存与并发的算例

70B、FP16、8K 上下文：

- MHA 每请求 20 GiB，8 卡几乎放不下几个请求；
- GQA（8 组）每请求 2.5 GiB，8×H100 可支持约 120–180 个并发；
- MQA 每请求 0.31 GiB，可以支持上千并发，但质量损失更大。

GQA 之所以成为默认选择，是因为它把 KV Cache 压到了可运营的范围，同时没有明显牺牲质量。

## 8.5 Causal Attention and Masking

自回归模型不能看未来，所以 attention 要加 causal mask。

### Causal mask

位置 `i` 只能看到 `j ≤ i`：

$$
M_{ij} = \begin{cases}
0 & j \le i \\
-\infty & j > i
\end{cases}
$$

实现上是一个下三角矩阵。加上之后，有效计算量减半：

```python
mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))
scores = scores.masked_fill(~mask, float("-inf"))
```

### Padding mask

batch 内序列长度不同，短序列要补齐。补齐的 token 不应该参与 attention：

```python
# attention_mask: [B, S_kv]，1 表示有效
scores = scores.masked_fill(attention_mask[:, None, None, :] == 0, float("-inf"))
```

这里有一个容易混淆的地方：不同框架对 mask 的语义相反。

| 框架/接口 | 语义 |
| --- | --- |
| Hugging Face `attention_mask` | 1 = 保留，0 = mask |
| PyTorch `masked_fill` | 传入的条件为 True 时填 `-inf` |
| FlashAttention `attn_mask` | 加性 mask，`0` 保留，`-inf` 屏蔽 |
| FlashAttention `causal=True` | 内部生成下三角 mask |

混用会得到完全相反的结果，而且不会报错。

### 左 padding 与右 padding

生成任务通常用左 padding，让最后一个位置对齐；训练通常用右 padding，让 loss 对齐。

左 padding 时，`position_ids` 必须从第一个有效 token 开始算，否则 RoPE 的位置会整体偏移。右 padding 时，padding 位置仍然会产出 logits，计算 loss 时要 mask 掉。

### Packed sequences

把多条短序列拼成一条长序列，可以提高 GPU 利用率。但 attention 不能跨文档：

```text
[doc A ...][doc B ...][doc C ...]
```

需要一个块对角 mask：A 的 token 只能看 A，B 只能看 B。实现上有两种：

- 构造一个 `[S, S]` 的布尔 mask，内存是 `S²` bits；
- 用 `cu_seqlens` 描述每段的起止，kernel 内部按段处理，不构造完整 mask。

`S = 128K` 时，完整 mask 是 `1.7e10` bits，约 2.1 GB。这就是变长 kernel 存在的原因。

### Mask 的常见 bug

- **softmax 之后再 mask**：概率已经归一化，再置零会导致行和不为 1；
- **用 0 而不是 `-inf`**：被 mask 的位置仍然有概率；
- **FP16 里用 `-65504` 代替 `-inf`**：一般能工作，但减去最大值后可能变成 `0`，需要确认；
- **整行被 mask**：`max` 是 `-inf`，产生 NaN；
- **mask 的 dtype 不匹配**：bool 和 float mask 混用，广播出错误的形状。

## 8.6 Sliding Window Attention

长序列里，每个 token 真的需要看前面所有 token 吗？

Sliding window attention 假设不需要：每个 token 只看最近的 `W` 个 token。

$$
M_{ij} = \begin{cases}
0 & i - W < j \le i \\
-\infty & \text{otherwise}
\end{cases}
$$

复杂度从 `O(S²)` 降到 `O(S·W)`。`W = 4096`、`S = 128K` 时，计算量降到 1/32。

### 感受野

单层只看 `W` 个 token，但多层叠加后，信息可以传得更远。每层把感受野扩大 `W - 1`，`L` 层之后：

$$
\text{receptive field} = 1 + L(W - 1)
$$

Mistral 7B 有 32 层、`W = 4096`，理论感受野约 131K。但这只是理论上界：信息要经过几十次 attention 混合才能传到，实际有效范围远小于这个数。

### KV Cache

滑动窗口的 KV Cache 只需要保留最近 `W` 个 token。实现上可以用环形缓冲区：

```text
slot = position % W
cache[slot] = k, v
```

写入新 token 时覆盖最老的槽位。attention 时按逻辑位置还原顺序，或者用位置取模处理 mask。

要注意：并不是所有层都能用滑动窗口。如果模型是局部/全局交替（Gemma 2），全局层仍然需要完整 KV Cache，只有局部层能省。

### 全局 token

纯滑动窗口会丢失远距离信息。Longformer 的做法是保留少量全局 token（如 `[CLS]`），它们可以 attend 所有位置，也被所有位置 attend。

StreamingLLM 发现，序列开头的几个 token 会吸收大量注意力，称为 attention sink。保留 4 个 sink token 加一个滑动窗口，就能让模型在无限长的流上稳定生成，但它并不具备真正的长程回忆能力。

### 代表模型

| 模型 | 窗口 | 说明 |
| --- | --- | --- |
| Longformer | 512 | 窗口 + 全局 token |
| Mistral 7B | 4096 | 全层滑动窗口 |
| Gemma 2 | 4096 | 局部/全局交替 |
| StreamingLLM | 任意 | sink token + 窗口 |

## 8.7 Sparse Attention

滑动窗口是一种固定的稀疏模式。更一般地，可以只计算注意力矩阵中选定的位置。

### 常见模式

| 模式 | 连接方式 | 复杂度 |
| --- | --- | --- |
| Strided | 每隔 `k` 个位置连一个 | `O(S√S)` |
| Fixed | 固定的列（如每段开头） | `O(S·c)` |
| Window | 最近 `W` 个 | `O(S·W)` |
| Block-sparse | 按 `128×128` 块选择 | 取决于密度 |
| BigBird | 窗口 + 随机 + 全局 | `O(S)` |
| Longformer | 窗口 + 全局 | `O(S·W)` |

### 块稀疏

GPU 上真正有效的稀疏是块稀疏。原因很简单：一个 `128×128` 的块可以映射到一个 warp 或一组 warp，跳过一个块就是跳过一整段计算；而不规则的单个元素稀疏会导致分支发散和无效访存。

块稀疏的代价是粒度。一个块里只要有一个位置需要计算，整块都要算。密度 1/16 的块稀疏，实际计算量可能接近 1/8。

DeepSeek 的 NSA（Native Sparse Attention）把稀疏做成三个分支：压缩的粗粒度表示、按块选择的细粒度表示、滑动窗口。三个分支的输出加权合并，并且整个模式对硬件对齐。这类工作的共同目标是：**让稀疏在 GPU 上真的变快，而不只是减少理论 FLOPs。**

### 稀疏不一定更快

一个常见的误判是“FLOPs 少了，所以更快”。实际上：

- 如果 kernel 仍然要读完整的 K/V 来做选择，带宽没省；
- 如果块太小，launch 和 index 开销会吃掉收益；
- 如果密度不均匀，负载不均衡，快的 SM 要等慢的。

稀疏 attention 的收益要在端到端时间线上验证，不能只看 FLOPs 计数。

## 8.8 Long-context Attention

上下文从 8K 涨到 128K，full attention 的计算量涨 256 倍，KV Cache 涨 16 倍。这不是一个可以靠“换块更大的 GPU”解决的问题。

### 五类方法

| 方法 | 代表 | 改变什么 |
| --- | --- | --- |
| 精确高效实现 | FlashAttention | 不改变 FLOPs，减少 HBM 读写 |
| 稀疏/窗口 | Mistral、NSA | 减少参与计算的位置 |
| KV 压缩 | GQA、MLA、量化 | 减少每个 token 的 KV 字节 |
| 分块/检索 | chunked prefill、RAG | 减少单次处理的长度 |
| 状态化 | attention sink、线性 attention | 用固定状态代替增长的 KV |

这几类方法可以叠加。一个 128K 上下文的系统，可能同时用 GQA、FlashAttention、chunked prefill 和 prefix cache。

### 计算量

以 70B、80 层、64 个头、`d = 128` 为例，full attention 的 prefill FLOPs（causal）是：

$$
2 \times 80 \times 64 \times S^2 \times 128
$$

| 上下文 `S` | Attention FLOPs | 线性层 FLOPs | Attention 占比 |
| --- | --- | --- | --- |
| 4K | 2.2e13 | 5.7e14 | 4% |
| 32K | 1.4e15 | 4.6e15 | 30% |
| 128K | 2.2e16 | 1.8e16 | 123% |

128K 时 attention 的计算量已经超过所有线性层。在 8×H100 上，仅 attention 的理想时间就是：

$$
\frac{2.2\text{e}16}{8 \times 989\text{e}12} \approx 2.8 \text{ 秒}
$$

按 50% 效率算约 5.6 秒。这解释了为什么长上下文 prefill 的 TTFT 是秒级。

### 显存

如果不分块，128K 上下文的分数矩阵是 `[64, 131072, 131072]`，FP16 下 2.2 TB。没有任何 GPU 放得下。所以长上下文 attention 必须分块，不能让 `S²` 落进 HBM。

KV Cache 也不小。GQA、8K 时每请求 2.5 GiB，128K 时是 40 GiB。单卡 80 GB 扣掉权重后放不下几个请求。这就是长上下文服务昂贵的原因。

### 有效上下文

模型声称支持 128K，不代表它能在 128K 上正确检索信息。

“Lost in the middle” 发现，信息放在长文档中间时，模型召回率明显低于开头和结尾。RULER 等基准进一步显示，很多模型的有效上下文只有标称值的一半甚至更少。

评估长上下文时，不能只看 perplexity。要用 needle-in-a-haystack、多跳检索、长文档问答等任务，分位置、分长度地测。

## 8.9 Attention Computation and Memory Costs

把前面的公式汇总。

### FLOPs

| 场景 | 公式 | 说明 |
| --- | --- | --- |
| Prefill，非 causal | `4 B n_h S² d` | QK + AV |
| Prefill，causal | `2 B n_h S² d` | 有效区域减半 |
| Decode | `4 B n_h S d` | 每个请求 1 个 Query |
| 投影 | `2 B S (2H² + 2H H_kv)` | Q/K/V/O |

### 访存

KV Cache 的字节数：

$$
\text{KV bytes} = 2 L B S n_{kv} d \times \text{bytes per element}
$$

decode 时每个 step 要把整个 KV Cache 读一遍（逻辑上），所以时间下限是：

$$
\text{TPOT}_{\text{attn}} \approx \frac{\text{KV bytes}}{\text{bandwidth}}
$$

prefill 时，如果不分块，分数矩阵要写回 HBM：

$$
\text{score bytes} = B n_h S^2 \times \text{bytes per element}
$$

分块之后，这部分变成片上读写，HBM 流量降到 `O(B S H)`。

### 算术强度

**Prefill**：假设只算 Q/K/V/O 的 HBM 流量（`8 B S H` 字节），强度是：

$$
I_{\text{prefill}} \approx \frac{2 B n_h S^2 d}{8 B S n_h d} = \frac{S}{4}
$$

`S = 4K` 时约 1,000 FLOP/byte，远高于脊点 295，是算力受限的。`S = 512` 时约 128，落到带宽受限区。短 prompt 的 attention 反而不划算。

**Decode**：每层每 token 的 FLOPs 是 `4 B n_h S d`，KV 读取是 `4 B n_kv S d` 字节，强度是：

$$
I_{\text{decode}} = \frac{n_h}{n_{kv}}
$$

注意这个值和 batch、序列长度都无关。每个请求有自己的 KV，batch 增大时 FLOPs 和字节同比例增长，不会像 GEMV 那样因为权重共享而提高强度。

| 方案 | `n_h / n_kv` | Decode 强度 |
| --- | --- | --- |
| MHA | 1 | 1 |
| GQA 8:1 | 8 | 8 |
| MQA 64:1 | 64 | 64 |
| H100 脊点 | — | 295 |

即使 MQA，decode attention 仍然低于脊点。**GQA/MQA 的收益主要来自减少字节，而不是让 attention 变成算力受限。**

### 一个算例

70B、GQA 8:1、8K 上下文、batch 1、decode：

- 每层 attention FLOPs：`4 × 64 × 8192 × 128 = 268M`；
- 每层 KV 读取：`4 × 8 × 8192 × 128 = 33.6 MB`；
- 80 层合计：21.5 GFLOPs、2.68 GB；
- 按 3.35 TB/s：约 0.8 毫秒。

同一时刻，线性层要读 140 GB 权重，约 42 毫秒。batch 1 时 attention 只占 2%。但 batch 增大后，权重读取被摊薄，KV 读取线性增长；到 batch 64、8K 上下文，KV 是 160 GiB，超过权重（约 130 GiB），attention 成为主要开销。

### 实现层面的成本

除了 FLOPs 和字节数，还有几项固定成本：

- **kernel launch**：每层一次 attention kernel，80 层 80 次；
- **layout 转换**：`[B, S, n_h, d]` 到 `[B, n_h, S, d]` 的 transpose 可能触发拷贝；
- **mask 构造**：每步生成 `S²` mask 会抵消分块带来的收益；
- **split-K / 并行度**：decode 时 `S` 很大但 `B` 很小，需要沿 `S` 切分才能填满 SM。

这些成本在短序列、小 batch 时可能超过 attention 本身的计算。分析时要看时间线，不只看 roofline。

## 本章小结

- Scaled dot-product attention 的核心是 `softmax(QK^T/√d)V`；除以 `√d` 是为了控制 logits 方差，避免 softmax 饱和。
- MHA 用 `n_h` 组独立的 Q/K/V，参数量 `4H²`；多头提供的是多种关注模式，不只是更多参数。
- MQA 让所有 Query 头共享一组 K/V，KV Cache 缩小 `n_h` 倍，但质量下降。
- GQA 分成 `G` 组，是 MHA 和 MQA 的折中；8:1 是当前主流配置，KV Cache 是 MHA 的 1/8。
- Mask 有 causal、padding、document 三种；不同框架的 mask 语义相反，混用不会报错但结果错误。
- 滑动窗口把复杂度降到 `O(S·W)`，多层叠加可以扩大感受野；环形缓冲区和全局 token 是常见实现。
- 稀疏 attention 只有在块对齐、负载均衡时才会真正变快，FLOPs 减少不等于时间减少。
- 长上下文下 attention 的计算量超过线性层；128K 上下文的 prefill 需要数秒，且 KV Cache 达到几十 GB。
- Decode attention 的算术强度是 `n_h / n_kv`，和 batch 无关；GQA/MQA 的收益来自减少字节，而不是改变受限类型。

## 延伸阅读

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
- Bahdanau et al., [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), 2014.
- Shazeer, [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150), 2019.
- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245), 2023.
- Beltagy et al., [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150), 2020.
- Zaheer et al., [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062), 2020.
- Child et al., [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509), 2019.
- Xiao et al., [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453), 2023.
- Liu et al., [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172), 2023.
- Hsieh et al., [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654), 2024.
- Yuan et al., [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089), 2025.
