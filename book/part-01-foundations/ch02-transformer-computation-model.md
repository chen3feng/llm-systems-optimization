---
chapter: 2
part: 1
title: Transformer Computation Model
level: tbd
status: draft
prerequisites: [1]
lab:
---

# Chapter 2. Transformer Computation Model

`hello world` 进入模型之前，先变成两个整数。

这两个整数查表得到两个向量，向量经过几十层矩阵乘法、注意力、归一化和激活函数，最后变成一个 128K 维的分数向量。分数最高的那个 token 被取出来，变成下一个字。重复这个过程，就是生成。

这一章不讲 Transformer 为什么有效，只讲它在算什么、每个张量是什么形状、每一步花多少计算和内存。后面所有性能分析，都建立在这套计算图上。

## 2.1 Tokenization and Token Representation

模型不直接处理字符串。它处理的是 token id。

Tokenization 把文本切成子词单元，再映射成整数。主流做法是 Byte-Pair Encoding（BPE）或其变体：从字符开始，反复合并出现频率最高的相邻对，直到词表达到预定大小。

几个具体数字：

| 模型 | 词表大小 | Tokenizer |
| --- | --- | --- |
| GPT-3 | 50,257 | byte-level BPE |
| Llama 3 | 128,256 | tiktoken-based BPE |
| Qwen2.5 | 151,643 | BPE |

词表变大，单个 token 能承载更多信息，同样文本的 token 数变少；但 embedding 和输出层的参数量也变大。Llama 3 把词表从 Llama 2 的 32K 扩到 128K，中文和代码的压缩率明显改善。

Token 数直接决定成本。推理按 token 计费，KV Cache 按 token 增长，训练数据按 token 计量。一个粗略的估算：

- 英文大约 4 个字符一个 token；
- 中文大约 1–2 个汉字一个 token，取决于 tokenizer；
- 代码的 token 密度和自然语言不同，缩进和符号也会占 token。

这些比例不是常数。同一段文本，用不同 tokenizer 可能差出 20%–50% 的 token 数。做成本估算时，必须用目标模型自己的 tokenizer 实测，不能拿字符数直接换算。

### Tokenizer 是模型的一部分

Tokenizer 不是预处理工具，它和权重绑定。换了 tokenizer，embedding 表就对不上，模型输出会变成乱码。

这也是为什么微调后的模型通常要保留原 tokenizer，新增特殊 token 时要同步扩展 embedding，并重新训练对应的行。

### 特殊 token

除了文本 token，tokenizer 还会插入特殊 token：

- `BOS` / `EOS`：序列开始和结束；
- `PAD`：batch 内对齐；
- `system` / `user` / `assistant`：对话角色标记；
- 工具调用、图像占位符等。

这些 token 不参与自然语言统计，但会影响注意力和位置编码。做 prompt 拼接或 batch 时，要确保特殊 token 的位置和模型训练时一致。

## 2.2 Embedding and Positional Encoding

Token id 是一个整数，模型需要把它变成向量。

Embedding 是一张形状为 `[V, H]` 的查找表：`V` 是词表大小，`H` 是 hidden dimension。输入 token id 后，直接取出对应的行：

```python
x = embedding_table[token_ids]   # [B, S] -> [B, S, H]
```

这一步没有矩阵乘法，只有访存。它的成本是 `B × S × H` 次读，以及 embedding 表本身的显存占用 `V × H × bytes`。

Llama 3 70B 的 `V = 128256`，`H = 8192`，FP16 下 embedding 表约 2.1 GB。如果输出层和 embedding 不共享权重，还要再占一份。

### 位置编码

Self-attention 本身没有顺序概念。打乱 token 顺序，注意力计算结果不变。所以必须把位置信息注入模型。

历史上主要有两种做法：

- **Learned positional embedding**：给每个位置一个可学习向量，和 token embedding 相加。GPT-2 用这种方式，最大长度在训练时就固定了。
- **RoPE（Rotary Position Embedding）**：不增加参数，而是把 Query 和 Key 的向量按位置旋转。两个 token 的点积只依赖它们的相对距离。

现代开源模型大多用 RoPE。Llama 3 的 RoPE base 是 500,000，比原始的 10,000 大得多，目的是支持更长的上下文。

RoPE 的直觉可以用二维旋转理解。把 head dimension 两两分组，第 `i` 组的旋转角是：

$$
\theta_i = \text{pos} \cdot \text{base}^{-2i/d}
$$

位置越远，旋转越多；两个位置的相对角度只由它们的差值决定。这样一来，模型在训练时见过的相对距离，可以在推理时泛化到更长的序列。

RoPE 不增加参数，也不增加显存，只在 attention 之前对 Q/K 做一次逐元素旋转。代价很小，但对长上下文至关重要。

## 2.3 Transformer Decoder Architecture

现代 LLM 基本都是 decoder-only Transformer。一个 decoder block 由两部分组成：

1. 多头自注意力（加上残差和归一化）；
2. 前馈网络（加上残差和归一化）。

用伪代码表示：

```python
for layer in layers:
    x = x + attention(norm1(x))
    x = x + feed_forward(norm2(x))
```

这是 pre-norm 结构：归一化放在子层之前，残差连接把原始输入直接加到输出上。GPT-2 之后的模型大多采用这种结构，因为深层网络更容易训练。

以 Llama 3 70B 为例，关键超参数是：

| 参数 | 值 |
| --- | --- |
| 层数 `L` | 80 |
| Hidden dimension `H` | 8192 |
| Query 头数 `n_h` | 64 |
| KV 头数 `n_kv` | 8 |
| Head dimension `d` | 128 |
| FFN 中间维度 `d_ff` | 28672 |
| 词表 `V` | 128256 |
| 归一化 | RMSNorm |
| 激活函数 | SwiGLU |
| 位置编码 | RoPE |

这张表基本决定了一个 token 要花多少计算、多少显存。后面所有估算都从它出发。

### 张量形状

把 batch size 记为 `B`，序列长度记为 `S`。一个 block 内的主要张量形状是：

| 张量 | 形状 |
| --- | --- |
| 输入 `x` | `[B, S, H]` |
| Q | `[B, S, n_h, d]` |
| K / V | `[B, S, n_kv, d]` |
| Attention scores | `[B, n_h, S, S]` |
| Attention 输出 | `[B, S, n_h, d]` |
| FFN 中间结果 | `[B, S, d_ff]` |

注意 `Q` 的头数是 64，`K/V` 的头数是 8。这就是 Grouped-Query Attention（GQA）：每 8 个 Query 头共享一组 K/V。它把 KV Cache 缩小到 MHA 的八分之一，同时保留了大部分表达能力。

## 2.4 Self-Attention Computation

Self-attention 的目标是：让每个 token 根据上下文，决定该关注哪些其他 token。

计算分成四步：

1. 用三个线性层把输入投影成 Query、Key、Value；
2. 计算 Q 和 K 的点积，得到注意力分数；
3. 除以 `sqrt(d)`，加 causal mask，做 softmax；
4. 用注意力权重对 V 加权求和。

公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + M\right)V
$$

其中 `M` 是 causal mask：位置 `i` 只能看到 `j ≤ i`，`j > i` 的位置被设为负无穷。

### 投影

对每个 token：

$$
Q = x W_Q,\quad K = x W_K,\quad V = x W_V
$$

`W_Q` 的形状是 `[H, n_h × d]`，`W_K` 和 `W_V` 是 `[H, n_kv × d]`。Llama 3 70B 里，`n_h × d = 8192`，`n_kv × d = 1024`。所以 Q 投影是 `8192 × 8192`，K/V 投影各是 `8192 × 1024`。

### 分数与 softmax

分数矩阵的形状是 `[B, n_h, S, S]`。序列长度 8192、64 个头、batch 1 时，这个矩阵有 `64 × 8192 × 8192 ≈ 43 亿` 个元素。FP16 下就是 8.6 GB。

这正是 FlashAttention 要解决的问题。标准实现会把整个 `S × S` 矩阵写回 HBM，再读出来做 softmax；FlashAttention 把 Q/K/V 分块，在 shared memory 里做 online softmax，避免物化完整的分数矩阵。它没有减少 FLOPs，减少的是 HBM 读写。

### Causal mask

自回归模型不能看到未来。实现上，在 softmax 之前把上三角部分设为 `-inf`：

```python
scores = Q @ K.transpose(-2, -1) / sqrt(d)
scores = scores.masked_fill(causal_mask == 0, float("-inf"))
attn = softmax(scores, dim=-1)
out = attn @ V
```

mask 让有效计算量减半。但 GPU 仍然要处理整个矩阵，除非 kernel 利用三角结构跳过无效块。FlashAttention 的 causal 版本会跳过完全被 mask 的块。

### GQA 与 KV Cache

推理时，K 和 V 会被缓存下来，避免每个 token 重新计算整个前缀。

每个 token 的 KV Cache 大小是：

$$
2 \times L \times n_{kv} \times d \times \text{bytes}
$$

Llama 3 70B、FP16：`2 × 80 × 8 × 128 × 2 = 327,680` 字节，约 320 KiB/token。8192 token 就是 2.5 GiB。

如果换成 MHA（`n_kv = 64`），同样上下文要 20 GiB。GQA 不是一个小优化，它直接决定了单卡能放多少个并发请求。

## 2.5 Feed-forward Networks and Gating

注意力负责 token 之间的信息交换，FFN 负责每个 token 内部的特征变换。

标准 FFN 是两层线性加一个激活：

$$
\text{FFN}(x) = W_2 \, \sigma(W_1 x)
$$

现代模型多用 SwiGLU：

$$
\text{FFN}(x) = W_{\text{down}} \left( \text{SiLU}(W_{\text{gate}} x) \odot (W_{\text{up}} x) \right)
$$

它有三个矩阵：`W_gate`、`W_up`、`W_down`。`SiLU` 是 `x * sigmoid(x)`，`⊙` 是逐元素乘。

### 为什么是三个矩阵

两个矩阵的 FFN 参数量是 `2 × H × d_ff`，SwiGLU 是 `3 × H × d_ff`。为了保持参数量相当，SwiGLU 通常把 `d_ff` 缩小到原来的 `2/3` 左右。Llama 3 70B 的 `d_ff = 28672`，大约是 `H` 的 3.5 倍；对应的总参数量约 `3 × 8192 × 28672 = 7.05 亿` 每层。

FFN 是模型里参数量最大的部分。70B 模型里，FFN 占每层参数的 80% 以上；注意力只占不到 20%。

### 计算特征

FFN 是逐 token 的矩阵乘法，没有 token 间交互。它的形状是：

- prefill：`[B × S, H] × [H, d_ff]`，大 GEMM，GPU 利用率高；
- decode：`[B, H] × [H, d_ff]`，batch 小时退化成 GEMV，受显存带宽限制。

这也是为什么 decode 阶段增大 batch 特别有效：同样的权重，一次读进来可以服务更多 token。

## 2.6 Normalization and Residual Connections

归一化让深层网络的激活值保持稳定。

LayerNorm 对每个 token 的 hidden 向量做归一化：

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta
$$

RMSNorm 去掉了均值中心化和偏移项，只保留缩放：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{H}\sum_i x_i^2 + \epsilon}} \odot \gamma
$$

RMSNorm 少了一次均值计算和一次减法，参数也少一半。对 LLM 来说，这种简化几乎不影响效果，却减少了 kernel 的访存和计算。Llama 系列都用 RMSNorm。

### Pre-norm 与 Post-norm

Post-norm 把归一化放在残差相加之后：

$$
x = \text{Norm}(x + \text{Attention}(x))
$$

这是原始 Transformer 的做法。层数一深，梯度容易不稳定，需要 warmup 和精细的超参。

Pre-norm 把归一化放在子层之前：

$$
x = x + \text{Attention}(\text{Norm}(x))
$$

残差路径上没有归一化，梯度可以更直接地回传。代价是表达能力略有变化，但训练稳定性好得多。现代 LLM 基本都用 pre-norm。

### 残差流

残差连接形成一条“残差流”（residual stream）：每一层的输出都加回主路径，信息可以跨层传递。

从系统角度看，残差流意味着一层要读写 `[B, S, H]` 的激活。层数越多、batch 越大、序列越长，这部分显存和带宽开销越可观。做 kernel fusion 时，残差加法通常和归一化、线性层融合在一起，减少一次完整的读写。

## 2.7 Autoregressive Generation

训练和推理的计算模式完全不同。

### 训练

训练时，整个序列是已知的。模型可以并行处理所有位置：

```python
logits = model(input_ids)          # [B, S, V]
loss = cross_entropy(logits[:, :-1], input_ids[:, 1:])
```

causal mask 保证位置 `i` 只能看到 `i` 之前的信息，所以一次 forward 就能算出所有位置的损失。这就是 teacher forcing。

### 推理

推理时，模型只能一个一个生成：

```text
prompt -> prefill -> token 1 -> token 2 -> ... -> token N
```

每生成一个 token，都要把它追加到序列末尾，再跑一次 forward。如果没有 KV Cache，每一步都要重新计算整个前缀的 K/V，复杂度是 `O(S^2)`；有了 KV Cache，每一步只计算新 token 的 K/V，复杂度降到 `O(S)`。

一个简化的 decode 循环：

```cpp
for (int step = 0; step < max_steps; ++step) {
  auto q = linear(x, Wq);          // [B, H]
  auto k = linear(x, Wk);          // [B, Hkv]
  auto v = linear(x, Wv);          // [B, Hkv]
  kv_cache.append(k, v);
  auto attn = attention(q, kv_cache);  // 读取历史 K/V
  auto out = linear(attn, Wo);
  x = residual + ffn(out);
  auto logits = lm_head(x);        // [B, V]
  auto next = sample(logits);
  if (next == eos) break;
}
```

每一步的主要开销不是 FLOPs，而是读取权重和 KV Cache。这就是 decode 受显存带宽限制的根本原因。

### 采样

模型输出的是 logits，不是 token。采样策略决定怎么从 logits 里选：

| 策略 | 做法 | 特点 |
| --- | --- | --- |
| Greedy | 取概率最大的 token | 确定性强，容易重复 |
| Temperature | 用 `T` 缩放 logits 再采样 | `T` 越低越保守 |
| Top-k | 只在前 k 个 token 里采样 | 限制候选集 |
| Top-p | 取累计概率达到 p 的最小集合 | 动态候选集，常用 |

采样本身的计算量很小，但它影响生成质量和长度分布，进而影响系统的吞吐和成本。

## 2.8 Computational Graph of a Transformer

把上面的内容拼起来，一个 decoder block 的计算图是：

```text
x ──► RMSNorm ──► Q/K/V 投影 ──► Attention ──► O 投影 ──► + ──► x
│                                                          ▲
└──────────────────────────────────────────────────────────┘

x ──► RMSNorm ──► Gate/Up 投影 ──► SiLU ⊙ ──► Down 投影 ──► + ──► x
│                                                            ▲
└────────────────────────────────────────────────────────────┘
```

### 参数量

一个 block 的参数量近似为：

$$
N_{\text{layer}} \approx \underbrace{2H^2 + 2H H_{kv}}_{\text{attention}} + \underbrace{3H d_{ff}}_{\text{FFN}} + \underbrace{2H}_{\text{norm}}
$$

其中 `H_{kv} = n_{kv} × d`。

代入 Llama 3 70B：

| 部分 | 参数量 |
| --- | --- |
| Q 投影 | `8192 × 8192 = 67.1M` |
| K 投影 | `8192 × 1024 = 8.4M` |
| V 投影 | `8192 × 1024 = 8.4M` |
| O 投影 | `8192 × 8192 = 67.1M` |
| FFN | `3 × 8192 × 28672 = 704.6M` |
| 两个 RMSNorm | `2 × 8192 = 16K` |
| **每层合计** | **约 855.7M** |

80 层就是约 68.5B，再加上 embedding 和输出层，总量约 70B。

### 每 token 的计算量

线性层占主导。每个 token 的 forward FLOPs 约等于：

$$
\text{FLOPs}_{\text{token}} \approx 2N
$$

其中 `N` 是参数量。70B 模型每 token 约 140 GFLOPs。

prefill 处理 `S` 个 token，计算量约 `2NS`；decode 每步只处理 1 个 token，但要把权重读一遍。**计算量和访存量在这里分道扬镳**，这也是下一章要展开的核心问题。

### 显存

一个 block 需要读写的内容包括：

- 权重：`N_layer × bytes`；
- 激活：`B × S × H` 及其若干中间张量；
- KV Cache：`2 × B × S × n_kv × d × bytes`。

推理时，权重和 KV Cache 是常驻开销，激活是临时开销。训练时，激活还要保存下来做反向传播，显存压力大得多。

下一章把这些形状变成公式：参数量、FLOPs、访存量、通信量，以及怎么用这些公式估算一张卡能跑多大的模型、一个请求要花多少时间。

## 本章小结

- Tokenizer 把文本变成 token id，token 数直接决定推理成本和 KV Cache 大小。
- Embedding 是查表，不产生 FLOPs；RoPE 不增加参数，但决定长上下文能力。
- 一个 decoder block 由 attention 和 FFN 组成，pre-norm + 残差是主流结构。
- GQA 把 KV Cache 缩小到 MHA 的 1/8，是推理系统最重要的结构优化之一。
- 训练可以并行处理整个序列；推理必须逐 token 生成，KV Cache 把每步复杂度从 `O(S^2)` 降到 `O(S)`。
- 每 token 的 forward FLOPs 约等于 `2N`，但 decode 的瓶颈是显存带宽，不是算力。

## 延伸阅读

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864), 2021.
- Zhang & Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467), 2019.
- Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202), 2020.
- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245), 2023.
- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), 2022.
- Meta AI, [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783), 2024.
- Hugging Face, [Llama 3 model documentation](https://huggingface.co/docs/transformers/model_doc/llama3).
