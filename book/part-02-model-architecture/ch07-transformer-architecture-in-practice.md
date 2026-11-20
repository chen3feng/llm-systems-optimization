---
chapter: 7
part: 2
title: Transformer Architecture in Practice
level: tbd
status: draft
prerequisites: [2, 3]
lab:
---

# Chapter 7. Transformer Architecture in Practice

打开 Llama 3 的 checkpoint，会发现一个和论文对不上的地方：线性层没有 bias。

再打开 Qwen2.5，`q_proj` 又有 bias。Gemma 2 的 `head_dim` 是 256，不等于 `hidden_size / num_heads`。DeepSeek-V3 的注意力根本没有单独的 K/V 投影，而是一个低秩压缩加一个 RoPE 分支。

“Transformer”是一个家族，不是一份规格。论文给出的是骨架，每个模型在归一化位置、偏置项、激活函数、位置编码排布、权重共享上都有自己的选择。这些选择不影响论文结论，但直接决定你能不能加载权重、kernel 怎么写、显存怎么算。

这一章讲实践：从一个 `config.json` 出发，把每一层映射成确切的张量形状和算子，写出能跑、能和参考实现对上数的代码。

## 7.1 Decoder-only Transformer

现代 LLM 几乎都是 decoder-only。一个 block 的结构可以写成两行：

```python
h = x + attention(norm1(x))
x = h + mlp(norm2(h))
```

所有模型都符合这个骨架。差别在 `norm` 放哪、`attention` 里 K/V 怎么组织、`mlp` 用什么激活、残差怎么加。

### 一个 checkpoint 里有什么

以 Hugging Face 格式的 Llama 3 8B 为例：

```text
config.json
model.safetensors.index.json
model-00001-of-00004.safetensors
...
```

`config.json` 是结构描述，safetensors 是权重。每个 safetensors 文件开头有一段 JSON header，记录每个张量的名字、dtype、shape 和字节偏移。加载时可以先读 header，校验形状，再决定是否 mmap。

一个 32 层的模型有这些权重：

```text
model.embed_tokens.weight                      [128256, 4096]
model.layers.0.self_attn.q_proj.weight         [4096, 4096]
model.layers.0.self_attn.k_proj.weight         [1024, 4096]
model.layers.0.self_attn.v_proj.weight         [1024, 4096]
model.layers.0.self_attn.o_proj.weight         [4096, 4096]
model.layers.0.mlp.gate_proj.weight            [14336, 4096]
model.layers.0.mlp.up_proj.weight              [14336, 4096]
model.layers.0.mlp.down_proj.weight            [4096, 14336]
model.layers.0.input_layernorm.weight          [4096]
model.layers.0.post_attention_layernorm.weight [4096]
...
model.norm.weight                              [4096]
lm_head.weight                                 [128256, 4096]
```

注意矩阵的形状是 `[out_features, in_features]`，这是 PyTorch 的约定，也是 safetensors 里的实际排布。做 `x @ W.T` 还是 `W @ x`，取决于你的 GEMM 怎么调用，但读文件时必须按这个顺序解释。

### 每层参数量

把上面的形状乘起来：

| 模块 | 计算 | 参数量 |
| --- | --- | --- |
| q_proj | `4096 × 4096` | 16.78M |
| k_proj | `1024 × 4096` | 4.19M |
| v_proj | `1024 × 4096` | 4.19M |
| o_proj | `4096 × 4096` | 16.78M |
| gate_proj | `14336 × 4096` | 58.72M |
| up_proj | `14336 × 4096` | 58.72M |
| down_proj | `4096 × 14336` | 58.72M |
| 两个 RMSNorm | `2 × 4096` | 8K |
| **每层合计** |  | **218.1M** |

32 层是 6.98B，加上 embedding（525M）和 lm_head（525M，不共享），总计约 8.03B。这就是“8B”的来历。

FFN 占每层的 80.8%，注意力只占 19.2%。模型越大，这个比例越稳定——所以 FFN 的优化（量化、稀疏、MoE）对总成本影响最大。

### 加载时的坑

- **tied embeddings**：`tie_word_embeddings: true` 时 checkpoint 里没有 `lm_head.weight`，它和 `embed_tokens.weight` 是同一块内存。Llama 3 8B 是 false，Gemma 2 是 true。
- **bias**：Qwen2 的 `q_proj`、`k_proj`、`v_proj` 有 bias，Llama 没有。按 Llama 的结构写加载器，读 Qwen 会直接少读三个张量。
- **`rotary_emb.inv_freq`**：老版本 checkpoint 会把它存成 buffer。它不是参数，现代实现通常在运行时计算，加载时应该跳过。
- **dtype**：BF16 checkpoint 的每个元素 2 字节；有的模型混合了 FP32 的 norm 权重。不要假设全文件一个 dtype。
- **分片**：70B 通常切成十几个文件，index 里有 `weight_map`。按名字查文件，不要按顺序拼。

## 7.2 Pre-Norm and Post-Norm

归一化的位置决定了残差路径的形状。

### 两种结构

Post-norm 是原始 Transformer 的做法：

$$
x = \text{Norm}(x + \text{Attention}(x))
$$

Pre-norm 把归一化移到子层之前：

$$
x = x + \text{Attention}(\text{Norm}(x))
$$

Pre-norm 的残差路径上没有归一化，梯度可以从最后一层直接回传到第一层。代价是深层的激活值会随层数增长，需要在最后加一个 `model.norm` 把尺度拉回来。

| | Post-norm | Pre-norm |
| --- | --- | --- |
| 梯度路径 | 经过每层 Norm | 恒等路径 |
| 训练稳定性 | 需要 warmup、精细调参 | 稳定 |
| 最终输出尺度 | 自然受控 | 需要 final norm |
| 代表模型 | 原始 Transformer、BERT | GPT-2 之后绝大多数 LLM |

### 变体

**Sandwich norm**：在子层前后各加一个 norm。Gemma 2 采用这种结构，每个子层的输出先归一化再加回残差。它的目的是控制深层激活的尺度，代价是每层多两个 norm。

**Parallel attention/FFN**：注意力和 FFN 共享同一个 norm，并行计算后一起加回：

$$
x = x + \text{Attention}(\text{Norm}(x)) + \text{FFN}(\text{Norm}(x))
$$

GPT-J 和 PaLM 用过。它把两个子层变成可以并行的分支，减少了依赖链，但表达能力略有下降。

**DeepNorm**：对残差做缩放，允许更深的 post-norm。主要用于超深模型，LLM 里不常见。

### 实现上的影响

Norm 的位置决定了每层有几个 norm kernel、残差在哪一步加、能不能融合。

Pre-norm 的两处残差加法都可以和后续的 norm 融合：`add + rmsnorm` 一个 kernel 完成，省一次读写。Post-norm 的 norm 在残差之后，融合方式不同。

移植模型时不要“顺手改进” norm 的位置。即使理论上等价，数值上也不等价，和参考实现的对数会对不上。

## 7.3 RMSNorm and LayerNorm

LayerNorm 做中心化和缩放：

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta
$$

RMSNorm 只做缩放：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{H}\sum_i x_i^2 + \epsilon}} \odot \gamma
$$

少了均值计算和偏移项，参数从 `2H` 降到 `H`，kernel 也少一次读写。

### 实现细节

**epsilon 的位置**。Llama 的实现是 `rsqrt(mean(x^2) + eps)`；有的框架写成 `rsqrt(mean(x^2)) + eps` 或 `mean(x^2 + eps)`。三者数值上接近，但在接近零的激活上会有可见差异。对齐参考实现时，先确认 eps 在哪。

**Gemma 的偏移**。Gemma 的 RMSNorm 权重初始化为 0，前向时用 `(1 + weight)`：

```python
output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
return output * (1.0 + weight)
```

如果按 Llama 的方式直接乘 `weight`，第一层的输出就全是 0。

**累加精度**。8192 个 FP16 数的平方和可能溢出，也可能丢精度。求和必须在 FP32 里做，最后再转回输入 dtype。这不是可选项，是正确性要求。

**一个 block 的实现**：

```cuda
__global__ void rms_norm(const half* x, const half* w, half* out,
                         int n, float eps) {
  float sum = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = __half2float(x[i]);
    sum += v * v;
  }
  // warp + block reduction
  sum = block_reduce_sum(sum);
  float scale = rsqrtf(sum / n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    out[i] = __float2half(__half2float(x[i]) * scale * __half2float(w[i]));
  }
}
```

这个 kernel 读两遍 `x`：一遍求和，一遍缩放。如果 `n` 不大，可以把 `x` 留在寄存器或 shared memory 里，只读一遍。H100 的 shared memory 放得下 8192 个 FP16（16 KB），所以单遍实现是可行的。

### 性能特征

对一个 token、一层、`H = 8192`：

- 读 `x` 16 KB，写 `out` 16 KB，读 `w` 16 KB；
- 两个 norm，约 96 KB；
- 80 层约 7.7 MB。

按 3.35 TB/s 算不到 3 微秒，带宽上完全不是瓶颈。真正的开销是 kernel launch：160 个 norm kernel，每个 2–4 微秒，累计 0.3–0.6 毫秒。当 decode 的 TPOT 只有 10 毫秒时，这是 3%–6%。

所以 RMSNorm 的优化方向不是带宽，而是融合：和残差加法融合、和量化融合、或者干脆用 CUDA Graph 把 launch 开销压掉。

## 7.4 Activation Functions: GELU and SwiGLU

激活函数决定了 FFN 的形状和计算模式。

### GELU

GELU 是 `x · Φ(x)`，`Φ` 是标准正态分布的累积分布函数。精确计算需要 `erf`，实际实现用 tanh 近似：

$$
\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)
$$

GPT-2 和 BERT 用的是这个近似。它和精确值的最大差异在 1e-3 量级，对 FP16 来说可以忽略。

### SiLU 与 GLU

SiLU（也叫 Swish）是 `x · sigmoid(x)`。它比 GELU 便宜，没有 `tanh` 或 `erf`。

GLU（Gated Linear Unit）把输入分成两路，一路做激活，一路做门控：

$$
\text{GLU}(x) = \sigma(W_1 x) \odot (W_2 x)
$$

把 `σ` 换成 SiLU 就是 SwiGLU：

$$
\text{SwiGLU}(x) = \text{SiLU}(W_{\text{gate}} x) \odot (W_{\text{up}} x)
$$

再经过 `W_down` 投影回 hidden dimension：

$$
\text{FFN}(x) = W_{\text{down}}\left(\text{SiLU}(W_{\text{gate}}x) \odot (W_{\text{up}}x)\right)
$$

### 为什么 d_ff 是 8/3 H

两个矩阵的 FFN 参数量是 `2 H d_ff`，三个矩阵的 SwiGLU 是 `3 H d_ff`。要保持参数量不变，`d_ff` 要乘 `2/3`。原始 Transformer 的 `d_ff = 4H`，所以 SwiGLU 的 `d_ff ≈ 8/3 H`。

实际模型会取整到 128 或 256 的倍数。Llama 3 8B 的 `d_ff = 14336`，而 `8/3 × 4096 = 10922`——它其实用了更大的 FFN，参数量比“参数等价”版本多 30%。这是一个设计选择，不是笔误。

| 模型 | 激活 | 门控 | `d_ff` | `d_ff / H` |
| --- | --- | --- | --- | --- |
| GPT-2 | GELU | 否 | 4H | 4.0 |
| Llama 3 8B | SwiGLU | 是 | 14336 | 3.5 |
| Llama 3 70B | SwiGLU | 是 | 28672 | 3.5 |
| Qwen2.5 7B | SwiGLU | 是 | 18944 | 5.3 |
| Mistral 7B | SwiGLU | 是 | 14336 | 3.5 |
| Gemma 2 9B | GeGLU | 是 | 14336 | 4.0 |

Qwen 的 5.3 倍明显大于其他模型，FFN 占比更高。做显存和 FLOPs 估算时不能套用 Llama 的比例。

### 融合

`SiLU(gate) * up` 是一个逐元素操作，输入两个 `[B, S, d_ff]` 张量，输出一个。如果不融合，要走三次 HBM：读 gate、读 up、写结果。

vLLM 的 `SiluAndMul` 把它做成一个 kernel。更进一步，gate/up 的 GEMM 可以拼成一个 `[H, 2 d_ff]` 的矩阵乘，输出直接进激活 kernel，减少一次中间张量的往返。

在 decode 阶段，`d_ff = 28672`、batch 1、FP16，一个 `[1, 28672]` 张量是 57 KB。三次往返约 170 KB，和 FFN 权重（每层 3 × 8192 × 28672 × 2 = 1.4 GB）比起来可以忽略。真正有意义的是 prefill：`[B, S, d_ff]` 可能到 GB 级，融合能省下可观的带宽。

## 7.5 RoPE and Positional Encoding

RoPE 把位置信息编码成旋转，作用在 Q 和 K 上。

### 公式

把 head dimension 两两分组，第 `i` 组的频率是：

$$
\theta_i = \text{base}^{-2i/d}, \quad i = 0, 1, \ldots, d/2 - 1
$$

位置 `p` 的旋转角是 `p · θ_i`。对一对分量 `(x_{2i}, x_{2i+1})`：

$$
\begin{aligned}
x'_{2i} &= x_{2i}\cos(p\theta_i) - x_{2i+1}\sin(p\theta_i) \\
x'_{2i+1} &= x_{2i}\sin(p\theta_i) + x_{2i+1}\cos(p\theta_i)
\end{aligned}
$$

两个 token 的 Q/K 点积只依赖它们的相对位置，这是 RoPE 的核心性质。

### 两种排布

上面的公式是交错排布（interleaved）：相邻两个分量是一组。GPT-J 用这种。

另一种是半分割（half-split）：前一半和后一半配对，`(x_i, x_{i+d/2})`。Hugging Face 的 Llama 实现用这种：

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

q_embed = q * cos + rotate_half(q) * sin
```

两种排布在数学上只差一个固定的维度置换。但 checkpoint 的权重是按某一种排布训练的，混用会让模型输出乱码。**移植时先确认参考实现用的是哪一种。**

### 预计算表

`cos` 和 `sin` 可以预计算成 `[max_seq, d/2]` 的表。128K 上下文、`d = 128`、FP32：

$$
131072 \times 64 \times 4 \times 2 \approx 67 \text{ MB}
$$

不算大，但也不小。有的实现存 `[max_seq, d]`（把两半拼成完整长度），内存翻倍。长上下文场景下，按需计算或分块缓存更划算。

### 位置 id

RoPE 用的是逻辑位置，不是物理偏移。以下几种情况容易出错：

- **左 padding**：batch 内补齐的 token 不应该消耗位置，`position_ids` 要从第一个有效 token 开始；
- **PagedAttention**：KV 存在不连续的 block 里，位置必须由 block table 之外的逻辑计数决定；
- **Prefix cache**：命中前缀的请求，新 token 的位置要接着前缀长度算；
- **Pack 多个序列**：每个序列的位置从 0 重新开始，不能共用一条递增的位置。

位置错一位，输出不会报错，只是质量下降。这是最难查的一类 bug。

### 长度扩展

模型训练时的 `max_position` 是 8K，推理时想跑到 128K，需要改 RoPE。

| 方法 | 做法 | 特点 |
| --- | --- | --- |
| Linear scaling | 位置除以因子 `s` | 简单，短文本性能下降 |
| NTK-aware | 调整 base | 不改代码，效果依赖参数 |
| YaRN | 分段缩放 + 温度 | 长文本效果好，配置复杂 |
| Dynamic NTK | 按当前长度动态调整 | 无需训练，推理开销略增 |

这些方法都改变 `θ_i` 或位置 `p` 的映射。加载模型时要读 `rope_scaling` 字段，不能假设 base 就是 `rope_theta`。

### 实现

RoPE 通常在 QKV 投影之后、写入 KV Cache 之前执行：

```cpp
// q: [B, S, n_h, d], k: [B, S, n_kv, d]
rope_apply(q, cos, sin, position_ids);   // in-place
rope_apply(k, cos, sin, position_ids);
kv_cache.append(k, v);
```

Q 和 K 都要旋转，V 不旋转。旋转是逐元素的，带宽受限，通常和 QKV 的 split 融合在一个 kernel 里。

## 7.6 Model Configuration and Architecture Variants

`config.json` 里的字段不多，但每一个都对应一处实现分支。

### 关键字段

| 字段 | 含义 | 影响 |
| --- | --- | --- |
| `hidden_size` | 残差流维度 `H` | 所有 GEMM 的输入输出 |
| `num_hidden_layers` | 层数 `L` | 参数量、显存、延迟 |
| `num_attention_heads` | Query 头数 `n_h` | 注意力并行度 |
| `num_key_value_heads` | KV 头数 `n_kv` | KV Cache 大小 |
| `head_dim` | 每个头的维度 `d` | 注意力形状，可能不等于 `H/n_h` |
| `intermediate_size` | FFN 中间维度 | FFN 参数量 |
| `vocab_size` | 词表大小 | embedding、lm_head、logits |
| `rms_norm_eps` | 归一化 epsilon | 数值稳定性 |
| `rope_theta` | RoPE base | 长上下文能力 |
| `attention_bias` | QKV 是否有 bias | 加载器和 GEMM 形状 |
| `tie_word_embeddings` | 是否共享 embedding | 参数量、加载逻辑 |
| `rope_scaling` | 长度扩展配置 | 位置编码实现 |

### 一致性检查

加载前应该校验：

```cpp
assert(c.num_heads % c.num_kv_heads == 0);
assert(c.head_dim * c.num_heads == q_proj_rows);
assert(c.head_dim * c.num_kv_heads == k_proj_rows);
assert(c.vocab_size * c.hidden_size == embed_rows * embed_cols);
```

`head_dim * num_heads == hidden_size` 不是永远成立。Gemma 2 9B 的 `H = 3584`、`n_h = 16`、`d = 256`，`16 × 256 = 4096 ≠ 3584`。如果按 `H / n_h` 推 `d`，注意力和输出投影的形状都会错。

### 模型家族对比

| 模型 | H | L | n_h | n_kv | d | d_ff | V | 激活 | QKV bias | tied |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama 3 8B | 4096 | 32 | 32 | 8 | 128 | 14336 | 128256 | SwiGLU | 否 | 否 |
| Llama 3 70B | 8192 | 80 | 64 | 8 | 128 | 28672 | 128256 | SwiGLU | 否 | 否 |
| Qwen2.5 7B | 3584 | 28 | 28 | 4 | 128 | 18944 | 152064 | SwiGLU | 是 | 否 |
| Mistral 7B | 4096 | 32 | 32 | 8 | 128 | 14336 | 32000 | SwiGLU | 否 | 否 |
| Gemma 2 9B | 3584 | 42 | 16 | 8 | 256 | 14336 | 256000 | GeGLU | 否 | 是 |

几个值得注意的差异：

- **Qwen2.5 的 KV 头只有 4 个**，GQA 比例 7:1，KV Cache 比 Llama 3 8B 更小；
- **Gemma 2 的 head_dim 是 256**，注意力 score 的形状和别的模型不同；
- **Gemma 2 层数 42**，比同尺寸模型深，单层更窄；
- **Mistral 的滑动窗口是 4096**，超出窗口的 token 不参与注意力；
- **DeepSeek-V3 用 MLA**，把 KV 压缩成低秩表示，KV Cache 比同规模的 GQA 还小 3–4 倍，比 MHA 小一个数量级以上，后面章节单独讲。

### 混合结构

有些模型不是纯 decoder-only：

- **MLA + MoE**：DeepSeek-V3，注意力用低秩压缩，FFN 用 256 个专家选 8 个；
- **局部/全局交替**：Gemma 2 每隔一层用滑动窗口，控制长上下文成本；
- **Mamba/Transformer 混合**：部分层换成状态空间模型，线性复杂度处理长序列；
- **MoE 稀疏 FFN**：Mixtral、Qwen-MoE、DeepSeek，参数量大但每 token 只激活一部分。

这些结构的共同点是：**config 字段会多出几组，加载器不能再假设固定的权重集合。**

## 7.7 Mapping Model Layers to Tensor Operations

把一层拆成算子，是写推理引擎的第一步。

### 一层有哪些算子

以 Llama 3 70B、decode、batch 1 为例：

| 算子 | 输入形状 | 输出形状 | FLOPs | 访存字节 |
| --- | --- | --- | --- | --- |
| RMSNorm | `[1, 8192]` | `[1, 8192]` | 约 16K | 48 KB |
| QKV GEMM | `[1, 8192]` | `[1, 10240]` | 168M | 168 MB |
| RoPE | `[1, 64, 128]` + `[1, 8, 128]` | 同左 | 约 18K | 37 KB |
| Attention | Q `[1,64,128]`，KV `[1,S,8,128]` | `[1, 8192]` | `4×64×S×128` | KV 字节 |
| O GEMM | `[1, 8192]` | `[1, 8192]` | 134M | 134 MB |
| 残差加法 | `[1, 8192]` | `[1, 8192]` | 8K | 48 KB |
| RMSNorm | `[1, 8192]` | `[1, 8192]` | 16K | 48 KB |
| Gate/Up GEMM | `[1, 8192]` | `[1, 57344]` | 939M | 939 MB |
| SiLU·Mul | `[1, 57344]` × 2 | `[1, 28672]` | 57K | 287 KB |
| Down GEMM | `[1, 28672]` | `[1, 8192]` | 470M | 470 MB |
| 残差加法 | `[1, 8192]` | `[1, 8192]` | 8K | 48 KB |

一层合计约 2.0G FLOPs（按 8K 上下文）、1.71 GB 权重读取。80 层就是约 160G FLOPs 和 137 GB。按 3.35 TB/s，仅权重读取就要 41 毫秒。

表格里的“读取字节”按 FP16 算，且假设每个权重只读一次。decode 时 batch 1，这正是实际情况——**计算量很小，访存量很大**。

### 融合机会

| 融合 | 节省 | 实现难度 |
| --- | --- | --- |
| add + RMSNorm | 一次 `[B,S,H]` 读写 | 低 |
| QKV 三个 GEMM 合成一个 | 两次 kernel launch、一次输出拼接 | 低 |
| RoPE + KV Cache 写入 | 一次 K 的读写 | 中 |
| gate/up GEMM 合成一个 | 一次中间张量 | 低 |
| SiLU·Mul + down GEMM | 一次 `[B,S,d_ff]` 读写 | 中 |
| Attention + O 投影 | 一次 `[B,S,H]` 读写 | 高 |

融合的收益在 decode 上主要是省 launch，在 prefill 上主要是省带宽。同一个融合，两种负载的收益来源不同。

### 一个 C++ 结构

```cpp
struct ModelConfig {
  int hidden_size;
  int num_layers;
  int num_heads;
  int num_kv_heads;
  int head_dim;
  int intermediate_size;
  int vocab_size;
  float rms_eps;
  float rope_theta;
  bool attention_bias;
  bool tie_word_embeddings;
};

struct LayerWeights {
  // 全部按 [out_features, in_features] 行主序
  const half* q_proj;        // [n_h * d, H]
  const half* k_proj;        // [n_kv * d, H]
  const half* v_proj;        // [n_kv * d, H]
  const half* o_proj;        // [H, n_h * d]
  const half* gate_proj;     // [d_ff, H]
  const half* up_proj;       // [d_ff, H]
  const half* down_proj;     // [H, d_ff]
  const half* input_norm;    // [H]
  const half* post_attn_norm;// [H]
};
```

前向：

```cpp
void decoder_layer(const ModelConfig& c, const LayerWeights& w,
                   Tensor& x, KVCache& kv, int pos) {
  auto h = rms_norm(x, w.input_norm, c.rms_eps);
  auto qkv = gemm(h, w.qkv_fused);            // [B, (n_h + 2 n_kv) * d]
  auto q = qkv.slice(0, c.num_heads * c.head_dim);
  auto k = qkv.slice(c.num_heads * c.head_dim,
                     c.num_kv_heads * c.head_dim);
  auto v = qkv.slice(c.num_heads * c.head_dim + c.num_kv_heads * c.head_dim,
                     c.num_kv_heads * c.head_dim);
  rope_apply(q, k, pos, c);
  kv.append(k, v);
  auto attn = paged_attention(q, kv, c);
  x = add(x, gemm(attn, w.o_proj));

  auto h2 = rms_norm(x, w.post_attn_norm, c.rms_eps);
  auto gate = gemm(h2, w.gate_proj);
  auto up = gemm(h2, w.up_proj);
  x = add(x, gemm(silu_mul(gate, up), w.down_proj));
}
```

这段代码对应 11 个 kernel。把 QKV 合并后是 1 个 GEMM，否则是 3 个；把 gate/up 合并后是 1 个，否则是 2 个。

### 内存布局

同一个矩阵，行主序 `[out, in]` 和列主序 `[in, out]` 在 GEMM 里对应不同的 transpose 标志。cuBLAS 默认列主序，PyTorch 的权重是行主序，调用时通常要传 `CUBLAS_OP_T`。

量化权重更复杂：GPTQ 按列分组，AWQ 按通道缩放，GGUF 的 Q4_K 有独立的 block 布局。kernel 必须按量化格式解释字节，不能当普通矩阵读。

### 和参考实现对数

写完一层，先别急着优化，先对 logits：

```python
ref = hf_model(input_ids).logits
out = engine.forward(input_ids)
diff = (ref - out).abs()
print(diff.max().item(), diff.mean().item())
```

BF16 下 `max diff` 在 0.01–0.05 是正常的，超过 0.1 说明有结构错误。FP32 参考实现应该能到 1e-5。

如果第一层就偏，检查 norm 的 eps 和 RoPE 排布；如果中间层开始偏，检查残差和 GQA 的头映射；如果只有最后几位偏，检查 lm_head 和 tied embedding。

## 本章小结

- “Transformer”是一族结构，模型之间的差异在 norm 位置、bias、激活、位置编码和权重共享上，加载器必须按 config 分支。
- 一个 decoder 层由 attention 和 FFN 组成；FFN 占每层参数的 80% 左右。
- Pre-norm 是现代 LLM 的默认选择；Gemma 2 的 sandwich norm 和 PaLM 的并行结构是重要变体。
- RMSNorm 要关注 eps 位置和 Gemma 的 `1 + weight`；求和必须在 FP32 里做。
- SwiGLU 用三个矩阵，`d_ff` 通常取 `8/3 H` 附近，但 Qwen 等模型会明显更大。
- RoPE 有交错和半分割两种排布，权重不能混用；位置 id 在 padding、分页和 prefix cache 下最容易出错。
- `head_dim` 不一定等于 `H / n_heads`，加载前要做一致性校验。
- 一层大约 11 个 kernel，decode 的瓶颈是权重读取，prefill 的瓶颈是 GEMM；融合和 CUDA Graph 是主要的优化手段。

## 延伸阅读

- Grattafiori et al., [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783), 2024.
- Touvron et al., [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288), 2023.
- Qwen Team, [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115), 2024.
- Gemma Team, [Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118), 2024.
- Jiang et al., [Mistral 7B](https://arxiv.org/abs/2310.06825), 2023.
- Black et al., [GPT-NeoX-20B: An Open-Source Autoregressive Language Model](https://arxiv.org/abs/2204.06745), 2022.
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864), 2021.
- Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202), 2020.
- Zhang & Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467), 2019.
- Hugging Face, [Transformers model implementations](https://github.com/huggingface/transformers/tree/main/src/transformers/models).
