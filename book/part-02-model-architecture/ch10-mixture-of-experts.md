---
chapter: 10
part: 2
title: Mixture of Experts
level: tbd
status: draft
prerequisites: [7, 9]
lab:
---

# Chapter 10. Mixture of Experts

MoE 做的事情可以用一句话概括：把模型的参数量和每个 token 的计算量解耦。

一个 671B 参数的模型，每个 token 只经过 37B 参数的计算。多出来的 634B 不是备份，是 specialization——不同的专家在不同的数据分布上变得擅长，路由器决定当前 token 该找谁。

这带来三个直接后果：同样的计算预算能装下更大的容量；显存必须装下全部专家；通信成为新的瓶颈。这一章讲 MoE 的结构、路由、调度和成本。

## 10.1 Dense versus Sparse Models

稠密模型里，每个 token 都经过全部参数。参数量 `N`，每 token 的 FLOPs 约 `2N`，没有例外。

MoE 把 FFN 拆成 `E` 个专家，每个 token 只选其中 `k` 个：

$$
\text{FLOPs}_{\text{token}} \approx 2 N_{\text{active}}, \quad N_{\text{active}} \ll N
$$

但显存要装下全部 `N`。**计算按激活参数算，显存按总参数算。**

| 模型 | 总参数 | 激活参数 | 比例 | 专家 / top-k |
| --- | --- | --- | --- | --- |
| Switch-C | 1.6T | 约 1/2048 | 极小 | 2048 / 1 |
| GShard | 600B | 约 1/1024 | 极小 | 2048 / 2 |
| Mixtral 8x7B | 46.7B | 12.9B | 28% | 8 / 2 |
| Grok-1 | 314B | 约 1/4 | 25% | 8 / 2 |
| DeepSeek-V2 | 236B | 21B | 9% | 160 / 6 |
| DeepSeek-V3 | 671B | 37B | 5.5% | 256 / 8 |

这张表说明 MoE 的谱系很宽：Switch 走极端稀疏，Mixtral 走温和稀疏，DeepSeek 用大量细粒度专家。

### 为什么有效

同样的 FLOPs，稠密模型只能有 `N` 个参数；MoE 可以有 `E/k` 倍的参数。更多的参数意味着更大的记忆容量和更强的 specialization 能力。

但这不是免费的：

- 显存要装下所有专家，即使大部分 token 用不到；
- 路由可能出错，把 token 送给不合适的专家；
- 专家分布不均，热专家排队，冷专家闲置；
- 多卡部署时，token 要在设备之间搬运。

MoE 的本质是用显存和通信换容量。

## 10.2 Expert Networks

一个专家就是一个 FFN。现代 MoE 的专家通常用 SwiGLU：

$$
\text{Expert}_i(x) = W_{\text{down}}^i\left(\text{SiLU}(W_{\text{gate}}^i x) \odot (W_{\text{up}}^i x)\right)
$$

每个专家有三个矩阵，参数量 `3 H d_ff`。

### 专家大小

| 模型 | `H` | 专家 `d_ff` | 每专家参数 | 专家数 |
| --- | --- | --- | --- | --- |
| Mixtral 8x7B | 4096 | 14336 | 176M | 8 |
| DeepSeek-V3 | 7168 | 2048 | 44M | 256 + 1 |

Mixtral 用少而大的专家，DeepSeek 用多而小的专家。后者的好处是组合空间更大：256 选 8 的组合数是 `C(256,8) ≈ 4×10^14`，而 8 选 2 只有 28 种。

### 共享专家

DeepSeek 的每个 MoE 层有一个共享专家，它对所有 token 都激活：

$$
y = \sum_{i \in \text{top-}k} g_i \text{Expert}_i(x) + \text{Expert}_{\text{shared}}(x)
$$

共享专家承担通用知识，路由专家负责差异化。这样可以减少路由专家之间的冗余——不用每个专家都学一遍基础语法。

### 稠密层

不是所有层都必须是 MoE。DeepSeek-V3 的前 3 层是稠密 FFN，后面 58 层才是 MoE。浅层处理的是低级的 token 表示，专家分工的意义不大。

## 10.3 Router and Gating Functions

路由器是一个小型线性分类器：

$$
u = x W_r, \quad W_r \in \mathbb{R}^{H \times E}
$$

`u` 是每个专家的分数。路由器参数量只有 `H × E`，DeepSeek-V3 是 `7168 × 256 ≈ 1.8M`，和 671B 比起来可以忽略。

### Softmax 与 Sigmoid

最直接的做法是 softmax 后取 top-k：

$$
g_i = \frac{e^{u_i}}{\sum_j e^{u_j}}
$$

DeepSeek 用的是 sigmoid：

$$
s_i = \sigma(u_i)
$$

然后只对选中的 `k` 个分数做归一化。sigmoid 的好处是每个专家独立判断，不像 softmax 那样互相竞争；在专家数很多时，softmax 会让分数被稀释。

### 门控输出

选中集合 `S = top-k(u)`，门控权重是：

$$
g_i = \frac{s_i}{\sum_{j \in S} s_j}, \quad i \in S
$$

最终输出：

$$
y = \sum_{i \in S} g_i \text{Expert}_i(x)
$$

门控权重决定专家输出的混合比例。它必须可微，否则路由器无法训练。

### 训练时的噪声

训练时给 router logits 加高斯噪声，可以鼓励探索、防止早期塌缩：

$$
u' = u + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

推理时不加噪声。噪声尺度要随训练衰减，否则会一直干扰收敛。

## 10.4 Top-k Routing

`k` 的选择直接决定计算量和质量。

| `k` | 代表 | 特点 |
| --- | --- | --- |
| 1 | Switch Transformer | 计算最省，路由不稳定，需要强辅助损失 |
| 2 | GShard、Mixtral | 质量和成本的常见平衡点 |
| 6 | DeepSeek-V2 | 细粒度专家，组合更多 |
| 8 | DeepSeek-V3 | 256 选 8，容量大，通信多 |

### 容量因子与丢 token

每个专家能处理的 token 数有上限：

$$
C = \text{capacity\_factor} \times \frac{kT}{E}
$$

`T` 是 batch 内的 token 总数，`k` 是每个 token 的专家数。超过容量的 token 被丢弃，直接走残差：

$$
y = x
$$

容量因子通常取 1.0–1.25。太小会丢很多 token，太大会浪费显存和计算。

丢 token 在训练时是一种正则化，在推理时是质量损失。现代实现更倾向 dropless：不设硬上限，按实际负载动态分配，代价是 kernel 要处理变长形状。

### Expert Choice

反过来让专家选 token：每个专家选分数最高的 `C` 个 token。这样每个专家的负载天然均衡，但每个 token 得到的专家数不固定——热门 token 可能被很多专家处理，冷门 token 一个都没有。

Expert Choice 在训练上有效，推理时实现复杂，实际部署不如 token choice 常见。

### 辅助损失无关的均衡

DeepSeek-V3 用一个可学习的偏置 `b_i` 来调节路由：

$$
s'_{i} = s_i + b_i
$$

用 `s'` 做 top-k 选择，但门控值仍然用原始的 `s_i`。训练时根据每个专家的负载调整 `b_i`：负载高的减一点，负载低的加一点。

这样不需要辅助损失，避免了“为了均衡牺牲语言建模质量”的问题。这是 DeepSeek 对 MoE 训练的一个重要改进。

## 10.5 Token Dispatch and Combine

路由决定“谁去哪”，dispatch 负责把 token 送过去，combine 负责把结果加权合并。

### 单卡实现

单卡上，所有专家都在同一块 GPU 上，dispatch 是一次 permutation：

```python
# x: [T, H]
logits = x @ W_r                       # [T, E]
scores = sigmoid(logits)
topk_scores, topk_idx = scores.topk(k, dim=-1)   # [T, k]

flat_idx = topk_idx.flatten()          # [T*k]
order = flat_idx.argsort()             # 按专家排序
sorted_x = x[order // k]               # [T*k, H]

# 每个专家处理自己的一段
counts = torch.bincount(flat_idx, minlength=E)
outs = grouped_gemm(sorted_x, counts, expert_weights)

# 合并
y = torch.zeros_like(x)
y.index_add_(0, order // k, outs * topk_scores.flatten()[order])
```

核心是 `grouped_gemm`：一次调用里，对每个专家执行不同大小的 GEMM。CUTLASS 的 grouped GEMM 和 Triton 都能做，但需要专家权重按连续内存排列。

### 多卡：All-to-All

专家并行（EP）下，每个 GPU 只放一部分专家。token 要发给专家所在的 GPU：

```text
GPU 0: token -> GPU 3 (expert 17)
GPU 0: token -> GPU 5 (expert 42)
...
计算完成后，结果再发回 GPU 0
```

这是一个 all-to-all：每张卡向其他卡发送不同的 token 子集，然后接收属于自己的 token。

### Dropless 的动态形状

不丢 token 时，每个专家收到的 token 数不等。kernel 必须支持：

- 变长的 batch；
- 每个专家的偏移量在运行时才知道；
- 负载不均时的 straggler。

这比固定形状的 GEMM 难写得多。DeepSeek、vLLM、TensorRT-LLM 都为此做了专门的 fused MoE kernel。

### 融合

一个高效的 MoE 层通常把这些步骤融合：

1. router GEMM；
2. top-k + 归一化；
3. permutation；
4. grouped GEMM（gate/up）；
5. SiLU·Mul；
6. grouped GEMM（down）；
7. unpermute + 加权求和。

融合减少中间张量的 HBM 往返。decode 时每个 token 的 hidden 只有几 KB，但来回搬几次也会累积成可观的带宽。

## 10.6 Expert Load Balancing

没有约束的话，路由器会塌缩：少数专家被大多数 token 选中，其余专家收不到梯度，逐渐退化。

### 辅助损失

Switch Transformer 的辅助损失：

$$
L_{\text{aux}} = \alpha N \sum_{i=1}^{N} f_i P_i
$$

其中：

- `f_i`：分到专家 `i` 的 token 比例；
- `P_i`：路由器给专家 `i` 的平均概率；
- `N`：专家数；
- `α`：系数，通常 0.01。

当所有专家负载均匀时，`Σ f_i P_i = 1/N`，损失最小。负载集中时，`f_i` 和 `P_i` 同时变大，损失上升。

### z-loss

路由器 logits 的绝对值变大会导致 softmax 数值不稳定。z-loss 惩罚 logits 的平方：

$$
L_z = \frac{1}{T} \sum_{t=1}^{T} \left(\log \sum_{i=1}^{E} e^{u_{t,i}}\right)^2
$$

系数通常很小，如 0.001。

### Jitter 与初始化

- 训练时给 logits 加噪声；
- 路由器权重用小方差初始化；
- 路由器的学习率单独设置，避免更新过快。

### 监控指标

| 指标 | 含义 | 健康范围 |
| --- | --- | --- |
| 最大/平均负载 | 最热专家有多热 | 小于 2–3 |
| 丢弃率 | 超过容量的 token 比例 | 小于 1% |
| 路由熵 | 分布的均匀程度 | 接近 `log k` |
| 专家梯度范数 | 是否有专家收不到梯度 | 都大于 0 |

负载不均衡的代价是双重的：热专家排队，冷专家浪费显存；整体吞吐由最慢的专家决定。

## 10.7 Auxiliary Loss and Routing Stability

辅助损失和语言建模目标是有冲突的。`α` 太大，模型会为了均衡把 token 送给不合适的专家，质量下降；`α` 太小，路由塌缩。

### 常见做法

- `α = 0.01` 是 Switch 的默认值；
- 训练后期可以衰减 `α`；
- 用 z-loss 单独稳定 logits；
- 用 bias 方法替代辅助损失（DeepSeek-V3）。

### 路由塌缩的早期信号

- 路由熵快速下降；
- 某些专家的梯度长期为 0；
- 最大负载比超过 5；
- 验证集 loss 在训练中期突然变差。

出现这些信号时，先检查学习率和初始化，再调 `α` 或容量因子。加专家数量不一定能解决问题——如果路由器已经塌缩，新专家也收不到 token。

### 稳定性与稀疏性的权衡

越稀疏（`k/E` 越小），路由越难学，越需要均衡机制。Switch 的 top-1 需要很强的辅助损失；DeepSeek 的 top-8/256 相对容易，因为每个 token 有更多选择。

## 10.8 Expert Parallelism Implications

MoE 的显存需求把专家推到多卡上。

### 放置策略

- **专家并行（EP）**：每张卡放一部分专家，token 通过 all-to-all 访问远程专家；
- **张量并行（TP）**：每个专家内部再切矩阵，通信频繁，通常不用于 MoE；
- **数据并行（DP）**：每张卡放全部专家，显存装不下。

大规模 MoE 基本是 EP。DeepSeek-V3 的 671B 参数需要跨节点部署，EP 组可能覆盖几十张卡。

### 节点限制路由

如果每个 token 的 8 个专家随机分布在 32 张卡上，all-to-all 会非常昂贵。DeepSeek 的做法是限制每个 token 最多去 4 个节点，把通信控制在可接受范围。

这是一个典型的系统-算法协同设计：路由策略不只考虑质量，也考虑拓扑。

### 热专家复制

如果某个专家持续过热，可以在多张卡上放副本，把流量分散。代价是显存和同步。

### 故障域

EP 下，一张卡挂掉，它上面的专家就不可用。整个 EP 组可能都要重启。这是 MoE 部署比稠密模型更脆弱的地方。

### 通信与计算的重叠

All-to-all 可以和 attention、dense 层或其他专家的计算重叠。DeepSeek 的 DualPipe 和类似技术把通信切成小块，塞进计算的空隙。不能重叠的通信会直接吃掉扩展性。

## 10.9 MoE Computation and Communication Costs

### 计算

每 token 的 FLOPs 约：

$$
2 N_{\text{active}} = 2 \left( N_{\text{attn}} + k \cdot N_{\text{expert}} + N_{\text{shared}} \right)
$$

DeepSeek-V3：`2 × 37B = 74 GFLOPs/token`，和 37B 稠密模型相当。

### 显存

权重显存按总参数算：

| 模型 | FP16 | FP8 | INT4 |
| --- | --- | --- | --- |
| Mixtral 46.7B | 93 GB | 47 GB | 23 GB |
| DeepSeek-V3 671B | 1.34 TB | 671 GB | 336 GB |

DeepSeek-V3 的 FP8 权重是 671 GB，单机 8×80 GB（640 GB）装不下，必须跨节点。

### 带宽

decode 时每个 token 只读激活专家的权重。batch 1 时读 `N_active`，batch 增大后，每个专家被多个 token 共享，总读取量趋近 `N_total`，但摊到每个 token 上下降。

这就是 MoE 在大 batch 下效率高的原因，也是小 batch 下浪费的原因：小 batch 只用了少数专家，其余专家的显存闲着。

### 通信

每个 token 每层的 all-to-all 数据量约：

$$
2 \times D \times H \times \text{bytes}
$$

`D` 是目的节点数（受 top-k 和节点限制约束），系数 2 是 dispatch 和 combine 各一次。

DeepSeek-V3 风格：`k = 8`、`H = 7168`、BF16、`D = 4`：

$$
2 \times 4 \times 7168 \times 2 \approx 115 \text{ KB/token/layer}
$$

58 个 MoE 层约 6.7 MB/token。按 8 卡、每卡 400 Gb/s（50 GB/s）的聚合带宽 400 GB/s 估算，通信吞吐上限约 6 万 token/s；同配置的计算上限约 10 万 token/s。

这是一个量级估算，实际取决于拓扑、重叠和负载均衡。但它说明：**MoE 的通信不是免费的，节点限制和重叠是必需项。**

### Roofline

- **Prefill**：大 batch、大 GEMM，算力受限；MoE 的 grouped GEMM 效率通常低于稠密 GEMM。
- **Decode**：小 batch 时带宽受限，读激活专家；大 batch 时读全部专家，但摊薄后效率提高。
- **All-to-all**：跨节点时可能成为主要瓶颈。

## 本章小结

- MoE 把参数量和每 token 计算量解耦；计算按激活参数算，显存按总参数算。
- 每个专家是一个 FFN，通常用 SwiGLU；共享专家承担通用知识，路由专家负责差异化。
- 路由器是小型线性分类器，输出 top-k 的门控权重；sigmoid 和 softmax 各有适用场景。
- `k` 决定计算量和质量；容量因子和丢 token 是训练与推理的权衡。
- Dispatch 是 permutation + grouped GEMM；多卡下变成 all-to-all，动态形状和负载均衡是主要难点。
- 辅助损失、z-loss、噪声和 bias 方法用于防止路由塌缩；DeepSeek 的 bias 方法避免了辅助损失与语言建模的冲突。
- 专家并行把专家分到多卡，节点限制路由和通信重叠是规模化的关键。
- MoE 的收益是容量，代价是显存、通信和路由复杂性。

## 延伸阅读

- Jacobs et al., [Adaptive Mixtures of Local Experts](https://doi.org/10.1162/neco.1991.3.1.79), 1991.
- Shazeer et al., [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538), 2017.
- Lepikhin et al., [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668), 2020.
- Fedus et al., [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961), 2021.
- Zhou et al., [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368), 2022.
- Jiang et al., [Mixtral of Experts](https://arxiv.org/abs/2401.04088), 2024.
- DeepSeek-AI, [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434), 2024.
- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), 2024.
- Gale et al., [Megablocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841), 2022.
