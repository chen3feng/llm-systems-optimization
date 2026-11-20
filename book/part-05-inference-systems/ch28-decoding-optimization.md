---
chapter: 28
part: 5
title: Decoding Optimization
level: tbd
status: draft
prerequisites: [25, 26]
lab:
---

# Chapter 28. Decoding Optimization

Batch 1、8K 上下文的 decode，一张 H100 每生成一个 token 要 42 毫秒，其中 GPU 的算力利用率不到 1%。瓶颈不在计算，而在每步都要把 140 GB 权重从 HBM 读一遍。既然一步只能出一个 token，能不能让一步多出几个？

投机解码（speculative decoding）就是这个思路：用小模型先猜几个 token，大模型一次前向并行验证。猜对的部分直接接受，猜错的丢掉重来。它不改变输出分布，却能把每步的有效 token 数提高到 2–4 个。这一章讲采样的数值细节、投机解码的接受与拒绝规则，以及各类 draft 结构的取舍。

## 28.1 Sampling Algorithms

模型每步输出一个长度为词表大小的 logits 向量，采样就是从它的 softmax 分布里取一个 token。完整流程是：

```text
logits (128K, FP32)
  → 除以 temperature
  → 可选：top-k / top-p / min-p 截断
  → softmax
  → 按概率采样
```

温度 `T` 缩放 logits：`T < 1` 让分布更尖锐，`T > 1` 更平坦，`T = 0` 退化为 argmax。截断类方法先把一部分候选的概率置零，再重新归一化。

数值上要注意两点。第一，softmax 必须先减去最大值，否则 FP16 下 `exp` 直接计算就会溢出；第二，归一化和采样要在 FP32 里做，FP16 的累积误差会让尾部概率失真。

GPU 上的采样不是一行 `torch.multinomial`。一个 batch 里每个请求的 temperature、top-k、top-p 都不同，kernel 要按请求分组处理。常见做法是每个请求一个 block，先在 shared memory 里做部分排序，再求累积和。词表 128K、batch 64 时，logits 本身就有 32 MB，采样 kernel 的访存相当于多读一遍全连接层的输出——这也是为什么采样必须留在 GPU 上，只回传 4 字节的 token id。

## 28.2 Greedy and Random Sampling

贪心采样取 `argmax`，输出确定，适合代码生成和事实问答。它的缺点是重复：一旦模型进入“the the the”这样的循环，argmax 会一直选同一个 token。

纯随机采样按 softmax 概率抽取，`T = 1` 时忠实于模型分布。它的问题相反：128K 词表的长尾里全是低质量 token，累积概率不小，抽中一次就可能毁掉整段输出。

两者之间的连续谱就是温度。实践中很少用纯随机：即使 `T = 1`，也会叠加 top-p 截断，把长尾切掉。

贪心的另一个用途是调试。同一输入、同一权重，贪心输出应当逐 token 可复现；如果两次运行不一致，说明 kernel 里有非确定性归约或 batch 组成影响了数值。

## 28.3 Top-k and Top-p

Top-k 保留概率最高的 `k` 个 token。`k = 1` 就是贪心，`k = 50` 是常见默认值。它对分布形状不敏感：分布很尖锐时，第 2 到第 50 个 token 的概率可能都接近零，等于白留了 49 个名额；分布很平坦时，`k = 50` 又可能切掉了大量有效概率。

Top-p（nucleus sampling，核采样）按累积概率截断：把 token 按概率降序排列，取累积和刚好超过 `p` 的最小集合。`p = 0.9` 时，保留的是“最可能的那部分，直到覆盖 90% 的概率质量”。分布尖锐时集合很小，平坦时集合自动变大，这正是 top-k 缺的适应性。

| 方法 | 参数 | 保留集合 | 优点 | 缺点 |
| --- | --- | --- | --- | --- |
| 贪心 | — | 1 | 确定、可复现 | 重复、死板 |
| 纯随机 | `T` | 全部 | 多样性最好 | 长尾噪声 |
| Top-k | `k` | 固定 `k` | 实现简单、开销可控 | 不适应分布形状 |
| Top-p | `p` | 动态大小 | 自适应 | 需要排序和累积和 |
| Min-p | `p_min` | 概率 ≥ `p_min × p_max` | 简单、随温度自适应 | 较新，生态支持少 |

实现上，top-p 不必完整排序。可以先取 top-`k`（如 64）再做累积和，因为 `p = 0.9` 时保留集合几乎不会超过 64 个；这一步把 `O(V log V)` 降到 `O(V)` 加一个小排序。

还有一个细节是“至少保留一个”。数值误差可能让所有概率之和略小于 `p`，或者 top-p 把唯一的候选也截掉；实现要在集合为空时回退到 argmax。

## 28.4 Speculative Decoding

投机解码的设定是：有一个小模型 `q`（draft）和一个大模型 `p`（target），两者词表相同。每一步：

1. draft 自回归生成 `γ` 个候选 token；
2. target 把这 `γ` 个 token 一次前向，得到每个位置的真实分布；
3. 从左到右逐个验证，接受或拒绝；
4. 第一个被拒绝的位置，用 target 的分布重新采样一个 token；
5. 全部接受时，target 的最后一个位置额外产生一个 token。

### 接受规则

对第 `i` 个候选 token `x`，draft 概率为 `q(x)`，target 概率为 `p(x)`。以概率 `min(1, p(x)/q(x))` 接受；拒绝时，从修正分布

$$
p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_{x'} \max(0, p(x') - q(x'))}
$$

中重新采样。这个规则保证最终输出分布严格等于 target 的分布——投机解码是精确加速，不是近似。

`p(x) ≥ q(x)` 时无条件接受，这符合直觉：target 比 draft 更想要这个 token，没有理由拒绝。

### 加速比

设单 token 接受率为 `α`。`γ` 个候选中期望接受的个数是

$$
E[\text{accepted}] = \frac{\alpha(1 - \alpha^\gamma)}{1 - \alpha}
$$

每个 target 前向产出 `E[\text{accepted}] + 1` 个 token（多出的 1 是拒绝位置或全部接受后的奖励 token）。

设一次 target 前向的耗时为 `T_t`，draft 生成 `γ` 个 token 的耗时为 `γ·T_d`，则每 token 的平均时间为

$$
T_{\text{spec}} = \frac{T_t + \gamma T_d}{E[\text{accepted}] + 1}
$$

以 70B、8×H100 为例：`T_t ≈ 13 ms`，draft 用 1B 模型，权重 2 GB，`T_d ≈ 0.6 ms`（2 GB / 3.35 TB/s，加 launch 开销），`γ = 5`，`α = 0.7`：

```text
E[accepted] = 0.7 × (1 - 0.7^5) / 0.3 ≈ 1.94
T_spec = (13 + 5 × 0.6) / 2.94 ≈ 5.4 ms
```

对比 13 ms，约 2.4 倍加速。`α` 降到 0.5 时，`E[accepted] ≈ 0.97`，`T_spec ≈ 8.1 ms`，只剩 1.6 倍。

加速比对上式里的三个参数都敏感：接受率、draft 成本、`γ`。`γ` 不是越大越好——draft 的成本线性增长，而接受数按 `1 - α^γ` 饱和。

### 系统代价

- **KV 回滚**：被拒绝 token 的 KV 已经写进块里，必须截断。分页布局下只要把 `num_tokens` 回退，尾块里的槽位下次会被覆盖，不需要释放块；
- **树形验证**：候选不止一条链时，KV 布局和 attention mask 都要支持树；
- **batch 内异构**：同一个 batch 里，有的请求在投机、有的不在，调度器要能混合；
- **采样正确性**：拒绝重采样需要每个位置独立的随机数，且不能用全局 RNG 的调用顺序影响结果。

## 28.5 Draft Model Architecture

draft 模型的选择标准只有一条：和 target 的分布足够接近，同时足够便宜。

| 方案 | 例子 | 接受率 | 成本 | 额外显存 |
| --- | --- | --- | --- | --- |
| 同系列小模型 | 70B + 1B/8B | 0.6–0.8 | 低 | 2–16 GB |
| 蒸馏 draft | 用 target 的输出训练 | 0.7–0.85 | 低 | 2–16 GB |
| n-gram / prompt lookup | 从上下文复制 | 0.3–0.6 | 几乎为零 | 无 |
| 额外预测头 | Medusa / EAGLE | 0.6–0.85 | 中 | 1–5% 参数 |

一般规律是：draft 越接近 target 的规模，接受率越高，但成本也越高。8B draft 配 70B target，接受率能到 0.8，但 8B 的 decode 本身要 2.4 ms/token（16 GB / 3.35 TB/s × 0.5 效率），`γ = 4` 时 draft 就花掉 10 ms，接近 target 的一步。这时加速比反而不如 1B draft。

Prompt lookup 不需要模型：直接用输入里最近出现的 n-gram 续写作为候选。它在摘要、改写、代码补全这类“大量复制输入”的任务上有效，在开放生成上几乎没有接受率。优点是零成本、零显存，适合作为兜底。

## 28.6 Tree-based Speculative Decoding

链式 draft 一次只能验一条路径。如果 draft 在某个位置不确定，可以同时展开多个候选，形成一棵树：

```text
        the
      /     \
   cat      dog
   / \      / \
 sat  ran  sat  ran
```

target 一次前向验证整棵树。关键是 attention mask：每个节点只能看到自己的祖先，不能看到兄弟节点。把树展平成序列、用自定义 mask 一次算完，就是 tree attention。

验证后取从根到叶的最长接受路径。树的宽度和深度由 draft 的置信度决定：高置信度处单链展开，低置信度处分叉。

代价是计算量。树有 `N` 个节点，target 就要算 `N` 个位置，而链式只有 `γ` 个。因为 decode 是带宽受限的，`N` 在几十以内时，target 的耗时增长远小于 `N` 倍；但上下文很长时，每个节点都要读一遍前缀 KV，树的成本会迅速上升。

## 28.7 Medusa and Multi-head Prediction

Medusa 不给 draft 模型，而是在 target 的最后隐状态上接几个额外的预测头，每个头预测未来第 `k` 个 token。头之间独立，所以可以并行给出多个候选，再组成树验证。

| 方案 | draft 形式 | 额外参数 | 训练成本 | 典型接受率 |
| --- | --- | --- | --- | --- |
| 独立 draft 模型 | 完整小模型 | 1–8B | 无需训练（若现成） | 0.6–0.8 |
| Medusa | 多个线性头 | < 1% | 需训练头 | 0.5–0.7 |
| EAGLE | 特征级自回归 | 1–2% | 需训练 | 0.7–0.85 |

Medusa 的头很便宜，但每个头独立预测，忽略了 token 之间的依赖，接受率不如自回归 draft。它的优势是部署简单：不增加一个模型，不增加一份 KV，头可以和主模型一起放进同一份权重。

## 28.8 EAGLE and Feature-level Drafting

EAGLE 的观察是：token 层面的 draft 要经过完整的 LM head 和 embedding，浪费在“把特征翻译成 token 再翻译回去”上。它改为在特征层做自回归：用 target 的倒数第二层隐状态作为输入，预测下一个隐状态，再统一过 LM head。

这样 draft 的每一步只是一个小的 Transformer 层，成本远低于完整小模型，接受率却更高，因为特征比 token 保留了更多信息。EAGLE-2 进一步把树的结构做成动态的，按 draft 的置信度分配宽度。

代价是耦合：draft 必须和 target 的隐空间对齐，换模型就要重新训练。它也不适合跨模型家族复用。

## 28.9 Multi-token Prediction

多 token 预测（multi-token prediction，MTP）在训练时就给模型加几个预测未来 token 的头，推理时这些头直接充当 draft。DeepSeek-V3 用的就是这种结构：每个 MTP 头预测第 `k` 个后续 token，串行或并行地产生候选。

它和 Medusa 的区别在于训练目标：MTP 是预训练的一部分，头在训练中见过完整数据分布，接受率高于事后加的头；Medusa 的头是在冻结的主模型上补训的。

MTP 的额外成本在训练侧：每个头都要算 loss，显存和计算增加 5%–10%。推理侧只在启用投机时才有开销。

## 28.10 Speculative Decoding Trade-offs

### 什么时候有效

投机解码的收益集中在“低 batch、带宽受限”的区间：

| 场景 | 单步瓶颈 | 投机收益 |
| --- | --- | --- |
| batch 1，短上下文 | 权重读取 | 高（2–3 倍） |
| batch 1，长上下文 | KV 读取 | 中，树形验证会放大 KV 流量 |
| batch 32 | 权重 + KV | 中（1.5–2 倍） |
| batch 128+ | 算力 | 低，验证的额外计算吃掉收益 |

batch 增大时，target 每步本来就能摊薄权重读取，投机的相对收益下降；而验证多个位置增加了算力和 KV 读取，成本上升。两条曲线在某个 batch 附近交叉，超过这个点，投机解码反而更慢。

### 正确性

投机解码的输出分布必须和原始采样完全一致，否则就是“近似加速”，评测结果会和基线对不上。验证方法是固定随机种子，对比开、关投机的输出分布，或者用大量样本做卡方检验。

贪心解码下的验证更简单：只要 target 对候选的 argmax 和接受路径一致，输出就应当逐 token 相同。

### 工程集成

- draft 的 KV 单独管理，通常用更小的块或直接连续分配；
- 调度器要为投机请求预留 `γ + 1` 个 KV 槽位；
- 前缀缓存对 draft 和 target 都要生效，否则 draft 的第一步就要重算整个前缀；
- 拒绝后的回滚要幂等，不能和取消、抢占的路径冲突；
- 指标的统计口径要区分“target 前向次数”和“生成 token 数”，否则吞吐会被高估。

### 收益的边界

投机解码不减少总 FLOPs，它用更多的计算换更少的串行步数。当系统已经算力饱和时，这个交换是负收益；当系统在等权重读取时，它是正收益。判断标准很简单：看 GPU 的算力利用率。低于 20% 时值得试，高于 60% 时基本无效。

## 本章小结

- 采样是 logits 经温度缩放、截断、softmax 后抽样的过程；数值上要先减最大值，归一化和抽样用 FP32。
- 贪心确定但易重复，纯随机多样但引入长尾噪声；top-k 固定保留数，top-p 按累积概率自适应，实际常用两者组合。
- 投机解码用小模型猜、大模型并行验证，接受规则 `min(1, p/q)` 加修正分布重采样，保证输出分布不变。
- 加速比由接受率 `α`、draft 成本 `T_d` 和猜测长度 `γ` 决定；70B 配 1B draft、`α = 0.7`、`γ = 5` 时约 2.4 倍，`α = 0.5` 时降到 1.6 倍。
- draft 越接近 target 接受率越高但越贵；prompt lookup 零成本、适合复制型任务，Medusa/EAGLE 用少量参数换取更高的接受率。
- 树形验证用自定义 attention mask 一次算完整棵树，代价是每个节点都要读前缀 KV，长上下文时成本放大。
- EAGLE 在特征层做自回归，比 token 级 draft 更便宜也更准；MTP 把预测头放进预训练，接受率更高。
- 投机解码的收益集中在低 batch、算力空闲的区间；batch 增大或算力饱和后，额外计算会吃掉收益。
- 工程上要处理 KV 回滚、树形布局、batch 内异构和前缀缓存；正确性用固定种子对比输出分布来验证。

## 延伸阅读

- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), ICML 2023.
- Chen et al., [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318), 2023.
- Cai et al., [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774), 2024.
- Li et al., [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077), ICML 2024.
- Miao et al., [SpecInfer: Accelerating Generative LLM Serving with Tree-based Speculative Inference](https://arxiv.org/abs/2305.09781), ASPLOS 2024.
- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), 2024.
- Holtzman et al., [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751), ICLR 2020.
- vLLM, [Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html).
