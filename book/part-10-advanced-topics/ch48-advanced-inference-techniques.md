---
chapter: 48
part: 10
title: Advanced Inference Techniques
level: tbd
status: draft
prerequisites: [28, 29, 47]
lab:
---

# Chapter 48. Advanced Inference Techniques

前面各章讲的是“把一次生成做快”。这一章讲另一类优化：**改变生成的方式**——多算几步换更高的正确率，少算几步换更低的延迟，或者干脆换一个模型来回答。

这些技术的共同点是：它们都不改变单个 kernel 的效率，而是改变请求在时间、模型和计算量上的分配。它们的收益要用“单位成本下的有效质量”来衡量，而不是 tokens/s。

## 48.1 Test-time Compute Scaling

同一个模型，多给它一些推理时间，答案会更准。这不是玄学，而是有明确的机制：采样多次取多数、让模型展开更长的推理链、或者用验证器筛掉错误答案。

三种常见形式：

| 方法 | 做法 | 成本 | 适用 |
| --- | --- | --- | --- |
| 自一致性 | 采样 N 次，投票 | N 倍 decode | 有唯一答案的任务 |
| Best-of-N | 采样 N 次，奖励模型选最好 | N 倍 decode + N 次打分 | 开放式生成 |
| 长链推理 | 生成更长的思维链 | 数倍输出 token | 数学、逻辑、代码 |

成本是线性的，收益是次线性的。经验数据是：N 从 1 增到 16，数学题的准确率可能提升 10–20 个点；再往上，边际收益迅速下降。所以在给定预算下，存在一个最优点——继续增加 N 不如换一个更大的模型。

从系统角度看，test-time scaling 改变了流量画像：

- **decode 占比上升**：一次请求的输出从 500 token 变成 5,000 token，KV Cache 和 decode 时间同比增长；
- **batch 变大**：N 个样本可以并行，实际是在同一 batch 里跑多个序列；
- **调度变复杂**：投票要等所有样本完成，最慢的那个决定延迟，尾延迟的影响被放大。

一个实用的优化是**早停**：投票结果已经收敛（比如 12 票对 2 票）就停止剩余样本。这能把平均成本降 20%–30%，代价是判定收敛本身需要一次同步。

## 48.2 Adaptive Computation

不是所有 token 都同样难。简单的 token 可能一层都不用多想，难的 token 需要更多计算。自适应计算让模型按需分配。

主要形式：

- **早退**（early exit）：在某些层插入分类器，判断“这个 token 已经够确定了”，直接跳到输出；
- **动态深度**：根据输入难度选择用多少层；
- **token 剪枝**：跳过对输出影响小的 token 的计算；
- **动态宽度**：MoE 的专家选择本身就是一种自适应计算。

早退在 batch 推理里很难落地。GPU 是 SIMT 的，一个 batch 里有的序列在第 20 层退出、有的到第 40 层，执行时必须按最长的走，早退的序列只是空转。要真正省算力，得把退出的序列从 batch 里摘出去——这会让 batch 变小、利用率下降，收益可能被抵消。

所以自适应计算目前的实际收益主要在**研究场景和单请求场景**。批量服务里，更现实的做法是模型级的路由（下一节），而不是层级的早退。

## 48.3 Dynamic Model Selection

在推理时决定用哪个模型，比训练时决定更灵活。

**按难度路由**：用一个小分类器或小模型本身，判断请求需要多大模型。分类器的成本很低（几十毫秒），相比省下的大模型调用可以忽略。

**按置信度路由**：小模型先答，如果 logprob 低、或者输出触发了某些信号（重复、格式错误、拒绝回答），升级到大模型。

**按任务类型路由**：代码请求去代码模型，长文档去长上下文模型，多模态请求去视觉模型。这不需要判断难度，只需要分类。

路由的收益取决于流量分布。如果 80% 的请求是简单的（比如闲聊、格式转换），小模型能覆盖，成本能降一个数量级；如果 80% 是复杂推理，路由几乎不省成本，还多了一次判断的延迟。

路由的系统挑战是**容量规划**。两个模型要各自预留容量，而流量在两个池之间的分配是动态的。如果路由器的判断和实际负载不匹配，会出现一个池打满、另一个池空闲的情况。实践中通常按历史比例固定分配，只在偏差大时调整。

## 48.4 Model Cascading

级联是路由的特例：按顺序尝试，不满足条件就升级。它和 47.3 讲的级联是同一件事，这里补充系统层面的设计。

一个三级级联的例子：

```text
7B  →  置信度低 / 格式错  →  70B  →  验证失败  →  人工或更强的模型
```

级联的关键参数是**升级率**和**验证成本**：

$$
\text{总成本} = C_1 + r_1 (C_2 + r_2 C_3)
$$

其中 `r` 是升级率。`C_1` 很小、`r_1` 在 20%–30% 时，总成本接近 `C_1 + 0.25 C_2`，相比全部走 `C_2` 省 75% 以上。

但级联的延迟是**串行**的：小模型的 200 ms 加大模型的 800 ms，最坏情况 1 秒。如果 SLO 是 800 ms，级联直接出局。缓解办法是并行发起（小模型和大模型同时跑，小模型先返回就取消大模型），代价是 GPU 资源翻倍——省下的成本又还回去了。

所以级联适合**离线批处理**和**对延迟不敏感的交互**（比如代码审查、文档摘要），不适合实时对话。

## 48.5 Advanced Speculative Execution

投机解码在 Chapter 28 讲过基础形式。这里看几种进阶变体及其系统含义。

**多草稿模型**：用多个不同的小模型并行起草，取最长的一致前缀。草稿成本乘以模型数，但接受长度也增加。实践中收益有限，因为草稿模型之间的多样性不足。

**树形投机**：草稿模型生成一棵树（每个节点多个候选），目标模型一次验证整棵树。验证时用树形 attention mask，一次前向验证多个路径。收益来自“一次前向验证多个候选”，代价是 KV Cache 要按树的结构组织，实现复杂。

**前瞻解码**（lookahead decoding）：不用草稿模型，用 Jacobi 迭代同时猜测多个位置，再验证。它不需要额外模型，但收敛慢，实际加速比通常低于 1.5 倍。

**Medusa / EAGLE**：在目标模型上训练额外的解码头或特征层。EAGLE 在特征层预测，比 token 层的草稿更准，接受率能到 70%–80%。

系统层面的关键约束是**batch 大小**。投机解码的收益随 batch 增大而下降：batch=1 时 decode 是访存受限，多算几个 token 几乎免费；batch=64 时 GPU 已经接近计算受限，验证多个候选反而拖慢。经验交叉点在大约 batch 16–32，超过这个规模，投机解码的净收益为负。

还有一个容易被忽略的成本：**KV Cache 的额外占用**。投机验证时，被拒绝的 token 的 KV 也要先算出来再丢弃，这部分显存和带宽是纯开销。树形投机的 KV 占用是线性的几倍，在长上下文场景下可能直接 OOM。

## 48.6 Long-context Optimization

上下文从 8K 涨到 128K，计算量和 KV Cache 都涨 16 倍。这不是简单的“加显存”能解决的。

**计算侧**：attention 的 FLOPs 随 `S²` 增长。128K 的 prefill，attention 部分约 22 PFLOP（按因果掩码算一半），而线性层是 `2 × 70e9 × 131072 ≈ 18` PFLOP——序列到了这个长度，attention 已经超过线性层，成为计算的主导。

**显存侧**：KV Cache 按每 token 320 KiB 算，128K 上下文是 40 GB，单请求就占掉半张卡。

主要手段：

| 手段 | 原理 | 代价 |
| --- | --- | --- |
| 环形 attention | 序列切块，KV 在设备间轮转 | 通信量随设备数增长 |
| 稀疏 attention | 只算部分 token 对 | 需要训练适配，召回损失 |
| 线性 attention / SSM | 复杂度降到 `O(S)` | 表达能力受限 |
| KV 压缩（MLA、GQA） | 减少每 token 的 KV 字节 | 精度损失 |
| KV 分层存储 | 冷 KV 放 CPU/SSD | 访存延迟增加 |

**环形 attention**（ring attention）是把序列维切到多个设备上，每个设备算自己那块的 query，KV 按环传递。它的通信量和 FlashAttention 的计算重叠，理论上可以做到“通信隐藏在计算里”。但实际的通信量是 `O(S × h × L × 设备数)`，128K 序列、8 设备下是几百 GB 的传输，对互联带宽要求很高。

**KV 分层存储**更适合服务场景：把长时间不访问的 KV 块换到 CPU 内存或 NVMe，需要时再换回。这和操作系统换页是一个思路，关键是预测哪些块会被再次访问——attention sink（开头几个 token 总是被关注）和最近窗口是可靠的启发式。

长上下文的真实瓶颈往往不是计算，而是**调度**：一个 128K 的请求会占用大量 KV 块，把并发从 150 压到 5。所以生产上常见的做法是长上下文请求单独资源池，或者限制单实例上长请求的数量。

## 本章小结

- Test-time scaling 用线性成本换次线性收益：N 从 1 到 16 可能提升 10–20 个点，再往上不如换更大的模型；早停能省 20%–30%。
- 自适应计算在 SIMT 的 batch 推理里收益有限，早退的序列仍需按最长路径执行；模型级路由比层级早退更现实。
- 动态模型选择按难度、置信度或任务类型分流，收益取决于流量分布；路由器会增加一次判断延迟，并让容量规划变复杂。
- 级联的延迟是串行的，适合离线和对延迟不敏感的场景；并行发起能降延迟但抵消成本收益。
- 投机解码的收益随 batch 增大而下降，交叉点约在 batch 16–32；树形投机和 Medusa/EAGLE 提高接受率，但 KV 占用成倍增加。
- 长上下文的主要成本是 `O(S²)` 的 attention 和线性增长的 KV；环形 attention 用通信换显存，KV 分层存储用延迟换容量。
- 长上下文的真实瓶颈常是调度：一个 128K 请求能把并发从 150 压到 5，需要单独的资源池。

## 延伸阅读

- Wei et al., [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903), NeurIPS 2022.
- Wang et al., [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171), ICLR 2023.
- Snell et al., [Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314), 2024.
- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), ICML 2023.
- Cai et al., [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774), ICML 2024.
- Li et al., [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077), ICML 2024.
- Liu et al., [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889), ICLR 2024.
- DeepSeek-AI, [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434), 2024.
