---
chapter: 37
part: 7
title: Training Execution Pipeline
level: tbd
status: draft
prerequisites: [7, 25]
lab:
---

# Chapter 37. Training Execution Pipeline

一次 70B 模型的预训练要 6.3 ZFLOP：15 万亿 token，每个 token 前向加反向约 420 GFLOP。按 40% MFU 折算，这是 440 万个 H100 小时；用 1,024 张卡跑，需要半年。训练系统的每一个百分点效率，都对应着几天的时间和一笔真实的电费。

和推理相比，训练多了反向传播、梯度同步和优化器状态，显存压力大一个数量级，计算也不再是“读一遍权重”而是持续的高强度矩阵乘。这一章沿一次训练迭代的顺序，把数据、前向、反向、优化器和吞吐指标串起来，重点是每个阶段的资源消耗从哪里来。

## 37.1 Training Data Pipeline

语言模型的训练样本是 token 序列。原始语料经过清洗、去重、质量过滤、分词，最后打包成固定长度的序列。

打包（packing）是关键一步。如果每条样本单独成序列、短的补 padding，padding 部分同样要参与前向和反向，纯属浪费。打包把多条短文档首尾相接，填满一条 `S = 8192` 的序列，利用率从 60% 左右提到 95% 以上。

代价是跨文档注意力：后一篇文档的 token 会看到前一篇的内容。两种处理方式——要么用 attention mask 隔断（如 `cu_seqlens` 变长注意力），要么接受这种污染。多数大规模预训练接受它，因为影响很小；对质量敏感的场景则用 mask 隔断。

数据管道的吞吐要求可以算出来。70B 在 8×H100、40% MFU 下每秒处理

$$
\text{tokens/s} = \frac{7.9\ \text{PFLOPS} \times 0.4}{420\ \text{GFLOP}} \approx 7{,}500
$$

每秒 7,500 个 token，按 4 字节的 int32 存，带宽需求只有 30 KB/s。相比 GPU 的算力，这个数字微不足道：数据管道几乎不可能是 70B 训练的瓶颈。

小模型则相反。1B 模型每 token 只要 6 GFLOP，8 卡每秒能吃掉 50 万 token，带宽需求到 2 MB/s，还要加上 tokenize 的 CPU 开销。这时的瓶颈往往不是 GPU，而是数据加载进程和存储。

这也解释了一个常见现象：大模型训练集群的 CPU 配置看起来很“过剩”，因为真正吃 CPU 的是数据预处理、去重和分词，它们在上游离线完成，而不是在训练循环里。

## 37.2 Forward and Backward Pass

前向传播的计算量是 `2N` FLOPs/token，`N` 是参数量。这个 2 来自每个参数参与一次乘法和一次加法。

反向传播要算两样东西：对输入的梯度（继续往前传）和对权重的梯度。前者是 `2N`，后者也是 `2N`，合计 `4N`。所以一次完整的训练迭代是 `6N` FLOPs/token，是推理的 3 倍。

反向的代价不只在 FLOPs。链式法则需要前向的中间结果：每层的输入、attention 的 softmax 输出、归一化的统计量。这些激活必须保存到反向时使用，显存占用随层数和序列长度线性增长。

一个粗略的估算：`L` 层、hidden `h`、序列 `S`、batch `B`，每层每 token 的激活约 `10h` 到 `20h` 个元素。70B、`h = 8192`、`S = 8192`、`B = 1` 时，一层就是 1 GB 量级，80 层根本放不下。这就是激活重计算（activation checkpointing）存在的原因，Chapter 38 会展开。

反向还有一个结构特点：它是前向的逆序，且每层的计算依赖下一层的梯度。这种串行依赖让反向的并行度和前向相同，但也意味着流水线并行里前向和反向必须交错调度。

## 37.3 Gradient Computation

每个参数的梯度是损失对该参数的偏导数。框架用自动微分（automatic differentiation）构建计算图，反向时逐节点应用链式法则。

矩阵乘的梯度形式很规整。对 `Y = XW`：

$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^T, \qquad
\frac{\partial L}{\partial W} = X^T \frac{\partial L}{\partial Y}
$$

两个都是矩阵乘，计算量和前向同阶。这也是为什么反向的 FLOPs 恰好是前向的两倍——两次矩阵乘。

梯度的数值精度要单独考虑。BF16 的梯度直接累加会丢精度，尤其在小 batch 或梯度量级差异大时。常见做法是梯度在 FP32 中累加，或者用 BF16 存但用随机舍入（stochastic rounding）。

梯度裁剪（gradient clipping）在全局范数上做：把所有参数的梯度拼成一个向量，算 L2 范数，超过阈值就整体缩放。它在梯度累积时必须跨越所有 micro-batch——先累加、再裁剪、最后更新，否则每个 micro-batch 单独裁剪会改变有效学习率。

梯度里还会出现异常值。loss spike 往往伴随梯度范数的突然跳升；监控梯度范数比监控 loss 更早发现问题。

## 37.4 Optimizer Update

Adam 为每个参数维护一阶矩 `m` 和二阶矩 `v`：

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

其中 `\hat{m}`、`\hat{v}` 是偏差修正后的矩估计。AdamW 把权重衰减从梯度里分离出来，直接作用在参数上，这是目前 LLM 训练的标准做法。

优化器状态的显存是训练显存的大头：`m` 和 `v` 各 4 字节/参数，加上 FP32 master 权重 4 字节，7B 模型就是 84 GB。这也是 ZeRO 要分片优化器状态的原因。

单步更新的耗时相对固定：读参数、读梯度、读 `m`/`v`、写回，约 `16` 字节/参数。70B 就是 1.12 TB 的访存，按 3.35 TB/s 要 334 ms——比一步前向还长。实际实现会把多个参数的更新融合成少量 kernel，并把优化器步和下一步的前向重叠（如果框架支持）。

学习率调度通常是 warmup 加 cosine 或 linear decay。warmup 的存在是因为训练初期梯度方向不稳定，直接上大学习率容易发散。

## 37.5 Gradient Accumulation

显存放不下大 batch 时，把一个大 batch 切成若干 micro-batch，逐个前向反向，梯度累加，最后统一更新一次：

```python
optimizer.zero_grad()
for i, micro in enumerate(micro_batches):
    loss = model(micro) / accumulation_steps
    loss.backward()
clip_grad_norm_(model.parameters(), max_norm)
optimizer.step()
```

数学上它等价于一个大 batch，因为梯度是线性的。但有几个细节会破坏等价性：

- **损失归一化**：每个 micro-batch 的 loss 要除以累积步数，否则梯度被放大 `k` 倍；
- **梯度裁剪**：必须在所有 micro-batch 累加完之后做一次全局裁剪；
- **dropout 和随机性**：每个 micro-batch 的 dropout mask 不同，和大 batch 的一次采样不完全等价；
- **BatchNorm**：LLM 用 RMSNorm/LayerNorm，不受 batch 组成影响；如果是 BatchNorm，统计量会不同。

累积步数还影响吞吐。micro-batch 越小，GPU 利用率越低；但累积本身不增加通信，梯度只在最后同步一次（DDP 下），所以大累积步数对通信友好。

## 37.6 Mixed Precision Training

BF16 训练的标准配置是：前向和反向用 BF16，参数更新用 FP32。

| 项目 | 精度 | 原因 |
| --- | --- | --- |
| 前向/反向计算 | BF16 | Tensor Core 吞吐最高 |
| 激活 | BF16 | 省显存 |
| 梯度 | BF16 或 FP32 | FP32 更稳，BF16 更省 |
| Master 权重 | FP32 | 更新时的小量不能被舍入吃掉 |
| 优化器状态 | FP32 | 矩估计需要动态范围 |

为什么需要 master 权重？BF16 只有 8 位尾数，相对精度约 0.4%。学习率 1e-4 时，一次更新量可能小于参数的 BF16 分辨率，直接更新会“更新了个寂寞”。保留一份 FP32 参数专门接收更新，再把结果转成 BF16 供前向使用，就能避免这个问题。

FP16 还需要 loss scaling：梯度太小会下溢成 0，先把 loss 放大 `2^k` 倍，反向得到放大后的梯度，更新前再缩回来。BF16 的指数范围和 FP32 相同，不需要这一步——这是 BF16 在训练中胜过 FP16 的主要原因。

FP8 训练在 Hopper 上开始实用。它需要 per-tensor 或 per-block 的缩放因子，以及更细的数值监控；收益是 GEMM 吞吐翻倍，代价是调参复杂度和稳定性风险。

## 37.7 Training Throughput and MFU

模型 FLOPs 利用率（model FLOPs utilization，MFU）衡量实际算力占峰值的比例：

$$
\text{MFU} = \frac{6 N \cdot \text{tokens/s}}{N_{\text{gpu}} \cdot F_{\text{peak}}}
$$

`F_peak` 是单卡峰值。注意分子用的是 `6N`，只计模型计算，不计 attention 的额外 FLOPs、重计算和通信。

70B、8×H100、实测 7,500 tokens/s：

```text
MFU = 6 × 70e9 × 7,500 / (8 × 989e12) = 3.15e15 / 7.91e15 ≈ 39.8%
```

40% 左右是稠密模型训练的典型值。剩下 60% 去哪了：

| 损耗来源 | 典型占比 |
| --- | --- |
| 矩阵形状不理想（小 batch、非对齐维度） | 10%–20% |
| 通信等待（all-reduce、all-gather） | 5%–15% |
| 激活重计算的额外前向 | 5%–10% |
| 优化器步、数据加载、气泡 | 5%–10% |

MoE 模型的 MFU 通常更低，因为专家并行的 all-to-all 和负载不均；用 MFU 衡量 MoE 也不完全公平，它的激活参数量远小于总参数量。

另一个指标是硬件 FLOPs 利用率（HFU），分子把重计算的 FLOPs 也算进去。HFU 高于 MFU 说明重计算比例大；两个指标一起看，才能判断是 kernel 效率问题还是重计算配置问题。

训练的总时间可以直接估：

$$
T_{\text{train}} = \frac{6 N D}{N_{\text{gpu}} \cdot F_{\text{peak}} \cdot \text{MFU}}
$$

`D` 是训练 token 数。70B、`D = 15T`、1,024 张 H100、MFU 40%：

```text
T = 6 × 70e9 × 15e12 / (1024 × 989e12 × 0.4) ≈ 1.56e7 秒 ≈ 180 天
```

这个数字对得上业界的公开数据：Llama 3 70B 报告的训练量在数百万 GPU 小时量级。它也说明为什么 MFU 提升 5 个百分点值得投入一个团队——那是 20 多天。

## 本章小结

- 训练的每 token 计算量是 `6N`：前向 `2N`、反向 `4N`；推理只有 `2N`，所以训练的同 token 成本是推理的 3 倍。
- 打包把短样本拼成定长序列，显著提高利用率；跨文档注意力可以用 mask 隔断，多数预训练选择接受轻微污染。
- 大模型的数据管道几乎不可能成为瓶颈（70B 只需 30 KB/s），小模型则可能被 tokenize 和存储拖住。
- 反向需要保存激活，显存随层数、序列长度和 batch 线性增长，这是重计算和并行策略的直接动因。
- Adam 为每参数维护 `m`、`v` 和 FP32 master 权重，共 12 字节；7B 模型仅优化器状态就有 84 GB。
- 梯度累积在数学上等价于大 batch，但要求 loss 归一化、全局裁剪，并注意 dropout 的随机性差异。
- BF16 训练不需要 loss scaling，但必须保留 FP32 master 权重，否则小学习率的更新会被舍入吃掉。
- MFU = `6N × tokens/s / (卡数 × 峰值)`；70B 在 8×H100 上 7,500 tokens/s 对应约 40%，损耗主要来自形状、通信和重计算。
- 70B、15T token、1,024 张 H100、40% MFU 的训练时间约 180 天；MFU 每提高 5 个百分点，节省约 20 天。

## 延伸阅读

- Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), 2020.
- Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), 2022.
- Narayanan et al., [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473), SC 2021.
- Korthikanti et al., [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198), MLSys 2023.
- Chowdhery et al., [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311), 2022.
- Micikevicius et al., [Mixed Precision Training](https://arxiv.org/abs/1710.03740), ICLR 2018.
- Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), ICLR 2019.
- Llama Team, [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783), 2024.
