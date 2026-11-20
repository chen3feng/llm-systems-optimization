---
chapter: 38
part: 7
title: Training Memory Optimization
level: tbd
status: draft
prerequisites: [34, 37]
lab:
---

# Chapter 38. Training Memory Optimization

一张 80 GB 的 H100 装不下 7B 模型的训练：权重 14 GB、梯度 14 GB、优化器状态 84 GB，加起来 112 GB，还没算激活。而到了 70B，仅优化器状态就是 840 GB。

训练显存由四部分组成，每一部分都有对应的优化手段：分片、卸载、重计算、低精度。这一章先把四笔账算清楚，再讨论每种手段省下多少、付出什么。选择哪种组合，决定了能用多少张卡、跑多长的序列。

## 38.1 Parameter Memory

参数按精度存储。BF16 是 2 字节/参数，7B 模型 14 GB，70B 模型 140 GB。

如果同时保留 FP32 master 权重，参数部分就是 6 字节/参数。master 权重只在优化器更新时需要，前向反向用的是 BF16 副本，但两份都必须常驻——除非用分片或卸载。

| 模型 | BF16 权重 | 含 FP32 master |
| --- | --- | --- |
| 7B | 14 GB | 42 GB |
| 13B | 26 GB | 78 GB |
| 70B | 140 GB | 420 GB |

参数本身无法压缩到比“参数量 × 存储精度”更小，除非量化。训练中的量化权重（如 FP8）能省一半，但需要处理缩放因子和精度回退，生态还不如推理成熟。

分片是更常见的做法：张量并行把每层矩阵切开，每卡只存 1/TP；ZeRO-3/FSDP 把参数本身切到所有 rank，用时再 all-gather。70B 在 TP=8 下每卡 17.5 GB，这是推理和训练都常用的起点。

## 38.2 Gradient Memory

梯度和参数同形，BF16 下也是 2 字节/参数。反向算出的梯度先累积在本地，再参与同步。

DDP 的实现会为梯度分配一块连续的 bucket，反向算完一层就把对应片段放进去，bucket 满了就触发 all-reduce，和后续层的反向重叠。bucket 的大小影响重叠效果：太小，通信次数多、延迟占比高；太大，第一层要等很久才开始通信。常见值是 25 MB 左右。

ZeRO-2 之后，梯度也被分片，每卡只保留 1/N 的梯度用于更新，其余在 all-reduce 后丢弃。这把梯度显存从 14 GB 降到 1.75 GB（N=8），代价是通信量不变、实现更复杂。

梯度的另一个细节是累积。梯度累积 `k` 步时，本地梯度要一直保留到更新，显存不随 `k` 增长（因为累加在同一个 buffer 里），但如果用 FP32 累积，就是 4 字节/参数。

## 38.3 Optimizer State Memory

Adam 的 `m` 和 `v` 各 4 字节/参数，加上 FP32 master 权重 4 字节，共 12 字节。7B 是 84 GB，70B 是 840 GB。

| 组成 | 字节/参数 | 7B | 70B |
| --- | --- | --- | --- |
| FP32 master | 4 | 28 GB | 280 GB |
| Adam `m` | 4 | 28 GB | 280 GB |
| Adam `v` | 4 | 28 GB | 280 GB |
| 合计 | 12 | 84 GB | 840 GB |

这就是为什么 7B 训练的显存账单是 112 GB（14 + 14 + 84），而不是很多人直觉里的“14 GB 权重再加一点”。

省优化器状态有几条路：

- **分片**：ZeRO-1 把优化器状态切到 N 个 rank，7B、N=8 时每卡 10.5 GB；
- **卸载**：把 `m`、`v` 放到 CPU 内存，更新时搬回来；
- **低精度**：8-bit Adam 把 `m`、`v` 各压到 1 字节，省 75%；
- **换优化器**：Adafactor 用分解的二阶矩，几乎不存 `v`；
- **只训一部分**：LoRA 只更新低秩增量，优化器状态随之缩小几个数量级。

## 38.4 Activation Memory

激活是训练显存里最容易失控的部分，因为它随 batch 和序列长度线性增长，还随层数累加。

一层 Transformer 在前向中产生的、反向需要用的中间结果包括：

| 张量 | 每 token 元素数 | 说明 |
| --- | --- | --- |
| 归一化输入/输出 | `2h` | 残差前后各一份 |
| QKV 投影 | `h(1 + 2 n_{kv}/n_h)` | GQA 下小于 `3h` |
| Attention 输出 | `h` | |
| 投影输出 | `h` | |
| MLP 中间 | `2 × 4h` | SwiGLU 的两个分支 |
| SwiGLU 输出 | `4h` | |
| 合计 | 约 `16h` | 不含 attention scores |

按 FP16 存，每层每 token 约 `32h` 字节。70B、`h = 8192` 时是 256 KB/token/层；序列 8,192、batch 1 时，一层就是 2.1 GB，80 层需要 168 GB。这还没算 attention scores——如果不使用 FlashAttention，`S² × n_h` 的分数矩阵单独一层就能到 8.6 GB。

两个关键优化把这块压下来：

- **FlashAttention**：不物化 `S × S` 的分数矩阵，attention 的激活从 `O(S²)` 降到 `O(S·h)`；
- **激活重计算**：只存每层输入，反向时重算其余部分。

## 38.5 Activation Checkpointing

激活重计算（activation checkpointing，也叫 gradient checkpointing）的思路是：前向时只保存少数“检查点”处的激活，反向需要中间结果时，从最近的检查点重新前向一遍。

最粗的粒度是每层一个检查点：存每层输入，反向时对该层重算一次前向。

```python
def checkpointed_layer(x, layer):
    x.requires_grad_(True)
    with torch.no_grad():
        y = layer(x)          # 前向不保存中间结果
    y.requires_grad_(True)
    return CheckpointFunction.apply(layer, x, y)
```

显存从“所有中间结果”降到“每层输入”：70B、S=8192 时，每层输入 `S × h × 2 = 134 MB`，80 层约 10.7 GB，降了一个数量级。

代价是计算。反向时每层多算一次前向，总 FLOPs 从 `6N` 变成 `8N`，增加 33%。对 40% MFU 的训练，这相当于效率降到 30%——除非显存本来就不够，否则不划算。

折中方案：

| 方案 | 显存 | 额外计算 | 适用 |
| --- | --- | --- | --- |
| 不重算 | `O(L·S·h)` | 0 | 短序列、小模型 |
| 全部重算 | `O(L·h)` | +33% | 显存紧张 |
| 选择性重算 | 中 | +5%–10% | 只重算 attention |
| 分段检查点 | `O(√L · S·h)` | +10%–20% | 折中 |

选择性重算（selective activation checkpointing）只重算显存占用大、计算便宜的部分，比如 attention 内部的 softmax 和 dropout，保留 MLP 的激活。Megatron 的报告显示，这种策略在 GPT-3 规模上能以约 5% 的额外计算换到 5 倍显存节省。

分段的思路是把 L 层分成 `√L` 段，每段存一个检查点，段内再重算。显存和计算都落在 `O(√L)` 的量级，是理论上的最优点。

## 38.6 Gradient Offloading

梯度卸载把梯度搬到 CPU 内存，需要时再搬回来。7B 的 BF16 梯度是 14 GB，PCIe Gen5 的 50 GB/s 下搬一次 280 ms。

单独的梯度卸载收益有限，因为它必须和计算重叠才有意义：反向算完一层就把该层梯度传走，不占 GPU 显存，传输藏在后续层的计算后面。

更大的收益来自和 ZeRO 结合。ZeRO-Offload 把优化器状态和更新计算都放在 CPU：GPU 只负责前向反向，梯度传到 CPU 后由 CPU 做 Adam 更新，再把新参数传回 GPU。这样 GPU 显存只留参数、梯度和激活。

代价是 CPU 侧的算力和带宽。CPU 做 Adam 更新的吞吐远低于 GPU，7B 模型的更新在单路 CPU 上可能要几十毫秒；同时 PCIe 上要传梯度、参数和优化器状态，PCIe 带宽很快成为瓶颈。实践中的经验是：卸载适合“GPU 显存不够但 CPU 和 PCIe 有余”的场景，不适合追求极致吞吐的训练。

## 38.7 Optimizer State Offloading

优化器状态卸载（如 ZeRO-Infinity、FSDP 的 CPU offload）把 `m`、`v` 和 master 权重放到 CPU 内存甚至 NVMe。7B 的 84 GB 优化器状态全部搬走，GPU 上只剩 28 GB 的参数和梯度。

每步的流程是：

```text
GPU: 前向 + 反向 → 梯度
PCIe: 梯度传到 CPU
CPU: Adam 更新（读 m/v，写参数）
PCIe: 新参数传回 GPU
```

一次更新的 PCIe 流量是“梯度 + 参数”约 28 GB（7B），50 GB/s 下 560 ms——比一步前向反向（约 100 ms）慢得多。这意味着卸载优化器状态会把训练变成 PCIe 受限，除非能和多步计算重叠（预取下一批的更新）或者用更大的 batch 摊薄。

所以优化器状态卸载通常只在“模型必须训、硬件只有这些”的时候用，不是性能优化手段，而是可行性手段。

## 38.8 Memory-efficient Training

把上面的手段组合起来，一个 7B 模型的训练显存可以从 112 GB 降到单卡可承受的范围：

| 配置 | 权重 | 梯度 | 优化器 | 激活（S=4K） | 合计 |
| --- | --- | --- | --- | --- | --- |
| 原始 | 14 | 14 | 84 | ~20 | 132 |
| ZeRO-1（N=8） | 14 | 14 | 10.5 | ~20 | 58.5 |
| ZeRO-2（N=8） | 14 | 1.75 | 10.5 | ~20 | 46.25 |
| ZeRO-3（N=8） | 1.75 | 1.75 | 10.5 | ~20 | 35 |
| + 重计算 | 1.75 | 1.75 | 10.5 | ~5 | 19 |

表中的激活按 FlashAttention 估算：7B、`h=4096`、`S=4096` 时每层每 token 约 128 KiB，32 层合计 16 GiB，加上临时 buffer 约 20 GB。

结论很清楚：显存优化的顺序应该是先分片、再重计算、最后卸载。分片不增加计算，只增加通信；重计算用算力换显存；卸载用带宽换显存，代价最大。

还有几条常被忽略的：

- **序列长度的影响大于 batch**：激活随 `S` 线性增长，随 batch 也线性增长，但长序列还会让 attention 的激活变成 `O(S²)`（未用 FlashAttention 时）；
- **优化器状态的 dtype**：`v` 用 BF16 存会让小梯度失真，通常只在显存极度紧张时用；
- **临时 buffer**：all-gather 的接收 buffer、通信 bucket、cuBLAS workspace 都要预留，实际占用比公式值高 5%–10%；
- **碎片**：训练中张量生命周期差异大，PyTorch 的 caching allocator 会保留已释放的块，`max_memory_allocated` 和 `reserved` 可能差几十 GB。

最后是微调场景。全参数微调的显存和预训练同量级；LoRA 只训练低秩增量，优化器状态按增量参数算，7B 模型在单张 80 GB 卡上就能跑。这也是 LoRA 成为主流微调方式的原因——它把显存问题从“分布式”降级成了“单卡”。

## 本章小结

- 训练显存由参数、梯度、优化器状态和激活四部分组成；7B BF16 Adam 的前三项是 112 GB，优化器状态占 75%。
- 参数和梯度各 2 字节/参数；Adam 的 `m`、`v` 和 FP32 master 各 4 字节，共 12 字节/参数。
- 激活随层数、序列长度和 batch 线性增长；不用 FlashAttention 时 attention scores 还会带来 `O(S²)` 的项。
- 全部重计算把激活从 `O(L·S·h)` 降到 `O(L·h)`，代价是 33% 的额外前向；选择性重算只增加 5%–10%，是更常用的折中。
- 梯度卸载和优化器状态卸载用 PCIe 带宽换显存；7B 的优化器状态卸载一次要搬 28 GB，容易变成 PCIe 受限。
- 显存优化的优先顺序是分片、重计算、卸载：分片只加通信，重计算加算力，卸载加带宽。
- ZeRO-3 加重计算能把 7B 的训练显存从 132 GB 压到 20 GB 以内，但通信和计算开销要单独核算。
- 长序列的显存压力主要来自激活；微调场景用 LoRA 可以把问题降级到单卡。

## 延伸阅读

- Rajbhandari et al., [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), SC 2020.
- Rajbhandari et al., [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://www.usenix.org/conference/atc21/presentation/rajbhandari), ATC 2021.
- Rajbhandari et al., [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857), SC 2021.
- Chen et al., [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174), 2016.
- Korthikanti et al., [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198), MLSys 2023.
- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), NeurIPS 2022.
- Dettmers et al., [8-bit Optimizers via Block-wise Quantization](https://arxiv.org/abs/2110.02861), ICLR 2022.
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), ICLR 2022.
- PyTorch, [FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html).
