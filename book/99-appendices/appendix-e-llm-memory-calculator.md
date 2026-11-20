# Appendix E. LLM Memory Calculator

本附录把所有显存公式集中在一起，配合算例。书中反复出现的 70B 配置为：`L = 80` 层，`h = 8192`，64 个 Q 头、8 个 KV 头，head_dim = 128。

## E.1 参数量

| 组件 | 公式 | 70B 的数值 |
| --- | --- | --- |
| 每层 attention | `2h² + 2h·h_kv`（GQA） | `2 × 8192² + 2 × 8192 × 1024 = 151 M` 参数，302 MB |
| 每层 MLP | `3 × h × h_ff`，`h_ff = 28672` | `3 × 8192 × 28672 = 705 M` 参数，1.41 GB |
| 每层合计 | — | 856 M 参数，1.71 GB（FP16） |
| 全部层 | `L × 每层` | 68.5 B 参数，137 GB |
| Embedding + LM head | `V × h`，`V = 128K` | 1.05 B 参数，2.1 GB |

合计约 69.5 B 参数，FP16 下 139 GB，与常见的“70B / 140 GB”一致。粗略公式 `12Lh²` 给出 64.4 B，偏小，因为它假设 MLP 只有两个矩阵且 `h_ff = 2.67h`；SwiGLU 有三个矩阵，`h_ff/h = 3.5` 时 MLP 部分是 `10.5h²` 而不是 `8h²`。

## E.2 权重显存

$$
\text{权重} = N \times \frac{\text{bits}}{8}
$$

| 精度 | 70B 权重 |
| --- | --- |
| FP32 | 280 GB |
| BF16 | 140 GB |
| FP8 | 70 GB |
| INT4 | 35 GB |

## E.3 KV Cache

每个 token 的 KV 字节数：

$$
\text{KV/token} = 2 \times L \times n_{kv} \times d_{head} \times \frac{\text{bits}}{8}
$$

70B、BF16：

$$
2 \times 80 \times 8 \times 128 \times 2 = 327{,}680\ \text{B} = 320\ \text{KiB}
$$

| 上下文 | 单请求 KV | 8 卡 TP=8 后每卡 |
| --- | --- | --- |
| 4K | 1.25 GiB | 160 MiB |
| 8K | 2.5 GiB | 320 MiB |
| 32K | 10 GiB | 1.25 GiB |
| 128K | 40 GiB | 5 GiB |

并发上限：`可用 KV 显存 / (每请求 KV)`。8×H100 上给 KV 留 400 GB，8K 上下文能跑约 150 个并发请求。

## E.4 激活（推理）

推理不保存中间激活，只有临时的 buffer：

| 张量 | 大小 |
| --- | --- |
| 最大中间结果（MLP 的 `h_ff`） | `B·S·h_ff × 2 B` |
| attention 分数（不用 FlashAttention） | `B·n_h·S² × 2 B` |

`B = 1`、`S = 8192` 时，MLP 中间是 `8192 × 28672 × 2 = 470 MB`；分数矩阵是 `32 × 8192² × 2 = 4.3 GB`——这就是必须用 FlashAttention 的原因。

## E.5 训练显存

$$
\text{训练显存} = \underbrace{2N}_{\text{BF16 权重}} + \underbrace{2N}_{\text{梯度}} + \underbrace{12N}_{\text{Adam + master}} + \text{激活}
$$

| 模型 | 权重 + 梯度 + 优化器 | 加激活（8K） |
| --- | --- | --- |
| 7B | 112 GB | 约 132 GB |
| 13B | 208 GB | 约 240 GB |
| 70B | 1,120 GB | 约 1,300 GB |

激活按 FlashAttention 估算：每层每 token 约 `32h` 字节，`S = 8192`、`B = 1` 时 70B 是 168 GB；用激活重计算后降到约 10.7 GB。

## E.6 并行后的每卡显存

以 70B、BF16、N 卡为例：

| 并行 | 权重/卡 | 说明 |
| --- | --- | --- |
| TP=8（BF16） | 17.5 GB | 每层按头或按列切分 |
| TP=8 + PP=2（BF16） | 8.75 GB | 层再对半分 |
| TP=8（FP8） | 8.75 GB | 权重减半 |
| TP=8 + PP=2（FP8） | 4.4 GB | 两者叠加 |

训练时再叠加 ZeRO/FSDP 对参数、梯度和优化器状态的分片，见 Chapter 38 的显存表。

推理时的实际占用还要加上 KV Cache 和临时 buffer，通常再留 20%–30%。

## E.7 快速估算清单

1. 参数量：`12Lh²`，再按 MLP 放大系数和 GQA 修正；
2. 权重：参数量 × 精度字节；
3. KV：`2 × L × n_kv × d_head × bytes × tokens`；
4. 训练：权重 × 16 字节/参数（BF16 + Adam），加激活；
5. 每卡：除以并行度，再加 KV 和 buffer；
6. 并发：可用 KV 显存 / 每请求 KV。
