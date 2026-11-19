---
chapter: 3
part: 1
title: Computational Complexity and Cost Modeling
level: core
status: draft
prerequisites: [1, 2]
lab:
---

# Chapter 3. Computational Complexity and Cost Modeling

H100 的规格表上，989 TFLOPS 和 3.35 TB/s 并排印着。

一个决定算力上限，一个决定带宽上限。哪个先成为瓶颈，不取决于 GPU，而取决于你让它算什么。同一个矩阵乘法，形状不同，可能跑在算力受限区，也可能跑在带宽受限区；同一个模型，prefill 和 decode 的瓶颈也完全不同。

这一章把上一章的计算图变成公式：参数量、FLOPs、访存量、通信量，以及怎么用这些公式估算延迟、吞吐和成本。公式不追求精确到小数点，目标是**在写代码之前，判断优化空间在哪里**。

## 3.1 Parameters and Model Size

参数量决定三件事：权重占多少显存、每个 token 要多少计算、模型能表达多少知识。

一个 decoder-only Transformer 的参数量可以按层拆开：

$$
N \approx L \left( \underbrace{2H^2 + 2H H_{kv}}_{\text{attention}} + \underbrace{3H d_{ff}}_{\text{FFN}} + \underbrace{2H}_{\text{norm}} \right) + V H
$$

其中：

- `L`：层数；
- `H`：hidden dimension；
- `H_kv = n_kv × d`：KV 投影的维度；
- `d_ff`：FFN 中间维度；
- `V`：词表大小；
- 最后一项是 embedding；如果输出层不共享权重，还要再加 `V × H`。

以 Llama 3 70B 为例，代入上一章的超参数：

| 部分 | 计算 | 参数量 |
| --- | --- | --- |
| Q 投影 | `8192 × 8192` | 67.1M |
| K 投影 | `8192 × 1024` | 8.4M |
| V 投影 | `8192 × 1024` | 8.4M |
| O 投影 | `8192 × 8192` | 67.1M |
| FFN | `3 × 8192 × 28672` | 704.6M |
| 每层合计 |  | 855.7M |
| 80 层 | `855.7M × 80` | 68.5B |
| Embedding | `128256 × 8192` | 1.05B |
| **总计** |  | **约 70B** |

### 常见模型的规模

| 模型 | 总参数 | 每 token 激活参数 | 结构特点 |
| --- | --- | --- | --- |
| Llama 3 8B | 8.0B | 8.0B | Dense |
| Llama 3 70B | 70.6B | 70.6B | Dense，GQA |
| Llama 3 405B | 405B | 405B | Dense |
| Qwen2.5 72B | 72.7B | 72.7B | Dense，GQA |
| DeepSeek-V3 | 671B | 37B | MoE |

“总参数”和“激活参数”是 MoE 模型必须区分的两个数。DeepSeek-V3 有 671B 参数，但每个 token 只经过 37B 参数的计算。权重占 671B 的显存，计算量按 37B 算。

### 权重显存

权重显存就是参数量乘以每个参数的字节数：

| 精度 | 每参数字节 | 70B 模型权重 |
| --- | --- | --- |
| FP32 | 4 | 280 GB |
| FP16 / BF16 | 2 | 140 GB |
| FP8 | 1 | 70 GB |
| INT4 | 0.5 | 35 GB |

实际占用还要加上量化 scale、zero point、对齐 padding 和框架开销。INT4 的 35 GB 是理想值，真实部署通常在 38–42 GB 之间。

这张表解释了很多部署决策：

- 70B FP16 需要至少 2 张 80 GB 的卡；
- 70B FP8 可以放进 1 张 80 GB 的卡，但 KV Cache 和激活会挤占剩余空间；
- 70B INT4 可以放进 1 张 48 GB 的卡，代价是精度和 kernel 复杂度。

## 3.2 FLOPs and MACs

FLOPs 是浮点运算次数。MAC 是乘加运算（multiply-accumulate）。

一次 MAC 包含一次乘法和一次加法，按惯例算作 2 FLOPs。所以看到一个算子的 MACs，乘以 2 才是 FLOPs。

矩阵乘法 `[M, K] × [K, N]` 的 FLOPs 是：

$$
\text{FLOPs} = 2 M K N
$$

其中 `2` 来自每个输出元素的“乘一次、加一次”。

### 峰值算力的口径

厂商规格表上的 TFLOPS 有几个容易混淆的口径：

| 口径 | 含义 |
| --- | --- |
| FP32 | 非 Tensor Core 的 FP32 算力 |
| TF32 | Tensor Core 的 TF32 算力 |
| FP16 / BF16 dense | 不启用稀疏时的半精度算力 |
| FP16 / BF16 sparse | 启用 2:4 稀疏时，标称值翻倍 |
| FP8 | 新一代 Tensor Core 的低精度算力 |

H100 SXM 的 FP16 dense 是 989 TFLOPS，sparse 标称 1979 TFLOPS。做估算时要用 dense 值，因为大多数 LLM kernel 并没有利用 2:4 稀疏。

峰值算力是上限，不是现实。一个 GEMM 能跑到峰值的 70% 已经很好；Attention、LayerNorm、采样这些算子通常是 10%–40%。**用峰值算出来的时间，要再除以一个经验效率系数。**

## 3.3 Computational Complexity of Linear Layers

线性层是 LLM 的主要计算来源。

对一个 token，输入 `x` 是 `[H]`，权重 `W` 是 `[H, O]`，输出是 `[O]`。FLOPs 是 `2 × H × O`。

把模型所有线性层加起来，每个 token 的 FLOPs 约等于：

$$
\text{FLOPs}_{\text{token}} \approx 2N
$$

这个近似很好用：参数量是多少，每 token 的 forward 就是它的两倍（单位 GFLOPs）。

| 模型 | 参数量 | 每 token FLOPs |
| --- | --- | --- |
| Llama 3 8B | 8B | 16 GFLOPs |
| Llama 3 70B | 70B | 140 GFLOPs |
| Llama 3 405B | 405B | 810 GFLOPs |

prefill 处理 `S` 个 token，计算量是：

$$
\text{FLOPs}_{\text{prefill}} \approx 2 N S
$$

70B 模型处理 4096 token 的 prompt，约 `2 × 70e9 × 4096 = 5.7e14` FLOPs。按 H100 的 989 TFLOPS 峰值，理论时间约 0.58 秒；按 50% 的实际效率，约 1.16 秒。用 8 卡张量并行，可以把这个时间摊到 8 张卡上。

### 线性层的访存

线性层的权重有 `H × O` 个元素。FP16 下，读取权重需要 `2HO` 字节。

如果 batch 是 1，每个权重只参与一次乘加，算术强度是：

$$
\frac{2HO}{2HO} = 1 \text{ FLOP/byte}
$$

这就是 decode 阶段 GEMV 的典型特征：算得少，读得多。

batch 增大到 `B` 后，同一份权重服务 `B` 个 token，算术强度变成 `B`。这是 batching 提升吞吐的根本原因。

## 3.4 Computational Complexity of Attention

Attention 的复杂度分成两部分：投影和分数计算。

### 投影

Q/K/V/O 四个投影的 FLOPs 已经在参数量里算过，每 token 约 `2 × (2H^2 + 2H H_kv)`。

### 分数计算

对 batch `B`、序列长度 `S`、`n_h` 个 Query 头、head dimension `d`：

- `QK^T`：`2 B n_h S^2 d` FLOPs；
- `softmax`：约 `5 B n_h S^2` FLOPs（指数、求和、除法）；
- `AV`：`2 B n_h S^2 d` FLOPs。

总计：

$$
\text{FLOPs}_{\text{attn}} \approx 4 B n_h S^2 d + 5 B n_h S^2
$$

causal mask 让有效计算量减半，近似为：

$$
\text{FLOPs}_{\text{attn, causal}} \approx 2 B n_h S^2 d
$$

### 与线性层的对比

线性层总量是 `2NS`。attention 是 `2 L n_h d S^2`。两者相等时：

$$
S \approx \frac{N}{L n_h d}
$$

对 Llama 3 70B：`70e9 / (80 × 64 × 128) ≈ 107,000`。

也就是说，上下文长度到 10 万 token 左右，attention 的计算量才追平线性层。对 Llama 3 8B，这个交叉点约为 61,000。

| 上下文长度 | 70B 模型 attention 占 prefill FLOPs 的比例 |
| --- | --- |
| 4K | 约 4% |
| 32K | 约 30% |
| 128K | 约 120% |

这里的“120%”不是错误：attention 的计算量已经超过线性层。长上下文推理的瓶颈，会从 GEMM 转向 attention。

### Decode 的 attention

decode 时 `S` 是历史长度，但 Query 只有新 token 一个。每个 token、每层要做：

- `QK^T`：`2 B n_h S d` FLOPs；
- `AV`：`2 B n_h S d` FLOPs。

合计 `4 B n_h S d`，随上下文线性增长。它比 prefill 的平方增长温和得多，但每步都要读一遍 KV Cache，访存压力大。

## 3.5 Prefill and Decode Complexity

Prefill 和 decode 是两种完全不同的负载。

| 维度 | Prefill | Decode |
| --- | --- | --- |
| 每步 token 数 | `S` | `B`（每个请求 1 个） |
| 主要算子 | 大 GEMM | GEMV / 小 GEMM |
| 算术强度 | 约 `S` | 约 `B` |
| 瓶颈 | 算力 | 显存带宽 |
| 优化手段 | FlashAttention、chunked prefill、并行 | KV Cache、batching、量化 |

### Prefill

prefill 的 FLOPs 是 `2NS`，访存主要是权重和激活。权重只读一遍，激活是 `B × S × H` 量级。算术强度大致是：

$$
\frac{2NS}{2N + \text{activation bytes}} \approx S
$$

当 `S` 远大于 1，算术强度很高，prefill 通常受算力限制。

### Decode

decode 每步的 FLOPs 是 `2NB`，需要读取：

- 全部权重：`2N` 字节（FP16）；
- KV Cache：`k × S × B` 字节，`k` 是每 token 的 KV 大小；
- 激活：少量。

算术强度：

$$
\text{Intensity} = \frac{2NB}{2N + k S B}
$$

当 `B` 很小，强度接近 `B`；当 `B` 很大，强度趋近于 `2N / (kS)`。

对 Llama 3 70B、FP16、8K 上下文：

- `2N = 1.4e11` 字节；
- `k = 327,680` 字节/token；
- `kS = 2.68e9` 字节/请求。

batch 1 时强度约 1；batch 32 时约 31；batch 512 时约 47，接近 52 的上限，仍远低于 H100 的 295。**即使 batch 很大，8K 上下文的 70B decode 仍然是带宽受限的。**

如果上下文降到 1K，`kS = 3.36e8`，大 batch 下的强度上限约 417，才可能进入算力受限区。

### 延迟估算

decode 的单 token 时间下限是：

$$
\text{TPOT}_{\min} = \frac{\text{权重字节} + \text{KV 字节}}{\text{带宽}}
$$

70B、FP16、batch 1：`140 GB / 3.35 TB/s ≈ 42 ms`。实际带宽利用率通常 50%–70%，所以真实 TPOT 在 60–85 ms 之间。

用 8 卡张量并行，每张卡只读 1/8 的权重：`17.5 GB / 3.35 TB/s ≈ 5.2 ms`，理想情况下 TPOT 可以降到 7–10 ms。代价是卡间通信。

prefill 的时间下限是：

$$
\text{TTFT}_{\min} = \frac{2NS}{\text{峰值算力} \times \text{效率}}
$$

70B、4K prompt、50% 效率：约 1.16 秒。这是单卡；8 卡张量并行可以降到约 0.15 秒。

这两个公式解释了一个常见现象：**长 prompt 的首 token 延迟主要由 prefill 决定，后续 token 的速度主要由带宽和 batch 决定。**

## 3.6 Activation and KV Cache Memory Requirements

推理显存由三部分组成：权重、KV Cache、激活。

### 激活

激活是前向传播中的中间张量。推理时，它们用完即释放，峰值通常不大；训练时，它们要保存到反向传播，显存压力大得多。

以 Llama 3 70B、batch 1、8K 上下文、FP16 为例，一个 `[B, S, H]` 张量是：

$$
1 \times 8192 \times 8192 \times 2 \approx 134 \text{ MB}
$$

一个 block 里同时存在的激活可能有十几个，80 层叠加，轻松超过 100 GB。这就是训练必须做激活检查点（activation checkpointing）的原因。

### KV Cache

每 token 的 KV Cache 是：

$$
2 \times L \times n_{kv} \times d \times \text{bytes}
$$

| 模型 | 每 token KV | 8K 上下文 | 32K | 128K |
| --- | --- | --- | --- | --- |
| Llama 3 70B（GQA） | 320 KiB | 2.5 GiB | 10 GiB | 40 GiB |
| GPT-3 175B（MHA） | 4.5 MiB | 36 GiB | 144 GiB | 576 GiB |

这张表是容量规划的起点。一个 80 GB 的卡，扣掉权重和激活，能放多少个并发请求，直接由 KV Cache 决定。

例如 8×H100 部署 70B FP16：

- 权重：140 GB；
- 激活和 workspace：约 50 GB；
- 剩余给 KV：约 450 GB；
- 8K 上下文每个请求 2.5 GiB，理论上可支持约 180 个并发请求。

实际还要留出碎片和调度余量，通常按 70% 估算，约 120–130 个并发。

### Prefix Cache

很多请求共享相同的前缀：系统提示词、few-shot 示例、长文档。

Prefix cache 把相同前缀的 KV Cache 块复用，避免重复 prefill。命中率高的场景，TTFT 可以下降一个数量级，KV 显存也更省。

代价是缓存管理复杂：要检测前缀、做引用计数、处理淘汰。vLLM 的 PagedAttention 和 SGLang 的 RadixAttention 都在这个方向做优化。

## 3.7 Training Memory Requirements

训练显存比推理复杂得多。

### 优化器状态

混合精度训练（Adam）通常需要：

| 内容 | 每参数字节 |
| --- | --- |
| FP16 权重 | 2 |
| FP16 梯度 | 2 |
| FP32 master weights | 4 |
| Adam 一阶矩 `m` | 4 |
| Adam 二阶矩 `v` | 4 |
| **合计** | **16** |

70B 模型仅这部分就是 `70e9 × 16 = 1.12 TB`。

### 激活

激活和 batch、序列长度、层数成正比。用激活检查点可以把大部分激活丢掉，只保留每层输入，反向时重算。代价是额外约 33% 的前向计算。

### 并行策略

单卡放不下，就要切分：

| 策略 | 切什么 | 主要通信 |
| --- | --- | --- |
| 数据并行（DP） | batch | 梯度 AllReduce |
| 张量并行（TP） | 单层内的矩阵 | 每层两次 AllReduce |
| 流水线并行（PP） | 层 | 阶段间激活传输 |
| ZeRO / FSDP | 优化器状态、梯度、参数 | 分片 AllGather / ReduceScatter |

ZeRO 有三个阶段：

- ZeRO-1：切优化器状态；
- ZeRO-2：再切梯度；
- ZeRO-3：再切参数，前向时按需 AllGather。

FSDP 可以理解为 PyTorch 原生的 ZeRO-3 实现。

70B 模型用 16 卡训练，1.12 TB 的优化器状态分摊到每张卡约 70 GB，加上权重、梯度和激活，仍然紧张。这就是为什么大模型训练通常要几十到几千张卡。

## 3.8 Arithmetic Intensity

算术强度是 FLOPs 与访存字节数的比值：

$$
I = \frac{\text{FLOPs}}{\text{Bytes}}
$$

Roofline 模型给出的性能上限是：

$$
\text{Performance} = \min(\text{Peak Compute}, \text{Bandwidth} \times I)
$$

两者的交点就是脊点：

$$
I_{\text{ridge}} = \frac{\text{Peak Compute}}{\text{Bandwidth}}
$$

| 硬件 | FP16 峰值 | 带宽 | 脊点 |
| --- | --- | --- | --- |
| A100 80GB | 312 TFLOPS | 2.0 TB/s | 156 |
| H100 SXM | 989 TFLOPS | 3.35 TB/s | 295 |

算术强度低于脊点，算子受带宽限制；高于脊点，受算力限制。

| 算子 | 典型算术强度 | 瓶颈 |
| --- | --- | --- |
| 大 GEMM（M=N=K=8192） | 数千 | 算力 |
| 小 GEMM（M=1） | 1–10 | 带宽 |
| Decode GEMV | 约 1 | 带宽 |
| Prefill Attention | 100–1000 | 算力/带宽混合 |
| Elementwise | 0.25–0.5 | 带宽 |

算术强度不是算子的固有属性。batch、精度、融合方式都会改变它：

- batch 从 1 增到 32，GEMV 的强度从 1 增到 32；
- FP16 换成 FP8，同样字节数的 FLOPs 不变，强度翻倍；
- 算子融合减少中间结果的读写，强度上升。

**优化的本质，就是把算术强度推到脊点以上，或者把带宽压力降下来。**

## 3.9 Communication Complexity

多卡之后，通信成为新的瓶颈。

### 集合通信的量

设消息大小为 `M` 字节，参与者为 `n`：

| 操作 | 每卡发送/接收量 | 说明 |
| --- | --- | --- |
| AllReduce（Ring） | `2(n-1)/n × M ≈ 2M` | 先 ReduceScatter，再 AllGather |
| AllGather | `(n-1)/n × M ≈ M` | 每卡拿到完整结果 |
| ReduceScatter | `(n-1)/n × M ≈ M` | 每卡拿到一部分 |
| All-to-All | `(n-1)/n × M ≈ M` | 每卡向其他卡发一部分 |

Ring AllReduce 的延迟是 `2(n-1) × α`，其中 `α` 是单次通信延迟。消息很小时，延迟主导；消息很大时，带宽主导。

### 张量并行

张量并行每层要做两次 AllReduce：

- attention 输出投影之后；
- FFN 输出投影之后。

每次的消息大小是 `B × S × H × bytes`。每层通信量约：

$$
4 B S H \frac{n-1}{n} \text{ bytes}
$$

70B、FP16、batch 1、decode（`S=1`）、TP=8：

$$
4 \times 8192 \times 2 \times \frac{7}{8} \approx 57 \text{ KB}
$$

80 层就是约 4.6 MB/token。NVLink 4 的带宽约 900 GB/s，传输时间约 5 微秒；但每次集合通信还有几微秒的固定延迟，80 层 × 2 次 = 160 次，延迟可能达到几百微秒。**decode 的 TP 通信是延迟敏感的，不只是带宽问题。**

prefill 时 `S` 很大，通信量变成带宽主导：

`4 × 4096 × 8192 × 2 × 7/8 ≈ 234 MB` 每层，80 层约 18.7 GB。按 900 GB/s 算，约 21 毫秒。这个开销和 prefill 的计算时间同一量级，不能被忽略。

### 数据并行与流水线并行

数据并行的梯度 AllReduce 每步同步一次，消息大小是 `2N` 字节（FP16 梯度）。70B 模型约 140 GB。

在 400 Gb/s InfiniBand（约 50 GB/s）上，传输时间约 2.8 秒。即使和反向传播重叠，梯度同步仍然是扩展性的关键约束。

流水线并行的通信量小得多：每个 microbatch 在阶段之间传 `B × S × H × bytes` 的激活。但流水线有气泡，阶段越多，气泡越大。

### 通信强度

和算术强度类似，可以定义通信强度：

$$
I_{\text{comm}} = \frac{\text{FLOPs}}{\text{Communication bytes}}
$$

如果计算时间大于通信时间，通信可以被隐藏；否则就会成为瓶颈。张量并行的通信强度随 `B × S` 增大而提高，所以 prefill 比 decode 更容易隐藏通信。

## 3.10 Cost Estimation for LLM Workloads

把前面的公式组合起来，可以估算推理成本。

### 每百万 token 成本

$$
\text{Cost}_{1M} = \frac{\text{GPU 小时单价} \times \text{GPU 数} \times \text{小时数}}{\text{生成的 token 数}} \times 10^6
$$

一个算例，假设：

- 8×H100 节点，每 GPU 每小时 3 美元，总价 24 美元/小时；
- 70B 模型，FP16，batch 64；
- 聚合输出吞吐 2,000 token/s。

每小时生成 `2,000 × 3,600 = 7.2M` token，成本是：

$$
\frac{24}{7.2} \approx 3.33 \text{ 美元/百万 token}
$$

如果吞吐做到 10,000 token/s，成本降到约 0.67 美元/百万 token。**吞吐每提高一倍，单位成本就减半。**

### 容量规划

先算显存能支持多少并发，再算吞吐，最后算成本。

以 8×H100、70B FP16、8K 上下文为例：

- KV 预算约 450 GB；
- 每请求 2.5 GiB；
- 并发上限约 180，按 70% 安全系数取 126。

如果每个请求平均生成 512 token，一轮大约 126 × 512 = 64,512 token。按 2,000 token/s 的吞吐，一轮约 32 秒。

用 Little's Law 估算所需副本：

$$
\text{并发数} = \text{到达率} \times \text{平均停留时间}
$$

如果每秒到达 20 个请求，每个请求停留 32 秒，就需要约 640 个并发槽位；按每节点 126 个算，至少 6 个节点。这还没算峰值和故障冗余。

### 成本杠杆

| 手段 | 影响 | 注意事项 |
| --- | --- | --- |
| FP8 / INT4 量化 | 带宽和显存下降，吞吐上升 | 精度损失、校准成本 |
| 增大 batch | 权重读取摊薄，吞吐上升 | 尾延迟、KV 显存 |
| Prefix cache | 重复前缀不再 prefill | 命中率、缓存管理 |
| 投机解码 | TPOT 下降 | 草稿模型开销、接受率 |
| 请求路由 | 长短请求分流 | 实现复杂、需要观测 |
| Spot 实例 | 单价下降 | 中断风险、恢复成本 |

这些手段的收益不是简单相加的。量化会改变 batch 的最优点，prefix cache 会改变 KV 的占用，路由会改变负载分布。**成本优化是一个联合优化问题，不是一张 checklist。**

### 估算的边界

上面的公式给出的是量级，不是精确值。真实系统还要考虑：

- kernel 效率随形状变化；
- 显存碎片和分配器开销；
- 网络拥塞和尾延迟；
- 请求长度分布的长尾；
- 故障、重试和冷启动。

所以估算的用途不是替代测量，而是**决定先测什么**。如果一个算例显示 decode 受带宽限制，就先看 HBM 利用率和 batch 组成；如果显示 prefill 受算力限制，就先看 GEMM 效率和并行度。

下一章讲硬件：这些带宽和算力数字从哪来，为什么 H100 的 3.35 TB/s 不是随便就能跑满。

## 本章小结

- 参数量决定权重显存和每 token 的计算量；MoE 模型要区分总参数和激活参数。
- 线性层每 token 约 `2N` FLOPs；prefill 是 `2NS`，是算力受限的。
- Attention 的计算量随 `S^2` 增长；70B 模型在约 10 万 token 时追平线性层。
- Decode 的算术强度约等于 batch size；即使 batch 很大，长上下文下仍然是带宽受限的。
- 显存由权重、KV Cache 和激活组成；训练还要加上 16 字节/参数的优化器状态。
- 多卡通信量可以用集合通信公式估算；张量并行在 decode 阶段是延迟敏感的。
- 成本估算的核心是吞吐和显存容量；所有优化最终都落在“每百万 token 多少钱”上。

## 延伸阅读

- Williams et al., [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://dl.acm.org/doi/10.1145/1498765.1498785), 2009.
- Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), 2020.
- Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), 2022.
- Rajbhandari et al., [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), 2019.
- Shoeybi et al., [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053), 2019.
- Narayanan et al., [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473), 2021.
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), 2023.
- NVIDIA, [H100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h100/).
