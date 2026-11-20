---
chapter: 25
part: 5
title: LLM Inference Execution
level: tbd
status: draft
prerequisites: [5, 7]
lab:
---

# Chapter 25. LLM Inference Execution

一份 serving trace 里，同一个请求的两段耗时对不上。8K 输入，首 token 用了 240 ms；之后 500 个输出 token，每个 13 ms，合计约 6.7 秒。前面 8,192 个 token 平均每个 29 微秒，后面 500 个平均每个 13 毫秒，相差约 450 倍。

同一个模型、同一份权重、同一张卡。区别只有一个：预填充（prefill）一次处理 8,192 个位置，解码（decode）一次只处理 1 个。这一章把这两段拆开，算清楚时间花在哪里，以及 batch、上下文长度和采样方式各自改变哪一段。

## 25.1 Inference Pipeline

一个请求进入引擎后，依次经过分词（tokenization）、排队与准入、prefill、decode 循环、反分词与流式返回。权重在服务启动时读入 HBM，之后被所有请求复用；KV Cache（键值缓存）随每个请求独立增长。引擎的调度目标，就是让这两类资源都不空转。

| 阶段 | 一次处理的位置数 | 主要资源 | 8K 输入、500 输出的占比 |
| --- | --- | --- | --- |
| 分词 | — | CPU | 约 1 ms，不到 0.1% |
| 排队与准入 | — | 调度器 | 取决于负载，空载为 0 |
| Prefill | S = 8,192 | 算力 | 240 ms，约 3.6% |
| Decode | 1 | HBM 带宽 | 500 × 13 ms = 6.5 s，约 96% |
| 反分词与流式返回 | — | CPU、网络 | 每 token 几十微秒 |

两段用的是同一份权重、同一套算子，但矩阵形状不同：prefill 的 GEMM 是 `[8192, 8192] × [8192, 10240]`，decode 的对应形状是 `[1, 8192] × [8192, 10240]`。前者是矩阵乘，后者退化成矩阵向量乘（GEMV）。形状一变，受限的资源就从算力换成了带宽。

真实引擎不会让一个请求走完整条流水线再处理下一个。连续批处理（continuous batching）在每个 decode 步重新组 batch，完成的请求立刻退出，排队的请求立刻补位；prefill 也会被切成块，和 decode 混在同一批里。阶段划分没有变，变的是多个请求在同一时刻共享同一个阶段。

## 25.2 Model Loading

70B 的 FP16 权重是 140 GB，一张 H100 的 80 GB 装不下，至少需要 2 路张量并行（tensor parallelism，TP）；生产上更常见的是 TP=8，每卡 17.5 GB。

| 步骤 | 做什么 | 70B 的量级 |
| --- | --- | --- |
| 解析 index | 读 safetensors header，校验形状 | 几百毫秒 |
| mmap 映射 | 建立虚拟地址映射，不复制数据 | 0 字节实际读入 |
| 按 TP 切分 | 每个 rank 取自己那一份 | 每卡 17.5 GB |
| dtype 转换 | BF16 转 FP16 | 读写各 140 GB，约 40 秒 |
| 拷贝到 HBM | H2D 传输 | 140 GB，PCIe 4.0 约 5.6 秒 |
| 预热与 CUDA Graph 捕获 | 跑几个 dummy batch | 数秒 |

冷启动的总量级：从 NVMe 顺序读 140 GB，按 7 GB/s 算约 20 秒；加上转换和传输，接近 1 分钟。如果文件还在 page cache 里，可以压到十几秒。这就是弹性扩容时新实例迟迟不进池的原因。

```cpp
struct WeightShard {
  const half* base;   // mmap 基址
  size_t bytes;       // 本 rank 负责的字节数
  int tp_rank;
  int tp_size;
};

// 列并行按输出维切，行并行按输入维切
inline std::pair<size_t, size_t> shard_range(size_t rows, int rank, int tp) {
  size_t per = rows / tp;
  return {rank * per, per};
}
```

权重常驻带来两个约束：显存预算里权重是固定成本，KV 是可变成本。量化会同时改变两者——FP8 权重 70 GB，INT4 约 35 GB，decode 的带宽下限也随之减半或降到四分之一。

80 GB 的卡上，权重、KV、激活和 CUDA context 要一起放下。TP=8 时每卡 17.5 GB 权重，按 0.9 的显存利用率留给推理 72 GB，剩下 54 GB 给 KV 和激活。这个数字决定了后面所有容量计算的上界。

## 25.3 Prefill Phase

prefill 把所有输入位置放进同一个矩阵乘。S = 8,192、hidden 为 8,192 时，QKV 投影的 M 足够大，Tensor Core 可以被喂满。总计算量是：

$$
C_{\text{prefill}} = 2NS + 2 n_h L S^2 d
$$

第一项是所有权重矩阵的乘加，第二项是 causal attention 的 `QK^\top` 和 `PV`。代入 70B（N = 70e9）、S = 8,192、L = 80、n_h = 64、d = 128：

$$
C_{\text{prefill}} = 2 \times 70\times10^9 \times 8192 + 2 \times 64 \times 80 \times 8192^2 \times 128 \approx 1.15\times10^{15} + 8.8\times10^{13}
$$

合计约 1.24 PFLOP，其中 attention 只占 7%。这是一个容易记错的分配：长上下文让 attention 变贵，但在 8K 这个尺度上，线性层仍然是绝对大头。

| 项目 | FLOPs | 8×H100 峰值时间 | 65% MFU |
| --- | --- | --- | --- |
| 线性层 | 1,147 TFLOP | 145 ms | 223 ms |
| Attention（causal） | 88 TFLOP | 11 ms | 17 ms |
| 合计 | 1,235 TFLOP | 156 ms | 240 ms |

8 张 H100 的 FP16 峰值是 7.9 PFLOPS，所以算力下限是 156 ms，工程实现大约落在 240 ms。

MFU（Model FLOPs Utilization，模型算力利用率）是这一段唯一值得追的指标：1,235 TFLOP 除以实测时间，再除以 7.9 PFLOPS。GEMM 效率高的实现能到 70%，attention 和 AllReduce 把整体拉到 60% 上下；低于 50% 通常说明 tile 配置或并行切分有问题，而不是卡不够。

prefill 还是吞吐友好的：一次 8K 的 prefill 和 8 次 1K 的 prefill，线性层的算力相同，但前者少 7 次权重读取和 7 次启动；attention 部分 8K 更贵，因为它是平方项。把长 prefill 切成固定大小的块，是为了调度公平，不是为了省算力。

访存侧：每卡读 17.5 GB 权重，按 3.35 TB/s 是 5.2 ms；KV 写入总共 2.6 GB，分摊到 8 卡只有 0.1 ms 量级。算力与带宽的时间比约 30:1，算术强度（arithmetic intensity）约等于 S = 8,192，是脊点 295 的 28 倍。这是典型的算力受限（compute-bound）。

所以 prefill 的优化方向是 GEMM 效率、FP8 和并行切分，而不是带宽。到了 128K 上下文，两项的量级会反过来：attention 变成 22.5 PFLOP，超过线性层的 18.4 PFLOP，瓶颈随之转移。

## 25.4 Decode Phase

decode 每个请求只送进 1 个位置。所有权重矩阵都要从 HBM 读一遍，却只参与一次乘加。每步的时间下限是：

$$
T_{\text{decode}} \ge \frac{2N + S \cdot k_{kv}}{BW}, \qquad k_{kv} = 2 L n_{kv} d b
$$

`k_kv` 是每 token 的 KV 字节数，70B 的 GQA 配置下是 320 KiB，8K 上下文就是 2.6 GB。代入单卡：

| 项目 | 字节或 FLOPs | 时间（3.35 TB/s、989 TFLOPS） |
| --- | --- | --- |
| 权重读取 | 140 GB | 41.8 ms |
| KV 读取 | 2.6 GB | 0.78 ms |
| Attention 计算 | 21.5 GFLOP | 0.02 ms |
| LM head 权重 | 2.1 GB | 0.63 ms |
| 合计 | 144.7 GB | 43.2 ms |

算术强度是 140 GFLOP / 144.7 GB ≈ 0.97 FLOP/byte，比脊点低约 300 倍。计算只要 0.14 ms，剩下的全是等待数据。

42 ms 是“一张卡把 140 GB 读一遍”的下限；70B 放不进一张 80 GB 的卡，所以这个数字是一把标尺，用来说明每 token 的字节成本。TP=8 时每卡 17.5 GB，下限 5.2 ms，实测 10–15 ms，差距来自 KV、小 kernel 和启动开销。

GQA（grouped-query attention，分组查询注意力）在这里的作用是把 KV 从 2.5 MiB/token 压到 320 KiB/token。如果是 64 个 KV 头的 MHA，8K 上下文的 KV 是 21 GB，单卡每步要读 161 GB，TPOT 下限从 43 ms 涨到 48 ms。看起来涨得不多，但它同时吃掉了显存，能并发的请求数降到八分之一。

## 25.5 Autoregressive Generation

自回归（autoregressive）循环本身只有几行：

```python
def generate(model, kv, input_ids, max_new, sampler):
    logits = model.prefill(input_ids, kv)        # [S, V]
    next_id = sampler(logits[-1])
    out = [next_id]
    pos = len(input_ids)
    for _ in range(max_new - 1):
        logits = model.decode(next_id, kv, pos)  # [1, V]
        next_id = sampler(logits[0])
        out.append(next_id)
        pos += 1
        if next_id == EOS:
            break
    return out
```

循环里没有可以并行的东西。第 t+1 步依赖第 t 步采样出的 token，采样又依赖第 t 步的 logits。一步之内有大量并行（64 个 Query 头、28,672 维的 FFN），步与步之间是严格串行。

| 组成 | 8×H100、8K、batch 1 的估算 | 说明 |
| --- | --- | --- |
| 权重读取 | 5.2 ms | 每卡 17.5 GB |
| KV 读取与 attention | 0.3 ms | 每卡 0.33 GB |
| 小 kernel 执行 | 1–2 ms | softmax、norm、RoPE |
| kernel launch | 2–3 ms | 80 层约 900 次，CUDA Graph 可压到 0.5 ms 以下 |
| 采样 | 0.05–0.2 ms | 128K 词表 |
| 合计 | 9–11 ms | 实测 10–15 ms |

提升单请求速度只有三条路：减少每步字节（量化、KV 压缩）、减少步数（投机解码，Chapter 28）、把多步合成一次（多 token 预测）。第一条改的是带宽，后两条改的是串行长度。

循环里还有两个容易忽略的细节。KV 的追加发生在每层 attention 之前，写入位置由逻辑位置决定，不是由显存地址决定；位置算错不会报错，只会让输出慢慢变差。停止条件也不止 EOS：还有 max_new_tokens、stop strings 和客户端取消。取消必须能中断正在执行的步，否则一次长生成会一直占着 KV 直到自然结束。

## 25.6 Sampling and Logits Processing

lm_head 是 `[8192, 128256]` 的矩阵，1.05B 参数，2.1 GB。每步都要完整读一遍，占 decode 权重流量的 1.5%。输出 logits 是长度为 128,256 的 FP32 向量，513 KB。

处理顺序固定：先在 logits 上做重复惩罚，再除以温度 T，然后 top-k、top-p 截断，softmax 归一化，最后做多项式采样。

$$
p_i = \frac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)}
$$

温度必须在截断之前，因为 top-p 的累积概率依赖分布形状；先截断再调温度，候选集会不一样。

```python
def sample(logits, temperature=1.0, top_k=0, top_p=1.0):
    z = logits.float() / max(temperature, 1e-6)
    if top_k > 0:
        kth = torch.topk(z, top_k).values[-1]
        z = torch.where(z < kth, float("-inf"), z)
    probs = torch.softmax(z, dim=-1)
    if top_p < 1.0:
        sp, idx = torch.sort(probs, descending=True)
        keep = (torch.cumsum(sp, -1) - sp) <= top_p
        sp = torch.where(keep, sp, torch.zeros_like(sp))
        probs = torch.zeros_like(probs).scatter_(0, idx, sp)
    return torch.multinomial(probs / probs.sum(), 1)
```

| 操作 | 复杂度 | 128K 词表的成本（GPU） | 备注 |
| --- | --- | --- | --- |
| 读取 logits | O(V) | 0.5 MB | 可忽略 |
| top-k（k=50） | O(V) | 0.05–0.1 ms | 选择而非全排序 |
| top-p | O(V log V) | 0.1–0.3 ms | 可换成直方图近似 |
| softmax 与采样 | O(V) | 0.02 ms | 单 kernel |
| 合计 | — | 0.1–0.4 ms | 占 13 ms 步长的 1%–3% |

成本不高，但它在关键路径上，而且每个请求独立。batch 128 时如果写成 128 次独立排序，就是 128 倍的串行；正确做法是按行并行的批量 kernel。贪心解码（T=0）可以跳过 softmax，但仍然要对 128K 个值求 argmax。

数值上有一个必做项：softmax 之前减去最大值。128K 个 logits 里只要出现一个 1e4 量级的值，`exp` 直接计算就会溢出成 inf，归一化之后全是 NaN。减去最大值后所有指数都不大于 0，结果不变，数值安全。采样用的随机数也要按请求播种并记录，否则同一份输入无法复现。

## 25.7 Batch Inference

decode 的权重读取与 batch 无关，这是批处理（batching）的全部理由。把 B 个请求放进同一步：

$$
T_{\text{step}}(B) \approx \frac{2N + B \cdot S \cdot k_{kv}}{BW_{\text{eff}}}
$$

8×H100 的聚合峰值带宽是 26.8 TB/s，按 40% 的实测带宽利用率（MBU）取 10.7 TB/s，8K 上下文：

| Batch | 每步字节 | TPOT | 吞吐（token/s） | KV 占用 |
| --- | --- | --- | --- | --- |
| 1 | 142.6 GB | 13.3 ms | 75 | 2.6 GB |
| 8 | 160.8 GB | 15.0 ms | 533 | 21 GB |
| 32 | 223.2 GB | 20.9 ms | 1,531 | 83 GB |
| 64 | 306.4 GB | 28.6 ms | 2,238 | 166 GB |
| 128 | 472.8 GB | 44.2 ms | 2,896 | 333 GB |

KV 超过权重发生在 B ≈ 140 / 2.6 ≈ 54。再往上，吞吐增长主要来自 KV 带宽，权重已经被摊薄；到 batch 128，KV 占每步流量的 70%。

batch 256 需要 666 GB KV 加上 140 GB 权重，超过 8 张卡 640 GB 的总显存，所以表停在 128。**batch 的上限通常由 KV 决定，不是由算力决定。** 表中把 8 张卡当作一个聚合带宽池估算，实际的 TP AllReduce 每层两次，总量不到 1 ms。

这张表也解释了静态批处理为什么浪费：固定 batch 里最短的请求结束后，它的槽位要等到整批结束才能释放，KV 和算力一起空转。连续批处理把等待时间从“整批”降到“一步”，代价是每步都要重新组织 block table 和 attention 的元数据。

## 25.8 Prefill and Decode Performance Characteristics

把两阶段放在同一张表里，差异一目了然：

| 维度 | Prefill | Decode |
| --- | --- | --- |
| 一次的位置数 | 8,192 | 1 |
| GEMM 的 M | 8,192 | 1 |
| FLOPs | 1.24 PFLOP | 140 GFLOP |
| 权重流量 | 140 GB | 140 GB |
| KV 流量 | 写 2.6 GB | 读 2.6 GB（8K） |
| 算术强度 | 约 8,000 | 约 1 |
| 相对脊点 295 | 27 倍以上 | 1/300 |
| 受限资源 | 算力 | HBM 带宽 |
| 时间量级 | 156–240 ms | 13 ms（8 卡），42 ms（单卡下限） |
| 主要手段 | GEMM 效率、FP8、并行 | 量化、batching、KV 压缩 |

两个强度可以写成闭式：

$$
I_{\text{prefill}} \approx \frac{2NS}{2N} = S, \qquad
I_{\text{decode}}(B) = \frac{2NB}{2N + B S k_{kv}}
$$

batch 把 decode 的强度从 1 提到 128 时的 38，但 KV 随 batch 线性增长，强度趋近上限 `2N / (S k) ≈ 52`，仍然远低于 295。**在 FP16 的 GQA KV 下，decode 不会因为增大 batch 变成算力受限。**

改变这个结论要靠压缩。FP8 权重把分子减半，FP8 KV 把分母减半，比值不变；只有对 KV 的压缩比超过对权重的压缩比时，曲线才会往上走。这也是 MLA 和 KV 量化在长上下文场景里收益更大的原因。

两个阶段之外还有第三种状态：batch 很小、上下文很短时，GPU 的并行度不足以填满任何一个上限，时间由 kernel launch 和依赖链决定。这时看 roofline 会得出“远未到顶”的结论，但无论怎么优化，都靠近不了任何一条线。判断方法很简单：把 batch 从 1 加到 8，如果 TPOT 几乎不变，说明本来就不是带宽或算力受限。

## 25.9 TTFT, TPOT and End-to-end Latency

三个指标的定义可以写成：

$$
\text{TTFT} = T_{\text{queue}} + T_{\text{tokenize}} + T_{\text{prefill}} + T_{\text{sample}}
$$

$$
\text{TPOT} = \frac{T_{\text{e2e}} - \text{TTFT}}{O - 1}, \qquad
T_{\text{e2e}} = \text{TTFT} + (O - 1) \cdot \text{TPOT}
$$

| 场景 | 输入 | 排队 | TTFT | TPOT | 500 token 的 E2E |
| --- | --- | --- | --- | --- | --- |
| 空载短请求 | 512 | 0 | 20 ms | 13.3 ms | 6.7 s |
| 空载长请求 | 8K | 0 | 240 ms | 13.3 ms | 6.9 s |
| 中等负载 | 8K | 100 ms | 340 ms | 20.9 ms | 10.8 s |
| 长上下文 | 128K | 0 | 8.0 s | 17.0 ms | 16.5 s |

前两行说明：8K 输入只把 TTFT 拉高 220 ms，对 6.7 秒的端到端几乎无感。第三行说明：TPOT 从 13.3 涨到 20.9 ms，是 batching 摊薄权重的代价，端到端因此多了 3.9 秒。第四行说明：128K 上下文里 attention 占 prefill 的 55%，TTFT 进入秒级；同时单请求 KV 42 GB，8 卡的 KV 池只装得下 9 条这样的请求，长上下文和并发不可兼得。

TPOT 的分位数由最慢的一步决定。一次抢占、一次 KV 换出、一次 batch 重排，都会在那一步留下尖峰，所以 p99 TPOT 往往比 p50 高 2–5 倍。生产上关心的是满足 SLO 的吞吐（goodput），不是裸吞吐：如果 SLO 是 TPOT 小于 50 ms，batch 128 那一档就不该进入统计。

拆解的意义在于定位。TTFT 高，先分清排队还是 prefill：把输入从 8K 降到 1K，如果 TTFT 没怎么变，问题在队列；如果按比例下降，问题在 prefill。TPOT 高，先看 batch 和上下文：同一 batch 下把上下文从 8K 降到 2K，如果 TPOT 明显下降，瓶颈在 KV；如果几乎不变，瓶颈在权重或 kernel 开销。

还有两项常被漏算：网络往返（同区域内 5–30 ms）和首 token 的反分词延迟。它们不占 GPU 时间，却直接加在 TTFT 上。当 TPOT 已经压到 10 ms 量级时，这两项的占比会变得显著，继续优化 GPU 反而看不到端到端收益。

## 本章小结

- 一次请求的时间被切成两段：prefill 处理 S 个位置，decode 每步处理 1 个，两段的单 token 成本相差两个数量级以上。
- Prefill 的 FLOPs 由 `2NS` 和 attention 的 `2 n_h L S^2 d` 组成；8K 时前者占 93%，128K 时后者反超。
- Prefill 算术强度约等于 S，是算力受限；decode 算术强度约等于 1，是带宽受限。
- 70B FP16 权重 140 GB，单卡 decode 的带宽下限约 42 ms/token；TP=8 降到 5.2 ms，实测 10–15 ms。
- 自回归循环不可并行，单步成本由权重读取、KV 读取、kernel launch 和采样构成；优化只有减字节、减步数、合并步三条路。
- Batching 摊薄权重但不摊薄 KV；8K 上下文的交叉点在 batch 54 附近，batch 上限由 KV 容量而不是算力决定。
- TTFT 由排队和 prefill 决定，TPOT 由 decode 的每步字节决定，端到端是两者之和；长上下文的 TTFT 会进入秒级。

## 延伸阅读

- Pope et al., [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102), 2022.
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), 2023.
- Yu et al., [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu), OSDI 2022.
- Zhong et al., [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670), 2024.
- Agrawal et al., [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310), 2024.
- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), 2022.
- Grattafiori et al., [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783), 2024.
- NVIDIA, [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).
- vLLM Project, [vLLM: Easy, Fast, and Cheap LLM Serving](https://github.com/vllm-project/vllm).
