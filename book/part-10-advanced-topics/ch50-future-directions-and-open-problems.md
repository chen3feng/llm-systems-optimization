---
chapter: 50
part: 10
title: Future Directions and Open Problems
level: tbd
status: draft
prerequisites: []
lab:
---

# Chapter 50. Future Directions and Open Problems

前面四十九章讲的都是“已经能用”的技术。这一章看还在变化的部分：硬件、数值格式、模型结构、部署形态。它们中的大部分不会立刻改变今天的工作方式，但它们决定了未来几年系统设计的边界在哪里。

写这一章的风险是很快过时。所以重点不放在具体型号和参数上，而放在**趋势背后的物理约束**：什么在变快，什么没有变快，瓶颈会往哪里移动。

## 50.1 Emerging GPU Architectures

过去几年 GPU 的演进有一条清晰的线：**算力涨得比带宽快，带宽涨得比容量快**。

| 指标 | 每代提升 | 后果 |
| --- | --- | --- |
| 张量算力 | 2–3× | 计算更容易“过剩” |
| HBM 带宽 | 1.5–2× | decode 仍受带宽限制 |
| HBM 容量 | 1.5× | KV Cache 仍是稀缺资源 |
| 互联带宽 | 1.5–2× | 跨节点并行仍受通信限制 |

这个不对称解释了本书反复出现的结论：decode 是访存受限，prefill 是计算受限，而两者的比例会随硬件演进而变化。算力涨得快，意味着同样的模型在 decode 阶段更“饿”带宽；也意味着低精度（FP8、FP4）的收益会越来越大，因为降低精度直接减少要搬的字节。

几个具体的架构趋势：

- **更大的片上存储和更强的异步能力**：TMA、warp specialization、thread block cluster，都是在减少数据搬运的启动开销；
- **更低的精度**：从 FP16 到 FP8 到 FP4，每一步都把带宽压力减半，代价是数值范围变窄、需要更精细的缩放；
- **更密的互联**：单机内的 NVLink 域越来越大，跨机用更快的网络（800G、1.6T），让 TP 和 EP 的边界外移；
- **专用单元**：除了矩阵乘，出现了专门的 attention 单元、通信引擎、压缩引擎。

对系统工程师的含义是：**优化手段的优先级会变**。今天 fusion 和 kernel 调优能带来 30% 的收益，未来可能降到 10%，因为硬件把很多开销藏掉了；而数据布局、精度选择、通信调度的权重会上升。

## 50.2 Alternative Accelerators

GPU 不是唯一的选择。几类加速器在不同维度上做了不同的取舍：

| 类型 | 代表 | 优势 | 局限 |
| --- | --- | --- | --- |
| 脉动阵列 ASIC | TPU、Trainium | 能效高、规模化成本低 | 编程模型受限、生态弱 |
| 确定性数据流 | Groq | 延迟极低且稳定 | 容量小、依赖片外存储 |
| 晶圆级 | Cerebras | 片上带宽极高 | 单机规模受限、成本高 |
| 存内计算 | 各类原型 | 消除访存瓶颈 | 精度和工艺尚不成熟 |

从 roofline 的角度看，这些架构的差异本质上是**算术强度的工作点不同**。GPU 的算力/带宽比大约 300 FLOP/byte（H100：989 TFLOPS / 3.35 TB/s）；LLM decode 的算术强度只有 1–2 FLOP/byte，远在拐点左侧，所以任何提高带宽或降低精度的方案都直接有效。

Groq 这类确定性架构的思路是：既然 decode 是访存受限，那就把访存做成完全可预测的，用巨大的片上 SRAM 和静态调度消除不确定性。代价是容量小（几百 MB），模型必须放片外，靠极高的带宽拉回来。

对本书的读者来说，重要的不是记住这些型号，而是理解：**LLM 推理的瓶颈在带宽，任何在带宽上做文章的架构都有机会**。

## 50.3 New Numerical Formats

数值格式的演进方向是“用更少的比特表示同样的信息，同时保持训练和推理的稳定性”。

当前的格局：

| 格式 | 位宽 | 动态范围 | 主要用途 |
| --- | --- | --- | --- |
| BF16 | 16 | 大 | 训练、推理基线 |
| FP8 E4M3 | 8 | 中 | 推理权重和激活 |
| FP8 E5M2 | 8 | 大 | 梯度 |
| MXFP4 / NVFP4 | 4 | 小（带块缩放） | 权重量化 |
| INT4 | 4 | 对称 | 权重量化 |

**微缩放格式**（microscaling，MX）是近年的关键进展：每 32 个元素共享一个 8 位指数，元素本身用 4 位或 6 位。这样既保留了块内的相对精度，又能覆盖较大的动态范围。FP4 的精度损失在权重上可以接受，在激活上仍有争议。

数值格式对系统的影响不止是带宽：

- **kernel 复杂度**：块缩放需要在 kernel 里做反量化，增加了寄存器压力和指令数；
- **累加精度**：低精度乘、高精度累加是标配，但累加器的位宽和顺序会影响结果的可复现性；
- **校准成本**：每种格式都需要一套缩放因子的计算和校准流程，格式越新，工具链越不成熟。

一个值得关注的开放问题是**训练的数值格式**。推理用 FP4 已经在部分场景落地，但训练用 FP8 仍在推广中，FP4 训练还停留在研究阶段——梯度对精度更敏感，且误差会在反向传播中累积。

## 50.4 Emerging Attention and Memory Architectures

Transformer 的 `O(S²)` attention 和线性增长的 KV Cache 是两个长期瓶颈。替代方案分三条路线。

**线性 attention 与状态空间模型**：用固定大小的状态代替不断增长的 KV。Mamba 这类 SSM 的复杂度是 `O(S)`，推理时每步只需要更新一个状态，显存恒定。代价是“精确回忆”能力弱——它擅长压缩长序列的整体信息，不擅长精确定位某个位置的内容。

**混合架构**：大部分层用线性 attention 或 SSM，少数层保留完整 attention。这样既有长序列的效率，又保留了精确检索能力。这是目前最务实的路线，多个开源模型采用了这种结构。

**KV 共享与压缩**：MLA 把 KV 压缩成低秩表示，GQA 让多个 query 头共享 KV 头。它们的共同点是用表达能力的少量损失换取显存和带宽的大幅下降。MLA 在 70B 规模上能把每 token 的 KV 从 320 KiB 压到几十 KiB。

对系统的影响很直接：如果 KV 不再随序列线性增长，paged KV、prefix cache、KV 换出这些机制的必要性都会下降，调度器会简化；但 attention 的计算模式会变，kernel 要重写。这也是为什么新架构的落地总是比论文慢一两年——模型结构变了，整个系统栈都要跟着变。

## 50.5 Disaggregated Compute and Memory

当前的部署形态是“计算和存储绑在一起”：一张卡有自己的 HBM，KV Cache 就在这张卡上。随着上下文变长、并发变高，这个绑定开始成为约束。

**PD 分离**已经在 Chapter 47 讲过：prefill 和 decode 用不同的资源池，KV 通过网络传输。

更彻底的是**KV Cache 的池化**：把整个集群的 CPU 内存、NVMe 甚至远端 HBM 组织成一层 KV 存储，decode 时按需拉取。Mooncake 这类系统的做法是把 KV 当作一级存储资源来调度，prefill 节点算完就写进池，decode 节点按需读。

这个方向的物理基础是**内存层级的价格差**：HBM 每 GB 几十美元，CPU 内存几美元，NVMe 几美分。把冷 KV 放到便宜的层级，用带宽换容量，是经济上必然的选择。

技术难点在**延迟和带宽的匹配**。decode 一步只有 10 ms，如果拉取 KV 要 5 ms，延迟就翻倍。所以池化 KV 只在两种场景可行：一是上下文很长、KV 很大，拉取的相对开销小；二是用 RDMA 或 CXL 这类低延迟互联，把访问延迟压到微秒级。

**CXL** 是这个方向的关键技术：它让 CPU 和加速器共享内存池，访问延迟比网络低一个数量级。目前 CXL 的带宽还不足以支撑高频访问，但作为 KV 的二级存储已经有实际价值。

## 50.6 New Runtime Architectures

运行时的演进方向是**减少 CPU 侧的介入**和**提高设备利用率**。

**Megakernel**：把整个模型的所有层编译成一个巨大的 kernel，层与层之间不回到 CPU，数据留在寄存器或 shared memory 里。这消除了所有 launch 开销和中间张量的显存往返，但要求模型结构固定、形状固定，且寄存器压力极大。目前的实现主要针对小模型或 decode 阶段。

**持久化 kernel**：kernel 常驻 GPU，从队列里取任务，避免反复启动。它和 megakernel 解决的是同一类问题，但更灵活。

**服务时编译**：把编译推迟到部署时，根据实际的模型、硬件和流量形状生成专用代码。TensorRT-LLM 的 ahead-of-time 编译就是这个思路。代价是编译时间长、灵活性差，换一个 batch 形状可能要重新编译。

**异构调度**：把 tokenize、采样、后处理放到 CPU 或 DPU 上，GPU 只做矩阵运算。这在端到端延迟里能省掉几毫秒。

这些方向的共同逻辑是：GPU 越来越快，CPU 侧的固定开销占比越来越高，所以要把 CPU 从关键路径上赶走。

## 50.7 Open Research Problems

最后列几个本书没有答案、但值得持续关注的问题。

**1. 长上下文的成本曲线**。128K 已经可用，1M 正在出现，但成本是超线性的。稀疏 attention、KV 压缩和检索式记忆的组合能否把成本压到线性？目前没有定论。

**2. 推理的能效**。数据中心的电力约束正在成为比芯片供应更硬的限制。每 token 的焦耳数会像今天的每 token 成本一样成为核心指标。

**3. 可靠性的形式化**。LLM 服务的 SLO 目前靠压测和经验，缺乏像传统分布式系统那样的排队模型和故障模型。如何为“生成质量”这种概率性输出定义 SLO，仍是开放问题。

**4. 训练与推理的协同设计**。模型结构越来越受推理成本驱动（GQA、MLA、MoE），但训练侧的约束（通信、显存、数值稳定性）又会限制这些选择。联合优化两边的工作才刚刚开始。

**5. 异构集群的调度**。一个集群里混有不同代际的 GPU、不同的互联拓扑、甚至不同的加速器。如何在这种异构性下保持高利用率和公平性，现有的调度器都还做不到。

**6. 可验证的性能模型**。本书给出了很多估算公式（FLOPs、带宽、通信量），但它们的误差经常在 2 倍以内。能否建立更精确的、能预测实际性能的模型，仍然依赖大量的实测数据。

## 本章小结

- GPU 的算力涨得比带宽和容量快，这决定了 decode 长期访存受限，低精度和带宽优化会持续增值。
- 替代加速器的本质差异是算术强度的工作点不同；LLM decode 的算术强度只有 1–2 FLOP/byte，任何降低带宽需求的架构都有机会。
- 数值格式向更低精度和微缩放演进，FP4 推理已落地，FP4 训练仍是开放问题；格式变化会带来 kernel 复杂度和校准成本。
- 线性 attention、混合架构和 KV 压缩分别从复杂度、精度和表示三个方向缓解长序列问题，落地慢是因为整个系统栈要跟着改。
- 计算与内存的分离由内存层级的价格差驱动，CXL 和 RDMA 决定 KV 池化的延迟能否被接受。
- 运行时的演进方向是减少 CPU 介入：megakernel、持久化 kernel、服务时编译、异构调度。
- 开放问题集中在长上下文成本、能效、可靠性形式化、训练推理协同设计、异构调度和可验证的性能模型。

## 延伸阅读

- Dao and Gu, [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060), ICML 2024.
- Gu and Dao, [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752), 2023.
- DeepSeek-AI, [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434), 2024.
- Qin et al., [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079), 2024.
- Rouhani et al., [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537), 2023.
- Jouppi et al., [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/abs/1704.04760), ISCA 2017.
- Williams et al., [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://dl.acm.org/doi/10.1145/1498765.1498785), CACM 2009.
