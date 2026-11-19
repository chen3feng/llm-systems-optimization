---
chapter: 1
part: 1
title: Introduction to LLM Systems
level: tbd
status: draft
prerequisites: []
lab:
---

# Chapter 1. Introduction to LLM Systems

一张 80 GB 的 H100，装不下一个 175B 参数的模型。

GPT-3 的权重用 FP16 存，需要 350 GB。要把模型放进去，至少得切到 5 张卡上；要让 100 个用户同时使用，还要再算上 KV Cache 和中间激活。模型没有变，变的是它从论文里的一个函数，变成了一个需要调度、通信、内存管理和故障恢复的系统。

这本书讲的就是这个系统。

## 1.1 Evolution from Neural Networks to LLM Systems

1958 年，Frank Rosenblatt 造出 Mark I Perceptron。它只有一层可学习的权重，却能通过调整连接强度完成简单的分类。当时的报道说它终将学会走路、说话、看东西。后来的故事大家都知道：感知机被证明连 XOR 都学不会，神经网络第一次进入冬天。

真正的转折发生在半个世纪之后。

1986 年，Rumelhart、Hinton 和 Williams 把反向传播算法系统地用于多层网络。神经网络重新变得可训练，但算力和数据都还不足。直到 2012 年，AlexNet 在 ImageNet 上把错误率从 26% 拉到 15%，人们才意识到 GPU 加深度网络意味着什么。AlexNet 只有 6000 万参数，用了两张 3 GB 的 GTX 580 训练。

2017 年，[Attention Is All You Need](https://arxiv.org/abs/1706.03762) 提出 Transformer。它去掉循环结构，把序列建模变成矩阵乘法和注意力计算。这个架构有一个当时未必被充分认识的特点：它对硬件非常友好——计算可以拆成大规模 GEMM，序列可以并行处理。

2020 年，[GPT-3](https://arxiv.org/abs/2005.14165) 用 175B 参数、300B token、约 $3.14 \times 10^{23}$ FLOPs 证明了“把同一个 Transformer 放大”可以带来跨任务的通用能力。2022 年底，ChatGPT 让这套能力进入普通用户的世界。

下表是这条路线上的几个关键节点：

| 年份 | 系统 / 模型 | 规模 | 当时最棘手的系统问题 |
| --- | --- | --- | --- |
| 1958 | Perceptron | 单层权重 | 如何用硬件实现学习 |
| 1986 | 多层网络 + 反向传播 | 数千参数 | 梯度计算与训练稳定性 |
| 2012 | AlexNet | 60M 参数 | 两张 GPU 上的模型切分 |
| 2017 | Transformer | 65M / 213M 参数 | 注意力计算的显存与带宽 |
| 2020 | GPT-3 | 175B 参数 | 模型并行、训练稳定性、数千张 GPU |
| 2024 | Llama 3 405B | 405B 参数、15.6T token | 16K H100 的集群、通信、故障恢复 |

规模每上一个台阶，瓶颈就换一个地方。

2012 年的瓶颈是“能不能用 GPU 训练”。2020 年的瓶颈是“能不能把 175B 参数放进一个集群，并让几千张卡一起有效工作”。到了推理阶段，瓶颈又变成“能不能在可接受的延迟和成本下，把模型服务给几百万用户”。

### 架构收敛之后，竞争转移到系统

Transformer 之后，模型架构没有再次发生同等量级的范式变化。GPT、Llama、Qwen、DeepSeek 的主力模型都还是 decoder-only Transformer，差别集中在注意力变体、归一化位置、激活函数、MoE 路由这些局部设计上。

这对写书是一件好事：我们不需要为每一种新架构重写一套系统原理。矩阵乘法、注意力、KV Cache、通信、调度，这些底层机制是共通的。

对工程师则意味着一件事：**模型结构和论文越来越公开，系统实现的质量差距反而越来越大。**

同一个 Llama 70B，用 Hugging Face `transformers` 的 `generate()`、vLLM、SGLang、TensorRT-LLM、llama.cpp 跑，吞吐量可以差出几倍到几十倍，首 token 延迟和单 token 成本也完全不同。模型权重是同一份，差别在运行时。

2020 年，[Kaplan 等人](https://arxiv.org/abs/2001.08361) 给出缩放定律：模型损失随参数量、数据量和计算量呈幂律下降。2022 年，[Chinchilla](https://arxiv.org/abs/2203.15556) 进一步指出，在固定计算预算下，参数量和数据量应该大致按 1:20 的比例增长。

缩放定律的意义不只是“模型越大越好”。它把模型质量变成了一个可以预估的工程预算：给定算力，能训练多大的模型、需要多少数据、预期损失是多少。既然质量可以预估，剩下的问题就变成了：**用什么样的系统，把这个预算高效地花出去。**

这就是本书的起点。

## 1.2 LLM System Architecture

一个 LLM 系统可以分成六层。每一层解决不同的问题，也对应不同的性能瓶颈。

| 层 | 职责 | 代表技术 |
| --- | --- | --- |
| 服务层 | 多租户、路由、弹性伸缩、可观测性 | Kubernetes、API Gateway、Prometheus |
| Runtime 层 | 批处理、KV Cache、内存分配、请求调度、采样 | vLLM、SGLang、TensorRT-LLM、llama.cpp |
| 框架层 | 模型定义、自动微分、训练循环 | PyTorch、JAX |
| 编译 / Kernel 层 | 算子实现、图优化、内存规划 | CUDA、CUTLASS、Triton、TorchInductor、TensorRT |
| 硬件层 | 算力、显存、互联 | H100、HBM、NVLink、InfiniBand |
| 模型层 | 权重、结构、tokenizer | Llama、Qwen、DeepSeek、GPT |

一次推理请求会穿过这些层：

```text
用户请求
   │
   ▼
API Gateway / Router
   │
   ▼
Serving Runtime
   ├── Tokenizer
   ├── Scheduler ──────► KV Cache Manager
   ├── Model Executor ─► CUDA Kernels / Compiled Graph
   └── Detokenizer
   │
   ▼
GPU / CPU / NPU
```

以 vLLM 为例，一次请求的生命周期大致是：

1. HTTP 层收到请求，tokenizer 把 prompt 转成 token id。
2. Scheduler 判断当前显存和 batch 预算，决定这一轮把哪些请求放进 batch。
3. KV Cache Manager 为每个请求分配 block，建立逻辑 token 位置到物理显存的映射。
4. Model Executor 调用 prefill kernel 处理 prompt，然后进入 decode 循环。
5. 每生成一个 token，detokenizer 把 token 转回文本，通过流式接口返回。
6. 请求结束，KV Cache block 被回收；如果开启 prefix caching，可复用的块会保留。

这里每一层都有自己的“优化语言”。Kernel 工程师关心 shared memory 和寄存器；Runtime 工程师关心 batch 组成和显存碎片；Serving 工程师关心 P99 延迟和副本数。**系统性能是这些层共同作用的结果，任何一层的短板都会把其他层的优化吃掉。**

一个常见的例子：Kernel 把 Attention 加速了 2 倍，但 Runtime 的 batch 组成很差，GPU 利用率仍然只有 30%。端到端吞吐不会翻倍。反过来，Runtime 把 batch 做得很大，但 KV Cache 频繁换入换出，Kernel 再快也没用。

### 边界正在变得模糊

上面这张分层图是理解系统的工具，不是严格的工程边界。

PyTorch 2.x 的 TorchInductor 把框架和编译器揉在了一起。vLLM 为了性能，自己写了 PagedAttention kernel、CUDA graph 捕获和自定义算子。TensorRT-LLM 同时包含编译器、kernel 库和 runtime。SGLang 把前缀缓存、结构化生成和调度放在同一个运行时里。

所以读这本书时，不必纠结“这个功能属于哪一层”。更重要的是问：**它解决的是哪个瓶颈？代价是什么？**

## 1.3 Model、Framework、Runtime and Serving

这四个词经常被混用。先把它们分开。

| 概念 | 回答的问题 | 典型代表 | 主要优化目标 |
| --- | --- | --- | --- |
| Model | 要计算什么 | Llama 3、Qwen、DeepSeek | 质量、能力、参数效率 |
| Framework | 如何表达和训练计算 | PyTorch、JAX | 开发效率、自动微分、算子覆盖 |
| Runtime | 如何在负载下高效执行 | vLLM、SGLang、TensorRT-LLM | 吞吐、延迟、显存利用率 |
| Serving | 如何稳定地对外提供服务 | Kubernetes、网关、可观测系统 | 可用性、弹性、成本、SLA |

### Model

模型是权重、计算图和 tokenizer 的组合。它定义了“给定输入 token，输出下一个 token 的概率分布”这件事。

对系统工程师来说，需要从模型里提取的是**计算特征**，而不是论文里的每个设计细节：

- 有多少层、每层有哪些算子；
- hidden dimension、注意力头数、KV 头数；
- 参数量、数值精度；
- 训练时用了什么并行策略、推理时有哪些结构可以利用。

例如，Llama 3 70B 有 80 层、64 个注意力头、8 个 KV 头（GQA）、head dimension 128。GQA 把 KV Cache 缩小到 MHA 的八分之一，这个结构选择直接决定了推理系统的显存预算。

### Framework

框架负责“把模型写出来，并让它可训练”。PyTorch 的 eager 模式让每个算子立即执行，调试直观；自动微分负责反向传播；分布式模块负责把计算切到多张卡上。

框架的强项是表达能力和生态，不是极限性能。PyTorch 的 `generate()` 是一个正确的参考实现，但它不是为高并发服务设计的：

```python
for _ in range(max_new_tokens):
    logits = model(input_ids)
    next_token = sample(logits[:, -1, :])
    input_ids = torch.cat([input_ids, next_token], dim=-1)
```

这段代码每个 token 都要跑一次完整的 forward。如果没有 KV Cache，每一步都会重新计算整个前缀的注意力；如果 batch size 是 1，GPU 的大部分计算单元在等内存。它能跑，但离“高效服务”很远。

### Runtime

Runtime 关心的是：在给定硬件和 SLA 的前提下，如何让尽可能多的请求跑得尽可能快。

它通常负责这些事：

- **KV Cache 管理**：分配、复用、回收、前缀共享；
- **批处理**：continuous batching、chunked prefill、优先级调度；
- **内存管理**：PagedAttention、block allocator、显存碎片控制；
- **执行优化**：CUDA Graph、算子融合、kernel 选择；
- **采样**：temperature、top-p、top-k、beam search、结构化输出。

[vLLM 的论文](https://arxiv.org/abs/2309.06180) 给出了一个常被引用的数字：在相同延迟下，PagedAttention 把吞吐量做到 Orca 的 2–4 倍；相比 Hugging Face Transformers，吞吐量最高可提升 24 倍。这个差距不是来自模型，而是来自 Runtime 对显存和 batch 的管理方式。

### Serving

Serving 是 Runtime 外面的那一层：多模型路由、副本管理、自动扩缩容、限流、灰度发布、监控和故障恢复。

一个推理服务真正的成本，往往由 Serving 层决定：

- 副本开少了，P99 延迟超标；
- 副本开多了，GPU 利用率低；
- 没有请求路由，长 prompt 把短请求堵死；
- 没有健康检查和优雅退出，滚动发布时大量请求失败。

对这本书来说，Serving 不是重点，但会反复出现。因为很多“系统优化”最终要落到一个可运维的服务上，而不是一个 benchmark 脚本里。

## 1.4 Training、Inference and Post-training Workloads

LLM 的生命周期有三个阶段，它们的计算特征完全不同。

| 阶段 | 目标 | 计算特征 | 主要瓶颈 | 典型系统 |
| --- | --- | --- | --- | --- |
| Pre-training | 从海量文本学习下一个 token 分布 | 大规模 GEMM、反向传播 | 算力、通信、训练稳定性 | Megatron-LM、DeepSpeed、FSDP |
| Post-training | 对齐、指令跟随、偏好优化 | SFT、DPO、RL rollout | 数据质量、 rollout 推理、显存 | PyTorch、TRL、verl |
| Inference | 给定 prompt 生成 token | Prefill + Decode | 显存带宽、KV Cache、调度 | vLLM、SGLang、TensorRT-LLM |

### Pre-training

预训练是“用算力换能力”。每一步都要对一批 token 做前向和反向传播，计算量巨大，通常需要几千张 GPU 跑几周。

以 GPT-3 为例：175B 参数、300B token、约 $3.14 \times 10^{23}$ FLOPs。按 H100 的 FP16 峰值 989 TFLOPS 算，即使达到 50% 的利用率，也需要约 7.2 天——这是在理想情况下、且假设所有算力都能用于有效计算。真实训练还要处理通信、流水线气泡、故障恢复和数据加载。

预训练的显存开销也不只是权重。混合精度训练通常需要：

| 内容 | 每参数字节数 |
| --- | --- |
| FP16 权重 | 2 |
| FP16 梯度 | 2 |
| FP32 master weights | 4 |
| Adam 一阶矩 | 4 |
| Adam 二阶矩 | 4 |
| **合计** | **16** |

70B 参数的模型，仅优化器状态就要约 1.1 TB。这就是为什么预训练必须做 ZeRO、FSDP、张量并行和流水线并行——单卡放不下，多卡之间还要频繁通信。

### Post-training

Post-training 包括监督微调（SFT）、偏好优化（DPO、RLHF、GRPO）和评测。它的计算量通常比预训练小几个数量级，但系统问题并不简单：

- SFT 的样本长度差异很大，batch 内 padding 会浪费算力；
- RL 训练需要先做 rollout（推理），再做训练，推理和训练的资源要动态分配；
- 偏好数据的质量比数量更重要，数据管道本身就是一个系统；
- 评测要可复现，否则无法判断模型是变好了还是运气好。

对系统工程师来说，Post-training 最值得关注的是 **rollout 推理**：它把推理系统的延迟和吞吐直接带进了训练循环。一个 RL 训练任务里，rollout 可能占用大部分 GPU 时间。

### Inference

推理分为两个阶段：

- **Prefill**：处理输入 prompt，把所有 token 的 Key/Value 算出来并写入 KV Cache。这个阶段是计算密集的，适合大 batch、长序列。
- **Decode**：逐 token 生成。每一步只处理一个新 token，但需要读取全部模型权重和 KV Cache。这个阶段是显存带宽密集的。

用公式表示，prefill 的 FLOPs 大致是：

$$
\text{FLOPs}_{\text{prefill}} \approx 2 \times N \times S
$$

其中 $N$ 是参数量，$S$ 是输入 token 数。decode 每生成一个 token 的 FLOPs 大致是：

$$
\text{FLOPs}_{\text{decode}} \approx 2 \times N
$$

但 decode 每步要读取约 $2N$ 字节的 FP16 权重（忽略 KV Cache），所以算术强度大约是：

$$
\frac{2N \text{ FLOPs}}{2N \text{ bytes}} \approx 1 \text{ FLOP/byte}
$$

H100 的 FP16 峰值是 989 TFLOPS，HBM 带宽是 3.35 TB/s，两者的比值（roofline 的“脊点”）约为 295 FLOP/byte。**decode 的算术强度比脊点低两个数量级，所以它几乎完全受显存带宽限制。**

这解释了一个反直觉的现象：单请求 decode 时，GPU 的算力大量闲置。Llama 3 70B 用 FP16 存权重约 140 GB，单 token 至少要读一遍权重；在 3.35 TB/s 的带宽下，理论下限是 42 ms/token，也就是约 24 token/s。想让吞吐更高，必须增大 batch，让同一份权重服务更多请求。

KV Cache 的显存也要算清楚。每个 token 的 KV Cache 大小是：

$$
2 \times L \times H_{\text{kv}} \times d \times \text{bytes}
$$

其中 $L$ 是层数，$H_{\text{kv}}$ 是 KV 头数，$d$ 是 head dimension，系数 2 表示 Key 和 Value。

- GPT-3（96 层、96 个 KV 头、head dim 128、FP16）：约 **4.5 MiB/token**，2048 token 就是 9 GiB。
- Llama 3 70B（80 层、8 个 KV 头、head dim 128、FP16）：约 **320 KiB/token**，8192 token 就是 2.5 GiB。

这就是 GQA 的价值：同样的上下文长度，KV Cache 缩小到八分之一，单卡能容纳的并发请求数直接翻几倍。

推理系统通常用三个指标描述性能：

- **TTFT（Time To First Token）**：从请求到第一个 token 返回的时间，主要受 prefill 和排队影响。
- **TPOT（Time Per Output Token）**：生成阶段每个 token 的平均时间，主要受 decode 带宽和 batch 影响。
- **Throughput**：单位时间处理的 token 数或请求数，通常用 tokens/s 或 requests/s 表示。

这三个指标互相牵制。增大 batch 可以提高吞吐，但会让 TTFT 和 TPOT 变差；减小 batch 可以降低延迟，但 GPU 利用率下降。Runtime 的核心工作，就是在这个三角里找平衡。

## 1.5 The LLM Optimization Landscape

LLM 优化不是一个技术清单，而是一组针对不同瓶颈的手段。

先看硬件。Roofline 模型把性能上限写成：

$$
\text{Performance} = \min(\text{Peak Compute}, \text{Bandwidth} \times \text{Arithmetic Intensity})
$$

| 硬件 | FP16 峰值 | HBM 带宽 | 脊点（FLOP/byte） |
| --- | --- | --- | --- |
| A100 80GB | 312 TFLOPS | 2.0 TB/s | 156 |
| H100 SXM | 989 TFLOPS | 3.35 TB/s | 295 |

算术强度低于脊点的算子受带宽限制，高于脊点的算子受算力限制。LLM 的大多数算子——尤其是 decode 阶段的 GEMM 和 Attention——都落在带宽受限的一侧。

所以优化方向可以按“把瓶颈移到哪里”来分类：

| 层次 | 手段 | 主要收益 | 主要代价 |
| --- | --- | --- | --- |
| 模型 | 蒸馏、剪枝、MoE、模型合并 | 减少参数量或激活计算 | 质量损失、结构复杂化 |
| 数值 | FP16、BF16、FP8、INT8、INT4 | 降低显存和带宽 | 精度损失、校准成本 |
| Kernel | FlashAttention、算子融合、CUTLASS | 提高算力利用率 | 实现复杂、硬件绑定 |
| Runtime | KV Cache、PagedAttention、continuous batching | 提高并发和显存利用率 | 调度复杂、尾延迟 |
| 分布式 | 张量并行、流水线并行、专家并行 | 突破单卡容量 | 通信开销、故障域扩大 |
| 服务 | 路由、扩缩容、优先级队列 | 降低成本、满足 SLA | 运维复杂、资源冗余 |

几个关键技术的出现时间，基本对应了 LLM 系统优化的几个阶段：

- **2017，Transformer**：把序列计算变成矩阵乘法，GPU 成为主战场。
- **2020，GPT-3**：模型大到一个 GPU 放不下，模型并行和训练系统成为必需。
- **2022，FlashAttention**：通过 tiling 和 online softmax 减少 HBM 读写，把 Attention 从带宽瓶颈里拉出来一部分。
- **2022，Orca**：引入 iteration-level scheduling，也就是后来常说的 continuous batching。
- **2023，PagedAttention / vLLM**：用分页管理 KV Cache，解决显存碎片和共享问题。
- **2023 之后， speculative decoding、chunked prefill、prefix caching、FP8、MoE 推理**：在延迟、吞吐和成本之间继续做细分优化。

这些技术不是互相替代的。量化会改变 kernel 的选择，batch 会改变 KV Cache 的压力，张量并行会改变通信和计算的平衡。**系统优化的难点不在于知道某个技术，而在于理解它们之间的相互作用。**

## 1.6 Quality、Performance、Cost and Reliability Trade-offs

LLM 系统没有“全面最优”的配置。每一个优化都在四个维度之间做交换：

| 维度 | 关注点 | 典型指标 |
| --- | --- | --- |
| Quality | 模型能力是否保持 | 准确率、评测集、长上下文表现 |
| Performance | 用户感知的速度 | TTFT、TPOT、P99 延迟、吞吐 |
| Cost | 每个 token 的成本 | GPU 小时、$/1M tokens、显存占用 |
| Reliability | 服务是否稳定 | 可用性、错误率、故障恢复时间 |

几个具体的交换：

**FP8 量化**。权重和 KV Cache 的带宽减半；在 decode 这种带宽受限的场景，吞吐上限接近翻倍，实际通常在 1.3–1.8 倍之间。但 FP8 的动态范围比 BF16 窄，遇到激活值离群点会导致精度下降。实际系统需要 per-tensor 或 per-channel scaling，甚至对某些层保留 BF16。

**增大 batch**。吞吐上升，GPU 利用率提高。但请求排队时间变长，TTFT 和 TPOT 都会变差。Continuous batching 缓解了这个问题，但没有消除它：batch 越大，单个请求的尾延迟越难控制。

**投机解码**。用小模型快速草拟多个 token，再用大模型一次验证。延迟可以降低 1.5–2 倍，但额外增加了草拟计算；如果接受率低，收益会被抵消。接受率取决于草稿模型和目标模型的一致性，不是一个固定常数。

**增加副本**。可用性和延迟都会改善，但成本线性上升。用 spot instance 可以降低成本，但需要处理随时被回收的风险；用多可用区部署可以提高可靠性，但跨区带宽和一致性又成为新问题。

**加长上下文**。模型能力增强，但 KV Cache 随上下文线性增长，prefill 的注意力计算随长度平方增长。128K 上下文仅 KV Cache 就可能达到几十 GB；对 MHA 模型，单条请求占满一张 80 GB 的卡并不罕见。

这些交换有一个共同点：**你无法同时优化所有维度，只能根据场景选择放弃哪一个。**

聊天场景更在意 TTFT 和 TPOT，批量离线推理更在意吞吐和成本，Agent 场景更在意长上下文和工具调用的稳定性。同一个模型，不同的 SLA，需要不同的系统配置。

## 本章小结

- LLM 系统不是“模型加一个 API”。它包含模型、框架、Runtime、Kernel、硬件和服务六个层次，每一层都有自己的瓶颈。
- Transformer 之后，模型架构趋于稳定，系统实现的质量成为主要差距来源。
- Prefill 是计算密集的，decode 是显存带宽密集的；decode 的算术强度约为 1 FLOP/byte，远低于 H100 的 295 FLOP/byte 脊点。
- KV Cache 是推理系统最重要的显存开销。GQA、PagedAttention、prefix caching 都是在不同层面控制这笔开销。
- 优化不是消灭瓶颈，而是把瓶颈移动到代价更低的地方。Quality、Performance、Cost、Reliability 四者不可兼得。

下一章从 Transformer 的计算图出发，把“模型在算什么”拆成可以度量、可以实现、可以优化的计算过程。

## 延伸阅读

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
- Brown et al., [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), 2020.
- Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), 2020.
- Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), 2022.
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), 2023.
- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), 2022.
- Yu et al., [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu), OSDI 2022.
- Williams et al., [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://dl.acm.org/doi/10.1145/1498765.1498785), 2009.
- Meta AI, [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783), 2024.
- NVIDIA, [H100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h100/).
- [vLLM](https://github.com/vllm-project/vllm)、[SGLang](https://github.com/sgl-project/sglang)、[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)、[llama.cpp](https://github.com/ggml-org/llama.cpp).
