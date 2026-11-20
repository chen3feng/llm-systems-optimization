---
chapter: 49
part: 10
title: Specialized Workloads
level: tbd
status: draft
prerequisites: [27, 29, 47]
lab:
---

# Chapter 49. Specialized Workloads

通用的“输入 prompt、输出文本”只是 LLM 服务的一部分。真实系统里还有检索增强的问答、多轮调用工具的 agent、必须返回合法 JSON 的结构化输出、图文混合的请求、以及要求端到端 300 毫秒的语音对话。

这些负载对系统的要求和普通文本生成不同：RAG 的瓶颈在检索和长 prompt 的 prefill，agent 的瓶颈在多轮之间的 KV 复用和工具等待，结构化生成的瓶颈在解码约束的开销，多模态的瓶颈在视觉编码器。这一章逐一分析它们的资源画像和优化点。

## 49.1 RAG Workloads

检索增强生成（retrieval-augmented generation，RAG）把外部文档检索出来塞进 prompt。它的延迟预算被切成几段：

| 阶段 | 典型耗时 | 瓶颈 |
| --- | --- | --- |
| 查询改写 | 50–150 ms | 一次短生成 |
| 向量检索 | 10–50 ms | 索引规模、并发 |
| 重排（rerank） | 30–100 ms | 交叉编码器的 batch |
| 拼接与 prefill | 200–800 ms | 文档长度 |
| 生成 | 500–3000 ms | 输出长度 |

从系统角度看，RAG 有两个特殊之处。

**prompt 长且高度重复**。同一个知识库的文档会被不同用户反复检索到，拼进不同请求的 prompt。这意味着 prefix cache 的命中率可以很高——如果路由能把这些请求聚到同一实例上，命中的部分可以直接跳过 prefill。实践中，把检索结果按文档 ID 排序后再拼接，能显著提高公共前缀的长度。

**质量对检索结果极度敏感**。检索召回 3 篇和召回 10 篇，答案质量的差别可能比换模型还大。但这部分不在 GPU 上——它在向量数据库和重排模型上。系统工程师常犯的错误是只优化生成，忽略了检索侧的 100 ms 可能才是 P99 的主要来源。

RAG 和长上下文是两条竞争路线。把整个知识库放进 1M 上下文的模型，省掉了检索，但每个请求的 prefill 都是百万 token 级，成本和延迟都不可接受。RAG 用一次检索把上下文压到几千 token，代价是可能漏掉相关信息。目前的分工是：高频、事实性的查询走 RAG，需要全局推理的任务走长上下文。

## 49.2 Agent Workloads

Agent 把一次问答变成多轮“思考—调用工具—观察结果”的循环。一次任务可能有 10–50 轮，每轮都要一次完整的模型调用。

资源画像和单轮生成完全不同：

- **输入输出比失衡**：每一轮都要把之前的完整历史作为输入，输入 token 随轮数线性增长，而输出只有几百 token；
- **KV 复用是核心**：第 `k` 轮的输入 = 前 `k-1` 轮的全部内容 + 新的观察结果。如果每轮都从头 prefill，成本会平方级增长；
- **工具等待**：调用外部 API 或执行代码时 GPU 空闲，这段时间要能跑别的请求。

第一点的量化很直观。假设每轮新增 500 token，第 20 轮的输入是 10,000 token。如果每轮都重算，总 prefill 量是 `500 × (1 + 2 + ... + 20) = 105,000` token；如果复用 KV，只需要 `20 × 500 = 10,000` token。差了 10 倍。

所以 agent 服务的调度器必须支持**跨请求的 KV 复用**，而且要在工具调用期间把 KV 保留住。这就带来两个设计问题：

- **KV 的生命周期**：一个 agent 任务可能持续几分钟，期间 KV 一直占着显存。并发 100 个 agent，每个 10K token，就是 320 GB——超出单机容量。实践中会把不活跃的 agent KV 换出到 CPU；
- **会话亲和**：同一 agent 的连续轮次必须落到同一实例，否则缓存无效。这和多租户的负载均衡是冲突的，通常按会话 ID 做一致性哈希。

工具调用的延迟也改变了 SLO 的定义。单轮生成的 TPOT 是 10 ms，但端到端任务时长可能是 30 秒，其中一半时间在等工具。优化方向不是让模型更快，而是让等待期间 GPU 不空转——把 agent 的请求当作可抢占的任务，工具等待时让出 batch 槽位。

## 49.3 Tool Calling and Structured Generation

要求模型输出合法的 JSON 或函数调用参数，靠“提示词里写清楚”是不够的，正确率通常只有 90% 左右，剩下的 10% 会让下游解析失败。

**约束解码**（constrained decoding）在采样时就把不合法的 token 屏蔽掉。实现方式是把 JSON Schema 或语法编译成有限状态机，每一步根据当前状态计算允许的 token 集合：

```text
状态：期望 key 字符串
允许：{ ", a-z, 0-9, _ }   ← 其他 token 的 logit 置为 -inf
```

工程上的难点有三个：

- **token 与字符不对齐**：一个 token 可能跨越多个语法位置，比如 `{"a` 是一个 token，进入了两个状态。需要用 tokenizer 的字节映射预先计算每个 token 的合法后继；
- **状态机规模**：复杂 schema 的状态数可能上万，每步查表的开销要控制在微秒级；
- **性能开销**：每步都要做一次 mask 计算和 logit 处理。实测开销在 5%–15%，取决于词表大小和 schema 复杂度。

一个重要的优化是 **jump-forward decoding**：当语法状态唯一确定接下来的若干字符时（比如 `"name":` 后面必然是引号），直接把这段 token 填进去，跳过模型的前向。结构化输出的模板部分往往很长，jump-forward 能省掉 30%–50% 的 decode 步。

结构化生成对 batch 的影响也值得注意：不同请求的 schema 不同，mask 也不同，无法共享同一个 kernel。实践中会把 schema 相同的请求分到一组，或者用通用的位掩码实现（每个请求一个 mask 张量），牺牲一点效率换取灵活性。

## 49.4 Multimodal Inference

多模态模型把图像、视频编码成 token，再和文本一起送进语言模型。系统上的关键问题是**编码成本**和**token 数量**。

以视觉语言模型为例，一张 1024×1024 的图，用 14×14 的 patch，产生约 5,300 个 patch；2×2 合并后约 1,300 个 token，进一步池化可以压到 256–1,024 个。一个 300M 参数的 ViT-L 编码 1,300 个 token 约 0.8 TFLOP，看起来不多——同样 1,300 个 token 送进 70B 语言模型做 prefill 要 180 TFLOP。真正的成本不在视觉编码器，而在这些 token 后续占用的上下文和 KV Cache。

这带来几个特点：

- **编码器和语言模型是两套负载**：ViT 是计算密集的（矩阵乘为主），语言模型 decode 是访存密集的。它们的 batch 策略、并行方式都不同；
- **图像 token 挤占上下文**：一张图 1,024 token，相当于 800 个汉字。多图请求的 KV Cache 会迅速膨胀；
- **编码结果可缓存**：同一张图被多次引用时，视觉 embedding 可以缓存，避免重复编码。这在文档问答、电商场景里很常见。

视频更极端。一段 30 秒、30 fps 的视频是 900 帧，即使每帧只取 64 token，也是 57,600 token。实际系统通常抽帧（1–4 fps）加时空压缩，把 token 数压到几千。但即使这样，视频 prefill 的时间也在秒级，必须和文本请求分开调度。

一个工程细节是**图像预处理的 CPU 开销**：解码 JPEG、缩放、归一化，一张 4K 图在单核上要几十毫秒。高并发下这部分会成为瓶颈，通常要用 GPU 做预处理，或者用专门的解码库。

## 49.5 Audio and Speech Workloads

语音负载有两类：语音识别（speech-to-text）和语音合成（text-to-speech），以及把两者串起来的实时对话。

**流式 ASR** 的要求是低延迟：用户说话时就要出字。做法是把音频切成 20–40 ms 的块，每块送进编码器，用带状态的模型（如 Conformer 的缓存机制）维持上下文。端到端延迟由“块大小 + 编码 + 解码”决定，通常在 200–500 ms。

**TTS** 的 token 率远高于文本：一段 10 秒的语音可能是 500 个音频 token，而对应的文本只有 50 个字。这意味着 TTS 的 decode 负载比文本生成重一个数量级，且对实时性要求高——生成速度必须快于播放速度（实时因子 < 1）。

**实时语音对话**把 ASR、LLM、TTS 串成流水线，端到端延迟预算是 500–800 ms：

| 环节 | 预算 |
| --- | --- |
| 音频采集与 VAD | 100 ms |
| ASR | 150 ms |
| LLM 首 token | 200 ms |
| TTS 首包 | 150 ms |
| 网络 | 100 ms |

这个预算非常紧，任何一环的 P99 抖动都会让对话变得不自然。系统上的对策是**全链路流式**：ASR 出部分结果就送给 LLM，LLM 出第一个短语就送给 TTS，TTS 出第一段音频就开始播放。代价是错误会向下游传播——ASR 听错了，后面的回答就全错。

另一个技巧是**打断处理**（barge-in）：用户中途说话时，要立刻停止 TTS 播放并清空 LLM 的生成。这要求整条链路都支持取消，且取消的延迟要小于 100 ms。

## 49.6 Real-time and Streaming Workloads

实时负载的共同约束是**尾延迟**而不是平均延迟。视频会议里 5% 的卡顿比平均延迟高 50 ms 更影响体验。

LLM 服务里，实时性的敌人是**排队**。一个请求的端到端延迟 = 排队时间 + prefill + decode。在负载 80% 时，排队时间的分布会急剧变长（排队论里的“拐点”）。所以实时服务的容量规划通常只跑到 50%–60% 的利用率，留出应对突发的空间。

几个针对实时场景的技术：

- **chunked prefill**：把长 prompt 的 prefill 切成小块，和 decode 交错执行，避免一个长 prefill 阻塞所有 decode（Chapter 29 讲过）；
- **优先级抢占**：实时请求优先，批处理请求可以被抢占，KV 换出后重算；
- **流式投机**：投机解码的验证和输出流式化，让首 token 更快；
- **地理分布**：把模型部署在离用户近的区域，网络延迟从 100 ms 降到 20 ms——这比任何 GPU 优化都有效。

还有一个反直觉的点：**更小的模型往往更适合实时场景**。70B 的 TPOT 是 10 ms，8B 是 3 ms；在同样的 SLO 下，8B 能支撑更高的并发和更低的尾延迟。质量差距可以通过 RAG、few-shot 和结构化输出来弥补一部分。实时场景的优化空间，很多时候在“选对模型”，而不是“把大模型调快”。

## 本章小结

- RAG 的延迟分五段，瓶颈常在检索和长 prompt 的 prefill；prompt 高度重复，prefix 感知路由能显著提高命中率。
- Agent 的输入随轮数线性增长，跨请求 KV 复用能把 20 轮的 prefill 从 105,000 token 降到 10,000；代价是长生命周期的 KV 占用和会话亲和。
- 约束解码用状态机屏蔽非法 token，开销 5%–15%；jump-forward 能跳过模板部分，省 30%–50% 的 decode 步。
- 多模态的成本主要在视觉编码和 token 数量：一张 1024×1024 的图约 5,300 个 patch token，可池化到 256–1,024；编码结果可缓存，视频需要抽帧加时空压缩。
- 语音负载的 token 率比文本高一个数量级，实时对话的端到端预算 500–800 ms，必须全链路流式并支持打断。
- 实时场景优化的是尾延迟而不是平均延迟，利用率只跑到 50%–60%；chunked prefill、抢占和地理分布比单点优化更有效。
- 实时场景里，选一个更小的模型往往比把大模型调快更有效。

## 延伸阅读

- Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), NeurIPS 2020.
- Yao et al., [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), ICLR 2023.
- Willard and Louf, [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702), 2023.
- Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104), NeurIPS 2024.
- Radford et al., [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356), ICML 2023.
- Liu et al., [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485), NeurIPS 2023.
- Agrawal et al., [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310), OSDI 2024.
