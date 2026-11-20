# Appendix H. Glossary

术语按主题分组，组内大致按中文拼音排序，括号内为书中首次出现的位置。

## H.1 模型与计算

| 中文 | English | 说明 |
| --- | --- | --- |
| 注意力机制 | attention | 按 query 与 key 的相似度加权聚合 value |
| 自回归生成 | autoregressive generation | 每次生成一个 token，依赖之前的全部 token |
| 批处理 | batching | 把多个请求合并成一次前向 |
| 因果掩码 | causal mask | 禁止看到未来 token 的注意力掩码 |
| 分组查询注意力 | grouped-query attention (GQA) | 多个 Q 头共享一组 K/V 头（Ch8） |
| 多头潜在注意力 | multi-head latent attention (MLA) | 把 K/V 压缩成低秩潜向量（Ch9） |
| 混合专家 | mixture of experts (MoE) | 每个 token 只激活部分专家（Ch10） |
| 旋转位置编码 | rotary position embedding (RoPE) | 通过旋转注入相对位置信息 |
| RMSNorm | root mean square normalization | 不含均值的归一化，比 LayerNorm 省一次归约 |
| 状态空间模型 | state space model (SSM) | 用固定状态代替增长的 KV（Ch11） |

## H.2 数值与压缩

| 中文 | English | 说明 |
| --- | --- | --- |
| 算术强度 | arithmetic intensity | 每字节访存对应的 FLOPs |
| 块缩放 | block scaling | 每若干元素共享一个缩放因子 |
| 校准 | calibration | 用少量数据确定量化的缩放因子 |
| 反量化 | dequantization | 把低精度值还原成浮点 |
| 逐通道量化 | per-channel quantization | 每个输出通道一组缩放 |
| 量化感知训练 | quantization-aware training (QAT) | 训练时模拟量化误差 |
| 训练后量化 | post-training quantization (PTQ) | 训练后直接量化，无需重训 |
| 微缩放格式 | microscaling (MX) | 4/6 位元素加 8 位块指数 |

## H.3 GPU 与 kernel

| 中文 | English | 说明 |
| --- | --- | --- |
| 占用率 | occupancy | 活跃 warp 占 SM 上限的比例 |
| 合并访存 | coalesced access | 相邻线程访问相邻地址 |
| 线程束 | warp | 32 个同步执行的线程 |
| 共享内存 | shared memory | block 内可见的片上存储 |
| 张量核心 | tensor core | 专用矩阵乘单元 |
| 流水 | pipeline | 访存与计算重叠执行 |
| 融合 | fusion | 多个算子合成一个 kernel |
| 持久化 kernel | persistent kernel | 常驻 GPU、循环取任务 |
| CUDA Graph | CUDA graph | 录制并重放 kernel 序列 |

## H.4 推理系统

| 中文 | English | 说明 |
| --- | --- | --- |
| 连续批处理 | continuous batching | 每步动态增减 batch 成员（Ch29） |
| 首 token 延迟 | time to first token (TTFT) | 从请求到第一个输出 token |
| 每 token 时间 | time per output token (TPOT) | 相邻输出 token 的间隔 |
| 键值缓存 | KV cache | 缓存历史 token 的 K/V |
| 页式 KV | paged KV cache | 按固定大小块管理 KV（Ch26） |
| 前缀缓存 | prefix cache | 复用公共前缀的 KV（Ch27） |
| 投机解码 | speculative decoding | 小模型起草、大模型验证（Ch28） |
| 分离式预填充解码 | prefill-decode disaggregation | prefill 与 decode 分开部署（Ch47） |
| 预填充 | prefill | 一次性计算整个 prompt 的 KV |
| 解码 | decode | 逐 token 生成阶段 |
| 抢占 | preemption | 为腾出显存而中止正在进行的请求 |

## H.5 分布式

| 中文 | English | 说明 |
| --- | --- | --- |
| 全归约 | all-reduce | 所有 rank 求和后广播 |
| 全收集 | all-gather | 收集所有 rank 的片段 |
| 全交换 | all-to-all | 每对 rank 之间交换数据 |
| 归约散射 | reduce-scatter | 求和后按 rank 切分 |
| 数据并行 | data parallelism (DP) | 每卡一份模型，切分 batch |
| 张量并行 | tensor parallelism (TP) | 按矩阵维度切分 |
| 流水并行 | pipeline parallelism (PP) | 按层切分 |
| 专家并行 | expert parallelism (EP) | MoE 的专家分布到不同卡 |
| 零冗余优化器 | ZeRO | 分片参数、梯度、优化器状态 |
| 完全分片数据并行 | FSDP | PyTorch 的 ZeRO-3 实现 |
| 环形注意力 | ring attention | 序列维切分、KV 环形传递 |

## H.6 训练与后训练

| 中文 | English | 说明 |
| --- | --- | --- |
| 激活重计算 | activation checkpointing | 反向时重算激活以省显存 |
| 梯度累积 | gradient accumulation | 多个 micro-batch 累积一次更新 |
| 模型浮点利用率 | MFU | 实际 FLOPs / 峰值 FLOPs |
| 监督微调 | supervised fine-tuning (SFT) | 在指令数据上做下一个 token 预测 |
| 直接偏好优化 | direct preference optimization (DPO) | 用偏好对直接优化策略 |
| 近端策略优化 | proximal policy optimization (PPO) | 带裁剪的 on-policy 策略梯度 |
| 分组相对策略优化 | group relative policy optimization (GRPO) | 用组内奖励归一化代替 critic |
| 可验证奖励强化学习 | RL with verifiable rewards (RLVR) | 用程序判定对错 |
| 奖励作弊 | reward hacking | 策略利用奖励函数的漏洞 |
| 低秩适配 | low-rank adaptation (LoRA) | 只训练低秩增量 |

## H.7 系统与服务

| 中文 | English | 说明 |
| --- | --- | --- |
| 计算图 | computational graph | 算子与数据依赖构成的有向无环图 |
| 下沉 | lowering | 把高层算子翻译成硬件表示 |
| 形状特化 | shape specialization | 针对具体形状生成代码 |
| 服务等级目标 | service level objective (SLO) | 可度量的服务承诺 |
| 多租户 | multi-tenancy | 多个租户共享同一集群 |
| 加权公平队列 | weighted fair queueing (WFQ) | 按权重分配资源 |
| 尾延迟 | tail latency | P99 等高百分位延迟 |
| 检索增强生成 | retrieval-augmented generation (RAG) | 检索文档后拼进 prompt |
| 约束解码 | constrained decoding | 屏蔽不合法的 token |
