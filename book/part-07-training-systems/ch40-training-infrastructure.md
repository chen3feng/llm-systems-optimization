---
chapter: 40
part: 7
title: Training Infrastructure
level: tbd
status: draft
prerequisites: [36, 39]
lab:
---

# Chapter 40. Training Infrastructure

一千张卡训练两个月，按单卡年故障率 2% 算，期间会遇到三四次 GPU 故障；再加上节点、网络和存储的故障，实际停机次数更多。每次停机都意味着从最近的 checkpoint 恢复——而 70B 训练的 checkpoint 是 1.1 TB，读回来就要几分钟。

训练基础设施处理的就是这些“与模型无关、但决定能不能跑完”的问题：数据怎么喂、状态怎么存、故障怎么恢复、集群怎么调度、成本怎么算。它们不改变模型结构，却决定了同样的算力能产出多少有效训练。

## 40.1 Data Loading and Prefetching

数据管道的目标是让 GPU 永远不等待。一条训练样本要经过读取、解码、tokenize、打包，才能变成一个 batch。

```text
存储（对象存储/NVMe）
  → 读取：预取 N 个 batch
  → 解码与 tokenize：CPU 线程池，多进程
  → 打包：拼成定长序列，写入共享队列
  → GPU：直接 H2D 拷贝
```

关键在于预取和并行。DataLoader 的 `num_workers` 要足够多，让 CPU 处理领先 GPU 至少一个 batch；如果每个 step 才去取数据，GPU 就要等 I/O。

大模型的训练对数据管道要求不高：70B 每秒只消费 7,500 个 token，约 30 KB。真正吃 CPU 的是预处理和去重，通常离线完成，不占训练循环。小模型则相反，1B 模型每秒消费 50 万 token，数据管道很容易成为瓶颈。

几个常见陷阱：

- **每个 worker 一份数据副本**：`num_workers` 大时，内存和页缓存被重复占用；
- **随机访问对象存储**：小文件、随机读的延迟远高于顺序读，应该先缓存到本地 NVMe；
- **tokenize 在 GPU 进程里做**：会阻塞训练循环，必须放在独立进程；
- **数据不均衡**：不同 worker 分到的样本长度差异大，导致某些 step 特别慢。

## 40.2 Checkpoint Format and Storage

checkpoint 要保存“恢复训练所需的一切”：

| 内容 | 70B 的大小 | 说明 |
| --- | --- | --- |
| 参数（BF16） | 140 GB | 推理也用它 |
| FP32 master 权重 | 280 GB | 更新需要 |
| 优化器 `m`、`v` | 560 GB | Adam 状态 |
| 梯度 | 140 GB | 可选，通常不存 |
| RNG、调度器、数据位置 | 几 KB | 决定能否精确续训 |
| 合计 | 约 1.1 TB | 不含梯度 |

格式上，分片存储是标准做法：每个 rank 写自己的分片，元数据记录全局布局：

```text
checkpoint-1000/
├── metadata.json       # 并行度、张量映射、步数、RNG
├── rank-0000.safetensors
├── rank-0001.safetensors
...
```

分片的好处是并行写、并行读，也避免了单文件的锁竞争。代价是恢复时必须用相同的并行度，或者做重分片（resharding）。元数据里记录全局张量布局，才能从 64 卡恢复到 32 卡。

写入要原子：先写临时目录，完成后 rename。写到一半崩溃留下的半成品，如果不加校验，恢复时会加载出损坏的权重——而且不一定报错，可能只是训练慢慢发散。

1.1 TB 写到 7 GB/s 的并行文件系统要 160 秒。如果每 100 步存一次，一步 10 秒，那就是 16% 的时间在写 checkpoint。这直接催生了异步 checkpoint。

## 40.3 Asynchronous Checkpointing

异步 checkpoint 把“快照”和“落盘”分开：

```text
1. 在 GPU 上把状态复制到 pinned host memory（阻塞训练）
2. 后台线程把 host 数据写到存储（与训练并行）
3. 写完后更新 checkpoint 指针
```

关键在第一步必须快。1.1 TB 通过 PCIe 50 GB/s 要 22 秒，仍然可观；可以用增量 checkpoint 只存变化的参数，或者用 GPU 直接写（GPUDirect Storage）绕开 host。

| 方式 | 训练停顿 | 适用 |
| --- | --- | --- |
| 同步写 | 160 s | 小模型、低频 |
| 异步（复制到 host） | 22 s | 主流方案 |
| 增量 | 几秒 | 参数变化密集时收益有限 |
| 无停顿（快照 + COW） | ~0 | 需要文件系统/驱动支持 |

异步的代价是内存：快照期间要额外保留一份 host 副本，1.1 TB 的主机内存不是每个节点都有。常见做法是分片快照，每个 rank 只存自己的部分，总量还是 1.1 TB，但分摊到各节点。

还有一个隐蔽问题：异步写期间如果训练继续，下一份 checkpoint 可能覆盖正在写的文件。实现上要用版本号或写时复制，确保每一份快照对应一个不可变的目录。

## 40.4 Fault Tolerance

训练的故障域比推理大：一个 rank 失败，整个作业都要停。

故障来源和占比大致是：

| 来源 | 典型表现 | 恢复方式 |
| --- | --- | --- |
| GPU 硬件 | Xid、ECC 不可纠正 | 换卡、重启作业 |
| 节点 | 掉电、网卡故障 | 重新调度 |
| 网络 | IB 链路抖动、交换机故障 | 重连、换路径 |
| 存储 | 读超时、元数据服务故障 | 重试、切备 |
| 软件 | NCCL 挂起、OOM | 重启进程 |

千卡集群的 MTBF 可以估算：单卡年故障率 2%，1,024 张卡一年约 20 次 GPU 故障，平均 18 天一次；加上节点和网络，实际可能几天一次。两个月的训练跑下来，遇到几次故障是常态。

恢复的流程是：检测 → 终止所有 rank → 重新调度 → 加载 checkpoint → 重建通信组 → 继续。时间主要花在重新调度和加载上，1.1 TB 的加载要几分钟。

降低恢复成本的手段：

- **高频 checkpoint**：间隔越短，丢失的进度越少，但写开销越大；
- **弹性训练**：故障后不重启整个作业，缩容继续，等资源回来再扩容；
- **冗余**：关键节点热备，故障时快速替换；
- **快速失败**：不要让挂起的 rank 拖到超时才退出。

## 40.5 Cluster Scheduling

训练作业的特点是“要么全有、要么全无”：1,024 个 rank 必须同时拿到资源，否则整个作业等着。这叫 gang scheduling。

调度器要处理的问题：

- **资源碎片**：集群里零散的空闲卡凑不出一个完整作业，作业一直排队；
- **拓扑**：TP 组要同节点，PP/DP 组要尽量同交换机，跨 spine 的通信会拖慢整个作业；
- **优先级与配额**：多团队共享集群时，配额防止一个团队占满；
- **抢占**：高优先级作业可以抢占低优先级，被抢占的作业从 checkpoint 恢复。

```text
Slurm:  srun --nodes=128 --ntasks-per-node=8 --gpus-per-task=1
K8s:    gang scheduling（Volcano、Kueue）+ GPU device plugin + 拓扑感知
```

拓扑感知调度在大规模下价值很高：同样的 1,024 张卡，跨 spine 的放置可能让 all-reduce 慢 30%。调度器要读取 NVLink 域、leaf switch、rack 的标签，把作业尽量放在一个拓扑单元内。

## 40.6 Training Monitoring

训练监控的核心指标分三类：

| 类别 | 指标 | 异常信号 |
| --- | --- | --- |
| 进度 | loss、学习率、token 数 | loss 尖刺、平台期 |
| 效率 | MFU、tokens/s、通信带宽 | 吞吐下降、NCCL 带宽不足 |
| 健康 | GPU 温度、Xid、ECC、显存 | 温度过高、ECC 计数上升 |

loss spike 是最需要关注的信号。它可能是数据问题（脏样本）、数值问题（梯度爆炸）或硬件问题（静默错误）。处理方式通常是回滚到上一个 checkpoint，跳过那段数据，或者降低学习率。

效率指标要看趋势。MFU 从 40% 慢慢掉到 35%，可能是碎片增加、某个 rank 变慢，或者 checkpoint 越来越频繁。慢节点（straggler）尤其隐蔽：平均值正常，但最慢的 rank 拖住了整个 all-reduce。

监控要按 rank 采集，而不是只看全局平均。一个 rank 的 NCCL 带宽只有别人的一半，全局看只是效率略降，实际是这条链路出了问题。

## 40.7 Experiment Reproducibility

完全复现一次大模型训练几乎不可能：硬件、驱动、通信顺序、原子操作的浮点非确定性都会影响结果。但“可复现”仍有明确的目标：同样的代码和数据，在同样的硬件上，能得到统计上等价的模型。

要做到这一点，需要固定：

- **代码和镜像**：容器镜像带版本，依赖锁定；
- **数据**：数据版本、采样顺序、shuffle 种子；
- **随机性**：初始化种子、dropout 种子、数据加载顺序；
- **并行配置**：TP/PP/DP 的度、microbatch 大小、梯度累积步数；
- **精度设置**：BF16/FP8、是否重计算、优化器实现。

数值非确定性来自 GPU 的并行归约：同样的输入，浮点加法顺序不同，结果就不同。`torch.use_deterministic_algorithms(True)` 能强制确定性，但会禁用一些高效 kernel，训练变慢。多数团队选择“统计可复现”，只固定种子和数据，不追求逐位一致。

实验管理上，每个 run 要记录完整的配置、指标曲线和 checkpoint 指针。没有这些，三个月后没人能说清某个结果是哪次实验跑出来的。

## 40.8 Resource Utilization and Cost Analysis

训练成本可以直接算：

$$
\text{成本} = \text{GPU 小时} \times \text{单价}
= \frac{6 N D}{F_{\text{peak}} \cdot \text{MFU}} \times \text{单价}
$$

70B、15T token、H100、MFU 40%：`6 × 70e9 × 15e12 / (989e12 × 0.4) = 1.59e10 秒`，即 440 万 GPU 小时。按每小时 2 个货币单位算（不同云和自建差异很大），约 880 万。

摊到 token 上：`880 万 / 15T ≈ 0.59 / 百万 token`。

这个数字的启示是：成本对 MFU 极其敏感。MFU 从 40% 提到 45%，成本降 11%；而从 30% 提到 40%，降 25%。这也是为什么大模型团队愿意投入大量工程去抠几个百分点的 MFU。

利用率不只看 MFU，还要看“有效 GPU 时间”：

```text
有效利用率 = MFU × 作业运行率 × (1 - checkpoint/恢复开销)
```

一个 MFU 45% 但每周停机两次的集群，有效利用率可能不如 MFU 40% 但稳定的集群。评估时要看端到端的 token 产出成本，而不是单个指标的峰值。

## 本章小结

- 数据管道的目标是让 GPU 不等待；大模型对带宽要求很低，小模型才可能被 tokenize 和存储拖住。
- 70B 的完整训练 checkpoint 约 1.1 TB；分片存储支持并行读写，元数据要记录全局布局以支持重分片。
- 同步写一份 checkpoint 要 160 秒，异步方案把停顿压到 22 秒（复制到 host），代价是额外的主机内存。
- 千卡集群几天就可能遇到一次故障；恢复时间主要花在重新调度和加载 checkpoint 上。
- 调度要解决 gang scheduling、资源碎片和拓扑感知；跨 spine 的放置可能让通信慢 30%。
- 监控要按 rank 采集，看 loss、MFU、通信带宽和硬件健康；慢节点会拖住整个 all-reduce。
- 完全逐位复现很难，实践目标是固定代码、数据、种子和并行配置后的统计可复现。
- 成本 = GPU 小时 × 单价，对 MFU 极其敏感；评估要看端到端的有效利用率，而不只是峰值 MFU。

## 延伸阅读

- Mohan et al., [CheckFreq: Frequent, Fine-Grained DNN Checkpointing](https://www.usenix.org/conference/fast21/presentation/mohan), FAST 2021.
- Eisenman et al., [Check-N-Run: A Checkpointing System for Training Deep Learning Recommendation Models](https://arxiv.org/abs/2010.08679), NSDI 2022.
- Jeon et al., [Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads](https://www.usenix.org/conference/atc19/presentation/jeon), ATC 2019.
- Weng et al., [MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters](https://www.usenix.org/conference/nsdi22/presentation/weng), NSDI 2022.
- Narayanan et al., [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473), SC 2021.
- PyTorch, [torch.distributed.checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html).
- NVIDIA, [GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/).
- Slurm, [Gang Scheduling and Resource Management](https://slurm.schedmd.com/).
