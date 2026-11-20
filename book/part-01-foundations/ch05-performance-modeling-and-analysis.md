---
chapter: 5
part: 1
title: Performance Modeling and Analysis
level: tbd
status: draft
prerequisites: [3, 4]
lab:
---

# Chapter 5. Performance Modeling and Analysis

“这个优化快了多少？”

这个问题有一半的时候问错了。快了多少，取决于你在哪个 batch、哪个序列长度、哪块 GPU 上测；也取决于你说的是首 token 延迟、单 token 延迟，还是整机吞吐。同一份代码，换一个指标，结论可能反过来。

这一章建立一套分析性能的方法：先定义指标，再用 roofline 判断上限，用 Amdahl 定律找串行部分，用排队论解释延迟为什么在高负载下爆炸，最后讨论怎么测才算数。目标不是背公式，而是**在看到一组数字时，能判断它离理论上限还有多远、差距来自哪里**。

## 5.1 Latency and Throughput

延迟和吞吐是两件事，优化其中一个经常会牺牲另一个。

### 四个常用指标

| 指标 | 定义 | 关注者 |
| --- | --- | --- |
| TTFT | Time To First Token，从请求到第一个 token 的时间 | 交互体验 |
| TPOT | Time Per Output Token，首 token 之后每个 token 的平均间隔 | 生成速度 |
| E2E latency | 端到端时间，`TTFT + TPOT × (输出长度 - 1)` | SLO |
| Throughput | 单位时间处理的 token 数或请求数 | 成本 |

TTFT 主要由 prefill 决定，和输入长度成正比。TPOT 主要由 decode 决定，和 batch、上下文长度、带宽有关。E2E 是两者之和，所以长输入和长输出会分别拉高不同的部分。

还有一个指标叫 **goodput**：满足 SLO 的吞吐。如果 TPOT 超过 100 ms 的请求不算数，那么吞吐再高，goodput 也可能很低。生产系统关心的是 goodput，不是裸吞吐。

### 延迟与吞吐的权衡

增大 batch 能提高吞吐，因为同一份权重服务更多 token；但它也拉长了每个 token 的时间，因为每个请求要等更久。

以一个 8×H100 节点、70B FP16、TP=8、平均上下文 2K 的 decode 为例（示意数据，用于说明趋势）：

| Batch | TPOT | 吞吐（token/s） |
| --- | --- | --- |
| 1 | 12 ms | 83 |
| 8 | 14 ms | 571 |
| 32 | 22 ms | 1,455 |
| 128 | 55 ms | 2,327 |
| 256 | 100 ms | 2,560 |

前几行吞吐几乎线性增长，因为权重读取被摊薄；到 128 以后曲线变平，因为 KV Cache 的读取开始占主导，而 KV 是每个请求独有的，摊不薄。

**吞吐的天花板由“每 token 必须搬多少字节”决定，延迟的地板由“一次要搬多少字节”决定。** 这两个数都来自上一章的带宽模型。

### 百分位数

平均值会骗人。100 个请求里 99 个用 10 ms、1 个用 2 秒，平均是 30 ms，但那个 2 秒的请求会让用户关掉页面。

生产系统看 p50、p90、p99，必要时看 p999。报告指标时至少给出 p50 和 p99，并说明样本量和测量时长。

## 5.2 Roofline Model

Roofline 把两个硬件上限画在同一张图上：一条水平的算力线，一条斜率为带宽的带宽线。

$$
\text{Performance} = \min(\text{Peak Compute}, \text{Bandwidth} \times I)
$$

其中 `I` 是算术强度，单位为 FLOP/byte。两条线的交点叫脊点：

$$
I_{\text{ridge}} = \frac{\text{Peak Compute}}{\text{Bandwidth}}
$$

在脊点左侧，性能随强度线性上升，受带宽限制；右侧是一条水平线，受算力限制。

### 怎么读这张图

把 kernel 的实测性能和算术强度画上去，看它离哪条线更近：

- 贴近斜线：带宽受限，优化方向是减少字节数或提高复用；
- 贴近水平线：算力受限，优化方向是换更快的指令或减少 FLOPs；
- 两条线都很远：既没跑满带宽也没跑满算力，通常是延迟、占用率或调度问题。

### 三个算例

**Elementwise**：`y = a*x + b`，读 4 字节、写 4 字节，2 FLOPs。

$$
I = \frac{2}{8} = 0.25 \text{ FLOP/byte}
$$

H100 上的上限是 `3.35e12 × 0.25 = 0.84 TFLOPS`，只有峰值的 0.08%。这类 kernel 无论怎么优化指令都没用，唯一的出路是融合——把多个 elementwise 操作合并成一次读写。

**GEMM**：`[4096, 4096] × [4096, 4096]`。

$$
I = \frac{2 \times 4096^3}{3 \times 4096^2 \times 2} = \frac{4096}{3} \approx 1{,}365
$$

远高于 295，是典型的算力受限。优化方向是 Tensor Core 利用率、tile 大小和流水线。

**Decode attention**：batch 1、GQA（64 个 Q 头、8 组 KV）、head dim 128。每层每 token 的 FLOPs 约 `4 n_h S d`，KV 读取约 `2 × 2 n_kv S d` 字节。

$$
I = \frac{4 \times 64 \times S \times 128}{2 \times 8 \times S \times 128 \times 2} = 8 \text{ FLOP/byte}
$$

GQA 让 64 个 Query 头共享 8 组 KV，强度从 1 提到了 8。但 8 仍然远低于 295，所以 decode attention 是带宽受限的。batch 增大到 `B` 时强度变成 `8B`，要到 batch 37 才可能进入算力受限区——而那时 KV 显存早就爆了。

### 多级 roofline

H100 的 L2 带宽大约是 HBM 的 4–5 倍。如果把 L2 也算进去，roofline 会多出几条斜率更陡的线，分别对应 L2、L1/shared memory。

这解释了一个常见现象：一个 kernel 的 HBM 利用率只有 40%，但性能已经到顶了——它的数据其实来自 L2，真正的上限是 L2 带宽。看 roofline 之前，先确认数据是从哪一级存储来的。

## 5.3 Compute-bound and Memory-bound Workloads

机器平衡点（machine balance）就是脊点。算术强度高于它，算力受限；低于它，带宽受限。

| 算子 | 算术强度 | 类型 | 主要优化手段 |
| --- | --- | --- | --- |
| 大 GEMM（prefill） | 1,000+ | 算力 | Tensor Core、tile、流水线 |
| 小 GEMM（batch 8） | 8–64 | 带宽/算力混合 | 增大 batch、融合 |
| Decode GEMV | 1–8 | 带宽 | 量化、batching、KV 压缩 |
| Softmax / LayerNorm | 0.25–1 | 带宽 | 融合、减少往返 |
| Elementwise | 0.25 | 带宽 | 融合 |
| AllReduce | — | 通信 | 重叠、分层归约 |

### 第三类：延迟受限

Roofline 假设有足够的并行度。但一个只有 1 个 block 的 kernel，只能用到 132 个 SM 中的 1 个，此时它既不是算力受限也不是带宽受限，而是**并行度不足**。

小 kernel 还受启动开销影响。一次 kernel launch 在 GPU 上约 3–5 微秒，CPU 侧提交约 5–10 微秒。如果一个 kernel 本身只跑 2 微秒，启动开销比计算还大。decode 阶段每层有十几个小 kernel，80 层就是上千次 launch，累计开销不可忽略。CUDA Graphs 把整串 launch 一次性提交，能把这部分开销压掉大半。

### 量化会改变什么

FP8 把峰值算力翻倍，同时把权重和 KV 的字节数减半。算术强度翻倍，脊点也翻倍，所以**一个 kernel 属于算力受限还是带宽受限，通常不会因为量化而改变**。

但绝对时间会变：带宽受限的 kernel 因为字节减半，时间减半；算力受限的 kernel 因为峰值翻倍，时间也减半。两条路都通向 2 倍，只是原因不同。

真正改变分类的是 batch 和融合。batch 提高强度但不改变脊点，所以它能把带宽受限推向算力受限；融合减少中间结果的读写，也是提高强度。

## 5.4 Amdahl's Law

Amdahl 定律说：如果程序有比例为 `p` 的部分被加速了 `s` 倍，整体加速比是：

$$
S = \frac{1}{(1-p) + p/s}
$$

当 `s → ∞`，上限是 `1/(1-p)`。

| 可加速比例 `p` | 无限加速后的上限 |
| --- | --- |
| 50% | 2× |
| 90% | 10× |
| 95% | 20× |
| 99% | 100× |

### 在 LLM 系统里的表现

**Python 开销**：假设一个 serving 循环里 95% 的时间在 GPU 上，你把 GPU kernel 优化到 2 倍快：

$$
S = \frac{1}{0.05 + 0.95/2} = 1.90
$$

不是 2 倍，是 1.9 倍。剩下 5% 的 Python 和调度开销在优化后占了总时间的 10%。

**Tokenization**：如果 tokenization 占端到端时间的 10%，那么无论 GPU 多快，整体加速比都不会超过 10 倍。长 prompt 场景下，tokenization 和 detokenization 经常是隐藏的串行部分。

**Decode 的串行性**：自回归生成必须一个 token 一个 token 地算。即使把单 token 的计算优化到无限快，TPOT 仍然受限于读取权重和 KV 的带宽。这是另一种形式的串行部分——不是代码串行，而是数据依赖串行。

### Gustafson 的补充

Amdahl 假设问题规模固定。Gustafson 指出，实际系统里问题规模会随着资源增加而变大：卡多了，batch 和上下文也会变大。

在 LLM 里，这表现为弱扩展：8 卡能服务 8 倍并发，单请求延迟不变。如果只盯着固定 batch 的加速比，会低估大规模系统的价值；如果只盯着弱扩展，会忽略单请求延迟的退化。两个视角都要看。

### 怎么用

先测量，再分解。用 profiler 把总时间拆成 GPU kernel、CPU 调度、通信、IO，算出每一部分的占比，再判断优化哪一块的收益上限。凭直觉猜“瓶颈在 GPU”，经常猜错。

## 5.5 Little's Law and Queueing Theory

Little's Law 说：系统里的平均请求数，等于到达率乘以平均停留时间。

$$
L = \lambda W
$$

它不假设任何分布，只要系统稳定就成立。

### 容量规划

如果每秒到达 20 个请求，每个请求平均停留 32 秒：

$$
L = 20 \times 32 = 640
$$

系统必须能同时容纳 640 个请求。按每节点 126 个并发槽位算，至少 6 个节点——还没算峰值和故障冗余。

反过来也成立：如果只有 4 个节点、504 个槽位，那么可持续的到达率上限是 `504 / 32 ≈ 15.75` 请求/秒。

### 为什么高负载下延迟会爆炸

把服务看成一个 M/M/1 队列：到达率 `λ`，服务率 `μ`，利用率 `ρ = λ/μ`。平均响应时间是：

$$
W = \frac{1}{\mu - \lambda} = \frac{1}{\mu(1-\rho)}
$$

| 利用率 `ρ` | 响应时间 / 服务时间 |
| --- | --- |
| 0.5 | 2× |
| 0.8 | 5× |
| 0.9 | 10× |
| 0.95 | 20× |
| 0.99 | 100× |

利用率从 80% 提到 95%，延迟翻四倍，吞吐只提高 19%。这是所有在线系统都会遇到的拐点。

对 LLM serving，这意味着：**如果你的 SLO 是 p99 TPOT 小于 100 ms，不要把 GPU 跑到 95% 利用率。** 留出的余量不是浪费，是买延迟的保险。

### 批处理也是一种排队

动态批处理把多个请求凑成一个 batch。等待凑批的时间是排队延迟，batch 越大，单 token 成本越低，但等待越久。

Continuous batching（连续批处理）解决了这个问题：每个 decode step 都重新组 batch，请求完成就立刻换出，新请求立刻加入，不需要等整批结束。vLLM、TensorRT-LLM、SGLang 都用这种方式。

即使如此，请求之间仍然会互相影响：一个长请求占着 KV Cache，会挤掉后来的短请求；一个大 batch 会让所有请求的 TPOT 一起变长。这就是 head-of-line blocking。按长度分桶、设置优先级、限制单请求最大长度，都是缓解手段。

### 尾延迟放大

一个页面请求可能 fan-out 到 50 个后端请求，每个后端的 p99 是 1 秒。页面等待所有后端，所以页面延迟的 p99 接近单个后端的最坏情况。后端越多，尾部越容易被触发。

LLM 系统里的对应现象是：一个 agent 任务串行调用 10 次模型，每次 p99 是 2 秒，端到端 p99 会明显高于 2 秒。优化单次调用的平均值，不如先把 p99 压下来。

## 5.6 Bottleneck Analysis

一个系统的吞吐由最慢的资源决定。列出所有可能的资源，算每个资源允许的上限，取最小值。

| 资源 | 上限公式 | 怎么测 |
| --- | --- | --- |
| 算力 | `FLOPs / 峰值算力` | SM throughput、Tensor Core 利用率 |
| HBM 带宽 | `bytes / 有效带宽` | DRAM throughput |
| L2 带宽 | `bytes / L2 带宽` | L2 hit rate、L2 throughput |
| NVLink | `comm bytes / 链路带宽` | NVLink 计数器、nsys |
| PCIe | `H2D/D2H bytes / 链路带宽` | nsys memcpy 时间线 |
| CPU | `指令数 / IPC / 频率` | perf、火焰图 |
| 网络 | `bytes / 网卡带宽` | 网卡计数器、NCCL 日志 |
| 存储 | `bytes / IOPS 或带宽` | iostat |

### 一个例子：70B prefill

8×H100、TP=8、4K prompt：

- 算力：`5.7e14 / (8 × 989e12) = 72 ms`，按 50% 效率约 144 ms；
- HBM：每卡读 17.5 GB 权重，`17.5 / 3.35e12 = 5.2 ms`；
- NVLink：每层约 234 MB，80 层约 18.7 GB，`18.7 / 900e9 = 21 ms`。

算力是瓶颈。这时优化 NVLink 没有意义，应该看 GEMM 效率和 Tensor Core 利用率。

### 同一个模型，decode

batch 64、2K 上下文：

- 算力：`2 × 70e9 × 64 / (8 × 989e12) = 1.1 ms`；
- HBM：每卡 `(17.5 GB + 64 × 0.625 GiB) / 3.35 TB/s ≈ 18 ms`；
- NVLink：`64 × 4.6 MB / 900 GB/s ≈ 0.33 ms`。

瓶颈变成 HBM。这时优化 Tensor Core 没有意义，应该做量化、增大 batch 或压缩 KV。

### 瓶颈会移动

优化完一个瓶颈，下一个就会浮现。这个循环是性能工程的全部：

1. 测量各资源利用率；
2. 找到最接近 100% 的那个；
3. 优化它；
4. 回到第 1 步。

常见的移动路径是：算力 → 带宽 → 通信 → 调度 → 存储。每移动一次，可获得的收益通常变小，成本变高。知道什么时候停下来，和知道怎么优化同样重要。

### 重叠

两个资源如果能在时间上重叠，它们的上限就不是相加，而是取最大值。

prefill 和 decode 可以重叠：chunked prefill 把长 prompt 切成小块，和 decode 请求混在一个 batch 里，让算力密集的 prefill 填补 decode 的带宽空闲。通信也可以和计算重叠：Megatron 的 TP 把 AllReduce 拆开，和下一层的 GEMM 交错执行。

能否重叠，取决于依赖关系和是否有独立的 stream。分析时要把时间线画出来，看资源是真的并行，还是只是看起来并行。

## 5.7 Performance Metrics and Measurement Errors

性能测量最大的敌人不是硬件，是测量方法本身。

### 异步执行

CUDA kernel launch 是异步的。CPU 提交完立刻返回，kernel 可能还没开始跑。用 CPU 的 `time.time()` 包住一段 GPU 代码，测到的是提交时间，不是执行时间。

正确做法是用 CUDA event：

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()
start.record()
for _ in range(iters):
    model(x)
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end) / iters
```

`torch.cuda.synchronize()` 只放在计时区间外。如果每次迭代都同步，测到的是“launch + 执行 + 同步”的串行时间，会显著偏高。

### 预热

第一次运行包含 CUDA context 初始化、JIT 编译、cuDNN autotune、显存分配器预热。这些开销可能比正式运行还大。

标准做法是预热至少 10–20 次，或者跑够 1 秒，再开始计时。框架的 benchmark 工具通常内置了 warmup 参数，不要关掉。

### 时钟与功耗

GPU 的 boost 频率取决于功耗和温度。短时间跑分可以冲到最高频，持续负载会降到稳定频率。如果基准测试只跑 100 ms，得到的数字在生产环境里复现不了。

要得到可比较的结果：

- 固定时钟：`nvidia-smi -lgc <freq>`；
- 记录功耗和温度；
- 跑够时间，让频率进入稳态；
- 报告时注明是否锁频。

### 测量噪声

同一份代码跑 10 次，结果可能差 5%–15%。原因包括频率波动、其他进程干扰、内存分配位置、NUMA 和 PCIe 拓扑。

报告时给出中位数和 p99，至少跑 30 次。如果两个版本的差异小于噪声，就不能下结论。可以做一个简单的统计检验，或者把差异放到更大的 batch 上放大。

### MFU 和 MBU

两个常用的效率指标：

$$
\text{MFU} = \frac{\text{实际 FLOPs/s}}{\text{峰值 FLOPs/s}}
$$

$$
\text{MBU} = \frac{\text{实际 bytes/s}}{\text{峰值带宽}}
$$

MFU 衡量算力利用，MBU 衡量带宽利用。Llama 3 405B 的训练报告 MFU 在 38%–43% 之间；推理的 decode 阶段 MBU 通常能到 60%–80%，而 MFU 只有个位数——这再次说明 decode 是带宽受限的。

两个指标要一起看。只看 MFU，会误判带宽受限的 kernel；只看 MBU，会忽略算力受限的 prefill。

### Profiler 的 overhead

Nsight Compute 需要 replay kernel 来收集所有计数器，开销可能是 10–100 倍。Nsight Systems 开销低得多，但也有 5%–20%。PyTorch Profiler 在开启 `with_stack` 和 `profile_memory` 后开销很大。

原则：**用 profiler 找瓶颈，用干净的计时验证收益。** 不要用 profiler 里的数字直接报告加速比。

## 5.8 Benchmark Design and Reproducibility

一个可信的 benchmark 需要说清楚五件事：测什么、怎么测、在什么上测、测了多久、结果怎么统计。

### 必须记录的信息

| 类别 | 内容 |
| --- | --- |
| 模型 | 名称、参数量、精度、量化方式 |
| 硬件 | GPU 型号、数量、互联拓扑、CPU、内存 |
| 软件 | 驱动、CUDA、框架、推理引擎版本 |
| 负载 | 输入长度分布、输出长度分布、并发数、采样参数 |
| 方法 | warmup 次数、测量时长、请求模式（开环/闭环） |
| 指标 | TTFT/TPOT 的 p50/p99、吞吐、goodput、MFU/MBU、功耗 |

少任何一项，结果都无法复现。

### 负载要像真实流量

用固定 512 输入、256 输出测出来的数字，和真实流量的差距可能很大。真实请求的长度是长尾分布：大部分短，少数很长，而长请求决定了 KV Cache 的压力和尾延迟。

至少要测几组：

- 短输入短输出（128 / 64）；
- 中等（2K / 256）；
- 长输入（16K–32K）；
- 混合分布（按真实日志采样）。

每组单独报告，不要只报一个平均值。

### 开环与闭环

闭环（closed-loop）压测：固定并发数，上一个请求完成才发下一个。它测的是“系统在 N 个并发下的表现”，但会低估延迟——因为当系统变慢时，客户端也变慢，不会继续按原速率施压。这叫 coordinated omission。

开环（open-loop）压测：按固定速率发请求，不管上一个是否完成。它能暴露排队延迟，更接近真实流量。

两种都要跑：闭环看吞吐上限，开环看延迟拐点。

### 常用工具

- vLLM 的 `benchmark_serving.py`：支持 TTFT/TPOT 统计和多种请求分布；
- LLMPerf：开环压测，输出延迟分位数；
- MLPerf Inference：标准化 benchmark，适合跨系统比较；
- `nsys` / `ncu`：定位瓶颈，不作为最终性能数字。

### 反模式

| 反模式 | 问题 |
| --- | --- |
| 只报最好的一次 | 掩盖方差，不可复现 |
| 只测 batch 1 | 忽略吞吐场景，结论片面 |
| 不同输出长度比较吞吐 | token 数不同，数字没有可比性 |
| 忽略 prefill | 长 prompt 场景下 TTFT 才是瓶颈 |
| 在共享机器上测 | 邻居干扰，结果漂移 |
| 用 profiler 数字报加速比 | profiler overhead 改变时间分布 |
| 只测一个 shape | 优化可能只对这个 shape 有效 |

### 回归测试

性能回归比功能回归更难发现，因为代码仍然正确，只是变慢了。

做法：

- 固定一组代表性 shape，每次提交跑微基准；
- 结果和基线比较，超过阈值（如 5%）就报警；
- 记录 commit、驱动版本和 GPU 型号；
- 夜间跑一次端到端 serving benchmark；
- 对关键 kernel 同时跑正确性测试，防止“变快但算错”。

阈值要大于噪声。先在空载机器上跑 30 次，算出变异系数，再定阈值。如果噪声是 8%，5% 的阈值只会制造误报。

## 本章小结

- 延迟和吞吐是不同的目标；TTFT 由 prefill 决定，TPOT 由 decode 和带宽决定，生产系统应看 goodput 和 p99。
- Roofline 用算术强度判断一个算子受算力还是带宽限制；脊点是分界线。
- Decode 的算术强度约为 batch size（attention 部分因 GQA 放大 8 倍），远低于脊点，所以几乎总是带宽受限。
- Amdahl 定律说明串行部分决定加速上限；Python 调度、tokenization、自回归依赖都是常见的串行部分。
- Little's Law 用于容量规划；排队论说明利用率超过 80% 后延迟会急剧上升。
- 瓶颈分析要列出所有资源的上限，取最小值，并在优化后重新测量。
- 测量要注意异步执行、预热、时钟、噪声和 profiler overhead；benchmark 要记录完整环境并使用真实负载分布。

## 延伸阅读

- Amdahl, [Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities](https://dl.acm.org/doi/10.1145/1465482.1465560), 1967.
- Gustafson, [Reevaluating Amdahl's Law](https://dl.acm.org/doi/10.1145/42411.42415), 1988.
- Little, [A Proof for the Queuing Formula: L = λW](https://doi.org/10.1287/opre.9.3.383), 1961.
- Williams et al., [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://dl.acm.org/doi/10.1145/1498765.1498785), 2009.
- Dean & Barroso, [The Tail at Scale](https://dl.acm.org/doi/10.1145/2408776.2408794), 2013.
- Schroeder et al., [Open Versus Closed: A Cautionary Tale](https://www.usenix.org/legacy/events/nsdi06/tech/full_papers/schroeder/schroeder.pdf), 2006.
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), 2023.
- MLCommons, [MLPerf Inference Benchmark](https://mlcommons.org/benchmarks/inference-datacenter/).
