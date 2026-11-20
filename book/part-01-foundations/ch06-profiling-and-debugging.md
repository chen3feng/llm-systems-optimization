---
chapter: 6
part: 1
title: Profiling and Debugging
level: tbd
status: draft
prerequisites: [4, 5]
lab:
---

# Chapter 6. Profiling and Debugging

一个 kernel 报错，Python 栈里只有一行：

```text
RuntimeError: CUDA error: an illegal memory access was encountered
```

没有行号，没有变量，没有哪个线程越界。CUDA 的异步执行把错误和现场分开了几毫秒，也分开了几十万个线程。

性能问题同样如此。代码能跑，loss 正常，但吞吐只有预期的一半。你看不到是哪一层慢、哪个 kernel 在等内存、哪次通信没有重叠。

这一章讲工具和方法：怎么从系统级时间线定位到 kernel，再从 kernel 定位到源码行；怎么查出越界、泄漏和数值错误；怎么把性能回归挡在合并之前。

## 6.1 CPU Profiling

GPU 系统的瓶颈经常在 CPU 上。请求解析、tokenization、调度、采样、日志，任何一段慢了，GPU 都会空等。

### perf

Linux 上最常用的采样 profiler：

```bash
perf stat -p <pid> sleep 10
perf record -F 999 -g -p <pid> -- sleep 30
perf report --hierarchy
```

`perf stat` 给出粗粒度指标：IPC、cache miss、分支预测失败、上下文切换。IPC 长期低于 1，通常意味着内存访问或锁竞争；cache miss 高，说明数据布局有问题。

`perf record -g` 采集调用栈，`perf report` 可以按函数展开。要得到火焰图，把 `perf script` 的输出交给 FlameGraph 工具：

```bash
perf script | stackcollapse-perf.pl | flamegraph.pl > cpu.svg
```

火焰图的宽度是采样占比，不是耗时绝对值。看最宽的塔，再看它下面的调用链。

### Off-CPU 分析

CPU 火焰图只显示“在 CPU 上跑”的时间。线程阻塞在锁、IO、page fault 上的时间不会出现。LLM 服务里，等待网络请求、等待 GPU 完成、等待锁的时间可能占大头。

```bash
# bcc 工具
offcputime -p <pid> 30
# 或者用 perf 跟踪调度事件
perf record -e sched:sched_switch -p <pid> -- sleep 30
```

如果 off-CPU 时间集中在 `cudaDeviceSynchronize` 或 `cudaStreamSynchronize`，说明 CPU 在等 GPU；如果集中在 `futex`，说明有锁竞争；如果集中在 `read`/`write`，说明 IO 是瓶颈。

### Python

`cProfile` 适合函数级统计，但对多线程和 C 扩展不友好：

```bash
python -m cProfile -s cumulative serve.py
```

`py-spy` 是采样 profiler，不需要改代码，能看到 Python 栈和 C 栈：

```bash
py-spy top --pid <pid>
py-spy record -o profile.svg --pid <pid> --duration 30
py-spy dump --pid <pid>          # 当前所有线程的栈
```

GIL 是常见陷阱。纯 Python 代码在多线程下不能并行；Hugging Face 的 fast tokenizer 在 Rust 侧释放了 GIL，可以并行，但 Python 层的循环、JSON 解析、回调不会。如果一个 serving 进程的 CPU 只用到 100%，先看是不是 GIL 限制了吞吐。

### 常见瓶颈

| 现象 | 可能原因 |
| --- | --- |
| CPU 单核 100%，GPU 空闲 | Python 调度、tokenization、采样循环 |
| 大量 futex 等待 | 锁竞争、线程池过小 |
| 高 page fault | 模型 mmap 未预热、内存不足 |
| 高 sys 时间 | 网络收发、日志、频繁小 IO |
| 上下文切换频繁 | 线程过多、阻塞式调用 |

## 6.2 GPU Profiling with Nsight Systems

Nsight Systems 看的是时间线：CPU 在做什么、GPU 在做什么、两者有没有重叠。

### 采集

```bash
nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --cuda-memory-usage=true \
  --sample=cpu \
  --duration=30 \
  -o /tmp/llm_trace \
  python serve.py
```

生成 `.nsys-rep` 文件，用 GUI 打开，或者用命令行统计：

```bash
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,osrt_sum /tmp/llm_trace.nsys-rep
```

### 时间线上看什么

按顺序检查四件事：

1. **GPU 有没有空洞。** kernel 之间如果有明显间隙，GPU 在等 CPU。看间隙前面是哪个 CUDA API 调用，通常是同步、内存分配或 Python 逻辑。
2. **有没有意外的 memcpy。** H2D/D2H 拷贝如果频繁出现，可能是 tensor 在 CPU 和 GPU 之间来回搬，或者用了 pageable memory。
3. **多个 stream 有没有真的并行。** 画在一起不等于重叠，要看时间条是否交叠。
4. **单个 kernel 的时长分布。** 一两个超长 kernel 和几千个微小 kernel，优化方向完全不同。

### NVTX 标注

Nsight Systems 能显示 NVTX 区间。在代码里埋点，把逻辑阶段标出来：

```python
import torch

with torch.cuda.nvtx.range("prefill"):
    logits = model(prompt_ids)
with torch.cuda.nvtx.range("decode"):
    for _ in range(max_new_tokens):
        logits = model(next_id, past_key_values=past)
        next_id = sample(logits)
```

时间线上就能直接看到 prefill 和 decode 各占多少，以及它们之间有没有空隙。

### 一个典型发现

某 serving 系统的吞吐只有理论值的一半。nsys 时间线显示：每个 decode step 之间有 3–4 ms 的空隙，空隙前是一次 `cudaStreamSynchronize`。

原因是采样在 CPU 上做，每步都要把 logits 拷回 CPU、同步、用 NumPy 采样、再把 token 拷回 GPU。一次往返 3 ms，80 层 kernel 只要 8 ms，开销占了 27%。

修复方式是把采样移到 GPU 上，或者至少用 pinned memory 加异步拷贝。这类问题在纯 kernel profiler 里看不到，只有系统级时间线能暴露。

## 6.3 Kernel Profiling with Nsight Compute

Nsight Systems 告诉你哪个 kernel 慢，Nsight Compute 告诉你它为什么慢。

### 采集

```bash
ncu --set full \
    --kernel-name-base demangled \
    --kernel-name regex:.*gemm.* \
    --launch-skip 20 --launch-count 1 \
    -o /tmp/gemm_report \
    python bench.py
```

`--set full` 收集所有 section，开销很大。日常分析可以只选需要的：

```bash
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --section Occupancy --section SchedulerStats \
    python bench.py
```

`--launch-skip` 跳过预热阶段，`--launch-count` 限制采集的 kernel 数，避免报告过大。

### 先看 SpeedOfLight

SpeedOfLight 给出两个百分比：SM 吞吐和内存吞吐，各自相对于峰值。

| 指标 | 含义 | 经验判断 |
| --- | --- | --- |
| `sm__throughput` | 计算单元利用率 | 接近 100%：算力受限 |
| `gpu__dram_throughput` | HBM 带宽利用率 | 接近 100%：带宽受限 |
| `l1tex__throughput` | L1/shared 吞吐 | 高：片上带宽受限 |
| `lts__throughput` | L2 吞吐 | 高：L2 受限 |

两个都低，说明既没吃满算力也没吃满带宽：可能是延迟受限、占用率不足，或者有同步/分支开销。

### 再看 stall reason

SchedulerStats 和 WarpStateStats 给出 warp 为什么停：

| Stall 原因 | 含义 | 常见修复 |
| --- | --- | --- |
| Long scoreboard | 等全局内存 | 提高占用率、预取、改善合并访问 |
| Short scoreboard | 等 shared memory | 消除 bank conflict |
| Wait | 等固定延迟指令 | 增加 ILP、展开循环 |
| Barrier | 等 `__syncthreads()` | 减少同步、调整 block 大小 |
| Not selected | 有别的 warp 被选中 | 正常，说明调度器忙 |
| Math pipe throttle | 计算单元排满 | 正常，说明算力受限 |

`Not selected` 占比高不是坏事，说明 SM 有很多 ready 的 warp。`Long scoreboard` 高才是内存延迟没藏住。

### MemoryWorkloadAnalysis

这一节回答“字节从哪来、浪费在哪”：

- DRAM read/write throughput；
- L1/L2 hit rate；
- sectors per request：理想是 4（128 字节 / 32 字节 sector），大于 4 说明访问没有合并；
- shared memory bank conflict 次数。

如果 `sectors per request` 明显偏高，先检查访问模式，而不是换指令。

### 源码关联

编译时加 `-lineinfo`，ncu 能把指标映射到源码行：

```bash
nvcc -lineinfo -O3 kernel.cu -o kernel
ncu --set full --import-source yes ./kernel
```

在 Source 视图里按 `sm__throughput` 或 stall 原因排序，直接看到哪一行最贵。

### Replay 的代价

ncu 为了收集所有计数器，会把 kernel 重放多次。一个 kernel 跑 10 次才能采完所有指标，整体耗时可能是原来的几十倍。

这带来两个限制：不能用来测端到端性能；对依赖状态的 kernel（如 atomic 累加），replay 可能改变结果。分析单 kernel 时用它，报告整体性能时用干净的计时。

### 一个分析例子

一个 LayerNorm kernel，SpeedOfLight 显示 SM 12%、DRAM 85%。算术强度约 0.25，确实是带宽受限。

进一步看 MemoryWorkloadAnalysis：读 2 次、写 2 次，因为 kernel 先算均值和方差，再归一化，每个元素被读了两遍。把两次遍历融合成一次（Welford 算法或分块统计），DRAM 流量减半，性能接近 2 倍。

这类优化不需要更快的指令，只需要少读一遍。

## 6.4 Memory Profiling

显存问题分三类：越界、泄漏、碎片。

### 越界与非法访问

`compute-sanitizer` 是 CUDA 的 memcheck：

```bash
compute-sanitizer --tool memcheck --leak-check full python train.py
```

它会在每次内存访问时检查边界，报出具体的 kernel、地址和线程。开销很大（10–50 倍），只在复现 bug 时用。

其他工具：

| 工具 | 检查内容 |
| --- | --- |
| `--tool memcheck` | 越界、非法地址、泄漏 |
| `--tool racecheck` | shared memory 数据竞争 |
| `--tool initcheck` | 读取未初始化内存 |
| `--tool synccheck` | 同步原语误用 |

`racecheck` 对 reduction、softmax 这类有 shared memory 协作的 kernel 特别有用。数据竞争不一定每次都错，但会在特定调度下产生随机结果。

### PyTorch 显存

```python
print(torch.cuda.memory_summary(device=0, abbreviated=False))
```

输出区分 `allocated`（张量实际占用）和 `reserved`（缓存分配器向驱动申请的）。两者差距大，说明有碎片或缓存未释放。

记录分配历史：

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)
# ... 跑一段负载 ...
torch.cuda.memory._dump_snapshot("mem_snapshot.pickle")
```

把文件拖到 [PyTorch Memory Visualizer](https://pytorch.org/memory_viz)，能看到每个张量的分配时间、大小和调用栈。OOM 时这是最快找到“谁占了 20 GB”的方法。

### 碎片

推理服务跑久了 OOM，但 `nvidia-smi` 显示的占用并没有到上限，通常是碎片。

PyTorch 的缓存分配器按 block 管理显存。不同大小的请求交替出现，会留下无法复用的空洞。缓解方式：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

expandable segments 允许物理页不连续，显著降低碎片。vLLM 这类自己管理 KV Cache 的系统则用 block pool 和 PagedAttention 从源头避免。

### 泄漏

Python 的引用计数通常能及时释放张量，但以下情况会漏：

- 全局列表或字典持有张量；
- closure 捕获了大张量；
- CUDA Graph 捕获了输入缓冲区；
- 自定义 CUDA 扩展没有释放 `cudaMalloc` 的内存；
- 异常路径跳过了 `free`。

排查方法：跑固定负载，每轮打印 `allocated`，看是否单调上升。稳定后不再增长是正常的缓存；持续增长就是泄漏。

`torch.cuda.empty_cache()` 只把未使用的缓存还给驱动，不能修复泄漏，也不应该在高频路径调用——它会让下一次分配重新向驱动申请，增加延迟。

## 6.5 PyTorch Profiler and Trace Analysis

PyTorch Profiler 介于 nsys 和框架代码之间：它能识别 `aten` 算子、模块名和 Python 栈，适合定位框架层的开销。

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=3, active=5, repeat=1),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step in range(10):
        train_step()
        prof.step()

print(prof.key_averages().table(
    sort_by="self_cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")
```

`key_averages()` 按算子聚合，列出调用次数、CPU 时间、GPU 时间。`trace.json` 可以拖进 Chrome 的 `chrome://tracing` 或 [Perfetto](https://ui.perfetto.dev/)。

### 常见发现

| 现象 | 含义 | 处理 |
| --- | --- | --- |
| 大量 `aten::copy_` | 设备间或 dtype 转换 | 统一 dtype，避免不必要的 `.cpu()` |
| 几千个微小 kernel | 逐元素操作未融合 | `torch.compile` 或手写融合 kernel |
| `cudaLaunchKernel` 占比高 | CPU launch 受限 | CUDA Graphs、增大 batch |
| `cudaStreamSynchronize` 长 | 隐式同步 | 找出 `.item()`、`.cpu()`、print |
| `aten::to` 频繁 | 精度转换 | 混合精度策略不一致 |

`with_stack=True` 会记录 Python 调用栈，能直接看到是哪一行代码触发了拷贝，但开销也最大。定位到问题后关掉它再测性能。

### 与 nsys 的分工

- PyTorch Profiler：框架层，算子级，Python 栈，适合改代码；
- Nsight Systems：系统层，CUDA API、kernel、memcpy、stream，适合看整体时间线；
- Nsight Compute：kernel 内部，指令、内存、占用率，适合写 CUDA。

先用 nsys 找可疑区间，再用 PyTorch Profiler 定位到算子，最后用 ncu 分析 kernel。反过来会陷入细节。

## 6.6 GPU Utilization and Occupancy Analysis

`nvidia-smi` 里的 `GPU-Util` 是最容易被误读的指标。它的定义是“过去采样周期内有 kernel 在跑的时间比例”，不是 SM 的利用率。

一个只用一个 SM 的 kernel，也能让 `GPU-Util` 显示 100%。一个全是内存等待的 kernel，也能显示 100%。**100% 只说明 GPU 没闲着，不说明它跑得快。**

### 理论占用率

占用率是每个 SM 上活跃 warp 数与最大 warp 数之比。它由三个资源中最早耗尽的一个决定：

$$
\text{blocks per SM} = \min\left(
\left\lfloor \frac{\text{registers}}{\text{regs per thread} \times \text{threads per block}} \right\rfloor,
\left\lfloor \frac{\text{shared memory}}{\text{smem per block}} \right\rfloor,
\left\lfloor \frac{\text{max warps}}{\text{warps per block}} \right\rfloor
\right)
$$

H100 每个 SM 有 65,536 个寄存器、228 KB shared memory、最多 64 个 warp。

一个 block 有 128 个线程（4 个 warp）、每线程 40 个寄存器、32 KB shared memory：

- 寄存器：`65536 / (40 × 128) = 12.8`，取 12 个 block；
- shared memory：`228 / 32 = 7.1`，取 7 个 block；
- warp：`64 / 4 = 16` 个 block。

瓶颈是 shared memory，只能放 7 个 block，即 28 个 warp，占用率 43.75%。

如果把 shared memory 降到 16 KB，就能放 14 个 block、56 个 warp，占用率 87.5%。这就是为什么 tile 大小要权衡：tile 越大，复用越多，但占用率越低。

### 实测占用率

```bash
ncu --section Occupancy --section LaunchStats python bench.py
```

关注 `Achieved Occupancy` 和 `Theoretical Occupancy` 的差距。差距大通常是因为：

- block 大小不是 32 的倍数，warp 内有空闲 lane；
- 负载不均衡，部分 block 提前结束；
- 有 barrier 或长依赖，warp 虽然驻留但不能发射。

### Wave quantization

GPU 一次能驻留的 block 数是有限的。假设 132 个 SM、每个 SM 8 个 block，一波能跑 1,056 个 block。

如果 grid 有 1,100 个 block，就要跑两波：第一波 1,056 个，第二波 44 个。第二波只用了 4% 的 SM，但时间上和满波差不多。这叫尾效应（tail effect）。

$$
\text{效率} = \frac{\text{block 总数}}{\text{波数} \times \text{每波容量}}
$$

1,100 个 block、两波、容量 1,056，效率约 52%。如果能把 grid 调整到 1,056 或 2,112 个 block，效率会显著提高。

GEMM 的 tile 量化是同一问题的另一面。M=4097、tile 高度 128，需要 33 个 tile，最后一个 tile 只有 1 行有效，却要花一个完整 tile 的时间。padding 到 4224 或换用更小的 tile，能减少浪费。

### 小 batch 的 decode

decode 时 batch 1、hidden 8192，一个 GEMV 的 grid 可能只有几百个 block。对 132 个 SM 来说，这个并行度不算低，但每个 block 的工作量很小，kernel 时间只有几微秒，launch 和调度开销占比很高。

这时提高占用率没用，因为总工作量太小。正确方向是减少 kernel 数量（融合）、用 CUDA Graphs 降低 launch 开销，或者增大 batch 把多个请求合并。

## 6.7 Numerical Correctness and Debugging

LLM 的数值错误往往不报错，只是让输出变差。Loss 曲线看起来正常，评测分数掉了两个点，很难归因。

### 浮点格式的差异

| 格式 | 指数位 | 尾数位 | 最大值 | 最小正规数 | 相对精度 |
| --- | --- | --- | --- | --- | --- |
| FP32 | 8 | 23 | 3.4e38 | 1.2e-38 | 约 1e-7 |
| TF32 | 8 | 10 | 3.4e38 | 1.2e-38 | 约 1e-3 |
| BF16 | 8 | 7 | 3.4e38 | 1.2e-38 | 约 8e-3 |
| FP16 | 5 | 10 | 65,504 | 6.1e-5 | 约 1e-3 |
| FP8 E4M3 | 4 | 3 | 448 | 约 1.6e-2 | 约 6e-2 |
| FP8 E5M2 | 5 | 2 | 57,344 | 约 6.1e-5 | 约 1e-1 |

BF16 和 FP16 的选择不是精度高低的问题，是范围问题。BF16 的尾数比 FP16 少，但指数范围和 FP32 一样。训练时梯度可能小到 1e-8，FP16 直接下溢成 0，BF16 还能表示。

FP8 的范围更窄，必须配合 per-tensor 或 per-block 的缩放因子。缩放因子估计错了，不是溢出就是精度损失。

### 常见的数值 bug

- **Softmax 溢出**：没有减去最大值，`exp(100)` 在 FP16 里是 inf。标准写法是先减 `max`。
- **除零**：LayerNorm 的方差为 0，`1/sqrt(0)` 变成 inf。加 `epsilon`。
- **累加精度**：FP16 累加 4096 项，误差会累积。用 FP32 累加器。
- **mask 用 0 而不是 -inf**：softmax 会给被 mask 的位置分配概率。
- **RoPE 位置偏移**：PagedAttention 的 block 索引和逻辑位置不一致，位置编码就错了。
- **GQA 头映射**：64 个 Q 头对 8 个 KV 头，映射写错不会报错，只是效果变差。

### 定位工具

```python
# 让异常在产生它的前向位置抛出，而不是在反向
torch.autograd.set_detect_anomaly(True)

# 让 kernel 同步执行，错误现场停在正确的调用上
# 环境变量：CUDA_LAUNCH_BLOCKING=1
```

`CUDA_LAUNCH_BLOCKING=1` 会严重降低性能，只用于调试。它把异步 launch 变成同步，报错时就能拿到准确的调用栈。

`compute-sanitizer` 能定位越界和未初始化读取。`TORCH_USE_CUDA_DSA` 打开设备端断言，越界时直接打印文件和行号。

### 确定性

默认情况下，CUDA 的 `atomicAdd` 顺序不固定，同样的输入可能得到不同的结果。要复现 bug，需要打开确定性模式：

```python
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
```

代价是性能下降，某些算子会退化成慢实现。确定性只用于调试和小规模回归，不要在生产环境默认打开。

### 与参考实现对比

写新 kernel 时，先写一个朴素但正确的版本，用 FP64 或 FP32 计算，再和新 kernel 对比：

```python
ref = reference_attention(q.double(), k.double(), v.double())
out = flash_attention(q, k, v)
torch.testing.assert_close(out.double(), ref, rtol=1e-3, atol=1e-3)
```

容差要按数据类型定。FP16 的 `rtol` 取 1e-3 是合理的，FP32 可以取 1e-5。不要用“看起来差不多”作为标准。

### 分布式问题

多卡训练的挂起和错误更隐蔽：

- 某个 rank 的 loss 变成 NaN，其他 rank 还在等 AllReduce；
- 数据加载器在某个 rank 上提前结束；
- NCCL 超时，但日志只说 `Watchdog caught collective operation timeout`。

排查手段：

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

`NCCL_DEBUG=INFO` 会打印每个集合通信的参与者、大小和耗时。如果某个 rank 的通信序列和别人不一致，就是代码里有条件分支没有同步。

### 静默错误

最危险的是不报错但算错。性能测试通过、loss 正常，但模型效果下降。防守手段是：

- 每次改动后跑一个小规模评测集，不只看 loss；
- 对比 logits 的统计量（均值、方差、最大值）；
- 在关键层加断言，检查 NaN/Inf 和数值范围；
- 保留一个端到端的 golden test，输出和基线在容差内一致。

## 6.8 Performance Regression Testing

性能回归不会让测试变红，只会让账单变高。它需要独立的工程化机制。

### 微基准

对关键 kernel 建一个独立的 benchmark，固定 shape、固定输入、固定 GPU：

```python
import torch

def bench(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters
```

Triton 的 `triton.testing.do_bench` 做了同样的事，并且会清理 L2 cache，避免缓存命中导致结果偏乐观。测真实负载时不要清 cache；测 kernel 本身时可以清。

### Shape 矩阵

不要只测一个 shape。至少覆盖：

| 场景 | 典型 shape |
| --- | --- |
| Decode，小 batch | B=1，S=1，past=2K |
| Decode，大 batch | B=64，S=1，past=2K |
| Prefill，短 | B=1，S=512 |
| Prefill，长 | B=1，S=16K |
| GEMM | M=N=K=4096 |
| Attention | B=8，H=32，S=4K |

一个优化可能让 decode 快 20%，却让 prefill 慢 5%。只看一个 shape 会做出错误判断。

### CI 集成

性能测试要跑在固定硬件上。共享 runner 的噪声太大，云上的实例型号也可能变。常见做法是自建一台带 GPU 的 runner，锁频，关掉其他负载。

流程：

1. 每个 PR 跑微基准，和主干基线比较；
2. 超过阈值（如 5%）标记为可疑；
3. 人工确认后，要么修复，要么更新基线；
4. 夜间跑端到端 serving benchmark 和显存峰值；
5. 结果写入数据库，画趋势图。

阈值要基于噪声设定。先在空闲机器上重复跑 30 次，算出变异系数。如果噪声是 3%，5% 的阈值合理；如果噪声是 10%，阈值就要放到 15%，否则全是误报。

### 正确性门禁

性能测试必须和正确性测试一起跑。一个 kernel 变快 2 倍但算错，比慢 2 倍更糟。

```python
out = kernel(x)
ref = reference(x)
torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
```

对量化 kernel，容差要放宽，但仍然要有一个上限。对采样、beam search 这类有随机性的路径，固定 seed 后比较输出序列。

### 记录环境

一次性能数字离开环境就没有意义。至少记录：

- GPU 型号、数量、互联方式；
- 驱动、CUDA、cuBLAS、框架版本；
- 是否锁频、功耗上限；
- 编译选项和 `-gencode` 架构；
- 输入数据来源和 shape。

把这些写进结果文件，和 commit hash 一起存。半年后回看时，才知道那个“快了 30%”是在什么条件下测的。

## 本章小结

- CPU 瓶颈常被忽略；`perf`、off-CPU 分析和 `py-spy` 能定位调度、锁、IO 和 GIL 问题。
- Nsight Systems 看时间线，找 GPU 空洞、意外拷贝和未重叠的 stream；NVTX 能把逻辑阶段标出来。
- Nsight Compute 看 kernel 内部：先看 SpeedOfLight 判断算力/带宽受限，再看 stall reason 和访存效率。
- `compute-sanitizer` 查越界、竞争和未初始化读取；PyTorch 的显存快照查泄漏和碎片。
- `nvidia-smi` 的利用率不等于 SM 效率；占用率由寄存器、shared memory 和 warp 上限共同决定，还要注意 wave quantization。
- 数值 bug 往往静默；要理解浮点格式的范围和精度，用参考实现和容差测试兜底。
- 性能回归需要固定硬件、shape 矩阵、噪声阈值和正确性门禁，单独一次跑分不算测试。

## 延伸阅读

- NVIDIA, [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).
- NVIDIA, [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html).
- NVIDIA, [Compute Sanitizer User Guide](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).
- NVIDIA, [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).
- PyTorch, [Profiler Recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).
- PyTorch, [Understanding GPU Memory](https://pytorch.org/blog/understanding-gpu-memory-1/).
- Gregg, [The Flame Graph](https://dl.acm.org/doi/10.1145/2909476), 2016.
- Goldberg, [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://dl.acm.org/doi/10.1145/103162.103163), 1991.
