---
chapter: 16
part: 3
title: Quantized and Compressed Runtime
level: tbd
status: draft
prerequisites: [12, 14]
lab:
---

# Chapter 16. Quantized and Compressed Runtime

Llama 2 7B 的 FP16 权重是 13.5 GB，INT4 只有 3.5 GB，字节数只有前者的 1/3.9。在 A100 上实测，decode 的 TPOT 从 11 毫秒降到 4 毫秒，加速约 2.7 倍，不是 3.9 倍。

差距来自三个地方：KV Cache 没有量化、kernel 的固定开销、以及反量化指令占用了发射槽。量化格式解决“存得小”，runtime 解决“跑得快”，两者之间的差距就是这一章的内容。

这一章从文件格式讲到 kernel：scale 存在哪、加载时怎么转换、反量化怎么融进 GEMM、CPU 和 GPU 的路径有什么不同、KV Cache 怎么量化，最后用一个端到端算例把数字对上。

## 16.1 Quantized Model Formats

量化模型不是一个“INT4 的 safetensors”，而是一组约定：权重怎么打包、scale 什么形状、group 多大、zero point 在哪。不同格式之间不能直接互读，因为打包顺序和 scale 布局都不一样。

| 格式 | 产出工具 | 容器 | 量化粒度 | scale 存在哪 | 绑定硬件 |
| --- | --- | --- | --- | --- | --- |
| GGUF | llama.cpp | 单文件 | block 32 / 256 | 每个 block 内联 FP16 | 否 |
| GPTQ | AutoGPTQ、GPTQModel | safetensors | group 128 | 独立 scales / qzeros 张量 | 否 |
| AWQ | AutoAWQ | safetensors | group 128 | 独立 scales / qzeros 张量 | 否 |
| NF4 | bitsandbytes | safetensors + quant_state | block 64 | absmax 与嵌套 absmax | 否 |
| TRT-LLM engine | trtllm-build | `.engine` plan | per-channel / per-block | 编译进 plan | 是 |
| safetensors + quant_config | 通用 | safetensors | 由 config 决定 | 由实现决定 | 否 |

**GGUF** 是 llama.cpp 的单文件格式：一个 header 存 key-value 元数据（架构、超参、tokenizer），后面是张量数据。量化参数不单独存，而是内联在每个 block 里。Q4_0 的 block 是 32 个权重加一个 FP16 的 `d`，共 18 字节；反量化是 `w = d * (q - 8)`，zero point 固定为 8，不占存储。

| 类型 | block 大小 | 每 block 字节 | bits/weight |
| --- | --- | --- | --- |
| Q2_K | 256 | 84 | 2.63 |
| Q4_0 | 32 | 18 | 4.50 |
| Q4_K | 256 | 144 | 4.50 |
| Q5_K | 256 | 176 | 5.50 |
| Q6_K | 256 | 210 | 6.56 |
| Q8_0 | 32 | 34 | 8.50 |

Q4_K、Q5_K 的 256 个权重分 8 个子块，每个子块有自己的 6-bit scale 和 6-bit min，外面还有一个 FP16 的 super-scale 和 super-min。所以 Q4_K 虽然也是 4.5 bpw，精度比 Q4_0 好，代价是反量化的分支更多。`Q4_K_M` 不是一种新类型，而是“大部分张量用 Q4_K、少数敏感张量用 Q6_K”的混合策略。

**GPTQ** 把 8 个 4-bit 值打包进一个 int32，`qweight`、`scales`、`qzeros` 是三个独立张量，放在同一个 safetensors 里。group size 写在 `config.json` 的 `quantization_config` 里。`desc_act` 为真时还会有一个 `g_idx`，记录激活重排后的列顺序。GPTQ 的 qweight 通常按 `in_features` 方向打包，scales 的形状是 `[in/group, out]`，但不同实现的转置约定不一致，加载时必须按 kernel 的期望核对。

**AWQ** 的存储布局和 GPTQ 接近，区别在量化前的变换：它按激活幅值挑出约 0.1% 的显著通道，对这些通道做 per-channel 缩放再量化，scale 在推理时折回权重。所以 AWQ 的 `scales` 同时承担 group scale 和通道缩放两个角色。

**bitsandbytes NF4** 的元数据不在 safetensors 的 header 里，而在一个 `quant_state` 对象中：每个 64 权重块的 absmax、第二层的嵌套 absmax、blocksize、原始 dtype。这也是为什么加载 bnb 模型需要额外的 `quant_state` 文件或字段。

**TensorRT-LLM engine** 是另一种东西：它不是权重格式，而是编译产物。权重、scale、选好的 kernel、甚至 grid 配置都序列化进 plan 文件，和 GPU 的 SM 版本、TensorRT 版本绑定。换一张卡就要重新 build，换来的是没有运行时解释开销。

一个统一的账：权重的实际位宽不是 4，而是

$$
\text{bpw} = b + \frac{16}{g} + \frac{16}{g}
$$

`b` 是量化位数，`g` 是 group size，两项分别是 scale 和 zero point（FP16 各 16 bit）。`b=4, g=128` 时是 4.25 bpw；如果 zero point 用 4-bit 打包，是 4.16 bpw；NF4 的 double quantization 是 4.13 bpw。7B 模型按 4.25 bpw 算是 3.6 GB，按 4.13 算是 3.5 GB。

## 16.2 Model Loading and Weight Conversion

加载的第一原则是不要复制。safetensors 支持 mmap：读 header 拿到每个张量的偏移，直接把文件映射进地址空间，权重留在 page cache 里，多个进程共享同一份物理内存。GGUF 同理。一个 3.5 GB 的 4-bit 模型，冷启动要从 NVMe 读 3.5 GB，按 3 GB/s 算是 1.2 秒；热启动直接命中 page cache，接近 0。这也是为什么服务重启后第一个请求的 TTFT 明显偏高。

mmap 的代价是 page fault 分散在首次访问时。如果在意首 token 延迟，可以在启动时做一次顺序预读，或者用 `MAP_POPULATE`。

对齐是第二个坑。Tensor Core 的 `ldmatrix` 按 16 字节读取，打包后的行如果长度不是 16 的倍数，每一行都要 padding。多数 LLM 的 `in_features` 是 128 的倍数，权重本身天然对齐；真正容易出错的是 scale 张量的布局，不是权重。

加载时做 dtype 转换是有成本的：

- BF16 转 FP16，7B 模型要读 14 GB、写 14 GB，按 2 TB/s 算是 14 毫秒；
- 如果不做流式处理，峰值内存会翻倍到 28 GB；
- 转换后的权重和原文件不能再共享 page cache。

更贵的转换是布局重排。某些 kernel 要求权重按 Tensor Core 的 swizzle 排列（比如 NVFP4 的 128×4 块布局），加载时要逐块重排，成本接近一次完整的读写。这类转换应该在离线做一次、存成最终格式，不要每次启动都做。

加载后建议做一个自检：取一行打包权重，用参考实现反量化，和原始 FP16 权重比对。GPTQ 和 AWQ 的打包顺序错一位，模型不会报错，只是输出逐渐退化，这种 bug 靠看 loss 是发现不了的。

## 16.3 On-the-fly Dequantization

W4A16 的推理路径是：权重以 4-bit 存储，进 GEMM 之前反量化成 FP16，Tensor Core 做 FP16 乘加。反量化的公式就是仿射变换：

$$
w = (q - z) \cdot s
$$

`q` 是解包出来的 4-bit 整数，`z` 是 zero point，`s` 是 group scale。权重 scale 按输出通道或 group；激活若也量化，scale 要按 token，因为不同 token 的幅值差很多。每个权重需要：一次移位和掩码解包、一次减法、一次整数转浮点、一次乘法，大约 4–6 条指令。

这笔开销能不能被藏住，取决于瓶颈在哪。decode 时 batch 1，每个权重只参与一次乘加，算术强度约 1 FLOP/byte，是彻底的带宽受限。A100 上每字节能分到的计算时间：

$$
\frac{0.5 \text{ B} / 2.04 \text{ TB/s}}{8 \text{ ops} / 312 \text{ TFLOPS}} \approx 9.6
$$

也就是说，带宽给每个权重留的时间是反量化所需时间的 9.6 倍，反量化完全被掩盖。真正的问题不是算力，是**指令发射**：如果解包写成标量循环，每个权重 6 条指令，SM 的发射带宽会被打满，反而拖慢访存。

反量化的收益什么时候消失？把两块时间写成 `M` 的函数。权重字节是 `0.5KN`，计算是 `(2M + c)KN`，`c` 是每个权重的反量化指令数：

$$
\frac{(2M + c)KN}{P} = \frac{0.5KN}{B}
\;\Longrightarrow\;
M^{*} = \frac{0.5P}{2B} - \frac{c}{2}
$$

A100 上 `P = 312` TFLOPS、`B = 2.04` TB/s、`c = 8`，得到 `M* ≈ 34`。`M < 34` 时反量化被带宽掩盖；`M > 34` 进入计算受限，这时候反量化的指令会计入总时间。但注意，计算受限时每个权重被 `M` 行复用，`c` 摊薄到 `2M` 次乘加上，占比是 `c / (2M)`，`M = 512` 时只有 0.8%。

真正会让收益归零的是**没有做寄存器复用**：每个 `(weight, row)` 对都反量化一次，成本变成 `c \cdot M`，prefill 上直接多出几倍的指令。kernel 的 tile 必须让一份反量化后的权重服务 `BLOCK_M` 行。

还有一个隐性成本是 scale 的加载。group 128 时每 128 个权重读一个 FP16 scale，看似不多，但如果 scale 按 `[N, K/group]` 存放而线程按 K 方向连续访问，就会变成跨步读，每次只取 2 字节，浪费整个 32 字节的 sector。scale 必须按 kernel 的访问顺序排布。

## 16.4 Fused Quantization Kernels

生产 kernel 不会“先反量化到一块临时 buffer，再调 GEMM”。反量化和 GEMM 在同一个 kernel 里，权重从 HBM 读进来、在寄存器里解包、直接进 Tensor Core。

下面是一段 Triton 伪代码，展示 W4A16 的结构：

```python
@triton.jit
def w4a16_gemm(X, Wq, Ws, Wz, Out,
               M, N, K, GROUP: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)

    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        # Wq: int32，沿 K 方向每字打包 8 个 4-bit（伪代码，实际每字只加载一次）
        word = tl.load(Wq + (rk // 8)[:, None] * N + rn[None, :])
        q = (word >> ((rk % 8) * 4)[:, None]) & 0xF
        g = rk // GROUP
        s = tl.load(Ws + g[:, None] * N + rn[None, :])   # [BK, BN]
        z = tl.load(Wz + g[:, None] * N + rn[None, :])
        w = (q.to(tl.float32) - z) * s
        x = tl.load(X + rm[:, None] * K + rk[None, :])   # [BM, BK]
        acc += tl.dot(x, w.to(tl.float16))

    tl.store(Out + rm[:, None] * N + rn[None, :], acc.to(tl.float16))
```

这段代码有几个刻意简化：真实的 kernel 用 `ld.global` 一次读 128 bit，在寄存器里用 `PRMT` 和 `LOP3` 一次解出 8 个值，不会对每个元素重复加载同一个 word；scale 会预先重排成和权重一致的 tile 顺序；`tl.dot` 之前可能还要做一次 `cvt` 到 BF16。

融合的价值在 epilogue。以 SwiGLU 为例，标准路径是：

```text
gate = gemm(x, W_gate)     # 写 [B, S, d_ff]
up   = gemm(x, W_up)       # 写 [B, S, d_ff]
h    = silu(gate) * up     # 读两个，写一个
```

`B=1, S=4096, d_ff=14336, BF16` 时每个张量是 117 MB。标准路径三次写、两次读，合计约 585 MB 流量；把 gate/up 合成一个 GEMM、再把 `silu(gate) * up` 放进 epilogue，中间结果只留在寄存器或 shared memory，只写最终的 `[B, S, d_ff]`，省掉约 470 MB。prefill 时这是一层上百微秒的量级，几十层累积起来可观。

更激进的融合是把反量化、GEMM、激活、残差加全部放进一个 persistent kernel，用 split-K 处理 decode 的小 `M`。这类 kernel 的调试成本很高，通常只在 TensorRT-LLM、Marlin 这样的专用实现里出现。

## 16.5 CPU and GPU Quantized Execution

CPU 的量化路径和 GPU 完全不同。CPU 没有 Tensor Core，靠 SIMD 向量指令：

- **AVX2**：256-bit 向量，一条指令处理 32 个 int8；
- **AVX-512**：512-bit，64 个 int8；
- **AVX-512 VNNI**：`VPDPBUSD` 把 4 对 int8 的点积累加到一个 int32，一条指令 64 次乘加，是 AVX2 的两倍以上。

llama.cpp 的做法是“权重保持量化，激活现场量化”：把激活按 32 个一组量化成 Q8_0（int8 + FP16 scale），和 Q4_0 权重做块内点积。核心是 `_mm256_maddubs_epi16`（无符号乘有符号，相邻两两相加）加 `_mm256_madd_epi16`（成对累加到 int32），最后用 FP32 乘回两个 scale。

CPU 上几乎总是带宽受限。DDR5-5600 双通道的理论带宽是 89.6 GB/s，实际能拿到 60–70 GB/s。一个 4 GB 的 Q4 模型，每 token 要读一遍权重：

$$
\frac{4 \text{ GB}}{60 \text{ GB/s}} \approx 67 \text{ ms}
$$

也就是 15 token/s 左右。这和量化算法关系不大，换更复杂的 4-bit 格式也不会更快，除非它能减少字节数。

GPU 走另一条路：

| 能力 | 引入代次 | 用途 |
| --- | --- | --- |
| INT8 Tensor Core | Turing（SM 75） | W8A8，INT8 GEMM |
| INT4 Tensor Core | Ampere | 很少用于 LLM，精度和布局都不友好 |
| FP8 Tensor Core | Hopper（SM 90） | W8A8 / FP8 KV，原生 GEMM |
| FP4 Tensor Core | Blackwell | NVFP4，原生 4-bit GEMM |
| W4A16 | 无原生指令 | 反量化到 FP16 后走 FP16 Tensor Core |

关键区别：**W4A16 在 Blackwell 之前没有原生指令**。A100、H100 上的 4-bit 权重必须反量化成 FP16 才能进 Tensor Core，所以 4-bit 省的是带宽，不是算力。Marlin 这类 kernel 的做法是把反量化压在寄存器里、用异步拷贝预取下一块权重，把 decode 的带宽利用率推到 80% 以上。

FP8 是另一回事：Hopper 的 FP8 GEMM 是原生的，算力是 FP16 的两倍，权重和激活的字节同时减半。compute-bound 的 prefill 上，FP8 能拿到接近 2 倍的加速；4-bit 权重在同样的场景下只能省带宽，算力不变。

## 16.6 Quantized KV Cache

长上下文时，KV Cache 会超过权重。Llama 2 7B 每 token 的 KV 是：

$$
2 \times 32 \times 32 \times 128 \times 2 \text{ B} = 512 \text{ KiB}
$$

4096 token 是 2 GiB，32768 token 是 16 GiB，比 13.5 GB 的 FP16 权重还大。这时 decode 的带宽瓶颈从权重转移到 KV，量化 KV 的收益比量化权重更直接。

KV 量化的粒度选择有明确的方向性：

- **K 按 per-channel 量化**。K 的某些通道在不同 token 上持续偏大，是稳定的离群通道；per-token 缩放会被这些通道主导，其他通道被压扁。
- **V 按 per-token 量化**。V 的离群值随 token 变化，per-channel 反而不好。
- **group-wise** 是两者的折中，group 32–128，兼顾精度和 scale 开销。

| 方案 | 粒度 | 位宽 | 相对 FP16 | 典型精度影响 |
| --- | --- | --- | --- | --- |
| FP8 E4M3 | per-tensor / per-head | 8 | 0.5 倍 | 几乎无损 |
| INT8 | per-token | 8 | 0.5 倍 | 很小 |
| INT4 group 128 | per-token + group | 4.25 | 0.27 倍 | 小到中 |
| KIVI | K per-channel，V per-token | 2 | 0.25 倍 | 中，长上下文明显 |

KIVI 是 2-bit KV 的代表工作。它不训练，直接对已有的 KV 做量化：K 沿 channel 维分组、V 沿 token 维分组，group size 32–128，最近的若干 token 保留 FP16 的 residual window，因为新 token 的 KV 还在被频繁写入和读取。在这个配置下，KV Cache 压到 FP16 的四分之一，长上下文吞吐明显提升。

精度风险集中在 attention score。softmax 对 score 的误差很敏感，K 的量化误差会被 `QK^\top` 放大 `\sqrt{d}` 倍；V 的误差则直接加到输出上。2-bit KV 在短上下文和普通对话上看起来正常，在 needle-in-a-haystack 这类检索任务上掉得很明显。生产上更常见的选择是 FP8 KV，它几乎无损，而且 Hopper 有原生支持。

## 16.7 Memory Bandwidth versus Compute Trade-offs

判断量化有没有用，先算算术强度：

$$
I = \frac{2MNK}{2(MK + NK + MN)} \;\text{FLOP/byte}
$$

`M` 是 batch×序列长度，`N` 和 `K` 是 GEMM 的输出和 reduction 维度。decode 时 `M=1`，分子是 `2NK`，分母里 `NK` 项主导，`I ≈ 1`。prefill 时 `M` 大，`I` 随 `M` 增长。

和机器的脊点比：

| GPU | FP16 算力（TFLOPS） | 带宽（TB/s） | 脊点（FLOP/byte） |
| --- | --- | --- | --- |
| A100 80GB | 312 | 2.04 | 153 |
| H100 SXM | 989 | 3.35 | 295 |
| B200 | 约 2250 | 约 8 | 约 280 |

decode 的 `I ≈ 1`，离脊点差两个数量级，是彻底的带宽受限；prefill 在 `M` 到几百时跨过脊点，进入计算受限。

这决定了量化的收益形态：

- **带宽受限（decode、小 batch）**：时间是 `字节数 / 带宽`，4-bit 权重直接把时间压到四分之一。KV 量化同理。
- **计算受限（prefill、大 batch）**：时间是 `FLOPs / 算力`，权重位宽不影响时间。只有换到有原生指令的格式（FP8、FP4）才能加速；W4A16 在这个区间甚至可能因为反量化略慢。
- **中间区间**：两者叠加，收益介于 1 倍和压缩比之间。

还有一个容易忽略的项：batch 增大后，权重的读取被摊薄，KV 的读取按 `batch × S` 增长。`M` 大到一定程度，KV 成为主要流量，权重是否 4-bit 不再重要，KV 是否量化才重要。

## 16.8 End-to-end Performance Evaluation

用一个具体算例把上面的数字串起来。配置：Llama 2 7B（6.74B 参数），单张 A100 80GB，输入 2048 token，输出 256 token，batch 1，vLLM 后端。

**权重和带宽**：

- FP16：13.5 GB，TPOT 下限 `13.5 / 2.04 = 6.6 ms`；
- W8A8：6.8 GB，下限 3.3 ms；
- W4A16 group 128：3.6 GB，下限 1.8 ms。

**TTFT**：prefill 的 FLOPs 是 `2 × 6.74e9 × 2048 ≈ 27.6 TFLOP`。A100 的 FP16 峰值是 312 TFLOPS，理论上限 88 ms；batch 1 的 prefill MFU 通常在 35%–60%，实际在 150–250 ms。W4A16 的 prefill 仍走 FP16 Tensor Core，反量化摊薄后开销很小，TTFT 和 FP16 接近；W8A8 用 INT8 Tensor Core，理论算力翻倍，TTFT 可能略好。

**显存**：FP16 权重 13.5 GB，KV（2048 token）1 GiB，加上激活和 CUDA context，约 16 GB；W4A16 权重 3.6 GB，总计约 6 GB。权重在请求间共享，所以 80GB 卡上的并发数主要由 KV 决定，FP16 和 4-bit 都能放几十个 2K 请求，差距不大。量化的显存收益在单请求场景更明显：4-bit 让 7B 装进 8 GB 消费级卡，FP16 做不到。

| 指标 | FP16 | W8A8 | W4A16 |
| --- | --- | --- | --- |
| 权重显存 | 13.5 GB | 6.8 GB | 3.6 GB |
| TPOT 理论下限 | 6.6 ms | 3.3 ms | 1.8 ms |
| TPOT 典型实测 | 10–13 ms | 6–8 ms | 4–5 ms |
| TTFT（2K prefill） | 150–250 ms | 130–220 ms | 150–250 ms |
| 单请求吞吐 | 约 80–100 tok/s | 约 130–160 tok/s | 约 200–250 tok/s |
| 质量（perplexity） | 基准 | +0.01–0.05 | +0.1–0.3 |

实测 TPOT 是理论下限的 1.5–2.8 倍。差距来自 KV 读取（2048 token 时约 0.5 ms，占 4-bit 路径的 12%）、kernel launch、采样、以及调度开销。4-bit 路径的固定开销占比更高，所以加速比 2.7 倍而不是 3.9 倍。

评估时要注意三件事：

- **分别报 TTFT 和 TPOT**。量化对 prefill 和 decode 的影响不同，混在一个“平均延迟”里会掩盖问题。
- **在目标 batch 上报吞吐**。batch 1 的结论不能外推到 batch 64；后者进入计算受限，4-bit 的收益会缩水。
- **质量要测任务指标**。perplexity 上升 0.2 可能对应 MMLU 掉 1 个点，也可能什么都不掉，必须在自己关心的任务上测。

最后，量化的收益不是单调的。一个 4-bit 模型如果因为精度问题需要更大的 batch 或更多的重试，端到端成本可能反而更高。压缩比是手段，单位请求的成本才是目标。

## 本章小结

- 量化格式的核心差异是元数据放在哪：GGUF 内联在 block 里，GPTQ/AWQ 用独立张量，bnb 用 quant_state，TRT-LLM 编译进 plan。
- 实际位宽是 `b + 16/g + 16/g`，4-bit group 128 是 4.25 bpw，不是 4.0。
- 加载要用 mmap 避免复制，注意 16 字节对齐和打包顺序；布局重排应该离线做。
- on-the-fly 反量化在 decode 被带宽掩盖，A100 上的分界点是 `M ≈ 34`；计算受限时每个权重被复用，反量化摊薄。
- 融合 kernel 把反量化和 GEMM 放在一起，epilogue 融合 SiLU·Mul 能省掉大量中间张量流量。
- CPU 靠 AVX2/AVX-512 VNNI 做块内点积，几乎总是带宽受限；GPU 的 W4A16 在 Blackwell 之前没有原生指令，省带宽不省算力。
- KV Cache 在长上下文会超过权重；K 按 channel、V 按 token 量化是更合理的方向，FP8 是低风险选择，2-bit 需要仔细评估。
- 算术强度决定量化收益：decode 带宽受限，收益接近压缩比；prefill 计算受限，只有原生低精度指令能加速。
- 端到端评估要分开看 TTFT、TPOT、吞吐、显存和质量；4-bit 的实测加速通常是 2.5–3 倍，不是 4 倍。

## 延伸阅读

- Frantar et al., [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323), 2023.
- Lin et al., [AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978), 2024.
- Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314), 2023.
- Liu et al., [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750), 2024.
- Xiao et al., [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438), 2022.
- Pope et al., [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102), 2022.
- ggml-org, [llama.cpp](https://github.com/ggml-org/llama.cpp).
- NVIDIA, [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).
- IST-DASLab, [Marlin: FP16xINT4 mixed-precision GEMM kernel](https://github.com/IST-DASLab/marlin).
- NVIDIA, [CUTLASS](https://github.com/NVIDIA/cutlass).
