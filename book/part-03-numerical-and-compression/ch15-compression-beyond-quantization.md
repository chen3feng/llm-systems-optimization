---
chapter: 15
part: 3
title: Compression Beyond Quantization
level: tbd
status: draft
prerequisites: [12]
lab:
---

# Chapter 15. Compression Beyond Quantization

SparseGPT 一次性剪掉 OPT-175B 一半的权重，不重新训练，perplexity 几乎没有变化。把同一份权重放到 A100 上，吞吐和 dense 版本一样。

GPU 没有“跳过零”的指令：非结构化稀疏在存储上少了，计算路径却不认识它，Tensor Core 仍按 dense 的形状读每个字。

压缩的收益要同时满足两个条件：**数据真的变少，执行路径也认识这种变少。** 这一章讲剪枝、稀疏计算、蒸馏、低秩、权重共享、模型合并和 LoRA，以及它们各自在哪里失效。

## 15.1 Structured and Unstructured Pruning

剪枝分两类。非结构化（unstructured）剪掉任意位置的单个权重；结构化（structured）剪掉整个通道、整个 attention head、整个 FFN 中间维度，或者一块连续的权重。

最基础的打分是 magnitude，即 `|W_ij|`，全局取阈值或逐层取阈值。它在小模型上有效，在 LLM 上超过 40%–50% 后迅速失效。原因是权重的重要性不由自身大小决定，而由它乘上的那个激活决定。

Wanda 把激活乘进来：

$$
S_{ij} = |W_{ij}| \cdot \|X_j\|_2
$$

`X_j` 是校准集中第 `j` 个输入特征的向量，范数在 token 维度上取。剪枝时在**每个输出行内部**比较 `S_ij`，全局比较会把某些行整体剪光。它不需要矩阵求逆，只要一次前向收集激活，7B 模型几分钟完成。

SparseGPT 走的是逐层重建。给定一层的权重 `W` 和校准输入 `X`，最小化输出误差：

$$
\min_{\hat{W}} \; \| WX - \hat{W}X \|_2^2
$$

按列剪枝，每剪掉一列，用 Hessian 逆把误差补偿到尚未剪的列上：

$$
\delta_j = -\frac{w_j - \hat{w}_j}{[H^{-1}]_{jj}} H^{-1}_{:,j}, \quad H = 2XX^\top
$$

这一步和 OBS（optimal brain surgeon）同源。代价是要对每层的 `H` 求逆，而 `H` 是 `[d_in, d_in]`，7B 模型最大的 `d_in` 到 11008，求逆在 GPU 上可行但比 Wanda 贵一个量级。收益是在 50%–60% 稀疏下仍然可用，模型越大越明显。

| 粒度 | 剪枝单位 | 50% 稀疏下的存储 | 硬件加速 | 7B 级精度 |
| --- | --- | --- | --- | --- |
| 非结构化 | 单个权重 | 需索引，可能比 dense 大 | 无 | 最好 |
| 2:4 | 每 4 个连续权重 | 约 1.88 倍压缩 | Ampere+ Sparse Tensor Core | 中等 |
| 块稀疏 32×32 | 32×32 的块 | 索引开销小于 1% | 需专用 kernel | 中等 |
| 通道/head | 整行、整个 head | 直接变小 | 直接变小 | 较差 |
| 层 | 整个 block | 直接变小 | 直接变小 | 最差 |

2:4 是唯一被主流硬件支持的稀疏模式：每 4 个连续的 K 维权重里恰好保留 2 个，A100 上 FP16 的 Sparse Tensor Core 算力因此从 312 提到 624 TFLOPS。约束作用在 reduction 维度（K）上，选哪两个非零由剪枝算法决定。

## 15.2 Sparsity and Sparse Computation

稀疏矩阵的经典存储是 CSR：

```text
values   : nnz 个数值
col_idx  : nnz 个 int32 列号
row_ptr  : N+1 个 int32 行偏移
```

存储量是：

$$
\text{bytes}_{\text{CSR}} = n_{nnz}(b + 4) + (N+1) \times 4
$$

`b` 是数值的字节数。反直觉的是：**在 50% 稀疏下，CSR 可能比 dense 还大。** 一个 4096×4096 的 FP16 权重是 33.6 MB；剪掉一半后 `nnz = 8.39M`，CSR 需要 `8.39M × 6 + 16 KB ≈ 50 MB`，比 dense 大 50%。索引（4 字节）比数值（2 字节）还贵。

稀疏要真正省空间，必须换格式：

| 格式 | 索引开销 | 每原始权重位数（50% 稀疏） | 能否加速 |
| --- | --- | --- | --- |
| CSR | 4 字节/非零 | 24 bit | 否，gather 太散 |
| 块稀疏 32×32 | 4 字节/块 | 约 8.0 bit | 需专用 kernel |
| 2:4 | 2 bit/4 权重 | 8.5 bit | Sparse Tensor Core |
| dense | 0 | 16 bit | 基准 |

2:4 的账这样算：每 4 个位置存 2 个 FP16 值（4 字节）加 2 bit 元数据，平均每个权重 `(32 + 2) / 4 = 8.5` bit，相对 FP16 是 1.88 倍压缩，不是 2 倍。

稀疏要变成真实的加速，需要四个条件同时成立：

- **模式匹配硬件**。只有 2:4 和部分块稀疏有指令支持，任意稀疏没有。
- **负载均衡**。一个 tile 里非零太多，这个 tile 就慢，其余 SM 在等它。
- **维度足够大**。K 或 N 小于一个 tile 时，metadata 和边界处理的固定开销盖过收益。
- **瓶颈在计算**。decode 带宽受限，稀疏只减少权重字节，收益上限是压缩比；prefill 的 M 大、进入计算受限，2:4 才可能接近 2 倍。

SwiGLU 的输出不会产生精确的零，ReLU 时代“激活一半是零”的经验在 LLM 上不成立。

## 15.3 Knowledge Distillation

蒸馏用一个大模型（teacher）的输出训练一个小模型（student）。Hinton 的做法是给 softmax 加温度：

$$
p_i^{(T)} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

损失是硬标签和软标签的加权和：

$$
L = \alpha \, L_{\text{CE}}(y, p_s) + (1-\alpha)\, T^2 \, D_{\text{KL}}\left(p_t^{(T)} \,\|\, p_s^{(T)}\right)
$$

`T^2` 用来补偿温度对梯度的缩小，让软标签项和硬标签项的梯度量可比。

中间层蒸馏（FitNets）对齐 hidden state，维度不同时加一个可学习投影；LLM 上常作为 logit 蒸馏的辅助项。

LLM 蒸馏有三个量化方法没有的难点：

- **词表巨大**。KL 要在 32k–128k 的词表上算，每个 token 一次 softmax。
- **tokenizer 必须一致**。teacher 和 student 的词表不同，logits 无法逐位置对齐，只能退到序列级蒸馏。
- **容量差距**。student 太小，soft label 里的细粒度分布学不进去，反而不如直接用硬标签。

MiniLLM 指出 forward KL 的问题：它要求 student 覆盖 teacher 的所有模式，包括概率极低的尾部，小模型只能把概率质量摊薄。它改成 reverse KL：

$$
L_{\text{RKL}} = \sum_t D_{\text{KL}}\left(p_s(\cdot \mid y_{<t}) \,\|\, p_t(\cdot \mid y_{<t})\right)
$$

reverse KL 是 mode-seeking 的，student 集中拟合高概率区域；代价是需要从 student 采样，要用 policy gradient 优化。

GKD（on-policy distillation）针对 exposure bias：teacher 在真实数据上的分布和 student 自己生成时不一致。它让 student 先自己生成序列，再由 teacher 打 soft label，用 JSD 在 forward 和 reverse KL 之间插值。

| 方法 | 监督信号 | teacher 何时运行 | 主要成本 | 适合 |
| --- | --- | --- | --- | --- |
| Logit KD | 全词表 soft label | 离线或在线 | 词表 softmax、logits 存储 | 同 tokenizer、容量接近 |
| Feature KD | 中间层 hidden | 在线 | 投影层、对齐 | 同架构 |
| Sequence KD | teacher 生成的序列 | 离线 | 一次生成 | 跨 tokenizer |
| MiniLLM | reverse KL | 在线 | policy gradient | 小 student |
| GKD | student 自生成序列 | 在线 | 两模型同时驻留 | 生成任务 |

## 15.4 Teacher-Student Execution

蒸馏的执行方式直接决定显存和吞吐。

**离线蒸馏**先让 teacher 跑完整个数据集，把 soft label 落盘，再单独训练 student。训练时显存里只有 student。问题在存储：128k 词表、FP32 logits，一个 token 是 512 KB；10 亿 token 就是 512 TB，不可能全存。实际做法是只存 top-k（比如 top-50，每 token 200 字节），或者干脆不存 logits、只存 teacher 生成的文本。

**在线蒸馏**在每一步同时跑 teacher 和 student。teacher 只需要前向，不需要梯度和优化器状态，但权重必须常驻。一个 7B teacher 用 BF16 是 14 GB；7B student 全参数微调用 Adam，权重 14 GB + 梯度 14 GB + 两份 FP32 动量 56 GB = 84 GB，加上激活，单卡 80 GB 放不下。teacher 再进来就是 98 GB 以上。

所以在线蒸馏的常见配置是：teacher 放另一张卡，或者 student 用 LoRA，基座量化到 4-bit。

吞吐上，teacher 前向的 FLOPs 和 student 同量级，所以在线一步大约是纯 student 训练的 1.5–2 倍。离线把这个成本挪到数据生产阶段。

| | 离线蒸馏 | 在线蒸馏 |
| --- | --- | --- |
| 训练时显存 | student + batch | teacher + student + 优化器 |
| 存储 | top-k logits 或生成文本 | 无 |
| 每步成本 | 1 倍 | 1.5–2 倍 |
| 数据新鲜度 | 固定 | 可随 student 变化 |
| 适合 | 大规模、teacher 固定 | on-policy、质量优先 |

## 15.5 Low-rank Approximation

SVD 把权重分解成 `W = U \Sigma V^\top`，截断到前 `r` 个奇异值：

$$
W \approx U_r \Sigma_r V_r^\top, \quad U_r \in \mathbb{R}^{m \times r}, \; V_r \in \mathbb{R}^{n \times r}
$$

参数量从 `mn` 降到 `r(m+n)`，压缩比是：

$$
\rho = \frac{mn}{r(m+n)}
$$

方阵 `n \times n` 时 `\rho = n / (2r)`。4096×4096、`r = 256`，参数量从 16.8M 降到 2.1M，8 倍压缩。

直接对训练好的权重做截断 SVD，效果通常不好：LLM 权重的奇异值衰减很慢，能量分散，砍掉一半的秩输出误差已经明显。原因是最小化权重误差 `\|W - \hat{W}\|`，而模型关心输出误差 `\|WX - \hat{W}X\|`。

改进方向是激活感知：用校准集估计输入协方差，白化后再分解，等价于最小化输出误差。ASVD、SVD-LLM 属于这一类，中等压缩率下明显更好，但仍不如 4-bit 量化。

张量分解（CP、Tucker）理论上压缩率更高，实际实现少，多停留在研究阶段。

低秩和 LoRA 的关系容易混淆：

- **SVD 压缩**是对已有权重做分解，减少参数；
- **LoRA** 是给冻结的权重加一个低秩增量 `\Delta W = BA`，增加少量参数来适配任务。

两者数学形式相同，目的相反。LoRA 的增量可以在推理前合并回 `W`，没有额外延迟，但合并后失去低秩的存储优势。

激活也有低秩性，这一点在架构上更有价值。DeepSeek-V2 的 MLA 把每个 token 每层的 KV 压缩成 512 维的 latent，再加 64 维解耦的 RoPE 分量；相比标准 MHA 的 `2 \times n_h \times d`，KV Cache 小一个数量级以上。这不是事后压缩，而是训练时就按低秩结构设计。

| 方法 | 压缩对象 | 是否需要校准 | 典型压缩比 | 主要风险 |
| --- | --- | --- | --- | --- |
| 截断 SVD | 权重 | 否 | 2–4 倍 | 精度掉得快 |
| 激活感知 SVD | 权重 | 是 | 2–4 倍 | 需要校准数据 |
| 张量分解 | 权重 | 是 | 4 倍以上 | 实现复杂 |
| LoRA | 增量 | 否 | 不压缩 | 只适合适配 |
| MLA | KV 表示 | 训练时 | 10 倍以上 | 需改架构 |

## 15.6 Weight Sharing

权重共享用聚类代替逐个存储。对每层做 k-means：

$$
\min_{C} \sum_{ij} \left\| W_{ij} - C_{k(i,j)} \right\|^2
$$

`C` 是码本（codebook），`k(i,j)` 是权重所属的簇。Deep Compression 的三个细节值得记住：

- **初始化用线性分位**，不用随机。随机初始化会让大量中心落空。
- **逐层聚类**，不同层的权重量级差别很大，共享一个码本没有意义。
- **和剪枝、Huffman 编码叠加**。Deep Compression 在 AlexNet 上把 240 MB 压到 6.9 MB（35 倍），VGG-16 从 552 MB 压到 11.3 MB（49 倍），精度不降。

存储的账很简单：FP16 是 16 bit，32 个簇的索引是 5 bit，压缩 3.2 倍；16 个簇是 4 bit，压缩 4 倍。码本只有 64 字节，可以忽略。

| 簇数 | 索引位数 | 相对 FP16 压缩 | 是否字节对齐 |
| --- | --- | --- | --- |
| 16 | 4 | 4.0 倍 | 是 |
| 32 | 5 | 3.2 倍 | 否，需位打包 |
| 64 | 6 | 2.7 倍 | 否 |
| 256 | 8 | 2.0 倍 | 是 |

GPU 上的问题是查表。每个权重都要做一次 gather，Tensor Core 用不了这种操作，decode 时指令数会压过带宽收益。所以权重共享在 LLM 上更多是存储优化，不是延迟优化。真正被广泛使用的是**层共享**：ALBERT 让 12 层共用一组参数，large 版本比 BERT-large 少 18 倍参数；更常见的是 embedding 和 lm_head 共享（tied embeddings）。

## 15.7 Model Merging

模型合并把多个微调模型的差异加起来，不需要联合训练。前提是它们来自同一个基座。

Task arithmetic 定义任务向量：

$$
\tau_i = \theta_i - \theta_0, \qquad \theta_{\text{merged}} = \theta_0 + \lambda \sum_i \tau_i
$$

`\lambda` 通常取 0.5–1.0。任务不冲突时线性叠加有效，冲突时会互相抵消。

TIES 针对冲突做了三步处理：

1. **Trim**：每个任务向量只保留绝对值最大的 20%，其余置零；
2. **Elect sign**：对每个参数位置，按正负两侧的幅值总和投票，选出主导符号；
3. **Disjoint merge**：只合并符号一致的参数，取平均。

DARE 随机丢掉 90%–99% 的 delta，再把剩下的按 `1/(1-p)` 放大；期望不变，方差变大但方向保留，参数冲突显著减少。

SLERP 只用于两个模型，在参数空间做球面插值：

$$
\text{slerp}(p_0, p_1; t) = \frac{\sin((1-t)\Omega)}{\sin\Omega} p_0 + \frac{\sin(t\Omega)}{\sin\Omega} p_1, \quad \cos\Omega = \frac{p_0 \cdot p_1}{\|p_0\|\|p_1\|}
$$

它保持路径上的范数，常用来融合 base 和 chat。

失败模式很明确：

- **基座不同**。两个模型的权重不在同一个损失面的同一个邻域，相加没有意义，输出会退化。
- **tokenizer 不同**。embedding 行对不上，合并直接错位。
- **任务冲突**。数学和代码的微调方向可能相反，TIES 只能缓解，不能消除。
- **只在小范围有效**。线性性只在基座附近成立，微调偏离太远时失效。

| 方法 | 模型数 | 是否需要基座 | 关键参数 | 适合 |
| --- | --- | --- | --- | --- |
| Task arithmetic | 多个 | 是 | `\lambda` | 同源微调 |
| TIES | 多个 | 是 | trim 比例 20% | 任务冲突明显 |
| DARE | 2 个或多个 | 是 | drop 率 90%–99% | 减少干扰 |
| SLERP | 2 个 | 否 | `t` | base + chat 融合 |

## 15.8 LoRA and Parameter-efficient Adaptation

LoRA 冻结原权重，只训练一个低秩增量：

$$
W' = W_0 + \frac{\alpha}{r} BA, \quad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k}
$$

`A` 用 Kaiming 初始化，`B` 初始化为 0，所以训练开始时增量恰好是 0，模型行为和基座完全一致。可训练参数是 `r(d+k)`。`d = k = 4096`、`r = 16` 时是 131K，占原矩阵 16.8M 的 0.78%。

`\alpha / r` 是缩放因子，让改变 `r` 时不必重调学习率：`r` 翻倍，`BA` 的量级大致翻倍，除以 `r` 后回到同一尺度。常见配置是 `r = 16, \alpha = 16`（缩放 1）或 `r = 8, \alpha = 16`（缩放 2）。`\alpha` 固定而 `r` 变化时，有效学习率会漂移，这是复现实验时最常见的坑。

QLoRA 在 LoRA 上叠了三项技术，让 65B 模型能在单张 48 GB 卡上微调：

- **NF4**。4-bit NormalFloat 的 16 个量化点取标准正态分布的分位点，对正态分布的权重是信息论意义上最优的。块大小 64，每块一个 absmax 缩放。
- **Double quantization**。absmax 本身也要存，FP32 下是每参数 `32/64 = 0.5` bit。把 absmax 再按 256 个一块量化成 8-bit，降到 `8/64 + 32/(64 \times 256) ≈ 0.127` bit，省下 0.373 bit/参数。
- **Paged optimizer**。优化器状态放在统一内存里，显存峰值时换页到 CPU，避免长序列 OOM。

计算时 NF4 权重先反量化到 BF16 再进 GEMM，所以省的是显存和带宽，不是算力。QLoRA 的 Guanaco 65B 在 Vicuna benchmark 上达到 ChatGPT 的 99.3%，说明 4-bit 基座加 LoRA 已接近全参数微调。

| 方法 | 可训练参数 | 推理额外延迟 | 显存（7B） | 备注 |
| --- | --- | --- | --- | --- |
| Full fine-tune | 100% | 0 | 84 GB+ | 需要多卡 |
| LoRA | 0.1%–1% | 合并后为 0 | 基座 + 少量 | 最常用 |
| QLoRA | 0.1%–1% | 合并后为 0 | 约 4 GB 基座 | 单卡 24 GB 可跑 |
| Adapter | 1%–5% | 有，串行插入 | 基座 + 适配器 | 延迟明显 |
| Prefix tuning | 0.01% | 占用上下文 | 基座 + 少量 | 挤占 KV |

## 15.9 Compression Trade-offs

把这一章的方法放在一张表里，按“省的是什么”分类：

| 方法 | 典型压缩比 | 精度损失 | kernel 复杂度 | 训练成本 | 适用场景 |
| --- | --- | --- | --- | --- | --- |
| INT8 W8A8 | 2 倍 | 很小 | 中 | 校准 | 服务端推理 |
| INT4 W4A16 | 3.5–4 倍 | 小到中 | 高 | 校准 | 单卡部署 |
| 2:4 稀疏 | 1.88 倍权重 | 中 | 中 | 剪枝 + 可选恢复 | 计算受限的 prefill |
| 非结构化 50% | 可能为负 | 低到中 | 无加速 | 低 | 研究、存储 |
| 蒸馏 | 不压缩，换小模型 | 取决于容量差 | 无 | 高 | 端侧、专用模型 |
| 低秩 SVD | 2–4 倍 | 中到高 | 低 | 低 | 与量化叠加 |
| 权重共享 | 3–4 倍 | 中 | 高（gather） | 中 | 存储受限 |
| LoRA / QLoRA | 不压缩 | 接近无损 | 低 | 低 | 适配、微调 |
| 模型合并 | 不压缩 | 任务相关 | 无 | 无 | 多任务整合 |

几条实践结论：

- **显存受限优先量化**。4-bit 权重量化在同样的精度损失下压缩比最高，kernel 也最成熟。
- **计算受限才考虑 2:4**。prefill 占比高的服务，2:4 能带来接近 2 倍的算力；decode 为主的场景，稀疏的收益上限就是权重字节的减少。
- **蒸馏和量化不冲突**。先蒸馏出小模型，再量化，是端侧部署的常见组合。
- **合并和 LoRA 是训练侧工具**，不减少推理成本，除非把 LoRA 合并回权重。
- **低秩和权重共享的 kernel 支持差**，要按端到端时间评估，不看压缩比。

## 本章小结

- 非结构化稀疏减少存储但不加速，稀疏要生效，模式必须匹配硬件。
- Wanda 用 `|W| \cdot \|X\|` 打分；SparseGPT 用逐层重建和 Hessian 补偿，能一次剪到 50%–60%。
- 2:4 权重压到 8.5 bit，算力翻倍，但只在计算受限时兑现。
- 蒸馏的 soft label 需要温度 `T` 和 `T^2` 缩放；MiniLLM 用 reverse KL，GKD 用 on-policy 数据。
- 在线蒸馏的显存是 teacher 加 student 加优化器状态，通常要把 teacher 放到另一张卡。
- 截断 SVD 直接压缩 LLM 权重效果有限，激活感知的变体更好；MLA 说明低秩最好在设计时引入。
- 权重共享能到 3–4 倍压缩，瓶颈是 GPU 上的 gather；层共享和 embedding tying 更实用。
- 模型合并只在同基座、同 tokenizer 的模型之间有效；TIES 处理符号冲突，DARE 降低 delta 密度。
- LoRA 的 `\alpha/r` 决定有效缩放；QLoRA 用 NF4、double quant 和 paged optimizer 把 65B 放进 48 GB。
- 压缩比的排序不等于加速比的排序，评估以端到端时间和质量为准。

## 延伸阅读

- Han et al., [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626), 2015.
- Han et al., [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149), 2016.
- Frantar & Alistarh, [SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774), 2023.
- Sun et al., [A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695), 2023.
- Hinton et al., [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), 2015.
- Romero et al., [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550), 2015.
- Gu et al., [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543), 2023.
- Agarwal et al., [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649), 2024.
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), 2021.
- Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314), 2023.
- Ilharco et al., [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089), 2022.
- Yadav et al., [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708), 2023.
- Yu et al., [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://arxiv.org/abs/2311.03099), 2023.
- DeepSeek-AI, [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434), 2024.
