# Appendix A. Linear Algebra Refresher

本附录只收录本书反复用到的线性代数与张量运算约定，不替代教材。

## A.1 矩阵乘法的形状与计算量

对 `A: [M, K]` 与 `B: [K, N]`：

$$
C_{ij} = \sum_{k=0}^{K-1} A_{ik} B_{kj}, \quad C \in \mathbb{R}^{M \times N}
$$

| 量 | 表达式 |
| --- | --- |
| 输出元素数 | `M × N` |
| 每个元素的乘加次数 | `K` |
| 乘加总数（MACs） | `M × N × K` |
| FLOPs（1 MAC = 2 FLOPs） | `2 × M × N × K` |
| 读入字节 | `(M×K + K×N) × sizeof(dtype)` |
| 算术强度 | `2MNK / ((MK + KN) · b)` |

算术强度在 `M, N ≫ K` 时接近 `2/b`（`b` 为字节数），在 `M = 1` 时退化到 `2N/(N + K)` 量级。这就是 decode 的 `M = 1` GEMM 属于访存受限的原因。

## A.2 批矩阵乘与分组矩阵乘

**批矩阵乘**（batched GEMM）：`A: [B, M, K]`、`B: [B, K, N]`，输出 `[B, M, N]`，FLOPs 为 `2BMNK`。每个 batch 独立，GPU 上映射为不同的 block。

**分组矩阵乘**（grouped GEMM）：多个不同形状的 GEMM 合并成一次调用，每个组有自己的 `M_i, N_i, K_i`。MoE 的专家计算就属于这一类：`E` 个专家的 token 数不同，合并后一次 kernel 处理，避免 `E` 次 launch。

## A.3 张量的内存布局

PyTorch 的默认布局是行优先（row-major），一个 `[M, N]` 张量的 stride 是 `(N, 1)`。对 `[B, S, H]`，stride 是 `(S·H, H, 1)`。

转置不移动数据，只交换 stride：`A.T` 的 stride 从 `(N, 1)` 变成 `(1, M)`。这看起来免费，但会让后续的访存变成跨步访问，带宽利用率大幅下降。所以 `mm(A.T, B)` 往往比 `mm(A, B)` 慢，除非 kernel 内部做了转置加载。

几种常见布局：

| 布局 | 形状顺序 | 使用场景 |
| --- | --- | --- |
| 行优先 | `[M, K]` | 通用 |
| 列优先 | `[K, M]` 存 `Aᵀ` | cuBLAS 的默认习惯 |
| NHWC | `[N, H, W, C]` | 卷积、Tensor Core 友好 |
| 分块布局 | `[M/tm, K/tk, tm, tk]` | shared memory 中的 tile |

## A.4 广播与归约

广播（broadcasting）在 stride 为 0 的维度上复制。`[B, 1, H] + [1, S, H]` 会广播成 `[B, S, H]`，计算量按输出算，但读入只有两个小张量。融合 kernel 时要注意，广播张量不应该被物化。

归约（reduction）沿某个维度求和：

```python
x.sum(dim=-1)      # [B, S, H] -> [B, S]
```

归约的代价取决于归约维度的长度：短归约适合一个线程处理，长归约需要跨线程或跨 block，涉及 shared memory 和多次同步。

## A.5 einsum 与注意力

注意力可以用 einsum 表达：

```python
scores = torch.einsum("bhid,bhjd->bhij", q, k) * scale
attn   = torch.softmax(scores, dim=-1)
out    = torch.einsum("bhij,bhjd->bhid", attn, v)
```

但这样会物化 `[B, H, S, S]` 的分数矩阵。`S = 8192`、`H = 32`、FP16 时是 `4.3 GB`，远超可接受范围。FlashAttention 的做法是在 tile 内做 softmax，用在线归一化避免物化整行，这也是理解 attention kernel 的起点。

## A.6 常用近似

| 场景 | 近似 |
| --- | --- |
| 参数量 | `12 × L × h²`（MLP 两个矩阵、`h_ff = 4h`，或 SwiGLU 的 `h_ff = 2.67h`） |
| GQA + SwiGLU 的参数量 | `L × (2h² + 2h·h_kv + 3h·h_ff)` |
| 前向 FLOPs | `2 × N × tokens` |
| 反向 FLOPs | `4 × N × tokens` |
| attention FLOPs（因果） | `2 × S² × h × L` |

最后一项的系数：每个 token 对约 `4h` FLOPs（QKᵀ 与 AV 各 `2h`），因果掩码下平均一半，再乘 `L` 层。
