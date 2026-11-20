# 符号与术语约定

全书尽量使用统一的符号。同一符号在不同章节有特殊含义时，会在该章重新定义。

## 数学符号

| 符号 | 含义 |
| --- | --- |
| $B$ | Batch size |
| $S$ | Sequence length（上下文长度） |
| $H$ 或 $h$ | Hidden dimension |
| $L$ | Transformer 层数 |
| $V$ | 词表大小 |
| $n_h$ | Query 头数 |
| $n_{kv}$ | KV 头数（GQA 下小于 $n_h$） |
| $d_{head}$ | 每个头的维度，通常 $n_h \cdot d_{head} = H$ |
| $h_{ff}$ | MLP 中间维度 |
| $N$ | 参数量；在分布式语境下也表示 rank 数（会明确标注） |
| $T$ | 生成的 token 数 |
| $\alpha$ | 投机解码的接受率 |
| $\gamma$ | 投机解码的草稿长度 |
| $\beta$ | DPO 中偏离参考策略的强度 |

## 性能量

| 符号/单位 | 含义 |
| --- | --- |
| FLOPs | 浮点运算次数（复数形式表示总量） |
| MACs | 乘加运算次数，1 MAC = 2 FLOPs |
| FLOPS | 每秒浮点运算次数（硬件能力） |
| MFU | 模型浮点利用率，实际 FLOPs/s ÷ 峰值 FLOPS |
| GB / GiB | 十进制 / 二进制容量，1 GiB = 1.074 GB |
| KiB | 1024 字节，用于 KV 块等小粒度 |
| tok/s | 每秒生成的 token 数 |
| TTFT | Time to first token，首 token 延迟 |
| TPOT | Time per output token，每输出 token 时间 |

## 常用规模

书中反复出现的参照配置：

| 配置 | 参数 |
| --- | --- |
| 70B 模型 | $L=80$，$H=8192$，$n_h=64$，$n_{kv}=8$，$d_{head}=128$，FP16 权重 140 GB |
| 每 token KV | 320 KiB（FP16） |
| H100 | BF16 989 TFLOPS，HBM 3.35 TB/s，80 GB |
| 8×H100 | 聚合 BF16 7.9 PFLOPS，HBM 640 GB |
| 单卡 decode | 70B 约 42 ms/token（访存受限估算） |
| 8K prefill | 线性层 1.15 PFLOP，attention 88 TFLOP |

## 术语

| 英文 | 中文 | 说明 |
| --- | --- | --- |
| Prefill | 预填充 | 处理输入 prompt 的阶段 |
| Decode | 解码 | 逐 token 生成的阶段 |
| KV Cache | KV 缓存 | 缓存历史 Key/Value，避免重复计算 |
| Attention | 注意力 | 按相似度加权聚合 value |
| Roofline | 屋顶线模型 | 用算术强度判断计算或访存受限 |
| Kernel | 核函数 | GPU 上执行的并行函数 |
| Warp | 线程束 | 32 个同步执行的线程 |
| Collective | 集合通信 | 多个 rank 参与的通信操作 |
| SLO | 服务等级目标 | 可度量的服务承诺 |

完整术语表见 [附录 H](../99-appendices/appendix-h-glossary.md)。
