# Part III — Numerical Optimization and Model Compression

**数值与压缩篇 · 约 65 页**

目标：理解模型参数和中间数据的数值表示，以及如何降低模型的存储和计算成本。

## 本篇章节

| 章 | 标题 | 级别 | 状态 |
| --- | --- | --- | --- |
| 12 | [Numerical Formats for LLMs](ch12-numerical-formats-for-llms.md) | TBD | draft |
| 13 | [Quantization Fundamentals](ch13-quantization-fundamentals.md) | TBD | draft |
| 14 | [Practical LLM Quantization](ch14-practical-llm-quantization.md) | TBD | draft |
| 15 | [Compression Beyond Quantization](ch15-compression-beyond-quantization.md) | TBD | draft |
| 16 | [Quantized and Compressed Runtime](ch16-quantized-and-compressed-runtime.md) | TBD | draft |

## 本篇导言

70B 模型 FP16 是 140 GB，FP8 是 70 GB，INT4 是 35 GB。数值格式和量化算法决定了模型能不能装进一张卡、decode 能跑多快、精度会掉多少。

本篇从浮点数的位布局讲起，经过量化的数学基础和主流算法，最后落到量化 kernel 的实现：反量化怎么融合进 GEMM、scale 怎么存、低精度累加怎么处理。

## 学习目标

- 能解释 FP16、BF16、FP8、FP4 和 INT4 的范围、精度与适用场景。
- 能推导仿射量化的 scale 和 zero point，并实现 per-channel / group-wise 量化。
- 能分析量化误差、离群值和累加精度对模型质量的影响。
- 能比较 GPTQ、AWQ、GGUF、SmoothQuant 等方法的取舍。
- 能在 CUDA kernel 中融合反量化，避免额外的 HBM 往返。

## 阅读建议

- Chapter 12 是后续所有内容的基础，建议先读。
- Chapter 13 和 Chapter 14 连续阅读：前者讲原理，后者讲工程算法。
- Chapter 15 可以在需要进一步压缩时查阅。
- Chapter 16 面向 runtime 和 kernel 开发者，建议对照源码阅读。
