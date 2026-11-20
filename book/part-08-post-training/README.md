# Part VIII — Post-training and Model Quality from a Systems Perspective

**模型质量篇 · 约 55 页**

目标：让系统工程师理解 Post-training 的数据、算法和计算流程，以及如何构建高效训练基础设施。

## 本篇章节

| 章 | 标题 | 级别 | 状态 |
| --- | --- | --- | --- |
| 41 | [Data Engineering for LLMs](ch41-data-engineering-for-llms.md) | TBD | draft |
| 42 | [Supervised Fine-tuning](ch42-supervised-fine-tuning.md) | TBD | draft |
| 43 | [Preference Optimization and RL](ch43-preference-optimization-and-rl.md) | TBD | draft |
| 44 | [Evaluation and Quality Measurement](ch44-evaluation-and-quality-measurement.md) | TBD | draft |

## 本篇导言

预训练决定模型的上限，后训练决定模型能用。而“能用”这件事，在整个系统里是成本最低、改动力度最大的一环：一次 SFT 的算力比预训练小三个数量级，几万元就能跑完；一次 DPO 在单机八卡上就能做；但换一批数据、改一个模板，效果差异往往超过在推理侧做的任何优化。

本篇从系统视角看后训练：数据怎么来、怎么清、怎么混、怎么去重；SFT 的 loss 怎么 mask、序列怎么打包、chat template 为什么错一个换行就出事；偏好优化的目标函数对应多少显存和多少 GPU 时间；RL 训练为什么大半时间花在 rollout 上；以及最后，怎么衡量这些改动到底带来了多少质量提升。

本篇不写强化学习教材，也不铺开讲对齐理论的争论。它关心的是资源画像和工程约束：一次后训练要多少 GPU 小时，显存账单是几份模型，rollout 和训练的吞吐差多少，评测的置信区间有多宽。

## 学习目标

- 能设计数据清洗、去重和混合流水线，估算各阶段的存储与计算开销，并检测评测污染。
- 能写出正确的 SFT loss mask 和序列打包逻辑，解释 chat template 不一致导致的线上事故。
- 能推导 DPO、PPO、GRPO 的内存账单，判断给定模型规模下需要几张卡、是否需要分离式 rollout。
- 能估算 RL 训练中 rollout 的占比，设计权重同步和 off-policy 的折中方案。
- 能设计可信的评测方案，计算置信区间和配对检验，识别人工评估与 LLM-as-a-judge 的系统性偏差。
- 能把质量指标和延迟、吞吐、成本放在同一张表里，评估量化与蒸馏的真实代价。

## 阅读建议

- Chapter 41 是后续三章的数据基础，MinHash 去重和混合比例的部分建议动手算一遍规模。
- Chapter 42 最贴近工程实践，loss mask 和 packing 的代码可以直接对照自己项目的实现检查。
- Chapter 43 的公式不必逐项推导，重点在 43.5 的显存账单和 43.9 的 rollout 架构，它们与 Part V、Part VI 的推理系统直接相连。
- Chapter 44 建议在读完前面三章后再读，它把“改了什么”和“好了多少”连起来；44.6 的统计部分值得反复看，大多数评测报告的结论都撑不住置信区间。
