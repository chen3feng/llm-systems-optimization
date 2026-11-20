---
chapter: 42
part: 8
title: Supervised Fine-Tuning
level: tbd
status: draft
prerequisites: [37, 41]
lab:
---

# Chapter 42. Supervised Fine-Tuning

一个预训练模型能续写网页，却不会回答问题。让它学会“按指令输出”的，通常是最多几万到几百万条的高质量对话数据，加上一次监督微调（supervised fine-tuning，SFT）。

SFT 的计算量比预训练小三个数量级，但工程上的坑一点不少：数据格式不对，模型学不会角色；loss 没有 mask，模型学会复述问题；chat template 和推理时不一致，线上效果直接塌掉。这一章讲 SFT 的目标、数据组织、mask 与打包，以及全参数微调和 LoRA 的成本。

## 42.1 SFT Objectives

SFT 的损失和预训练一样，是下一个 token 的交叉熵：

$$
\mathcal{L} = -\sum_{t \in \text{target}} \log p_\theta(x_t \mid x_{<t})
$$

区别在求和范围：只对“回答”部分的 token 计算 loss，问题和系统提示不参与。

为什么？如果对全部 token 计 loss，模型会把大量容量花在“复述用户问题”上。推理时它确实需要理解问题，但不需要被奖励去预测问题本身——尤其是在多轮对话里，历史消息会越来越长，loss 会被这些内容主导。

SFT 的数据量通常远小于预训练。经验值是每个任务几千到几万条高质量样本；把模型“教成”某种风格，几万条足够；要覆盖广泛能力，需要几十万到上百万条。

学习率也要小得多。预训练用 `1e-4` 到 `3e-4`，SFT 通常用 `1e-5` 到 `5e-5`，轮数 2–3 轮。过大或过多会让模型遗忘预训练知识（灾难性遗忘），表现为通用能力下降。

## 42.2 Instruction Data Format

一条 SFT 样本是“输入 + 期望输出”。常见的组织方式是消息列表：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个严谨的代码助手。"},
    {"role": "user", "content": "解释一下 PagedAttention。"},
    {"role": "assistant", "content": "PagedAttention 把 KV Cache ..."}
  ]
}
```

数据质量比数量重要得多。常见的质量问题：

- **答案错误**：模型会忠实学会错误；
- **风格不一致**：同一批数据里有的用 markdown、有的用纯文本，模型输出会摇摆；
- **过度冗长**：为了“显得详细”而堆砌，模型会学会啰嗦；
- **模板痕迹**：数据里残留“以下是回答：”这类前缀，模型会在输出里重复。

过滤和重写数据往往比增加数据更有效。一个常见的做法是用强模型对弱模型生成的数据打分，只保留高分样本，或者直接用强模型重写。

## 42.3 Chat Templates

模型不认识“角色”，它只看到 token。chat template 把消息列表渲染成带特殊 token 的序列：

```text
<|im_start|>system
你是一个严谨的代码助手。<|im_end|>
<|im_start|>user
解释一下 PagedAttention。<|im_end|>
<|im_start|>assistant
PagedAttention 把 KV Cache ...<|im_end|>
```

模板的关键约束：

- **训练和推理必须一致**：同一套特殊 token、同样的换行和空格。不一致时，模型看到的分布变了，输出质量会明显下降；
- **结束符要明确**：模型要学会在 `<|im_end|>` 处停止，否则会一直生成；
- **系统提示可选**：没有 system 消息时，模板不能凭空插入或留下空段。

模板 bug 是最常见的线上事故之一。训练用 Jinja 模板，推理用另一套拼接逻辑，两者差一个换行，模型就开始输出奇怪的内容。解决办法是训练和推理共用同一个模板文件，并在 CI 里做渲染快照测试。

## 42.4 Loss Masking

有了模板，还要决定哪些 token 计入 loss。标准做法是只对 assistant 部分计算：

```python
labels = input_ids.clone()
labels[:prompt_len] = -100        # PyTorch 的 ignore_index
loss = cross_entropy(logits, labels, ignore_index=-100)
```

多轮对话时，每一轮的 user 部分都要 mask，只保留各轮 assistant 的输出。

一个细节是边界 token。`<|im_start|>assistant` 这个前缀算 prompt 还是回答？如果算 prompt，模型学不会在正确的位置开始；如果算回答，loss 会奖励它生成模板。多数实现把 assistant 的起始标记算作 prompt 的结尾，从第一个内容 token 开始计 loss。

另一个细节是 padding。batch 内序列长度不同，padding 位置必须 mask，否则 loss 会被大量无意义的 pad token 拉低，梯度也会被污染。

## 42.5 Sequence Packing

SFT 样本长度差异很大：短的几十个 token，长的几千。如果按 batch 内最长序列 padding，利用率可能只有 30%。

打包（packing）把所有样本拼成定长序列，用 attention mask 或位置重置隔开：

```text
[样本 A ...][样本 B ...][样本 C ...]  → 一条 8192 的序列
```

两种隔断方式：

- **块对角 attention mask**：每个样本只能看到自己那段，数学上等价于单独训练；
- **重置 position id**：让每个样本的位置编码从 0 开始，否则后面的样本会“看到”前面的位置。

不隔断也能训，但样本 B 会看到样本 A 的内容，相当于引入了跨样本上下文，多数情况下有害。

打包能把吞吐提高 2–3 倍，代价是实现复杂：mask 要额外构造，packing 边界要和 loss mask 对齐，否则会把 padding 或上一个样本的内容算进 loss。

## 42.6 Fine-tuning Memory and Compute Costs

SFT 的显存和预训练同构：参数、梯度、优化器状态、激活。7B 全参数微调仍需约 112 GB 的前三项，单卡放不下。

| 方式 | 7B 显存（约） | 硬件 | 适用 |
| --- | --- | --- | --- |
| 全参数 + ZeRO-3 | 20 GB/卡（N=8） | 8×80 GB | 数据多、要改全部能力 |
| 全参数 + 重计算 | 30 GB/卡（N=8） | 8×80 GB | 折中 |
| LoRA（r=16） | 25–40 GB | 单卡 80 GB | 大多数微调场景 |
| QLoRA（4-bit + LoRA） | 10–16 GB | 单卡 24–48 GB | 显存受限 |

计算量可以估：7B、10 亿 token 的 SFT，`6 × 7e9 × 1e9 = 4.2e19 FLOPs`；8×H100、40% MFU 下约 3.7 小时。实际 SFT 的数据量通常更小（几亿 token），几十分钟到几小时就能跑完。

正因为成本低，SFT 的实验迭代很快，数据质量的作用被放大：同样的算力，换一批更好的数据，效果提升往往超过调参。

## 42.7 LoRA-based Fine-tuning

LoRA 冻结原权重，只训练低秩增量：

$$
W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k}
$$

`r` 通常取 8–64，可训练参数只有原模型的 0.1%–1%。

优势很直接：

- **优化器状态按增量算**：7B 模型的 LoRA 参数可能只有几百万，Adam 状态可以忽略；
- **权重可共享**：一份基座模型可以挂多个 adapter，服务多租户；
- **不遗忘**：冻结原权重，通用能力保持得更好；
- **可合并**：推理时把 `BA` 加回 `W`，没有额外延迟。

代价是表达能力受限。LoRA 适合“改变风格、注入领域知识、适配格式”，不适合“大幅改变模型能力”。任务与预训练分布差得远时，LoRA 的效果明显不如全参数微调。

超参上，`alpha` 通常取 `2r`，学习率比全参数微调高一个量级（`1e-4` 到 `3e-4`），作用在哪些模块上也有讲究：只加在 `q`、`v` 上是最省参数的配置，加在全部线性层上效果更好但参数更多。

QLoRA 在 LoRA 基础上把基座量化到 4-bit，进一步降低显存，代价是训练速度略慢（反量化开销）和少量精度损失。

## 本章小结

- SFT 用下一个 token 的交叉熵，只对 assistant 输出计 loss；对全部 token 计 loss 会让模型学会复述问题。
- 数据质量比数量重要；答案错误、风格不一致、模板痕迹都会直接被模型学走。
- chat template 训练与推理必须完全一致，结束符要明确；模板 bug 是最常见的线上事故之一。
- Loss masking 要覆盖多轮对话的每个 user 段和 padding 位置，assistant 起始标记的归属要统一。
- 打包把利用率从 30% 提到 90% 以上，但需要块对角 mask 或位置重置，否则样本之间会互相污染。
- 7B 全参数微调仍需 112 GB 的前三项，LoRA 把可训练参数降到 0.1%–1%，单卡即可。
- LoRA 适合风格与领域适配，不适合大幅改变能力；QLoRA 进一步把基座压到 4-bit。
- SFT 的计算成本很低，迭代快，数据质量的作用被放大。

## 延伸阅读

- Ouyang et al., [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155), NeurIPS 2022.
- Wei et al., [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652), ICLR 2022.
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), ICLR 2022.
- Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314), NeurIPS 2023.
- Zhou et al., [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206), NeurIPS 2023.
- Taori et al., [Stanford Alpaca: An Instruction-following LLaMA Model](https://github.com/tatsu-lab/stanford_alpaca), 2023.
- Hugging Face, [Chat Templates](https://huggingface.co/docs/transformers/chat_templating).
