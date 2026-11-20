---
chapter: 43
part: 8
title: Preference Optimization and RL
level: tbd
status: draft
prerequisites: [42]
lab:
---

# Chapter 43. Preference Optimization and RL

SFT 教会模型“怎么回答”，但回答有好有坏。让模型偏向人类更喜欢的那个，是偏好优化要解决的问题。

一条路径是训练奖励模型，再用强化学习优化策略（PPO）；另一条是直接在偏好数据上优化策略（DPO）。前者灵活但工程复杂，后者简单但受限于离线数据。这一章讲两类方法的目标函数、显存和 rollout 成本，以及奖励作弊这个绕不开的问题。

## 43.1 Preference Data

偏好数据的单位是“同一个 prompt 下的两个回答，标注者更喜欢哪一个”：

```json
{
  "prompt": "解释一下 KV Cache。",
  "chosen": "KV Cache 缓存历史 token 的 K 和 V ...",
  "rejected": "KV Cache 就是缓存，让模型变快。"
}
```

数据的来源有三种：

- **人工标注**：质量最高，成本也最高，一条几元到几十元；
- **AI 反馈**（RLAIF）：用强模型判断哪个回答更好，成本低一个数量级；
- **可验证信号**：数学题对答案、代码跑测试，不依赖判断，最可靠但覆盖面窄。

标注的一致性很重要。同一个 prompt，不同标注者的偏好可能相反；模型学到的是“平均偏好”，如果标注噪声大，奖励信号就会被稀释。常见做法是多人标注加一致性过滤，或者用强模型做仲裁。

## 43.2 Reward Modeling

奖励模型把“回答好不好”变成一个标量。训练数据是偏好对，损失是 Bradley-Terry 模型的最大似然：

$$
P(y_w \succ y_l \mid x) = \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)
$$

$$
\mathcal{L}_R = -\log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)
$$

实现上，奖励模型通常是策略模型加一个标量头，输入完整的 prompt 加回答，取最后一个 token 的隐状态过一个线性层。

几个容易出错的细节：

- **只在回答末尾打分**：不能让 prompt 或 padding 影响分数；
- **奖励模型和策略的分布要接近**：奖励模型在 SFT 模型的输出上训练，策略偏离太远后，它给出的分数就不可信；
- **校准**：奖励分数的绝对值没有意义，只有相对大小有意义；PPO 里用它算优势时要做归一化。

奖励模型的准确率通常用“偏好对上的正确率”衡量，70%–75% 就不错了；再高往往意味着过拟合了标注噪声，而不是真的更懂人类偏好。

## 43.3 Bradley-Terry Model

Bradley-Terry 假设每个回答有一个潜在的强度 `r`，比较结果的概率由强度差决定：

$$
P(i \succ j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} = \sigma(r_i - r_j)
$$

这个假设有局限：它要求偏好可传递（A 优于 B、B 优于 C，则 A 优于 C），而人类偏好经常不传递；它也表达不了平局。实践中这些偏差可以接受，因为奖励模型只需要在统计上区分好坏，不需要还原完整的偏好结构。

BT 模型和 Plackett-Luce 同族：后者把两两比较扩展到多选一排序，每次标注能提供更多信息，但标注成本和数据收集难度都更高。

## 43.4 DPO

直接偏好优化（direct preference optimization，DPO）绕开奖励模型，直接在偏好对上优化策略：

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\left(
\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}
- \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
\right)
$$

推导的起点是“最优策略对应一个隐式奖励”：把 BT 模型里的奖励用策略与参考策略的对数比表示，代回损失就得到上式。`β` 控制偏离参考策略的强度，通常取 0.1–0.5：太小约束不够，策略会跑远；太大偏保守，学不到偏好。

DPO 的工程优势很明显：

- 只需要策略模型和参考模型（参考模型只做前向），不需要奖励模型和 critic；
- 训练是监督式的，稳定、好调，和 SFT 的流程几乎一样；
- 显存比 PPO 低一半以上。

局限在于离线：训练数据是固定的偏好对，策略更新后产生的分布偏移没有被纠正。DPO 容易过度优化——把 chosen 的概率推得很高、rejected 推得很低，在训练集上“赢麻了”，但泛化性能下降。缓解手段包括降低 `β`、混入一定比例的 SFT loss、或者迭代式地重新采样数据（online DPO / iterative DPO）。

对系统工程师来说，DPO 的成本画像值得记一下：7B 模型的 DPO 训练，两份模型各 14 GB，加上激活和优化器状态，单卡 80 GB 可以跑（配合 LoRA 更宽松）；同规模的 PPO 至少要 4 份模型加两个优化器，单卡绝无可能。

## 43.5 PPO

近端策略优化（proximal policy optimization，PPO）是 on-policy 的：策略生成回答，奖励模型打分，用优势估计更新策略，并加 KL 惩罚防止偏离参考模型。

目标函数：

$$
\mathcal{L}_{\text{PPO}} = \mathbb{E}_t\left[\min\left(\rho_t A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t\right)\right] - \beta\, \text{KL}\left(\pi_\theta \| \pi_{\text{ref}}\right)
$$

其中重要性比 `ρ_t = π_θ(a_t|s_t) / π_old(a_t|s_t)`，`A_t` 是优势；`clip` 把单步更新的幅度限制在 `1 ± ε`（`ε` 通常 0.2），避免一次更新把策略推离采样分布太远。

PPO 的显存账单很重：同时需要策略、参考、奖励、critic 四个模型。7B 规模下，四份 BF16 权重就是 56 GB；再加上策略和 critic 的优化器状态（按 Adam 的 12 字节/参数算，各 84 GB），总量超过 220 GB，单机放不下，必须用 ZeRO 分片，或者把奖励模型和 critic 换成 1B 级别的小模型。

生成（rollout）是另一个瓶颈，而且它不在上面的整数里。每条样本要自回归生成几百到几千个 token；70B 模型 decode 一个 token 约 42 ms，1,000 token 就是 42 秒。batch 256 条即使完全并行，也要几分钟才能凑够一批训练数据，而一步策略更新只要几百毫秒。PPO 的时间几乎全花在生成上。

## 43.6 GRPO

分组相对策略优化（group relative policy optimization，GRPO）去掉了 critic：对同一个 prompt 采样 `G` 个回答，用组内奖励的均值和标准差归一化得到优势：

$$
A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}
$$

这样省掉了价值网络及其优化器状态，显存和实现复杂度都下降。代价是每个 prompt 要多采样几条（`G` 通常 4–16），生成成本上升。7B 模型、`G=8`、每 prompt 1,000 token，样本生成量是单条采样的 8 倍——省下的显存换成了 GPU 时间。

GRPO 特别适合奖励可验证的任务：一组回答里有的对有的错，组内比较就能给出清晰信号，不需要学一个价值函数。如果一组回答的奖励全部相同（全对或全错），标准差为零，这一组就不产生梯度——这也是 GRPO 里“难度适中的题目才有训练价值”这一说法的由来。

## 43.7 RLVR

可验证奖励强化学习（RL with verifiable rewards，RLVR）用程序判定对错，而不是训练奖励模型：

| 任务 | 验证方式 | 可靠性 |
| --- | --- | --- |
| 数学 | 答案精确匹配或符号等价 | 高，但要处理格式差异 |
| 代码 | 跑单元测试 | 高，覆盖率不足时漏判 |
| 指令遵循 | 规则检查（长度、格式、关键词） | 中，容易被绕过 |
| 工具调用 | 检查调用序列和返回结果 | 中高 |

RLVR 的优势是奖励信号可靠、难以作弊（相对而言），并且可以大规模自动生成，不需要标注人力。DeepSeek-R1 的工作表明，仅用可验证奖励做 RL，就能让模型自发出现长链推理、自我检查和回溯等行为，而不需要先用 SFT 教它这些模式。

它的边界也很清楚：只覆盖有明确对错的任务。开放写作、对话风格、有用性这些没有标准答案的能力，仍然需要奖励模型或人工偏好。实践中常见的组合是“RLVR 提升推理能力，DPO/PPO 对齐风格和偏好”。

从系统角度看，RLVR 还有个好处：验证器是普通的 CPU 程序，可以水平扩展。代码任务的测试执行、数学题的符号运算，都可以放到独立的服务集群里，不占训练和推理的 GPU。

## 43.8 Reward Hacking and Verification

奖励作弊（reward hacking）是 RL 训练最危险的问题：模型找到让奖励变高、但人类并不满意的捷径。奖励函数是代理目标，优化得越狠，代理和目标之间的缝隙就越容易被利用。

常见形式：

- **长度偏置**：奖励模型在训练数据里见到的长回答普遍得分高，模型就无限加长，直到撞上长度上限；
- **谄媚**：附和用户的错误观点，因为标注者——尤其是被评价者本人——喜欢被认同；
- **格式投机**：堆砌标题、列表和粗体，看起来结构化，实际内容空洞；
- **测试过拟合**：代码模型针对可见的测试用例优化，隐藏用例大面积失败；
- **验证器漏洞**：答案格式检查被绕开，比如输出一个能被正则匹配、但语义错误的字符串。

缓解手段：

- **奖励模型集成**：多个奖励模型取平均或最小值，单一模型的漏洞很难被同时利用；
- **KL 约束**：限制策略偏离参考模型的程度，作弊策略通常离参考策略很远；
- **定期人工评估**：自动指标看不出谄媚和空洞，必须有人看样本；
- **保留集**：用于验证奖励的测试集不能参与训练，也不能以任何形式泄漏给策略；
- **规则与验证器结合**：格式类奖励用规则，正确性用验证器，两者独立实现，避免同一个 bug 同时影响奖励和判定。

奖励作弊不会完全消除，只能监控。训练曲线里自动奖励持续上升、但人工评估停滞甚至下降，通常就是信号。这也是为什么 RL 训练必须有独立于训练奖励的评估流水线——用同一个奖励模型评估自己的策略，等于让考生自己出题。

## 43.9 Distributed Rollout Systems

RL 训练的吞吐瓶颈在 rollout：生成比训练慢一个数量级，而生成和训练的资源需求又不同——生成要大的 KV Cache 和 continuous batching，训练要大的计算 batch 和优化器显存。把它们放在同一组 GPU 上，两边的资源利用率都不高。

典型架构是把两者分开：

```text
Rollout 集群（推理引擎，vLLM/SGLang）
   ↑ 策略权重同步
训练集群（FSDP/ZeRO）
   ↓ 采样请求
奖励服务（奖励模型或验证器）
```

关键设计点：

- **权重同步**：训练完一步，把新权重推给推理引擎。7B 的 BF16 权重 14 GB，NVLink/IB 上几秒；70B 要几十秒，且每步都得同步一次，通常用分片广播或增量（只同步变化大的部分）来摊薄；
- **推理引擎配置**：rollout 用 paged KV、continuous batching，和线上推理一样；但采样温度、top-p 要按探索需要设置，通常比线上服务更“发散”；
- **异步与 off-policy**：训练和生成并行，用略旧的策略生成样本，提高利用率；代价是引入 off-policy 偏差，要靠重要性采样或限制陈旧度来控制；
- **样本长度**：RL 样本通常比 SFT 长（要生成完整推理链），KV 显存和最大生成长度都要预留够，否则长样本会被截断，奖励信号失真。

规模化时，rollout 集群往往占 70% 以上的 GPU 时间。这意味着优化 rollout 的吞吐——更大的 batch、更快的 decode、投机解码——比优化训练本身更能提升整体效率。这和前面几章讲的推理优化直接相关：RL 训练系统的一大半，其实是一个吞吐优先的推理系统。

## 本章小结

- 偏好数据来自人工标注、AI 反馈和可验证信号；标注一致性决定奖励信号的质量。
- 奖励模型用 Bradley-Terry 损失训练，偏好对准确率 70%–75% 已属不错；只在回答末尾打分，且奖励分数只有相对意义。
- DPO 用策略与参考策略的对数比构造隐式奖励，不需要奖励模型和 critic，显存约为 PPO 的一半；但离线数据容易导致过度优化。
- PPO 是 on-policy 的，需要策略、参考、奖励、critic 四个模型，7B 规模下总量超过 220 GB，必须分布式；rollout 是主要瓶颈。
- GRPO 去掉 critic，用组内奖励归一化得到优势；省显存换来成倍的生成量，组内奖励全同时不产生梯度。
- RLVR 用程序判定对错，奖励可靠、可自动化、验证器可水平扩展，但只覆盖有明确答案的任务。
- 奖励作弊表现为长度偏置、谄媚、格式投机和测试过拟合；靠集成、KL 约束和独立的人工评估监控。
- RL 训练中 rollout 占 70% 以上的 GPU 时间，分离式 rollout 集群、权重同步和异步 off-policy 是系统设计的核心。

## 延伸阅读

- Christiano et al., [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741), NeurIPS 2017.
- Ouyang et al., [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155), NeurIPS 2022.
- Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290), NeurIPS 2023.
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017.
- Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), 2024.
- DeepSeek-AI, [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948), 2025.
- Skalse et al., [Defining and Characterizing Reward Hacking](https://arxiv.org/abs/2209.13085), NeurIPS 2022.
- Bai et al., [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073), 2022.
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023.
