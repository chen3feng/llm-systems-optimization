---
chapter: 27
part: 5
title: Advanced KV Cache Management
level: tbd
status: draft
prerequisites: [26]
lab:
---

# Chapter 27. Advanced KV Cache Management

一个客服系统里，八成请求带着同一段 2,000 token 的系统提示词。每个请求都重新 prefill，8×H100 上多花约 90 毫秒；一天一百万次请求，累计约 200 GPU 小时的重复计算。把这些 KV 留下来复用，看起来只是加一个哈希表，实际要回答三个问题：怎么判断两段前缀相同，共享的块什么时候可以写，缓存满了淘汰谁。

这一章从命中条件讲到 radix tree、引用计数、copy-on-write、offload、压缩、量化和跨实例传输，最后讨论正确性：这些机制出错时通常不会崩溃，而是悄悄返回错误的文本。

## 27.1 Prefix Caching

Prefix caching 的前提是：两个请求的开头若干 token 完全相同，那么这些 token 在每一层产生的 K 和 V 也完全相同，可以直接复用，不必重新计算。

命中判定比字符串比较严格。缓存键至少包含：

- 完整的 token ID 序列，不是原始文本；
- 模型权重版本，或权重文件的哈希；
- tokenizer 版本与 chat template；
- RoPE 配置，包括 scaling 方式和 base；
- LoRA adapter 的 ID；
- KV 量化的格式与 group size；
- 租户 ID。

任何一项不同，复用都会得到错误的 KV。文本相同但 tokenizer 版本不同，token 序列就可能不同；同一个模型换了 adapter，K/V 分布也会变。多模态请求还要把图像哈希放进键里。

匹配以 block 为单位。设 block 大小为 `B`（常见 16），前缀长度为 `L_h`，实际能复用的长度是 `⌊L_h / B⌋ × B`，剩下不足一块的 token 要重新计算。短前缀场景下，这个取整会明显拉低命中率。

Prefix caching 省的是 prefill，不是 decode。prefill 的 FLOPs 约为 `2PL`，`P` 是参数量，`L` 是 token 数。在 8×H100、70B、40% MFU 下：

$$
T_{\text{prefill}}(L) \approx \frac{2PL}{\eta F}, \qquad
\Delta T = \frac{2PL_h}{\eta F}
$$

其中 `F = 8 × 989 TFLOPS`，`η = 0.4`。代入 `L_h = 2048`，省下约 90.6 毫秒。

| 共享前缀长度 | 省下的 prefill | KV 占用 | 典型场景 |
| --- | --- | --- | --- |
| 512 | 22.7 ms | 160 MiB | 短 system prompt |
| 2K | 90.6 ms | 640 MiB | 客服机器人 |
| 8K | 362 ms | 2.6 GB | 长文档问答 |
| 32K | 1.45 s | 10.5 GB | 代码仓库上下文 |

KV 占用按 320 KiB/token 计算。缓存不是免费的：留下来的块占着 HBM，挤占并发空间。生产系统通常把 10%–30% 的 KV 预算给缓存，再用命中率和淘汰率调整。命中率过低时，缓存的开销会超过它省下的计算。

## 27.2 Radix Tree and Prefix Indexing

Prefix cache 需要按 token 序列查找、插入和淘汰的索引。Radix tree 的每条边对应一段 token，从根到节点的路径就是一个前缀。SGLang 的 RadixAttention 用的就是这个结构。

```cpp
struct RadixNode {
  std::vector<int32_t> label;   // 边上压缩存储的 token 片段
  std::vector<int32_t> blocks;  // 该片段对应的物理 KV block
  std::unordered_map<int32_t, std::unique_ptr<RadixNode>> children;
  RadixNode* parent;
  int32_t ref_count;            // 正在使用该前缀的请求数
  uint64_t last_access;
  bool sealed;                  // KV 是否已经写完
};
```

`label` 做路径压缩：只有单个孩子的节点与孩子合并，避免每个 token 建一个节点。`blocks` 按 block 边界切分，一条边可能覆盖多个块。

### 匹配

从根开始，按第一个 token 找孩子，再逐 token 比较 `label`：

```python
def match(node, tokens, i):
    while i < len(tokens):
        child = node.children.get(tokens[i])
        if child is None:
            break
        j = 0
        while (j < len(child.label) and i + j < len(tokens)
               and child.label[j] == tokens[i + j]):
            j += 1
        if j < len(child.label):
            child = split(child, j)   # 在边中间分裂
        node = child
        i += j
    return node, i
```

失配在边中间时要分裂：新建节点保存前 `j` 个 token，把原节点挂到它下面，`blocks` 按 `j` 切开。切分点不在 block 边界上时，那个块只能复制，或让新节点少复用半个块。

插入时，剩余 token 按 block 切好，分配物理块，写入 KV，再挂到树上；写完前标记为未 sealed。并发插入同一前缀时，用树锁或 compare-and-swap 保证只有一个成功。

淘汰从叶子开始。只有 `ref_count == 0` 的叶子可以删；父节点若也变成无引用叶子，继续向上回收。内部节点只要还有孩子就不能删，否则会切断孩子的路径。策略通常是 LRU。

Radix tree 的查找是 `O(L)` 次 token 比较。块哈希链每个块的键是 `hash(parent_hash, block_tokens)`，用哈希表直接定位，vLLM 采用后者。

| 维度 | Radix tree | 块哈希链 |
| --- | --- | --- |
| 匹配粒度 | 任意 token 前缀，可分裂 | block 对齐 |
| 查找成本 | 逐 token 比较，`O(L)` | 每块一次哈希，`O(L/B)` |
| 插入与分裂 | 需要节点分裂、路径压缩 | 只追加块 |
| 冲突处理 | 逐 token 精确比较 | 需要额外校验 token |

Radix tree 能表达任意长度的共享，包括两个请求只在中间一段分叉；块哈希链更简单，但只能按块共享。system prompt 固定时两者差别不大；分支多的 agent 轨迹，radix tree 的共享率更高。

## 27.3 Reference Counting and Shared Blocks

物理块可以被多个请求的 block table 同时指向。每个块维护一个引用计数，记录有多少个所有者。

```cpp
struct Block {
  int32_t id;
  int32_t ref_count;      // block table 条目 + 树节点引用
  bool sealed;            // 是否已写完， sealed 块只读
  uint64_t last_used;
  std::vector<int32_t> token_ids;  // 用于命中校验
};
```

引用计数的语义要精确：块被请求的 block table 指向一次，计数加一；被 radix tree 节点持有，再算一个引用。请求结束或节点淘汰时减一，计数归零且不在树上的块才回到空闲链表。

块有四种状态：

| 状态 | ref_count | 能否写入 | 能否淘汰 |
| --- | --- | --- | --- |
| FREE | 0 | 分配后可写 | 本就在空闲链表 |
| CACHED | 0 | 否，内容仍有效 | 可以，LRU 优先 |
| SHARED | ≥2 | 否，写前要 COW | 不可以 |
| PRIVATE | 1 | 可以 | 不可以 |

CACHED 最微妙：没有请求在用，但树还挂着它，随时可能被命中。它和 FREE 的区别是内容有效，和 SHARED 的区别是没有活跃读者。命中 CACHED 块时，必须在同一把锁里完成“查找 + 计数加一”，否则可能被淘汰并重新分配。

共享的收益很直接。100 个请求共享同一段 2,000 token 的系统提示词，只需要 125 个块，每块 `16 × 320 KiB = 5 MiB`，合计 625 MiB；各存一份则是 12,500 个块，即 62,500 MiB，约 61 GiB。8×H100 的 HBM 共 640 GB，这一项差了约 10% 的整机显存。

引用计数本身开销可忽略，代价在同步。调度器单线程拥有 GPU 时，多数计数不需要原子指令；有异步传输或分离式 prefill 时，就必须原子操作并规定锁序。

## 27.4 Copy-on-write

共享块默认只读。一个请求要往共享块里写新 token 时，必须先复制出一份私有副本，这就是 copy-on-write。

触发条件很窄。写满的块本来就不可变，新 token 会分配新块；只有最后一个未写满的块可能被共享，典型场景是从缓存前缀继续生成且前缀不按块对齐，或者 beam search、并行采样从共同前缀分叉。

Copy-on-write 的成本是一次块复制：

$$
T_{\text{cow}} = \frac{2 B \cdot 320\text{ KiB}}{3.35\text{ TB/s}}
$$

`B = 16` 时约 3.1 微秒，`B = 128` 时约 25 微秒。相对 42 毫秒/token 的 decode 下限，这个开销不到 0.1%。真正需要小心的是同步：复制必须在写 kernel 之前完成，且不能让其他读者看到半成品。

```cpp
BlockId copy_on_write(BlockId old, BlockTable& table, int slot) {
  if (pool[old].ref_count == 1) return old;  // 已经私有
  BlockId fresh = pool.allocate();
  cudaMemcpyAsync(pool[fresh].data, pool[old].data,
                  kBlockBytes, cudaMemcpyDeviceToDevice, stream);
  pool[old].ref_count--;
  pool[fresh].ref_count = 1;
  table[slot] = fresh;
  return fresh;
}
```

复制和后续的写用同一个 stream，顺序自然有保证；跨 stream 要插事件同步。元数据更新必须在锁内完成，否则两个请求可能同时认为自己是唯一所有者。

多数生产系统只共享完整块，牺牲尾部十几个 token 的复用，换取无 COW 的简单路径。并行采样这类高频分叉场景，则在分叉时直接分配私有尾块，把复制消灭在分配阶段。

## 27.5 KV Cache Offloading

显存不够时，可以把不活跃请求的 KV 搬到 CPU 内存或 NVMe，恢复时再搬回。可行性由带宽决定。

| 层级 | 容量 | 带宽 | 8K KV 传输 |
| --- | --- | --- | --- |
| HBM | 80 GB/卡 | 3.35 TB/s | 0.8 ms |
| CPU DRAM | 1 TB/节点 | 200–300 GB/s | 9–13 ms |
| PCIe Gen5 x16 | — | 64 GB/s | 41 ms |
| NVMe Gen4 | 30 TB | 7 GB/s | 371 ms |
| RDMA 400 Gb/s | — | 50 GB/s | 52 ms |

8K 上下文的 KV 是 2.6 GB。按 PCIe Gen5 的实际 50 GB/s 算，往返 104 毫秒；重新 prefill 这 8K 在 8×H100、40% MFU 下约 362 毫秒，所以 offload 仍然划算。

什么时候不划算？把两条时间写成上下文长度 `L` 的函数，令它们相等：

$$
\frac{2 \cdot 320\text{ KiB}\cdot L}{BW}
=
\frac{2PL}{\eta F}
\quad\Longrightarrow\quad
BW^{*} = \frac{320\text{ KiB}\cdot \eta F}{P}
$$

代入 70B、8×H100、`η = 0.4`，得到 `BW* ≈ 14.8 GB/s`。带宽高于它，offload 加重新加载比重新计算便宜；低于它，不如丢掉 KV 重算。PCIe Gen4 的 25 GB/s 在线上，NVMe 的 7 GB/s 在线下。

这个比较只算时间，没算显存的机会成本。即使 offload 比重算慢，把 KV 搬走能立刻释放 GPU，整机吞吐可能反而更高。

实现要点：用 pinned memory 避免额外的 CPU 拷贝；按层分块传输，和 block 布局对齐；用独立 copy stream 和前向计算重叠；恢复时预取；记录每个块的 CPU 地址，防止重复搬移。

NVMe 容量大，但 7 GB/s 只适合冷请求。128K 请求的 KV 是 42 GB，按 NVMe 要 6 秒，已超过重新 prefill 的时间。生产上更常见的是 DRAM 做热缓存、NVMe 做冷备份。

## 27.6 KV Cache Compression

压缩的思路是只保留一部分 token 的 KV。

H2O 用累计注意力分数给 token 打分：把每个 token `j` 在各层、各头、各 query 上收到的注意力权重加起来，保留分数最高的若干 token，加上最近的窗口。预算为 `r` 时，KV 占用是 `r × L × 320 KiB`；`r = 0.2`、`L = 128K` 时约 8.4 GB，省了八成。

SnapKV 换了一个视角：在 prompt 末尾取观察窗口（如最后 32 个 token），用这些 query 对前面所有 KV 算注意力，选出每个头最重要的位置，并把相邻 token 聚成簇。StreamingLLM 保留 4 个 attention sink 加滑动窗口，内存恒定，但没有长程回忆。

| 方法 | 评分依据 | 预算 | 优点 | 风险 |
| --- | --- | --- | --- | --- |
| H2O | 累计注意力 | 20%–30% | 简单，在线可用 | 早期淘汰后无法恢复 |
| SnapKV | 末尾观察窗口 | 1K–2K token | 保留 prompt 关键信息 | 对未来 query 的预测有限 |
| StreamingLLM | sink + 窗口 | 固定 | 内存恒定，流式稳定 | 丢失远距离信息 |

压缩的核心风险是不可逆。被淘汰的 token 后面可能又需要，尤其在多轮对话、代码引用和精确检索里。H2O 假设重要的 token 会持续重要，但问题出现在末尾时，前面被判定为不重要的细节可能正是答案。

压缩适合内存紧张、偏开放生成的场景；需要精确召回时优先量化。压缩还会破坏 prefix 共享：两个请求对同一前缀选了不同 token 集合就不能共用缓存。

## 27.7 KV Cache Quantization

量化不改变保留哪些 token，只改变每个 token 占多少字节。FP16 下 70B 的 KV 是 320 KiB/token；FP8 或 INT8 减半，INT4 再减半。

| 格式 | 每 token | 8K 上下文 | 128K 上下文 | 质量 |
| --- | --- | --- | --- | --- |
| FP16 | 320 KiB | 2.6 GB | 42 GB | 基准 |
| FP8 E4M3 | 160 KiB | 1.3 GB | 21 GB | 几乎无损 |
| INT8 | 160 KiB | 1.3 GB | 21 GB | 很小 |
| INT4 group 128 | 85 KiB | 0.68 GB | 11 GB | 小到中 |
| KIVI 2-bit | 40 KiB | 0.33 GB | 5.2 GB | 中，长上下文明显 |

量化的基本公式是仿射变换：

`x̂ = (q − z) · s`，`q = clamp(round(x / s) + z, q_min, q_max)`

粒度选择有明确方向：K 按 channel 量化，V 按 token 量化。

K 的某些通道在不同 token 上持续偏大，是稳定的离群通道。按 token 算一个 scale 会被它主导，其他通道被压成很少几个量化级；按 channel 量化，每个通道有自己的 scale。V 的离群值随 token 变化，且误差直接加到输出上，按 token 量化更合适。

Scale 开销很小：K 按 channel 共 81,920 个 FP16 scale，约 160 KB；V 按 token 在 8K 下约 1.3 MB，不到 KV 的 0.05%。Kernel 层面，量化 KV 不能直接喂给只接受 FP16 的 FlashAttention。常见做法是在 attention kernel 里把 K/V tile 反量化到 BF16 放进 shared memory，QK 用 FP32 累加；Hopper 上也可以用 FP8 MMA 直接算 QK，但 softmax 仍在 FP32 里做。真正的麻烦是 kernel 要同时支持 paged layout、GQA 和量化 scale 三套索引。

精度风险集中在 attention score。K 的误差经过 `QK^T` 放大，量级约 `√d`；FP8 相对误差约 1%，`d = 128` 时 logits 误差约 0.1，通常可接受。INT4 的误差大一个量级，要单独评估。K 一般在 RoPE 之后量化，因为缓存里存的就是加过位置编码的 K。

## 27.8 Cache Persistence and Transfer

单机缓存之外，还有两类跨实例的用法：把 KV 存到外部存储，以及在 prefill 和 decode 实例之间传输。

LMCache 把 KV 当成可寻址对象：每个 chunk 按 token 哈希索引，存在 CPU DRAM、本地磁盘或远端存储。vLLM 通过 connector 接入，命中时直接加载 KV，跳过 prefill。

Prefill-decode 分离让 prefill 实例算完长 prompt 的 KV，通过 RDMA 或 NVLink 传给 decode 实例。两个阶段独立扩缩，prefill 的长请求不会打断 decode 的 TPOT。

| 机制 | 介质 | 带宽 | 范围 | 典型用途 |
| --- | --- | --- | --- | --- |
| LMCache 本地 | DRAM / NVMe | 50 / 7 GB/s | 单节点 | 前缀复用、重启恢复 |
| LMCache 远端 | RDMA | 25–50 GB/s | 集群 | 跨实例共享 |
| Mooncake | RDMA | 50 GB/s | 集群 | KV 池化 |
| PD 分离 | NVLink / RDMA | 50–900 GB/s | 节点内/间 | TTFT 与 TPOT 解耦 |
| NVMe 持久化 | PCIe | 7 GB/s | 单节点 | 冷启动预热 |

传输量按 320 KiB/token 算。8K 是 2.6 GB，400 Gb/s 链路上要 52 毫秒；128K 是 42 GB，要 840 毫秒。这个时间和 prefill 同量级，所以分离是否划算取决于重叠：传输请求 A 的 KV 时，prefill 实例继续处理请求 B。

跨实例共享还有两个非性能问题。安全上，缓存命中与否会泄漏前缀是否存在，攻击者可以用计时探测其他租户的 prompt；生产系统要么按租户隔离，要么在哈希里加盐。版本上，缓存键必须包含模型权重哈希、tokenizer 哈希、RoPE 配置、量化格式和 block 大小，任何一项变了旧 KV 都不能用。

## 27.9 KV Cache Correctness

Prefix cache 的 bug 很少让程序崩溃，它们让输出变差，或者只在特定请求组合下出错。维护不变量比事后调试有效。

| 不变量 | 违反后的表现 | 检测方法 |
| --- | --- | --- |
| `ref_count` 等于块表条目数加树引用 | 提前释放或泄漏 | 周期性审计，求和比对 |
| sealed 块不被写入 | 其他请求读到被污染的 KV | 写前检查 sealed 标志 |
| 空闲块不在任何块表或树中 | 重复分配，内容互相覆盖 | 分配时断言状态 |
| 命中块的 token 序列与请求一致 | 静默返回错误文本 | 逐块比较 token |

并发是最大的风险。查找和计数加一必须在同一临界区完成；否则线程 A 查到块还没加计数，线程 B 把它淘汰并重新分配，A 随后读到别人的 KV。可以用树锁保护查找路径，也可以用 RCU：读者拿 epoch，淘汰者等读者退出后再释放。COW 的元数据和块表更新同样要在锁内完成。

哈希碰撞也是静默错误源。64 位哈希在 `n` 个块上的碰撞概率约为：

$$
P_{\text{coll}} \approx \frac{n^2}{2^{65}}
$$

`n = 10^6` 时约 `2.7 × 10^-8`，可以忽略；`n = 10^9` 时约 2.7%，不能忽略。长跑集群的块数量很容易到十亿级。解决办法有两个：用 128 位哈希，或者命中时逐 token 比较。Radix tree 本身就是精确比较，块哈希链必须在块里存 token ID 并校验。

最后是语义正确性。复用的 KV 必须和重新计算的结果数值等价；position id 没有从正确位置继续、attention mask 跨过文档边界、量化或 RoPE 配置不同，都会破坏这个等价。

测试上最有效的是差分测试：同一组请求开、关缓存各跑一遍，固定种子，逐 token 比较输出；再加并发模糊和 soak，检查引用计数。

## 本章小结

- Prefix caching 复用相同 token 前缀的 KV；命中键要包含 tokenizer、模型、RoPE、adapter、量化和租户，匹配按 block 对齐。
- Radix tree 用路径压缩索引任意前缀，支持分裂和 LRU 叶子淘汰；块哈希链更简单，但要校验碰撞。
- 引用计数区分 FREE、CACHED、SHARED、PRIVATE；查找和加一必须原子；COW 只在写共享未满块时触发，一次约 3–25 微秒。
- Offload 的盈亏平衡点约 14.8 GB/s，PCIe 够用，NVMe 通常不够；KV 压缩按注意力保留 token，不可逆，量化把 320 KiB/token 降到 160 KiB 或更低。
- 跨实例传输（LMCache、PD 分离）靠重叠摊薄，传输量和 prefill 同量级。
- 正确性靠不变量、原子计数、精确命中校验和差分测试；缓存错误通常表现为静默的质量下降。

## 延伸阅读

- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), 2023.
- Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104), 2024.
- Gim et al., [Prompt Cache: Modular Attention Reuse for Low-Latency Inference](https://arxiv.org/abs/2311.04934), 2023.
- vLLM, [Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html).
- Zhang et al., [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048), 2023.
- Li et al., [SnapKV: LLM Knows What You are Looking for Before Generation](https://arxiv.org/abs/2404.14469), 2024.
- Liu et al., [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750), 2024.
- Xiao et al., [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453), 2023.
- Cheng et al., [LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference](https://arxiv.org/abs/2510.09665), 2025.
- Qin et al., [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079), 2024.
- Zhong et al., [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670), 2024.
- Patel et al., [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677), 2023.
