---
chapter: 31
part: 5
title: LLM Inference Engine Implementation
level: tbd
status: draft
prerequisites: [16, 20, 22, 25, 26, 29, 30]
lab:
---

# Chapter 31. LLM Inference Engine Implementation

前六章讲的是组件：KV Cache 怎么分配、请求怎么调度、decode 怎么加速。这一章把它们装成一个能跑的系统。

目标不是复刻 vLLM，而是实现一个最小但完整的引擎：单卡、单模型、支持连续批处理、分页 KV Cache、流式输出和取消。代码量控制在几千行，每个设计决策都能对应到前文的某个约束。做完之后，再读 vLLM 或 SGLang 的源码，会看到它们在同样的问题上做了更极端的工程取舍。

## 31.1 Runtime Architecture Design

先定约束：

- 单张 GPU，模型能完整放下；
- 支持多请求并发，decode 阶段连续批处理；
- KV Cache 分页，支持前缀共享；
- 流式返回 token；
- 请求可以取消，资源必须回收。

### 组件

```text
HTTP/SSE  ──►  API Server  ──►  Request Queue
                                    │
                              ┌─────▼─────┐
                              │ Scheduler │
                              └─────┬─────┘
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              KV Manager      Model Runner      Sampler
                    │               │               │
                    └───────────────┴───────────────┘
                                    │
                                  GPU
```

- **API Server**：解析请求、tokenize、管理 SSE 连接；
- **Scheduler**：决定每一步跑哪些请求、各跑多少 token；
- **KV Manager**：分配、共享、回收 block；
- **Model Runner**：执行 forward，返回 logits；
- **Sampler**：把 logits 变成 token。

### 线程模型

CUDA context 绑定线程，多个线程同时 launch 会引入同步和竞争。最简单的正确做法是**一个 engine 线程拥有 GPU**：

```text
API 线程池 ──push──► 无锁队列 ──► Engine 线程 ──► GPU
                                     │
                                     └──► 输出队列 ──► API 线程池
```

API 线程只做网络和 tokenize，不碰 CUDA。Engine 线程串行执行“调度 → forward → 采样 → 回写”，天然避免了并发问题。

代价是 tokenize 和 detokenize 也在关键路径外，需要额外的线程池；好处是调度器的状态不需要锁。

### 一步做什么

```cpp
while (running_) {
  Batch batch = scheduler_.next_batch();   // 选请求、分配 KV
  if (batch.empty()) { wait_for_request(); continue; }

  Tensor logits = runner_.forward(batch);  // prefill 或 decode
  sampler_.sample(batch, logits);          // 写回 token
  scheduler_.commit(batch);                // 更新状态、释放完成的请求
  streamer_.publish(batch);                // 推给 API 层
}
```

这五行就是整个引擎的主循环。后面的章节把它拆开。

## 31.2 Tensor and Model Abstractions

### Tensor

不引入完整的框架，只保留 kernel 需要的信息：

```cpp
enum class DType : uint8_t { kF16, kBF16, kF32, kI8, kI4 };

struct Tensor {
  void* data = nullptr;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;   // 以元素为单位
  DType dtype = DType::kF16;
  int device = 0;

  int64_t numel() const {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
  }
  size_t nbytes() const { return numel() * dtype_size(dtype); }
};
```

`strides` 支持非连续视图，比如 KV Cache 的 `[block, token, head, dim]` 按需切片。kernel 必须显式处理 stride，不能假设连续。

### ModelConfig

```cpp
struct ModelConfig {
  int hidden_size       = 8192;
  int intermediate_size = 28672;
  int num_layers        = 80;
  int num_heads         = 64;
  int num_kv_heads      = 8;
  int head_dim          = 128;
  int vocab_size        = 128256;
  float rms_eps         = 1e-5f;
  float rope_theta      = 500000.0f;
};
```

这些字段直接决定 KV Cache 大小、GEMM 形状和 kernel 的 tile 选择。引擎启动时从 `config.json` 读取，并做一致性校验：`hidden_size == num_heads * head_dim`，`num_heads % num_kv_heads == 0`。

### 层权重

```cpp
struct LayerWeights {
  Tensor* attn_norm;
  Tensor* wq, *wk, *wv, *wo;
  Tensor* ffn_norm;
  Tensor* w_gate, *w_up, *w_down;
};
```

权重用指针而不是值，加载后地址固定，方便 CUDA Graph 捕获。量化权重用 `kI4` 加独立的 scale 张量表示，反量化在 kernel 内完成。

## 31.3 Weight Loading

### mmap safetensors

safetensors 的 header 是一个 JSON，记录每个张量的 dtype、shape 和字节偏移。加载时只解析 header，数据用 mmap 映射：

```cpp
int fd = open(path.c_str(), O_RDONLY);
size_t file_size = lseek(fd, 0, SEEK_END);
void* base = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

// header_length 是文件前 8 字节
const char* json = static_cast<const char*>(base) + 8;
// 解析出 {name: {dtype, shape, data_offsets}}
```

张量指针是 `base + 8 + header_len + offset`。多个进程可以共享同一份 page cache，3.5 GB 的 4-bit 模型不会每个 worker 复制一份。

### 名称映射

HuggingFace 的命名和内部实现往往不同：

```text
model.layers.0.self_attn.q_proj.weight  →  layers[0].wq
model.layers.0.mlp.down_proj.weight     →  layers[0].w_down
```

维护一张映射表，加载时逐项检查。缺失或多余的张量都要报错，不能静默跳过——量化模型最常见的 bug 就是某一层的 scale 没加载，输出缓慢退化。

### 布局转换

如果 kernel 要求特定的打包顺序（比如 GPTQ 的 `qweight` 按 K 方向打包、NVFP4 的 128×4 块），在加载时做一次转换，并把结果缓存成最终格式。不要每次启动都重排。

### 校验

加载后取几个张量做数值抽查：

```cpp
// 取一行权重，反量化后和原始 FP16 对比
float max_err = check_row(layers[0].wq, ref_wq, row=0);
assert(max_err < 1e-2);
```

这一步能抓住打包顺序、scale 索引、转置约定三类错误。

## 31.4 Operator Execution

### 算子接口

```cpp
struct OpContext {
  cudaStream_t stream;
  Workspace* workspace;
};

using OpFn = void (*)(const Tensor* const* inputs,
                      Tensor* const* outputs,
                      const OpParams& params,
                      OpContext* ctx);
```

每个算子注册名字和函数指针：

```cpp
struct OpEntry { const char* name; OpFn fn; };
std::unordered_map<std::string, OpFn> registry;
```

模型执行时按名字查找。查找发生在初始化阶段，不在每步的热路径上。

### 组合成层

```cpp
void decoder_layer(const LayerWeights& w, Tensor& h,
                   KVCache& kv, int layer, OpContext* ctx) {
  Tensor norm = ctx->workspace->alloc_like(h);
  rms_norm(h, w.attn_norm, norm, ctx);
  Tensor q = ctx->workspace->alloc({seq, hidden});
  Tensor k = ctx->workspace->alloc({seq, kv_dim});
  Tensor v = ctx->workspace->alloc({seq, kv_dim});
  qkv_gemm(norm, w.wq, w.wk, w.wv, q, k, v, ctx);

  kv.append(layer, k, v);
  Tensor attn = ctx->workspace->alloc_like(h);
  paged_attention(q, kv, layer, attn, ctx);

  Tensor proj = ctx->workspace->alloc_like(h);
  gemm(attn, w.wo, proj, ctx);
  add_inplace(h, proj, ctx);

  rms_norm(h, w.ffn_norm, norm, ctx);
  Tensor gate = ctx->workspace->alloc({seq, intermediate});
  Tensor up = ctx->workspace->alloc({seq, intermediate});
  gemm(norm, w.w_gate, gate, ctx);
  gemm(norm, w.w_up, up, ctx);
  silu_mul(gate, up, gate, ctx);
  Tensor down = ctx->workspace->alloc_like(h);
  gemm(gate, w.w_down, down, ctx);
  add_inplace(h, down, ctx);
}
```

这是“能跑”的版本，每个算子一次 launch。生产版本会把 `rms_norm + qkv_gemm`、`silu_mul + down_gemm` 融合，减少 HBM 往返。

### Workspace

每步的中间张量从 workspace 分配。workspace 是一块预分配的显存，按生命周期复用：

```cpp
class Workspace {
  char* base_;
  size_t offset_ = 0;
 public:
  Tensor alloc(std::vector<int64_t> shape);
  void reset() { offset_ = 0; }
};
```

同一层内的临时张量可以复用；跨层保留的只有 hidden state 和 KV。`reset()` 在每步开始时调用，避免反复 `cudaMalloc`。

### GEMM 选择

- prefill 的 `M` 大，用 cuBLASLt 的 heuristic 选 tile；
- decode 的 `M` 小，用专门的 GEMV 或 W4A16 kernel；
- 按 `M` 分派，而不是一个 GEMM 打天下。

## 31.5 CUDA Backend

### 设备与流

```cpp
class CudaBackend {
 public:
  CudaBackend(int device) {
    cudaSetDevice(device);
    cudaStreamCreate(&stream_);
    cublasCreate(&blas_);
    cublasSetStream(blas_, stream_);
  }
  cudaStream_t stream() { return stream_; }
 private:
  cudaStream_t stream_;
  cublasHandle_t blas_;
};
```

整个引擎只有一个流。多流在这里没有收益：每一步的算子有严格依赖，拆流只会增加同步。

### Kernel launch 封装

```cpp
template <typename... Args>
void launch(void (*kernel)(Args...), dim3 grid, dim3 block,
            size_t smem, cudaStream_t stream, Args... args) {
  kernel<<<grid, block, smem, stream>>>(args...);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}
```

错误检查不要省。kernel 越界在 decode 时可能表现为“某个请求的输出乱码”，很难定位。

### CUDA Graph

decode 的形状固定（batch 大小决定 grid），适合捕获：

```cpp
cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
run_decoder_layer(...);   // 所有 launch 都在 stream 上
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
```

batch 大小变化时重新捕获，或者维护一组按 batch 分档的 graph（1、2、4、8、16、32）。分档会浪费一点显存带宽，但避免每步重捕获。

### 异步 H2D

输入 token、采样参数等小数据用 pinned buffer 异步拷贝：

```cpp
cudaMemcpyAsync(d_tokens, h_tokens, bytes, cudaMemcpyHostToDevice, stream);
```

不要在这之后立刻 `cudaStreamSynchronize`，让后续 kernel 自然排队。

## 31.6 KV Cache Implementation

### Block 池

```cpp
class BlockPool {
 public:
  BlockPool(int64_t num_blocks, int64_t block_bytes)
      : free_(num_blocks), ref_(num_blocks, 0) {
    for (int64_t i = 0; i < num_blocks; ++i) free_[i] = i;
  }

  int32_t allocate() {
    if (free_.empty()) return -1;
    int32_t id = free_.back();
    free_.pop_back();
    ref_[id] = 1;
    return id;
  }

  void add_ref(int32_t id) { ++ref_[id]; }

  void release(int32_t id) {
    if (--ref_[id] == 0) free_.push_back(id);
  }

 private:
  std::vector<int32_t> free_;
  std::vector<int32_t> ref_;
};
```

`ref_` 支持 prefix sharing：两个请求可以指向同一个 block，最后一个释放时才真正回收。

### Block table

每个序列维护自己的逻辑块表：

```cpp
struct SequenceKV {
  std::vector<int32_t> block_table;  // 逻辑块 → 物理块
  int32_t num_tokens = 0;
};
```

追加 token 时，如果 `num_tokens % block_size == 0`，分配一个新块：

```cpp
void append(SequenceKV& s, int32_t token_count) {
  int32_t needed = (s.num_tokens + token_count + block_size - 1) / block_size;
  while ((int32_t)s.block_table.size() < needed) {
    int32_t id = pool_->allocate();
    if (id < 0) throw OutOfMemory();
    s.block_table.push_back(id);
  }
  s.num_tokens += token_count;
}
```

### 物理布局

一个 block 存 `block_size` 个 token 的 K 和 V：

```text
K: [num_blocks, block_size, num_kv_heads, head_dim]
V: [num_blocks, block_size, num_kv_heads, head_dim]
```

70B、FP16、`block_size = 16` 时，一个 block 覆盖所有层的 K 和 V，字节数是：

$$
B_{\text{block}} = 2 \times 80 \times 16 \times 8 \times 128 \times 2 = 5 \text{ MiB}
$$

60 GB 的 KV 池能放 12,288 块，即 196,608 个 token。按 8K 上下文算，约 24 个并发序列。

### Attention 读取

kernel 不假设 KV 连续，而是按 block table 寻址：

```cuda
int phys = block_table[logical_block];
const half* k = K + ((int64_t)phys * block_size + offset)
                    * kv_heads * head_dim;
```

每个 block 内部是连续的，所以一个 tile 的读取仍然可以合并。跨 block 的跳转由 block table 承担，代价是一次额外的间接寻址。

## 31.7 Batch Scheduler Implementation

### 请求状态

```cpp
enum class State { kWaiting, kPrefill, kDecode, kFinished, kAborted };

struct Request {
  int64_t id;
  std::vector<int32_t> prompt;
  std::vector<int32_t> output;
  int32_t max_new_tokens;
  SamplingParams params;
  State state = State::kWaiting;
  SequenceKV kv;
};
```

### 一步的调度

```cpp
std::deque<std::unique_ptr<Request>> waiting_, decoding_;

Batch Scheduler::next_batch() {
  Batch batch;

  // 1. 所有 decode 请求各取 1 个 token
  for (auto& r : decoding_) {
    if (batch.tokens + 1 > token_budget_) break;
    batch.add_decode(r.get());
    batch.tokens += 1;
  }

  // 2. 用剩余预算做 prefill，按优先级/到达时间
  while (!waiting_.empty() && batch.tokens < token_budget_) {
    Request* r = waiting_.front().get();
    int32_t chunk = std::min<int32_t>(
        r->prompt.size() - r->prefilled, token_budget_ - batch.tokens);
    chunk = std::min(chunk, max_prefill_chunk_);
    batch.add_prefill(r, r->prefilled, chunk);
    batch.tokens += chunk;
    r->prefilled += chunk;
    if (r->prefilled == (int32_t)r->prompt.size()) {
      r->state = State::kDecode;
      decoding_.push_back(std::move(waiting_.front()));
      waiting_.pop_front();
    }
  }
  return batch;
}
```

关键点：

- decode 优先。已经在生成的请求不能被 prefill 饿死，否则 TPOT 抖动。
- prefill 分块。`max_prefill_chunk_` 通常 512–2048；太大时这一步的耗时会长到几百毫秒，decode 的 TPOT 出现尖峰。
- token budget。一步总 token 数有上限，防止某个长 prompt 独占。

以 70B 为例，512 token 的 prefill chunk 需要 `2 × 70e9 × 512 ≈ 72 TFLOP`；按 50% 峰值算约 145 ms。同一迭代里的 decode 请求会被这一块拖慢同样的时间。chunk 越小，TPOT 越平滑，但 prefill 的固定开销占比越高，TTFT 变长。

decode 侧的吞吐上限是：

$$
\text{throughput} = \frac{\text{batch}}{\text{TPOT}}
$$

batch 32、TPOT 42 ms 时约 762 token/s。再往上加 batch，权重读取被摊薄，但 KV 读取和 attention 会逐渐变成瓶颈，TPOT 开始上升。

### 抢占

显存不够时，调度器要腾出 KV：

- **recompute**：丢掉 KV，回到 waiting，下次重新 prefill。代价是重复计算，但省显存。
- **swap**：把 KV 拷到 CPU，之后拷回来。代价是 PCIe 带宽，但保留计算。

decode 请求被抢占的代价很高（用户已经等了很多 token），通常优先抢占刚进入 decode 或 prompt 很长的请求。

### 完成与回收

```cpp
void Scheduler::commit(const Batch& batch) {
  for (auto* r : batch.finished) {
    for (int32_t id : r->kv.block_table) pool_->release(id);
    r->state = State::kFinished;
  }
}
```

每个 block 的引用计数必须和分配严格配对。泄漏一个 block，长跑服务最终会 OOM；重复释放会破坏空闲链表，表现为随机崩溃。

## 31.8 Sampling Implementation

### 处理顺序

采样参数的应用顺序会影响结果，必须固定：

```text
logits → repetition penalty → temperature → top-k → top-p → multinomial
```

temperature 放在 top-k/top-p 之前，因为缩放会改变排序后的概率分布；反过来先截断再缩放，等价于改变了截断阈值。

### 代码

```cpp
int32_t sample(float* logits, int32_t vocab,
               const SamplingParams& p, std::mt19937& rng) {
  // repetition penalty
  for (int32_t t : p.recent_tokens) {
    logits[t] = logits[t] > 0 ? logits[t] / p.rep_penalty
                              : logits[t] * p.rep_penalty;
  }

  // temperature
  if (p.temperature != 1.0f) {
    for (int32_t i = 0; i < vocab; ++i) logits[i] /= p.temperature;
  }

  // top-k: 选出前 k 个，其余置 -inf
  top_k_filter(logits, vocab, p.top_k);

  // top-p: 排序后累加，超过 p 的截断
  top_p_filter(logits, vocab, p.top_p);

  // softmax + multinomial
  return multinomial(logits, vocab, rng);
}
```

### 性能

词表 128k，每步都要做一次 top-k 和 top-p。全排序是 `O(V log V)`，128k 时约 2M 次比较，在 GPU 上几微秒，但在 CPU 上会成为瓶颈。

优化：

- top-k 用 radix select，`O(V)`；
- top-p 只需要部分排序，或者用直方图近似；
- 采样在 GPU 上做，只把选中的 token 拷回 host；
- batch 内每个请求的采样参数不同，用一个 kernel 处理整个 batch。

### 确定性

给每个请求独立的 RNG 种子。相同 prompt、相同参数、相同种子，输出必须一致。这对测试和复现至关重要。

注意 batch 组合会影响浮点归约顺序，进而影响 logits 的末位。如果要求跨 batch 完全确定，需要固定 kernel 的归约顺序，代价是性能。

## 31.9 API and Streaming

### 请求接口

```json
POST /v1/completions
{
  "prompt": "...",
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": true
}
```

API 层负责：

- 校验参数；
- tokenize；
- 生成请求 ID，推入队列；
- 注册取消回调。

### 流式返回

SSE 格式：

```text
data: {"id":"req-1","choices":[{"text":"Hello"}]}

data: {"id":"req-1","choices":[{"text":" world"}]}

data: [DONE]
```

每个 token 到达时 flush 一次。不要攒够一批再发，那会破坏“流式”的意义。

### 增量 detokenize

tokenizer 的 token 可能对应半个 UTF-8 字符。直接 `decode` 单个 token 会产生替换字符。正确做法是维护字节缓冲：

```python
buf = b""
for token in stream:
    buf += tokenizer.token_to_bytes(token)
    text, buf = split_valid_utf8(buf)
    if text:
        send(text)
```

`split_valid_utf8` 只输出完整的字符，把不完整的字节留在缓冲里。

### 取消

客户端断开连接时：

1. API 层标记请求为 `kAborted`；
2. 调度器在下一步跳过它；
3. 释放 KV block，从 decode 队列移除；
4. 停止生成，不再消耗 GPU。

取消必须及时。一个已经断开的请求如果继续 decode，会白白占用带宽和 KV，长跑服务里这是常见的容量泄漏。

## 31.10 End-to-end Testing

### 单元测试

每个算子与 PyTorch 参考对比：

```python
ref = torch.nn.functional.rms_norm(x, (hidden,), weight, eps)
err = (out - ref).abs().max()
assert err < 1e-2
```

覆盖边界：`seq = 1`、非 tile 倍数、全零、极大值。

### 端到端测试

- **贪心确定性**：`temperature = 0`，同一 prompt 两次输出必须完全一致；
- **batch 不变性**：单独跑和混批跑，贪心输出应一致（允许 logits 末位差异，但 argmax 一致）；
- **前缀共享正确性**：两个请求共享 prefix，分别生成，结果应和独立跑一致；
- **长上下文**：8K、32K 输入，检查不越界、不 OOM；
- **取消**：生成中断开，确认 block 全部回收。

### 压测

用泊松到达模拟真实流量：

```text
arrival rate λ = 10 req/s
prompt 长度 ~ 对数正态，均值 512
output 长度 ~ 指数分布，均值 128
```

测：

- TTFT 的 p50/p99；
- TPOT 的 p50/p99；
- 吞吐（tokens/s）；
- 显存峰值；
- 取消后的显存回收。

p99 比 p50 更能反映调度问题。一个长 prompt 如果没有分块，会把 p99 TPOT 拉高几倍。

### 故障注入

- 显存耗尽：确认调度器触发抢占，而不是崩溃；
- 非法请求：超长 prompt、负的 max_tokens、非法采样参数；
- 客户端中途断开；
- GPU 错误：kernel 越界、ECC 错误。

这些路径在正常测试里覆盖不到，但生产环境一定会遇到。

## 本章小结

- 引擎的主循环是“调度 → forward → 采样 → 提交”，一个线程拥有 GPU 可以避免大部分并发问题。
- Tensor 只需携带 data、shape、stride、dtype；workspace 复用中间张量，避免每步分配。
- 权重用 mmap 加载，名称映射和打包顺序要显式校验；错误不会报错，只会让输出缓慢退化。
- 算子按名字注册，按形状分派 GEMM；decode 和 prefill 用不同的 kernel。
- KV Cache 用 block 池加引用计数，block table 完成逻辑到物理的映射；70B 的 5 MiB block 在 60 GB 池里能放约 19.6 万 token。
- 调度器 decode 优先、prefill 分块、token budget 限制单步规模；显存不足时 recompute 或 swap。
- 采样顺序固定为 repetition penalty → temperature → top-k → top-p；每个请求独立 RNG。
- 流式输出要处理半个 UTF-8 字符；取消要及时释放 KV。
- 测试覆盖算子正确性、贪心确定性、batch 不变性、前缀共享、长上下文、取消和压测。

## 延伸阅读

- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023.
- vLLM, [GitHub Repository](https://github.com/vllm-project/vllm).
- SGLang, [GitHub Repository](https://github.com/sgl-project/sglang).
- ggml-org, [llama.cpp](https://github.com/ggml-org/llama.cpp).
- Hugging Face, [safetensors](https://github.com/huggingface/safetensors).
- NVIDIA, [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).
