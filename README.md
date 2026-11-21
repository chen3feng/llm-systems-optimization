# LLM Systems Optimization

**Algorithms, GPU Kernels, Runtime and Distributed Execution**

一本约 800 页、以 C++/CUDA 为主要实践语言、以 LLM 底层系统为核心的技术专著。

在线阅读：<https://blog.chen3feng.top/llm-systems-optimization/>

EPUB 下载：<https://github.com/chen3feng/llm-systems-optimization/releases/latest/download/llm-systems-optimization.epub>（每次合并到 main 后更新）

PDF 下载：[A4 普通版](https://github.com/chen3feng/llm-systems-optimization/releases/latest/download/llm-systems-optimization.pdf) ·
[6 寸 Kindle 大字版](https://github.com/chen3feng/llm-systems-optimization/releases/latest/download/llm-systems-optimization-kindle.pdf)

> 当前状态：初稿完成。Part I–Part X 共 50 章、前言与附录 A–I 均已完成初稿，待统一审校。

## 这本书回答什么问题

一个 LLM 从模型结构、数值表示，到 GPU Kernel、内存管理、并行执行和在线服务，
究竟是如何运行的？我们如何理解、实现、分析和优化这个系统？

## 读者对象

具备 C++、操作系统、GPU 或分布式系统背景，希望理解并实现 LLM 底层系统的工程师。

读完本书应能够：

- 从 Transformer 计算图推导计算量、内存占用和通信量。
- 阅读和实现关键 CUDA Kernel。
- 理解量化、Attention、KV Cache、MoE 等技术的底层机制。
- 实现一个具备动态批处理和 KV Cache 的推理引擎。
- 理解多 GPU 训练、推理中的并行策略和通信开销。
- 分析 vLLM、SGLang、llama.cpp 等系统的架构与关键实现。

## 内容边界

| 范围 | 处理方式 |
| --- | --- |
| LLM 模型与算法 | 讲到能够理解计算过程、系统影响和优化机制 |
| LLM 底层系统 | 本书核心，深入数据结构、代码、内存、Kernel、并行和调度 |
| LLM 应用开发 | RAG、Agent、Tool Calling 等只讨论其对 Runtime 和 Serving 的系统需求 |

不详细讲解 NLP 历史、深度学习入门和各种模型的产品使用方法，也不把 RLHF 写成
一本独立的强化学习教材。CUDA 是主要实践平台，但性能模型、调度、内存管理和
分布式执行的原理尽量保持平台无关。

## 全书结构

| Part | 主题 | 篇幅 |
| --- | --- | --- |
| Part I | Foundations: Understanding LLM Computation | 基础篇 · 约 75 页 |
| Part II | Model Architecture and Computation | 模型计算篇 · 约 65 页 |
| Part III | Numerical Optimization and Model Compression | 数值与压缩篇 · 约 65 页 |
| Part IV | GPU Computing and Kernel Optimization | 核心篇 · 约 125 页 |
| Part V | LLM Inference Systems | 核心篇 · 约 125 页 |
| Part VI | Distributed LLM Systems | 核心篇 · 约 100 页 |
| Part VII | LLM Training Systems | 训练系统篇 · 约 75 页 |
| Part VIII | Post-training and Model Quality from a Systems Perspective | 模型质量篇 · 约 55 页 |
| Part IX | Compiler, Runtime and Production Serving | 核心篇 · 约 75 页 |
| Part X | Advanced Topics and Research Directions | 前沿篇 · 约 35 页 |
| **合计** | **10 Parts / 50 Chapters** | **约 795 页** |

完整目录见 [SUMMARY.md](SUMMARY.md)。

## 配套代码

书中的 C++/CUDA 实践工程位于独立仓库：

- `llm-systems-lab`（Book Companion）— 从 CPU reference 到 CUDA kernel
  和分布式执行的完整实现。

书稿仓库只保存正文、图和写作规范，不混入代码。

## 写作约定

- 每章使用统一的 11 段式模板，见 [templates/chapter-template.md](templates/chapter-template.md)。
- 正文使用中文，技术术语保留英文；首次出现时给出中英文对照。
- 章节文件使用 `chNN-kebab-case-title.md` 命名，编号与全书目录一致。
- 图片集中存放在 [figures/](figures/)，正文使用相对路径引用。

## 构建 HTML、EPUB 与 PDF

需要 [mdBook](https://rust-lang.github.io/mdBook/)、[mdbook-katex](https://github.com/lzanini/mdbook-katex) 和 [mdbook-epub](https://github.com/Michael-F-Bryan/mdbook-epub)：

```bash
brew install mdbook
cargo install mdbook-katex
cargo install mdbook-epub
```

```bash
./scripts/render.sh
open .build/html/index.html
```

脚本把 `book/`、`figures/` 和 `SUMMARY.md` 复制到 `.build/src/`，去掉章节的 YAML front matter，再由 mdbook-katex 在 Markdown 解析前把公式渲染成 HTML（KaTeX）。产物有两个：

- `.build/html/`：网页版，KaTeX 的样式和字体放在 `theme/katex/`，不依赖 CDN；
- `.build/llm-systems-optimization.epub`：电子书版，公式用 MathML 渲染，适配不加载 KaTeX 资源的阅读器。

`.build/` 不入库；正文改动后重跑脚本。

### 导出 PDF

需要本机安装 Chrome/Chromium：

```bash
python3 -I scripts/make_pdf.py
```

脚本把 mdBook 生成的整本 `print.html` 注入打印样式后交给无头 Chrome，输出两个版本：

- `.build/llm-systems-optimization.pdf`：A4 普通版；
- `.build/llm-systems-optimization-kindle.pdf`：6 寸 Kindle 大字版（91mm × 122mm，正文 12.5pt），`text` 代码块会按最大列宽自动缩放，避免 ASCII 示意图折行错位。

PDF 使用系统字体和本地 KaTeX 资源渲染，不需要联网。

构建后可以检查公式：

```bash
python3 -I scripts/check_math.py
```

它逐页比对源文件里的公式数量与渲染结果，并报告 KaTeX 解析错误。

## 仓库结构

```text
book/          书稿正文：前置内容、10 个 Part、50 章、附录
figures/       全书插图
assets/        封面等非正文资源
templates/     章节与 Part 模板
theme/katex/   KaTeX 样式与字体（构建用）
scripts/       HTML、EPUB、PDF 构建脚本
book.toml      mdBook 配置
SUMMARY.md     全书导航
```

## 版权

Copyright © 2026 chen3feng.

本书正文、插图及仓库中的其他原创内容采用 [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
（知识共享“署名—非商业性使用—禁止演绎”4.0 国际）许可协议发布：

- **署名（BY）**：复制或分发时须注明作者 chen3feng 和本书名称，并附上许可协议链接；
- **非商业性使用（NC）**：不得将本书内容用于商业目的；
- **禁止演绎（ND）**：不得修改、翻译或改编本书内容，也不得基于本书创作衍生作品。

完整法律文本见 [LICENSE](LICENSE)，书内的单页版权说明见 [版权说明](book/00-front-matter/03-copyright.md)。书中的代码示例不受上述条件限制，可自由使用、修改和分发。如需商业使用、翻译或改编授权，请联系作者另行协商。
