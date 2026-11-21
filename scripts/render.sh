#!/usr/bin/env bash
# 把书稿装配成 mdBook 源目录，渲染为 HTML 和 EPUB。
# 产物：.build/html/、.build/llm-systems-optimization.epub。
# 正文改动后重新执行本脚本；mdbook serve 监听的是装配后的副本，不会感知仓库里的修改。
set -euo pipefail

cd "$(dirname "$0")/.."

# mdbook-katex / mdbook-epub 通常由 cargo 安装到这里
export PATH="$HOME/.cargo/bin:$PATH"

if ! command -v mdbook >/dev/null; then
  echo "需要先安装 mdBook：brew install mdbook" >&2
  exit 1
fi
if ! command -v mdbook-katex >/dev/null; then
  echo "需要先安装 mdbook-katex：cargo install mdbook-katex" >&2
  exit 1
fi
if ! command -v mdbook-epub >/dev/null; then
  echo "需要先安装 mdbook-epub：cargo install mdbook-epub" >&2
  exit 1
fi

rm -rf .build/src
mkdir -p .build/src/theme
cp -R book figures .build/src/
cp -R theme/katex .build/src/theme/
cp SUMMARY.md LICENSE .build/src/

python3 -I scripts/prepare_src.py .build/src
mdbook build

echo "HTML: $(pwd)/.build/html/index.html"

# EPUB 用 MathML 公式：电子书阅读器不加载 KaTeX 的 CSS 和字体
rm -rf .build/epub
mkdir -p .build/epub
cat > .build/epub/book.toml <<'TOML'
[book]
title = "LLM Systems Optimization"
authors = ["chen3feng"]
description = "Algorithms, GPU Kernels, Runtime and Distributed Execution"
language = "zh-CN"
src = "../src"

[build]
build-dir = "out"

[preprocessor.katex]
after = ["links"]
output = "mathml"
no-css = true

[output.epub]
TOML

mdbook build .build/epub
cp ".build/epub/out/LLM Systems Optimization.epub" .build/llm-systems-optimization.epub

echo "EPUB: $(pwd)/.build/llm-systems-optimization.epub"
