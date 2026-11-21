#!/usr/bin/env bash
# 把书稿装配成 mdBook 源目录并渲染为 HTML，产物在 .build/html/。
# 正文改动后重新执行本脚本；mdbook serve 监听的是装配后的副本，不会感知仓库里的修改。
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v mdbook >/dev/null; then
  echo "需要先安装 mdBook：brew install mdbook" >&2
  exit 1
fi

rm -rf .build/src
mkdir -p .build/src
cp -R book figures .build/src/
cp SUMMARY.md LICENSE .build/src/

python3 -I scripts/prepare_src.py .build/src
mdbook build

echo "HTML: $(pwd)/.build/html/index.html"
