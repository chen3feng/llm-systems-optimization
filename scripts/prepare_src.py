#!/usr/bin/env python3
"""装配 mdBook 源目录前的预处理：删除章节开头的 YAML front matter。

mdBook 不识别书稿使用的 `chapter:` / `part:` 等 front matter，会把整块当正文
渲染，所以在装配 mdBook 源目录时统一去掉。

公式不需要在这里处理：mdbook-katex 预处理器在 Markdown 解析前就把公式渲染成
HTML，公式里的 `_`、`\\` 等不会再被 Markdown 语法干扰。
"""
import pathlib
import sys


def strip_front_matter(text):
    """去掉文件开头的 `--- ... ---` 块，返回 (新文本, 是否命中)。"""
    lines = text.splitlines(keepends=True)
    if lines and lines[0].strip() == '---':
        for i in range(1, len(lines)):
            if lines[i].strip() == '---':
                return ''.join(lines[i + 1:]).lstrip('\n'), True
    return text, False


def main(root):
    root = pathlib.Path(root)
    stripped = 0
    for path in root.rglob('*.md'):
        text, had = strip_front_matter(path.read_text(encoding='utf-8'))
        stripped += had
        path.write_text(text, encoding='utf-8')
    print(f'stripped front matter: {stripped} files')


if __name__ == '__main__':
    main(sys.argv[1])
