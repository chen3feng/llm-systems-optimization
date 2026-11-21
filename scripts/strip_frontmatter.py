#!/usr/bin/env python3
"""删除章节开头的 YAML front matter。

mdBook 不识别书稿使用的 `chapter:` / `part:` 等 front matter，会把整块当正文
渲染，所以在装配 mdBook 源目录时统一去掉。
"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
count = 0
for path in root.rglob('*.md'):
    lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
    if lines and lines[0].strip() == '---':
        for i in range(1, len(lines)):
            if lines[i].strip() == '---':
                path.write_text(''.join(lines[i + 1:]).lstrip('\n'), encoding='utf-8')
                count += 1
                break
print(f'stripped front matter: {count} files')
