#!/usr/bin/env python3
"""装配 mdBook 源目录前的预处理。

1. 删除章节开头的 YAML front matter——mdBook 不识别它，会把整块当正文渲染。
2. 保护数学公式——mdBook 先做 Markdown 解析再交给 MathJax，公式里的 `_` 会被
   解析成强调，`\\`、`\\,` 等反斜杠转义会被吞掉，MathJax 收到的是残缺的 TeX。
   这里在解析前把公式里的特殊字符转义，Markdown 解析后它们会还原成原始字符。
"""
import pathlib
import re
import sys

FENCE = re.compile(r'^ {0,3}(`{3,}|~{3,})[^\n]*\n.*?^ {0,3}\1[ \t]*$', re.M | re.S)
CODE_SPAN = re.compile(r'(`+)(?!`).*?(?<!`)\1(?!`)', re.S)
MATH = re.compile(r'\$\$.+?\$\$|\$(?!\$)[^$\n]+?\$', re.S)
SPECIAL = '_*`[]<>|#'


def protect_math(text):
    """把公式里的 Markdown 特殊字符转义；解析后 Markdown 会把反斜杠去掉。"""
    def repl(match):
        tex = match.group(0).replace('\\', '\\\\')
        for ch in SPECIAL:
            tex = tex.replace(ch, '\\' + ch)
        return tex

    return MATH.sub(repl, text)


def protect_segment(text):
    """处理一段非代码块文本，跳过其中的行内代码。"""
    out, pos = [], 0
    for match in CODE_SPAN.finditer(text):
        out.append(protect_math(text[pos:match.start()]))
        out.append(match.group(0))
        pos = match.end()
    out.append(protect_math(text[pos:]))
    return ''.join(out)


def transform(text):
    """按围栏代码块切分，只处理代码块以外的内容。"""
    out, pos = [], 0
    for match in FENCE.finditer(text):
        out.append(protect_segment(text[pos:match.start()]))
        out.append(match.group(0))
        pos = match.end()
    out.append(protect_segment(text[pos:]))
    return ''.join(out)


def strip_front_matter(text):
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
        path.write_text(transform(text), encoding='utf-8')
    print(f'stripped front matter: {stripped} files')


if __name__ == '__main__':
    main(sys.argv[1])
