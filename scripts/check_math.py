#!/usr/bin/env python3
"""核对公式渲染：源文件里的每段公式都应渲染成一个 KaTeX 公式，且没有解析错误。

用法：
    ./scripts/render.sh
    python3 -I scripts/check_math.py
"""
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / 'book'
OUT = REPO / '.build' / 'html' / 'book'

FENCE = re.compile(r'^ {0,3}(`{3,}|~{3,})[^\n]*\n.*?^ {0,3}\1[ \t]*$', re.M | re.S)
CODE_SPAN = re.compile(r'(`+)(?!`).*?(?<!`)\1(?!`)', re.S)
MATH = re.compile(r'\$\$.+?\$\$|\$(?!\$)[^$\n]+?\$', re.S)


def count_math(text):
    """统计围栏代码块和行内代码之外的公式数量。"""
    chunks, pos = [], 0
    for match in FENCE.finditer(text):
        chunks.append(text[pos:match.start()])
        pos = match.end()
    chunks.append(text[pos:])
    count = 0
    for chunk in chunks:
        cpos = 0
        pieces = []
        for match in CODE_SPAN.finditer(chunk):
            pieces.append(chunk[cpos:match.start()])
            cpos = match.end()
        pieces.append(chunk[cpos:])
        for piece in pieces:
            count += len(MATH.findall(piece))
    return count


def main():
    problems = []
    pages = formulas = 0
    for md in sorted(SRC.rglob('*.md')):
        rel = md.relative_to(SRC)
        name = 'index.html' if rel.name == 'README.md' else rel.with_suffix('.html').name
        page = OUT / rel.parent / name
        if not page.exists():
            problems.append(f'{rel}: 缺少对应页面')
            continue
        expected = count_math(md.read_text(encoding='utf-8'))
        page_text = page.read_text(encoding='utf-8')
        rendered = page_text.count('class="katex"')
        errors = page_text.count('katex-error')
        pages += 1
        formulas += expected
        if rendered != expected:
            problems.append(f'{rel}: 源 {expected} 段，渲染 {rendered} 个')
        if errors:
            problems.append(f'{rel}: {errors} 个 KaTeX 解析错误')
    print(f'检查 {pages} 页、{formulas} 段公式，问题 {len(problems)} 个')
    for line in problems:
        print(f'  {line}')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
