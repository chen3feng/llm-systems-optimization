#!/usr/bin/env python3
"""用无头 Chrome 把 mdBook 的 print.html 打印成 PDF。

用法：
    ./scripts/render.sh
    python3 -I scripts/make_pdf.py

产物：
    .build/llm-systems-optimization.pdf          A4 普通版
    .build/llm-systems-optimization-kindle.pdf   6 寸 Kindle 大字版（91mm × 122mm）

print.html 是 mdBook 自动生成的整本合并页，已包含全部章节、插图和 KaTeX
公式。脚本向其中注入打印样式后交给 Chrome 渲染，不依赖网络。封面由
assets/cover/cover.png 单独打印成一页，再用 poppler 的 pdfunite 合并到正文前。
"""
import html as html_lib
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata

REPO = pathlib.Path(__file__).resolve().parent.parent
BUILD = REPO / '.build'
HTML_DIR = BUILD / 'html'
PRINT_HTML = HTML_DIR / 'print.html'
COVER_IMAGE = REPO / 'assets' / 'cover' / 'cover.png'

CHROME_CANDIDATES = [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
    shutil.which('google-chrome'),
    shutil.which('google-chrome-stable'),
    shutil.which('chromium'),
    shutil.which('chromium-browser'),
    shutil.which('chrome'),
]

COMMON_CSS = """
html, body { background: transparent !important; }
body {
    font-family: "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC",
                 "Source Han Sans SC", "Microsoft YaHei", sans-serif;
    color: #111;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
}
#mdbook-content { max-width: none; margin: 0; padding: 0; }
#mdbook-content p { text-align: justify; }
#mdbook-content img { max-width: 100%; height: auto; page-break-inside: avoid; }
#mdbook-content pre {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    background: #f6f8fa;
    border-radius: 4px;
}
#mdbook-content code {
    font-family: "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
}
#mdbook-content table { width: 100%; border-collapse: collapse; }
#mdbook-content th, #mdbook-content td {
    border: 1px solid #ccc;
    vertical-align: top;
    overflow-wrap: break-word;
}
#mdbook-content blockquote,
#mdbook-content .katex-display { page-break-inside: avoid; }
#mdbook-content h1, #mdbook-content h2, #mdbook-content h3,
#mdbook-content h4, #mdbook-content h5, #mdbook-content h6 {
    line-height: 1.35;
    page-break-after: avoid;
}
a { color: #1a5fb4; text-decoration: none; }
"""

# 封面单独打印成一页，再与正文 PDF 合并。Chrome 的命名页（@page cover）
# 会留下约 2mm 白边，所以封面页直接用整页画布背景绘制。
COVER_HTML = """<!doctype html><meta charset="utf-8"><style>
@page {{ size: {width}mm {height}mm; margin: 0; }}
html, body {{
    margin: 0;
    padding: 0;
    width: {width}mm;
    height: {height}mm;
    background: #faf8f2 url("{uri}") center / contain no-repeat;
}}
</style>"""

A4_CSS = """
@page { size: A4; margin: 20mm 18mm; }
#mdbook-content { font-size: 10.5pt; line-height: 1.75; }
#mdbook-content p { margin: 0.6em 0; }
#mdbook-content h1 { font-size: 21pt; margin: 0 0 0.9em; }
#mdbook-content h2 { font-size: 15.5pt; margin: 1.5em 0 0.6em; }
#mdbook-content h3 { font-size: 13pt; margin: 1.2em 0 0.5em; }
#mdbook-content h4 { font-size: 11.5pt; margin: 1em 0 0.4em; }
#mdbook-content h5, #mdbook-content h6 { font-size: 11pt; }
#mdbook-content pre { font-size: 8pt; line-height: 1.5; padding: 8px 10px; }
#mdbook-content table { font-size: 8.5pt; line-height: 1.5; }
#mdbook-content th, #mdbook-content td { padding: 4px 6px; }
#mdbook-content .katex-display { margin: 0.9em 0; }
"""

KINDLE_CSS = """
@page { size: 91mm 122mm; margin: 3.5mm 3.5mm 4mm; }
#mdbook-content { font-size: 12.5pt; line-height: 1.65; }
/* 窄栏两端对齐会出现大片空白，左对齐更耐读 */
#mdbook-content p { margin: 0.5em 0; text-align: left; }
#mdbook-content h1 { font-size: 18pt; margin: 0 0 0.7em; }
#mdbook-content h2 { font-size: 14.5pt; margin: 1.1em 0 0.45em; }
#mdbook-content h3 { font-size: 13pt; margin: 0.9em 0 0.4em; }
#mdbook-content h4 { font-size: 12pt; }
#mdbook-content h5, #mdbook-content h6 { font-size: 11.5pt; }
#mdbook-content pre { font-size: 6.5pt; line-height: 1.35; padding: 4px 5px; }
#mdbook-content table { font-size: 7.5pt; line-height: 1.35; }
#mdbook-content th, #mdbook-content td { padding: 2px 3px; }
#mdbook-content .katex { font-size: 0.95em; }
#mdbook-content .katex-display { font-size: 0.85em; margin: 0.6em 0; }
"""


def find_chrome():
    """返回本机可用的 Chrome/Chromium 路径，找不到则返回 None。"""
    for candidate in CHROME_CANDIDATES:
        if candidate and pathlib.Path(candidate).exists():
            return candidate
    return None


def write_print_html(out_path, css, fit_text_blocks=False):
    """复制 print.html，注入打印样式；base 让相对路径仍指向构建目录。

    fit_text_blocks 为 True 时，按每块 text 代码的最大列宽缩小字号，
    避免窄页（Kindle 版）下 ASCII 示意图折行错位。
    """
    html = PRINT_HTML.read_text(encoding='utf-8')
    base = HTML_DIR.as_uri() + '/'
    head = f'<base href="{base}">\n<style>{COMMON_CSS}\n{css}</style>\n'
    html = html.replace('<head>', '<head>\n' + head, 1)
    if fit_text_blocks:
        html = fit_text_blocks_to_page(html)
    out_path.write_text(html, encoding='utf-8')


TEXT_CODE_RE = re.compile(r'<pre><code class="language-text">(.*?)</code></pre>', re.S)


def display_cells(line):
    """按等宽字符位计算显示宽度，全角字符算两格。"""
    return sum(2 if unicodedata.east_asian_width(ch) in 'WF' else 1 for ch in line)


def fit_text_blocks_to_page(html, width_pt=216.0):
    """把超宽的 text 代码块缩到页面宽度内，保持 ASCII 图对齐。"""
    char_width_em = 0.68  # 等宽字体的实际字宽，留一点余量

    def repl(match):
        body = match.group(1)
        plain = html_lib.unescape(re.sub(r'<[^>]+>', '', body))
        cells = max((display_cells(line) for line in plain.splitlines()), default=0)
        if cells <= 0:
            return match.group(0)
        size = min(6.5, width_pt / (cells * char_width_em))
        if size >= 6.4:
            return match.group(0)
        size = max(size, 4.4)
        return (f'<pre><code class="language-text" style="font-size:{size:.2f}pt">'
                f'{body}</code></pre>')

    return TEXT_CODE_RE.sub(repl, html)


def print_pdf(chrome, src, out, profile):
    """调用无头 Chrome 把 src 打印成 out。"""
    cmd = [
        chrome,
        '--headless=new',
        '--disable-gpu',
        '--no-pdf-header-footer',
        '--allow-file-access-from-files',
        '--no-first-run',
        '--no-default-browser-check',
        f'--user-data-dir={profile}',
        f'--print-to-pdf={out}',
        src.as_uri(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(f'Chrome 打印失败：{out}')


def print_cover(chrome, out, width_mm, height_mm, profile, tmp):
    """打印一页整页出血的封面。"""
    if not COVER_IMAGE.exists():
        raise SystemExit(f'找不到封面：{COVER_IMAGE.relative_to(REPO)}')
    src = tmp / f'cover-{width_mm}x{height_mm}.html'
    src.write_text(
        COVER_HTML.format(width=width_mm, height=height_mm,
                          uri=COVER_IMAGE.as_uri()),
        encoding='utf-8')
    print_pdf(chrome, src, out, profile)


def merge_pdfs(cover, content, out):
    """用 poppler 的 pdfunite 把封面页拼到正文前面。"""
    pdfunite = shutil.which('pdfunite')
    if not pdfunite:
        raise SystemExit('找不到 pdfunite（poppler-utils），'
                         '请先安装：brew install poppler 或 apt-get install poppler-utils')
    result = subprocess.run(
        [pdfunite, str(cover), str(content), str(out)],
        capture_output=True, text=True, check=False)
    if result.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(f'合并 PDF 失败：{out}')


def main():
    """生成 A4 普通版和 6 寸 Kindle 大字版两个 PDF。"""
    if not PRINT_HTML.exists():
        raise SystemExit(f'找不到 {PRINT_HTML.relative_to(REPO)}，请先运行 ./scripts/render.sh')
    chrome = find_chrome()
    if not chrome:
        raise SystemExit('找不到 Chrome/Chromium，请先安装 Google Chrome')

    jobs = [
        ('A4 普通版', A4_CSS, (210, 297),
         BUILD / 'llm-systems-optimization.pdf', False),
        ('6 寸 Kindle 大字版', KINDLE_CSS, (91, 122),
         BUILD / 'llm-systems-optimization-kindle.pdf', True),
    ]

    with tempfile.TemporaryDirectory(prefix='llm-book-pdf-') as tmp:
        tmp = pathlib.Path(tmp)
        profile = tmp / 'chrome-profile'
        for label, css, (cover_w, cover_h), out, fit_text in jobs:
            src = tmp / (out.stem + '.html')
            content_pdf = tmp / (out.stem + '-content.pdf')
            cover_pdf = tmp / (out.stem + '-cover.pdf')
            write_print_html(src, css, fit_text_blocks=fit_text)
            print(f'打印{label}：{out.name}')
            print_pdf(chrome, src, content_pdf, profile)
            print_cover(chrome, cover_pdf, cover_w, cover_h, profile, tmp)
            merge_pdfs(cover_pdf, content_pdf, out)
            print(f'  {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
