"""Markdown 转 HTML 工具"""

import logging

from .indent_processor import IndentProcessor

_logger = logging.getLogger(__name__)

# HTML 样式：用于 QTextEdit 显示
HTML_STYLE = """
<style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 14px;
        line-height: 1.6;
        color: #333;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 10px 0;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    th {
        background-color: #f5f5f5;
        font-weight: bold;
    }
    tr:nth-child(even) {
        background-color: #fafafa;
    }
    code {
        background-color: #f4f4f4;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: "Consolas", "Monaco", monospace;
        font-size: 13px;
    }
    pre {
        background-color: #f4f4f4;
        padding: 10px;
        border-radius: 5px;
        overflow-x: auto;
    }
    pre code {
        background-color: transparent;
        padding: 0;
    }
    /* LaTeX 公式样式 */
    .latex-formula {
        background-color: #f8f9fa;
        padding: 10px 15px;
        border-radius: 5px;
        font-family: "Consolas", "Monaco", monospace;
        font-size: 13px;
        margin: 10px 0;
        border-left: 3px solid #0078d4;
    }
    .latex-inline {
        background-color: #f8f9fa;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: "Consolas", "Monaco", monospace;
        font-size: 13px;
    }
    /* 中文段落首行缩进 */
    .zh-paragraph {
        text-indent: 2em;
    }
    .zh-paragraph p:first-child {
        text-indent: 2em;
    }
    /* 列表嵌套缩进 */
    ul, ol {
        margin-left: 1.5em;
        padding-left: 0.5em;
    }
    ul ul, ol ol, ul ol, ol ul {
        margin-left: 1.2em;
    }
    li {
        margin: 0.2em 0;
    }
    li p {
        text-indent: 0;
    }
</style>
"""


def markdown_to_html(
    markdown_text: str,
    *,
    include_style: bool = True,
    extensions: list | None = None,
) -> str:
    """将 Markdown 转换为 HTML

    Args:
        markdown_text: Markdown 格式的文本
        include_style: 是否包含 CSS 样式
        extensions: 自定义扩展列表

    Returns:
        HTML 格式的文本
    """
    if not markdown_text:
        return ""

    # markdown 仅由 OCR/MinerU 子进程实际调用，主进程只用本模块的 HTML_STYLE
    # 字符串常量。此处延迟 import，使主进程（PyInstaller 冻结 exe，已排除
    # markdown）加载 markdown_converter 时不触发 ImportError。
    import markdown
    from markdown.extensions.tables import TableExtension

    # 预处理：将 LaTeX 公式转换为 HTML span 标签
    processed_text = _process_latex_formulas(markdown_text)

    # 预处理：处理中文段落缩进
    indent_processor = IndentProcessor()
    processed_text = indent_processor.process_markdown(processed_text)

    # 默认扩展：支持表格等
    if extensions is None:
        extensions = [
            TableExtension(),
            "fenced_code",
            "nl2br",  # 换行转 <br>
            "sane_lists",  # 更严格的列表处理
        ]

    try:
        # 转换 Markdown 为 HTML
        html_body = markdown.markdown(
            processed_text,
            extensions=extensions,
        )

        # 组装完整 HTML
        if include_style:
            return f"{HTML_STYLE}<body>{html_body}</body>"
        return html_body

    except Exception as e:
        _logger.warning(f"Markdown 转换失败: {e}")
        # 转换失败时返回原文本（转义 HTML）
        escaped = (
            markdown_text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        if include_style:
            return f"{HTML_STYLE}<body><pre>{escaped}</pre></body>"
        return f"<pre>{escaped}</pre>"


def _process_latex_formulas(text: str) -> str:
    """处理 LaTeX 公式，将其转换为带样式的 HTML

    将 $$...$$ 转换为 <div class="latex-formula">...</div>
    将 $...$ 转换为 <span class="latex-inline">...</span>
    """
    import re

    # 处理块级公式 $$...$$
    # 注意：Markdown 表格中可能包含 $ 符号，需要小心处理
    def replace_block_formula(match):
        formula = match.group(1).strip()
        # 转义 HTML 特殊字符
        formula = (
            formula.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        return f'<div class="latex-formula">{formula}</div>'

    # 处理行内公式 $...$
    # 只匹配非表格行的单行公式
    def replace_inline_formula(match):
        formula = match.group(1).strip()
        # 转义 HTML 特殊字符
        formula = (
            formula.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        return f'<span class="latex-inline">{formula}</span>'

    # 先处理块级公式 $$...$$（多行）
    result = re.sub(r"\$\$(.*?)\$\$", replace_block_formula, text, flags=re.DOTALL)

    # 再处理行内公式 $...$
    # 使用负向前瞻和后顾，避免匹配表格分隔符 |---|
    # 只匹配不包含换行的简单公式
    return re.sub(
        r"(?<![|\\])\$([^\$\n]+?)\$(?![|\\])",
        replace_inline_formula,
        result,
    )


def extract_plain_text(html_text: str) -> str:
    """从 HTML 中提取纯文本

    Args:
        html_text: HTML 格式的文本

    Returns:
        纯文本
    """
    if not html_text:
        return ""

    import re

    # 移除 style 标签
    text = re.sub(r"<style.*?</style>", "", html_text, flags=re.DOTALL | re.IGNORECASE)

    # 将块级元素替换为换行
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</div>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</tr>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</td>", "\t", text, flags=re.IGNORECASE)
    text = re.sub(r"</th>", "\t", text, flags=re.IGNORECASE)

    # 移除所有 HTML 标签
    text = re.sub(r"<[^>]+>", "", text)

    # 解码 HTML 实体
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&nbsp;", " ")
    text = text.replace("&quot;", '"')

    # 清理多余空白
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)
