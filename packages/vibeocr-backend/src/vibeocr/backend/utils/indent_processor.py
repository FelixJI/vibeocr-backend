"""缩进处理器模块"""

import re
from dataclasses import dataclass


@dataclass
class IndentConfig:
    """缩进配置"""

    chinese_indent: str = "2em"
    chinese_threshold: float = 0.05


class IndentProcessor:
    """处理文本缩进的处理器"""

    CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df]")

    def __init__(self, config: IndentConfig | None = None):
        self.config = config or IndentConfig()

    def is_chinese_text(self, text: str) -> bool:
        """检测文本是否主要为中文

        Args:
            text: 待检测文本

        Returns:
            如果中文字符占比 >= chinese_threshold 则返回 True
        """
        if not text.strip():
            return False
        chinese_chars = len(self.CHINESE_PATTERN.findall(text))
        total_chars = len(text.strip())
        return chinese_chars / total_chars >= self.config.chinese_threshold

    def process_markdown(self, markdown_text: str) -> str:
        """处理 Markdown 文本，为中文段落添加标记

        Args:
            markdown_text: 原始 Markdown 文本

        Returns:
            处理后的 Markdown 文本，中文段落被包装在 zh-paragraph div 中
        """
        if not markdown_text:
            return ""

        # 如果内容包含大量 HTML 块级标签（来自 VLM 管道的输出），跳过处理
        html_block_count = len(
            re.findall(r"<(div|table|tr|td|th|p|span)\b", markdown_text, re.IGNORECASE)
        )
        if html_block_count >= 3:
            return markdown_text

        # 用于检测特殊元素的正则
        table_line_pattern = re.compile(r"^\|.*\|$")
        list_line_pattern = re.compile(r"^[\*\-\+]\s|^\d+\.\s")

        # 先按代码块分割，保留代码块内容不变
        code_block_pattern = re.compile(r"(```.*?```)", re.DOTALL)
        parts = code_block_pattern.split(markdown_text)

        processed_parts = []
        for part in parts:
            # 如果是代码块，直接保留
            if part.startswith("```") and part.endswith("```"):
                processed_parts.append(part)
                continue

            # 对非代码块部分，按双换行分割段落
            paragraphs = part.split("\n\n")
            processed_paras = []

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # 检查段落是否包含表格行或列表行
                lines = para.split("\n")
                has_table_or_list = any(
                    table_line_pattern.match(line.strip())
                    or list_line_pattern.match(line.strip())
                    for line in lines
                )

                # 跳过表格和列表段落
                if has_table_or_list:
                    processed_paras.append(para)
                    continue

                # 跳过单行代码块标记
                if para.startswith("```"):
                    processed_paras.append(para)
                    continue

                if self.is_chinese_text(para):
                    processed_paras.append(f'<div class="zh-paragraph">{para}</div>')
                else:
                    processed_paras.append(para)

            if processed_paras:
                processed_parts.append("\n\n".join(processed_paras))

        return "\n\n".join(processed_parts)
