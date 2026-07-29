"""导出设置数据类"""

from dataclasses import dataclass


@dataclass
class ExportSettings:
    """导出设置

    Attributes:
        format: 导出格式 (markdown, html, docx, xlsx, txt)
        location_mode: 导出位置模式 (same_as_source, custom)
        custom_directory: 自定义导出目录
        last_custom_directory: 上次使用的自定义目录（用于记忆）
    """

    format: str = "markdown"
    location_mode: str = "same_as_source"
    custom_directory: str = ""
    last_custom_directory: str = ""

    FORMAT_EXTENSIONS = {
        "markdown": ".md",
        "html": ".html",
        "docx": ".docx",
        "xlsx": ".xlsx",
        "txt": ".txt",
    }

    FORMAT_LABELS = {
        "markdown": "Markdown 文档 (.md)",
        "html": "HTML 网页 (.html)",
        "docx": "Word 文档 (.docx)",
        "xlsx": "Excel 表格 (.xlsx)",
        "txt": "纯文本 (.txt)",
    }

    def get_extension(self) -> str:
        return self.FORMAT_EXTENSIONS.get(self.format, ".txt")

    def get_label(self) -> str:
        return self.FORMAT_LABELS.get(self.format, "未知格式")
