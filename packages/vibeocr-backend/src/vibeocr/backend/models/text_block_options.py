# src/vibeocr/models/text_block_options.py
"""文本块处理选项数据模型

控制单识别标签页中 OCR 文本块的拼接排版策略，与 OCR 引擎 / 管道无关，
因此独立于 OCROptions（不参与 get_pipeline_supported_options 矩阵）。

由 TextBlockProcessor 在识别完成后消费，重写 OCRResult.raw_text。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# 换行模式常量
LINE_MODE_KEEP = "keep"  # 保留原样：每个文本块占一行
LINE_MODE_MERGE = "merge"  # 合并成一段：去除断行
LINE_MODE_SMART = "smart"  # 智能分段：按垂直间距判定段落边界

_VALID_LINE_MODES = frozenset({LINE_MODE_KEEP, LINE_MODE_MERGE, LINE_MODE_SMART})


@dataclass
class TextBlockOptions:
    """文本块处理选项

    Attributes:
        line_mode: 换行模式 keep / merge / smart。
        block_join_space: 合并相关模式下，块之间是否插入半角空格。
        chinese_indent: smart/merge 模式下，每段首行是否加两个全角空格缩进。
        drop_blank_blocks: 是否过滤掉 text.strip() 为空的文本块。
    """

    line_mode: str = LINE_MODE_MERGE
    block_join_space: bool = False
    chinese_indent: bool = False
    drop_blank_blocks: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_mode": self.line_mode,
            "block_join_space": self.block_join_space,
            "chinese_indent": self.chinese_indent,
            "drop_blank_blocks": self.drop_blank_blocks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TextBlockOptions:
        if not data:
            return cls()
        line_mode = data.get("line_mode", LINE_MODE_MERGE)
        if line_mode not in _VALID_LINE_MODES:
            line_mode = LINE_MODE_MERGE
        return cls(
            line_mode=line_mode,
            block_join_space=bool(data.get("block_join_space", False)),
            chinese_indent=bool(data.get("chinese_indent", False)),
            drop_blank_blocks=bool(data.get("drop_blank_blocks", True)),
        )
