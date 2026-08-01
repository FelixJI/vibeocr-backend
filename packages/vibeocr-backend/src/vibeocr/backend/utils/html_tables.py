"""HTML 表格规整化与转换工具（UI 层展示逻辑）。

纯函数、无 Qt 依赖、无 OCR 引擎依赖。将 PaddleX/MinerU 输出的表格 HTML
规整化为纯净的 ``<table>``（剥离 inline style、补齐空单元格），并支持
HTML → Markdown 表格转换。按 ADR §5.2，UI 负责「展示结果」，此模块属于
UI/工具层。

从 ``vibeocr.backend.services.ocr_service`` 迁移（Phase 3 单图 OCR 切片）。
"""

from __future__ import annotations

import html as _html
import re as _re

_RE_TABLE = _re.compile(r"(<table\b.*?</table>)", _re.DOTALL | _re.IGNORECASE)
_RE_TR = _re.compile(r"<tr[^>]*>(.*?)</tr>", _re.DOTALL | _re.IGNORECASE)
_RE_TD = _re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", _re.DOTALL | _re.IGNORECASE)
# 单格匹配：捕获标签名（td/th）以区分表头、属性段（含 colspan/rowspan）、单元格内容
_RE_CELL = _re.compile(r"<(td|th)([^>]*)>(.*?)</\1>", _re.DOTALL | _re.IGNORECASE)


def extract_table_html(html_str: str) -> str:
    """从 HTML 字符串中提取第一个 ``<table>`` 块。"""
    match = _RE_TABLE.search(html_str)
    return match.group(1) if match else html_str


def html_table_to_markdown(html: str) -> str:
    """将 ``<table>`` HTML 转换为 GFM Markdown 表格。"""
    rows: list[list[str]] = []
    for tr_match in _RE_TR.finditer(html):
        cells = []
        for td in _RE_TD.finditer(tr_match.group(1)):
            # 复用 _cell_text：剥标签（<br>→\n）、unescape 实体、规整空白。
            text = _cell_text(td.group(1))
            # GFM 表格单元格内换行需表示为 <br>（python-markdown 的
            # TableExtension 会吞掉单元格内的 \n），故把 \n 转回 <br>。
            # pipe 是 markdown 表格分隔符，必须转义。
            text = text.replace("\n", "<br>").replace("|", "\\|")
            cells.append(text)
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    max_cols = max(len(r) for r in rows)
    for r in rows:
        r.extend("" for _ in range(max_cols - len(r)))
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join("---" for _ in range(max_cols)) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(part for part in (header, sep, body) if part)


def _cell_text(inner: str) -> str:
    """剥离单元格内容里的 HTML 标签，规整空白并解码常见实体。"""
    # <br> / <br/> → 换行
    text = _re.sub(r"<br\s*/?>", "\n", inner, flags=_re.IGNORECASE)
    # 其余标签直接去掉
    text = _re.sub(r"<[^>]+>", "", text)
    text = _html.unescape(text)
    # 行内空白规整，但保留显式换行
    lines = [_re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _span_value(attrs: str, name: str) -> int:
    match = _re.search(rf"\b{name}\s*=\s*['\"]?([1-9]\d*)", attrs, flags=_re.IGNORECASE)
    return min(int(match.group(1)), 1000) if match else 1


def _layout_rows(
    rows: list[list[tuple[str, str, int, int]]], target_width: int | None = None
) -> tuple[list[str], int]:
    """按 rowspan/colspan 占位布局并生成已清洗的行 HTML。"""
    active: list[int] = []
    rendered: list[str] = []
    max_width = 0
    for row in rows:
        cells_html: list[str] = []
        col = 0
        for tag, text, rowspan, colspan in row:
            while col < len(active) and active[col] > 0:
                col += 1
            needed = col + colspan
            if needed > len(active):
                active.extend([0] * (needed - len(active)))
            for occupied_col in range(col, needed):
                active[occupied_col] = max(active[occupied_col], rowspan)
            attrs = ""
            if rowspan > 1:
                attrs += f' rowspan="{rowspan}"'
            if colspan > 1:
                attrs += f' colspan="{colspan}"'
            safe = _html.escape(text).replace("\n", "<br>")
            cells_html.append(f"<{tag}{attrs}>{safe}</{tag}>")
            col = needed

        width = max(len(active), col)
        max_width = max(max_width, width)
        if target_width is not None:
            if len(active) < target_width:
                active.extend([0] * (target_width - len(active)))
            for pad_col in range(col, target_width):
                if active[pad_col] == 0:
                    cells_html.append("<td></td>")
        rendered.append(f"<tr>{''.join(cells_html)}</tr>")
        active = [max(0, remaining - 1) for remaining in active]
    return rendered, max_width


def normalize_table_html(html: str) -> str:
    """规整化表格 HTML：剥离 inline style、补齐空单元格、统一标签。

    解决两类问题：
    1. **复制带底纹**：PaddleX pred_html 的单元格常带 ``style="background:..."``
       inline 属性，渲染→原生 Ctrl+C 会把样式带进剪贴板。这里剥离展示
       属性，仅保留决定表格结构的数字 ``rowspan``/``colspan``。
    2. **空单元格错位**：若某行单元格数不足（空 ``<td>`` 缺失或 HTML 不规则），
       Excel/Word 粘贴时会把后续单元格前移（如 A1 空、A2 有内容，结果 A2
       内容跑到 A1）。这里按最大列数补齐，保证每行规整矩形。

    本函数**保留原 td/th 标签类型和合并单元格结构**（不强制首行 th），
    适合渲染展示与复制。

    Args:
        html: 原始表格 HTML（含/不含 ``<html><body>`` 外壳均可）。

    Returns:
        规整化的 ``<table>...</table>``，无展示属性、逻辑列数一致。
    """
    table_match = _RE_TABLE.search(html)
    table_html = table_match.group(1) if table_match else html

    # 只保留安全的数字 rowspan/colspan；style/class/id 等展示属性全部剥离。
    rows: list[list[tuple[str, str, int, int]]] = []
    for tr_match in _RE_TR.finditer(table_html):
        row: list[tuple[str, str, int, int]] = []
        for cm in _RE_CELL.finditer(tr_match.group(1)):
            tag = cm.group(1).lower()  # td 或 th
            text = _cell_text(cm.group(3))
            attrs = cm.group(2)
            row.append(
                (
                    tag,
                    text,
                    _span_value(attrs, "rowspan"),
                    _span_value(attrs, "colspan"),
                )
            )
        if row:  # 跳过空 <tr></tr>
            rows.append(row)

    if not rows:
        return "<table></table>"

    _, max_cols = _layout_rows(rows)
    rows_html, _ = _layout_rows(rows, max_cols)
    return f"<table>{''.join(rows_html)}</table>"


def html_table_to_cell_grid(html: str) -> list[list[str]]:
    """解析单个 ``<table>`` HTML 为 ``rows×cols`` 字符串矩阵。

    复用 :func:`_cell_text`（处理 ``<br>`` / HTML 实体 / 空白规整），
    兼容含 inline ``style``、``<thead>``/``<tbody>``、``<html><body>``
    外壳的输入。空行（无单元格）被丢弃。

    Args:
        html: 含单个 ``<table>`` 的 HTML 片段（可带外壳）。

    Returns:
        ``[[cell, cell, ...], ...]``；无表格或表格为空时返回 ``[]``。
    """
    table_html = extract_table_html(html)
    rows: list[list[str]] = []
    active: list[int] = []
    for tr_match in _RE_TR.finditer(table_html):
        cells: list[str] = []
        col = 0
        for cell in _RE_CELL.finditer(tr_match.group(1)):
            while col < len(active) and active[col] > 0:
                while len(cells) <= col:
                    cells.append("")
                col += 1
            rowspan = _span_value(cell.group(2), "rowspan")
            colspan = _span_value(cell.group(2), "colspan")
            needed = col + colspan
            if len(active) < needed:
                active.extend([0] * (needed - len(active)))
            while len(cells) < needed:
                cells.append("")
            cells[col] = _cell_text(cell.group(3))
            for occupied_col in range(col, needed):
                active[occupied_col] = max(active[occupied_col], rowspan)
            col = needed
        if cells or any(active):
            if len(cells) < len(active):
                cells.extend([""] * (len(active) - len(cells)))
            rows.append(cells)
        active = [max(0, remaining - 1) for remaining in active]
    width = max((len(row) for row in rows), default=0)
    for row in rows:
        row.extend([""] * (width - len(row)))
    return rows


def html_tables_to_cell_grid(html: str) -> list[list[list[str]]]:
    """解析 HTML 中**所有** ``<table>``，每个返回一个 ``rows×cols`` 矩阵。

    与 :func:`html_table_to_cell_grid` 的区别：本函数找出全部表格
    （PaddleX/MinerU 文档解析可能输出多表），按文档顺序返回。

    Args:
        html: 任意 HTML（可含 ``<html><body>`` 外壳、CSS、多个表格）。

    Returns:
        ``[table_grid, ...]``，每个 ``table_grid`` 是 ``[[cell, ...], ...]``；
        无表格时返回 ``[]``。
    """
    grids: list[list[list[str]]] = []
    for table_match in _RE_TABLE.finditer(html):
        grid = html_table_to_cell_grid(table_match.group(1))
        if grid:
            grids.append(grid)
    return grids


def _collect_table_htmls_from_text(text: str) -> list[str]:
    """从任意文本里提取所有 ``<table>...</table>`` 片段（去重、保序）。"""
    seen: set[str] = set()
    out: list[str] = []
    for m in _RE_TABLE.finditer(text or ""):
        frag = m.group(1)
        if frag not in seen:
            seen.add(frag)
            out.append(frag)
    return out


def tables_from_result(result: object) -> list[str]:
    """从 OCR 结果里按优先级收集表格 HTML 片段（去重）。

    覆盖前后端分离下表格可能存放的任意位置，作为导出/复制的单一数据源：

    1. ``content_list`` 中 ``type == "table"`` 块的 ``table_body`` / ``html``；
    2. ``text_blocks`` 中 ``label == "table"`` 块的 ``text``；
    3. ``html_text`` / ``markdown_text`` 中正则提取的 ``<table>``；
    4. ``raw_text`` 里残留的 ``<table>``（表格管道 ``raw_text`` 含裸 HTML）。

    Args:
        result: ``OCRResult`` dataclass、``SimpleNamespace`` 或 wire ``dict``。

    Returns:
        去重后的 ``<table>`` HTML 片段列表（按发现顺序）。
    """

    def _get(name: str) -> object:
        if isinstance(result, dict):
            return result.get(name)
        return getattr(result, name, None)

    seen: set[str] = set()
    tables: list[str] = []

    # 1. content_list table 块
    content_list = _get("content_list") or []
    if isinstance(content_list, (list, tuple)):
        canonical_present = any(
            isinstance(block, dict)
            and str(block.get("type", "")).lower() == "table"
            and isinstance(block.get("table"), dict)
            for block in content_list
        )
        if canonical_present:
            from vibeocr.backend.tables.blocks import table_model_from_block
            from vibeocr.backend.tables.html_adapter import table_model_to_html

            for index, block in enumerate(content_list):
                if (
                    not isinstance(block, dict)
                    or str(block.get("type", "")).lower() != "table"
                ):
                    continue
                table_html = table_model_to_html(
                    table_model_from_block(
                        block,
                        fallback_table_id=f"table-{index}",
                        strict_canonical=False,
                    )
                )
                if table_html not in seen:
                    seen.add(table_html)
                    tables.append(table_html)
            return tables
        for block in content_list:
            if not isinstance(block, dict):
                continue
            if str(block.get("type", "")).lower() != "table":
                continue
            html = block.get("table_body") or block.get("html") or ""
            if isinstance(html, str) and html and html not in seen:
                seen.add(html)
                tables.append(html)

    # 2. text_blocks label=table
    text_blocks = _get("text_blocks") or []
    if isinstance(text_blocks, list):
        for block in text_blocks:
            label = ""
            text = ""
            if isinstance(block, dict):
                label = str(block.get("label", "")).lower()
                text = block.get("text", "")
            else:
                label = str(getattr(block, "label", "")).lower()
                text = getattr(block, "text", "")
            if label != "table" or not isinstance(text, str) or not text:
                continue
            # text 可能是裸 <table>，也可能是含外壳的 HTML
            for frag in _collect_table_htmls_from_text(text):
                if frag not in seen:
                    seen.add(frag)
                    tables.append(frag)

    # 3. html_text / markdown_text / raw_text 里的 <table>
    for field in ("html_text", "markdown_text", "raw_text"):
        for frag in _collect_table_htmls_from_text(str(_get(field) or "")):
            if frag not in seen:
                seen.add(frag)
                tables.append(frag)

    return tables


# Backward-compat aliases matching the original private names.
_extract_table_html = extract_table_html
_html_table_to_markdown = html_table_to_markdown


__all__ = [
    "extract_table_html",
    "html_table_to_cell_grid",
    "html_table_to_markdown",
    "html_tables_to_cell_grid",
    "normalize_table_html",
    "tables_from_result",
]
