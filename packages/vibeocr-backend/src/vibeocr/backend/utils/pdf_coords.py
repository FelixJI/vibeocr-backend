"""PDF 坐标变换纯函数(无 fitz 依赖)。

主进程 PDF 预览窗口渲染文字层高亮时,需要把 bbox(PDF points 或归一化坐标)
转换为渲染图像的像素坐标。此模块从 ``pdf_service.py`` 抽出,使主进程不再触发
``pdf_service`` 的顶层 ``import fitz``,从而 fitz 可从主 exe 排除(子进程仍用)。

设计要点:
- 纯数学,不 import fitz/numpy/Qt,任意进程均可安全加载。
- ``page_rect`` 同时支持 4-tuple ``(x0,y0,x1,y1)`` 与带 ``.width/.height`` 的对象
  (兼容遗留调用方可能传入的 ``fitz.Rect``,虽然主进程统一用 tuple)。
"""

from __future__ import annotations


def bbox_to_pixel(
    bbox: tuple[float, float, float, float],
    page_rect: tuple[float, float, float, float],
    render_dpi: int,
    source: str = "pdf",
    rotation: int = 0,
    mediabox: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float]:
    """将 bbox 转换为渲染图像的像素坐标。

    Args:
        bbox: 输入 bbox。
        page_rect: PDF 页面矩形 (points)。接受 4-tuple ``(x0,y0,x1,y1)`` 或
            带 ``.width/.height`` 属性的对象(兼容 ``fitz.Rect``,主进程统一用 tuple)。
            这是 **显示空间** 尺寸（= page.rect，含旋转/CropBox 归零）。
        render_dpi: 渲染 DPI。
        source: ``"pdf"`` 表示 bbox 是 PDF points 坐标（MediaBox 未旋转空间），
                ``"normalized"`` 表示 ``[0, 1000]`` 归一化坐标（显示空间）。
        rotation: 页面 /Rotate（0/90/180/270）。source="pdf" 时把 MediaBox 空间
            bbox 转到显示空间（与渲染 pixmap 同空间），否则旋转页文字层高亮会
            位置/角度全错。source="normalized" 时 bbox 已在显示空间，rotation 忽略。
        mediabox: MediaBox ``(x0,y0,x1,y1)``，rotation≠0 时用于转换。None 时
            从 page_rect 推断（假设无 CropBox，mediabox = 未旋转的 page_rect）。

    Returns:
        像素坐标 ``(x0, y0, x1, y1)``。
    """
    # 兼容 fitz.Rect(有 .width/.height)与 4-tuple (x0,y0,x1,y1)
    pw = getattr(page_rect, "width", None)
    if pw is None:
        pw = page_rect[2] - page_rect[0]
        ph = page_rect[3] - page_rect[1]
    else:
        ph = page_rect.height  # type: ignore[attr-defined]

    if source == "normalized":
        # 归一化 bbox 已在显示空间，直接映射到 page_rect 像素
        x0 = bbox[0] / 1000 * pw
        y0 = bbox[1] / 1000 * ph
        x1 = bbox[2] / 1000 * pw
        y1 = bbox[3] / 1000 * ph
    else:
        # source="pdf": bbox 在 MediaBox（未旋转）空间，需转到显示空间。
        # page_rect 是显示空间（含旋转，宽高可能互换）。mediabox 宽高用于旋转转换。
        mb_w = mb_h = 0.0
        if mediabox is not None:
            mb_w = mediabox[2] - mediabox[0]
            mb_h = mediabox[3] - mediabox[1]
        else:
            # 无 CropBox 时：rot∈{0,180} mediabox=page_rect 尺寸；
            # rot∈{90,270} mediabox 宽高 = page_rect 互换后的尺寸
            if rotation in (90, 270):
                mb_w = ph
                mb_h = pw
            else:
                mb_w = pw
                mb_h = ph
        bx0, by0, bx1, by1 = bbox
        if rotation == 90:
            # (mb_x, mb_y) -> (mb_h - mb_y, mb_x)
            x0, y0 = mb_h - by1, bx0
            x1, y1 = mb_h - by0, bx1
        elif rotation == 180:
            # (mb_x, mb_y) -> (mb_w - mb_x, mb_h - mb_y)
            x0, y0 = mb_w - bx1, mb_h - by1
            x1, y1 = mb_w - bx0, mb_h - by0
        elif rotation == 270:
            # (mb_x, mb_y) -> (mb_y, mb_w - mb_x)
            x0, y0 = by0, mb_w - bx1
            x1, y1 = by1, mb_w - bx0
        else:
            x0, y0, x1, y1 = bx0, by0, bx1, by1

    # 显示空间 points → pixels: coord / 72 * dpi
    scale = render_dpi / 72.0
    return (x0 * scale, y0 * scale, x1 * scale, y1 * scale)
