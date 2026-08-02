"""pdf_coords bbox 像素坐标变换的旋转/归一化/DPI 边缘用例测试。"""

from __future__ import annotations

import pytest
from vibeocr.backend.utils.pdf_coords import bbox_to_pixel


class TestNormalizedSource:
    """source='normalized' 归一化坐标映射。"""

    def test_origin_maps_to_origin(self):
        """归一化原点 (0,0) 映射到像素原点。"""
        result = bbox_to_pixel((0, 0, 0, 0), (0, 0, 100, 200), 72, source="normalized")
        assert result == (0.0, 0.0, 0.0, 0.0)

    def test_full_range_maps_to_page_size(self):
        """归一化 (0,0,1000,1000) 映射到整页像素（72dpi 即 points）。"""
        result = bbox_to_pixel(
            (0, 0, 1000, 1000), (0, 0, 100, 200), 72, source="normalized"
        )
        assert result == (0.0, 0.0, 100.0, 200.0)

    def test_midpoint_proportional(self):
        """归一化中点按比例映射。"""
        result = bbox_to_pixel(
            (500, 500, 500, 500), (0, 0, 200, 400), 72, source="normalized"
        )
        assert result == (100.0, 200.0, 100.0, 200.0)

    def test_dpi_scaling_applied(self):
        """DPI 缩放：150dpi 把 72dpi 结果放大约一倍。"""
        at_72 = bbox_to_pixel(
            (0, 0, 1000, 1000), (0, 0, 100, 200), 72, source="normalized"
        )
        at_144 = bbox_to_pixel(
            (0, 0, 1000, 1000), (0, 0, 100, 200), 144, source="normalized"
        )
        assert at_144[0] == pytest.approx(at_72[0] * 2)
        assert at_144[3] == pytest.approx(at_72[3] * 2)


class TestPdfSourceRotation:
    """source='pdf' 在不同 rotation 下的坐标变换。"""

    def test_rotation_zero_identity(self):
        """rotation=0 时 bbox 恒等缩放。"""
        result = bbox_to_pixel(
            (10, 20, 30, 40), (0, 0, 100, 100), 72, source="pdf", rotation=0
        )
        assert result == (10.0, 20.0, 30.0, 40.0)

    def test_rotation_90(self):
        """rotation=90： (mb_x, mb_y) -> (mb_h - mb_y, mb_x)。"""
        # mediabox = (0,0,100,80): mb_w=100, mb_h=80
        # bbox (10,20,30,40) -> x0=80-40=40, y0=10, x1=80-20=60, y1=30
        result = bbox_to_pixel(
            (10, 20, 30, 40),
            (0, 0, 80, 100),  # 显示空间旋转后宽高互换：pw=80, ph=100
            72,
            source="pdf",
            rotation=90,
            mediabox=(0, 0, 100, 80),
        )
        assert result == (40.0, 10.0, 60.0, 30.0)

    def test_rotation_180(self):
        """rotation=180：(mb_x, mb_y) -> (mb_w - mb_x, mb_h - mb_y)。"""
        # mediabox = (0,0,100,100): mb_w=100, mb_h=100
        # bbox (10,20,30,40) -> x0=100-30=70, y0=100-40=60, x1=100-10=90, y1=100-20=80
        result = bbox_to_pixel(
            (10, 20, 30, 40),
            (0, 0, 100, 100),
            72,
            source="pdf",
            rotation=180,
            mediabox=(0, 0, 100, 100),
        )
        assert result == (70.0, 60.0, 90.0, 80.0)

    def test_rotation_270(self):
        """rotation=270：(mb_x, mb_y) -> (mb_y, mb_w - mb_x)。"""
        # mediabox = (0,0,100,80): mb_w=100, mb_h=80
        # bbox (10,20,30,40) -> x0=20, y0=100-30=70, x1=40, y1=100-10=90
        result = bbox_to_pixel(
            (10, 20, 30, 40),
            (0, 0, 80, 100),  # 旋转后互换
            72,
            source="pdf",
            rotation=270,
            mediabox=(0, 0, 100, 80),
        )
        assert result == (20.0, 70.0, 40.0, 90.0)

    def test_mediabox_inferred_for_90(self):
        """rotation=90 无显式 mediabox 时从 page_rect 互换推断。"""
        # page_rect=(0,0,80,100): pw=80, ph=100；rotation=90 推断 mb_w=ph=100, mb_h=pw=80
        # 与上例显式 mediabox=(0,0,100,80) 等价
        result = bbox_to_pixel(
            (10, 20, 30, 40),
            (0, 0, 80, 100),
            72,
            source="pdf",
            rotation=90,
        )
        assert result == (40.0, 10.0, 60.0, 30.0)

    def test_mediabox_inferred_for_zero(self):
        """rotation=0 无 mediabox 时推断 mb=page_rect。"""
        result = bbox_to_pixel(
            (10, 20, 30, 40),
            (0, 0, 100, 100),
            72,
            source="pdf",
            rotation=0,
        )
        assert result == (10.0, 20.0, 30.0, 40.0)


class TestPageRectShapes:
    """page_rect 兼容 4-tuple 与带 width/height 的对象。"""

    def test_rect_like_object(self):
        """page_rect 为带 .width/.height 的对象（兼容 fitz.Rect）。"""

        class FakeRect:
            def __init__(self, x0, y0, x1, y1):
                self.x0 = x0
                self.y0 = y0
                self.x1 = x1
                self.y1 = y1
                self.width = x1 - x0
                self.height = y1 - y0

        rect = FakeRect(0, 0, 100, 200)
        result = bbox_to_pixel((0, 0, 1000, 1000), rect, 72, source="normalized")
        assert result == (0.0, 0.0, 100.0, 200.0)

    def test_tuple_page_rect(self):
        """page_rect 为 4-tuple。"""
        result = bbox_to_pixel(
            (0, 0, 1000, 1000), (0, 0, 100, 200), 72, source="normalized"
        )
        assert result == (0.0, 0.0, 100.0, 200.0)

    def test_offset_origin_tuple(self):
        """非零原点的 page_rect 仍按宽高计算（不用原点偏移）。"""
        # page_rect (10,20,110,220) 宽100 高200
        result = bbox_to_pixel(
            (0, 0, 1000, 1000), (10, 20, 110, 220), 72, source="normalized"
        )
        assert result == (0.0, 0.0, 100.0, 200.0)
