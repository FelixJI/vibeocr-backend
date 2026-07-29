# tests/services/test_pdf_service_bbox.py
"""Tests for PDF bbox coordinate inverse transform."""

import fitz

from vibeocr.backend.services.pdf_service import PdfService

# 标准信纸尺寸 page_rect: 612×792 points
_PAGE_RECT = fitz.Rect(0, 0, 612, 792)


class TestDenormalizeAndUnrotateBbox:
    """测试 _denormalize_and_unrotate_bbox 的四种旋转角度。"""

    def test_no_rotation(self):
        """0°：直接映射，无变换。"""
        # bbox 覆盖整个页面 [0, 1000]
        result = PdfService._denormalize_and_unrotate_bbox(
            (0.0, 0.0, 1000.0, 1000.0), 0, _PAGE_RECT
        )
        assert result.is_empty is False
        assert abs(result.x0 - 0) < 1
        assert abs(result.y0 - 0) < 1
        assert abs(result.x1 - 612) < 1
        assert abs(result.y1 - 792) < 1

    def test_no_rotation_partial(self):
        """0°：部分区域 bbox。"""
        result = PdfService._denormalize_and_unrotate_bbox(
            (100.0, 200.0, 500.0, 800.0), 0, _PAGE_RECT
        )
        assert abs(result.x0 - 61.2) < 1  # 100/1000 * 612
        assert abs(result.y0 - 158.4) < 1  # 200/1000 * 792
        assert abs(result.x1 - 306.0) < 1  # 500/1000 * 612
        assert abs(result.y1 - 633.6) < 1  # 800/1000 * 792

    def test_rotation_180_center(self):
        """180°：中心点保持不变。"""
        # bbox 在归一化空间的中心 (500, 500)
        result = PdfService._denormalize_and_unrotate_bbox(
            (450.0, 450.0, 550.0, 550.0), 180, _PAGE_RECT
        )
        cx = (result.x0 + result.x1) / 2
        cy = (result.y0 + result.y1) / 2
        # 页面中心 (306, 396)
        assert abs(cx - 306) < 2
        assert abs(cy - 396) < 2

    def test_rotation_180_corner(self):
        """180°：左上角映射到右下角。"""
        result = PdfService._denormalize_and_unrotate_bbox(
            (0.0, 0.0, 100.0, 100.0), 180, _PAGE_RECT
        )
        # 原始左上 → 旋转后映射到右下区域
        assert result.x0 > 300  # 应在右半部分
        assert result.y0 > 400  # 应在下半部分

    def test_rotation_90_width_height_swap(self):
        """90°：旋转后宽度方向映射到页面高度方向。"""
        # 窄长条 (归一化 x: 0-100, y: 0-900)
        result = PdfService._denormalize_and_unrotate_bbox(
            (0.0, 0.0, 100.0, 900.0), 90, _PAGE_RECT
        )
        # 90° 逆变换：y 方向映射到 x 方向，x 方向映射到 y 方向
        # 横向范围 = 0-900/1000*612 → 约 0-550.8
        # 纵向范围 = (1-100/1000)*792 - (1-0/1000)*792 → 约 712.8-792
        assert result.width > result.height  # 旋转后窄条变横条

    def test_rotation_270_width_height_swap(self):
        """270°：与 90° 方向相反。"""
        result = PdfService._denormalize_and_unrotate_bbox(
            (0.0, 0.0, 100.0, 900.0), 270, _PAGE_RECT
        )
        assert result.width > result.height

    def test_roundtrip_90(self):
        """90° 旋转后完整 bbox 覆盖整个页面。"""
        result = PdfService._denormalize_and_unrotate_bbox(
            (0.0, 0.0, 1000.0, 1000.0), 90, _PAGE_RECT
        )
        assert abs(result.width - 612) < 2
        assert abs(result.height - 792) < 2

    def test_roundtrip_270(self):
        """270° 旋转后完整 bbox 覆盖整个页面。"""
        result = PdfService._denormalize_and_unrotate_bbox(
            (0.0, 0.0, 1000.0, 1000.0), 270, _PAGE_RECT
        )
        assert abs(result.width - 612) < 2
        assert abs(result.height - 792) < 2

    def test_invalid_angle_defaults_to_zero(self):
        """无效角度视为 0°。"""
        result = PdfService._denormalize_and_unrotate_bbox(
            (100.0, 200.0, 500.0, 800.0), 45, _PAGE_RECT
        )
        expected = PdfService._denormalize_and_unrotate_bbox(
            (100.0, 200.0, 500.0, 800.0), 0, _PAGE_RECT
        )
        assert abs(result.x0 - expected.x0) < 1
        assert abs(result.y0 - expected.y0) < 1


class TestUnrotatePosition:
    """位置敏感回归：90/270 角度的逆变换必须落在正确象限。

    复现用户报告「开启文档方向分类后，部分页面文字层方向旋转 90°」。
    根因：_denormalize_and_unrotate_bbox 的 90 与 270 分支被写反。

    PaddleOCR 实测约定（scripts/verify_orient_roundtrip2/3.py）：
      reported angle == 内容相对正向「顺时针」偏转的度数；PaddleOCR 把图
      「逆时针」旋转 angle 度得到正向 output_img。bbox 在 output(正向)空间。
      要还原回「显示空间」(= 顺时针偏转 angle 度的图)，需把 output bbox
      「顺时针」旋转 angle 度。

    本组测试用一个「正向图顶部窄条」的归一化 bbox，验证它经各角度逆变换后
    在显示空间落到的正确位置（左/右/上/下），而非仅校验宽高互换。
    """

    def test_90_top_band_lands_on_right(self):
        """angle=90：正向图「顶部窄条」还原到显示空间应在「右侧」。

        正向图顶部窄条 (x:40-60%, y:5-10%)。CW90 后该条在显示图右侧（x 大）。
        """
        result = PdfService._denormalize_and_unrotate_bbox(
            (400.0, 50.0, 600.0, 100.0), 90, _PAGE_RECT
        )
        # 应落在右半部分（接近右边缘）
        assert result.x0 > _PAGE_RECT.width * 0.5, (
            f"angle=90: 顶部窄条还原后应在右侧(x0>{_PAGE_RECT.width*0.5:.0f})，"
            f"实际 rect={result}（90/270 分支写反的症状）"
        )

    def test_270_top_band_lands_on_left(self):
        """angle=270：正向图「顶部窄条」还原到显示空间应在「左侧」。"""
        result = PdfService._denormalize_and_unrotate_bbox(
            (400.0, 50.0, 600.0, 100.0), 270, _PAGE_RECT
        )
        # 应落在左半部分（接近左边缘）
        assert result.x1 < _PAGE_RECT.width * 0.5, (
            f"angle=270: 顶部窄条还原后应在左侧(x1<{_PAGE_RECT.width*0.5:.0f})，"
            f"实际 rect={result}（90/270 分支写反的症状）"
        )

    def test_90_left_band_lands_on_top(self):
        """angle=90：正向图「左侧窄条」还原到显示空间应在「顶部」。"""
        # 左侧竖条 (x:5-10%, y:40-60%)
        result = PdfService._denormalize_and_unrotate_bbox(
            (50.0, 400.0, 100.0, 600.0), 90, _PAGE_RECT
        )
        assert result.y1 < _PAGE_RECT.height * 0.5, (
            f"angle=90: 左侧竖条还原后应在顶部(y1<{_PAGE_RECT.height*0.5:.0f})，"
            f"实际 rect={result}"
        )

    def test_270_left_band_lands_on_bottom(self):
        """angle=270：正向图「左侧窄条」还原到显示空间应在「底部」。"""
        result = PdfService._denormalize_and_unrotate_bbox(
            (50.0, 400.0, 100.0, 600.0), 270, _PAGE_RECT
        )
        assert result.y0 > _PAGE_RECT.height * 0.5, (
            f"angle=270: 左侧竖条还原后应在底部(y0>{_PAGE_RECT.height*0.5:.0f})，"
            f"实际 rect={result}"
        )


class TestBboxToPixel:
    """测试 bbox_to_pixel 的坐标转换。"""

    def test_pdf_points_to_pixel(self):
        """source=pdf：PDF points → pixels。"""
        # 100pt @ 72dpi = 100px；@ 144dpi = 200px
        result = PdfService.bbox_to_pixel(
            (100.0, 100.0, 200.0, 200.0), _PAGE_RECT, render_dpi=72
        )
        assert abs(result[0] - 100) < 0.01
        assert abs(result[2] - 200) < 0.01

        result_144 = PdfService.bbox_to_pixel(
            (100.0, 100.0, 200.0, 200.0), _PAGE_RECT, render_dpi=144
        )
        assert abs(result_144[0] - 200) < 0.01
        assert abs(result_144[2] - 400) < 0.01

    def test_normalized_to_pixel(self):
        """source=normalized：[0,1000] → pixels。"""
        # 完整页面归一化 bbox @ 72dpi 应等于页面尺寸
        result = PdfService.bbox_to_pixel(
            (0.0, 0.0, 1000.0, 1000.0), _PAGE_RECT, render_dpi=72, source="normalized"
        )
        assert abs(result[0] - 0) < 0.01
        assert abs(result[2] - 612) < 0.01
        assert abs(result[3] - 792) < 0.01

    def test_pdf_points_rotation_90(self):
        """source=pdf + rotation=90：MediaBox bbox 转到显示空间。

        Bug：预览已有文字层时 get_text 返回 MediaBox（未旋转）bbox，但预览
        pixmap 是显示（旋转）空间，bbox_to_pixel 不做旋转 → 高亮位置/角度全错。
        修复：rotation 参数把 MediaBox bbox 转到显示空间。
        公式（无 CropBox）：rot=90 时 (mb_x,mb_y) -> (mb_h-mb_y, mb_x)。
        """
        # mediabox 595.2×841.68，rot=90 显示空间 page_rect = (0,0,841.68,595.2)
        disp_rect = (0.0, 0.0, 841.68, 595.2)
        # mb(200,300) 点 → 显示 (841.68-300, 200) = (541.68, 200)
        result = PdfService.bbox_to_pixel(
            (200.0, 300.0, 200.0, 300.0), disp_rect, render_dpi=72,
            source="pdf", rotation=90,
        )
        assert abs(result[0] - 541.68) < 1, f"rot=90 x 应≈541.68，实际 {result[0]:.1f}"
        assert abs(result[1] - 200) < 1, f"rot=90 y 应≈200，实际 {result[1]:.1f}"

    def test_pdf_points_rotation_180(self):
        """rotation=180：(mb_x,mb_y) -> (mb_w-mb_x, mb_h-mb_y)。"""
        disp_rect = (0.0, 0.0, 595.2, 841.68)
        # mb(200,300) -> (595.2-200, 841.68-300) = (395.2, 541.68)
        result = PdfService.bbox_to_pixel(
            (200.0, 300.0, 200.0, 300.0), disp_rect, render_dpi=72,
            source="pdf", rotation=180,
        )
        assert abs(result[0] - 395.2) < 1
        assert abs(result[1] - 541.68) < 1

    def test_pdf_points_rotation_270(self):
        """rotation=270：(mb_x,mb_y) -> (mb_y, mb_w-mb_x)。"""
        disp_rect = (0.0, 0.0, 841.68, 595.2)  # rot=270 display
        # mb(200,300) -> (300, 595.2-200) = (300, 395.2)
        result = PdfService.bbox_to_pixel(
            (200.0, 300.0, 200.0, 300.0), disp_rect, render_dpi=72,
            source="pdf", rotation=270,
        )
        assert abs(result[0] - 300) < 1
        assert abs(result[1] - 395.2) < 1

    def test_rotation_ignored_for_normalized(self):
        """source=normalized 时 rotation 应被忽略（bbox 已在显示空间）。"""
        result = PdfService.bbox_to_pixel(
            (500.0, 500.0, 500.0, 500.0), _PAGE_RECT, render_dpi=72,
            source="normalized", rotation=90,
        )
        # 500/1000 * 612 = 306, 500/1000 * 792 = 396（不受 rotation 影响）
        assert abs(result[0] - 306) < 1
        assert abs(result[1] - 396) < 1
