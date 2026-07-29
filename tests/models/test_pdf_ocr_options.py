# tests/models/test_pdf_ocr_options.py
"""Tests for PdfGlobalSettings data model."""

from vibeocr.backend.models.pdf_ocr_options import PdfGlobalSettings


class TestPdfGlobalSettingsDefaults:
    def test_default_values(self):
        s = PdfGlobalSettings()
        assert s.render_dpi == 300
        assert s.max_pixels == 16_000_000
        assert s.font_size_ratio == 0.8
        assert s.text_layer_visible is False
        assert s.font_size_retry_count == 5
        assert s.font_size_shrink_factor == 0.75

    def test_custom_values(self):
        s = PdfGlobalSettings(render_dpi=150, max_pixels=8_000_000, font_size_ratio=0.6)
        assert s.render_dpi == 150
        assert s.max_pixels == 8_000_000
        assert s.font_size_ratio == 0.6

    def test_min_font_size_default(self):
        s = PdfGlobalSettings()
        assert s.min_font_size == 4.0


class TestPdfGlobalSettingsSerialization:
    def test_to_dict_roundtrip(self):
        s = PdfGlobalSettings(render_dpi=200, font_size_ratio=0.7, min_font_size=6.0)
        d = s.to_dict()
        assert d["render_dpi"] == 200
        assert d["font_size_ratio"] == 0.7
        assert d["min_font_size"] == 6.0

        s2 = PdfGlobalSettings.from_dict(d)
        assert s2.render_dpi == 200
        assert s2.font_size_ratio == 0.7
        assert s2.min_font_size == 6.0

    def test_from_dict_missing_fields_use_defaults(self):
        s = PdfGlobalSettings.from_dict({"render_dpi": 150})
        assert s.render_dpi == 150
        assert s.max_pixels == 16_000_000
        assert s.font_size_retry_count == 5
        # min_font_size 缺失时回退默认值（向后兼容旧偏好）
        assert s.min_font_size == 4.0

    def test_from_dict_empty(self):
        s = PdfGlobalSettings.from_dict({})
        assert s == PdfGlobalSettings()


class TestPdfGlobalSettingsAdjustDpi:
    def test_adjust_dpi_no_change_when_within_limit(self):
        s = PdfGlobalSettings(render_dpi=300, max_pixels=16_000_000)
        # A4 at 300dpi = 2480*3508 = ~8.7M pixels, well within 16M
        adjusted = s.adjust_dpi(612, 792)
        assert adjusted == 300

    def test_adjust_dpi_reduces_when_exceeds_limit(self):
        s = PdfGlobalSettings(render_dpi=600, max_pixels=4_000_000)
        # A4 at 600dpi = 4960*7016 = ~34.8M, way over 4M
        adjusted = s.adjust_dpi(612, 792)
        assert adjusted < 600
        # Verify the adjusted DPI stays within limit
        w = int(612 / 72 * adjusted)
        h = int(792 / 72 * adjusted)
        assert w * h <= 4_000_000

    def test_adjust_dpi_floors_at_72(self):
        """极端小像素上限时，DPI 不低于 72。"""
        s = PdfGlobalSettings(render_dpi=600, max_pixels=10_000)
        assert s.adjust_dpi(612, 792) == 72

    def test_adjust_dpi_at_limit_boundary(self):
        """像素恰好等于上限时，DPI 保持不变。"""
        # A4 @ 300dpi ≈ 8.7M，把上限设为略高于该值
        s = PdfGlobalSettings(render_dpi=300, max_pixels=9_000_000)
        assert s.adjust_dpi(612, 792) == 300
