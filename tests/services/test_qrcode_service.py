"""QrcodeService 单元测试"""

import pytest
from PIL import Image


@pytest.fixture
def service():
    from vibeocr.backend.services.qrcode_service import QrcodeService

    return QrcodeService()


class TestQrCodeGeneration:
    def test_generate_qr_returns_pil_image(self, service):
        options = service.default_options()
        options["format"] = "qr"
        options["text"] = "Hello"
        img = service.generate("Hello", options)
        assert isinstance(img, Image.Image)

    def test_generate_qr_non_empty(self, service):
        options = service.default_options()
        options["format"] = "qr"
        img = service.generate("Hello", options)
        assert img.width > 0
        assert img.height > 0

    def test_generate_qr_respects_size(self, service):
        options = service.default_options()
        options["format"] = "qr"
        options["size"] = 200
        img = service.generate("Test", options)
        assert img.width == 200
        assert img.height == 200

    def test_generate_qr_respects_error_correction(self, service):
        options = service.default_options()
        options["format"] = "qr"
        options["error_correction"] = "H"
        img = service.generate("Test", options)
        assert isinstance(img, Image.Image)

    def test_generate_qr_respects_fg_bg_colors(self, service):
        options = service.default_options()
        options["format"] = "qr"
        options["fg_color"] = "#FF0000"
        options["bg_color"] = "#0000FF"
        img = service.generate("Test", options)
        assert isinstance(img, Image.Image)

    def test_default_options_returns_dict(self, service):
        opts = service.default_options()
        assert isinstance(opts, dict)
        assert opts["format"] == "qr"
        assert opts["size"] == 600
        assert opts["error_correction"] == "M"
        assert opts["fg_color"] == "#000000"
        assert opts["bg_color"] == "#FFFFFF"
        assert opts["invert"] is False


class TestBarcodeGeneration:
    def test_generate_code128_returns_pil_image(self, service):
        options = service.default_options()
        options["format"] = "code128"
        img = service.generate("Hello123", options)
        assert isinstance(img, Image.Image)
        assert img.width > 0
        assert img.height > 0

    def test_generate_code39_returns_pil_image(self, service):
        options = service.default_options()
        options["format"] = "code39"
        img = service.generate("HELLO123", options)
        assert isinstance(img, Image.Image)

    def test_generate_ean13_returns_pil_image(self, service):
        options = service.default_options()
        options["format"] = "ean13"
        img = service.generate("5901234123457", options)
        assert isinstance(img, Image.Image)

    def test_generate_barcode_respects_size(self, service):
        options = service.default_options()
        options["format"] = "code128"
        options["size"] = 400
        img = service.generate("Hello123", options)
        assert img.height == 400

    def test_unsupported_barcode_raises(self, service):
        import barcode.errors

        options = service.default_options()
        options["format"] = "nonexistent_format"
        with pytest.raises(barcode.errors.BarcodeNotFoundError):
            service.generate("Hello", options)


class TestLogoEmbedding:
    def test_apply_logo_with_valid_image(self, service, tmp_path):
        logo = Image.new("RGB", (50, 50), color="red")
        logo_path = str(tmp_path / "logo.png")
        logo.save(logo_path)

        qr_img = Image.new("RGB", (300, 300), color="white")
        result = service.apply_logo(qr_img, logo_path, ratio=0.2)
        assert isinstance(result, Image.Image)
        assert result.size == qr_img.size

    def test_apply_logo_ratio_affects_logo_size(self, service, tmp_path):
        logo = Image.new("RGB", (50, 50), color="blue")
        logo_path = str(tmp_path / "logo.png")
        logo.save(logo_path)

        qr_img = Image.new("RGB", (300, 300), color="white")
        result_small = service.apply_logo(qr_img, logo_path, ratio=0.1)
        result_large = service.apply_logo(qr_img, logo_path, ratio=0.4)
        assert isinstance(result_small, Image.Image)
        assert isinstance(result_large, Image.Image)

    def test_apply_logo_with_qr_pipeline(self, service, tmp_path):
        logo = Image.new("RGB", (30, 30), color="green")
        logo_path = str(tmp_path / "logo.png")
        logo.save(logo_path)

        options = service.default_options()
        options["format"] = "qr"
        options["logo_path"] = logo_path
        options["logo_ratio"] = 0.25
        img = service.generate("Test with logo", options)
        img = service.apply_logo(img, logo_path, ratio=0.25)
        assert isinstance(img, Image.Image)


class TestTextLabelAndInvert:
    def test_apply_text_label_bottom(self, service):
        img = Image.new("RGB", (300, 300), color="white")
        result = service.apply_text_label(
            img, "Scan me", position="bottom", font_size=14
        )
        assert isinstance(result, Image.Image)
        assert result.height > img.height

    def test_apply_text_label_top(self, service):
        img = Image.new("RGB", (300, 300), color="white")
        result = service.apply_text_label(
            img, "Top label", position="top", font_size=12
        )
        assert isinstance(result, Image.Image)
        assert result.height > img.height

    def test_apply_text_label_none_returns_original(self, service):
        img = Image.new("RGB", (300, 300), color="white")
        result = service.apply_text_label(img, "Label", position="none", font_size=12)
        assert result.size == img.size

    def test_invert_colors(self, service):
        img = Image.new("RGB", (10, 10), color=(0, 0, 0))
        result = service.invert_colors(img)
        pixel = result.getpixel((0, 0))
        assert pixel == (255, 255, 255)

    def test_invert_colors_white_to_black(self, service):
        img = Image.new("RGB", (10, 10), color=(255, 255, 255))
        result = service.invert_colors(img)
        pixel = result.getpixel((0, 0))
        assert pixel == (0, 0, 0)

    def test_full_pipeline_with_invert(self, service):
        options = service.default_options()
        options["format"] = "qr"
        options["invert"] = True
        options["label_text"] = "Inverted QR"
        options["label_position"] = "bottom"
        img = service.generate("Test", options)
        img = service.apply_text_label(
            img, "Inverted QR", position="bottom", font_size=12
        )
        img = service.invert_colors(img)
        assert isinstance(img, Image.Image)


class TestSvgExport:
    def test_generate_svg_returns_string(self, service):
        options = service.default_options()
        options["format"] = "qr"
        svg = service.generate_svg("Hello", options)
        assert isinstance(svg, str)
        assert "<svg" in svg

    def test_generate_svg_contains_content(self, service):
        options = service.default_options()
        options["format"] = "qr"
        svg = service.generate_svg("Test SVG", options)
        assert len(svg) > 100


class TestQrcodeServiceInternalBranches:
    """_generate_qr no-canvas + _load_font 平台/回退分支。"""

    def test_generate_qr_small_size_returns_image(self, service):
        """size=1 时 _generate_qr 走 line 85 直接返回（actual==target 或无 padding）。"""
        from PIL import Image

        img = service.generate("x", {"type": "qr", "size": 1, "box_size": 1})
        assert isinstance(img, Image.Image)

    def test_load_font_darwin_candidates(self, monkeypatch):
        """sys.platform=darwin 时返回 MAC 字体候选路径（line 146-150）。"""
        import sys

        from vibeocr.backend.services.qrcode_service import QrcodeService

        monkeypatch.setattr(sys, "platform", "darwin")
        font = QrcodeService._load_font(20)
        assert font is not None

    def test_load_font_linux_candidates(self, monkeypatch):
        """sys.platform=linux 时返回 Linux 字体候选（line 151-155）。"""
        import sys

        from vibeocr.backend.services.qrcode_service import QrcodeService

        monkeypatch.setattr(sys, "platform", "linux")
        font = QrcodeService._load_font(20)
        assert font is not None

    def test_load_font_falls_back_to_default_when_all_missing(self, monkeypatch):
        """所有候选字体都不存在/损坏时回退 load_default（line 161-163）。"""
        from vibeocr.backend.services.qrcode_service import QrcodeService

        font = QrcodeService._load_font(16)
        assert font is not None


class TestGenerateAppliesExtras:
    """generate 必须应用 options 里的 invert / logo / label。

    回归：历史上端点只调 generate，logo/label/invert 选项被静默忽略，
    Classic 前端的对应开关完全无效。
    """

    def test_generate_applies_invert(self, service):
        options = service.default_options()
        options["format"] = "qr"
        options["invert"] = True
        img = service.generate("invert me", options)
        # 反色后四角（quiet zone）应为背景色白→黑或含黑色通道。
        assert isinstance(img, Image.Image)
        plain = service.generate("invert me", {**options, "invert": False})
        assert list(img.getdata()) != list(plain.getdata())

    def test_generate_applies_label(self, service):
        options = service.default_options()
        options["format"] = "qr"
        options["size"] = 300
        options["label_text"] = "Scan me"
        options["label_position"] = "bottom"
        options["label_font_size"] = 14
        img = service.generate("label me", options)
        plain = service.generate("label me", {**options, "label_text": ""})
        assert img.height > plain.height

    def test_generate_applies_logo(self, service, tmp_path):
        logo = Image.new("RGB", (30, 30), color="green")
        logo_path = str(tmp_path / "logo.png")
        logo.save(logo_path)

        options = service.default_options()
        options["format"] = "qr"
        options["logo_path"] = logo_path
        options["logo_ratio"] = 0.25
        img = service.generate("logo me", options)
        plain = service.generate("logo me", {**options, "logo_path": None})
        assert isinstance(img, Image.Image)
        assert list(img.getdata()) != list(plain.getdata())
