"""QrcodeDecodeService 单元测试"""

import pytest

pytest.importorskip("pyzbar")  # pyzbar 缺失时整个文件跳过

from PIL import Image

from vibeocr.backend.services.qrcode_decode_service import (
    DecodedItem,
    QrcodeDecodeService,
)
from vibeocr.backend.services.qrcode_service import QrcodeService


@pytest.fixture
def decode_service():
    return QrcodeDecodeService()


@pytest.fixture
def gen_service():
    return QrcodeService()


def _make_qr_image(text: str, gen_service) -> Image.Image:
    opts = gen_service.default_options()
    opts["format"] = "qr"
    return gen_service.generate(text, opts)


class TestDecodeServiceStructure:
    def test_default_options_returns_dict(self, decode_service):
        opts = decode_service.default_options()
        assert isinstance(opts, dict)

    def test_decode_returns_list(self, decode_service, gen_service):
        img = _make_qr_image("Hello", gen_service)
        results = decode_service.decode(img)
        assert isinstance(results, list)

    def test_decoded_item_fields(self):
        item = DecodedItem(data="x", type="QRCODE", is_url=False)
        assert item.data == "x"
        assert item.type == "QRCODE"
        assert item.is_url is False


class TestDecodeRoundtrip:
    def test_decode_url_qr(self, decode_service, gen_service):
        url = "https://example.com"
        img = _make_qr_image(url, gen_service)
        results = decode_service.decode(img)
        assert len(results) == 1
        assert results[0].data == url
        assert results[0].is_url is True

    def test_decode_non_url_text(self, decode_service, gen_service):
        # 注：用 ASCII 文本，避免 qrcode 库对 CJK 的已知编码缺陷
        # （qrcode 把非 ASCII 字节按 kanji 模式错误转换，是生成侧问题，与本解码服务无关）
        text = "Plain text 12345"
        img = _make_qr_image(text, gen_service)
        results = decode_service.decode(img)
        assert len(results) == 1
        assert results[0].data == text
        assert results[0].is_url is False

    def test_decode_type_is_qrcode(self, decode_service, gen_service):
        img = _make_qr_image("test", gen_service)
        results = decode_service.decode(img)
        assert (
            results[0].type.upper() == "qrcode".upper()
            or "QR" in results[0].type.upper()
        )


class TestDecodeEdgeCases:
    def test_decode_blank_image_returns_empty(self, decode_service):
        blank = Image.new("RGB", (100, 100), "white")
        assert decode_service.decode(blank) == []

    def test_decode_multiple_codes(self, decode_service, gen_service):
        img1 = _make_qr_image("first-payload", gen_service)
        img2 = _make_qr_image("second-payload", gen_service)
        w = img1.width + img2.width + 40
        h = max(img1.height, img2.height)
        canvas = Image.new("RGB", (w, h), "white")
        canvas.paste(img1, (0, 0))
        canvas.paste(img2, (img1.width + 40, 0))
        results = decode_service.decode(canvas)
        datas = {r.data for r in results}
        assert "first-payload" in datas
        assert "second-payload" in datas
        assert len(results) >= 2

    def test_decode_file(self, decode_service, gen_service, tmp_path):
        img = _make_qr_image("file-test", gen_service)
        path = tmp_path / "qr.png"
        img.save(str(path))
        results = decode_service.decode_file(str(path))
        assert len(results) == 1
        assert results[0].data == "file-test"

    def test_decode_bytes(self, decode_service, gen_service):
        import io

        img = _make_qr_image("bytes-test", gen_service)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        results = decode_service.decode_bytes(buf.getvalue())
        assert len(results) == 1
        assert results[0].data == "bytes-test"


class TestDecodeLargeImage:
    def test_huge_image_does_not_crash(self, decode_service, gen_service):
        """构造一张含小二维码的大图，验证大图保护路径不抛异常。"""
        qr = _make_qr_image("big-img-test", gen_service).resize((100, 100))
        # 粘贴到远大于 4096 的白画布上
        canvas = Image.new("RGB", (5000, 5000), "white")
        canvas.paste(qr, (0, 0))
        results = decode_service.decode(canvas)
        datas = {r.data for r in results}
        assert "big-img-test" in datas


class TestUrlDetection:
    def test_http_url_detected(self, decode_service, gen_service):
        img = _make_qr_image("http://foo.bar/baz", gen_service)
        results = decode_service.decode(img)
        assert results[0].is_url is True

    def test_javascript_scheme_not_url(self):
        from vibeocr.backend.services.qrcode_decode_service import _is_http_url

        assert _is_http_url("javascript:alert(1)") is False

    def test_file_scheme_not_url(self):
        from vibeocr.backend.services.qrcode_decode_service import _is_http_url

        assert _is_http_url("file:///etc/passwd") is False

    def test_plain_text_not_url(self):
        from vibeocr.backend.services.qrcode_decode_service import _is_http_url

        assert _is_http_url("just some text") is False

    def test_uppercase_scheme_is_url(self):
        from vibeocr.backend.services.qrcode_decode_service import _is_http_url

        assert _is_http_url("HTTPS://example.com/path") is True


def test_is_http_url_rejects_invalid_url(decode_service):
    """_is_http_url 对非 http scheme 与无 netloc 返回 False。"""
    from vibeocr.backend.services.qrcode_decode_service import _is_http_url

    # 正常情况
    assert _is_http_url("https://example.com") is True
    assert _is_http_url("http://example.com/path") is True
    # 非 http scheme
    assert _is_http_url("ftp://example.com") is False
    assert _is_http_url("javascript:alert(1)") is False
    assert _is_http_url("not a url") is False
    # http:// 但无 netloc
    assert _is_http_url("http://") is False


def test_is_http_url_returns_false_when_urlparse_raises(monkeypatch):
    """urlparse 抛 ValueError/TypeError 时返回 False（line 34-36）。"""
    from vibeocr.backend.services import qrcode_decode_service

    def _raise(_v):
        raise ValueError("bad url")

    monkeypatch.setattr(qrcode_decode_service, "urlparse", _raise)
    assert qrcode_decode_service._is_http_url("https://example.com") is False


def test_decode_skips_empty_and_undecodable_results(
    decode_service, gen_service, monkeypatch
):
    """decode 跳过空数据/解码失败的结果（line 70-73）。"""
    from PIL import Image

    # 构造 fake pyzbar decode 返回：一个空数据 + 一个解码失败 + 一个有效
    class _FakeResult:
        def __init__(self, data_bytes, rtype="QRCODE"):
            self.data = data_bytes
            self.type = rtype

    valid = b"https://example.com"

    def fake_zbar_decode(_gray):
        return [
            _FakeResult(None),  # None.data.decode 抛 AttributeError → 跳过 (line 70-71)
            _FakeResult(b"   "),  # 空白 → 跳过 (line 72-73)
            _FakeResult(valid),  # 有效
        ]

    # patch pyzbar.decode 在 decode() 内部 import
    import sys

    fake_pyzbar = type(sys)("pyzbar.pyzbar")
    fake_pyzbar.decode = fake_zbar_decode
    fake_pkg = type(sys)("pyzbar")
    fake_pkg.pyzbar = fake_pyzbar
    monkeypatch.setitem(sys.modules, "pyzbar", fake_pkg)
    monkeypatch.setitem(sys.modules, "pyzbar.pyzbar", fake_pyzbar)

    img = Image.new("RGB", (50, 50), "white")
    results = decode_service.decode(img)
    assert len(results) == 1  # 只有有效的那条
    assert results[0].data == "https://example.com"
    assert results[0].is_url is True
