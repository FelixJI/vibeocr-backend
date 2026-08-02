"""MIME 类型映射工具的边缘用例测试。"""

from __future__ import annotations

import pytest
from vibeocr.backend.utils.mime_types import (
    DOCUMENT_EXTENSIONS,
    EXT_TO_MIME,
    FILE_FILTER_ALL,
    FILE_FILTER_DOCUMENTS,
    FILE_FILTER_IMAGES,
    extension_to_mime,
    guess_mime_from_bytes,
    guess_mime_from_filename,
    is_document_file,
    is_office_file,
    mime_to_extension,
)


class TestExtensionToMime:
    """extension_to_mime 边缘用例。"""

    def test_known_extensions(self):
        """已知扩展名返回对应 MIME。"""
        assert extension_to_mime(".pdf") == "application/pdf"
        assert extension_to_mime(".png") == "image/png"
        assert extension_to_mime(".jpg") == "image/jpeg"
        assert extension_to_mime(".xlsx") == (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    def test_lowercase_normalization(self):
        """大写扩展名归一化为小写后命中。"""
        assert extension_to_mime(".PNG") == "image/png"
        assert extension_to_mime(".PDF") == "application/pdf"
        assert extension_to_mime(".JpEg") == "image/jpeg"

    def test_unknown_extension_returns_none(self):
        """未注册扩展名返回 None。"""
        assert extension_to_mime(".txt") is None
        assert extension_to_mime(".gif.xyz") is None

    def test_requires_leading_dot(self):
        """无前导点查不到（按设计 ext 需带点）。"""
        assert extension_to_mime("pdf") is None


class TestMimeToExtension:
    """mime_to_extension 边缘用例。"""

    def test_known_mime(self):
        """已知 MIME 返回带点扩展名。"""
        assert mime_to_extension("application/pdf") == ".pdf"
        assert mime_to_extension("image/png") == ".png"

    def test_unknown_mime_returns_none(self):
        """未注册 MIME 返回 None。"""
        assert mime_to_extension("text/plain") is None
        assert mime_to_extension("") is None

    def test_round_trip_for_all_known(self):
        """所有 EXT_TO_MIME 条目经正向查表后反向至少能回到该扩展名。

        jpg/jpeg 共享同一 MIME，反向取首个匹配，故仅校验其值在表内一致。
        """
        for ext, mime in EXT_TO_MIME.items():
            assert mime_to_extension(mime) in EXT_TO_MIME
            assert extension_to_mime(mime_to_extension(mime)) == mime


class TestGuessMimeFromFilename:
    """guess_mime_from_filename 边缘用例。"""

    def test_known_filename(self):
        """带路径的已知文件名正确解析。"""
        assert guess_mime_from_filename("a/b/c.png") == "image/png"
        assert guess_mime_from_filename("report.PDF") == "application/pdf"

    def test_unknown_extension_defaults_to_pdf(self):
        """未知扩展名回退默认 application/pdf。"""
        assert guess_mime_from_filename("notes.txt") == "application/pdf"
        assert guess_mime_from_filename("archive.zip") == "application/pdf"

    def test_no_extension_defaults_to_pdf(self):
        """无后缀文件名回退默认值。"""
        assert guess_mime_from_filename("README") == "application/pdf"


class TestGuessMimeFromBytes:
    """guess_mime_from_bytes 魔数嗅探边缘用例。"""

    @pytest.mark.parametrize(
        "payload,expected",
        [
            (b"\x89PNG\r\n\x1a\n\x00\x00", "image/png"),
            (b"%PDF-1.7...", "application/pdf"),
            (b"\xff\xd8\xff\xe0", "image/jpeg"),
            (b"GIF87a", "image/gif"),
            (b"GIF89a", "image/gif"),
            (b"BM\x00\x00", "image/bmp"),
            (b"II*\x00", "image/tiff"),
            (b"MM\x00*", "image/tiff"),
            (b"RIFF\x00\x00\x00\x00WEBP", "image/webp"),
            (b"PK\x03\x04...", "application/pdf"),  # ZIP 容器归到 pdf
        ],
    )
    def test_magic_numbers(self, payload, expected):
        """各文件类型魔数正确识别。"""
        assert guess_mime_from_bytes(payload) == expected

    def test_empty_bytes_defaults_to_pdf(self):
        """空内容回退默认 application/pdf。"""
        assert guess_mime_from_bytes(b"") == "application/pdf"

    def test_unknown_bytes_defaults_to_pdf(self):
        """无匹配魔数回退默认值。"""
        assert guess_mime_from_bytes(b"\x00\x01\x02random") == "application/pdf"

    def test_short_riff_not_webp(self):
        """RIFF 但偏移 8:12 非 WEBP 不命中 webp。"""
        assert guess_mime_from_bytes(b"RIFF\x00\x00\x00\x00WAVE") == "application/pdf"


class TestOfficeAndDocumentChecks:
    """is_office_file / is_document_file 边缘用例。"""

    @pytest.mark.parametrize("name", ["a.docx", "B.PPTX", "sheet.Xlsx"])
    def test_office_true(self, name):
        """Office 文档扩展名返回 True（大小写无关）。"""
        assert is_office_file(name) is True

    @pytest.mark.parametrize("name", ["a.pdf", "b.png", "c.txt"])
    def test_office_false(self, name):
        """非 Office 扩展名返回 False。"""
        assert is_office_file(name) is False

    @pytest.mark.parametrize("name", ["a.pdf", "a.docx", "a.pptx", "a.xlsx"])
    def test_document_true(self, name):
        """文档类型（含 Office）返回 True。"""
        assert is_document_file(name) is True

    @pytest.mark.parametrize("name", ["a.png", "a.jpg", "a.gif", "a.bmp"])
    def test_document_false_images(self, name):
        """图片扩展名返回 False。"""
        assert is_document_file(name) is False

    def test_document_extensions_constant(self):
        """DOCUMENT_EXTENSIONS 常量包含四类文档。"""
        assert DOCUMENT_EXTENSIONS == frozenset({".pdf", ".docx", ".pptx", ".xlsx"})


class TestFileFilters:
    """文件过滤器常量完整性。"""

    def test_filters_are_nonempty_strings(self):
        """三个过滤器均为非空字符串。"""
        for flt in (FILE_FILTER_IMAGES, FILE_FILTER_DOCUMENTS, FILE_FILTER_ALL):
            assert isinstance(flt, str)
            assert flt.strip() != ""

    def test_all_filter_combines_image_and_document(self):
        """FILE_FILTER_ALL 同时包含图片与文档扩展名提示。"""
        assert "*.pdf" in FILE_FILTER_ALL
        assert "*.png" in FILE_FILTER_ALL
        assert "*.docx" in FILE_FILTER_ALL
