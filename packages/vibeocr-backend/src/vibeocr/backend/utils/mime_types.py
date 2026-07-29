"""共享 MIME 类型映射和文件过滤器

单一来源（Single Source of Truth），所有 MIME 类型映射和文件对话框过滤器统一定义。
"""

from __future__ import annotations

from pathlib import Path

# 扩展名（小写，带点） → MIME 类型
EXT_TO_MIME: dict[str, str] = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".jp2": "image/jp2",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

# MIME → 扩展名（反向映射，取第一个匹配的）
_MIME_TO_EXT: dict[str, str] = {v: k for k, v in EXT_TO_MIME.items()}


def extension_to_mime(ext: str) -> str | None:
    """扩展名 → MIME 类型。ext 需带前导点，如 '.png'。"""
    return EXT_TO_MIME.get(ext.lower())


def mime_to_extension(mime: str) -> str | None:
    """MIME 类型 → 扩展名（带前导点）。"""
    return _MIME_TO_EXT.get(mime)


# 文件对话框过滤器
FILE_FILTER_IMAGES = "图片 (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.tif *.webp *.jp2)"
FILE_FILTER_DOCUMENTS = "文档 (*.pdf *.docx *.pptx *.xlsx)"
FILE_FILTER_ALL = (
    "所有支持的格式 (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.tif *.webp *.jp2"
    " *.pdf *.docx *.pptx *.xlsx)"
)


def guess_mime_from_filename(filename: str) -> str:
    """从文件名猜测 MIME 类型，未知时默认 application/pdf。"""
    suffix = Path(filename).suffix.lower()
    return EXT_TO_MIME.get(suffix, "application/pdf")


def guess_mime_from_bytes(data: bytes) -> str:
    """从文件字节内容嗅探 MIME 类型，未知时默认 application/pdf。

    仅做轻量魔数（magic number）匹配，用于调用方无文件名时仍能正确
    路由到 MinerU 文档解析服务（mime 决定扩展名，进而影响 mineru-api 解析）。
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:4] == b"%PDF":
        return "application/pdf"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data.startswith(b"BM"):
        return "image/bmp"
    if data[:4] == b"II*\x00" or data[:4] == b"MM\x00*":
        return "image/tiff"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    # DOCX/PPTX/XLSX 是 ZIP 容器，统一归到文档类型
    if data[:4] == b"PK\x03\x04":
        return "application/pdf"
    return "application/pdf"


def is_office_file(path_or_name: str) -> bool:
    """判断文件是否为 Office 文档（docx/pptx/xlsx）。"""
    suffix = Path(path_or_name).suffix.lower()
    return suffix in {".docx", ".pptx", ".xlsx"}


# 非图片文档扩展名（需要 MinerU 文档解析管道处理）
DOCUMENT_EXTENSIONS = frozenset({".pdf", ".docx", ".pptx", ".xlsx"})


def is_document_file(path_or_name: str) -> bool:
    """判断文件是否为文档类型（PDF / Office），即必须使用 MinerU 管道的文件。

    图片文件返回 False，表示可以走 PaddleX 管道。
    """
    return Path(path_or_name).suffix.lower() in DOCUMENT_EXTENSIONS
