"""二维码/条形码识别（解码）服务"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlparse

from PIL import Image

logger = logging.getLogger(__name__)

_MAX_DECODE_DIM = 4096  # 超过此尺寸先缩放，防止 pyzbar OOM/超时


@dataclass
class DecodedItem:
    """单条解码结果。"""

    data: str
    type: str
    is_url: bool


def _is_http_url(value: str) -> bool:
    """严格判定 http/https URL，拒绝 javascript:/file: 等其他 scheme。

    scheme 大小写不敏感（部分二维码生成器会大写化 scheme），但前缀
    快速检查已覆盖绝大多数情况；urlparse 会归一化 scheme 到小写。
    """
    if not value.lower().startswith(("http://", "https://")):
        return False
    try:
        parsed = urlparse(value)
    except (ValueError, TypeError):
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


class QrcodeDecodeService:
    """二维码和条形码识别服务（基于 pyzbar）。"""

    def default_options(self) -> dict:
        return {
            "max_decode_dim": _MAX_DECODE_DIM,
        }

    def decode(self, image: Image.Image) -> list[DecodedItem]:
        from pyzbar.pyzbar import (  # type: ignore[import-not-found]
            decode as _zbar_decode,
        )

        # 大图保护：任一边超过上限先等比缩放（在副本上操作，不改原图）
        max_dim = max(image.size)
        opts_max = _MAX_DECODE_DIM
        if max_dim > opts_max:
            working = image.copy()
            working.thumbnail((opts_max, opts_max))
        else:
            working = image

        # pyzbar 优先用灰度
        gray = working.convert("L") if working.mode != "L" else working

        raw_results = _zbar_decode(gray)
        items: list[DecodedItem] = []
        for r in raw_results:
            try:
                data = r.data.decode("utf-8", errors="replace")
            except Exception:
                continue
            if not data.strip():
                continue
            items.append(DecodedItem(data=data, type=r.type, is_url=_is_http_url(data)))
        return items

    def decode_bytes(self, data: bytes) -> list[DecodedItem]:
        import io

        img = Image.open(io.BytesIO(data))
        return self.decode(img)

    def decode_file(self, path: str) -> list[DecodedItem]:
        img = Image.open(path)
        return self.decode(img)
