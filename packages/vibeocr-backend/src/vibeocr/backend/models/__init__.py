"""OCR 数据模型"""

from vibeocr.backend.models.ocr_result import OCRResult
from vibeocr.backend.models.ocr_result_serializer import (
    ocr_result_from_payload,
    ocr_result_to_payload,
)

__all__ = [
    "OCRResult",
    "ocr_result_from_payload",
    "ocr_result_to_payload",
]
