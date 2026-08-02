"""OCR 批次预算与稳定顺序分批。"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Generic, TypeVar

from vibeocr.backend.core.constants import Constants

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class BatchBudget:
    max_items: int
    max_encoded_bytes: int
    max_pixels: int

    def __post_init__(self) -> None:
        if min(self.max_items, self.max_encoded_bytes, self.max_pixels) <= 0:
            raise ValueError("batch budget limits must be positive")

    @classmethod
    def ocr_default(cls) -> BatchBudget:
        return cls(
            max_items=Constants.OCR_BATCH_GPU_SIZE_CAP,
            max_encoded_bytes=Constants.OCR_BATCH_MAX_ENCODED_BYTES,
            max_pixels=Constants.OCR_BATCH_MAX_PIXELS,
        )


@dataclass(frozen=True, slots=True)
class BatchEntry(Generic[T]):
    value: T
    encoded_bytes: int
    pixels: int | None = None


@dataclass(frozen=True, slots=True)
class BatchChunk(Generic[T]):
    entries: tuple[BatchEntry[T], ...]
    encoded_bytes: int
    pixels: int
    oversized_single: bool = False

    @property
    def values(self) -> list[T]:
        return [entry.value for entry in self.entries]


def partition_batches[T](
    entries: Iterable[BatchEntry[T]], budget: BatchBudget
) -> list[BatchChunk[T]]:
    """按数量、encoded bytes、pixels 稳定分批。

    未知 pixels 不计入像素和，但仍受数量/字节约束。任何单项即使超过预算
    也单独进入一批，保证调用者不会因极端输入陷入空批或死循环。
    """
    chunks: list[BatchChunk[T]] = []
    current: list[BatchEntry[T]] = []
    current_bytes = 0
    current_pixels = 0

    def flush() -> None:
        nonlocal current, current_bytes, current_pixels
        if not current:
            return
        only = current[0]
        oversized = len(current) == 1 and (
            only.encoded_bytes > budget.max_encoded_bytes
            or (only.pixels or 0) > budget.max_pixels
        )
        chunks.append(
            BatchChunk(
                entries=tuple(current),
                encoded_bytes=current_bytes,
                pixels=current_pixels,
                oversized_single=oversized,
            )
        )
        current = []
        current_bytes = 0
        current_pixels = 0

    for entry in entries:
        encoded_bytes = max(0, int(entry.encoded_bytes))
        pixels = max(0, int(entry.pixels or 0))
        exceeds = bool(current) and (
            len(current) + 1 > budget.max_items
            or current_bytes + encoded_bytes > budget.max_encoded_bytes
            or current_pixels + pixels > budget.max_pixels
        )
        if exceeds:
            flush()
        current.append(entry)
        current_bytes += encoded_bytes
        current_pixels += pixels
    flush()
    return chunks


def image_pixel_count(
    source: bytes | bytearray | memoryview | str | Path,
) -> int | None:
    """只读取图像头部返回像素数；不可识别时返回 None。"""
    try:
        from PIL import Image

        target = (
            BytesIO(bytes(source))
            if isinstance(source, (bytes, bytearray, memoryview))
            else source
        )
        with Image.open(target) as image:
            width, height = image.size
        pixels = int(width) * int(height)
        return pixels if pixels > 0 else None
    except Exception:
        return None
