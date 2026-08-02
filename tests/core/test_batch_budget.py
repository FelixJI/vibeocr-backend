"""OCR 批次预算的纯函数边界测试。"""

from __future__ import annotations

from vibeocr.backend.core.batch_budget import BatchBudget, BatchEntry, partition_batches
from vibeocr.backend.core.constants import Constants


def _values(entries, budget):
    return [chunk.values for chunk in partition_batches(entries, budget)]


def test_ocr_default_pixel_budget_is_64m() -> None:
    assert BatchBudget.ocr_default().max_pixels == 64_000_000
    assert Constants.OCR_BATCH_MAX_PIXELS == 64_000_000


def test_ocr_default_a4_300dpi_pages_fit_seven_per_chunk() -> None:
    entries = [
        BatchEntry(value=index, encoded_bytes=1, pixels=8_700_000) for index in range(8)
    ]

    assert [
        len(chunk.entries)
        for chunk in partition_batches(entries, BatchBudget.ocr_default())
    ] == [7, 1]


def test_item_limit_and_order_are_stable():
    entries = [BatchEntry(value=index, encoded_bytes=1, pixels=1) for index in range(5)]
    budget = BatchBudget(max_items=2, max_encoded_bytes=100, max_pixels=100)

    assert _values(entries, budget) == [[0, 1], [2, 3], [4]]


def test_encoded_byte_and_pixel_limits_are_independent():
    budget = BatchBudget(max_items=10, max_encoded_bytes=5, max_pixels=100)
    byte_entries = [
        BatchEntry(value="a", encoded_bytes=3, pixels=10),
        BatchEntry(value="b", encoded_bytes=2, pixels=10),
        BatchEntry(value="c", encoded_bytes=1, pixels=10),
    ]
    assert _values(byte_entries, budget) == [["a", "b"], ["c"]]

    pixel_entries = [
        BatchEntry(value="a", encoded_bytes=1, pixels=60),
        BatchEntry(value="b", encoded_bytes=1, pixels=50),
        BatchEntry(value="c", encoded_bytes=1, pixels=10),
    ]
    assert _values(pixel_entries, budget) == [["a"], ["b", "c"]]


def test_unknown_pixels_fall_back_to_item_and_byte_limits():
    entries = [
        BatchEntry(value="a", encoded_bytes=3, pixels=None),
        BatchEntry(value="b", encoded_bytes=3, pixels=None),
    ]
    budget = BatchBudget(max_items=10, max_encoded_bytes=5, max_pixels=1)

    assert _values(entries, budget) == [["a"], ["b"]]


def test_oversized_single_always_enters_one_batch():
    entries = [
        BatchEntry(value="huge", encoded_bytes=101, pixels=101),
        BatchEntry(value="small", encoded_bytes=1, pixels=1),
    ]
    budget = BatchBudget(max_items=2, max_encoded_bytes=10, max_pixels=10)

    chunks = partition_batches(entries, budget)

    assert [chunk.values for chunk in chunks] == [["huge"], ["small"]]
    assert chunks[0].oversized_single is True
    assert chunks[1].oversized_single is False


def test_batch_budget_rejects_non_positive_limits() -> None:
    """任一限制 <=0 时 raise ValueError（line 25-26）。"""
    import pytest
    from vibeocr.backend.core.batch_budget import BatchBudget

    with pytest.raises(ValueError, match="positive"):
        BatchBudget(max_items=0, max_encoded_bytes=100, max_pixels=100)
    with pytest.raises(ValueError, match="positive"):
        BatchBudget(max_items=1, max_encoded_bytes=-1, max_pixels=100)
    with pytest.raises(ValueError, match="positive"):
        BatchBudget(max_items=1, max_encoded_bytes=100, max_pixels=0)


def test_image_pixel_count_reads_header_from_bytes() -> None:
    """image_pixel_count 从字节读取图像尺寸（line 109-122）。"""
    import io

    from PIL import Image
    from vibeocr.backend.core.batch_budget import image_pixel_count

    img = Image.new("RGB", (10, 20), "red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    assert image_pixel_count(buf.getvalue()) == 200


def test_image_pixel_count_reads_header_from_path(tmp_path) -> None:
    """image_pixel_count 从文件路径读取（line 115 分支）。"""

    from PIL import Image
    from vibeocr.backend.core.batch_budget import image_pixel_count

    img = Image.new("RGB", (4, 5), "blue")
    p = tmp_path / "t.png"
    img.save(p, format="PNG")
    assert image_pixel_count(p) == 20


def test_image_pixel_count_returns_none_on_garbage() -> None:
    """无法识别的数据返回 None（line 121-122）。"""
    from vibeocr.backend.core.batch_budget import image_pixel_count

    assert image_pixel_count(b"not an image") is None
    assert image_pixel_count(b"") is None


def test_partition_batches_empty_entries_returns_empty() -> None:
    """空 entries 返回空 chunks 列表（flush 早返回 line 72）。"""
    from vibeocr.backend.core.batch_budget import BatchBudget, partition_batches

    budget = BatchBudget(max_items=2, max_encoded_bytes=100, max_pixels=100)
    assert partition_batches([], budget) == []
    assert partition_batches(iter([]), budget) == []
