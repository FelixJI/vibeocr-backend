from __future__ import annotations

import pytest
from vibeocr.backend.models.ocr_result import (
    OCRResult,
    normalize_bbox,
    normalize_polygon,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ((0.1, 0.2, 0.8, 1.0), (100.0, 200.0, 800.0, 1000.0)),
        ((10, 20, 900, 1000), (10.0, 20.0, 900.0, 1000.0)),
    ],
)
def test_normalize_bbox_handles_normalized_ranges(raw, expected) -> None:
    assert normalize_bbox(raw) == expected


def test_normalize_bbox_scales_pixel_coordinates_per_axis() -> None:
    assert normalize_bbox((0, 0, 2000, 1000), img_w=4000, img_h=2000) == (
        0.0,
        0.0,
        500.0,
        500.0,
    )


def test_normalize_bbox_preserves_unscaled_pixels_and_warns(caplog) -> None:
    raw = (0, 0, 1600, 1200)

    assert normalize_bbox(raw) == (0.0, 0.0, 1600.0, 1200.0)
    assert "unexpected pixel coords" in caplog.text


def test_normalize_polygon_accepts_nested_points() -> None:
    polygon = [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]

    assert normalize_polygon(polygon) == (
        100.0,
        200.0,
        900.0,
        200.0,
        900.0,
        800.0,
        100.0,
        800.0,
    )


def test_normalize_polygon_scales_x_and_y_with_independent_dimensions() -> None:
    polygon = (0, 0, 4000, 0, 4000, 2000, 0, 2000)

    assert normalize_polygon(polygon, img_w=4000, img_h=2000) == (
        0.0,
        0.0,
        1000.0,
        0.0,
        1000.0,
        1000.0,
        0.0,
        1000.0,
    )


@pytest.mark.parametrize(
    "polygon",
    [
        (0, 0, 1, 1, 2, 2),
        (0, 0, 1, 1, 2, 2, 3, 3, 4),
        [[0, 0], [1, 1], "invalid", [2, 2]],
    ],
)
def test_normalize_polygon_rejects_incomplete_geometry(polygon) -> None:
    assert normalize_polygon(polygon) is None


@pytest.mark.parametrize(
    ("result", "rich", "display", "copy"),
    [
        (OCRResult(raw_text="plain"), False, "plain", "plain"),
        (
            OCRResult(
                raw_text="plain",
                markdown_text="**plain**",
                html_text="<b>plain</b>",
            ),
            True,
            "<b>plain</b>",
            "**plain**",
        ),
        (OCRResult(raw_text="same", html_text="same"), False, "same", "same"),
    ],
)
def test_result_text_properties_follow_documented_fallbacks(
    result: OCRResult,
    rich: bool,
    display: str,
    copy: str,
) -> None:
    assert result.has_rich_content is rich
    assert result.display_text == display
    assert result.copy_text == copy


def test_normalize_polygon_passthrough_when_pixel_coords_without_dims() -> None:
    """max_val>1001 但未提供 img_w/img_h 时直接 passthrough（走 line 89）。"""
    poly = [[0, 0], [2000, 0], [2000, 1000], [0, 1000]]
    result = normalize_polygon(poly)
    assert result == (0.0, 0.0, 2000.0, 0.0, 2000.0, 1000.0, 0.0, 1000.0)


def test_ocr_result_has_content_list_property() -> None:
    """has_content_list 反映 content_list 是否非空。"""
    empty = OCRResult(raw_text="x")
    assert empty.has_content_list is False
    with_content = OCRResult(raw_text="x", content_list=[{"type": "text", "text": "a"}])
    assert with_content.has_content_list is True
