"""Unit tests for ocr_result_to_payload (the wire serializer).

These pin the structured key set the supervisor stores on job results and
guard against regressing to the old ``{"text": str(result)}`` fallback that
dropped all structure.
"""

from __future__ import annotations

from vibeocr.backend.models import ocr_result_from_payload, ocr_result_to_payload
from vibeocr.backend.models.ocr_result import OCRResult, TextBlock


def test_dict_input_passes_through_unchanged() -> None:
    payload = {"raw_text": "hi", "markdown_text": "# hi"}
    assert ocr_result_to_payload(payload) is payload


def test_ocr_result_produces_structured_key_set() -> None:
    result = OCRResult(
        raw_text="hello",
        markdown_text="# hello",
        html_text="<h1>hello</h1>",
        avg_score=0.91,
        pipeline_type="OCR",
        preproc_angle=90,
        content_list=[{"type": "text", "text": "hello"}],
        text_with_scores=[("hello", 0.9), ("world", 0.8)],
        text_blocks=[
            TextBlock(text="hello", score=0.9, bbox=(1.0, 2.0, 3.0, 4.0)),
        ],
    )
    payload = ocr_result_to_payload(result)

    # The headline assertion: the structured keys exist, not just `text`.
    for key in (
        "raw_text",
        "markdown_text",
        "html_text",
        "avg_score",
        "pipeline_type",
        "preproc_angle",
        "content_list",
        "text_with_scores",
        "text_blocks",
    ):
        assert key in payload, f"missing key {key!r}"

    assert payload["raw_text"] == "hello"
    assert payload["markdown_text"] == "# hello"
    assert payload["pipeline_type"] == "OCR"
    assert payload["preproc_angle"] == 90
    assert payload["avg_score"] == 0.91
    assert payload["content_list"] == [{"type": "text", "text": "hello"}]


def test_text_blocks_converted_to_dicts_with_list_bbox() -> None:
    result = OCRResult(
        text_blocks=[
            TextBlock(
                text="a",
                score=0.5,
                bbox=(1.0, 2.0, 3.0, 4.0),
                polygon=(0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0),
                page_idx=0,
                label="text",
                order=2,
            ),
        ],
    )
    payload = ocr_result_to_payload(result)
    block = payload["text_blocks"][0]
    assert block["text"] == "a"
    assert block["score"] == 0.5
    # bbox/polygon are tuples in the dataclass but must serialize to JSON lists.
    assert block["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert isinstance(block["bbox"], list)
    assert block["polygon"] == [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    assert block["page_idx"] == 0
    assert block["label"] == "text"
    assert block["order"] == 2


def test_images_bytes_not_inlined_into_payload() -> None:
    # Raw image bytes can be large; the serializer must NOT inline them.
    result = OCRResult(images={"fig1.png": b"\x89PNG large bytes"})
    payload = ocr_result_to_payload(result)
    images = payload["images"]
    assert "fig1.png" in images
    assert images["fig1.png"]["present"] is True
    assert images["fig1.png"]["size"] == len(b"\x89PNG large bytes")
    # The raw bytes must not leak into the JSON payload.
    assert b"\x89PNG large bytes" not in str(payload).encode()


def test_text_with_scores_tuples_become_lists() -> None:
    result = OCRResult(text_with_scores=[("a", 0.1), ("b", 0.2)])
    payload = ocr_result_to_payload(result)
    assert payload["text_with_scores"] == [["a", 0.1], ["b", 0.2]]


def test_non_ocrresult_non_dict_falls_back_to_text_str() -> None:
    payload = ocr_result_to_payload(42)
    assert payload == {"text": "42"}


def test_empty_ocr_result_still_has_keys() -> None:
    payload = ocr_result_to_payload(OCRResult())
    assert payload["raw_text"] == ""
    assert payload["text_blocks"] == []
    assert payload["pipeline_type"] == "OCR"


def test_structured_payload_roundtrips_to_client_model() -> None:
    original = OCRResult(
        raw_text="hello",
        markdown_text="**hello**",
        avg_score=0.8,
        text_blocks=[
            TextBlock(
                text="hello",
                score=0.8,
                bbox=(1.0, 2.0, 3.0, 4.0),
            )
        ],
    )
    restored = ocr_result_from_payload(ocr_result_to_payload(original))

    assert restored.raw_text == "hello"
    assert restored.markdown_text == "**hello**"
    assert restored.text_blocks[0].bbox == (1.0, 2.0, 3.0, 4.0)


def test_canonical_table_and_stable_content_id_roundtrip_strictly() -> None:
    from vibeocr.runtime_contracts.contracts.tables import TableCellV1, TableModelV1

    table = TableModelV1(
        table_id="table-wire",
        row_count=2,
        column_count=2,
        cells=(
            TableCellV1(
                cell_id="merged",
                row=0,
                column=0,
                colspan=2,
                text="标题",
            ),
            TableCellV1(cell_id="left", row=1, column=0, text="左"),
            TableCellV1(cell_id="right", row=1, column=1, text="右"),
        ),
    )
    original = OCRResult(
        content_list=[{"type": "table", "table": table.to_payload()}],
        text_blocks=[
            TextBlock(
                text="标题\n左\t右",
                score=0.9,
                bbox=None,
                content_index=0,
                content_id="block-wire",
                label="table",
            )
        ],
    )

    payload = ocr_result_to_payload(original)
    restored = ocr_result_from_payload(payload)

    assert restored.content_list == original.content_list
    assert restored.text_blocks[0].content_id == "block-wire"


def test_unknown_table_schema_is_rejected_on_both_wire_directions() -> None:
    invalid = {
        "type": "table",
        "table": {
            "schema_version": 999,
            "table_id": "future",
            "row_count": 0,
            "column_count": 0,
            "coordinate_space": "unknown",
            "cells": [],
            "provenance": None,
        },
    }

    for operation in (
        lambda: ocr_result_to_payload(OCRResult(content_list=[invalid])),
        lambda: ocr_result_from_payload({"content_list": [invalid]}),
    ):
        try:
            operation()
        except ValueError as error:
            assert "schema_version" in str(error)
        else:
            raise AssertionError("unknown table schemas must fail at the wire boundary")


def test_images_non_bytes_value_surfaces_present_without_size() -> None:
    """images 值不是 bytes/bytearray 时，只标记 present，不报 size（line 110）。"""
    result = OCRResult(images={"marker.txt": "not-bytes"})
    payload = ocr_result_to_payload(result)
    entry = payload["images"]["marker.txt"]
    assert entry == {"present": True}
    assert "size" not in entry


def test_ocr_result_from_payload_skips_non_dict_text_blocks() -> None:
    """payload.text_blocks 含非 dict 元素时跳过（line 140 continue）。"""
    payload = {
        "raw_text": "x",
        "markdown_text": "",
        "html_text": "",
        "avg_score": 0.0,
        "pipeline_type": "OCR",
        "preproc_angle": 0,
        "content_list": [],
        "text_with_scores": [],
        "low_confidence_items": [],
        "text_blocks": [None, "str-entry", 42, {"text": "valid", "score": 0.9}],
        "images": {},
        "image_width": 0,
        "image_height": 0,
    }
    restored = ocr_result_from_payload(payload)
    assert len(restored.text_blocks) == 1
    assert restored.text_blocks[0].text == "valid"


def test_non_dict_images_attribute_is_ignored_safely() -> None:
    """result.images 不是 dict（如 None 或 list）时不进入汇总（line 105->112）。"""
    result = OCRResult()
    # OCRResult 默认 images={}，这里直接构造一个 images 非 dict 的场景
    object.__setattr__(result, "images", None)
    payload = ocr_result_to_payload(result)
    assert payload["images"] == {}

    object.__setattr__(result, "images", ["not", "a", "dict"])
    payload = ocr_result_to_payload(result)
    assert payload["images"] == {}
