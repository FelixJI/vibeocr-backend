"""Convert live :class:`OCRResult` instances to JSON-native wire payloads.

The supervisor's v2 HTTP transport stores each job-item result as an opaque
``dict[str, Any]`` (see :class:`vibeocr.runtime_contracts.ResultEntry`). The
adapters that drive real models (Paddle / MinerU) return live
:class:`OCRResult` objects, so they must be converted to a dict before the
result is stored on the job record.

This module is the single source of truth for that conversion. It produces a
key set aligned with the dict shapes already consumed downstream:

- :meth:`pdf_service.add_text_layer_batch` reads ``text_blocks`` + ``preproc_angle``;
- the v2 ``/v2/export`` endpoint reads ``raw_text`` / ``markdown_text`` /
  ``html_text`` / ``content_list``.

Heavy binary fields (e.g. ``images`` values, ``preprocessed_image``) are NOT
inlined into the JSON payload — they can be large and break the HTTP body
budget. Only structural counts/keys are surfaced; callers that need the raw
bytes fetch them through a dedicated streaming endpoint later.
"""

from __future__ import annotations

from typing import Any

# Keys every serialized text_block dict carries (order-stable for readability).
_TEXT_BLOCK_KEYS = (
    "text",
    "score",
    "bbox",
    "polygon",
    "page_idx",
    "is_manually_edited",
    "content_index",
    "content_id",
    "label",
    "order",
)


def text_block_to_dict(block: Any) -> dict[str, Any]:
    """Convert a :class:`TextBlock` to a JSON-native dict.

    ``block`` is duck-typed (anything with the TextBlock attributes) so this
    works against either the client-py or backend copy of the dataclass.
    ``bbox`` (4-tuple) and ``polygon`` (N-tuple) are tuples in the dataclass
    but become JSON lists here so they round-trip through ``json.dumps``.
    """
    bbox = getattr(block, "bbox", None)
    polygon = getattr(block, "polygon", None)
    return {
        "text": getattr(block, "text", ""),
        "score": float(getattr(block, "score", 0.0)),
        "bbox": list(bbox) if isinstance(bbox, (tuple, list)) else bbox,
        "polygon": list(polygon) if isinstance(polygon, (tuple, list)) else polygon,
        "page_idx": getattr(block, "page_idx", None),
        "is_manually_edited": bool(getattr(block, "is_manually_edited", False)),
        "content_index": getattr(block, "content_index", None),
        "content_id": getattr(block, "content_id", None),
        "label": getattr(block, "label", "text"),
        "order": getattr(block, "order", -1),
    }


def ocr_result_to_payload(result: Any) -> dict[str, Any]:
    """Convert an OCR result to a JSON-native wire payload.

    - ``dict`` inputs are returned unchanged so test fakes / pre-serialized
      results keep working without conversion.
    - :class:`OCRResult` instances are serialized via :func:`text_block_to_dict`
      and the field set documented in this module's docstring.

    Anything else is wrapped as ``{"text": str(result)}`` as a last-resort
    fallback so a misbehaving adapter still yields a non-empty payload rather
    than raising during result storage.
    """
    if isinstance(result, dict):
        _validate_table_blocks(result.get("content_list", ()))
        return result

    # Duck-type an OCRResult: it must expose the core text fields. We avoid a
    # hard import so this works against either the client-py or backend copy
    # of the dataclass (they share the top-level ``vibeocr`` namespace).
    if not (hasattr(result, "raw_text") and hasattr(result, "text_blocks")):
        return {"text": str(result)}

    text_blocks = [
        text_block_to_dict(b)
        for b in getattr(result, "text_blocks", []) or []
        if b is not None
    ]
    text_with_scores = [
        [t, float(s)]
        for t, s in (getattr(result, "text_with_scores", []) or [])
    ]
    low_confidence_items = [
        [t, float(s)]
        for t, s in (getattr(result, "low_confidence_items", []) or [])
    ]

    # images: do NOT inline raw bytes. Surface only structural information so
    # the HTTP payload stays small and JSON-safe.
    images = getattr(result, "images", {}) or {}
    images_summary: dict[str, Any] = {}
    if isinstance(images, dict):
        for name, value in images.items():
            if isinstance(value, (bytes, bytearray)):
                images_summary[name] = {"present": True, "size": len(value)}
            else:
                images_summary[name] = {"present": True}

    content_list = list(getattr(result, "content_list", []) or [])
    _validate_table_blocks(content_list)
    payload: dict[str, Any] = {
        "raw_text": getattr(result, "raw_text", "") or "",
        "markdown_text": getattr(result, "markdown_text", "") or "",
        "html_text": getattr(result, "html_text", "") or "",
        "avg_score": float(getattr(result, "avg_score", 0.0) or 0.0),
        "pipeline_type": getattr(result, "pipeline_type", "OCR") or "OCR",
        "preproc_angle": int(getattr(result, "preproc_angle", 0) or 0),
        "content_list": content_list,
        "text_with_scores": text_with_scores,
        "low_confidence_items": low_confidence_items,
        "text_blocks": text_blocks,
        "images": images_summary,
        "image_width": int(getattr(result, "image_width", 0) or 0),
        "image_height": int(getattr(result, "image_height", 0) or 0),
    }
    return payload


def ocr_result_from_payload(payload: dict[str, Any]) -> Any:
    """Rebuild the stable client model from one typed ``ocr.v1`` payload."""
    from vibeocr.backend.models.ocr_result import OCRResult, TextBlock

    _validate_table_blocks(payload.get("content_list", ()))
    blocks: list[TextBlock] = []
    for raw in payload.get("text_blocks", []) or []:
        if not isinstance(raw, dict):
            continue
        bbox = raw.get("bbox")
        polygon = raw.get("polygon")
        blocks.append(
            TextBlock(
                text=str(raw.get("text", "")),
                score=float(raw.get("score", 0.0)),
                bbox=tuple(bbox) if isinstance(bbox, list) else None,
                polygon=(
                    tuple(polygon) if isinstance(polygon, list) else None
                ),
                page_idx=raw.get("page_idx"),
                is_manually_edited=bool(
                    raw.get("is_manually_edited", False)
                ),
                content_index=raw.get("content_index"),
                content_id=(
                    str(raw["content_id"])
                    if raw.get("content_id") is not None
                    else None
                ),
                label=str(raw.get("label", "text")),
                order=int(raw.get("order", -1)),
            )
        )
    return OCRResult(
        raw_text=str(payload.get("raw_text", payload.get("text", ""))),
        markdown_text=str(payload.get("markdown_text", "")),
        html_text=str(payload.get("html_text", "")),
        text_with_scores=[
            (str(item[0]), float(item[1]))
            for item in payload.get("text_with_scores", []) or []
            if isinstance(item, (list, tuple)) and len(item) >= 2
        ],
        avg_score=float(payload.get("avg_score", 0.0)),
        low_confidence_items=[
            (str(item[0]), float(item[1]))
            for item in payload.get("low_confidence_items", []) or []
            if isinstance(item, (list, tuple)) and len(item) >= 2
        ],
        pipeline_type=str(payload.get("pipeline_type", "OCR")),
        content_list=list(payload.get("content_list", []) or []),
        text_blocks=blocks,
        image_width=int(payload.get("image_width", 0)),
        image_height=int(payload.get("image_height", 0)),
        preproc_angle=int(payload.get("preproc_angle", 0)),
    )


def _validate_table_blocks(content_list: Any) -> None:
    """Reject unsupported canonical table versions at transport boundaries."""

    from vibeocr.backend.tables.blocks import validate_table_blocks

    validate_table_blocks(content_list)


__all__ = [
    "ocr_result_from_payload",
    "ocr_result_to_payload",
    "text_block_to_dict",
]
