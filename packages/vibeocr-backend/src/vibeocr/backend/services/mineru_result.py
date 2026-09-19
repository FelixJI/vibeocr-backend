"""Project native MinerU 4 outputs into the existing VibeOCR result contract."""

from __future__ import annotations

import base64
import binascii
import io
import re
import zipfile
from dataclasses import replace
from pathlib import PurePosixPath

from vibeocr.backend.models.ocr_result import (
    DISCARDED_BLOCK_TYPES,
    OCRResult,
    TextBlock,
    normalize_bbox,
)
from vibeocr.backend.tables.blocks import (
    canonicalize_table_block,
    table_model_from_block,
)
from vibeocr.backend.tables.projections import table_model_to_plain_text
from vibeocr.backend.tables.reducer import rebuild_result_projections
from vibeocr.runtime_contracts.contracts.tables import TableProvenanceV1

from .mineru_api import MineruApiError, MineruDocument, object_value, string_value

_TYPE_MAP = {"doc_title": "title", "paragraph_title": "title", "paragraph": "text"}


def _plain_content(value: object) -> str:
    """Read semantic body/span content, never rendered Markdown wrappers."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            _plain_content(object_value(span, "inline span").get("content"))
            for span in value
        )
    raise MineruApiError("Invalid MinerU semantic content")


def _list_items(value: object) -> list[str]:
    if not isinstance(value, list):
        raise MineruApiError("Invalid MinerU semantic list")
    items: list[str] = []
    for value_item in value:
        item = object_value(value_item, "list item")
        if item.get("type") == "list":
            items.extend(_list_items(item.get("content")))
        else:
            items.append(_plain_content(item.get("content")))
    return items


def _image_paths(value: object) -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        path = value.get("image_path")
        if isinstance(path, str) and path:
            paths.add(path)
        for child in value.values():
            paths.update(_image_paths(child))
    elif isinstance(value, list):
        for child in value:
            paths.update(_image_paths(child))
    return paths


def _read_images(document: MineruDocument) -> dict[str, bytes]:
    images: dict[str, bytes] = {}
    paths = _image_paths(document.middle_json)
    if not paths:
        return images
    try:
        with zipfile.ZipFile(io.BytesIO(document.archive)) as archive:
            names = archive.namelist()
            for path in sorted(paths):
                normalized = PurePosixPath(path)
                if (
                    "\\" in path
                    or ":" in path
                    or normalized.is_absolute()
                    or ".." in normalized.parts
                    or normalized.as_posix() != path
                ):
                    raise MineruApiError("Unsafe MinerU image sidecar path")
                if names.count(path) != 1:
                    raise MineruApiError("Missing or duplicate MinerU image sidecar")
                info = archive.getinfo(path)
                if info.file_size > 128 * 1024 * 1024:
                    raise MineruApiError("MinerU image sidecar exceeds size limit")
                images[path] = archive.read(info)
    except zipfile.BadZipFile as exc:
        raise MineruApiError("Invalid MinerU output archive") from exc
    return images


def project_document(document: MineruDocument) -> OCRResult:
    middle = document.middle_json
    if (
        middle.get("schema") != "docvortex.middle"
        or middle.get("schema_version") != "2.0"
    ):
        raise MineruApiError("Unsupported MinerU middle JSON schema")
    pages = document.structured_content.get("pages")
    source_pages = middle.get("pages")
    if not isinstance(pages, list) or not isinstance(source_pages, list):
        raise MineruApiError("MinerU output has no pages")
    page_indices = [
        object_value(page, "source page").get("page_idx") for page in source_pages
    ]
    if [object_value(page, "page").get("page_idx") for page in pages] != page_indices:
        raise MineruApiError("MinerU output page identity mismatch")
    if any(type(index) is not int or index < 0 for index in page_indices):
        raise MineruApiError("Invalid MinerU page index")
    if len(set(page_indices)) != len(page_indices):
        raise MineruApiError("Duplicate MinerU page index")
    images = _read_images(document)
    blocks: list[dict] = []
    texts: list[TextBlock] = []
    raw_parts: list[str] = []
    for page, source_page in zip(pages, source_pages, strict=True):
        native_page = object_value(page, "page")
        page_idx = native_page["page_idx"]
        page_blocks = native_page.get("blocks")
        if not isinstance(page_blocks, list):
            raise MineruApiError("Invalid MinerU page blocks")
        source_blocks = object_value(source_page, "source page").get("blocks")
        if not isinstance(source_blocks, list) or len(source_blocks) != len(
            page_blocks
        ):
            raise MineruApiError("MinerU block identity mismatch")
        for index, value in enumerate(page_blocks):
            native = object_value(value, "block")
            kind = string_value(native.get("type"), "block type")
            source_block = object_value(source_blocks[index], "source block")
            if source_block.get("type") != kind:
                raise MineruApiError("MinerU block type mismatch")
            content = native.get("content")
            if not isinstance(content, str):
                raise MineruApiError("Invalid MinerU structured block content")
            block_id = f"mineru4-page-{page_idx}-block-{index}"
            bbox = native.get("bbox")
            if bbox is not None and (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or any(type(x) not in (float, int) or not 0 <= x <= 1 for x in bbox)
            ):
                raise MineruApiError("Invalid MinerU normalized bounding box")
            block: dict = {
                "block_id": block_id,
                "type": _TYPE_MAP.get(kind, kind),
                "text": content
                if kind in {"table", "image", "chart", "code", "list"}
                else _plain_content(source_block.get("content")),
                "page_idx": page_idx,
                "source": {"provider_schema": "docvortex.middle/2.0", "block": native},
            }
            if bbox is not None:
                block["bbox"] = list(normalize_bbox(bbox))
            image_source = native.get("image_source")
            if isinstance(image_source, str) and image_source in images:
                block["img_path"] = image_source
            elif isinstance(image_source, str) and image_source.startswith(
                "data:image/"
            ):
                header, separator, encoded = image_source.partition(",")
                if not separator or not header.endswith(";base64"):
                    raise MineruApiError("Unsupported MinerU embedded image")
                extensions = {
                    "image/png": "png",
                    "image/jpeg": "jpg",
                    "image/webp": "webp",
                    "image/gif": "gif",
                }
                extension = extensions.get(header[5:].removesuffix(";base64"))
                if extension is None or len(encoded) > 180 * 1024 * 1024:
                    raise MineruApiError("Unsupported MinerU embedded image")
                try:
                    data = base64.b64decode(encoded, validate=True)
                except binascii.Error as exc:
                    raise MineruApiError("Invalid MinerU embedded image") from exc
                image_path = f"images/{block_id}.{extension}"
                images[image_path] = data
                block["img_path"] = image_path
                block["source"]["block"] = {
                    k: v for k, v in native.items() if k != "image_source"
                }
            if block["type"] == "title":
                block["level"] = native.get("level", 1)
            if kind in ("table", "image", "chart", "code"):
                source_children = source_block.get("content")
                if not isinstance(source_children, list):
                    raise MineruApiError("Invalid MinerU visual content")
                for suffix in ("caption", "footnote"):
                    block[f"{kind}_{suffix}"] = [
                        _plain_content(child.get("content"))
                        for child in source_children
                        if isinstance(child, dict)
                        and child.get("type") == f"{kind}_{suffix}"
                    ]
            if kind == "table":
                bodies = source_block.get("content", [])
                html_bodies = (
                    [
                        body.get("content")
                        for body in bodies
                        if isinstance(body, dict) and body.get("type") == "table_body"
                    ]
                    if isinstance(bodies, list)
                    else []
                )
                table_html = html_bodies[0] if len(html_bodies) == 1 else content
                if isinstance(table_html, str) and re.search(
                    r"<table\b", table_html, flags=re.IGNORECASE
                ):
                    block["table_body"] = table_html
                    block = canonicalize_table_block(
                        block, table_id=block_id, pipeline="MinerU"
                    )
                    model = table_model_from_block(block)
                    model = replace(
                        model,
                        provenance=TableProvenanceV1(
                            pipeline="MinerU", provider_schema="docvortex.middle/2.0"
                        ),
                    )
                    block["table"] = model.to_payload()
                    block["text"] = table_model_to_plain_text(model)
                else:
                    block["type"] = "table_unparsed"
                    block["source_type"] = "table"
                    block["projection_warnings"] = [
                        f"{block_id}:structured-table-unsupported"
                    ]
            elif kind in ("image", "chart"):
                block["text"] = "\n".join(
                    [
                        content,
                        *block.get(f"{kind}_caption", []),
                        *block.get(f"{kind}_footnote", []),
                    ]
                ).strip()
            elif kind == "code":
                source_content = source_block.get("content")
                bodies = (
                    [
                        body.get("content")
                        for body in source_content
                        if isinstance(body, dict)
                        and body.get("type") in {"code_body", "algorithm_body"}
                    ]
                    if isinstance(source_content, list)
                    else []
                )
                if len(bodies) != 1:
                    raise MineruApiError(
                        "MinerU code block must have one semantic body"
                    )
                block["text"] = block["code_body"] = _plain_content(bodies[0])
            elif kind == "list":
                block["list_items"] = _list_items(source_block.get("content"))
                block["text"] = "\n".join(block["list_items"])
            blocks.append(block)
            if kind in DISCARDED_BLOCK_TYPES:
                continue
            text = block["text"]
            if text:
                raw_parts.append(text)
                if bbox is not None or kind == "table":
                    texts.append(
                        TextBlock(
                            text=text,
                            score=1.0,
                            bbox=normalize_bbox(bbox) if bbox is not None else None,
                            page_idx=page_idx,
                            content_index=len(blocks) - 1,
                            content_id=block_id,
                        )
                    )
    result = OCRResult(
        raw_text="\n".join(raw_parts),
        markdown_text=document.markdown,
        html_text="",
        text_with_scores=[(block.text, block.score) for block in texts],
        avg_score=1.0 if texts else 0.0,
        low_confidence_items=[],
        pipeline_type="MinerU",
        images=images,
        content_list=blocks,
        text_blocks=texts,
    )
    rebuild_result_projections(result)
    return result
