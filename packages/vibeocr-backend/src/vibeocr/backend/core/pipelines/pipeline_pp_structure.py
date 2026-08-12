# src/vibeocr/core/pipelines/pipeline_pp_structure.py
"""PP-StructureV3 管道选项与规格

定义 PP-StructureV3 管道的选项类和 PipelineSpec，
支持文档结构分析，包括表格、公式、印章、图表识别。
"""

from __future__ import annotations

import gc
import html
import io
import logging
import re as _re
from dataclasses import dataclass, replace
from typing import Any

from vibeocr.backend.core.pipelines.base_options import BasePipelineOptions
from vibeocr.backend.core.pipelines.registry import PipelineSpec
from vibeocr.backend.tables.blocks import (
    canonicalize_table_block,
    table_model_from_block,
)
from vibeocr.backend.tables.projections import (
    table_model_to_markdown,
    table_model_to_plain_text,
)
from vibeocr.backend.tables.reducer import rebuild_result_projections
from vibeocr.runtime_contracts.contracts.tables import TableProvenanceV1

_logger = logging.getLogger(__name__)

# 表格 HTML 解析正则
_RE_TABLE = _re.compile(r"(<table\b.*?</table>)", _re.DOTALL | _re.IGNORECASE)
_RE_TR = _re.compile(r"<tr[^>]*>(.*?)</tr>", _re.DOTALL | _re.IGNORECASE)
_RE_TD = _re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", _re.DOTALL | _re.IGNORECASE)


def _extract_table_html(html_str: str) -> str:
    """从 HTML 字符串中提取第一个 <table>...</table>"""
    match = _RE_TABLE.search(html_str)
    return match.group(1) if match else html_str


def _html_table_to_markdown(html: str) -> str:
    """将 HTML 表格转换为 Markdown 格式"""
    rows: list[list[str]] = []
    for tr_match in _RE_TR.finditer(html):
        cells = [
            _re.sub(r"<[^>]+>", "", td.group(1)).strip().replace("|", "\\|")
            for td in _RE_TD.finditer(tr_match.group(1))
        ]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    max_cols = max(len(r) for r in rows)
    for r in rows:
        r.extend("" for _ in range(max_cols - len(r)))
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join("---" for _ in range(max_cols)) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(part for part in (header, sep, body) if part)


def _consume_generator_safely(output) -> list:
    """Consume model output while restoring the caller's GC state."""
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        return list(output)
    finally:
        if gc_was_enabled:
            gc.enable()


def _build_ocr_result(
    raw_text: str,
    markdown_text: str = "",
    html_text: str = "",
    text_with_scores: list[tuple[str, float]] | None = None,
    pipeline_type: str = "PP-StructureV3",
    images: dict[str, Any] | None = None,
    text_blocks: list | None = None,
    content_list: list[dict[str, Any]] | None = None,
) -> Any:
    """构建 OCRResult 对象"""
    from vibeocr.backend.models.ocr_result import OCRResult

    if text_with_scores is None:
        text_with_scores = []

    avg_score = 0.0
    if text_with_scores:
        avg_score = sum(s for _, s in text_with_scores) / len(text_with_scores)

    low_confidence_items = [
        (text, score) for text, score in text_with_scores if score < 0.80
    ]

    final_html = html_text or raw_text

    return OCRResult(
        raw_text=raw_text,
        markdown_text=markdown_text or raw_text,
        html_text=final_html,
        text_with_scores=text_with_scores,
        avg_score=avg_score,
        low_confidence_items=low_confidence_items,
        pipeline_type=pipeline_type,
        images=images or {},
        text_blocks=text_blocks or [],
        content_list=content_list or [],
    )


@dataclass
class PPStructureV3Options(BasePipelineOptions):
    """PP-StructureV3 管道选项

    文档结构分析，支持表格、公式、印章、图表识别。
    """

    pipeline: str = "PP-StructureV3"
    use_doc_orientation_classify: bool = True
    use_doc_unwarping: bool = True
    use_textline_orientation: bool = False
    use_table_recognition: bool = True
    use_formula_recognition: bool = True
    use_seal_recognition: bool = False
    use_chart_recognition: bool = False


def _create_pp_structure_pipeline(device: str, **kwargs: Any) -> Any:
    """创建 PP-StructureV3 管道实例

    额外 kwargs 透传给 PPStructureV3（例如 enable_mkldnn）。
    """
    from paddleocr import PPStructureV3

    return PPStructureV3(device=device, **kwargs)


def _recognize_pp_structure(
    service: Any, image: Any, options: PPStructureV3Options
) -> Any:
    """PP-StructureV3 文档结构分析

    从 OCRService._recognize_structure 迁移而来。
    """
    from vibeocr.backend.models.ocr_result import TextBlock

    pipeline = service.get_or_create_pipeline("PP-StructureV3")
    output = pipeline.predict(
        input=image,
        use_doc_orientation_classify=options.use_doc_orientation_classify,
        use_doc_unwarping=options.use_doc_unwarping,
        use_textline_orientation=options.use_textline_orientation,
        use_table_recognition=options.use_table_recognition,
        use_formula_recognition=options.use_formula_recognition,
        use_seal_recognition=options.use_seal_recognition,
        use_chart_recognition=options.use_chart_recognition,
    )
    output_list = _consume_generator_safely(output)

    preproc_angle = 0
    preprocessed_png: bytes | None = None
    preproc_w = preproc_h = 0
    if output_list:
        res = output_list[0]
        dp_res = res.get("doc_preprocessor_res")
        if dp_res is not None:
            preproc_angle = dp_res.get("angle", 0)
            out_arr = dp_res.get("output_img")
            if out_arr is not None:
                from PIL import Image as _PILImage

                # output_img 已是 RGB，不可做 [::-1] 翻转（否则 R/B 对调）
                rgb = out_arr.copy()
                pil_img = _PILImage.fromarray(rgb)
                preproc_w, preproc_h = pil_img.size
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                preprocessed_png = buf.getvalue()

    text_blocks: list[TextBlock] = []
    text_with_scores: list[tuple[str, float]] = []
    content_list: list[dict[str, Any]] = []
    markdown_parts: list[str] = []
    images: dict[str, Any] = {}
    table_sequence = 0

    for res in output_list:
        # 提取内建 markdown 作为参考
        if hasattr(res, "markdown"):
            md_info = getattr(res, "markdown", None)
            if isinstance(md_info, dict):
                md_text = md_info.get("markdown_texts", "")
                if md_text:
                    markdown_parts.append(md_text)
                md_imgs = md_info.get("markdown_images", {})
                if md_imgs:
                    images.update(md_imgs)

        # 从 parsing_res_list 提取结构化结果
        parsing_res_list: list[Any] = []
        if hasattr(res, "__getitem__"):
            parsing_res_list = (
                res["parsing_res_list"]
                if "parsing_res_list" in (res.keys() if hasattr(res, "keys") else [])
                else []
            )
        if not parsing_res_list and hasattr(res, "parsing_res_list"):
            parsing_res_list = res.parsing_res_list

        for block in parsing_res_list:
            label = getattr(block, "label", "text")
            bbox = getattr(block, "bbox", None)
            content = getattr(block, "content", "")
            order_index = getattr(block, "order_index", -1)
            block_image = getattr(block, "image", None)

            if not content and label not in ("image", "chart"):
                continue

            cl_idx = len(content_list)
            bbox_tuple = (
                (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                if bbox
                else None
            )

            if label == "table":
                table_html = _extract_table_html(content)
                table_id = f"pp-structure-v3-table-{table_sequence}"
                table_sequence += 1
                canonical_block = canonicalize_table_block(
                    {
                        "type": "table",
                        "table_body": table_html,
                        "bbox": bbox_tuple,
                    },
                    table_id=table_id,
                    pipeline="PP-StructureV3",
                )
                table_model = table_model_from_block(canonical_block)
                table_model = replace(
                    table_model,
                    provenance=TableProvenanceV1(
                        pipeline="PP-StructureV3",
                        provider_schema="paddlex-pp-structure-v3",
                    ),
                )
                canonical_block["table"] = table_model.to_payload()
                table_plain_text = table_model_to_plain_text(table_model)
                table_md = table_model_to_markdown(table_model).text
                if table_md:
                    markdown_parts.append(table_md)
                text_blocks.append(
                    TextBlock(
                        text=table_plain_text,
                        score=0.9,
                        bbox=bbox_tuple,
                        label=label,
                        order=order_index if order_index is not None else -1,
                        content_index=cl_idx,
                        content_id=table_id,
                    )
                )
                text_with_scores.append((table_plain_text, 0.9))
                content_list.append(canonical_block)

            elif label == "formula":
                block_id = f"pp-structure-v3-block-{cl_idx}"
                formula_md = f"$${content}$$"
                markdown_parts.append(formula_md)
                text_blocks.append(
                    TextBlock(
                        text=content,
                        score=1.0,
                        bbox=bbox_tuple,
                        label=label,
                        order=order_index if order_index is not None else -1,
                        content_index=cl_idx,
                        content_id=block_id,
                    )
                )
                text_with_scores.append((content, 1.0))
                content_list.append(
                    {
                        "type": "formula",
                        "text": content,
                        "bbox": bbox_tuple,
                        "block_id": block_id,
                    }
                )

            else:
                # text, doc_title, seal, chart, image, etc.
                block_id = f"pp-structure-v3-block-{cl_idx}"
                text_blocks.append(
                    TextBlock(
                        text=content,
                        score=0.9,
                        bbox=bbox_tuple,
                        label=label,
                        order=order_index if order_index is not None else -1,
                        content_index=cl_idx,
                        content_id=block_id,
                    )
                )
                text_with_scores.append((content, 0.9))
                content_entry: dict[str, Any] = {
                    "type": label,
                    "text": content,
                    "bbox": bbox_tuple,
                    "block_id": block_id,
                }
                if block_image and isinstance(block_image, dict):
                    img_path = block_image.get("path", "")
                    if img_path:
                        content_entry["img_path"] = img_path
                content_list.append(content_entry)

    raw_text = "\n".join(b.text for b in text_blocks if b.label not in ("table",))
    markdown_text = "\n\n".join(markdown_parts) if markdown_parts else raw_text

    html_parts = [
        (
            str(block.get("table_body") or "")
            if block.get("type") == "table"
            else f"<p>{html.escape(str(block.get('text') or ''))}</p>"
        )
        for block in content_list
    ]

    result = _build_ocr_result(
        raw_text=raw_text,
        markdown_text=markdown_text,
        html_text="\n".join(part for part in html_parts if part),
        text_with_scores=text_with_scores,
        pipeline_type="PP-StructureV3",
        images=images if images else None,
        text_blocks=text_blocks,
        content_list=content_list,
    )
    rebuild_result_projections(result)
    result.preproc_angle = preproc_angle
    result.preprocessed_image = preprocessed_png
    result.preproc_img_w = preproc_w
    result.preproc_img_h = preproc_h
    return result


PP_STRUCTURE_V3_SPEC = PipelineSpec(
    name="PP-StructureV3",
    display_name="PP-StructureV3",
    description="文档结构分析，支持表格、公式、印章、图表识别",
    options_class=PPStructureV3Options,
    create_pipeline=_create_pp_structure_pipeline,
    recognize=_recognize_pp_structure,
)
