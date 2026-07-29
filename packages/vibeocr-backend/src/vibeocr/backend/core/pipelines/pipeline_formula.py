# src/vibeocr/core/pipelines/pipeline_formula.py
"""公式识别管道选项与规格

定义公式识别管道的选项类和 PipelineSpec，
基于 PPStructureV3，仅启用公式识别功能。
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

from vibeocr.backend.core.pipelines.base_options import BasePipelineOptions
from vibeocr.backend.core.pipelines.registry import PipelineSpec


@dataclass
class FormulaRecognitionOptions(BasePipelineOptions):
    """公式识别管道选项

    基于 PPStructureV3，仅启用公式识别，禁用表格/印章/图表。
    """

    pipeline: str = "FORMULA_RECOGNITION"
    use_doc_orientation_classify: bool = True
    use_doc_unwarping: bool = True
    formula_recognition_model_name: str | None = None
    formula_recognition_model_dir: str | None = None
    formula_recognition_batch_size: int = 1


def _create_formula_pipeline(device: str, **kwargs: Any) -> Any:
    """创建公式识别管道实例

    PaddleOCR 3.x 没有独立的公式识别管道类，
    因此使用 PPStructureV3 作为底层引擎。
    额外 kwargs 透传给 PPStructureV3（例如 enable_mkldnn）。
    """
    from paddleocr import PPStructureV3

    return PPStructureV3(device=device, **kwargs)


def _recognize_formula(
    service: Any, image: Any, options: FormulaRecognitionOptions
) -> Any:
    """执行公式识别并返回 OCRResult

    通过 PPStructureV3 进行识别，但仅提取 label=="formula" 的区块，
    内容以 LaTeX 格式输出，包裹在 $$...$$ 中用于 Markdown 显示。
    """
    from enum import Enum

    from vibeocr.backend.models.ocr_result import OCRResult, TextBlock

    pipeline_name = (
        options.pipeline.value
        if isinstance(options.pipeline, Enum)
        else options.pipeline
    )
    pipeline = service.get_or_create_pipeline(pipeline_name)

    predict_kwargs: dict[str, Any] = {
        "use_doc_orientation_classify": options.use_doc_orientation_classify,
        "use_doc_unwarping": options.use_doc_unwarping,
        "use_table_recognition": False,
        "use_formula_recognition": True,
        "use_seal_recognition": False,
        "use_chart_recognition": False,
    }
    if options.formula_recognition_batch_size != 1:
        predict_kwargs["formula_recognition_batch_size"] = (
            options.formula_recognition_batch_size
        )

    output = pipeline.predict(input=image, **predict_kwargs)
    output_list = list(output)

    text_blocks: list[TextBlock] = []
    text_with_scores: list[tuple[str, float]] = []
    markdown_parts: list[str] = []
    content_list: list[dict[str, Any]] = []

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

    for res in output_list:
        # PaddleX 结果是 dict 子类，parsing_res_list 是 dict key（非属性），
        # 必须用下标取值；getattr 对 dict 会恒返回默认值 []（属性不存在）。
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
            if not content:
                continue
            if label != "formula":
                continue

            cl_idx = len(content_list)
            bbox_tuple = (
                (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                if bbox
                else None
            )
            formula_md = f"$${content}$$"
            markdown_parts.append(formula_md)
            text_blocks.append(
                TextBlock(
                    text=content,
                    score=1.0,
                    bbox=bbox_tuple,
                    label=label,
                    order=order_index or -1,
                    content_index=cl_idx,
                )
            )
            text_with_scores.append((content, 1.0))
            content_list.append(
                {"type": "formula", "text": content, "bbox": bbox_tuple}
            )

    raw_text = "\n".join(b.text for b in text_blocks)
    markdown_text = "\n\n".join(markdown_parts) if markdown_parts else raw_text

    from vibeocr.backend.utils.markdown_converter import markdown_to_html

    result = OCRResult(
        raw_text=raw_text,
        markdown_text=markdown_text,
        html_text=markdown_to_html(markdown_text) if markdown_text else "",
        text_with_scores=text_with_scores,
        pipeline_type="FORMULA_RECOGNITION",
        text_blocks=text_blocks,
        content_list=content_list,
    )
    result.preproc_angle = preproc_angle
    result.preprocessed_image = preprocessed_png
    result.preproc_img_w = preproc_w
    result.preproc_img_h = preproc_h
    return result


FORMULA_RECOGNITION_SPEC = PipelineSpec(
    name="FORMULA_RECOGNITION",
    display_name="公式识别",
    description="独立数学公式识别（LaTeX 输出）",
    options_class=FormulaRecognitionOptions,
    create_pipeline=_create_formula_pipeline,
    recognize=_recognize_formula,
)
