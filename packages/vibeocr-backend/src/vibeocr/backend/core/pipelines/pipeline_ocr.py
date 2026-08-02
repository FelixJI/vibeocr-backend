# src/vibeocr/core/pipelines/pipeline_ocr.py
"""通用 OCR 管道选项与规格

定义通用 OCR 管道的选项类和 PipelineSpec，
适用于纯文本识别场景。
"""

from __future__ import annotations

import gc
import io
import logging
import os
import sys
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vibeocr.backend.core.pipelines.base_options import BasePipelineOptions
from vibeocr.backend.core.pipelines.registry import PipelineSpec

if TYPE_CHECKING:
    from vibeocr.backend.models.ocr_result import TextBlock

_logger = logging.getLogger(__name__)


def _isolate_unused_modelscope_import() -> bool:
    """Keep Paddle's process free of Torch when ModelScope is not selected.

    PaddleX 3.7 imports ``modelscope`` unconditionally while defining all
    possible model hosters.  ModelScope imports Torch at module import time,
    even when the configured OCR model source is HuggingFace/BOS/AIStudio.
    On Windows, Paddle GPU and Torch then load competing cuDNN copies in the
    same process and one of them fails with ``WinError 127``.

    MinerU/Torch already runs in a supervisor-owned child process.  For the
    Paddle process, install a deliberately unusable placeholder only when
    ModelScope is not the configured hoster.  An explicit ModelScope choice is
    left untouched so the downloader is never silently faked.
    """
    source = os.environ.get("PADDLE_PDX_MODEL_SOURCE", "huggingface").lower()
    if source == "modelscope" or "modelscope" in sys.modules:
        return False

    placeholder = types.ModuleType("modelscope")

    def unavailable(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError(
            "ModelScope is disabled in the Paddle process; choose "
            "PADDLE_PDX_MODEL_SOURCE=modelscope to enable it explicitly."
        )

    placeholder.snapshot_download = unavailable  # type: ignore[attr-defined]
    sys.modules["modelscope"] = placeholder
    return True


@dataclass
class OCROptions(BasePipelineOptions):
    """通用 OCR 管道选项

    适用于纯文本识别场景。
    """

    pipeline: str = "OCR"
    use_doc_orientation_classify: bool = True
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False


def _create_ocr_pipeline(device: str, **kwargs: Any) -> Any:
    """创建通用 OCR 管道实例

    额外 kwargs 透传给 PaddleOCR（例如 enable_mkldnn）。

    GPU 模式下显式注入 ``text_recognition_batch_size``（对应 PaddleOCR 2.x
    的 ``rec_batch_num``）。PaddleOCR 3.x 默认识别批量偏保守，导致 4090 等
    大显存卡 GPU 占用仅 20-40%；这里在 GPU 模式给一个较大的批量以喂满 GPU。
    调用方显式传入该参数时不覆盖。
    """
    _isolate_unused_modelscope_import()
    from paddleocr import PaddleOCR

    if device == "gpu" and "text_recognition_batch_size" not in kwargs:
        kwargs["text_recognition_batch_size"] = 8
    return PaddleOCR(device=device, **kwargs)


def _consume_generator_safely(output) -> list:
    """安全地消费 generator（禁用 GC 避免 CUDA 内存管理冲突）"""
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        return list(output)
    except Exception as e:
        _logger.error("[安全消费] 消费 generator 时出错: %s", e, exc_info=True)
        return []
    finally:
        if gc_was_enabled:
            gc.enable()


def _extract_bbox(rec_boxes, index: int) -> tuple[float, float, float, float] | None:
    """从 rec_boxes 提取第 index 个文本框的 bbox [x0, y0, x1, y1]

    支持格式:
    - (N, 4): [x0, y0, x1, y1] 轴对齐矩形
    - (N, 4, 2): [[x0,y0], [x1,y1], [x2,y2], [x3,y3]] 四点多边形
    - (N, 2, 2): [[x0,y0], [x1,y1]] 两点矩形
    """
    try:
        box = rec_boxes[index]
        if hasattr(box, "tolist"):
            box = box.tolist()
        if len(box) == 4:
            # (N, 4) 或 (N, 4, 2)
            if isinstance(box[0], (int, float)):
                return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            # (N, 4, 2) 多边形格式: 取外接矩形
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            return (min(xs), min(ys), max(xs), max(ys))
        if len(box) == 2 and len(box[0]) == 2 and len(box[1]) == 2:
            return (
                float(box[0][0]),
                float(box[0][1]),
                float(box[1][0]),
                float(box[1][1]),
            )
    except (IndexError, TypeError, ValueError):
        pass
    return None


def _extract_polygon(rec_polys, index: int) -> tuple[float, ...] | None:
    """从 rec_polys 提取第 index 个文本行的 4 点检测多边形 [x0,y0,x1,y1,...]。

    PaddleOCR 的 DB 检测输出 4 点多边形（顺时针 TL,TR,BR,BL），其顶点排序
    编码阅读方向：横排顶边为长边、竖排左边为长边。AABB 把它塌缩成轴对齐
    矩形会丢失这个方向信号，故单独透传给 PDF 文字层用于判横/竖排。

    与 _extract_bbox 同源数据（rec_polys/dt_polys），但保留 4 点几何而非
    取外接矩形。失败返回 None（调用方回退到 bbox 长宽比）。
    """
    try:
        poly = rec_polys[index]
        if hasattr(poly, "tolist"):
            poly = poly.tolist()
        if len(poly) < 4:
            return None
        flat: list[float] = []
        for pt in poly:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                flat.append(float(pt[0]))
                flat.append(float(pt[1]))
        if len(flat) >= 8 and len(flat) % 2 == 0:
            return tuple(flat)
    except (IndexError, TypeError, ValueError):
        pass
    return None


def _build_ocr_result(
    raw_text: str,
    markdown_text: str = "",
    html_text: str = "",
    text_with_scores: list[tuple[str, float]] | None = None,
    pipeline_type: str = "OCR",
    images: dict[str, Any] | None = None,
    text_blocks: list | None = None,
    content_list: list[dict[str, Any]] | None = None,
) -> Any:
    """构建 OCRResult 对象"""
    from vibeocr.backend.models.ocr_result import OCRResult

    if text_with_scores is None:
        text_with_scores = []

    # 计算平均置信度
    avg_score = 0.0
    if text_with_scores:
        avg_score = sum(s for _, s in text_with_scores) / len(text_with_scores)

    # 收集低置信度项（低于 80%）
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


def _parse_single_result(res: Any) -> tuple[list[tuple[str, float]], list]:
    """从单个 OCRResult/dict 提取 (text, score) 列表与 TextBlock 列表。

    供单图与批量识别共享，统一处理 res 对象的多种字段形态。
    """
    from vibeocr.backend.models.ocr_result import TextBlock

    tws: list[tuple[str, float]] = []
    blocks: list[TextBlock] = []
    try:
        if hasattr(res, "rec_texts") and hasattr(res, "rec_scores"):
            rec_boxes = getattr(res, "rec_boxes", None)
            rec_polys = getattr(res, "rec_polys", None) or getattr(
                res, "dt_polys", None
            )
            for i, (text, score) in enumerate(
                zip(res.rec_texts, res.rec_scores, strict=False)
            ):
                if text:
                    fs = float(score)
                    tws.append((text, fs))
                    bbox = (
                        _extract_bbox(rec_boxes, i) if rec_boxes is not None else None
                    )
                    polygon = (
                        _extract_polygon(rec_polys, i)
                        if rec_polys is not None
                        else None
                    )
                    blocks.append(
                        TextBlock(text=text, score=fs, bbox=bbox, polygon=polygon)
                    )
        elif hasattr(res, "rec_texts"):
            rec_boxes = getattr(res, "rec_boxes", None)
            rec_polys = getattr(res, "rec_polys", None) or getattr(
                res, "dt_polys", None
            )
            for i, text in enumerate(res.rec_texts):
                if text:
                    tws.append((text, 1.0))
                    bbox = (
                        _extract_bbox(rec_boxes, i) if rec_boxes is not None else None
                    )
                    polygon = (
                        _extract_polygon(rec_polys, i)
                        if rec_polys is not None
                        else None
                    )
                    blocks.append(
                        TextBlock(text=text, score=1.0, bbox=bbox, polygon=polygon)
                    )
        elif hasattr(res, "ocr_text"):
            tws.append((res.ocr_text, 1.0))
            blocks.append(TextBlock(text=res.ocr_text, score=1.0, bbox=None))
        elif isinstance(res, dict):
            rec_texts = res.get("rec_texts", [])
            rec_scores = res.get("rec_scores", [])
            rec_boxes = res.get("rec_boxes")
            rec_polys = res.get("rec_polys") or res.get("dt_polys")
            if rec_scores:
                for i, (text, score) in enumerate(
                    zip(rec_texts, rec_scores, strict=False)
                ):
                    if text:
                        fs = float(score)
                        tws.append((text, fs))
                        bbox = (
                            _extract_bbox(rec_boxes, i)
                            if rec_boxes is not None
                            else None
                        )
                        polygon = (
                            _extract_polygon(rec_polys, i)
                            if rec_polys is not None
                            else None
                        )
                        blocks.append(
                            TextBlock(text=text, score=fs, bbox=bbox, polygon=polygon)
                        )
            else:
                for i, text in enumerate(rec_texts):
                    if text:
                        tws.append((text, 1.0))
                        bbox = (
                            _extract_bbox(rec_boxes, i)
                            if rec_boxes is not None
                            else None
                        )
                        polygon = (
                            _extract_polygon(rec_polys, i)
                            if rec_polys is not None
                            else None
                        )
                        blocks.append(
                            TextBlock(text=text, score=1.0, bbox=bbox, polygon=polygon)
                        )
    except Exception as e:
        _logger.error("[_parse_single_result] 处理结果项时出错: %s", e)
    return tws, blocks


def _extract_preproc_info(
    res: Any, *, include_image: bool = True
) -> tuple[int, bytes | None, int, int]:
    """从单个结果项提取预处理信息 (angle, preprocessed_png, w, h)。

    实际预处理图在 doc_preprocessor_res['output_img']（numpy RGB，
    PaddleOCR 3.7 实测即为 RGB），res.img['preprocessed_img'] 是拼接可视化，不用。
    注意：不可对 output_img 做 [:, :, ::-1] 通道翻转——旧注释误以为是 BGR，
    翻转会导致 R/B 对调（黄色变青蓝）。见 scripts/verify_preproc_channels.py。
    """
    preproc_angle = 0
    preprocessed_png: bytes | None = None
    preproc_w = preproc_h = 0
    dp_res = res.get("doc_preprocessor_res") if hasattr(res, "get") else None
    if dp_res is not None:
        preproc_angle = dp_res.get("angle", 0)
        out_arr = dp_res.get("output_img")
        if out_arr is not None:
            preproc_h, preproc_w = out_arr.shape[:2]
            if include_image:
                from PIL import Image as _PILImage

                # output_img 已是 RGB，直接转换；取可写副本避免 PIL 共享只读缓冲
                rgb = out_arr.copy()
                pil_img = _PILImage.fromarray(rgb)
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                preprocessed_png = buf.getvalue()
    return preproc_angle, preprocessed_png, preproc_w, preproc_h


def _recognize_ocr(service: Any, image: Any, options: OCROptions) -> Any:
    """通用 OCR 识别

    从 OCRService._recognize_ocr 和 _process_ocr_output_safe 迁移而来。
    使用 service.get_or_create_pipeline("OCR") 获取管道实例。
    """
    _logger.debug("[_recognize_ocr] 获取 OCR 管道...")
    pipeline = service.get_or_create_pipeline("OCR")
    _logger.debug("[_recognize_ocr] 执行 predict...")
    try:
        output = pipeline.predict(
            input=image,
            use_doc_orientation_classify=options.use_doc_orientation_classify,
            use_doc_unwarping=options.use_doc_unwarping,
            use_textline_orientation=options.use_textline_orientation,
        )
        _logger.debug("[_recognize_ocr] predict 返回，类型: %s", type(output))
    except Exception as e:
        _logger.error("[_recognize_ocr] predict 调用失败: %s", e, exc_info=True)
        raise

    _logger.debug("[_recognize_ocr] 开始处理输出...")
    text_with_scores: list[tuple[str, float]] = []
    text_blocks: list[TextBlock] = []

    output_list = _consume_generator_safely(output)

    # 提取预处理信息：旋转角度和实际预处理后图像
    # 注意：res.img['preprocessed_img'] 是拼接可视化，不用；
    #       实际预处理图在 doc_preprocessor_res['output_img']（numpy RGB）
    preproc_angle, preprocessed_png, preproc_w, preproc_h = (0, None, 0, 0)
    if output_list:
        preproc_angle, preprocessed_png, preproc_w, preproc_h = _extract_preproc_info(
            output_list[0]
        )

    result_count = 0
    for res in output_list:
        result_count += 1
        if result_count > 100:
            _logger.warning("[_recognize_ocr] 结果项过多，可能有问题")
            break
        tws, blocks = _parse_single_result(res)
        text_with_scores.extend(tws)
        text_blocks.extend(blocks)

    raw_text = "\n".join(t for t, _ in text_with_scores)
    _logger.debug(
        "[_recognize_ocr] 处理完成: 共 %d 个结果项, %d 个文本块",
        result_count,
        len(text_with_scores),
    )

    result = _build_ocr_result(
        raw_text=raw_text,
        text_with_scores=text_with_scores,
        pipeline_type="OCR",
        text_blocks=text_blocks,
    )
    result.preproc_angle = preproc_angle
    result.preprocessed_image = preprocessed_png
    result.preproc_img_w = preproc_w
    result.preproc_img_h = preproc_h
    _logger.debug("[_recognize_ocr] 结果处理完成: %d 字符", len(result.raw_text))
    return result


def _recognize_ocr_batch(service: Any, images: list, options: OCROptions) -> list:
    """通用 OCR 批量识别

    将多张图像一次性送入单次 pipeline.predict(list) 调用，由 PaddleOCR 内部
    的 ImageBatchSampler 按 batch_size 分批处理，避免逐张调用的管道开销。
    返回结果顺序与输入 images 一致。

    每张图像的预处理信息（旋转角度、预处理图尺寸）取自其对应的输出项，
    bbox 归一化由 OCRService._normalize_result_bbox 负责。
    """
    _logger.debug("[_recognize_ocr_batch] 获取 OCR 管道...")
    pipeline = service.get_or_create_pipeline("OCR")
    _logger.debug("[_recognize_ocr_batch] 执行批量 predict (%d 张)...", len(images))
    import time as _time

    _predict_start = _time.perf_counter()
    try:
        output_list = _consume_generator_safely(
            pipeline.predict(
                input=images,
                use_doc_orientation_classify=options.use_doc_orientation_classify,
                use_doc_unwarping=options.use_doc_unwarping,
                use_textline_orientation=options.use_textline_orientation,
            )
        )
        _predict_elapsed = _time.perf_counter() - _predict_start
        _logger.info(
            "[识别] pipeline.predict 耗时 %.2fs (%d 张, %.2fs/页)",
            _predict_elapsed,
            len(images),
            _predict_elapsed / max(len(images), 1),
        )
        _logger.debug(
            "[_recognize_ocr_batch] predict 返回 %d 个结果项", len(output_list)
        )
    except Exception as e:
        _predict_elapsed = _time.perf_counter() - _predict_start
        _logger.error(
            "[_recognize_ocr_batch] predict 调用失败 (耗时 %.2fs): %s",
            _predict_elapsed,
            e,
            exc_info=True,
        )
        raise

    # predict(list) 逐图产出结果；若个别图失败，对应项为 {"error": ...}。
    results: list = []
    for res in output_list:
        tws, blocks = _parse_single_result(res)
        preproc_angle, preprocessed_png, preproc_w, preproc_h = (0, None, 0, 0)
        if not (isinstance(res, dict) and "error" in res):
            preproc_angle, preprocessed_png, preproc_w, preproc_h = (
                _extract_preproc_info(
                    res,
                    include_image=bool(
                        getattr(options, "_include_preprocessed_image", True)
                    ),
                )
            )
        raw_text = "\n".join(t for t, _ in tws)
        result = _build_ocr_result(
            raw_text=raw_text,
            text_with_scores=tws,
            pipeline_type="OCR",
            text_blocks=blocks,
        )
        result.preproc_angle = preproc_angle
        result.preprocessed_image = preprocessed_png
        result.preproc_img_w = preproc_w
        result.preproc_img_h = preproc_h
        results.append(result)

    # 容错：输出项少于输入图（极端异常）时补空结果，保持索引对齐
    while len(results) < len(images):
        _logger.warning("[_recognize_ocr_batch] 输出项少于输入图，补空结果")
        results.append(_build_ocr_result(raw_text="", pipeline_type="OCR"))
    return results


OCR_SPEC = PipelineSpec(
    name="OCR",
    display_name="通用 OCR",
    description="识别图片中的文字内容，适用于纯文本场景",
    options_class=OCROptions,
    create_pipeline=_create_ocr_pipeline,
    recognize=_recognize_ocr,
    recognize_batch=_recognize_ocr_batch,
)
