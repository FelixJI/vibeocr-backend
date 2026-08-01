# src/vibeocr/core/pipelines/pipeline_table.py
"""表格识别管道选项与规格

定义表格识别管道的选项类和 PipelineSpec，
支持有线/无线表格结构识别。
"""

from __future__ import annotations

import html
import logging
import math
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any

from vibeocr.backend.core.pipelines.base_options import BasePipelineOptions
from vibeocr.backend.core.pipelines.registry import PipelineSpec
from vibeocr.backend.tables.blocks import (
    canonicalize_table_block,
    table_model_from_block,
)
from vibeocr.backend.tables.html_adapter import (
    parse_table_source_layout,
    table_model_from_html,
    table_model_to_html,
)
from vibeocr.backend.tables.projections import (
    table_model_to_markdown,
    table_model_to_plain_text,
)
from vibeocr.backend.tables.reducer import rebuild_result_projections
from vibeocr.runtime_contracts.contracts.tables import (
    CoordinateSpace,
    TableProvenanceV1,
)

_logger = logging.getLogger(__name__)


class TableDependencyError(RuntimeError):
    """表格识别所需依赖（paddlex[ocr] leaf 包）缺失。

    PaddleX 会把真因 ``DependencyError`` 包成无信息的 ``RuntimeError: A dependency
    error occurred...``，本类在管道创建前主动探测并报告**具体缺失包名**，引导用户
    走设置页重装，而非展示无信息的泛化错误。
    """


def _check_table_deps() -> None:
    """检测表格识别所需 paddlex[ocr] 依赖，缺失则抛 TableDependencyError。

    核心策略：**直接复用 PaddleX 的判定**（``is_extra_available("ocr")`` +
    ``is_dep_available``），与 ``TableRecognitionPipelineV2`` 实例化时
    ``@pipeline_requires_extra("ocr")`` 走的是**同一条代码路径**，杜绝
    "本探测通过但 PaddleX 判否"的盲区。

    历史问题：早期版本用 ``importlib.util.find_spec`` 探测一份手工维护的
    leaf 清单（``OCR_CHECK_LEAF_MODULES``），但 PaddleX 的 ``is_dep_available``
    对绝大多数包走 ``importlib.metadata.version``（查发行版元数据），与
    ``find_spec``（查 import 名）是两套机制：包可能 ``find_spec`` 命中却无
    ``.dist-info`` 元数据，反之亦然。便携环境曾出现 leaf 探测全通过、PaddleX
    却判 extra 不可用、实例化时爆炸为无信息 ``RuntimeError`` 的情况。

    缺失时列出 PaddleX 视角的具体发行版名，引导用户在设置页精准重装。
    """
    try:
        from paddlex.utils.deps import EXTRAS, is_dep_available, is_extra_available
    except ImportError:
        # paddlex 未安装（极端残缺环境）：回退到本项目的 leaf 清单兜底探测，
        # 总比静默放过、让 PaddleX 抛无信息 RuntimeError 强。
        import importlib.util

        from vibeocr.backend.services.env_config import OCR_CHECK_LEAF_MODULES

        missing = [
            pkg
            for mod, pkg in OCR_CHECK_LEAF_MODULES.items()
            if importlib.util.find_spec(mod) is None
        ]
        if missing:
            raise TableDependencyError(
                f"表格识别缺少依赖：{', '.join(missing)}。"
                "请在「设置 → 重装 OCR 依赖」修复后重试。"
            ) from None
        return

    # PaddleX 判 extra 不可用 → 用其同一判定路径列出具体缺失发行版。
    # 注意 is_dep_available / is_extra_available 均 @lru_cache，此处结果与
    # 实例化时的检查完全一致。
    if not is_extra_available("ocr"):
        missing = [dep for dep in EXTRAS.get("ocr", []) if not is_dep_available(dep)]
        raise TableDependencyError(
            f"表格识别缺少 PaddleX[ocr] 依赖：{', '.join(missing) or '（未知）'}。"
            "请在「设置 → 重装 OCR 依赖」修复后重试。"
        )


@dataclass
class TableRecognitionOptions(BasePipelineOptions):
    """表格识别管道选项

    支持有线和无线表格结构识别，可配置各种 OCR 检测/识别参数。
    """

    pipeline: str = "TABLE_RECOGNITION"
    use_doc_orientation_classify: bool = True
    use_doc_unwarping: bool = True
    use_wireless_table: bool = True
    wireless_table_model_name: str = "SLANeXt_wireless"
    wired_table_model_name: str = "SLANeXt_wired"
    use_table_orientation_classify: bool = True
    use_ocr_results_with_table_cells: bool = True
    text_det_limit_side_len: int | None = None
    # 检测/识别阈值默认 None → 吃 PaddleX 模型默认值（PP-OCRv4 在通用场景
    # 验证过的工程值，准确率与误检平衡最佳）。漏字根因在 PaddleX 内部
    # match_table_and_ocr 的 IoU>0.7 匹配（见 _backfill_empty_table_cells
    # 的下游兜底），不应在此处改默认值引入全局误检。用户遇极端场景可
    # 在设置页显式覆盖这些字段。
    text_det_thresh: float | None = None
    text_det_box_thresh: float | None = None
    text_det_unclip_ratio: float | None = None
    text_rec_score_thresh: float | None = None
    use_wired_table_cells_trans_to_html: bool = False
    use_wireless_table_cells_trans_to_html: bool = False
    use_e2e_wired_table_rec_model: bool = False
    use_e2e_wireless_table_rec_model: bool = True
    formula_recognition_batch_size: int = 1


def _diagnose_paddlex_ocr_extra() -> list[str]:
    """从 PaddleX 视角列出 ``ocr`` extra 中被判不可用的发行版名。

    供 ``_create_table_pipeline`` 的 except 分支补充日志：理论上
    ``_check_table_deps`` 已用同一 ``is_extra_available`` 前置拦截，但若
    PaddleX 内部状态在两次调用间发生变化（如 lru_cache 时序），此处兜底
    把具体缺失包名落进 error 日志，便于定位。
    """
    try:
        from paddlex.utils.deps import EXTRAS, is_dep_available

        return [dep for dep in EXTRAS.get("ocr", []) if not is_dep_available(dep)]
    except Exception as diag_err:  # 诊断本身不能掩盖原始错误
        _logger.exception("[表格依赖诊断] 诊断失败: %s", diag_err)
        return []


def _create_table_pipeline(device: str, **kwargs: Any) -> Any:
    """创建表格识别管道实例

    额外 kwargs 透传给 TableRecognitionPipelineV2（例如 enable_mkldnn）。
    """
    # 创建前主动探测 paddlex[ocr] leaf 包。PaddleX 会把真因 DependencyError 包成
    # 无信息的 RuntimeError，此处提前拦截并报告具体缺失包，引导用户修复。
    _check_table_deps()
    from paddleocr import TableRecognitionPipelineV2

    try:
        return TableRecognitionPipelineV2(device=device, **kwargs)
    except Exception:
        # _check_table_deps 已用 is_extra_available 前置拦截，正常情况下此处
        # 不会因 ocr extra 缺失触发。但 defense-in-depth：若 PaddleX 内部状态
        # 在两次调用间变化（lru_cache 时序 / 并发），把具体缺失包名落进日志，
        # 避免又退回无信息的 RuntimeError。
        paddlex_missing = _diagnose_paddlex_ocr_extra()
        if paddlex_missing:
            _logger.error(
                "[表格管道] PaddleX 判定 ocr extra 不可用，缺失发行版: %s",
                ", ".join(paddlex_missing),
            )
        raise


def _recognize_table(service: Any, image: Any, options: TableRecognitionOptions) -> Any:
    """执行表格识别并返回 OCRResult"""
    from enum import Enum

    from vibeocr.backend.models.ocr_result import OCRResult, TextBlock

    pipeline_name = (
        options.pipeline.value
        if isinstance(options.pipeline, Enum)
        else options.pipeline
    )
    pipeline = service.get_or_create_pipeline(pipeline_name)

    predict_kwargs: dict[str, Any] = {}
    predict_kwargs["use_doc_orientation_classify"] = (
        options.use_doc_orientation_classify
    )
    predict_kwargs["use_doc_unwarping"] = options.use_doc_unwarping
    predict_kwargs["use_table_orientation_classify"] = (
        options.use_table_orientation_classify
    )
    predict_kwargs["use_ocr_results_with_table_cells"] = (
        options.use_ocr_results_with_table_cells
    )
    predict_kwargs["use_wired_table_cells_trans_to_html"] = (
        options.use_wired_table_cells_trans_to_html
    )
    predict_kwargs["use_wireless_table_cells_trans_to_html"] = (
        options.use_wireless_table_cells_trans_to_html
    )
    predict_kwargs["use_e2e_wired_table_rec_model"] = (
        options.use_e2e_wired_table_rec_model
    )
    predict_kwargs["use_e2e_wireless_table_rec_model"] = (
        options.use_e2e_wireless_table_rec_model
    )

    # 同时传入有线和无线模型名，管道内部的表格分类器
    # (PP-LCNet_x1_0_table_cls) 会自动判断表格类型并选用对应模型。
    predict_kwargs["wireless_table_structure_recognition_model_name"] = (
        options.wireless_table_model_name
    )
    predict_kwargs["wired_table_structure_recognition_model_name"] = (
        options.wired_table_model_name
    )

    if options.text_det_limit_side_len is not None:
        predict_kwargs["text_det_limit_side_len"] = options.text_det_limit_side_len
    if options.text_det_thresh is not None:
        predict_kwargs["text_det_thresh"] = options.text_det_thresh
    if options.text_det_box_thresh is not None:
        predict_kwargs["text_det_box_thresh"] = options.text_det_box_thresh
    if options.text_det_unclip_ratio is not None:
        predict_kwargs["text_det_unclip_ratio"] = options.text_det_unclip_ratio
    if options.text_rec_score_thresh is not None:
        predict_kwargs["text_rec_score_thresh"] = options.text_rec_score_thresh

    output = pipeline.predict(input=image, **predict_kwargs)
    output_list = list(output)

    text_blocks: list[TextBlock] = []
    text_with_scores: list[tuple[str, float]] = []
    markdown_parts: list[str] = []
    content_list: list[dict[str, Any]] = []
    table_sequence = 0

    for res in output_list:
        # PaddleX 表格结果（TableRecognitionResult）是 dict 子类，键需下标访问。
        # 表格内容在 "table_res_list"（每项含 pred_html），而非 parsing_res_list
        # （后者属于版面解析管道）。早期代码误读 parsing_res_list 导致永远取空，
        # 表现为"未识别到文字"。
        table_res_list: list[Any] = []
        if hasattr(res, "__getitem__"):
            table_res_list = (
                res["table_res_list"]
                if "table_res_list" in (res.keys() if hasattr(res, "keys") else [])
                else []
            )

        # 记录每个表格的外接框，用于后续过滤 overall_ocr_res 中的重复文本。
        # 注意：PaddleX 的 SingleTableRecognitionResult 不含 ``table_bbox``
        # 字段，只有 ``cell_box_list``（各单元格 [x1,y1,x2,y2]，原图坐标系，
        # 已在 post-processing 中 clip 到原图范围）。表格整体外接框需从
        # cell_box_list 的并集推导，否则过滤条件永远为空，overall_ocr_res
        # 里整图文字（含表格内文字）会被原样再展示一遍。
        table_match_regions: list[tuple[tuple[float, float, float, float], str]] = []

        # 先把 overall_ocr_res 解析为统一的 ocr_items（text + center + score +
        # bbox），供回填与去重共享。center 为文本框中心点（原图坐标），
        # None 表示无 poly（无法定位）。
        overall_ocr_res = res.get("overall_ocr_res") if hasattr(res, "get") else None
        ocr_items: list[dict[str, Any]] = []
        if overall_ocr_res is not None:
            rec_texts = (
                overall_ocr_res.get("rec_texts")
                if hasattr(overall_ocr_res, "get")
                else None
            )
            rec_scores = (
                overall_ocr_res.get("rec_scores")
                if hasattr(overall_ocr_res, "get")
                else None
            )
            rec_polys = (
                overall_ocr_res.get("rec_polys")
                if hasattr(overall_ocr_res, "get")
                else None
            )
            if rec_texts:
                for i, text in enumerate(rec_texts):
                    score = (
                        float(rec_scores[i])
                        if rec_scores and i < len(rec_scores)
                        else 0.9
                    )
                    poly = rec_polys[i] if rec_polys and i < len(rec_polys) else None
                    bbox_tuple = None
                    center: tuple[float, float] | None = None
                    if poly is not None and hasattr(poly, "shape") and poly.size >= 4:
                        xs = poly[:, 0].tolist() if poly.ndim == 2 else None
                        ys = poly[:, 1].tolist() if poly.ndim == 2 else None
                        if xs and ys:
                            bbox_tuple = (
                                float(min(xs)),
                                float(min(ys)),
                                float(max(xs)),
                                float(max(ys)),
                            )
                            center = (
                                (bbox_tuple[0] + bbox_tuple[2]) / 2,
                                (bbox_tuple[1] + bbox_tuple[3]) / 2,
                            )
                    ocr_items.append(
                        {
                            "text": text,
                            "score": score,
                            "bbox": bbox_tuple,
                            "center": center,
                        }
                    )

        # 已被回填进空单元格的 OCR 索引（后续不再作为独立 text 块展示，
        # 避免与表格内容重复）。
        consumed_ocr_indices: set[int] = set()

        for idx, table_res in enumerate(table_res_list):
            # pred_html: <html><body><table>...</table></body></html>
            pred_html = (
                table_res.get("pred_html") if hasattr(table_res, "get") else None
            )
            if not pred_html:
                continue

            # 从 cell_box_list 并集推导当前表格的外接框（原图像素坐标）。
            # 该值既用于下方过滤重复文本，也挂回表格块自身的 bbox（替代
            # 早期写死的 None），让左侧画布能正确绘制表格 bbox。
            # 归一化到 [0,1000] 由 service 层 _normalize_result_bbox 统一完成。
            cell_box_list = (
                table_res.get("cell_box_list") if hasattr(table_res, "get") else None
            )
            current_bbox: tuple[float, float, float, float] | None = None
            if cell_box_list:
                valid_cell_boxes = [
                    parsed
                    for cell in cell_box_list
                    if (parsed := _parse_cell_box(cell)) is not None
                ]
                if valid_cell_boxes:
                    current_bbox = (
                        min(box[0] for box in valid_cell_boxes),
                        min(box[1] for box in valid_cell_boxes),
                        max(box[2] for box in valid_cell_boxes),
                        max(box[3] for box in valid_cell_boxes),
                    )

            from vibeocr.backend.services.ocr_service import _extract_table_html

            table_html = _extract_table_html(pred_html)
            original_table_html = table_html
            cl_idx = len(content_list)
            table_id = f"table-recognition-table-{table_sequence}"
            table_sequence += 1
            cell_mapping: dict[int, tuple[tuple[float, float, float, float], int]] = {}
            provider_warnings: list[str] = []

            # 空单元格回填（soft fallback）：PaddleX IoU 失配时输出空 <td></td>，
            # 但那些字其实躺在 overall_ocr_res 里。把落在空单元格内的 OCR
            # 文本回填进单元格，救回上游漏填的字，避免彻底漏字。
            available_ocr = [
                (ocr_index, item)
                for ocr_index, item in enumerate(ocr_items)
                if ocr_index not in consumed_ocr_indices
            ]
            table_html, consumed_local = _backfill_empty_table_cells(
                table_html,
                cell_box_list,
                [item for _, item in available_ocr],
                table_id=table_id,
                mapping_out=cell_mapping,
                warnings_out=provider_warnings,
            )
            if any(
                warning.endswith(":cell-box-count-mismatch")
                for warning in provider_warnings
            ):
                provider_warnings.append(f"{table_id}:cell-all:cell-mapping-unproven")
            consumed_ocr_indices.update(
                available_ocr[local_index][0] for local_index in consumed_local
            )

            if current_bbox is not None:
                table_match_regions.append(
                    (
                        current_bbox,
                        _normalize_match_text(
                            table_model_to_plain_text(
                                table_model_from_html(table_html, table_id=table_id)
                            )
                        ),
                    )
                )

            canonical_block = canonicalize_table_block(
                {
                    "type": "table",
                    "table_body": table_html,
                    "bbox": list(current_bbox) if current_bbox else None,
                    "source": {
                        "source_html": original_table_html,
                        "provider_schema": "paddlex-table-res-list",
                        "provider_table_id": (
                            table_res.get("table_region_id")
                            if hasattr(table_res, "get")
                            else None
                        ),
                    },
                },
                table_id=table_id,
                pipeline="TABLE_RECOGNITION",
            )
            table_model = table_model_from_block(canonical_block)
            original_model = table_model_from_html(
                original_table_html, table_id=table_id
            )
            provider_table_id = (
                table_res.get("table_region_id") if hasattr(table_res, "get") else None
            )
            enriched_cells = []
            for cell_index, cell in enumerate(table_model.cells):
                mapping = cell_mapping.get(cell_index)
                if mapping is None:
                    enriched_cells.append(cell)
                    continue
                cell_box, source_index = mapping
                source_ref = (
                    (
                        "paddlex:"
                        f"table_region:{provider_table_id}:"
                        f"cell_box:{source_index}"
                    )
                    if provider_table_id is not None
                    else f"paddlex-cell:{source_index}"
                )
                enriched_cells.append(
                    replace(
                        cell,
                        bbox=cell_box,
                        source_refs=(source_ref,),
                    )
                )
                if (
                    cell_index < len(original_model.cells)
                    and original_model.cells[cell_index].text != cell.text
                ):
                    provider_warnings.append(
                        f"{table_id}:cell-{cell.cell_id}:ocr-backfilled"
                    )
            table_model = replace(
                table_model,
                cells=tuple(enriched_cells),
                coordinate_space=(
                    CoordinateSpace.PIXEL if cell_mapping else CoordinateSpace.UNKNOWN
                ),
                provenance=TableProvenanceV1(
                    pipeline="TABLE_RECOGNITION",
                    provider_schema="paddlex-table-res",
                    warnings=tuple(dict.fromkeys(provider_warnings)),
                ),
            )
            canonical_block["table"] = table_model.to_payload()
            canonical_block["table_body"] = table_model_to_html(table_model)
            table_plain_text = table_model_to_plain_text(table_model)
            table_md = table_model_to_markdown(table_model).text
            if table_md:
                markdown_parts.append(table_md)

            text_blocks.append(
                TextBlock(
                    text=table_plain_text,
                    score=0.9,
                    bbox=current_bbox,
                    label="table",
                    order=idx,
                    content_index=cl_idx,
                    content_id=table_id,
                )
            )
            text_with_scores.append((table_plain_text, 0.9))
            content_list.append(canonical_block)

        # 表格外的普通文字（overall_ocr_res）：截图场景多为整图表格，此处通常为空，
        # 但保留以兼容"表格 + 周边文字"的图片。
        # 注意：overall_ocr_res 包含整图所有文本（含表格内文字）。处理策略：
        # 1. 已被回填进空单元格的 OCR（consumed_ocr_indices）→ 跳过（已在表格中展示）。
        # 2. 落在单元格内的 OCR 已被合并进表格；无精确单元格但文本已存在于
        #    表格中的 OCR 也按表格 bbox + 文字匹配去重。
        # 3. 表格外的 OCR → 正常展示。
        if ocr_items:
            for i, item in enumerate(ocr_items):
                text = item["text"]
                if not text:
                    continue
                if i in consumed_ocr_indices:
                    continue
                bbox_tuple = item["bbox"]
                center = item["center"]
                normalized_text = _normalize_match_text(text)
                if center is not None and normalized_text:
                    if any(
                        _point_in_box(float(center[0]), float(center[1]), bbox)
                        and normalized_text in table_text
                        for bbox, table_text in table_match_regions
                    ):
                        continue
                # 剩余项属于表格外文字，或无法安全定位/去重的 OCR；保留以防漏字。
                cl_idx = len(content_list)
                block_id = f"table-recognition-block-{cl_idx}"
                text_blocks.append(
                    TextBlock(
                        text=text,
                        score=item["score"],
                        bbox=bbox_tuple,
                        label="text",
                        order=-1,
                        content_index=cl_idx,
                        content_id=block_id,
                    )
                )
                text_with_scores.append((text, item["score"]))
                content_list.append(
                    {
                        "type": "text",
                        "text": text,
                        "bbox": bbox_tuple,
                        "block_id": block_id,
                    }
                )

    raw_text = "\n".join(b.text for b in text_blocks)
    markdown_text = "\n\n".join(markdown_parts) if markdown_parts else raw_text

    html_parts = [
        (
            str(block.get("table_body") or "")
            if block.get("type") == "table"
            else f"<p>{html.escape(str(block.get('text') or ''))}</p>"
        )
        for block in content_list
    ]

    result = OCRResult(
        raw_text=raw_text,
        markdown_text=markdown_text,
        html_text="\n".join(part for part in html_parts if part),
        text_with_scores=text_with_scores,
        pipeline_type="TABLE_RECOGNITION",
        text_blocks=text_blocks,
        content_list=content_list,
    )
    rebuild_result_projections(result)
    return result


def _point_in_box(cx: float, cy: float, box: tuple[float, float, float, float]) -> bool:
    """点是否落在矩形框内（含边界）。"""
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _parse_cell_box(raw_box: Any) -> tuple[float, float, float, float] | None:
    """Parse one finite, non-degenerate provider cell box."""

    box = raw_box.tolist() if hasattr(raw_box, "tolist") else raw_box
    if (
        not isinstance(box, (list, tuple))
        or len(box) < 4
        or any(
            isinstance(value, bool) or not isinstance(value, Real) for value in box[:4]
        )
    ):
        return None
    parsed = (
        float(box[0]),
        float(box[1]),
        float(box[2]),
        float(box[3]),
    )
    if (
        not all(math.isfinite(value) for value in parsed)
        or parsed[0] >= parsed[2]
        or parsed[1] >= parsed[3]
    ):
        return None
    return parsed


def _normalize_match_text(text: str) -> str:
    """用于表格 OCR 去重的宽松文本规范化。"""
    return "".join(str(text).split()).casefold()


def _backfill_empty_table_cells(
    table_html: str,
    cell_box_list: list[Any] | None,
    ocr_items: list[dict[str, Any]],
    *,
    table_id: str = "table",
    mapping_out: dict[int, tuple[tuple[float, float, float, float], int]] | None = None,
    warnings_out: list[str] | None = None,
) -> tuple[str, set[int]]:
    """把表内 OCR 吸收到对应单元格，并救回上游漏填的字。

    根因：PaddleX ``match_table_and_ocr`` 要求 IoU>0.7 才把 OCR 文字填进
    单元格，失配时输出空 ``<td></td>`` / ``<th></th>``；但那些文字其实躺在
    ``overall_ocr_res`` 里。本函数按几何落点吸收所有表内 OCR：空格回填，
    已填格去重或追加缺失文字，从而避免 UI 再显示独立文本框。

    匹配策略——**几何优先，不依赖位置序号**：
    PaddleX 的 cell_box_list 与 pred_html 单元格虽都是 row-major，但二者
    的 row 切分机制独立（一个几何、一个结构 token），错位时 1:1 位置对应
    不保证。故本函数对每个单元格，按其在 cell_box_list 的位置取候选框，
    再用 OCR 文本中心点是否落在框内做匹配——位置序号仅用于取候选框，几何
    才是判据。

    Args:
        table_html: 经 ``_extract_table_html`` 提取的 ``<table>...</table>``。
        cell_box_list: PaddleX 各单元格 [x1,y1,x2,y2]（row-major，与 HTML
            单元格顺序对齐；结构错位时按几何兜底）。
        ocr_items: 每项 ``{"text": str, "center": (cx,cy)|None}``，center
            为文本框中心点（原图坐标）；None 表示无坐标，跳过。

    Returns:
        (new_table_html, consumed_indices)：合并后的 HTML，与被消费的
        ocr_items 索引集合。无坐标或落点不在任何单元格的项由调用方处理。
    """
    if not cell_box_list:
        return table_html, set()

    # 收集全部单元格：表格内已有内容对应的 OCR 也需要吸收，否则 UI 会
    # 同时显示表格和一套独立文本框。
    # 注意：_RE_CELL 在 tr_match.group(1) 内匹配，其 start/end 是相对该子串
    # 的偏移；需加上 tr_match.start(1) 换算成 table_html 内的绝对偏移，否则
    # 注入会落到错误位置（如把字插进 <table> 标签中间）。
    try:
        source_layout = parse_table_source_layout(table_html, table_id=table_id)
    except ValueError as error:
        detail = str(error).casefold()
        warning_kind = (
            "span-limit-exceeded"
            if "span" in detail or "coverage" in detail or "supported limit" in detail
            else "layout-invalid"
        )
        warning = f"{table_id}:cell-0:{warning_kind}"
        if warnings_out is not None:
            warnings_out.append(warning)
        _logger.warning("[表格回填] %s：%s", warning, error)
        return table_html, set()
    layout = source_layout.model
    all_cell_spans = [
        (cell.content_start, cell.content_end) for cell in source_layout.cells
    ]
    all_cell_texts = [cell.text.strip() for cell in layout.cells]
    if not source_layout.cells:
        return table_html, set()
    if len(source_layout.cells) != len(layout.cells):
        warning = f"{table_id}:cell-all:layout-cell-count-mismatch"
        if warnings_out is not None:
            warnings_out.append(warning)
        _logger.warning(
            "[表格回填] %s：无法从 cell_box_list 证明 HTML 逻辑网格映射",
            warning,
        )
        return table_html, set()

    logical_cells = [
        (cell.row, cell.column, cell.rowspan, cell.colspan) for cell in layout.cells
    ]
    logical_column_count = layout.column_count
    logical_row_count = layout.row_count

    # PaddleX 未承诺 cell_box_list 的序列顺序；先过滤退化框，再按可验证的
    # 视觉坐标（上→下、左→右）排序后对应 HTML 的 row-major 单元格。
    valid_boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for source_index, raw_box in enumerate(cell_box_list):
        parsed = _parse_cell_box(raw_box)
        if parsed is not None:
            valid_boxes.append((source_index, parsed))
    if len(valid_boxes) != len(layout.cells):
        warning = f"{table_id}:cell-all:cell-box-count-mismatch"
        if warnings_out is not None:
            warnings_out.append(warning)
        _logger.warning(
            "[表格回填] %s：无法从 cell_box_list 证明 HTML 逻辑网格映射",
            warning,
        )
        return table_html, set()

    def cluster_boundaries(
        values: list[float], expected_count: int
    ) -> tuple[list[float], float] | None:
        ordered = sorted(values)
        if not ordered or expected_count < 2:
            return None
        grid_step = (ordered[-1] - ordered[0]) / (expected_count - 1)
        tolerance = max(grid_step * 0.15, 1e-6)
        clusters: list[list[float]] = []
        for value in ordered:
            if (
                clusters
                and abs(value - sum(clusters[-1]) / len(clusters[-1])) <= tolerance
            ):
                clusters[-1].append(value)
            else:
                clusters.append([value])
        if len(clusters) != expected_count:
            return None
        return ([sum(cluster) / len(cluster) for cluster in clusters], tolerance)

    x_grid = cluster_boundaries(
        [value for _, box in valid_boxes for value in (box[0], box[2])],
        logical_column_count + 1,
    )
    y_grid = cluster_boundaries(
        [value for _, box in valid_boxes for value in (box[1], box[3])],
        logical_row_count + 1,
    )
    if x_grid is None or y_grid is None:
        warning = f"{table_id}:cell-all:cell-mapping-unproven"
        if warnings_out is not None:
            warnings_out.append(warning)
        _logger.warning(
            "[表格回填] %s：无法从 cell_box_list 证明 HTML 逻辑网格映射",
            warning,
        )
        return table_html, set()

    x_boundaries, x_tolerance = x_grid
    y_boundaries, y_tolerance = y_grid

    def boundary_index(
        value: float, boundaries: list[float], tolerance: float
    ) -> int | None:
        index = min(
            range(len(boundaries)),
            key=lambda candidate: abs(value - boundaries[candidate]),
        )
        return index if abs(value - boundaries[index]) <= tolerance else None

    candidate_boxes: dict[int, tuple[float, float, float, float]] = {}
    for source_index, box in valid_boxes:
        boundary_span = (
            boundary_index(box[1], y_boundaries, y_tolerance),
            boundary_index(box[0], x_boundaries, x_tolerance),
            boundary_index(box[3], y_boundaries, y_tolerance),
            boundary_index(box[2], x_boundaries, x_tolerance),
        )
        if any(value is None for value in boundary_span):
            continue
        top, left, bottom, right = boundary_span
        matches = [
            cell_index
            for cell_index, (row, column, rowspan, colspan) in enumerate(logical_cells)
            if (top, left, bottom, right)
            == (row, column, row + rowspan, column + colspan)
        ]
        if len(matches) != 1 or matches[0] in candidate_boxes:
            warning = f"{table_id}:cell-{matches[0] if matches else 'unknown'}:mapping-ambiguous"
            if warnings_out is not None:
                warnings_out.append(warning)
            _logger.warning("[表格回填] %s：cell_box 无法唯一映射", warning)
            continue
        candidate_boxes[matches[0]] = box
        if mapping_out is not None:
            mapping_out[matches[0]] = (box, source_index)

    if not candidate_boxes:
        return table_html, set()
    if not ocr_items:
        return table_html, set()

    # 几何匹配：OCR 文本中心落在任意单元格框内 → 吸收进该格。
    cell_to_text: dict[int, list[str]] = {ci: [] for ci in candidate_boxes}
    consumed: set[int] = set()
    for i, item in enumerate(ocr_items):
        center = item.get("center")
        if center is None:
            continue
        cx, cy = float(center[0]), float(center[1])
        text = (item.get("text") or "").strip()
        if not text:
            continue
        matches = [
            (ci, box)
            for ci, box in candidate_boxes.items()
            if _point_in_box(cx, cy, box)
        ]
        if matches:
            ci, _ = min(
                matches,
                key=lambda pair: (
                    (pair[1][2] - pair[1][0]) * (pair[1][3] - pair[1][1]),
                    pair[0],
                ),
            )
            cell_to_text[ci].append(text)
            consumed.add(i)

    if not consumed:
        return table_html, set()

    # 空格直接回填；已有内容只追加尚未出现的 OCR 文本，既消除独立文本框，
    # 又避免结构模型漏字时静默丢失。
    import html as _html

    replacements: list[tuple[int, int, str]] = []
    for ci, texts in cell_to_text.items():
        if not texts:
            continue
        span_start, span_end = all_cell_spans[ci]
        existing = all_cell_texts[ci]
        normalized_existing = _normalize_match_text(existing)
        additions = [
            value
            for value in texts
            if _normalize_match_text(value)
            and _normalize_match_text(value) not in normalized_existing
        ]
        if not additions:
            continue
        escaped = _html.escape(" ".join(additions))
        original = table_html[span_start:span_end]
        filled = f"{original}<br>{escaped}" if existing else escaped
        replacements.append((span_start, span_end, filled))

    # 按位置降序替换
    replacements.sort(key=lambda r: r[0], reverse=True)
    new_html = table_html
    for start, end, filled in replacements:
        new_html = new_html[:start] + filled + new_html[end:]

    return new_html, consumed


TABLE_RECOGNITION_SPEC = PipelineSpec(
    name="TABLE_RECOGNITION",
    display_name="表格识别",
    description="独立表格结构识别，支持有线和无线表格",
    options_class=TableRecognitionOptions,
    create_pipeline=_create_table_pipeline,
    recognize=_recognize_table,
)
