"""Backend pipeline registry with compatibility exports for shared contracts.

Pipeline identifiers and presentation metadata live in ``vibeocr.runtime_contracts.contracts``
so importing them from a frontend does not initialize backend implementations.
"""

from vibeocr.backend.core.pipelines.pipeline_formula import FORMULA_RECOGNITION_SPEC
from vibeocr.backend.core.pipelines.pipeline_mineru import MINERU_SPEC
from vibeocr.backend.core.pipelines.pipeline_ocr import OCR_SPEC
from vibeocr.backend.core.pipelines.pipeline_paddlocr_vl import PADDLEOCR_VL_SPEC
from vibeocr.backend.core.pipelines.pipeline_pp_structure import PP_STRUCTURE_V3_SPEC
from vibeocr.backend.core.pipelines.pipeline_table import TABLE_RECOGNITION_SPEC
from vibeocr.backend.core.pipelines.registry import PipelineRegistry
from vibeocr.runtime_contracts.contracts.pipelines import (
    OCRPipeline,
    RecognitionModeLifecycleKind,
    get_all_pipelines,
    get_all_recognition_modes,
    get_heavy_pipelines,
    get_mineru_pipelines,
    get_paddle_pipelines,
    get_pipeline_description,
    get_pipeline_display_name,
    get_pipeline_short_name,
    get_pipeline_supported_options,
    get_preloadable_pipelines,
    get_recognition_mode_definition,
    is_option_supported,
)

_registry = PipelineRegistry()
_registry.register(OCR_SPEC)
_registry.register(PP_STRUCTURE_V3_SPEC)
_registry.register(TABLE_RECOGNITION_SPEC)
_registry.register(FORMULA_RECOGNITION_SPEC)
_registry.register(MINERU_SPEC)
_registry.register(PADDLEOCR_VL_SPEC)


def get_registry() -> PipelineRegistry:
    """获取全局管道注册表单例。"""
    return _registry


def get_paddle_residency_pipelines() -> list[OCRPipeline]:
    """Return execution pipelines owned by Paddle model residency.

    Protocol 2.8 correctly classifies the shared ``OCR`` pipeline as routed,
    so the generic pipeline cache helpers no longer identify its physical
    owner.  Inside the Paddle adapter/cache boundary, derive ownership from
    the formal Recognition Mode lifecycle instead.
    """
    return list(
        dict.fromkeys(
            definition.pipeline
            for mode_id in get_all_recognition_modes()
            if (definition := get_recognition_mode_definition(mode_id)).lifecycle.kind
            is RecognitionModeLifecycleKind.MODEL_RESIDENCY
        )
    )


__all__ = [
    "OCRPipeline",
    "get_all_pipelines",
    "get_heavy_pipelines",
    "get_mineru_pipelines",
    "get_paddle_pipelines",
    "get_paddle_residency_pipelines",
    "get_pipeline_description",
    "get_pipeline_display_name",
    "get_pipeline_short_name",
    "get_pipeline_supported_options",
    "get_preloadable_pipelines",
    "get_registry",
    "is_option_supported",
]
