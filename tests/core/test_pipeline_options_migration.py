# tests/core/test_pipeline_options_migration.py
"""Task 2 测试：验证各管道 Options 类继承 BasePipelineOptions 且字段拆分正确。"""

from vibeocr.backend.core.pipelines.pipeline_mineru import MinerUOptions
from vibeocr.backend.core.pipelines.pipeline_ocr import OCROptions
from vibeocr.backend.core.pipelines.pipeline_paddlocr_vl import PaddleOCRVLOptions
from vibeocr.backend.core.pipelines.pipeline_pp_structure import PPStructureV3Options


def test_ocr_options_roundtrip():
    opts = OCROptions(
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )
    d = opts.to_dict()
    assert d["pipeline"] == "OCR"
    assert d["use_doc_unwarping"] is False
    restored = OCROptions.from_dict(d)
    assert restored.use_doc_unwarping is False


def test_pp_structure_options():
    opts = PPStructureV3Options(use_table_recognition=False)
    d = opts.to_dict()
    assert d["pipeline"] == "PP-StructureV3"
    assert d["use_table_recognition"] is False


def test_mineru_options():
    opts = MinerUOptions(parse_method="ocr")
    d = opts.to_dict()
    assert d["pipeline"] == "MinerU"
    assert d["parse_method"] == "ocr"


def test_paddlocr_vl_options():
    opts = PaddleOCRVLOptions(vl_use_layout_detection=False)
    d = opts.to_dict()
    assert d["pipeline"] == "PaddleOCR-VL"
    assert d["vl_use_layout_detection"] is False
