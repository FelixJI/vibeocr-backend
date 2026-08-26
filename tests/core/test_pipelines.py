# tests/core/test_pipelines.py
"""管道定义模块测试"""

from vibeocr.backend.core.pipelines import (
    OCRPipeline,
    get_pipeline_description,
    get_pipeline_display_name,
    get_pipeline_supported_options,
)


class TestOCRPipeline:
    def test_pipeline_count(self):
        """验证管道数量为 6"""
        assert len(OCRPipeline) == 6

    def test_pipeline_values(self):
        """验证管道值"""
        assert OCRPipeline.OCR.value == "OCR"
        assert OCRPipeline.PP_STRUCTURE_V3.value == "PP-StructureV3"
        assert OCRPipeline.DOCUMENT_PARSING.value == "MinerU"
        assert OCRPipeline.PADDLEOCR_VL.value == "PaddleOCR-VL"
        assert OCRPipeline.TABLE_RECOGNITION.value == "TABLE_RECOGNITION"
        assert OCRPipeline.FORMULA_RECOGNITION.value == "FORMULA_RECOGNITION"

    def test_get_display_name(self):
        """验证显示名称获取"""
        assert get_pipeline_display_name(OCRPipeline.OCR) == "通用 OCR"
        assert get_pipeline_display_name(OCRPipeline.PP_STRUCTURE_V3) == (
            "文档结构识别（PP-StructureV3）"
        )
        assert "MinerU" in get_pipeline_display_name(OCRPipeline.DOCUMENT_PARSING)
        assert "PaddleOCR-VL" in get_pipeline_display_name(OCRPipeline.PADDLEOCR_VL)

    def test_get_description(self):
        """验证描述获取"""
        desc = get_pipeline_description(OCRPipeline.OCR)
        assert "文字" in desc or "文本" in desc
        desc_vl = get_pipeline_description(OCRPipeline.PADDLEOCR_VL)
        assert "PaddleOCR-VL" in desc_vl

    def test_get_supported_options(self):
        """验证支持的选项"""
        options = get_pipeline_supported_options(OCRPipeline.OCR)
        assert "use_doc_orientation_classify" in options
        assert "use_doc_unwarping" in options
        assert "use_textline_orientation" in options

    def test_pp_structure_v3_options(self):
        """PP-StructureV3 应支持预处理 + 结构分析选项"""
        options = get_pipeline_supported_options(OCRPipeline.PP_STRUCTURE_V3)
        assert "use_doc_orientation_classify" in options
        assert "use_table_recognition" in options
        assert "use_formula_recognition" in options
        assert "use_seal_recognition" in options
        assert "use_chart_recognition" in options

    def test_document_parsing_options(self):
        """文档解析应支持 MinerU 选项"""
        options = get_pipeline_supported_options(OCRPipeline.DOCUMENT_PARSING)
        assert "parse_method" in options
        assert "enable_formula" in options
        assert "enable_table" in options

    def test_document_parsing_supports_lang_and_page_range(self):
        """文档解析应支持语言和页码范围选项"""
        options = get_pipeline_supported_options(OCRPipeline.DOCUMENT_PARSING)
        assert "lang_list" in options
        assert "start_page_id" in options
        assert "end_page_id" in options

    def test_paddlocr_vl_options(self):
        """PaddleOCR-VL 应支持布局、图表、印章、图片OCR选项"""
        options = get_pipeline_supported_options(OCRPipeline.PADDLEOCR_VL)
        assert "vl_use_layout_detection" in options
        assert "vl_use_chart_recognition" in options
        assert "vl_use_seal_recognition" in options
        assert "use_ocr_for_image_block" in options

    def test_table_recognition_options(self):
        """表格识别应支持表格方向和单元格文字识别选项"""
        options = get_pipeline_supported_options(OCRPipeline.TABLE_RECOGNITION)
        assert "use_doc_orientation_classify" in options
        assert "use_table_orientation_classify" in options
        assert "use_ocr_results_with_table_cells" in options

    def test_formula_recognition_options(self):
        """公式识别应支持公式相关选项"""
        options = get_pipeline_supported_options(OCRPipeline.FORMULA_RECOGNITION)
        assert "use_doc_orientation_classify" in options
        assert "use_doc_unwarping" in options
        assert "formula_recognition_batch_size" in options
