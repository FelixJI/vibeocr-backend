# tests/integration/test_pipeline_registry_e2e.py
"""管道注册表端到端测试

验证注册表完整性：所有 spec 可查询、options 可序列化/反序列化，
UI 管道下拉与注册表同步，旧导入路径仍可用。
"""

from vibeocr.backend.core.pipelines import OCRPipeline, get_all_pipelines, get_registry


def test_full_registry_flow():
    """验证注册表完整性：所有 spec 可查询、options 可序列化/反序列化"""
    reg = get_registry()
    for spec in reg.list_all():
        assert spec.name
        assert spec.display_name
        assert spec.description
        assert spec.options_class
        assert spec.create_pipeline
        assert spec.recognize

        opts = spec.options_class()
        d = opts.to_dict()
        assert d["pipeline"] == spec.name
        restored = spec.options_class.from_dict(d)
        assert restored.to_dict() == d


def test_pipeline_combo_has_all_registered():
    """UI 管道下拉应包含所有注册管道"""
    reg = get_registry()
    spec_names = {s.name for s in reg.list_all()}
    # All spec names should have a corresponding OCRPipeline enum value
    enum_values = {e.value for e in OCRPipeline}
    assert spec_names.issubset(enum_values)


def test_old_imports_still_work():
    """验证旧导入路径仍然可用"""
    from vibeocr.backend.models.ocr_options import OCROptions

    opts = OCROptions()
    assert opts.pipeline == OCRPipeline.OCR
    all_pipelines = get_all_pipelines()
    assert len(all_pipelines) == 6


def test_ocr_service_get_or_create():
    """OCRService 有 get_or_create_pipeline 方法"""
    from vibeocr.backend.services.ocr_service import OCRService

    svc = OCRService()
    assert hasattr(svc, "get_or_create_pipeline")
