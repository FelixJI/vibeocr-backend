# tests/core/test_pipeline_registry_all.py
"""验证全局注册表包含所有已定义的 PipelineSpec"""

from vibeocr.backend.core.pipelines import get_registry


def test_registry_has_all_pipelines():
    reg = get_registry()
    names = [s.name for s in reg.list_all()]
    assert "OCR" in names
    assert "PP-StructureV3" in names
    assert "TABLE_RECOGNITION" in names
    assert "FORMULA_RECOGNITION" in names
    assert "MinerU" in names
    assert "PaddleOCR-VL" in names


def test_registry_get_each():
    reg = get_registry()
    for name in [
        "OCR",
        "PP-StructureV3",
        "TABLE_RECOGNITION",
        "FORMULA_RECOGNITION",
        "MinerU",
        "PaddleOCR-VL",
    ]:
        spec = reg.get(name)
        assert spec.name == name
        assert spec.options_class is not None


def test_registry_spec_count():
    """注册表应恰好包含 6 个管道"""
    reg = get_registry()
    assert len(reg.list_all()) == 6


def test_registry_has_method():
    reg = get_registry()
    assert reg.has("OCR")
    assert reg.has("PP-StructureV3")
    assert not reg.has("NONEXISTENT")


def test_each_spec_has_callable_create_and_recognize():
    """每个 spec 的 create_pipeline 和 recognize 应可调用"""
    reg = get_registry()
    for spec in reg.list_all():
        assert callable(spec.create_pipeline), (
            f"{spec.name} create_pipeline not callable"
        )
        assert callable(spec.recognize), f"{spec.name} recognize not callable"


def test_mineru_spec_raises_not_implemented():
    """MinerU 的 create_pipeline 和 recognize 应抛出 NotImplementedError"""
    reg = get_registry()
    mineru = reg.get("MinerU")
    import pytest

    with pytest.raises(NotImplementedError):
        mineru.create_pipeline("cpu")
    with pytest.raises(NotImplementedError):
        mineru.recognize(None, None, None)


def test_get_registry_returns_same_instance():
    """get_registry() 应返回同一个注册表实例"""
    from vibeocr.backend.core.pipelines import get_registry as gr2

    assert get_registry() is gr2()
