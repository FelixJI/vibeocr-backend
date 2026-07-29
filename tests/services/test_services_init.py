"""Backend services 使用最终物理模块，不保留旧命名空间兼容导出。"""

import pytest


def test_ocr_service_constructs_from_final_module():
    """OCRService 由 Backend 最终所有者模块直接导出。"""
    from vibeocr.backend.services.ocr_service import OCRService

    svc = OCRService()
    assert svc is not None
    assert svc.__class__.__name__ == "OCRService"


def test_services_namespace_has_no_compat_ocr_service_export():
    import vibeocr.backend.services as services

    with pytest.raises(AttributeError):
        _ = services.OCRService


def test_getattr_unknown_attribute_raises():
    """请求已删除的历史导出时 raise AttributeError（line 33）。"""
    import vibeocr.backend.services as services

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = services.OCRServiceSubprocess


def test_getattr_other_unknown_attribute_raises():
    with pytest.raises(AttributeError, match="MinerUBatchService"):
        import vibeocr.backend.services as services

        _ = services.MinerUBatchService
