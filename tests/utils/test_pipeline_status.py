"""pipeline_status 成功标记读写的边缘用例测试。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from vibeocr.backend import pipeline_status, runtime_state


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """临时 project_root。"""
    return tmp_path


@pytest.fixture(autouse=True)
def reset_machine_id(monkeypatch: pytest.MonkeyPatch):
    """隔离 runtime_state 的进程级 _machine_id。"""
    monkeypatch.setattr(runtime_state, "_machine_id", None)
    yield
    monkeypatch.setattr(runtime_state, "_machine_id", None)


@pytest.fixture
def fixed_today(monkeypatch: pytest.MonkeyPatch):
    """固定 _today() 返回值，使跨天断言确定。"""
    today = date(2026, 8, 2)
    monkeypatch.setattr(pipeline_status, "_today", lambda: today)
    return today


class TestIsSuccessEntryToday:
    """_is_success_entry_today 边缘用例。"""

    def test_legacy_bool_true(self, fixed_today):
        """旧格式布尔 True 直通。"""
        assert pipeline_status._is_success_entry_today(True) is True

    def test_legacy_bool_false(self, fixed_today):
        """旧格式布尔 False 直通。"""
        assert pipeline_status._is_success_entry_today(False) is False

    def test_dict_today_succeeded(self, fixed_today):
        """新格式当天 succeeded=True。"""
        entry = {"succeeded": True, "date": fixed_today.isoformat()}
        assert pipeline_status._is_success_entry_today(entry) is True

    def test_dict_other_day(self, fixed_today):
        """新格式跨天视为未成功。"""
        entry = {"succeeded": True, "date": "2020-01-01"}
        assert pipeline_status._is_success_entry_today(entry) is False

    def test_dict_succeeded_false(self, fixed_today):
        """succeeded=False 即使日期今天也返回 False。"""
        entry = {"succeeded": False, "date": fixed_today.isoformat()}
        assert pipeline_status._is_success_entry_today(entry) is False

    def test_dict_non_str_date(self, fixed_today):
        """date 非 str 返回 False。"""
        assert (
            pipeline_status._is_success_entry_today({"succeeded": True, "date": 123})
            is False
        )

    def test_dict_missing_date(self, fixed_today):
        """缺 date 键返回 False。"""
        assert pipeline_status._is_success_entry_today({"succeeded": True}) is False

    def test_unknown_type(self, fixed_today):
        """未知类型（int/list/None）返回 False。"""
        assert pipeline_status._is_success_entry_today(42) is False
        assert pipeline_status._is_success_entry_today([1, 2]) is False
        assert pipeline_status._is_success_entry_today(None) is False


class TestPipelineNamesConstants:
    """常量完整性。"""

    def test_local_markable_subset_of_pipeline_names(self):
        """LOCAL_MARKABLE_PIPELINES 必须全部在 PIPELINE_NAMES 内。"""
        assert pipeline_status.LOCAL_MARKABLE_PIPELINES.issubset(
            pipeline_status.PIPELINE_NAMES
        )

    def test_mineru_not_in_local_markable(self):
        """MinerU 走远程独立标记，不在本地集合。"""
        assert "MinerU" not in pipeline_status.LOCAL_MARKABLE_PIPELINES
        assert "MinerU" in pipeline_status.PIPELINE_NAMES


class TestIsPipelineEverSucceeded:
    """is_pipeline_ever_succeeded 边缘用例。"""

    def test_invalid_cache_returns_false(self, project_root: Path):
        """缓存无效返回 False。"""
        assert pipeline_status.is_pipeline_ever_succeeded("OCR", project_root) is False

    def test_today_success_true(self, project_root: Path, fixed_today):
        """当天成功标记返回 True。"""
        runtime_state.create_cache_entry(project_root, {}, {})
        pipeline_status.mark_pipeline_success("OCR", project_root)
        assert pipeline_status.is_pipeline_ever_succeeded("OCR", project_root) is True

    def test_other_day_success_false(self, project_root: Path, fixed_today):
        """跨天成功标记返回 False。"""
        runtime_state.create_cache_entry(project_root, {}, {})
        # 直接写入昨天的成功标记
        runtime_state.update_cache_field(
            project_root,
            "pipeline_success",
            {"OCR": {"succeeded": True, "date": "2020-01-01"}},
        )
        assert pipeline_status.is_pipeline_ever_succeeded("OCR", project_root) is False

    def test_missing_pipeline_false(self, project_root: Path, fixed_today):
        """有效缓存但无该管道记录返回 False。"""
        runtime_state.create_cache_entry(project_root, {}, {})
        assert (
            pipeline_status.is_pipeline_ever_succeeded("NeverRun", project_root)
            is False
        )

    def test_legacy_bool_success_true(self, project_root: Path, fixed_today):
        """旧格式布尔 True 视为今天成功（宽容向后兼容）。"""
        runtime_state.create_cache_entry(project_root, {}, {})
        runtime_state.update_cache_field(
            project_root, "pipeline_success", {"OCR": True}
        )
        assert pipeline_status.is_pipeline_ever_succeeded("OCR", project_root) is True


class TestMarkPipelineSuccess:
    """mark_pipeline_success 边缘用例。"""

    def test_invalid_cache_no_write(self, project_root: Path):
        """缓存无效时静默不写，且不创建缓存文件。"""
        pipeline_status.mark_pipeline_success("OCR", project_root)
        assert not runtime_state.get_cache_path(project_root).exists()

    def test_writes_new_format_today(self, project_root: Path, fixed_today):
        """有效缓存写入新格式 {succeeded, date=今天}。"""
        runtime_state.create_cache_entry(project_root, {}, {})
        pipeline_status.mark_pipeline_success("OCR", project_root)
        _, data = runtime_state.is_cache_valid(project_root)
        assert data["pipeline_success"]["OCR"] == {
            "succeeded": True,
            "date": fixed_today.isoformat(),
        }

    def test_preserves_other_pipelines(self, project_root: Path, fixed_today):
        """标记新管道不破坏已有其它管道记录。"""
        runtime_state.create_cache_entry(project_root, {}, {})
        runtime_state.update_cache_field(
            project_root,
            "pipeline_success",
            {"FORMULA_RECOGNITION": {"succeeded": True, "date": "2020-01-01"}},
        )
        pipeline_status.mark_pipeline_success("OCR", project_root)
        _, data = runtime_state.is_cache_valid(project_root)
        assert "FORMULA_RECOGNITION" in data["pipeline_success"]
        assert data["pipeline_success"]["OCR"]["succeeded"] is True
