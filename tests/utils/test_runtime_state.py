"""runtime_state 缓存读写与有效性校验的边缘用例测试。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from vibeocr.backend import runtime_state


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """临时 project_root，提供干净的缓存目录。"""
    return tmp_path


@pytest.fixture(autouse=True)
def reset_machine_id(monkeypatch: pytest.MonkeyPatch):
    """每个用例前重置进程级 _machine_id 缓存，避免互相污染。"""
    monkeypatch.setattr(runtime_state, "_machine_id", None)
    yield
    monkeypatch.setattr(runtime_state, "_machine_id", None)


class TestGenerateMachineId:
    """generate_machine_id 边缘用例。"""

    def test_is_stable_hex(self):
        """同一进程多次调用返回相同的 SHA256 hex。"""
        first = runtime_state.generate_machine_id()
        second = runtime_state.generate_machine_id()
        assert first == second
        assert len(first) == 64
        # 确为十六进制
        int(first, 16)

    def test_global_cached(self):
        """第二次调用走缓存（_machine_id 已赋值）。"""
        runtime_state.generate_machine_id()
        assert runtime_state._machine_id is not None
        # 再次调用应返回同一对象字符串
        assert runtime_state.generate_machine_id() == runtime_state._machine_id


class TestCachePaths:
    """get_cache_dir / get_cache_path 边缘用例。"""

    def test_paths_under_state_dir(self, project_root: Path):
        """缓存目录与文件落在 data/backend 下。"""
        cache_dir = runtime_state.get_cache_dir(project_root)
        cache_path = runtime_state.get_cache_path(project_root)
        assert cache_dir == project_root.resolve() / "data" / "backend"
        assert cache_path == cache_dir / "runtime-state.json"


class TestLoadCache:
    """load_cache 边缘用例。"""

    def test_loads_valid_dict(self, project_root: Path):
        """正常 JSON dict 可读回。"""
        path = runtime_state.get_cache_path(project_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 1, "x": 1}), encoding="utf-8")
        assert runtime_state.load_cache(project_root) == {"version": 1, "x": 1}

    def test_missing_file_returns_none(self, project_root: Path):
        """文件不存在返回 None。"""
        assert runtime_state.load_cache(project_root) is None

    def test_corrupt_json_returns_none(self, project_root: Path):
        """损坏 JSON 返回 None。"""
        path = runtime_state.get_cache_path(project_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not valid json", encoding="utf-8")
        assert runtime_state.load_cache(project_root) is None

    def test_non_dict_top_level_returns_none(self, project_root: Path):
        """顶层非 dict（如 list/数字）返回 None。"""
        path = runtime_state.get_cache_path(project_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        for raw in ("[1, 2, 3]", '"a string"', "42", "true", "null"):
            path.write_text(raw, encoding="utf-8")
            assert runtime_state.load_cache(project_root) is None


class TestSaveCache:
    """save_cache 边缘用例。"""

    def test_writes_atomically_and_content_correct(self, project_root: Path):
        """原子写后内容可正确读回，无残留 tmp。"""
        data = {"version": 1, "machine_id": "abc", "deps": {}}
        assert runtime_state.save_cache(project_root, data) is True

        cache_path = runtime_state.get_cache_path(project_root)
        tmp_path = cache_path.with_suffix(".json.tmp")
        assert cache_path.exists()
        assert not tmp_path.exists()
        assert json.loads(cache_path.read_text(encoding="utf-8")) == data

    def test_creates_parent_dirs(self, project_root: Path):
        """父目录不存在时自动创建。"""
        assert not runtime_state.get_cache_dir(project_root).exists()
        runtime_state.save_cache(project_root, {"v": 1})
        assert runtime_state.get_cache_dir(project_root).exists()

    def test_oserror_returns_false_and_cleans_tmp(
        self, project_root: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """写失败时返回 False 并清理 tmp 文件。"""
        cache_path = runtime_state.get_cache_path(project_root)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # 让 temporary.write_text 抛 OSError，但 unlink 正常工作。
        real_write = Path.write_text
        tmp_created = {"path": None}

        def fake_write(self, data, *args, **kwargs):
            if self.suffix == ".tmp":
                tmp_created["path"] = self
                raise OSError("disk full")
            return real_write(self, data, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fake_write)
        assert runtime_state.save_cache(project_root, {"v": 1}) is False
        # tmp 已被 unlink(missing_ok=True)
        if tmp_created["path"] is not None:
            assert not tmp_created["path"].exists()


class TestIsCacheValid:
    """is_cache_valid 边缘用例。"""

    def test_valid_cache(self, project_root: Path):
        """version 与 machine_id 匹配时为有效。"""
        mid = runtime_state.generate_machine_id()
        runtime_state.save_cache(
            project_root,
            {"version": runtime_state.CACHE_VERSION, "machine_id": mid},
        )
        ok, data = runtime_state.is_cache_valid(project_root)
        assert ok is True
        assert data is not None
        assert data["machine_id"] == mid

    def test_version_mismatch(self, project_root: Path):
        """version 不匹配返回 False。"""
        runtime_state.save_cache(
            project_root,
            {"version": 9999, "machine_id": runtime_state.generate_machine_id()},
        )
        assert runtime_state.is_cache_valid(project_root) == (False, None)

    def test_machine_id_mismatch(
        self, project_root: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """machine_id 不匹配返回 False。"""
        runtime_state.save_cache(
            project_root,
            {
                "version": runtime_state.CACHE_VERSION,
                "machine_id": "different-machine",
            },
        )
        ok, data = runtime_state.is_cache_valid(project_root)
        assert ok is False
        assert data is None

    def test_missing_cache_invalid(self, project_root: Path):
        """缓存不存在视为无效。"""
        assert runtime_state.is_cache_valid(project_root) == (False, None)


class TestUpdateCacheField:
    """update_cache_field 边缘用例。"""

    def test_invalid_cache_returns_false(self, project_root: Path):
        """缓存无效时返回 False 且不写。"""
        assert runtime_state.update_cache_field(project_root, "k", "v") is False
        assert not runtime_state.get_cache_path(project_root).exists()

    def test_writes_single_field_preserves_others(self, project_root: Path):
        """有效缓存时增量写单字段，保留其余字段。"""
        mid = runtime_state.generate_machine_id()
        runtime_state.save_cache(
            project_root,
            {"version": runtime_state.CACHE_VERSION, "machine_id": mid, "keep": 1},
        )
        assert runtime_state.update_cache_field(project_root, "new", 42) is True
        _, data = runtime_state.is_cache_valid(project_root)
        assert data["new"] == 42
        assert data["keep"] == 1


class TestGetCacheAgeSeconds:
    """get_cache_age_seconds 边缘用例。"""

    def test_valid_age(self, project_root: Path):
        """正常 ISO 时间返回非负秒数。"""
        recent = (datetime.now() - timedelta(seconds=10)).isoformat()
        runtime_state.save_cache(project_root, {"last_check_time": recent})
        age = runtime_state.get_cache_age_seconds(project_root)
        assert age is not None
        assert age >= 10

    def test_missing_field_returns_none(self, project_root: Path):
        """缺失 last_check_time 返回 None。"""
        runtime_state.save_cache(project_root, {"other": 1})
        assert runtime_state.get_cache_age_seconds(project_root) is None

    def test_malformed_field_returns_none(self, project_root: Path):
        """字段非字符串或非法 ISO 返回 None。"""
        runtime_state.save_cache(project_root, {"last_check_time": 12345})
        assert runtime_state.get_cache_age_seconds(project_root) is None
        runtime_state.save_cache(project_root, {"last_check_time": "not-a-date"})
        assert runtime_state.get_cache_age_seconds(project_root) is None

    def test_missing_cache_returns_none(self, project_root: Path):
        """无缓存文件返回 None。"""
        assert runtime_state.get_cache_age_seconds(project_root) is None


class TestCreateCacheEntry:
    """create_cache_entry 边缘用例。"""

    def test_creates_entry_with_metadata(self, project_root: Path):
        """生成含 version/machine_id/python_version 的有效条目。"""
        entry = runtime_state.create_cache_entry(project_root, {"d": 1}, {"h": 2})
        assert entry is not None
        assert entry["version"] == runtime_state.CACHE_VERSION
        assert entry["dependencies"] == {"d": 1}
        assert entry["hardware_info"] == {"h": 2}
        assert "machine_id" in entry
        assert "python_version" in entry
        assert "last_check_time" in entry
        # 落盘且有效
        ok, data = runtime_state.is_cache_valid(project_root)
        assert ok and data is not None
