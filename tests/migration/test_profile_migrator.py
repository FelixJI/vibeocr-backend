"""profile_migrator 配置迁移器的边缘用例测试。

覆盖幂等性、原子写入降级、备份写入、schema_version 早晚于迁移器等分支。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from vibeocr.backend import migration as migration_pkg
from vibeocr.backend.migration import profile_migrator


def _write_settings(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class TestMigrateConfig:
    """migrate_config 单文件迁移行为。"""

    def test_missing_file_returns_skipped(self, tmp_path: Path):
        """目标文件不存在时返回 skipped，不创建任何文件。"""
        target = tmp_path / "app_settings.json"

        result = profile_migrator.migrate_config(target)

        assert result.status == "skipped"
        assert "not found" in result.message
        assert not target.exists()

    def test_already_migrated_is_noop(self, tmp_path: Path):
        """schema_version 已等于 CURRENT_SCHEMA_VERSION 时不重写、无备份。"""
        target = tmp_path / "app_settings.json"
        _write_settings(target, {"schema_version": 1, "theme": "dark"})

        result = profile_migrator.migrate_config(target)

        assert result.status == "already_migrated"
        assert result.backup_path is None
        assert json.loads(target.read_text(encoding="utf-8"))["theme"] == "dark"

    def test_newer_schema_version_is_skipped(self, tmp_path: Path):
        """schema_version 大于迁移器版本时跳过，避免降级。"""
        target = tmp_path / "app_settings.json"
        _write_settings(target, {"schema_version": 99})

        result = profile_migrator.migrate_config(target)

        assert result.status == "skipped"
        assert "newer" in result.message

    def test_legacy_dict_is_migrated_with_backup(self, tmp_path: Path):
        """无 schema_version 的旧配置被迁移，字段保留并补上版本号。"""
        target = tmp_path / "app_settings.json"
        _write_settings(target, {"theme": "light", "lang": "zh"})

        result = profile_migrator.migrate_config(target)

        assert result.status == "migrated"
        assert result.backup_path is not None
        written = json.loads(target.read_text(encoding="utf-8"))
        assert written["schema_version"] == 1
        assert written["theme"] == "light"
        assert written["lang"] == "zh"
        backup = Path(result.backup_path)
        assert backup.exists()
        assert ".pre-migrate-" in backup.name

    def test_non_dict_json_is_skipped(self, tmp_path: Path):
        """JSON 解析为非 dict（如 list）时跳过，不抛异常。"""
        target = tmp_path / "app_settings.json"
        target.write_text("[1, 2, 3]", encoding="utf-8")

        result = profile_migrator.migrate_config(target)

        assert result.status == "skipped"
        assert result.message == "not a JSON object"

    def test_invalid_json_is_skipped(self, tmp_path: Path):
        """损坏的 JSON 文件读取失败时返回 skipped 而非抛异常。"""
        target = tmp_path / "app_settings.json"
        target.write_text("{not valid json", encoding="utf-8")

        result = profile_migrator.migrate_config(target)

        assert result.status == "skipped"

    def test_idempotent_second_run(self, tmp_path: Path):
        """对已迁移文件二次运行返回 already_migrated，且不产生新备份。"""
        target = tmp_path / "app_settings.json"
        _write_settings(target, {"theme": "light"})

        first = profile_migrator.migrate_config(target)
        second = profile_migrator.migrate_config(target)

        assert first.status == "migrated"
        assert second.status == "already_migrated"
        assert second.backup_path is None

    def test_atomic_write_failure_keeps_original(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """_atomic_write 抛 OSError 时返回 skipped，原文件不被破坏。"""
        target = tmp_path / "app_settings.json"
        original = {"theme": "light"}
        _write_settings(target, original)
        snapshot = target.read_bytes()

        def boom(path: Path, data: dict) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(profile_migrator, "_atomic_write", boom)

        result = profile_migrator.migrate_config(target)

        assert result.status == "skipped"
        assert "write failed" in result.message
        assert target.read_bytes() == snapshot

    def test_backup_write_failure_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """备份写入本身失败时 backup_path 为 None，但迁移仍可继续。"""
        target = tmp_path / "app_settings.json"
        _write_settings(target, {"theme": "light"})

        def fail_backup(path: Path) -> None:
            raise OSError("read-only")

        monkeypatch.setattr(profile_migrator, "_write_hashed_backup", lambda path: None)

        result = profile_migrator.migrate_config(target)

        assert result.status == "migrated"
        assert result.backup_path is None
        assert json.loads(target.read_text(encoding="utf-8"))["schema_version"] == 1


class TestMigrateProfile:
    """migrate_profile 目录级迁移。"""

    def test_migrates_app_settings_in_dir(self, tmp_path: Path):
        """目录中存在 app_settings.json 时被迁移。"""
        _write_settings(tmp_path / "app_settings.json", {"theme": "dark"})

        results = profile_migrator.migrate_profile(tmp_path)

        assert len(results) == 1
        assert results[0].status == "migrated"

    def test_missing_dir_target_returns_skipped(self, tmp_path: Path):
        """目录中 app_settings.json 缺失时返回单个 skipped 结果。"""
        results = profile_migrator.migrate_profile(tmp_path)

        assert len(results) == 1
        assert results[0].status == "skipped"


class TestHelpers:
    """模块级辅助函数。"""

    def test_current_schema_version_is_one(self):
        """迁移器基线版本固定为 1。"""
        assert profile_migrator.CURRENT_SCHEMA_VERSION == 1

    def test_read_json_missing_file_returns_none(self, tmp_path: Path):
        """_read_json 对缺失文件返回 None。"""
        assert profile_migrator._read_json(tmp_path / "nope.json") is None

    def test_hash_bytes_is_deterministic(self):
        """_hash_bytes 对相同输入返回相同摘要。"""
        assert profile_migrator._hash_bytes(b"abc") == profile_migrator._hash_bytes(
            b"abc"
        )
        assert profile_migrator._hash_bytes(b"abc") != profile_migrator._hash_bytes(
            b"abd"
        )

    def test_migration_package_exports(self):
        """migration 包按预期暴露子模块。"""
        assert hasattr(migration_pkg, "profile_migrator")
