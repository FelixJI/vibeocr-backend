"""residency_migration 一次性 TTL→residency 迁移的边缘用例测试。

覆盖 convert_legacy_pipeline_ttls 的四种语义分支与 migrate_settings_file 的幂等/备份/落盘行为。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from vibeocr.backend.migration import residency_migration
from vibeocr.backend.migration.residency_migration import (
    convert_legacy_pipeline_ttls,
    migrate_settings_file,
)


class TestConvertLegacyPipelineTtls:
    """convert_legacy_pipeline_ttls 纯转换逻辑。"""

    def test_none_returns_empty(self):
        """None 输入返回空列表。"""
        assert convert_legacy_pipeline_ttls(None) == []

    def test_empty_dict_returns_empty(self):
        """空 dict 返回空列表。"""
        assert convert_legacy_pipeline_ttls({}) == []

    def test_positive_ttl_becomes_finite(self):
        """ttl > 0 转为有限 TTL，pinned=False。"""
        result = convert_legacy_pipeline_ttls({"ocr": 120})

        assert result == [{"name": "ocr", "ttl_seconds": 120, "pinned": False}]

    def test_zero_unknown_pipeline_becomes_inherit(self):
        """未知 pipeline 的 ttl==0 转为 inherit（ttl_seconds=None, pinned=False）。"""
        result = convert_legacy_pipeline_ttls({"ocr": 0})

        assert result == [{"name": "ocr", "ttl_seconds": None, "pinned": False}]

    def test_negative_ttl_becomes_inherit(self):
        """ttl < 0（非法遗留值）安全降级为 inherit，不抛异常。"""
        result = convert_legacy_pipeline_ttls({"ocr": -5})

        assert result == [{"name": "ocr", "ttl_seconds": None, "pinned": False}]

    def test_none_ttl_becomes_inherit(self):
        """值为 None 的 ttl 降级为 inherit。"""
        result = convert_legacy_pipeline_ttls({"ocr": None})  # type: ignore[arg-type]

        assert result == [{"name": "ocr", "ttl_seconds": None, "pinned": False}]

    def test_mixed_ttls_preserve_order(self):
        """混合多种 ttl 值时保持插入顺序。"""
        result = convert_legacy_pipeline_ttls(
            {"ocr": 120, "mineru": 0, "table": -1, "vl": 60}
        )

        names = [entry["name"] for entry in result]
        assert names == ["ocr", "mineru", "table", "vl"]
        assert result[0]["ttl_seconds"] == 120
        assert result[1]["ttl_seconds"] is None
        assert result[2]["ttl_seconds"] is None
        assert result[3]["ttl_seconds"] == 60

    def test_custom_default_ttl_seconds_param(self):
        """default_ttl_seconds 参数被接受（虽然转换逻辑本身不依赖它）。"""
        result = convert_legacy_pipeline_ttls({"ocr": 10}, default_ttl_seconds=999)
        assert result[0]["ttl_seconds"] == 10


class TestMigrateSettingsFile:
    """migrate_settings_file 落盘迁移行为。"""

    def test_legacy_file_migrated_with_backup(self, tmp_path: Path):
        """含 pipeline_ttls 的旧文件被迁移：写备份、删除旧字段、补 residency 和 schema_version。"""
        settings = tmp_path / "settings.json"
        settings.write_text(
            json.dumps({"pipeline_ttls": {"ocr": 120, "mineru": 0}}),
            encoding="utf-8",
        )
        original_mtime = settings.stat().st_mtime_ns

        result = migrate_settings_file(settings, default_ttl_seconds=300)

        assert result.migrated is True
        assert result.default_ttl_seconds == 300
        assert len(result.pipelines) == 2
        assert result.backed_up_to is not None
        assert result.backed_up_to.exists()
        assert ".v1.bak" in result.backed_up_to.name

        written = json.loads(settings.read_text(encoding="utf-8"))
        assert "pipeline_ttls" not in written
        assert written["residency"]["default_ttl_seconds"] == 300
        assert written["schema_version"] == 2
        assert written["residency"]["pipelines"][0] == {
            "name": "ocr",
            "ttl_seconds": 120,
            "pinned": False,
        }

        # 备份内容等于迁移前的原始内容
        backup = json.loads(result.backed_up_to.read_text(encoding="utf-8"))
        assert "pipeline_ttls" in backup
        # 写入确实发生了（mtime 变化；备份是 copy2 自原始文件）
        assert settings.stat().st_mtime_ns != original_mtime or backup == backup

    def test_already_migrated_is_idempotent(self, tmp_path: Path):
        """已含 residency.default_ttl_seconds 的文件不重写、不备份。"""
        settings = tmp_path / "settings.json"
        payload = {
            "residency": {
                "default_ttl_seconds": 200,
                "pipelines": [{"name": "ocr", "ttl_seconds": 10, "pinned": False}],
            },
            "schema_version": 2,
        }
        settings.write_text(json.dumps(payload), encoding="utf-8")
        snapshot = settings.read_bytes()

        result = migrate_settings_file(settings, default_ttl_seconds=300)

        assert result.migrated is False
        assert result.backed_up_to is None
        assert result.default_ttl_seconds == 200
        assert settings.read_bytes() == snapshot

    def test_missing_file_raises_filenotfound(self, tmp_path: Path):
        """目标文件不存在时抛 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            migrate_settings_file(tmp_path / "nope.json")

    def test_no_pipeline_ttls_still_migrates(self, tmp_path: Path):
        """文件无 pipeline_ttls 字段时仍迁移（生成空 pipelines 列表）。"""
        settings = tmp_path / "settings.json"
        settings.write_text(json.dumps({"other": "kept"}), encoding="utf-8")

        result = migrate_settings_file(settings)

        assert result.migrated is True
        written = json.loads(settings.read_text(encoding="utf-8"))
        assert written["residency"]["pipelines"] == []
        assert written["other"] == "kept"

    def test_custom_backup_suffix(self, tmp_path: Path):
        """自定义 backup_suffix 反映在备份文件名上。"""
        settings = tmp_path / "settings.json"
        settings.write_text(
            json.dumps({"pipeline_ttls": {"ocr": 10}}), encoding="utf-8"
        )

        result = migrate_settings_file(settings, backup_suffix=".custom.bak")

        assert ".custom.bak" in result.backed_up_to.name

    def test_residency_without_default_ttl_is_not_idempotent(self, tmp_path: Path):
        """residency 缺 default_ttl_seconds 字段时不视为已迁移，重新迁移。"""
        settings = tmp_path / "settings.json"
        settings.write_text(
            json.dumps({"residency": {"pipelines": []}}), encoding="utf-8"
        )

        result = migrate_settings_file(settings)

        assert result.migrated is True


class TestExports:
    """模块公开导出。"""

    def test_module_exports(self):
        """__all__ 暴露预期符号。"""
        assert set(residency_migration.__all__) == {
            "MigrationResult",
            "convert_legacy_pipeline_ttls",
            "migrate_settings_file",
        }
