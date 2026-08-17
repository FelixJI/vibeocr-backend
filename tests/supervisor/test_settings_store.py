"""Durable Runtime Settings store contracts."""

from __future__ import annotations

import json

import pytest
from vibeocr.backend.runtime_selection import RuntimeSelectionError
from vibeocr.backend.supervisor import main as supervisor_main
from vibeocr.backend.supervisor.module import SupervisorModule, SupervisorOptions
from vibeocr.backend.supervisor.settings_store import (
    RuntimeSettingsStore,
    RuntimeSettingsStoreError,
)
from vibeocr.runtime_contracts import ResidencyStatus, SettingsSnapshot


def _snapshot(*, ttl: int = 300, sources: tuple[str, ...] = ()) -> SettingsSnapshot:
    return SettingsSnapshot(
        default_ttl_seconds=ttl,
        pipelines=(),
        extra={},
        download_source_ids=sources,
    )


def test_load_returns_default_when_no_settings_file_exists(tmp_path) -> None:
    default = _snapshot(ttl=123)

    loaded = RuntimeSettingsStore(tmp_path / "settings.json").load(default)

    assert loaded is default


def test_replace_survives_a_new_store_instance(tmp_path) -> None:
    path = tmp_path / "settings.json"
    expected = _snapshot(ttl=600, sources=("pypi",))

    RuntimeSettingsStore(path).replace(expected)

    assert RuntimeSettingsStore(path).load(_snapshot()) == expected


def test_load_fails_closed_for_invalid_json_or_unknown_schema(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("not json", encoding="utf-8")

    with pytest.raises(RuntimeSettingsStoreError, match="invalid"):
        RuntimeSettingsStore(path).load(_snapshot())

    path.write_text(
        json.dumps({"schema_version": 999, "settings": {}}), encoding="utf-8"
    )
    with pytest.raises(RuntimeSettingsStoreError, match="schema"):
        RuntimeSettingsStore(path).load(_snapshot())

    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "settings": {
                    "schema_version": 999,
                    "residency": {"default_ttl_seconds": 300, "pipelines": []},
                    "extra": {},
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeSettingsStoreError, match="invalid"):
        RuntimeSettingsStore(path).load(_snapshot())


def test_replace_failure_preserves_the_previous_durable_snapshot(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "settings.json"
    store = RuntimeSettingsStore(path)
    previous = _snapshot(ttl=300)
    store.replace(previous)
    monkeypatch.setattr(
        "vibeocr.backend.supervisor.settings_store.os.replace",
        lambda _source, _destination: (_ for _ in ()).throw(OSError("disk busy")),
    )

    with pytest.raises(RuntimeSettingsStoreError, match="replace"):
        store.replace(_snapshot(ttl=600))

    assert RuntimeSettingsStore(path).load(_snapshot()) == previous


class _RecordingExecutor:
    def __init__(self) -> None:
        self.configured: list[SettingsSnapshot] = []

    def configure_settings(self, snapshot: SettingsSnapshot) -> ResidencyStatus:
        self.configured.append(snapshot)
        return ResidencyStatus(
            default_ttl_seconds=snapshot.default_ttl_seconds,
            pipelines=snapshot.pipelines,
        )


class _FailingStore:
    def load(self, default: SettingsSnapshot) -> SettingsSnapshot:
        return default

    def replace(self, snapshot: SettingsSnapshot) -> None:
        del snapshot
        raise RuntimeSettingsStoreError("replace failed")


def _module(tmp_path, executor: _RecordingExecutor, store) -> SupervisorModule:
    return SupervisorModule(
        options=SupervisorOptions(instance_id="settings-test"),
        stager_root=tmp_path / "staging",
        executor=executor,
        settings_store=store,
    )


def test_settings_survive_a_supervisor_module_restart(tmp_path) -> None:
    store = RuntimeSettingsStore(tmp_path / "settings.json")
    first_executor = _RecordingExecutor()
    first = _module(tmp_path, first_executor, store)
    expected = _snapshot(ttl=600, sources=("pypi",))

    first.update_settings(expected)

    second_executor = _RecordingExecutor()
    second = _module(tmp_path, second_executor, RuntimeSettingsStore(store.path))
    assert second.settings() == expected
    assert second_executor.configured == [expected]


def test_supervisor_restart_fails_closed_for_unknown_persisted_source(
    tmp_path,
) -> None:
    store = RuntimeSettingsStore(tmp_path / "settings.json")
    store.replace(_snapshot(sources=("removed-source",)))

    with pytest.raises(RuntimeSelectionError, match="unknown download source"):
        _module(tmp_path, _RecordingExecutor(), store)


def test_failed_persist_keeps_module_memory_and_restores_executor(tmp_path) -> None:
    executor = _RecordingExecutor()
    module = _module(tmp_path, executor, _FailingStore())
    previous = module.settings()
    requested = _snapshot(ttl=600)

    with pytest.raises(RuntimeSettingsStoreError, match="replace failed"):
        module.update_settings(requested)

    assert module.settings() == previous
    assert executor.configured[-2:] == [requested, previous]


def test_supervisor_settings_path_prefers_explicit_environment(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "explicit.json"
    monkeypatch.setenv("VIBEOCR_SUPERVISOR_SETTINGS", str(explicit))
    monkeypatch.setenv("VIBEOCR_SUP_ROOT", str(tmp_path / "root"))

    assert supervisor_main._settings_store_from_environment().path == explicit


def test_supervisor_settings_path_falls_back_to_supervisor_root(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    monkeypatch.delenv("VIBEOCR_SUPERVISOR_SETTINGS", raising=False)
    monkeypatch.setenv("VIBEOCR_SUP_ROOT", str(root))

    assert supervisor_main._settings_store_from_environment().path == (
        root / "supervisor-settings.json"
    )
