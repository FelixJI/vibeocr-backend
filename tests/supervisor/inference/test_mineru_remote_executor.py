"""远程 MinerU 连接的 Supervisor 链回归（Backend #113）。

覆盖：远程不 spawn/不 kill 本地进程、连接配置沿 settings 链应用、
活动任务门禁、store 失败回滚、无 mineru 包的远程解析。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from vibeocr.backend.services import mineru_readiness as readiness
from vibeocr.backend.services.mineru_config import MineruConfigError
from vibeocr.backend.supervisor import composition
from vibeocr.backend.supervisor.inference.budgets import InputItem
from vibeocr.backend.supervisor.inference.mineru_adapter import MinerUProcessAdapter
from vibeocr.backend.supervisor.inference.mineru_executor import MinerUExecutor
from vibeocr.backend.supervisor.inference.ocr_engines import (
    EngineAvailability,
    EngineDescriptor,
    OcrEngineRegistry,
    OcrEngineResolver,
)
from vibeocr.backend.supervisor.module import SupervisorModule, SupervisorOptions
from vibeocr.backend.supervisor.settings_store import RuntimeSettingsStoreError
from vibeocr.runtime_contracts import SettingsSnapshot
from vibeocr.runtime_contracts.dtos import OcrEngine

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch

REMOTE_EXTRA = {
    "mineru_connection": {
        "mode": "remote",
        "api_url": "https://mineru.example.com/api",
        "api_key": "secret-token-1",
    }
}


@pytest.fixture()
def isolated_connection(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(readiness, "_connection", readiness.MineruConnection())
    monkeypatch.setattr(readiness, "_observations", {})
    yield readiness
    monkeypatch.setattr(readiness, "_connection", readiness.MineruConnection())


class _RecordingLifecycle:
    """记录 start/stop，任何调用都会被断言捕获。"""

    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def file_parse(self, files, backend=None, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append([name for name, _ in files])
        return {name: {"markdown": "remote"} for name, _ in files}


def _adapter(
    lifecycle: _RecordingLifecycle | None = None, client: Any = None
) -> MinerUProcessAdapter:
    return MinerUProcessAdapter(
        client_factory=lambda: client or _RecordingClient(),
        lifecycle=lifecycle,
    )


def _raw_item(item_id: str, data: bytes = b"%PDF-a") -> InputItem:
    return InputItem(
        item_id=item_id,
        display_name=f"{item_id}.pdf",
        data=data,
        encoded_bytes=len(data),
        decoded_pixels=0,
        estimated_pages=1,
    )


def _snapshot(extra: dict[str, Any] | None = None, ttl: int = 300) -> SettingsSnapshot:
    return SettingsSnapshot(default_ttl_seconds=ttl, extra=extra or {})


# ---------------------------------------------------------------------------
# 远程不 spawn / 不 kill 本地
# ---------------------------------------------------------------------------


class TestRemoteNeverSpawnsOrKillsLocal:
    def test_remote_ensure_started_skips_lifecycle(self, isolated_connection):
        lifecycle = _RecordingLifecycle()
        adapter = _adapter(lifecycle)
        isolated_connection.configure_connection(
            isolated_connection.connection_from_extra(REMOTE_EXTRA)
        )
        adapter.ensure_started()
        adapter.recognize_many([_raw_item("it-0")])
        adapter.release_idle()
        adapter.close()
        assert lifecycle.started == 0
        assert lifecycle.stopped == 0

    def test_remote_residency_claims_no_local_process(self, isolated_connection):
        adapter = _adapter()
        isolated_connection.configure_connection(
            isolated_connection.connection_from_extra(REMOTE_EXTRA)
        )
        adapter.ensure_started()
        from vibeocr.runtime_contracts import ResidencyKind

        entry = adapter.residency_status().entries[0]
        assert entry.pipeline == "MinerU"
        assert entry.kind is ResidencyKind.EVICTED
        assert entry.remaining_ttl_seconds is None
        adapter.close()

    def test_remote_residency_hides_leftover_local_process(self, isolated_connection):
        """local 会话切到 remote 后，遗留本地进程不得显示为当前驻留。

        旧实现（无遮蔽）会把遗留的 _process_started 报成 SOFT_TTL 驻留；
        本地子进程仍由既有 release/TTL/close 路径回收，不影响远程服务器。
        """
        from vibeocr.runtime_contracts import ResidencyKind

        lifecycle = _RecordingLifecycle()
        adapter = _adapter(lifecycle)
        adapter.ensure_started()  # local 会话启动（fake lifecycle，无真实进程）
        assert adapter.residency_status().entries[0].kind is ResidencyKind.SOFT_TTL

        isolated_connection.configure_connection(
            isolated_connection.connection_from_extra(REMOTE_EXTRA)
        )
        entry = adapter.residency_status().entries[0]
        assert entry.kind is ResidencyKind.EVICTED
        assert entry.remaining_ttl_seconds is None

        # 遗留本地子进程仍由本地清理路径回收（本地清理，不碰远程）。
        adapter.release_idle()
        assert lifecycle.stopped == 1
        adapter.close()
        assert lifecycle.stopped == 1  # close 对已停进程不再重复 stop

    def test_remote_parse_runs_without_local_package(self, isolated_connection):
        """Base Runtime（无 mineru 包）远程解析：adapter 链全程不触发本地启动。"""
        client = _RecordingClient()
        lifecycle = _RecordingLifecycle()
        adapter = _adapter(lifecycle, client)
        isolated_connection.configure_connection(
            isolated_connection.connection_from_extra(REMOTE_EXTRA)
        )
        results = adapter.recognize_many([_raw_item("it-0"), _raw_item("it-1")])
        assert [item["markdown"] for item in results] == ["remote", "remote"]
        assert len(client.calls) == 1 and len(client.calls[0]) == 2
        assert lifecycle.started == 0
        adapter.close()


# ---------------------------------------------------------------------------
# settings 链：连接应用、活动任务门禁、回滚
# ---------------------------------------------------------------------------


class _FailingStore:
    def load(self, default: SettingsSnapshot) -> SettingsSnapshot:
        return default

    def replace(self, snapshot: SettingsSnapshot) -> None:
        raise RuntimeSettingsStoreError("replace failed")


def _module(tmp_path: Path, executor: Any, store: Any = None) -> SupervisorModule:
    return SupervisorModule(
        options=SupervisorOptions(instance_id="mineru-remote-test"),
        stager_root=tmp_path / "staging",
        executor=executor,
        settings_store=store,
    )


def _executor(lifecycle: _RecordingLifecycle | None = None) -> MinerUExecutor:
    lifecycle = lifecycle or _RecordingLifecycle()
    return MinerUExecutor(adapter_factory=lambda: _adapter(lifecycle))


class TestSettingsChain:
    def test_lazy_executor_applies_connection_before_adapter_exists(
        self, isolated_connection, tmp_path
    ):
        executor = _executor()
        module = _module(tmp_path, executor)
        assert isolated_connection.current_connection().mode == "local"
        module.update_settings(_snapshot(REMOTE_EXTRA))
        assert isolated_connection.current_connection().mode == "remote"
        assert module.settings().extra["mineru_connection"]["mode"] == "remote"
        module.shutdown_now()

    def test_invalid_connection_fails_update_without_persisting(
        self, isolated_connection, tmp_path
    ):
        executor = _executor()
        module = _module(tmp_path, executor)
        previous = module.settings()
        invalid = {
            "mineru_connection": {
                "mode": "remote",
                "api_url": "ftp://mineru.example.com",
            }
        }
        with pytest.raises(MineruConfigError):
            module.update_settings(_snapshot(invalid))
        assert module.settings() is previous
        assert isolated_connection.current_connection().mode == "local"
        module.shutdown_now()

    def test_store_failure_rolls_connection_back(self, isolated_connection, tmp_path):
        executor = _executor()
        module = _module(tmp_path, executor, _FailingStore())
        with pytest.raises(RuntimeSettingsStoreError):
            module.update_settings(_snapshot(REMOTE_EXTRA))
        assert isolated_connection.current_connection().mode == "local"
        assert module.settings().extra == {}
        module.shutdown_now()

    def test_store_failure_rolls_back_after_adapter_materialized(
        self, isolated_connection, tmp_path
    ):
        executor = _executor()
        module = _module(tmp_path, executor, _FailingStore())
        materialized = executor.adapter  # force adapter creation
        with pytest.raises(RuntimeSettingsStoreError):
            module.update_settings(_snapshot(REMOTE_EXTRA))
        assert isolated_connection.current_connection().mode == "local"
        assert materialized._settings is module.settings()
        module.shutdown_now()

    def test_active_tasks_block_connection_change(self, isolated_connection, tmp_path):
        executor = _executor()
        module = _module(tmp_path, executor)
        adapter = executor.adapter
        adapter._active_leases = 1  # 模拟在途 MinerU 解析
        try:
            with pytest.raises(MineruConfigError) as caught:
                module.update_settings(_snapshot(REMOTE_EXTRA))
            assert (
                caught.value.reason
                == "mineru_connection_change_blocked_by_active_tasks"
            )
            # 拒绝时既不落盘也不生效。
            assert module.settings().extra == {}
            assert isolated_connection.current_connection().mode == "local"
        finally:
            adapter._active_leases = 0
        # 任务结束后同一变更可以生效。
        module.update_settings(_snapshot(REMOTE_EXTRA))
        assert isolated_connection.current_connection().mode == "remote"
        module.shutdown_now()

    def test_active_tasks_allow_same_connection_settings_change(
        self, isolated_connection, tmp_path
    ):
        executor = _executor()
        module = _module(tmp_path, executor)
        module.update_settings(_snapshot(REMOTE_EXTRA))
        adapter = executor.adapter
        adapter._active_leases = 1
        try:
            # 连接不变，仅 TTL 调整：允许。
            module.update_settings(_snapshot(REMOTE_EXTRA, ttl=600))
            assert module.settings().default_ttl_seconds == 600
            assert isolated_connection.current_connection().mode == "remote"
        finally:
            adapter._active_leases = 0
        module.shutdown_now()


# ---------------------------------------------------------------------------
# mode 目录动态门禁
# ---------------------------------------------------------------------------


class _FakeEngine:
    def __init__(self, engine_id: OcrEngine) -> None:
        self.engine_id = engine_id

    def descriptor(self) -> EngineDescriptor:
        return EngineDescriptor(
            engine_id=self.engine_id,
            availability=EngineAvailability.READY,
            included_in_base=True,
        )


def _registry(use_mineru: bool):
    resolver = OcrEngineResolver(
        OcrEngineRegistry([_FakeEngine(engine) for engine in OcrEngine])
    )
    return composition._build_recognition_mode_registry(
        engine_resolver=resolver,
        use_paddle=False,
        use_mineru=use_mineru,
    )


class TestModeCatalogGating:
    def test_remote_mode_ready_without_local_package(
        self, isolated_connection, monkeypatch: MonkeyPatch
    ):
        monkeypatch.setattr(composition, "_mineru_available", lambda: False)
        isolated_connection.configure_connection(
            isolated_connection.connection_from_extra(REMOTE_EXTRA)
        )
        modes = {
            mode["id"]: mode for mode in _registry(True).catalog_payload()["modes"]
        }
        assert modes["mineru_document"]["availability"] == "ready"
        assert modes["mineru_document"]["required_component"] is None

    def test_local_mode_without_package_stays_blocked(
        self, isolated_connection, monkeypatch: MonkeyPatch
    ):
        monkeypatch.setattr(composition, "_mineru_available", lambda: False)
        isolated_connection.configure_connection(
            isolated_connection.connection_from_extra(REMOTE_EXTRA)
        )
        registry = _registry(True)
        # 切回 local：目录读取时动态降回准备态。
        isolated_connection.configure_connection(readiness.MineruConnection())
        modes = {mode["id"]: mode for mode in registry.catalog_payload()["modes"]}
        assert modes["mineru_document"]["availability"] == "preparation_required"
        assert modes["mineru_document"]["required_component"] == "mineru-cpu"

    def test_explicit_optout_stays_blocked_even_remote(self, isolated_connection):
        isolated_connection.configure_connection(
            isolated_connection.connection_from_extra(REMOTE_EXTRA)
        )
        modes = {
            mode["id"]: mode for mode in _registry(False).catalog_payload()["modes"]
        }
        assert modes["mineru_document"]["availability"] == "preparation_required"


# ---------------------------------------------------------------------------
# settings wire：PUT /v2/settings 接受/拒绝远程连接配置
# ---------------------------------------------------------------------------


class TestSettingsWire:
    async def test_put_settings_applies_remote_and_rejects_crlf_key(
        self, isolated_connection, tmp_path
    ):
        import httpx
        from vibeocr.backend.supervisor.app import create_app
        from vibeocr.backend.supervisor.bootstrap import generate_session_token

        executor = _executor()
        module = _module(tmp_path, executor)
        token = generate_session_token()
        transport = httpx.ASGITransport(app=create_app(module, token))
        body = {
            "schema_version": 2,
            "residency": {"default_ttl_seconds": 300, "pipelines": []},
            "extra": REMOTE_EXTRA,
        }
        bad_body = {
            "schema_version": 2,
            "residency": {"default_ttl_seconds": 300, "pipelines": []},
            "extra": {
                "mineru_connection": {
                    "mode": "remote",
                    "api_url": "https://mineru.example.com/api",
                    "api_key": "a\r\nb",
                }
            },
        }
        headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1", headers=headers
        ) as http:
            accepted = await http.put("/v2/settings", json=body)
            assert accepted.status_code == 200, accepted.text
            assert isolated_connection.current_connection().mode == "remote"
            rejected = await http.put("/v2/settings", json=bad_body)
            assert rejected.status_code == 400
            assert rejected.json()["code"] == "VALIDATION_ERROR"
            # 拒绝后连接保持原值。
            assert isolated_connection.current_connection().mode == "remote"
        module.shutdown_now()
