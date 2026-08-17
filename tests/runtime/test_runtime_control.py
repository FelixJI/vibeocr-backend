from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from vibeocr.backend import runtime_maintenance
from vibeocr.backend.runtime_control import RuntimeControl
from vibeocr.backend.runtime_lock import RuntimeLockTimeout
from vibeocr.backend.runtime_maintenance import (
    RuntimeCapabilityError,
    RuntimeCommandConflict,
    RuntimeCursorExpired,
    RuntimeInstallFailure,
    RuntimeOperationConflict,
    RuntimeOperationNotRetryable,
    RuntimeOperationStore,
    RuntimeSourceIdentityMismatch,
)


def _snapshot(operation_id: str, sequence: int, state: str = "running") -> dict:
    return {
        "operation_id": operation_id,
        "sequence": sequence,
        "operation": "repair",
        "operation_state": state,
        "phase": "install_profile",
        "profile_id": "win-x64-cpu",
        "updated_at": "2026-08-05T12:00:00Z",
    }


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing violation contract")
def test_atomic_json_waits_for_transient_windows_reader(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    runtime_maintenance._atomic_json(metadata_path, {"value": 1})
    opened = threading.Event()

    def hold_target_open() -> None:
        with metadata_path.open("r", encoding="utf-8"):
            opened.set()
            time.sleep(0.05)

    with ThreadPoolExecutor(max_workers=1) as executor:
        reader = executor.submit(hold_target_open)
        assert opened.wait(timeout=1)
        runtime_maintenance._atomic_json(metadata_path, {"value": 2})
        reader.result(timeout=1)

    assert json.loads(metadata_path.read_text(encoding="utf-8")) == {"value": 2}


def test_operation_id_is_durable_and_bound_to_normalized_intent(tmp_path: Path) -> None:
    intent = {
        "operation": "repair",
        "profile_id": "win-x64-cpu",
        "component_ids": ["ocr_engine"],
        "required_capabilities": ["runtime.component-repair.v1"],
        "source_identity": {"backend_source_sha": "a" * 40},
    }
    store = RuntimeOperationStore(tmp_path)

    started = store.start("op-1", intent)
    assert started.created is True
    assert started.snapshot is not None
    assert started.snapshot["sequence"] == 1
    assert store.start("op-1", {**intent}).created is False

    restarted = RuntimeOperationStore(tmp_path)
    replayed = restarted.start("op-1", {**intent})
    assert replayed.created is False
    assert replayed.snapshot == started.snapshot
    with pytest.raises(RuntimeOperationConflict):
        restarted.start("op-1", {**intent, "component_ids": ["runtime_base"]})
    with pytest.raises(RuntimeOperationConflict):
        restarted.start("op-1", {**intent, "required_capabilities": []})
    with pytest.raises(RuntimeOperationConflict):
        restarted.start(
            "op-1",
            {**intent, "source_identity": {"backend_source_sha": "b" * 40}},
        )


def test_retry_reuses_source_intent_and_links_new_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_identity = {"backend_source_sha": "a" * 40}
    store = RuntimeOperationStore(tmp_path)
    store.start(
        "op-1",
        {
            "operation": "repair",
            "profile_id": "win-x64-cpu",
            "component_ids": ["ocr_engine"],
            "required_capabilities": ["runtime.component-repair.v1"],
            "source_identity": source_identity,
        },
    )
    store.append(
        "op-1",
        event_type="snapshot",
        snapshot=_snapshot("op-1", 2, "failed"),
        message_code="runtime.failed",
    )
    control = object.__new__(RuntimeControl)
    control._store = store
    control._installer = lambda: type("Installer", (), {"manifest": object()})()  # type: ignore[method-assign]
    monkeypatch.setattr(
        "vibeocr.backend.runtime_control.runtime_source_identity",
        lambda _manifest: source_identity,
    )
    calls: list[dict] = []

    def execute(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return {"operation_id": kwargs["operation_id"]}

    control.execute = execute  # type: ignore[method-assign]

    assert control.command(
        command_id="retry-1",
        command="retry",
        target_operation_id="op-1",
        new_operation_id="op-2",
    ) == {"operation_id": "op-2"}
    assert calls == [
        {
            "operation": "repair",
            "operation_id": "op-2",
            "component_ids": ("ocr_engine",),
            "required_capabilities": ("runtime.component-repair.v1",),
            "source_operation_id": "op-1",
            "profile_id": "win-x64-cpu",
            # retry 省略选择字段：source intent 未携带 install/source，
            # 传 None / 空集由 installer 解析为 Backend 缺省。
            "install_component_ids": None,
            "download_source_ids": (),
        }
    ]


def test_start_recovers_reservation_crash_before_initial_event(tmp_path: Path) -> None:
    intent = {"operation": "repair", "profile_id": "win-x64-cpu"}
    store = RuntimeOperationStore(tmp_path)
    store.start("op-1", intent)
    operation_directory = next((tmp_path / "runtime-operations").iterdir())
    metadata_path = operation_directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["snapshot"] = None
    metadata["through_sequence"] = 0
    metadata["oldest_sequence"] = 1
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    (operation_directory / "events.jsonl").unlink()

    recovered = RuntimeOperationStore(tmp_path).start("op-1", intent)

    assert recovered.created is True
    assert recovered.snapshot is not None
    assert recovered.snapshot["sequence"] == 1
    assert recovered.event is not None


def test_start_reconciles_durable_event_after_metadata_replace_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent = {"operation": "repair", "profile_id": "win-x64-cpu"}
    store = RuntimeOperationStore(tmp_path)
    original_atomic_json = runtime_maintenance._atomic_json

    def crash_metadata(path: Path, value: dict) -> None:
        if path.name == "metadata.json" and value.get("through_sequence") == 1:
            raise OSError("simulated crash after event fsync")
        original_atomic_json(path, value)

    monkeypatch.setattr(
        "vibeocr.backend.runtime_maintenance._atomic_json", crash_metadata
    )
    with pytest.raises(OSError, match="after event fsync"):
        store.start("op-1", intent)

    monkeypatch.undo()
    projection = RuntimeOperationStore(tmp_path).latest_projection()
    recovered = RuntimeOperationStore(tmp_path).start("op-1", intent)
    operation_directory = next((tmp_path / "runtime-operations").iterdir())
    events = (
        (operation_directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    )

    assert projection is not None
    assert projection["sequence"] == 1
    assert projection["message_code"] == "runtime.validate_binding"
    assert recovered.created is False
    assert recovered.snapshot is not None
    assert recovered.snapshot["sequence"] == 1
    assert len(events) == 1


def test_latest_projection_prefers_active_operation_over_newer_terminal(
    tmp_path: Path,
) -> None:
    store = RuntimeOperationStore(tmp_path)
    store.start(
        "active-repair",
        {"operation": "repair", "profile_id": "win-x64-cpu"},
        initial_snapshot={
            **_snapshot("active-repair", 1),
            "updated_at": "2026-08-05T12:00:00Z",
        },
    )
    store.start(
        "later-inspect",
        {"operation": "inspect", "profile_id": "win-x64-cpu"},
        initial_snapshot={
            **_snapshot("later-inspect", 1),
            "operation": "inspect",
            "updated_at": "2026-08-05T12:01:00Z",
        },
    )
    store.append(
        "later-inspect",
        event_type="snapshot",
        snapshot={
            **_snapshot("later-inspect", 2, "succeeded"),
            "operation": "inspect",
            "updated_at": "2026-08-05T12:02:00Z",
        },
        message_code="runtime.inspect_complete",
    )

    projection = store.latest_projection()

    assert projection is not None
    assert projection["operation_id"] == "active-repair"
    assert projection["operation_state"] == "running"


def test_cancel_intent_recovers_from_event_before_metadata_replace_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RuntimeOperationStore(tmp_path)
    store.start("op-1", {"operation": "repair", "profile_id": "win-x64-cpu"})
    original_atomic_json = runtime_maintenance._atomic_json

    def crash_cancel_projection(path: Path, value: dict) -> None:
        if path.name == "metadata.json" and value.get("cancel_requested") is True:
            raise OSError("simulated crash after durable cancel event")
        original_atomic_json(path, value)

    monkeypatch.setattr(
        "vibeocr.backend.runtime_maintenance._atomic_json",
        crash_cancel_projection,
    )
    with pytest.raises(OSError, match="durable cancel event"):
        store.request_cancel("op-1")

    monkeypatch.undo()
    restarted = RuntimeOperationStore(tmp_path)
    assert restarted.cancel_requested("op-1") is True
    snapshot = restarted.request_cancel("op-1")
    assert snapshot["sequence"] == 2
    update = restarted.observe("op-1", after_sequence=0, limit=10)
    assert [event["message_code"] for event in update["events"]].count(
        "runtime.cancel_requested"
    ) == 1
    assert all("_projection" not in event for event in update["events"])


def test_failure_taxonomy_recovers_from_event_before_metadata_replace_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RuntimeOperationStore(tmp_path)
    store.start("op-1", {"operation": "repair", "profile_id": "win-x64-cpu"})
    failure = runtime_maintenance._command_error(RuntimeLockTimeout("busy again"))
    original_atomic_json = runtime_maintenance._atomic_json

    def crash_failure_projection(path: Path, value: dict) -> None:
        if path.name == "metadata.json" and isinstance(value.get("failure"), dict):
            raise OSError("simulated crash after durable failed event")
        original_atomic_json(path, value)

    monkeypatch.setattr(
        "vibeocr.backend.runtime_maintenance._atomic_json",
        crash_failure_projection,
    )
    with pytest.raises(OSError, match="durable failed event"):
        store.append(
            "op-1",
            event_type="snapshot",
            snapshot=_snapshot("op-1", 2, "failed"),
            message_code="runtime.operation_failed",
            failure=failure,
        )

    monkeypatch.undo()
    with pytest.raises(RuntimeLockTimeout, match="busy again"):
        RuntimeOperationStore(tmp_path).raise_replayed_failure("op-1")


def test_compaction_recovers_after_journal_replace_before_metadata_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime(2026, 8, 5, tzinfo=UTC)
    store = RuntimeOperationStore(tmp_path, clock=lambda: now)
    store.start("op-1", {"operation": "repair", "profile_id": "win-x64-cpu"})
    for sequence in range(2, 6):
        store.append(
            "op-1",
            event_type="snapshot",
            snapshot=_snapshot(
                "op-1", sequence, "succeeded" if sequence == 5 else "running"
            ),
            message_code="runtime.install.profile",
        )
    expired = RuntimeOperationStore(tmp_path, clock=lambda: now + timedelta(days=8))
    original_atomic_json = runtime_maintenance._atomic_json

    def crash_final_compaction_metadata(path: Path, value: dict) -> None:
        if (
            path.name == "metadata.json"
            and value.get("oldest_sequence") == 4
            and "compaction_pending" not in value
        ):
            raise OSError("simulated crash after compacted journal replace")
        original_atomic_json(path, value)

    monkeypatch.setattr(
        "vibeocr.backend.runtime_maintenance._atomic_json",
        crash_final_compaction_metadata,
    )
    with pytest.raises(OSError, match="compacted journal replace"):
        expired.compact("op-1", keep_last=2)

    monkeypatch.undo()
    restarted = RuntimeOperationStore(tmp_path, clock=lambda: now + timedelta(days=8))
    update = restarted.observe("op-1", after_sequence=3, limit=10)
    assert update["oldest_sequence"] == 4
    assert [event["sequence"] for event in update["events"]] == [4, 5]


def test_command_id_applies_side_effect_once_and_conflicts_on_reuse(
    tmp_path: Path,
) -> None:
    store = RuntimeOperationStore(tmp_path)
    calls: list[str] = []

    def apply() -> dict:
        calls.append("called")
        return {"operation_id": "op-2"}

    payload = {
        "command": "retry",
        "target_operation_id": "op-1",
        "new_operation_id": "op-2",
    }
    assert store.apply_command("cmd-1", payload, apply) == {"operation_id": "op-2"}
    assert store.apply_command("cmd-1", {**payload}, apply) == {"operation_id": "op-2"}
    assert calls == ["called"]
    with pytest.raises(RuntimeCommandConflict):
        store.apply_command("cmd-1", {**payload, "new_operation_id": "op-3"}, apply)


def test_command_reservation_write_crash_leaves_id_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RuntimeOperationStore(tmp_path)
    payload = {"command": "cancel", "target_operation_id": "op-1"}
    calls: list[str] = []

    def apply() -> dict:
        calls.append("applied")
        return {"status": "ok"}

    original_atomic_json = runtime_maintenance._atomic_json

    def crash_reservation(path: Path, value: dict) -> None:
        if path.name == "command.json" and value.get("outcome") is None:
            raise OSError("simulated reservation write crash")
        original_atomic_json(path, value)

    monkeypatch.setattr(
        "vibeocr.backend.runtime_maintenance._atomic_json", crash_reservation
    )
    with pytest.raises(OSError, match="reservation write crash"):
        store.apply_command("cmd-1", payload, apply)

    monkeypatch.undo()
    assert store.apply_command("cmd-1", payload, apply) == {"status": "ok"}
    assert calls == ["applied"]


def test_failed_command_id_remains_bound_to_original_payload(tmp_path: Path) -> None:
    store = RuntimeOperationStore(tmp_path)
    store.start("op-1", {"operation": "repair", "profile_id": "win-x64-cpu"})
    store.append(
        "op-1",
        event_type="snapshot",
        snapshot=_snapshot("op-1", 2),
        message_code="runtime.install.profile",
    )
    control = object.__new__(RuntimeControl)
    control._store = store

    for _ in range(2):
        with pytest.raises(RuntimeOperationNotRetryable):
            control.command(
                command_id="cmd-failed",
                command="retry",
                target_operation_id="op-1",
                new_operation_id="op-2",
            )
    with pytest.raises(RuntimeCommandConflict):
        control.command(
            command_id="cmd-failed",
            command="cancel",
            target_operation_id="op-1",
        )


def test_failed_command_replays_identity_error_taxonomy(tmp_path: Path) -> None:
    store = RuntimeOperationStore(tmp_path)
    calls: list[str] = []

    def fail() -> dict:
        calls.append("called")
        raise RuntimeSourceIdentityMismatch("source changed")

    for _ in range(2):
        with pytest.raises(RuntimeSourceIdentityMismatch, match="source changed"):
            store.apply_command("cmd-identity", {"command": "retry"}, fail)

    assert calls == ["called"]


@pytest.mark.parametrize(
    "exception",
    [
        RuntimeLockTimeout("busy"),
        OSError("disk"),
        RuntimeCapabilityError("missing capability"),
        RuntimeInstallFailure("install failed"),
    ],
)
def test_failed_command_replays_transport_error_taxonomy(
    tmp_path: Path, exception: Exception
) -> None:
    store = RuntimeOperationStore(tmp_path)
    calls: list[str] = []

    def fail() -> dict:
        calls.append("called")
        raise exception

    with pytest.raises(type(exception), match=str(exception)):
        store.apply_command("cmd-taxonomy", {"command": "retry"}, fail)
    with pytest.raises(type(exception), match=str(exception)):
        store.apply_command("cmd-taxonomy", {"command": "retry"}, fail)

    assert calls == ["called"]


def test_cross_instance_cancel_and_progress_keep_unique_sequence(
    tmp_path: Path,
) -> None:
    first = RuntimeOperationStore(tmp_path)
    second = RuntimeOperationStore(tmp_path)
    first.start("op-1", {"operation": "repair", "profile_id": "win-x64-cpu"})

    def append_progress() -> None:
        try:
            first.append(
                "op-1",
                event_type="progress",
                snapshot=_snapshot("op-1", 2),
                message_code="runtime.install.profile",
            )
        except RuntimeError:
            pass

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(append_progress),
            executor.submit(second.request_cancel, "op-1"),
        ]
        for future in futures:
            future.result()

    update = first.observe("op-1", after_sequence=0, limit=10)
    sequences = [event["sequence"] for event in update["events"]]
    assert sequences == list(range(1, len(sequences) + 1))
    assert len(sequences) == len(set(sequences))


def test_sequence_cursor_replays_after_restart_and_reports_expiry(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 5, tzinfo=UTC)
    store = RuntimeOperationStore(tmp_path, clock=lambda: now)
    store.start("op-1", {"operation": "repair", "profile_id": "win-x64-cpu"})
    for sequence in range(2, 6):
        store.append(
            "op-1",
            event_type="snapshot",
            snapshot=_snapshot(
                "op-1", sequence, "succeeded" if sequence == 5 else "running"
            ),
            message_code="runtime.install.profile",
        )

    restarted = RuntimeOperationStore(tmp_path, clock=lambda: now)
    update = restarted.observe("op-1", after_sequence=2, limit=1)
    assert [event["snapshot"]["sequence"] for event in update["events"]] == [3]
    assert update["through_sequence"] == 3
    assert update["more"] is True

    with pytest.raises(RuntimeError, match="retention has not expired"):
        restarted.compact("op-1", keep_last=2)
    expired = RuntimeOperationStore(tmp_path, clock=lambda: now + timedelta(days=8))
    expired.compact("op-1", keep_last=2)
    with pytest.raises(RuntimeCursorExpired) as raised:
        expired.observe("op-1", after_sequence=0, limit=10)
    assert raised.value.oldest_sequence == 4
    assert raised.value.snapshot["sequence"] == 5


def test_cancel_request_is_idempotent_and_ordered_before_terminal(
    tmp_path: Path,
) -> None:
    store = RuntimeOperationStore(tmp_path)
    store.start("op-1", {"operation": "repair", "profile_id": "win-x64-cpu"})
    first = store.request_cancel("op-1")
    second = store.request_cancel("op-1")

    assert first == second
    assert first["sequence"] == 2
    update = store.observe("op-1", after_sequence=0, limit=10)
    assert [event["message_code"] for event in update["events"]] == [
        "runtime.validate_binding",
        "runtime.cancel_requested",
    ]


def test_cancel_expected_sequence_is_checked_inside_operation_lock(
    tmp_path: Path,
) -> None:
    store = RuntimeOperationStore(tmp_path)
    store.start("op-1", {"operation": "repair", "profile_id": "win-x64-cpu"})
    store.append(
        "op-1",
        event_type="progress",
        snapshot=_snapshot("op-1", 2),
        message_code="runtime.install.profile",
    )

    with pytest.raises(RuntimeOperationConflict, match="expected_sequence"):
        store.request_cancel("op-1", expected_sequence=1)

    assert store.cancel_requested("op-1") is False


def test_pending_cancel_command_recovers_idempotently_after_outcome_write_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RuntimeOperationStore(tmp_path)
    store.start("op-1", {"operation": "repair", "profile_id": "win-x64-cpu"})
    payload = {"command": "cancel", "target_operation_id": "op-1"}

    def apply() -> dict:
        return store.request_cancel("op-1", expected_sequence=1)

    original_atomic_json = runtime_maintenance._atomic_json

    def crash_outcome(path: Path, value: dict) -> None:
        if path.name == "command.json" and value.get("outcome") is not None:
            raise OSError("simulated process crash before command outcome persistence")
        original_atomic_json(path, value)

    monkeypatch.setattr(
        "vibeocr.backend.runtime_maintenance._atomic_json", crash_outcome
    )
    with pytest.raises(OSError, match="simulated process crash"):
        store.apply_command("cmd-cancel", payload, apply)

    monkeypatch.undo()
    recovered = store.apply_command("cmd-cancel", payload, apply)
    assert recovered["sequence"] == 2
    assert store.observe("op-1", after_sequence=0, limit=10)["through_sequence"] == 2


def test_pending_retry_with_running_operation_replays_retryable_busy_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_identity = {"backend_source_sha": "a" * 40}
    intent = {
        "operation": "repair",
        "profile_id": "win-x64-cpu",
        "component_ids": [],
        "required_capabilities": [],
        "source_identity": source_identity,
    }
    store = RuntimeOperationStore(tmp_path)
    store.start("op-1", intent)
    store.append(
        "op-1",
        event_type="snapshot",
        snapshot=_snapshot("op-1", 2, "failed"),
        message_code="runtime.failed",
    )
    control = object.__new__(RuntimeControl)
    control._store = store
    control._installer = lambda: type("Installer", (), {"manifest": object()})()  # type: ignore[method-assign]
    monkeypatch.setattr(
        "vibeocr.backend.runtime_control.runtime_source_identity",
        lambda _manifest: source_identity,
    )
    calls: list[str] = []

    def execute(**_kwargs) -> dict:  # type: ignore[no-untyped-def]
        calls.append("called")
        retry_intent = {**intent, "source_operation_id": "op-1"}
        started = store.start("op-2", retry_intent, source_operation_id="op-1")
        assert started.snapshot is not None
        return {
            "schema_version": 2,
            "operation_id": "op-2",
            "snapshot": started.snapshot,
            "negotiated_capabilities": [],
        }

    control.execute = execute  # type: ignore[method-assign]
    original_atomic_json = runtime_maintenance._atomic_json

    def crash_command_outcome(path: Path, value: dict) -> None:
        if path.name == "command.json" and value.get("outcome") is not None:
            raise OSError("simulated retry outcome crash")
        original_atomic_json(path, value)

    monkeypatch.setattr(
        "vibeocr.backend.runtime_maintenance._atomic_json", crash_command_outcome
    )
    with pytest.raises(OSError, match="retry outcome crash"):
        control.command(
            command_id="retry-command",
            command="retry",
            target_operation_id="op-1",
            new_operation_id="op-2",
        )
    monkeypatch.setattr(
        "vibeocr.backend.runtime_maintenance._atomic_json", original_atomic_json
    )

    for _ in range(2):
        with pytest.raises(RuntimeLockTimeout, match="interrupted or is still running"):
            control.command(
                command_id="retry-command",
                command="retry",
                target_operation_id="op-1",
                new_operation_id="op-2",
            )

    assert calls == ["called"]


def _retry_control_with_failed_source_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, intent: dict
) -> tuple[RuntimeControl, list[dict]]:
    source_identity = {"backend_source_sha": "a" * 40}
    intent.setdefault("source_identity", source_identity)
    store = RuntimeOperationStore(tmp_path)
    store.start("op-1", intent)
    store.append(
        "op-1",
        event_type="snapshot",
        snapshot=_snapshot("op-1", 2, "failed"),
        message_code="runtime.failed",
    )
    control = object.__new__(RuntimeControl)
    control._store = store
    control._installer = lambda: type("Installer", (), {"manifest": object()})()  # type: ignore[method-assign]
    monkeypatch.setattr(
        "vibeocr.backend.runtime_control.runtime_source_identity",
        lambda _manifest: source_identity,
    )
    calls: list[dict] = []

    def execute(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return {"operation_id": kwargs["operation_id"]}

    control.execute = execute  # type: ignore[method-assign]
    return control, calls


def test_retry_reuses_selection_intent_when_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control, calls = _retry_control_with_failed_source_operation(
        tmp_path,
        monkeypatch,
        {
            "operation": "ensure",
            "profile_id": "win-x64-cpu",
            "component_ids": [],
            "required_capabilities": [],
            # source operation 是显式 base-only + 已选源：retry 省略时复用。
            "install_component_ids": [],
            "download_source_ids": ["pypi"],
        },
    )
    assert control.command(
        command_id="retry-1",
        command="retry",
        target_operation_id="op-1",
        new_operation_id="op-2",
    ) == {"operation_id": "op-2"}
    assert calls[0]["install_component_ids"] == ()
    assert calls[0]["download_source_ids"] == ("pypi",)


def test_retry_explicit_selection_replaces_source_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control, calls = _retry_control_with_failed_source_operation(
        tmp_path,
        monkeypatch,
        {
            "operation": "ensure",
            "profile_id": "win-x64-cpu",
            "component_ids": [],
            "required_capabilities": [],
            "install_component_ids": [],
            "download_source_ids": ["pypi"],
        },
    )
    assert control.command(
        command_id="retry-2",
        command="retry",
        target_operation_id="op-1",
        new_operation_id="op-3",
        install_component_ids=("document_parsing",),
    ) == {"operation_id": "op-3"}
    assert calls[0]["install_component_ids"] == ("document_parsing",)
    # 显式只给 component 时源意图仍复用 source operation。
    assert calls[0]["download_source_ids"] == ("pypi",)


@pytest.mark.parametrize(
    ("first_selection", "changed_selection"),
    [
        ({}, {"install_component_ids": ()}),
        (
            {"install_component_ids": ()},
            {"install_component_ids": ("document_parsing",)},
        ),
        ({}, {"download_source_ids": ("pypi",)}),
    ],
)
def test_retry_command_identity_includes_selection_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_selection: dict,
    changed_selection: dict,
) -> None:
    control, calls = _retry_control_with_failed_source_operation(
        tmp_path,
        monkeypatch,
        {
            "operation": "ensure",
            "profile_id": "win-x64-cpu",
            "component_ids": [],
            "required_capabilities": [],
            "download_source_ids": ["pypi"],
        },
    )
    command = {
        "command_id": "retry-selection",
        "command": "retry",
        "target_operation_id": "op-1",
        "new_operation_id": "op-2",
    }

    first = control.command(**command, **first_selection)
    assert control.command(**command, **first_selection) == first
    assert len(calls) == 1

    with pytest.raises(RuntimeCommandConflict):
        control.command(**command, **changed_selection)


def test_retry_command_identity_ignores_unordered_selection_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control, calls = _retry_control_with_failed_source_operation(
        tmp_path,
        monkeypatch,
        {
            "operation": "ensure",
            "profile_id": "win-x64-cu126",
            "component_ids": [],
            "required_capabilities": [],
            "download_source_ids": ["pypi"],
        },
    )
    command = {
        "command_id": "retry-unordered-selection",
        "command": "retry",
        "target_operation_id": "op-1",
        "new_operation_id": "op-2",
    }

    first = control.command(
        **command,
        install_component_ids=("gpu_runtime", "document_parsing"),
    )
    assert (
        control.command(
            **command,
            install_component_ids=("document_parsing", "gpu_runtime"),
        )
        == first
    )
    assert len(calls) == 1


def test_selection_fields_are_rejected_outside_ensure_and_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control, _ = _retry_control_with_failed_source_operation(
        tmp_path,
        monkeypatch,
        {"operation": "inspect", "profile_id": "win-x64-cpu"},
    )
    with pytest.raises(ValueError, match="only valid for operation ensure"):
        control.execute_with_result(
            operation="inspect",
            install_component_ids=(),
            download_source_ids=("pypi",),
        )
    with pytest.raises(ValueError, match="only valid for command retry"):
        control.command(
            command_id="cancel-1",
            command="cancel",
            target_operation_id="op-1",
            install_component_ids=(),
        )
