"""Tests for the remaining branches of MinerUProcessAdapter.

Existing test_mineru_adapter.py covers the happy path; this file targets the
gaps: the ``_policy_locked`` loop skipping non-MinerU specs, the preload
short-circuit when ``MinerU`` is not requested, and the preload failure
log-and-re-raise path.
"""

from __future__ import annotations

import logging

import pytest
from vibeocr.backend.supervisor.inference.budgets import InputItem
from vibeocr.backend.supervisor.inference.mineru_adapter import MinerUProcessAdapter
from vibeocr.runtime_contracts import PipelineSpec, SettingsSnapshot


def _raw_item(item_id: str, display_name: str, data: bytes) -> InputItem:
    return InputItem(
        item_id=item_id,
        display_name=display_name,
        data=data,
        encoded_bytes=len(data),
        decoded_pixels=0,
        estimated_pages=1,
    )


class _FakeMinerUClient:
    def __init__(self, *, raises: bool = False) -> None:
        self.raises = raises

    def file_parse(self, files, backend=None, **kwargs):  # type: ignore[no-untyped-def]
        if self.raises:
            raise RuntimeError("file_parse boom")
        return {name: {"markdown": "md"} for name, _ in files}


class _BoomLifecycle:
    """Lifecycle whose start() raises so preload's failure path runs."""

    def __init__(self) -> None:
        self.started = False

    def start(self) -> None:
        self.started = True
        raise RuntimeError("mineru start failed")

    def stop(self) -> None:
        return


# ---------------------------------------------------------------------------
# _policy_locked: skip non-MinerU specs
# ---------------------------------------------------------------------------


def test_policy_skips_non_mineru_pipeline_specs() -> None:
    """A settings snapshot with non-MinerU pipeline specs only affects policy
    via the MinerU entry; non-MinerU entries are skipped (line 163)."""
    adapter = MinerUProcessAdapter(
        client_factory=lambda: _FakeMinerUClient(),
        lifecycle=_BoomLifecycle(),
    )
    adapter.configure_settings(
        SettingsSnapshot(
            default_ttl_seconds=200,
            pipelines=(
                PipelineSpec(name="OCR", ttl_seconds=999),  # not MinerU → skipped
                PipelineSpec(name="MinerU", ttl_seconds=400),  # used
            ),
        )
    )
    pinned, ttl = adapter._policy_locked()
    assert pinned is False
    assert ttl == 400


def test_policy_uses_default_ttl_when_no_mineru_spec() -> None:
    """When no MinerU spec exists, the default_ttl_seconds is used."""
    adapter = MinerUProcessAdapter(client_factory=lambda: _FakeMinerUClient())
    adapter.configure_settings(SettingsSnapshot(default_ttl_seconds=350))
    pinned, ttl = adapter._policy_locked()
    assert pinned is False
    assert ttl == 350


# ---------------------------------------------------------------------------
# preload: short-circuit when MinerU not requested
# ---------------------------------------------------------------------------


def test_preload_returns_status_when_mineru_not_requested() -> None:
    """A preload request that does not include ``MinerU`` returns the current
    status without touching the lifecycle (line 273)."""
    lifecycle = _BoomLifecycle()  # start() would raise
    adapter = MinerUProcessAdapter(
        client_factory=lambda: _FakeMinerUClient(),
        lifecycle=lifecycle,
    )
    # Should NOT raise despite the boom lifecycle, because we never call start.
    status = adapter.preload(("OCR",))
    assert status.entries[0].pipeline == "MinerU"
    assert lifecycle.started is False


# ---------------------------------------------------------------------------
# preload: failure path logs and re-raises
# ---------------------------------------------------------------------------


def test_preload_logs_and_reraises_when_lifecycle_start_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When preload's ensure_started raises, the failure is logged and re-raised
    (lines 278-284)."""
    adapter = MinerUProcessAdapter(
        client_factory=lambda: _FakeMinerUClient(),
        lifecycle=_BoomLifecycle(),
    )
    with (
        caplog.at_level(
            logging.ERROR,
            logger="vibeocr.backend.supervisor.inference.mineru_adapter",
        ),
        pytest.raises(RuntimeError, match="mineru start failed"),
    ):
        adapter.preload(("MinerU",))

    assert any(
        "[Supervisor][Preload]" in record.getMessage()
        and "result=failed" in record.getMessage()
        for record in caplog.records
    )


def test_mineru_executor_property_returns_adapter() -> None:
    """MinerUExecutor.adapter returns the materialised adapter (line 33)."""
    from vibeocr.backend.supervisor.inference.mineru_executor import MinerUExecutor

    inner_adapter = MinerUProcessAdapter(client_factory=lambda: _FakeMinerUClient())
    executor = MinerUExecutor(adapter_factory=lambda: inner_adapter)
    assert executor.adapter is inner_adapter


def test_executor_fails_closed_per_item_on_unsupported_type(tmp_path) -> None:
    """未知类型只失败对应 item，其余输入照常上传（Backend #116）。

    display_name 无后缀且无可信 MIME 的输入在 adapter 边界显式拒绝
    （VALIDATION_ERROR），不猜测类型，也不影响同批其它文件。
    """
    from vibeocr.backend.supervisor.inference.mineru_executor import MinerUExecutor
    from vibeocr.backend.supervisor.jobs.registry import JobRegistry
    from vibeocr.backend.supervisor.jobs.staging import StagedInput
    from vibeocr.runtime_contracts import (
        ItemState,
        JobItem,
        JobKind,
        JobPriority,
        JobState,
    )

    class _RecordingClient:
        def __init__(self) -> None:
            self.uploads: list[list[str]] = []

        def file_parse(self, files, backend=None, **kwargs):  # type: ignore[no-untyped-def]
            self.uploads.append([name for name, _ in files])
            return {name: {"markdown": "md"} for name, _ in files}

    client = _RecordingClient()
    adapter = MinerUProcessAdapter(client_factory=lambda: client)
    executor = MinerUExecutor(adapter_factory=lambda: adapter)

    good = tmp_path / "0000-a.pdf-00000001"
    good.write_bytes(b"%PDF-a")
    bad = tmp_path / "0001-clipboard-00000002"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n")
    staged = [
        StagedInput(
            item_id="it-0",
            display_name="a.pdf",
            path=good,
            size_bytes=6,
            content_type="application/pdf",
        ),
        StagedInput(
            item_id="it-1",
            display_name="clipboard",
            path=bad,
            size_bytes=8,
            content_type=None,
        ),
    ]
    registry = JobRegistry(instance_id="t")
    record = registry.create(
        kind=JobKind.MINERU_PARSE,
        priority=JobPriority.INTERACTIVE,
        items=[
            JobItem(item_id="it-0", display_name="a.pdf", state=ItemState.QUEUED),
            JobItem(item_id="it-1", display_name="clipboard", state=ItemState.QUEUED),
        ],
        progress_total=2,
    )
    record.transition(JobState.QUEUED)

    executor.execute(record, staged)

    snap = record.snapshot()
    states = {item.item_id: item.state for item in snap.items}
    assert states["it-0"] is ItemState.SUCCEEDED
    assert states["it-1"] is ItemState.FAILED
    assert record.item_errors["it-1"] == "VALIDATION_ERROR"
    # 仅受支持文件到达上传缝；无后缀名从未被上传。
    assert len(client.uploads) == 1
    assert len(client.uploads[0]) == 1
    assert client.uploads[0][0].endswith(".pdf")
