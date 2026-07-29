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
