"""Tests for MinerUProcessAdapter: unique stems, multi-file order, lifecycle."""

from __future__ import annotations

import logging
import threading
import time

import pytest

from vibeocr.backend.supervisor.inference.budgets import InputItem
from vibeocr.backend.supervisor.inference.mineru_adapter import (
    MinerUProcessAdapter,
    unique_stem,
)


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
    """Returns a dict keyed by the uploaded filename."""

    def __init__(self) -> None:
        self.received: list[list[tuple[str, bytes]]] = []

    def file_parse(self, files, backend=None, **kwargs):
        self.received.append(files)
        # Echo one result per file, keyed by the unique stem.
        return {name: {"markdown": f"md-{i}"} for i, (name, _data) in enumerate(files)}


# ---------------------------------------------------------------------------
# unique_stem
# ---------------------------------------------------------------------------


def test_unique_stem_disambiguates_duplicates() -> None:
    a = unique_stem("copy.pdf", 0)
    b = unique_stem("copy.pdf", 1)
    assert a != b
    assert a.endswith(".pdf")
    assert b.endswith(".pdf")


def test_unique_stem_strips_unsafe_chars() -> None:
    stem = unique_stem("..\\evil<> name.pdf", 0)
    assert "\\" not in stem
    assert "<" not in stem
    assert ">" not in stem
    assert " " not in stem


# ---------------------------------------------------------------------------
# recognize_many
# ---------------------------------------------------------------------------


def test_recognize_many_single_request_multi_file() -> None:
    fake = _FakeMinerUClient()
    adapter = MinerUProcessAdapter(client_factory=lambda: fake)
    items = [
        _raw_item("it-0", "a.pdf", b"%PDF-a"),
        _raw_item("it-1", "b.pdf", b"%PDF-b"),
    ]
    results = adapter.recognize_many(items)
    # Exactly one /file_parse call with both files.
    assert len(fake.received) == 1
    assert len(fake.received[0]) == 2
    assert len(results) == 2


def test_mineru_logs_preload_and_recognition_model_with_elapsed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = _FakeMinerUClient()
    adapter = MinerUProcessAdapter(client_factory=lambda: fake)
    items = [_raw_item("it-0", "a.pdf", b"%PDF-a")]

    with caplog.at_level(
        logging.INFO,
        logger="vibeocr.backend.supervisor.inference.mineru_adapter",
    ):
        adapter.preload(("MinerU",))
        adapter.recognize_many(items)

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "[Supervisor][Preload]" in message
        and "pipeline=MinerU" in message
        and "result=success" in message
        and "elapsed_ms=" in message
        for message in messages
    )
    assert any(
        "[Supervisor][Recognize]" in message
        and "pipeline=MinerU" in message
        and "items=1" in message
        and "result=success" in message
        and "elapsed_ms=" in message
        for message in messages
    )
    adapter.close()


def test_recognize_many_restores_input_order() -> None:
    fake = _FakeMinerUClient()
    adapter = MinerUProcessAdapter(client_factory=lambda: fake)
    items = [
        _raw_item("it-0", "a.pdf", b"a"),
        _raw_item("it-1", "b.pdf", b"b"),
        _raw_item("it-2", "c.pdf", b"c"),
    ]
    results = adapter.recognize_many(items)
    # The fake returns a dict; the adapter maps stems back to indices 0,1,2.
    assert [r.get("markdown") for r in results] == ["md-0", "md-1", "md-2"]


def test_recognize_many_handles_duplicate_stems_without_collision() -> None:
    fake = _FakeMinerUClient()
    adapter = MinerUProcessAdapter(client_factory=lambda: fake)
    items = [
        _raw_item("it-0", "copy.pdf", b"0"),
        _raw_item("it-1", "copy.pdf", b"1"),
        _raw_item("it-2", "copy.pdf", b"2"),
    ]
    results = adapter.recognize_many(items)
    assert len(results) == 3
    # All three uploads had unique stems.
    names = [name for name, _ in fake.received[0]]
    assert len(set(names)) == 3


def test_recognize_many_raises_on_missing_raw_bytes() -> None:
    adapter = MinerUProcessAdapter(client_factory=lambda: _FakeMinerUClient())
    plain = InputItem(item_id="x", encoded_bytes=1, decoded_pixels=1, estimated_pages=1)
    with pytest.raises(ValueError, match="no raw bytes"):
        adapter.recognize_many([plain])


def test_recognize_many_empty_returns_empty() -> None:
    adapter = MinerUProcessAdapter(client_factory=lambda: _FakeMinerUClient())
    assert adapter.recognize_many([]) == []


def test_capability_does_not_promise_compute_batch() -> None:
    adapter = MinerUProcessAdapter(client_factory=lambda: _FakeMinerUClient())
    cap = adapter.capabilities()
    assert cap.real_batch is False
    assert cap.max_compute_batch == 1


def test_release_idle_stops_subprocess() -> None:
    adapter = MinerUProcessAdapter(client_factory=lambda: _FakeMinerUClient())
    adapter.ensure_started()
    status = adapter.residency_status()
    entry = next(e for e in status.entries if e.pipeline == "MinerU")
    from vibeocr.runtime_contracts import ResidencyKind

    assert entry.kind is ResidencyKind.SOFT_TTL
    adapter.release_idle()
    status2 = adapter.residency_status()
    entry2 = next(e for e in status2.entries if e.pipeline == "MinerU")
    assert entry2.kind is ResidencyKind.EVICTED


def test_residency_status_is_bounded_while_mineru_process_starts() -> None:
    class BlockingLifecycle:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()

        def start(self) -> None:
            self.entered.set()
            self.release.wait(timeout=1.0)

        def stop(self) -> None:
            return

    lifecycle = BlockingLifecycle()
    adapter = MinerUProcessAdapter(
        client_factory=lambda: _FakeMinerUClient(),
        lifecycle=lifecycle,
    )
    starter = threading.Thread(target=adapter.ensure_started, daemon=True)
    starter.start()
    assert lifecycle.entered.wait(timeout=0.5)

    started_at = time.monotonic()
    try:
        status = adapter.residency_status()
        assert time.monotonic() - started_at < 0.2
        assert status.entries[0].pipeline == "MinerU"
    finally:
        lifecycle.release.set()
        starter.join(timeout=0.5)
        adapter.close()


def test_finite_ttl_stops_idle_mineru_process() -> None:
    from vibeocr.runtime_contracts import (
        EvictionReason,
        PipelineSpec,
        ResidencyKind,
        SettingsSnapshot,
    )

    adapter = MinerUProcessAdapter(client_factory=lambda: _FakeMinerUClient())
    adapter.configure_settings(
        SettingsSnapshot(
            default_ttl_seconds=1,
            pipelines=(PipelineSpec(name="MinerU", ttl_seconds=1),),
        )
    )
    adapter.ensure_started()
    deadline = time.monotonic() + 2.5
    entry = adapter.residency_status().entries[0]
    while time.monotonic() < deadline:
        entry = adapter.residency_status().entries[0]
        if entry.kind is ResidencyKind.EVICTED:
            break
        time.sleep(0.05)
    assert entry.kind is ResidencyKind.EVICTED
    assert entry.eviction_reason is EvictionReason.TTL_EXPIRED
    adapter.close()


def test_pinned_mineru_is_not_released() -> None:
    from vibeocr.runtime_contracts import PipelineSpec, ResidencyKind, SettingsSnapshot

    adapter = MinerUProcessAdapter(client_factory=lambda: _FakeMinerUClient())
    adapter.configure_settings(
        SettingsSnapshot(
            default_ttl_seconds=1,
            pipelines=(PipelineSpec(name="MinerU", ttl_seconds=1, pinned=True),),
        )
    )
    adapter.ensure_started()
    adapter.release_idle()
    assert adapter.residency_status().entries[0].kind is ResidencyKind.PINNED
    adapter.close()


def test_positional_list_results_mapped_in_order() -> None:
    class _ListClient:
        def file_parse(self, files, backend=None, **kwargs):
            return [{"markdown": f"md-{i}"} for i in range(len(files))]

    adapter = MinerUProcessAdapter(client_factory=lambda: _ListClient())
    items = [
        _raw_item("it-0", "a.pdf", b"a"),
        _raw_item("it-1", "b.pdf", b"b"),
    ]
    results = adapter.recognize_many(items)
    assert [r["markdown"] for r in results] == ["md-0", "md-1"]
