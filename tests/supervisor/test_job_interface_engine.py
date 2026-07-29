"""Execution invariants for the supervisor-only deep job interface."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pytest

from vibeocr.backend.supervisor.inference.budgets import AdapterCapability
from vibeocr.backend.supervisor.inference.paddle_executor import AdapterExecutor
from vibeocr.backend.supervisor.jobs.staging import InputExpiredError
from vibeocr.backend.supervisor.module import SupervisorModule, SupervisorOptions
from vibeocr.runtime_contracts import (
    JobKind,
    JobPriority,
    JobState,
    PipelineSelection,
    SubmitItem,
    SubmitRequest,
)

if TYPE_CHECKING:
    from pathlib import Path


class _ScriptedAdapter:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[Any], Any]] = []

    def recognize_many(self, items, *, options=None, compute_batch=None):
        self.calls.append((list(items), options))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def capabilities(self, options=None):
        del options
        return AdapterCapability(
            name="scripted", real_batch=True, max_compute_batch=64
        )

    def residency_status(self):
        from vibeocr.runtime_contracts import ResidencyStatus

        return ResidencyStatus()

    def release_idle(self, pipeline=None):
        return self.residency_status()


def _request(count: int = 2) -> SubmitRequest:
    return SubmitRequest(
        request_id="req-1",
        kind=JobKind.RECOGNITION,
        priority=JobPriority.BACKGROUND,
        pipeline=PipelineSelection(
            "OCR", options={"use_doc_orientation_classify": False}
        ),
        items=tuple(
            SubmitItem(
                client_item_key=f"file-{index}",
                ordinal=index,
                display_name=f"{index}.png",
                source={"type": "upload.v1", "attachment": f"file-{index}"},
            )
            for index in range(count)
        ),
    )


def _module(
    tmp_path: Path, adapter: _ScriptedAdapter, *, max_per_file_bytes: int = 1024
) -> SupervisorModule:
    return SupervisorModule(
        options=SupervisorOptions(
            instance_id="sup-test", max_per_file_bytes=max_per_file_bytes
        ),
        stager_root=tmp_path,
        executor=AdapterExecutor(adapter_factory=lambda: adapter),
    )


def _terminal(module: SupervisorModule, job_id: str):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        snapshot = module.status(job_id)
        if snapshot.state in {
            JobState.COMPLETED,
            JobState.COMPLETED_WITH_ERRORS,
            JobState.CANCELLED,
            JobState.FAILED,
        }:
            return snapshot
        time.sleep(0.005)
    raise AssertionError("job did not reach terminal state")


def test_submit_observe_preserves_manifest_and_returns_keyed_outcomes(tmp_path) -> None:
    adapter = _ScriptedAdapter(
        [[{"raw_text": "A"}, {"raw_text": "B"}]]
    )
    module = _module(tmp_path, adapter)
    request = _request()

    ref = module.submit_request(
        request,
        {
            "file-0": ("image/png", b"a"),
            "file-1": ("image/png", b"b"),
        },
    )
    snapshot = _terminal(module, ref.job_id)
    update = module.observe(ref.job_id)

    assert snapshot.priority is JobPriority.BACKGROUND
    assert snapshot.request_id == "req-1"
    assert snapshot.pipeline == request.pipeline
    assert [item.client_item_key for item in snapshot.items] == ["file-0", "file-1"]
    assert [outcome.item_id for outcome in update.outcomes] == [
        item.item_id for item in snapshot.items
    ]
    payloads = [outcome.payload for outcome in update.outcomes]
    assert all(payload is not None for payload in payloads)
    assert [payload["raw_text"] for payload in payloads if payload is not None] == [
        "A",
        "B",
    ]
    assert update.through_sequence == update.snapshot.event_sequence
    assert adapter.calls[0][1] == request.pipeline


def test_staging_failure_cannot_shift_the_next_items_result(tmp_path) -> None:
    adapter = _ScriptedAdapter([[{"raw_text": "SECOND"}]])
    module = _module(tmp_path, adapter, max_per_file_bytes=3)

    ref = module.submit_request(
        _request(),
        {
            "file-0": ("image/png", b"too-large"),
            "file-1": ("image/png", b"ok"),
        },
    )
    snapshot = _terminal(module, ref.job_id)
    results = module.result(ref.job_id)

    assert snapshot.state is JobState.COMPLETED_WITH_ERRORS
    assert snapshot.summary.failed == 1
    assert snapshot.summary.succeeded == 1
    assert results[0].error_code == "QUOTA_EXCEEDED"
    assert results[1].payload == {"raw_text": "SECOND"}


def test_short_adapter_result_fails_items_instead_of_faking_empty_success(
    tmp_path,
) -> None:
    adapter = _ScriptedAdapter([[]])
    module = _module(tmp_path, adapter)

    ref = module.submit_request(
        _request(),
        {
            "file-0": ("image/png", b"a"),
            "file-1": ("image/png", b"b"),
        },
    )
    snapshot = _terminal(module, ref.job_id)
    results = module.result(ref.job_id)

    assert snapshot.state is JobState.FAILED
    assert snapshot.summary.failed == 2
    assert all(
        result.error_code == "ADAPTER_PROTOCOL_VIOLATION" for result in results
    )
    assert all(result.payload == {} for result in results)


def test_retry_clones_real_failed_input_and_links_item_identity(tmp_path) -> None:
    adapter = _ScriptedAdapter(
        [RuntimeError("deterministic inference failure"), [{"raw_text": "retried"}]]
    )
    module = _module(tmp_path, adapter)

    first = module.submit_request(
        _request(1),
        {"file-0": ("image/png", b"original-bytes")},
    )
    assert _terminal(module, first.job_id).state is JobState.FAILED

    retry = module.retry(first.job_id)
    retry_snapshot = _terminal(module, retry.job_id)

    assert retry_snapshot.state is JobState.COMPLETED
    assert retry_snapshot.source_job_id == first.job_id
    assert retry_snapshot.items[0].source_item_id == first.items[0].item_id
    assert retry_snapshot.items[0].attempt == 1
    assert adapter.calls[1][0][0].data == b"original-bytes"
    assert module.result(retry.job_id)[0].payload == {"raw_text": "retried"}


def test_retry_rejects_failed_item_without_retained_input(tmp_path) -> None:
    adapter = _ScriptedAdapter([[]])
    module = _module(tmp_path, adapter, max_per_file_bytes=1)
    first = module.submit_request(
        _request(1),
        {"file-0": ("image/png", b"too-large")},
    )
    _terminal(module, first.job_id)

    with pytest.raises(InputExpiredError, match="expired or unavailable"):
        module.retry(first.job_id)
