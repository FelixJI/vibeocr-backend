from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from vibeocr.backend import runtime_installer as installer
from vibeocr.backend.runtime_maintenance import (
    RuntimeMaintenanceReporter,
    RuntimeOperationCancelled,
    RuntimeOperationStore,
    RuntimeProfileDescriptor,
)
from vibeocr.runtime_contracts import parse_runtime_maintenance_event


def _reporter(root: Path, sink=None):
    def validated_sink(event):
        parse_runtime_maintenance_event(event)
        if sink is not None:
            sink(event)

    reporter = RuntimeMaintenanceReporter(
        state_root=root,
        profile=RuntimeProfileDescriptor("win-x64-cpu", "cpu", ()),
        event_sink=validated_sink,
    )
    reporter.start("ensure", total_steps=7, operation_id="download-test")
    return reporter


def _artifact(payload=b"wheel", url="https://example.invalid/a.whl"):
    digest = hashlib.sha256(payload).hexdigest()
    return installer._ResolvedArtifact("a", url, digest, "a.whl"), {"a": {digest}}


def _transport(monkeypatch, handler):
    monkeypatch.setattr(
        installer,
        "_download_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def test_slow_real_http_head_is_visible_and_cancellable(tmp_path, monkeypatch):
    events = []
    entered = threading.Event()
    release = threading.Event()
    observed = []
    store = RuntimeOperationStore(tmp_path / "state")

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            observed.append(
                next(
                    e["message_code"]
                    for e in reversed(events)
                    if e["event_type"] == "progress"
                )
            )
            entered.set()
            release.wait(3)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    def sink(event):
        events.append(event)
        if event["event_type"] == "heartbeat" and entered.is_set():
            store.request_cancel("download-test")

    monkeypatch.setattr(installer, "_CHILD_HEARTBEAT_INTERVAL_SECONDS", 0.1)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    reporter = _reporter(tmp_path / "state", sink)
    artifact, allowed = _artifact(url=f"http://127.0.0.1:{server.server_port}/a.whl")
    try:
        with pytest.raises(RuntimeOperationCancelled):
            installer._download_resolved_artifacts(
                (artifact,), allowed, tmp_path / "cache", reporter
            )
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(2)
    assert observed == ["runtime.probe_package_metadata"]
    heartbeats = [e for e in events if e["event_type"] == "heartbeat"]
    assert heartbeats and all(
        e["snapshot"]["progress"]["current"] == 0 for e in heartbeats
    )
    assert float(heartbeats[-1]["message_args"]["last_activity_seconds"]) > 0
    replay = RuntimeOperationStore(tmp_path / "state").observe(
        "download-test", after_sequence=0, limit=100
    )
    assert replay["events"][-1]["snapshot"]["operation_state"] == "cancelled"
    assert replay["events"][-1]["message_args"]["reason_code"] == "cancelled"


def test_parallel_download_checks_observe_one_terminal_cancellation(tmp_path):
    reporter = _reporter(tmp_path / "state")
    store = RuntimeOperationStore(tmp_path / "state")
    store.request_cancel("download-test")

    # The download task and its heartbeat supervisor may both see this request.
    with pytest.raises(RuntimeOperationCancelled):
        reporter.check_cancelled()
    with pytest.raises(RuntimeOperationCancelled):
        reporter.check_cancelled()

    replay = store.observe("download-test", after_sequence=0, limit=100)
    terminal = [
        event
        for event in replay["events"]
        if event["snapshot"]["operation_state"] == "cancelled"
    ]
    assert len(terminal) == 1
    assert replay["events"][-1]["snapshot"]["operation_state"] == "cancelled"


@pytest.mark.parametrize("failure", ["cancel", "disconnect"])
def test_interrupted_body_closes_stream_and_removes_only_own_partial(
    tmp_path, monkeypatch, failure
):
    closed = []
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "other.whl").write_bytes(b"keep")
    reporter = _reporter(tmp_path / "state")
    artifact, allowed = _artifact()
    store = RuntimeOperationStore(tmp_path / "state")

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"w"
            if failure == "cancel":
                store.request_cancel("download-test")
                await asyncio.sleep(10)
            else:
                raise httpx.ReadError("https://user:secret@example.invalid/private")

        async def aclose(self):
            closed.append(True)

    def respond(request):
        return (
            httpx.Response(200, stream=Body())
            if request.method == "GET"
            else httpx.Response(200)
        )

    _transport(monkeypatch, respond)
    expected = (
        RuntimeOperationCancelled
        if failure == "cancel"
        else installer.RuntimeInstallError
    )
    with pytest.raises(expected) as exc:
        installer._download_resolved_artifacts((artifact,), allowed, cache, reporter)
    if failure != "cancel":
        reporter.fail(exc.value)
    assert closed == [True]
    assert sorted(p.name for p in cache.iterdir()) == ["other.whl"]
    terminal = RuntimeOperationStore(tmp_path / "state").observe(
        "download-test", after_sequence=0, limit=100
    )["events"][-1]
    assert terminal["message_args"]["reason_code"] == (
        "cancelled" if failure == "cancel" else "network_error"
    )
    assert "secret" not in json.dumps(terminal)


@pytest.mark.parametrize("head_length", [None, "1", "-2", "broken"])
def test_unreliable_total_never_becomes_invented_percentage(
    tmp_path, monkeypatch, head_length
):
    payload = b"wheel"
    artifact, allowed = _artifact(payload)
    events = []
    reporter = _reporter(tmp_path / "state", events.append)

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield payload

    def respond(request):
        if request.method == "HEAD":
            return httpx.Response(
                200,
                headers={} if head_length is None else {"Content-Length": head_length},
            )
        return httpx.Response(200, stream=Body())

    _transport(monkeypatch, respond)
    installer._download_resolved_artifacts(
        (artifact,), allowed, tmp_path / "cache", reporter
    )
    assert events[-1]["snapshot"]["progress"] == {
        "unit": "bytes",
        "current": len(payload),
    }
    assert events[-1]["message_code"] == "runtime.download_verified"
    assert reporter.snapshot["operation_state"] == "running"


def test_bad_hash_never_reports_completed_and_retry_downloads_again(
    tmp_path, monkeypatch
):
    expected_payload = b"valid"
    artifact, allowed = _artifact(expected_payload)
    events = []
    reporter = _reporter(tmp_path / "state", events.append)
    payload = b"wrong"
    requests = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield payload

    def respond(request):
        if request.method == "GET":
            requests.append(request.url)
        return httpx.Response(
            200,
            headers={"Content-Length": "5"},
            stream=Body() if request.method == "GET" else None,
        )

    _transport(monkeypatch, respond)
    with pytest.raises(installer.RuntimeInstallError) as exc:
        installer._download_resolved_artifacts(
            (artifact,), allowed, tmp_path / "cache", reporter
        )
    reporter.fail(exc.value)
    assert events[-1]["message_args"]["reason_code"] == "hash_mismatch"
    assert not any(
        e.get("snapshot", {}).get("progress", {}).get("current") == 5 for e in events
    )
    assert not list((tmp_path / "cache").iterdir())
    payload = expected_payload
    installer._download_resolved_artifacts(
        (artifact, artifact), allowed, tmp_path / "cache", None
    )
    assert len(requests) == 2
    assert (tmp_path / "cache/a.whl").read_bytes() == expected_payload


@pytest.mark.parametrize(
    "error,reason",
    [
        (OSError(errno.ENOSPC, "no space"), "disk_full"),
        (PermissionError("denied"), "permission_denied"),
        (
            ValueError(
                "Authorization: Bearer hidden-token https://u:p@example.invalid/private C:/Users/private/file"
            ),
            "unknown",
        ),
    ],
)
def test_failure_guidance_and_safe_tail_survive_restart(tmp_path, error, reason):
    reporter = _reporter(tmp_path)
    reporter.fail(error)
    replay = RuntimeOperationStore(tmp_path).observe(
        "download-test", after_sequence=0, limit=100
    )
    terminal = replay["events"][-1]
    assert terminal["snapshot"]["operation_state"] == "failed"
    assert terminal["message_args"]["reason_code"] == reason
    assert terminal["message_args"]["next_action"]
    serialized = json.dumps(terminal)
    assert "hidden-token" not in serialized and "private" not in serialized


@pytest.mark.parametrize("order", [("paddle", "mineru"), ("mineru", "paddle")])
def test_shared_cache_mixed_batch_and_version_change_count_only_new_bytes(
    tmp_path, monkeypatch, order
):
    payloads = {"a.whl": b"shared", "b.whl": b"engine"}
    shared, allowed = _artifact(payloads["a.whl"])
    other = installer._ResolvedArtifact(
        "b",
        "https://example.invalid/b.whl",
        hashlib.sha256(payloads["b.whl"]).hexdigest(),
        "b.whl",
    )
    allowed["b"] = {other.sha256}
    requests = []

    class Body(httpx.AsyncByteStream):
        def __init__(self, data):
            self.data = data

        async def __aiter__(self):
            yield self.data

    def respond(request):
        data = payloads[request.url.path.rsplit("/", 1)[-1]]
        if request.method == "GET":
            requests.append(str(request.url))
        return httpx.Response(
            200,
            headers={"Content-Length": str(len(data))},
            stream=Body(data) if request.method == "GET" else None,
        )

    _transport(monkeypatch, respond)
    cache = tmp_path / "cache"
    installer._download_resolved_artifacts(
        (shared, replace(shared, url="https://mirror.invalid/a.whl")),
        allowed,
        cache,
        _reporter(tmp_path / order[0]),
    )
    assert len(requests) == 1
    events = []
    installer._download_resolved_artifacts(
        (shared, other), allowed, cache, _reporter(tmp_path / order[1], events.append)
    )
    assert len(requests) == 2
    hits = [e for e in events if e["message_code"] == "runtime.download_cache_hit"]
    assert len(hits) == 1 and hits[0]["snapshot"]["progress"]["current"] == 0
    assert events[-1]["snapshot"]["progress"] == {
        "unit": "bytes",
        "current": 6,
        "total": 6,
    }
    payloads["a.whl"] = b"new-version"
    updated = replace(shared, sha256=hashlib.sha256(payloads["a.whl"]).hexdigest())
    allowed["a"].add(updated.sha256)
    changed = []
    installer._download_resolved_artifacts(
        (updated,), allowed, cache, _reporter(tmp_path / "upgrade", changed.append)
    )
    assert len(requests) == 3
    assert any(e["message_code"] == "runtime.download_cache_invalid" for e in changed)
    assert changed[-1]["snapshot"]["progress"]["current"] == len(payloads["a.whl"])


def test_heartbeat_keeps_last_real_activity_and_progress(tmp_path, monkeypatch):
    from vibeocr.backend import runtime_maintenance

    tick = [100.0]
    monkeypatch.setattr(runtime_maintenance.time, "monotonic", lambda: tick[0])
    events = []
    reporter = _reporter(tmp_path, events.append)
    reporter.advance_measured(
        phase="install_profile",
        unit="bytes",
        current=3,
        total=10,
        message_code="runtime.download_package",
    )
    for value in (101.0, 103.0):
        tick[0] = value
        reporter.heartbeat(message_code="runtime.download_wait")
    assert [e["message_args"]["last_activity_seconds"] for e in events[-2:]] == [
        "1.000",
        "3.000",
    ]
    assert events[-1]["snapshot"]["progress"]["current"] == 3


@pytest.mark.parametrize(
    "detail",
    [
        '{"token": "synthetic-credential with spaces", "status": "failed"}',
        "{'password': 'synthetic-credential', 'status': 'failed'}",
        'Authorization: "Bearer synthetic-credential"',
        'secret="synthetic-credential without closing quote',
        "access_token=synthetic-credential",
        "HF_TOKEN=synthetic-credential",
        '{"client_secret": "synthetic-credential"}',
        'accessToken="synthetic-credential"',
        "X-API-Key: synthetic-credential",
    ],
)
def test_credential_key_families_are_redacted_before_durable_public_events(
    tmp_path, detail
):
    reporter = _reporter(tmp_path)
    reporter.fail(ValueError(detail))
    replay = RuntimeOperationStore(tmp_path).observe(
        "download-test", after_sequence=0, limit=100
    )
    assert "synthetic-credential" not in json.dumps(replay)
    assert "[redacted]" in json.dumps(replay)
