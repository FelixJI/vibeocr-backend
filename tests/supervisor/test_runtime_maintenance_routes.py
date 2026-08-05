"""Authenticated HTTP adapters for durable Runtime maintenance control."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from vibeocr.backend.runtime_maintenance import RuntimeCursorExpired
from vibeocr.backend.supervisor.app import create_app


def _snapshot(
    *, sequence: int = 1, operation_state: str = "succeeded"
) -> dict[str, Any]:
    return {
        "operation_id": "op-1",
        "sequence": sequence,
        "operation": "repair",
        "operation_state": operation_state,
        "phase": "commit_runtime",
        "profile_id": "win-x64-cpu",
        "updated_at": "2026-08-05T12:00:00Z",
    }


def _event(*, sequence: int = 1, operation_state: str = "succeeded") -> dict[str, Any]:
    return {
        "schema_version": 2,
        "event_type": "snapshot",
        "sequence": sequence,
        "operation": "repair",
        "snapshot": _snapshot(sequence=sequence, operation_state=operation_state),
        "message_code": "runtime.commit_runtime",
        "message_args": {},
    }


def _update(*, sequence: int = 1, operation_state: str = "succeeded") -> dict[str, Any]:
    return {
        "schema_version": 2,
        "operation_id": "op-1",
        "snapshot": _snapshot(sequence=sequence, operation_state=operation_state),
        "events": [_event(sequence=sequence, operation_state=operation_state)],
        "oldest_sequence": 1,
        "through_sequence": sequence,
        "more": False,
        "replay_expires_at": "2026-08-06T12:00:00Z",
    }


class FakeRuntimeControl:
    def __init__(self) -> None:
        self.execute_calls: list[dict[str, Any]] = []
        self.command_calls: list[dict[str, Any]] = []
        self.observe_calls: list[tuple[str, int, int]] = []
        self.update = _update()
        self.observe_error: Exception | None = None

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        self.execute_calls.append(kwargs)
        return {
            "schema_version": 2,
            "operation_id": "op-1",
            "snapshot": _snapshot(),
            "negotiated_capabilities": kwargs["required_capabilities"],
        }

    def command(self, **kwargs: Any) -> dict[str, Any]:
        self.command_calls.append(kwargs)
        return {
            "schema_version": 2,
            "operation_id": kwargs.get("new_operation_id") or "op-1",
            "snapshot": _snapshot(),
            "negotiated_capabilities": [],
        }

    def observe(
        self, operation_id: str, *, after_sequence: int, limit: int
    ) -> dict[str, Any]:
        self.observe_calls.append((operation_id, after_sequence, limit))
        if self.observe_error is not None:
            raise self.observe_error
        return self.update


def _http(
    token: str, app, *, headers: dict[str, str] | None = None
) -> httpx.AsyncClient:
    merged = {"Authorization": f"Bearer {token}", **(headers or {})}
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1",
        headers=merged,
    )


async def test_health_projects_capability_lifecycle_metadata(
    pdf_module, supervisor_token: str
) -> None:
    app = create_app(pdf_module, supervisor_token, runtime_control=FakeRuntimeControl())
    async with _http(supervisor_token, app) as http:
        response = await http.get("/v2/health")
    assert response.status_code == 200
    descriptors = {
        item["name"]: item for item in response.json()["capability_descriptors"]
    }
    assert descriptors["runtime.maintenance.v2"] == {
        "name": "runtime.maintenance.v2",
        "lifecycle": "active",
        "introduced_in": "2.3.0",
        "deprecated_in": None,
        "sunset_at": None,
        "replacement": None,
    }


async def test_start_and_retry_forward_idempotency_and_negotiation_fields(
    pdf_module, supervisor_token: str
) -> None:
    control = FakeRuntimeControl()
    app = create_app(pdf_module, supervisor_token, runtime_control=control)
    async with _http(supervisor_token, app) as http:
        started = await http.post(
            "/v2/runtime/maintenance",
            json={
                "operation_id": "op-1",
                "operation": "repair",
                "profile_id": "win-x64-cpu",
                "component_ids": ["ocr_engine"],
                "required_capabilities": ["runtime.component-repair.v1"],
            },
        )
        retried = await http.post(
            "/v2/runtime/maintenance/command",
            json={
                "command_id": "retry-1",
                "command": "retry",
                "target_operation_id": "op-1",
                "new_operation_id": "op-2",
                "expected_sequence": 1,
            },
        )
    assert started.status_code == 200
    assert started.json()["negotiated_capabilities"] == ["runtime.component-repair.v1"]
    assert control.execute_calls == [
        {
            "operation": "repair",
            "operation_id": "op-1",
            "component_ids": ("ocr_engine",),
            "required_capabilities": ("runtime.component-repair.v1",),
            "profile_id": "win-x64-cpu",
        }
    ]
    assert retried.status_code == 200
    assert control.command_calls[0]["new_operation_id"] == "op-2"


async def test_observe_maps_expired_cursor_to_canonical_410(
    pdf_module, supervisor_token: str
) -> None:
    control = FakeRuntimeControl()
    control.observe_error = RuntimeCursorExpired(
        oldest_sequence=5,
        snapshot=_snapshot(sequence=8, operation_state="running"),
    )
    app = create_app(pdf_module, supervisor_token, runtime_control=control)
    async with _http(supervisor_token, app) as http:
        response = await http.get(
            "/v2/runtime/operations/op-1/observe", params={"after_sequence": 2}
        )
    assert response.status_code == 410
    body = response.json()
    assert body["code"] == "RUNTIME_CURSOR_EXPIRED"
    assert body["category"] == "not_found"
    assert body["retryable"] is False
    assert body["detail"]["oldest_sequence"] == 5
    assert body["detail"]["snapshot"]["sequence"] == 8


async def test_sse_replays_last_event_id_and_ends_at_terminal(
    pdf_module, supervisor_token: str
) -> None:
    control = FakeRuntimeControl()
    control.update = _update(sequence=4)
    app = create_app(pdf_module, supervisor_token, runtime_control=control)
    async with _http(
        supervisor_token,
        app,
        headers={"Accept": "text/event-stream", "Last-Event-ID": "3"},
    ) as http:
        response = await http.get("/v2/runtime/operations/op-1/events")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.startswith("id: 4\nevent: runtime-maintenance\ndata: {")
    assert control.observe_calls == [("op-1", 3, 128)]


async def test_ndjson_stream_replays_events_and_requires_authentication(
    pdf_module, supervisor_token: str
) -> None:
    control = FakeRuntimeControl()
    control.update = _update(sequence=2)
    app = create_app(pdf_module, supervisor_token, runtime_control=control)
    async with _http(
        supervisor_token, app, headers={"Accept": "application/x-ndjson"}
    ) as http:
        response = await http.get(
            "/v2/runtime/operations/op-1/events", params={"after_sequence": 1}
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert response.json()["sequence"] == 2

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as anonymous:
        unauthorized = await anonymous.get(
            "/v2/runtime/operations/op-1/events",
            headers={"Accept": "application/x-ndjson"},
        )
    assert unauthorized.status_code == 401


async def test_stream_polls_running_operation_until_terminal(
    pdf_module, supervisor_token: str
) -> None:
    control = FakeRuntimeControl()
    control.update = _update(sequence=1, operation_state="running")

    def observe(
        operation_id: str, *, after_sequence: int, limit: int
    ) -> dict[str, Any]:
        control.observe_calls.append((operation_id, after_sequence, limit))
        if len(control.observe_calls) == 1:
            return _update(sequence=1, operation_state="running")
        return _update(sequence=2, operation_state="succeeded")

    control.observe = observe  # type: ignore[method-assign]
    app = create_app(pdf_module, supervisor_token, runtime_control=control)
    async with _http(
        supervisor_token, app, headers={"Accept": "application/x-ndjson"}
    ) as http:
        response = await http.get(
            "/v2/runtime/operations/op-1/events", params={"after_sequence": 0}
        )

    assert response.status_code == 200
    assert [json.loads(line)["sequence"] for line in response.text.splitlines()] == [
        1,
        2,
    ]
    assert control.observe_calls == [("op-1", 0, 128), ("op-1", 1, 128)]


async def test_stream_disconnect_stops_nonterminal_polling(
    pdf_module, supervisor_token: str
) -> None:
    control = FakeRuntimeControl()
    control.update = _update(sequence=1, operation_state="running")
    app = create_app(pdf_module, supervisor_token, runtime_control=control)
    disconnect = asyncio.Event()
    request_sent = False
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            disconnect.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/v2/runtime/operations/op-1/events",
        "raw_path": b"/v2/runtime/operations/op-1/events",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"authorization", f"Bearer {supervisor_token}".encode()),
            (b"accept", b"application/x-ndjson"),
        ],
        "client": ("127.0.0.1", 52000),
        "server": ("127.0.0.1", 80),
        "state": {},
    }

    await asyncio.wait_for(app(scope, receive, send), timeout=1)

    chunks = [
        message["body"]
        for message in messages
        if message["type"] == "http.response.body" and message.get("body")
    ]
    assert len(chunks) == 1
    assert json.loads(chunks[0])["sequence"] == 1
    assert control.observe_calls == [("op-1", 0, 128)]


async def test_stream_rejects_unnegotiated_media_type(
    pdf_module, supervisor_token: str
) -> None:
    app = create_app(pdf_module, supervisor_token, runtime_control=FakeRuntimeControl())
    async with _http(
        supervisor_token, app, headers={"Accept": "application/json"}
    ) as http:
        response = await http.get("/v2/runtime/operations/op-1/events")
    assert response.status_code == 426
    assert response.json()["code"] == "RUNTIME_CAPABILITY_UNAVAILABLE"
