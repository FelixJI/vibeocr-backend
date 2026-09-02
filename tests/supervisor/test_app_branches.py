"""Tests for the defensive / error branches of ``vibeocr.backend.supervisor.app``.

Existing test files cover the happy-path of each route; this file covers the
branches left behind: middleware loopback/auth rejection, validation errors
in submit/observe/command/release/preload/settings/export, the QR decode +
generate routes (including service-failure paths), and the per-PDF-route
``except Exception`` handlers (exercised by injecting a failing adapter).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest
from fastapi.exceptions import ResponseValidationError
from vibeocr.backend.supervisor.app import create_app
from vibeocr.backend.supervisor.bootstrap import new_instance_id
from vibeocr.backend.supervisor.module import SupervisorModule, SupervisorOptions

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import FakePdfAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _http(token: str, app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1",
        headers={"Authorization": f"Bearer {token}"},
    )


class _FailingPdfAdapter:
    """Every method raises so the per-route except branches are exercised."""

    def open_session(self, path: str) -> Any:
        raise RuntimeError("open failed")

    def close_session(self, session_id: str) -> None:
        raise RuntimeError("close failed")

    def get_model(self, session_id: str) -> Any:
        raise RuntimeError("model failed")

    def load_stream(self, session_id: str):  # type: ignore[no-untyped-def]
        raise RuntimeError("load failed")
        yield  # pragma: no cover - generator marker

    def render_thumbnail(self, session_id: str, page: int, size: int = 160) -> bytes:
        raise RuntimeError("thumb failed")

    def render_preview(self, session_id: str, page: int, dpi: int = 150) -> bytes:
        raise RuntimeError("preview failed")

    def detect_text_layers(self, session_id: str, page: int) -> Any:
        raise RuntimeError("detect failed")

    def rotate(self, session_id: str, pages: list[int], angle: int) -> Any:
        raise RuntimeError("rotate failed")

    def delete_pages(self, session_id: str, pages: list[int]) -> Any:
        raise RuntimeError("delete_pages failed")

    def insert_blank(
        self,
        session_id: str,
        after_index: int,
        width: float = 612.0,
        height: float = 792.0,
    ) -> Any:
        raise RuntimeError("insert_blank failed")

    def insert_from(self, session_id: str, source_path: str, after_index: int) -> Any:
        raise RuntimeError("insert_from failed")

    def move_page(self, session_id: str, from_index: int, to_index: int) -> Any:
        raise RuntimeError("move_page failed")

    def reorder(self, session_id: str, new_order: list[int]) -> Any:
        raise RuntimeError("reorder failed")

    def add_text_layer(
        self,
        session_id: str,
        page: int,
        ocr_result: dict,
        pdf_settings=None,
        overwrite=False,
    ) -> Any:
        raise RuntimeError("add_text_layer failed")

    def add_text_layer_batch(
        self,
        session_id: str,
        pages_data: list,
        pdf_settings=None,
        overwrite=False,
        save=False,
    ) -> Any:
        raise RuntimeError("add_text_layer_batch failed")

    def rewrite_text_layer(
        self,
        session_id: str,
        page: int,
        text_blocks: list,
        preproc_angle: int = 0,
        pdf_settings=None,
    ) -> Any:
        raise RuntimeError("rewrite_text_layer failed")

    def update_block_text(
        self, session_id: str, page: int, block_index: int, new_text: str
    ) -> Any:
        raise RuntimeError("update_block_text failed")

    def delete_text_layers_stream(self, session_id: str, pages: list):  # type: ignore[no-untyped-def]
        raise RuntimeError("delete_text_layers failed")
        yield  # pragma: no cover - generator marker

    def save(
        self, session_id: str, path=None, pdf_settings=None, *, rewrite_text_layers=True
    ) -> Any:
        raise RuntimeError("save failed")

    def save_transactional(self, session_id: str, target_path: str) -> str:
        raise RuntimeError("save_transactional failed")

    def cancel(self, session_id: str) -> None:
        raise RuntimeError("cancel failed")

    def reset_cancel(self, session_id: str) -> None:
        raise RuntimeError("reset_cancel failed")

    def stop(self) -> None:
        return


class _NonDictAdapter:
    """Adapter whose open_session returns a non-DTO/non-dict value (line 408)."""

    def open_session(self, path: str) -> Any:
        # Returning a non-dict, non-pydantic value exercises the
        # ``_pdf_response`` ``else`` branch ({"value": payload}).
        return 42

    def stop(self) -> None:
        return


@pytest.fixture()
def failing_module(tmp_path: Path) -> SupervisorModule:
    opts = SupervisorOptions(instance_id=new_instance_id())
    # Use a null-ish executor: PDF routes don't run jobs.
    from conftest import NullExecutor

    return SupervisorModule(
        options=opts,
        stager_root=tmp_path / "staging",
        executor=NullExecutor(),
        pdf_adapter=_FailingPdfAdapter(),
    )


@pytest.fixture()
def failing_app(failing_module: SupervisorModule, supervisor_token: str):
    return create_app(failing_module, supervisor_token)


# ---------------------------------------------------------------------------
# Middleware: loopback + bearer
# ---------------------------------------------------------------------------


async def test_loopback_rejection_returns_forbidden(
    pdf_app, supervisor_token: str
) -> None:
    """A non-loopback client host must be rejected by the middleware."""
    # The ASGI transport presents 127.0.0.1; patch the middleware path by
    # using a custom transport that injects a non-loopback client.
    transport = httpx.ASGITransport(app=pdf_app, client=("10.0.0.5", 12345))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://10.0.0.5",
        headers={"Authorization": f"Bearer {supervisor_token}"},
    ) as http:
        resp = await http.get("/v2/runtime/residency")
    assert resp.status_code == 403
    assert resp.json()["code"] == "FORBIDDEN_LOOPBACK"


async def test_malformed_authorization_header_is_unauthorized(
    pdf_app, supervisor_token: str
) -> None:
    """A header that is not ``Bearer <token>`` is rejected (line 84 / 53)."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=pdf_app),
        base_url="http://127.0.0.1",
        headers={"Authorization": "Basic xyz"},
    ) as http:
        resp = await http.get("/v2/runtime/residency")
    assert resp.status_code == 401
    assert resp.json()["code"] == "UNAUTHORIZED"


async def test_runtime_status_returns_injected_profile_and_maintenance(
    pdf_module: SupervisorModule,
    supervisor_token: str,
) -> None:
    calls: list[tuple[str, str]] = []

    def status_provider(instance_id: str, service_state: str) -> dict[str, Any]:
        calls.append((instance_id, service_state))
        return {
            "schema_version": 2,
            "instance_id": instance_id,
            "service_state": service_state,
            "backend_version": "0.8.2",
            "profile": {
                "profile_id": "win-x64-cpu",
                "accelerator": "cpu",
                "components": [
                    {
                        "component_id": "ocr_engine",
                        "display_name": "OCR engine",
                        "state": "ready",
                        "version": "3.7.0",
                    }
                ],
            },
            "maintenance": None,
        }

    app = create_app(
        pdf_module,
        supervisor_token,
        runtime_status_provider=status_provider,
    )
    async with _http(supervisor_token, app) as http:
        response = await http.get("/v2/runtime/status")

    assert response.status_code == 200
    assert response.json()["profile"]["components"][0]["version"] == "3.7.0"
    assert calls == [(pdf_module.options.instance_id, "ready")]


# ---------------------------------------------------------------------------
# Submit job: content-type / manifest / quota / shutdown / generic error
# ---------------------------------------------------------------------------


async def test_submit_rejects_non_multipart_content_type(
    pdf_app, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/jobs",
            json={"x": 1},
            headers={"Authorization": f"Bearer {supervisor_token}"},
        )
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "VALIDATION_ERROR"
    assert body["detail"]["field"] == "content-type"


async def test_submit_rejects_when_manifest_is_not_a_string_field(
    pdf_app, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/jobs",
            files={"manifest": ("manifest", b"", "text/plain")},
        )
    assert resp.status_code == 400
    assert resp.json()["code"] == "VALIDATION_ERROR"


async def test_submit_rejects_when_manifest_is_invalid_json(
    pdf_app, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/jobs",
            data={"manifest": "{not json"},
        )
    assert resp.status_code == 400
    assert resp.json()["code"] == "VALIDATION_ERROR"


async def test_submit_rejects_when_attachment_count_is_wrong(
    pdf_app, supervisor_token: str
) -> None:
    """An attachment referenced by the manifest but absent from the form
    triggers a ValueError → VALIDATION_ERROR (line 130-133 + handler)."""
    manifest = (
        '{"schema_version":2,"request_id":"r","kind":"recognition",'
        '"priority":"interactive","items":[{"client_item_key":"k","ordinal":0,'
        '"display_name":"a.png","source":{"type":"upload.v1","attachment":"missing"}}]}'
    )
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post("/v2/jobs", data={"manifest": manifest})
    assert resp.status_code == 400


async def test_submit_returns_draining_when_module_rejects(
    pdf_app, supervisor_token: str, pdf_module: SupervisorModule
) -> None:
    """When the module raises ShutdownRequested, the route returns DRAINING."""
    from vibeocr.backend.supervisor.module import ShutdownRequested

    def raise_draining(*a, **k):  # type: ignore[no-untyped-def]
        raise ShutdownRequested("draining")

    pdf_module.submit_request = raise_draining  # type: ignore[assignment]
    manifest = (
        '{"schema_version":2,"request_id":"r","kind":"recognition",'
        '"priority":"interactive","pipeline":{"pipeline_id":"OCR",'
        '"options_version":1,"options":{}},'
        '"items":[{"client_item_key":"k","ordinal":0,'
        '"display_name":"a.png","source":{"type":"upload.v1","attachment":"f"}}]}'
    )
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/jobs",
            data={"manifest": manifest},
            files={"f": ("a.png", b"x", "image/png")},
        )
    assert resp.status_code == 503
    assert resp.json()["code"] == "SUPERVISOR_DRAINING"


# ---------------------------------------------------------------------------
# Observe: not found + validation
# ---------------------------------------------------------------------------


async def test_observe_unknown_job_returns_not_found(
    pdf_app, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.get("/v2/jobs/unknown/observe")
    assert resp.status_code == 404
    assert resp.json()["code"] == "JOB_NOT_FOUND"


async def test_observe_negative_after_sequence_returns_validation_error(
    pdf_app, supervisor_token: str, pdf_module: SupervisorModule
) -> None:
    """observe() with after_sequence=-1 raises ContractError → VALIDATION_ERROR.
    Needs a real job id because JobNotFoundError is checked first."""
    from vibeocr.runtime_contracts import JobKind, JobPriority

    ref = pdf_module.submit(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        uploads=[("a.png", None, b"1")],
    )
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.get(f"/v2/jobs/{ref.job_id}/observe?after_sequence=-1")
    assert resp.status_code == 400
    assert resp.json()["code"] == "VALIDATION_ERROR"


# ---------------------------------------------------------------------------
# Command: not found + cancel terminal + retry expired + validation
# ---------------------------------------------------------------------------


async def test_command_unknown_returns_not_found(
    pdf_app, supervisor_token: str
) -> None:
    body = {
        "command_id": "c1",
        "kind": "forget",
        "job_id": "unknown",
    }
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post("/v2/jobs/command", json=body)
    assert resp.status_code == 404
    assert resp.json()["code"] == "JOB_NOT_FOUND"


async def test_command_malformed_body_returns_validation_error(
    pdf_app, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post("/v2/jobs/command", json={"not": "a command"})
    assert resp.status_code == 400
    assert resp.json()["code"] == "VALIDATION_ERROR"


async def test_command_cancel_on_terminal_returns_not_cancellable(
    pdf_app, supervisor_token: str, pdf_module: SupervisorModule
) -> None:
    """Cancelling an already-terminal job raises ShutdownRequested in
    request_cancel; the route maps that to JOB_NOT_CANCELLABLE."""
    from vibeocr.runtime_contracts import TERMINAL_JOB_STATES, JobKind, JobPriority

    ref = pdf_module.submit(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        uploads=[("a.png", None, b"1")],
    )
    # Wait for terminal (NullExecutor fails immediately).
    import time

    deadline = time.time() + 2.0
    while time.time() < deadline:
        snap = pdf_module.status(ref.job_id)
        if snap.state in TERMINAL_JOB_STATES:
            break
        time.sleep(0.01)
    body = {
        "command_id": "c1",
        "kind": "cancel",
        "job_id": ref.job_id,
    }
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post("/v2/jobs/command", json=body)
    assert resp.status_code == 409
    assert resp.json()["code"] == "JOB_NOT_CANCELLABLE"


# ---------------------------------------------------------------------------
# Release runtime: malformed JSON rejected
# ---------------------------------------------------------------------------


async def test_release_runtime_rejects_invalid_json(
    pdf_app, supervisor_token: str
) -> None:
    """Invalid JSON cannot bypass the generated request contract."""
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/runtime/release",
            content=b"not json",
            headers={"Authorization": f"Bearer {supervisor_token}"},
        )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Preload: invalid body / empty pipelines / unknown pipelines
# ---------------------------------------------------------------------------


async def test_preload_rejects_invalid_json(pdf_app, supervisor_token: str) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/runtime/preload",
            content=b"not json",
            headers={"Authorization": f"Bearer {supervisor_token}"},
        )
    assert resp.status_code == 400
    assert resp.json()["code"] == "VALIDATION_ERROR"


async def test_preload_rejects_empty_pipelines(pdf_app, supervisor_token: str) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post("/v2/runtime/preload", json={"pipelines": []})
    assert resp.status_code == 400


async def test_preload_rejects_non_string_pipelines(
    pdf_app, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post("/v2/runtime/preload", json={"pipelines": [1, 2]})
    assert resp.status_code == 400


async def test_preload_rejects_duplicate_pipelines(
    pdf_app, supervisor_token: str
) -> None:
    """The formal uniqueItems constraint is enforced at runtime."""
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/runtime/preload",
            json={"pipelines": ["OCR", "OCR"]},
        )
    assert resp.status_code == 400
    assert resp.json()["code"] == "VALIDATION_ERROR"


async def test_preload_rejects_unknown_pipeline(pdf_app, supervisor_token: str) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/runtime/preload", json={"pipelines": ["DoesNotExist"]}
        )
    assert resp.status_code == 400
    assert "DoesNotExist" in resp.json()["detail"]["reason"]


# ---------------------------------------------------------------------------
# Settings PUT: validation error matrix
# ---------------------------------------------------------------------------


async def test_put_settings_rejects_invalid_json(
    pdf_app, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.request(
            "PUT",
            "/v2/settings",
            content=b"not json",
        )
    assert resp.status_code == 400


async def test_put_settings_rejects_non_object_body(
    pdf_app, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.request("PUT", "/v2/settings", json=[1, 2, 3])
    assert resp.status_code == 400


async def test_put_settings_rejects_negative_ttl(
    pdf_app, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.request(
            "PUT",
            "/v2/settings",
            json={"residency": {"default_ttl_seconds": -1}},
        )
    assert resp.status_code == 400


async def test_put_settings_rejects_string_ttl(pdf_app, supervisor_token: str) -> None:
    """Nested JSON Schema integer fields are validated without coercion."""
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.put(
            "/v2/settings",
            json={
                "schema_version": 2,
                "residency": {
                    "default_ttl_seconds": "300",
                    "pipelines": [],
                },
                "extra": {},
            },
        )
    assert resp.status_code == 400
    assert resp.json()["code"] == "VALIDATION_ERROR"


async def test_put_settings_rejects_non_dict_residency(
    pdf_app, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.request(
            "PUT",
            "/v2/settings",
            json={"residency": 5},
        )
    assert resp.status_code == 400


async def test_put_settings_rejects_non_list_pipelines(
    pdf_app, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.request(
            "PUT",
            "/v2/settings",
            json={"residency": {"pipelines": "OCR"}},
        )
    assert resp.status_code == 400


async def test_put_settings_rejects_non_dict_extra(
    pdf_app, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.request(
            "PUT",
            "/v2/settings",
            json={"residency": {}, "extra": "no"},
        )
    assert resp.status_code == 400


async def test_put_settings_rejects_invalid_pipeline_spec(
    pdf_app, supervisor_token: str
) -> None:
    """An invalid ttl_seconds triggers ValueError → VALIDATION_ERROR."""
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.request(
            "PUT",
            "/v2/settings",
            json={
                "residency": {
                    "pipelines": [{"name": "OCR", "ttl_seconds": -5}],
                }
            },
        )
    assert resp.status_code == 400


async def test_put_settings_accepts_valid_snapshot(
    pdf_app, supervisor_token: str
) -> None:
    """A well-formed body round-trips (covers the success path)."""
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.request(
            "PUT",
            "/v2/settings",
            json={
                "schema_version": 2,
                "residency": {
                    "default_ttl_seconds": 600,
                    "pipelines": [],
                },
                "extra": {},
            },
        )
    assert resp.status_code == 200


async def test_put_settings_uses_recognition_mode_lifecycle_registry(
    pdf_app, supervisor_token: str
) -> None:
    def payload(name: str, recognition_mode: str) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "residency": {
                "default_ttl_seconds": 600,
                "pipelines": [
                    {
                        "name": name,
                        "recognition_mode": recognition_mode,
                        "ttl_seconds": None,
                        "pinned": False,
                    }
                ],
            },
            "extra": {},
        }

    async with _http(supervisor_token, pdf_app) as http:
        mineru = await http.put(
            "/v2/settings", json=payload("MinerU", "mineru_document")
        )
        mismatch = await http.put("/v2/settings", json=payload("MinerU", "paddle_text"))
        unmanaged = await http.put("/v2/settings", json=payload("OCR", "rapid_text"))

    assert mineru.status_code == 200, mineru.text
    pipeline = mineru.json()["residency"]["pipelines"][0]
    assert pipeline["recognition_mode"] == "mineru_document"
    assert mismatch.json()["code"] == "RECOGNITION_MODE_PIPELINE_MISMATCH"
    assert unmanaged.json()["code"] == "RECOGNITION_MODE_LIFECYCLE_UNSUPPORTED"


# ---------------------------------------------------------------------------
# Export route: validation + failure paths
# ---------------------------------------------------------------------------


async def test_export_rejects_invalid_json(pdf_app, supervisor_token: str) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/export",
            content=b"not json",
            headers={"Authorization": f"Bearer {supervisor_token}"},
        )
    assert resp.status_code == 400


async def test_export_rejects_non_object_body(pdf_app, supervisor_token: str) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post("/v2/export", json=[1, 2])
    assert resp.status_code == 400


async def test_export_rejects_invalid_request_shape(
    pdf_app, supervisor_token: str
) -> None:
    """A malformed table block triggers validate_table_blocks ValueError → 400."""
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/export",
            json={
                "raw_text": "x",
                "output_path": "out.md",
                "format": "markdown",
                "raw_blocks": [{"type": "table", "table": {"version": "bad"}}],
            },
        )
    assert resp.status_code == 400


async def test_export_returns_internal_error_when_service_fails(
    pdf_app, supervisor_token: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force ExportService.export to raise → INTERNAL_ERROR path (line 367-368)."""
    from vibeocr.backend.services import export_service

    def boom(*a, **k):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    monkeypatch.setattr(export_service.ExportService, "export", boom)
    target = tmp_path / "out.md"
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/export",
            json={
                "raw_text": "x",
                "output_path": str(target),
                "format": "markdown",
            },
        )
    assert resp.status_code == 500
    assert resp.json()["code"] == "INTERNAL_ERROR"


async def test_export_returns_internal_error_when_export_returns_false(
    pdf_app, supervisor_token: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ExportService.export returning False → INTERNAL_ERROR (lines 362-365)."""
    from vibeocr.backend.services import export_service

    monkeypatch.setattr(export_service.ExportService, "export", lambda *a, **k: False)
    target = tmp_path / "out.md"
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/export",
            json={
                "raw_text": "x",
                "output_path": str(target),
                "format": "markdown",
            },
        )
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# PDF routes — failing adapter exercises every per-route except branch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("POST", "/v2/pdf/sessions/open", {"path": "x.pdf"}),
        ("POST", "/v2/pdf/sessions/sid/close", None),
        ("POST", "/v2/pdf/sessions/sid/model", None),
        ("POST", "/v2/pdf/sessions/sid/load", None),
        ("POST", "/v2/pdf/sessions/sid/render_thumbnail", {"page": 0}),
        ("POST", "/v2/pdf/sessions/sid/render_preview", {"page": 0}),
        ("POST", "/v2/pdf/sessions/sid/detect_text_layers", {"page": 0}),
        ("POST", "/v2/pdf/sessions/sid/rotate", {"pages": [0], "angle": 90}),
        ("POST", "/v2/pdf/sessions/sid/delete_pages", {"pages": [0]}),
        ("POST", "/v2/pdf/sessions/sid/insert_blank", {"after_index": 0}),
        (
            "POST",
            "/v2/pdf/sessions/sid/insert_from",
            {"source_path": "x.pdf", "after_index": 0},
        ),
        ("POST", "/v2/pdf/sessions/sid/move_page", {"from_index": 0, "to_index": 1}),
        ("POST", "/v2/pdf/sessions/sid/reorder", {"new_order": [0]}),
        ("POST", "/v2/pdf/sessions/sid/add_text_layer", {"page": 0, "ocr_result": {}}),
        ("POST", "/v2/pdf/sessions/sid/add_text_layer_batch", {"pages": []}),
        (
            "POST",
            "/v2/pdf/sessions/sid/rewrite_text_layer",
            {"page": 0, "text_blocks": []},
        ),
        (
            "POST",
            "/v2/pdf/sessions/sid/update_block_text",
            {"page": 0, "block_index": 0, "new_text": "updated"},
        ),
        ("POST", "/v2/pdf/sessions/sid/delete_text_layers", {"pages": [0]}),
        ("POST", "/v2/pdf/sessions/sid/save", {"path": "out.pdf"}),
        ("POST", "/v2/pdf/sessions/sid/save_transactional", {"path": "/tmp/out.pdf"}),
        ("POST", "/v2/pdf/sessions/sid/cancel", None),
        ("POST", "/v2/pdf/sessions/sid/reset_cancel", None),
    ],
)
async def test_pdf_route_returns_internal_error_when_adapter_fails(
    failing_app, supervisor_token: str, method: str, path: str, body
) -> None:
    async with _http(supervisor_token, failing_app) as http:
        resp = await http.request(method, path, json=body)
    # All routes should map RuntimeError to INTERNAL_ERROR (500) — except
    # streaming routes which emit an error line and stay 200.
    if path.endswith(("/load", "/delete_text_layers")):
        assert resp.status_code == 200
        assert "error:" in resp.text
    else:
        assert resp.status_code == 500, (path, resp.text)
        assert resp.json()["code"] == "INTERNAL_ERROR"


async def test_pdf_open_rejects_non_contract_adapter_result(
    tmp_path: Path, supervisor_token: str
) -> None:
    """Generated response DTOs reject malformed Backend adapter output."""
    from conftest import NullExecutor

    opts = SupervisorOptions(instance_id=new_instance_id())
    module = SupervisorModule(
        options=opts,
        stager_root=tmp_path / "staging",
        executor=NullExecutor(),
        pdf_adapter=_NonDictAdapter(),
    )
    app = create_app(module, supervisor_token)
    with pytest.raises(ResponseValidationError):
        async with _http(supervisor_token, app) as http:
            await http.post("/v2/pdf/sessions/open", json={"path": "x.pdf"})


# ---------------------------------------------------------------------------
# QR routes
# ---------------------------------------------------------------------------


async def test_qrcode_decode_rejects_invalid_json(
    pdf_app, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/qrcode/decode",
            content=b"not json",
            headers={"Authorization": f"Bearer {supervisor_token}"},
        )
    assert resp.status_code == 400


async def test_qrcode_decode_rejects_missing_image_field(
    pdf_app, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post("/v2/qrcode/decode", json={})
    assert resp.status_code == 400
    assert resp.json()["detail"]["field"] == "image"


async def test_qrcode_decode_rejects_non_object_body(
    pdf_app, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post("/v2/qrcode/decode", json=[1, 2])
    assert resp.status_code == 400


async def test_qrcode_decode_rejects_invalid_image_payload(
    pdf_app, supervisor_token: str
) -> None:
    import base64

    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/qrcode/decode",
            json={"image": base64.b64encode(b"not an image").decode()},
        )
    assert resp.status_code == 400
    assert resp.json()["detail"]["reason"] == "invalid image"


async def test_qrcode_decode_returns_internal_error_when_service_fails(
    pdf_app, supervisor_token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force the decode service to raise → INTERNAL_ERROR (line 810-811)."""
    from vibeocr.backend.services import qrcode_decode_service

    class _Boom:
        def __init__(self, *a, **k):  # type: ignore[no-untyped-def]
            pass

        def decode(self, img):  # type: ignore[no-untyped-def]
            raise RuntimeError("decode boom")

    monkeypatch.setattr(qrcode_decode_service, "QrcodeDecodeService", _Boom)

    # Build a tiny valid PNG so the PIL.open branch succeeds.
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post("/v2/qrcode/decode", json={"image": img_b64})
    assert resp.status_code == 500
    assert resp.json()["code"] == "INTERNAL_ERROR"


async def test_qrcode_decode_round_trips_with_real_image(
    pdf_app, supervisor_token: str
) -> None:
    """End-to-end: encode a QR, decode it. Covers the happy path (lines 786-816)."""
    import base64
    import io

    from vibeocr.backend.services.qrcode_service import QrcodeService

    png = QrcodeService().generate("vibeocr-branch-test", {})
    buf = io.BytesIO()
    png.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post("/v2/qrcode/decode", json={"image": img_b64})
    assert resp.status_code == 200
    codes = resp.json()["codes"]
    assert any(c["data"] == "vibeocr-branch-test" for c in codes)


async def test_qrcode_generate_rejects_invalid_json(
    pdf_app, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/qrcode/generate",
            content=b"not json",
            headers={"Authorization": f"Bearer {supervisor_token}"},
        )
    assert resp.status_code == 400


async def test_qrcode_generate_rejects_missing_data_field(
    pdf_app, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post("/v2/qrcode/generate", json={})
    assert resp.status_code == 400
    assert resp.json()["detail"]["field"] == "data"


async def test_qrcode_generate_rejects_non_object_body(
    pdf_app, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post("/v2/qrcode/generate", json="hi")
    assert resp.status_code == 400


async def test_qrcode_generate_returns_internal_error_when_service_fails(
    pdf_app, supervisor_token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vibeocr.backend.services import qrcode_service

    class _Boom:
        def __init__(self, *a, **k):  # type: ignore[no-untyped-def]
            pass

        def generate(self, *a, **k):  # type: ignore[no-untyped-def]
            raise RuntimeError("gen boom")

        def generate_svg(self, *a, **k):  # type: ignore[no-untyped-def]
            raise RuntimeError("svg boom")

    monkeypatch.setattr(qrcode_service, "QrcodeService", _Boom)
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/qrcode/generate", json={"data": "x", "format": "qr"}
        )
    assert resp.status_code == 500
    assert resp.json()["code"] == "INTERNAL_ERROR"


async def test_qrcode_generate_unknown_barcode_format_returns_validation_error(
    pdf_app, supervisor_token: str
) -> None:
    """未知条形码格式是请求侧问题：400 VALIDATION_ERROR，不是 500。

    回归：旧实现把 BarcodeNotFoundError 兜底成 INTERNAL_ERROR，
    Classic 前端 format 契约错位时只能看到无指向的"内部错误"。
    """
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/qrcode/generate",
            json={
                "data": "x",
                "format": "qrcode",
                "options": {"format": "nonexistent_format"},
            },
        )
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "VALIDATION_ERROR"
    assert "NONEXISTENT_FORMAT" in body["detail"]["reason"].upper()


async def test_qrcode_generate_applies_label_options(
    pdf_app, supervisor_token: str
) -> None:
    """options 中的 label_text 必须体现在返回图（高度增加）。"""
    import base64
    import io

    from PIL import Image

    async with _http(supervisor_token, pdf_app) as http:
        plain = await http.post(
            "/v2/qrcode/generate", json={"data": "labeled", "format": "qrcode"}
        )
        labeled = await http.post(
            "/v2/qrcode/generate",
            json={
                "data": "labeled",
                "format": "qrcode",
                "options": {
                    "format": "qr",
                    "size": 300,
                    "label_text": "Scan",
                    "label_position": "bottom",
                },
            },
        )
    assert plain.status_code == 200 and labeled.status_code == 200
    img_plain = Image.open(io.BytesIO(base64.b64decode(plain.json()["image"])))
    img_labeled = Image.open(io.BytesIO(base64.b64decode(labeled.json()["image"])))
    assert img_labeled.height > img_plain.height


async def test_qrcode_generate_svg_path_emits_svg(
    pdf_app, supervisor_token: str
) -> None:
    """The svg branch (lines 836-838) is covered by the e2e test, but include
    a direct assertion here too for completeness."""
    import base64

    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/qrcode/generate", json={"data": "svg-branch", "format": "svg"}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["media_type"] == "image/svg+xml"
    assert b"<svg" in base64.b64decode(body["image"])


# ---------------------------------------------------------------------------
# Additional happy-path / branch coverage
# ---------------------------------------------------------------------------


async def test_submit_skips_non_upload_source_items(
    pdf_app, supervisor_token: str
) -> None:
    """Items whose source type is not ``upload.v1`` are skipped (line 127).
    A pdf_page.v1 source is recognised by the parser but produces no upload,
    so submit_request then fails with a StagingQuotaError → QUOTA_EXCEEDED
    (also covers line 141)."""
    manifest = (
        '{"schema_version":2,"request_id":"r","kind":"recognition",'
        '"priority":"interactive","pipeline":{"pipeline_id":"OCR",'
        '"options_version":1,"options":{}},'
        '"items":[{"client_item_key":"k","ordinal":0,'
        '"display_name":"a.pdf","source":{"type":"pdf_page.v1",'
        '"session_id":"s","session_revision":1,"page_index":0}}]}'
    )
    # Include a dummy unreferenced file so httpx sends multipart/form-data;
    # submit_request then rejects the unreferenced attachment OR the empty
    # uploads list — either way it must be a typed validation/quota error,
    # not a 200 success.
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/jobs",
            data={"manifest": manifest},
            files={"unused": ("u.png", b"x", "image/png")},
        )
    assert resp.status_code in (400, 413)
    assert resp.json()["code"] in ("VALIDATION_ERROR", "QUOTA_EXCEEDED")


async def test_submit_returns_quota_when_attachment_missing(
    pdf_app, supervisor_token: str
) -> None:
    """Referenced attachment is absent from the form (line 130-133)."""
    manifest = (
        '{"schema_version":2,"request_id":"r","kind":"recognition",'
        '"priority":"interactive","pipeline":{"pipeline_id":"OCR",'
        '"options_version":1,"options":{}},'
        '"items":[{"client_item_key":"k","ordinal":0,'
        '"display_name":"a.png","source":{"type":"upload.v1","attachment":"f"}}]}'
    )
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/jobs",
            data={"manifest": manifest},
        )
    assert resp.status_code == 400
    assert resp.json()["code"] == "VALIDATION_ERROR"


async def test_command_cancel_returns_mode_when_job_exists(
    pdf_app, supervisor_token: str, pdf_module: SupervisorModule
) -> None:
    """Successful cancel command returns the cancel_mode (line 179-186)."""
    # Submit then immediately cancel — even if the job completes first,
    # cancelling a terminal job returns JOB_NOT_CANCELLABLE which covers
    # the ShutdownRequested path. To hit the success branch we need a
    # non-terminal job, so we use a hanging executor.
    import threading

    from vibeocr.runtime_contracts import JobKind, JobPriority

    hang_entered = threading.Event()
    hang_release = threading.Event()

    class HangingExecutor:
        def execute(self, record, staged):  # type: ignore[no-untyped-def]
            from vibeocr.runtime_contracts import JobState

            record.transition(JobState.RUNNING)
            hang_entered.set()
            hang_release.wait(timeout=2.0)

        def cancel_mode_for(self, record):  # type: ignore[no-untyped-def]
            from vibeocr.runtime_contracts import CancelMode

            return CancelMode.COOPERATIVE

    pdf_module._executor = HangingExecutor()  # type: ignore[attr-defined]
    ref = pdf_module.submit(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        uploads=[("a.png", None, b"1")],
    )
    assert hang_entered.wait(timeout=1.0)
    body = {"command_id": "c1", "kind": "cancel", "job_id": ref.job_id}
    try:
        async with _http(supervisor_token, pdf_app) as http:
            resp = await http.post("/v2/jobs/command", json=body)
        assert resp.status_code == 200
        assert resp.json()["cancel_mode"] == "cooperative"
    finally:
        hang_release.set()


async def test_command_forget_returns_success_when_job_terminal(
    pdf_app, supervisor_token: str, pdf_module: SupervisorModule
) -> None:
    """Successful forget command (line 198-205)."""
    import time

    from vibeocr.runtime_contracts import TERMINAL_JOB_STATES, JobKind, JobPriority

    ref = pdf_module.submit(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        uploads=[("a.png", None, b"1")],
    )
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if pdf_module.status(ref.job_id).state in TERMINAL_JOB_STATES:
            break
        time.sleep(0.01)
    body = {"command_id": "c1", "kind": "forget", "job_id": ref.job_id}
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post("/v2/jobs/command", json=body)
    assert resp.status_code == 200
    assert resp.json()["kind"] == "forget"


async def test_release_runtime_forwards_pipeline_argument(
    pdf_app, supervisor_token: str, fake_pdf_adapter: FakePdfAdapter
) -> None:
    """Valid JSON body with a ``pipeline`` field forwards it (lines 236-237)."""
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post("/v2/runtime/release", json={"pipeline": "OCR"})
    assert resp.status_code == 200


async def test_release_runtime_rejects_non_object_json(
    pdf_app, supervisor_token: str
) -> None:
    """The formal request schema requires a JSON object."""
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post("/v2/runtime/release", json=[1, 2, 3])
    assert resp.status_code == 400


async def test_preload_success_path(
    pdf_app, supervisor_token: str, pdf_module: SupervisorModule
) -> None:
    """An explicit Paddle model preload request returns 200."""
    from vibeocr.backend.supervisor.inference.recognition_modes import (
        ModeAvailability,
        RecognitionModeRegistry,
    )

    pdf_module.recognition_mode_registry = RecognitionModeRegistry(
        availability_probe=lambda _definition: ModeAvailability("ready")
    )
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/runtime/preload",
            json={"pipelines": ["PP-StructureV3"]},
        )
    assert resp.status_code == 200


async def test_preload_returns_internal_error_on_executor_failure(
    pdf_app, supervisor_token: str, pdf_module: SupervisorModule
) -> None:
    """When the executor's preload raises, the route returns INTERNAL_ERROR."""
    from vibeocr.backend.supervisor.inference.recognition_modes import (
        ModeAvailability,
        RecognitionModeRegistry,
    )

    pdf_module.recognition_mode_registry = RecognitionModeRegistry(
        availability_probe=lambda _definition: ModeAvailability("ready")
    )

    def boom(_pipelines):  # type: ignore[no-untyped-def]
        raise RuntimeError("preload boom")

    pdf_module._executor.preload = boom  # type: ignore[attr-defined]
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/runtime/preload",
            json={"pipelines": ["PP-StructureV3"]},
        )
    assert resp.status_code == 500
    assert resp.json()["code"] == "INTERNAL_ERROR"


async def test_export_success_returns_bytes_written(
    pdf_app, supervisor_token: str, tmp_path: Path
) -> None:
    """A successful export returns bytes_written + output_path (lines 366-374)."""
    target = tmp_path / "out.md"
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/export",
            json={
                "raw_text": "hello",
                "output_path": str(target),
                "format": "markdown",
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["output_path"] == str(target)
    assert body["bytes_written"] > 0


async def test_pdf_load_returns_internal_error_when_no_adapter(
    tmp_path: Path, supervisor_token: str
) -> None:
    """The streaming load route maps ``_PdfUnavailable`` to INTERNAL_ERROR
    before the generator starts (lines 459-460)."""
    from conftest import NullExecutor

    opts = SupervisorOptions(instance_id=new_instance_id())
    module = SupervisorModule(
        options=opts,
        stager_root=tmp_path / "staging",
        executor=NullExecutor(),
        pdf_adapter=None,
    )
    app = create_app(module, supervisor_token)
    async with _http(supervisor_token, app) as http:
        resp = await http.post("/v2/pdf/sessions/sid/load")
    assert resp.status_code == 500
    assert resp.json()["code"] == "INTERNAL_ERROR"


async def test_pdf_delete_text_layers_returns_internal_error_when_no_adapter(
    tmp_path: Path, supervisor_token: str
) -> None:
    """The streaming delete_text_layers route maps no-adapter to INTERNAL_ERROR
    before the generator starts (lines 695-696)."""
    from conftest import NullExecutor

    opts = SupervisorOptions(instance_id=new_instance_id())
    module = SupervisorModule(
        options=opts,
        stager_root=tmp_path / "staging",
        executor=NullExecutor(),
        pdf_adapter=None,
    )
    app = create_app(module, supervisor_token)
    async with _http(supervisor_token, app) as http:
        resp = await http.post(
            "/v2/pdf/sessions/sid/delete_text_layers", json={"pages": [0]}
        )
    assert resp.status_code == 500
    assert resp.json()["code"] == "INTERNAL_ERROR"


async def test_pdf_render_get_returns_internal_error_when_no_adapter(
    tmp_path: Path, supervisor_token: str
) -> None:
    """The GET render route maps no-adapter to INTERNAL_ERROR (lines 516-517)."""
    from conftest import NullExecutor

    opts = SupervisorOptions(instance_id=new_instance_id())
    module = SupervisorModule(
        options=opts,
        stager_root=tmp_path / "staging",
        executor=NullExecutor(),
        pdf_adapter=None,
    )
    app = create_app(module, supervisor_token)
    async with _http(supervisor_token, app) as http:
        resp = await http.get("/v2/pdf/sessions/sid/render")
    assert resp.status_code == 500
    assert resp.json()["code"] == "INTERNAL_ERROR"


async def test_pdf_insert_from_rejects_missing_source_path(
    pdf_app, supervisor_token: str
) -> None:
    """insert_from with empty source_path raises _PdfBadRequest (line 574)."""
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/pdf/sessions/sid/insert_from", json={"after_index": 0}
        )
    assert resp.status_code == 400
    assert resp.json()["code"] == "VALIDATION_ERROR"


async def test_pdf_body_rejects_non_object_json(pdf_app, supervisor_token: str) -> None:
    """A JSON list body raises _PdfBadRequest (line 398)."""
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post("/v2/pdf/sessions/sid/rotate", content=b"[1, 2, 3]")
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "VALIDATION_ERROR"
    assert "JSON object" in body["detail"]["reason"]


async def test_pdf_body_rejects_numeric_strings(pdf_app, supervisor_token: str) -> None:
    """Wire integer fields cannot rely on Pydantic's coercion."""
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/pdf/sessions/sid/rotate",
            json={"pages": ["1"], "angle": "90"},
        )
    assert resp.status_code == 400
    assert resp.json()["code"] == "VALIDATION_ERROR"


async def test_submit_rejects_when_attachment_occurs_zero_or_two_times(
    pdf_app, supervisor_token: str
) -> None:
    """A referenced attachment that does not occur exactly once in the form
    triggers a ValueError (line 130-133)."""
    manifest = (
        '{"schema_version":2,"request_id":"r","kind":"recognition",'
        '"priority":"interactive","pipeline":{"pipeline_id":"OCR",'
        '"options_version":1,"options":{}},'
        '"items":[{"client_item_key":"k","ordinal":0,'
        '"display_name":"a.png","source":{"type":"upload.v1","attachment":"f"}}]}'
    )
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/jobs",
            data={"manifest": manifest},
            files=[
                ("f", ("a.png", b"x", "image/png")),
                ("f", ("a.png", b"y", "image/png")),
            ],
        )
    assert resp.status_code == 400
    assert resp.json()["code"] == "VALIDATION_ERROR"


async def test_command_retry_returns_new_job_ref(
    pdf_app, supervisor_token: str, tmp_path: Path
) -> None:
    """Successful retry command returns a new job_ref (line 188-196).

    Uses a real module with a failing-then-succeeding executor swap to land
    a retryable (failed-item) source job, then a successful retry."""
    import time

    from vibeocr.runtime_contracts import (
        TERMINAL_JOB_STATES,
        JobKind,
        JobPriority,
    )

    opts = SupervisorOptions(instance_id=new_instance_id())
    mod = SupervisorModule(
        options=opts,
        stager_root=tmp_path / "staging",
        executor=None,  # set below
        pdf_adapter=None,
    )

    # First executor: fail every item then mark the job completed-with-errors.
    class _FailItems:
        def execute(self, record, staged):  # type: ignore[no-untyped-def]
            from vibeocr.runtime_contracts import JobState

            record.transition(JobState.RUNNING)
            for it in list(record.items):
                record.commit_item_failure(
                    it.item_id, error_code="BACKEND_UNAVAILABLE", error="boom"
                )
            record.transition(JobState.COMPLETED_WITH_ERRORS)

        def cancel_mode_for(self, record):  # type: ignore[no-untyped-def]
            from vibeocr.runtime_contracts import CancelMode

            return CancelMode.COOPERATIVE

    mod._executor = _FailItems()  # type: ignore[attr-defined]
    app = create_app(mod, supervisor_token)
    ref = mod.submit(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        uploads=[("a.png", None, b"1")],
    )
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if mod.status(ref.job_id).state in TERMINAL_JOB_STATES:
            break
        time.sleep(0.01)

    # Swap to a succeeding executor for the retry.
    class _Succeed:
        def execute(self, record, staged):  # type: ignore[no-untyped-def]
            from vibeocr.runtime_contracts import JobState

            record.transition(JobState.RUNNING)
            for it in list(record.items):
                record.commit_item_success(
                    it.item_id, payload_type="ocr.v1", payload={"text": "ok"}
                )
            record.transition(JobState.COMPLETED)

        def cancel_mode_for(self, record):  # type: ignore[no-untyped-def]
            from vibeocr.runtime_contracts import CancelMode

            return CancelMode.COOPERATIVE

    mod._executor = _Succeed()  # type: ignore[attr-defined]
    body = {"command_id": "c1", "kind": "retry", "job_id": ref.job_id}
    async with _http(supervisor_token, app) as http:
        resp = await http.post("/v2/jobs/command", json=body)
    assert resp.status_code == 200
    out = resp.json()
    assert out["kind"] == "retry"
    assert out["job_ref"]["job_id"] != ref.job_id
