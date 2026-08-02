"""Tests for the supervisor PDF v2 routes (plan §6).

Drives the real FastAPI app over httpx ASGI transport with a fake
``pdf_adapter`` injected via the shared ``pdf_module`` fixture in conftest.
Covers: open/close/model/load stream/render thumbnail+preview/detect/mutate/
text-layer/save/cancel/reset_cancel, plus 503 when no adapter is wired and
400 on bad JSON.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

# pytest puts conftest.py on sys.path for each test session, so importing the
# shared NullExecutor from it is safe at runtime.
from conftest import NullExecutor
from vibeocr.backend.supervisor.app import create_app
from vibeocr.backend.supervisor.bootstrap import new_instance_id
from vibeocr.backend.supervisor.module import SupervisorModule, SupervisorOptions

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import FakePdfAdapter
    from fastapi import FastAPI


def _http(token: str, app) -> httpx.AsyncClient:
    """Build a raw httpx client over ASGI transport with the bearer token."""
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1",
        headers={"Authorization": f"Bearer {token}"},
    )


# ---------------------------------------------------------------------------
# 503 when no adapter
# ---------------------------------------------------------------------------


async def test_pdf_route_returns_internal_error_when_no_adapter(
    tmp_path: Path, supervisor_token: str
) -> None:
    opts = SupervisorOptions(instance_id=new_instance_id())
    module = SupervisorModule(
        options=opts,
        stager_root=tmp_path / "staging",
        executor=NullExecutor(),
        pdf_adapter=None,
    )
    app = create_app(module, supervisor_token)
    async with _http(supervisor_token, app) as http:
        resp = await http.post("/v2/pdf/sessions/open", json={"path": "x.pdf"})
    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "INTERNAL_ERROR"


# ---------------------------------------------------------------------------
# Lifecycle: open / close / model
# ---------------------------------------------------------------------------


async def test_open_returns_session_id_and_model(
    pdf_app: FastAPI, supervisor_token: str, fake_pdf_adapter: FakePdfAdapter
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post("/v2/pdf/sessions/open", json={"path": "doc.pdf"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "sid-1"
    assert body["schema_version"] == 2
    assert body["model"]["file_path"] == "doc.pdf"
    assert fake_pdf_adapter.calls[-1][0] == "open_session"


async def test_close_proxies_and_returns_closed_flag(
    pdf_app: FastAPI, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post("/v2/pdf/sessions/sid-1/close")
    assert resp.status_code == 200
    assert resp.json()["closed"] is True


async def test_model_returns_full_mirror(
    pdf_app: FastAPI, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post("/v2/pdf/sessions/sid-1/model")
    assert resp.status_code == 200
    assert resp.json()["pages"] == []


# ---------------------------------------------------------------------------
# Streaming load / delete_text_layers
# ---------------------------------------------------------------------------


async def test_load_streams_ndjson_progress_events(
    pdf_app: FastAPI, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post("/v2/pdf/sessions/sid-1/load")
    assert resp.status_code == 200
    lines = [ln for ln in resp.text.splitlines() if ln]
    assert len(lines) == 1
    import json

    event = json.loads(lines[0])
    assert event["phase"] == "load"
    assert event["message"] == "done"


async def test_delete_text_layers_streams_ndjson(
    pdf_app: FastAPI, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/pdf/sessions/sid-1/delete_text_layers", json={"pages": [0, 1]}
        )
    assert resp.status_code == 200
    lines = [ln for ln in resp.text.splitlines() if ln]
    assert len(lines) == 1
    import json

    event = json.loads(lines[0])
    assert event["phase"] == "delete"


# ---------------------------------------------------------------------------
# Render (binary PNG)
# ---------------------------------------------------------------------------


async def test_render_thumbnail_returns_png_bytes(
    pdf_app: FastAPI, supervisor_token: str, fake_pdf_adapter: FakePdfAdapter
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/pdf/sessions/sid-1/render_thumbnail",
            json={"page": 0, "size": 128},
        )
    assert resp.status_code == 200
    assert resp.content.startswith(b"\x89PNG")
    name, _args, kwargs = fake_pdf_adapter.calls[-1]
    assert name == "render_thumbnail"
    assert kwargs == {"size": 128}


async def test_render_preview_returns_png_bytes_and_forwards_dpi(
    pdf_app: FastAPI, supervisor_token: str, fake_pdf_adapter: FakePdfAdapter
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/pdf/sessions/sid-1/render_preview",
            json={"page": 2, "dpi": 300},
        )
    assert resp.status_code == 200
    assert resp.content.startswith(b"\x89PNG")
    name, _args, kwargs = fake_pdf_adapter.calls[-1]
    assert name == "render_preview"
    assert kwargs == {"dpi": 300}


async def test_render_get_route_returns_png_and_reads_query_params(
    pdf_app: FastAPI, supervisor_token: str, fake_pdf_adapter: FakePdfAdapter
) -> None:
    """The .NET client issues GET /render?page=&size= for quick previews."""
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.get(
            "/v2/pdf/sessions/sid-1/render", params={"page": 3, "size": 96}
        )
    assert resp.status_code == 200
    assert resp.content.startswith(b"\x89PNG")
    name, args, kwargs = fake_pdf_adapter.calls[-1]
    assert name == "render_thumbnail"
    assert args == ("sid-1", 3)
    assert kwargs == {"size": 96}


async def test_save_transactional_returns_path_and_calls_adapter(
    pdf_app: FastAPI, supervisor_token: str, fake_pdf_adapter: FakePdfAdapter
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/pdf/sessions/sid-1/save_transactional",
            json={"path": "/tmp/out.pdf"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["path"] == "/tmp/out.pdf"
    name, args, _kwargs = fake_pdf_adapter.calls[-1]
    assert name == "save_transactional"
    assert args == ("sid-1", "/tmp/out.pdf")


async def test_save_transactional_rejects_missing_path(
    pdf_app: FastAPI, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post("/v2/pdf/sessions/sid-1/save_transactional", json={})
    assert resp.status_code == 400
    assert resp.json()["code"] == "VALIDATION_ERROR"


async def test_rotate_proxies_pages_and_angle(
    pdf_app: FastAPI, supervisor_token: str, fake_pdf_adapter: FakePdfAdapter
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/pdf/sessions/sid-1/rotate", json={"pages": [0, 1], "angle": 90}
        )
    assert resp.status_code == 200
    name, args, _kwargs = fake_pdf_adapter.calls[-1]
    assert name == "rotate"
    assert args == ("sid-1", [0, 1], 90)


async def test_delete_pages_marks_structural_change(
    pdf_app: FastAPI, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/pdf/sessions/sid-1/delete_pages", json={"pages": [2]}
        )
    assert resp.status_code == 200
    assert resp.json()["diff"]["structural_change"] is True


@pytest.mark.parametrize(
    "op,body,expected_name",
    [
        (
            "insert_blank",
            {"after_index": 0, "width": 100.0, "height": 200.0},
            "insert_blank",
        ),
        ("insert_from", {"source_path": "src.pdf", "after_index": 0}, "insert_from"),
        ("move_page", {"from_index": 0, "to_index": 1}, "move_page"),
        ("reorder", {"new_order": [1, 0]}, "reorder"),
    ],
)
async def test_structural_mutations_proxy_body(
    pdf_app: FastAPI,
    supervisor_token: str,
    fake_pdf_adapter: FakePdfAdapter,
    op: str,
    body: dict,
    expected_name: str,
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(f"/v2/pdf/sessions/sid-1/{op}", json=body)
    assert resp.status_code == 200
    assert fake_pdf_adapter.calls[-1][0] == expected_name


# ---------------------------------------------------------------------------
# Text layer
# ---------------------------------------------------------------------------


async def test_add_text_layer_batch_returns_saved_extra(
    pdf_app: FastAPI, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/pdf/sessions/sid-1/add_text_layer_batch",
            json={
                "pages": [{"page": 0, "ocr_result": {"text": "hi"}}],
                "save": True,
            },
        )
    assert resp.status_code == 200
    assert resp.json()["extra"]["saved"] is True


async def test_update_block_text_proxies_text(
    pdf_app: FastAPI, supervisor_token: str, fake_pdf_adapter: FakePdfAdapter
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/pdf/sessions/sid-1/update_block_text",
            json={"page": 0, "block_index": 1, "new_text": "edited"},
        )
    assert resp.status_code == 200
    name, args, _kwargs = fake_pdf_adapter.calls[-1]
    assert name == "update_block_text"
    assert args == ("sid-1", 0, 1, "edited")


# ---------------------------------------------------------------------------
# Save + cancel
# ---------------------------------------------------------------------------


async def test_save_returns_path_and_forwards_rewrite_flag(
    pdf_app: FastAPI, supervisor_token: str, fake_pdf_adapter: FakePdfAdapter
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/pdf/sessions/sid-1/save",
            json={"path": "out.pdf", "rewrite_text_layers": False},
        )
    assert resp.status_code == 200
    assert resp.json()["path"] == "out.pdf"
    name, _args, kwargs = fake_pdf_adapter.calls[-1]
    assert name == "save"
    assert kwargs == {"rewrite_text_layers": False}


async def test_cancel_and_reset_cancel_toggle_state(
    pdf_app: FastAPI, supervisor_token: str, fake_pdf_adapter: FakePdfAdapter
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        await http.post("/v2/pdf/sessions/sid-1/cancel")
        await http.post("/v2/pdf/sessions/sid-1/reset_cancel")
    names = [c[0] for c in fake_pdf_adapter.calls]
    assert "cancel" in names
    assert "reset_cancel" in names


# ---------------------------------------------------------------------------
# Bad input
# ---------------------------------------------------------------------------


async def test_open_without_path_returns_validation_error(
    pdf_app: FastAPI, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post("/v2/pdf/sessions/open", json={})
    assert resp.status_code == 400
    assert resp.json()["code"] == "VALIDATION_ERROR"


async def test_open_with_invalid_json_returns_validation_error(
    pdf_app: FastAPI, supervisor_token: str
) -> None:
    async with _http(supervisor_token, pdf_app) as http:
        resp = await http.post(
            "/v2/pdf/sessions/open",
            content=b"not json",
            headers={"Authorization": f"Bearer {supervisor_token}"},
        )
    assert resp.status_code == 400
