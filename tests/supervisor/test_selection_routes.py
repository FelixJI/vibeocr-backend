"""component/source selection 的 HTTP 面契约（Protocol 2.8.0）。"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from vibeocr.backend.runtime_installer import (
    RuntimeInstallError,
    _request,
)
from vibeocr.backend.runtime_selection import (
    component_variant_catalog_payload,
    download_source_catalog_payload,
)
from vibeocr.backend.supervisor.app import create_app


class _RecordingControl:
    def __init__(self) -> None:
        self.execute_calls: list[dict[str, Any]] = []
        self.command_calls: list[dict[str, Any]] = []

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        self.execute_calls.append(kwargs)
        return {
            "schema_version": 2,
            "operation_id": kwargs.get("operation_id") or "op-1",
            "snapshot": {
                "operation_id": kwargs.get("operation_id") or "op-1",
                "sequence": 1,
                "operation": kwargs["operation"],
                "operation_state": "succeeded",
                "phase": "commit_runtime",
                "profile_id": "win-x64-cpu",
                "updated_at": "2026-08-17T00:00:00Z",
                "effective_download_source_ids": ["pypi"],
            },
            "negotiated_capabilities": [],
        }

    def command(self, **kwargs: Any) -> dict[str, Any]:
        self.command_calls.append(kwargs)
        return {
            "schema_version": 2,
            "operation_id": kwargs.get("new_operation_id") or "op-2",
            "snapshot": {
                "operation_id": kwargs.get("new_operation_id") or "op-2",
                "sequence": 1,
                "operation": "ensure",
                "operation_state": "succeeded",
                "phase": "commit_runtime",
                "profile_id": "win-x64-cpu",
                "updated_at": "2026-08-17T00:00:00Z",
            },
            "negotiated_capabilities": [],
        }

    def observe(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("not used in these fixtures")


def _client(module, token: str, control: _RecordingControl) -> httpx.AsyncClient:
    app = create_app(module, token, runtime_control=control)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1",
        headers={"Authorization": f"Bearer {token}"},
    )


async def test_health_declares_selection_capability_catalogs(
    pdf_module, supervisor_token: str
) -> None:
    app = create_app(pdf_module, supervisor_token)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1",
        headers={"Authorization": f"Bearer {supervisor_token}"},
    ) as client:
        health = (await client.get("/v2/health")).json()

    descriptors = {d["name"]: d for d in health["capability_descriptors"]}
    source_descriptor = descriptors["runtime.download-sources.v1"]
    variant_descriptor = descriptors["runtime.component-selection.v1"]

    # catalog 结构与业务键唯一性（fail closed 的构建期防线在
    # runtime_selection 单测，这里钉住 HTTP 面的真实输出形状）。
    sources = source_descriptor["download_source_catalog"]["sources"]
    assert sources == download_source_catalog_payload()["sources"]
    assert len({s["id"] for s in sources}) == len(sources)
    assert {s["id"] for s in sources} == {
        "tuna-pypi",
        "pypi",
        "huggingface",
        "modelscope",
    }
    assert {s["kind"] for s in sources} == {"package_index", "model_registry"}

    variants = variant_descriptor["component_variant_catalog"]["variants"]
    assert variants == component_variant_catalog_payload()["variants"]
    keys = {(v["feature_id"], v["accelerator"]) for v in variants}
    assert len(keys) == len(variants)
    component_ids = {v["component_id"] for v in variants}
    assert {"ocr_engine", "pdf_document_tools"}.isdisjoint(component_ids)


async def test_settings_roundtrips_download_source_ids(
    pdf_module, supervisor_token: str
) -> None:
    app = create_app(pdf_module, supervisor_token)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1",
        headers={"Authorization": f"Bearer {supervisor_token}"},
    ) as client:
        put = await client.put(
            "/v2/settings",
            json={
                "schema_version": 2,
                "residency": {"default_ttl_seconds": 300, "pipelines": []},
                "extra": {},
                "download_source_ids": ["pypi"],
            },
        )
        assert put.status_code == 200
        assert put.json()["download_source_ids"] == ["pypi"]

        got = await client.get("/v2/settings")
        assert got.json().get("download_source_ids") == ["pypi"]

        same_kind = await client.put(
            "/v2/settings",
            json={
                "schema_version": 2,
                "residency": {"default_ttl_seconds": 300, "pipelines": []},
                "extra": {},
                "download_source_ids": ["tuna-pypi", "pypi"],
            },
        )
        assert same_kind.status_code == 400

        model_source = await client.put(
            "/v2/settings",
            json={
                "schema_version": 2,
                "residency": {"default_ttl_seconds": 300, "pipelines": []},
                "extra": {},
                "download_source_ids": ["pypi", "huggingface"],
            },
        )
        assert model_source.status_code == 200
        assert model_source.json()["download_source_ids"] == ["pypi", "huggingface"]

        unknown = await client.put(
            "/v2/settings",
            json={
                "schema_version": 2,
                "residency": {"default_ttl_seconds": 300, "pipelines": []},
                "extra": {},
                "download_source_ids": ["unknown-model-source"],
            },
        )
        assert unknown.status_code == 400
        assert unknown.json()["code"] == "DOWNLOAD_SOURCE_UNKNOWN"


async def test_maintenance_ensure_forwards_selection_fields(
    pdf_module, supervisor_token: str
) -> None:
    control = _RecordingControl()
    async with _client(pdf_module, supervisor_token, control) as client:
        started = await client.post(
            "/v2/runtime/maintenance",
            json={
                "operation": "ensure",
                "operation_id": "op-1",
                "install_component_ids": [],
                "download_source_ids": ["pypi"],
            },
        )
        assert started.status_code == 200
        body = started.json()
        assert body["snapshot"]["effective_download_source_ids"] == ["pypi"]

        # 请求省略源且 Settings 为空时保留 None，由 policy 采用 TUNA 缺省源。
        omitted = await client.post(
            "/v2/runtime/maintenance",
            json={"operation": "ensure", "operation_id": "op-1b"},
        )
        assert omitted.status_code == 200

        # schema 条件校验：inspect 携带选择字段 fail closed。
        inspect_rejected = await client.post(
            "/v2/runtime/maintenance",
            json={
                "operation": "inspect",
                "install_component_ids": [],
            },
        )
        assert inspect_rejected.status_code == 400
        assert inspect_rejected.json()["code"] == "VALIDATION_ERROR"

        retried = await client.post(
            "/v2/runtime/maintenance/command",
            json={
                "command_id": "retry-1",
                "command": "retry",
                "target_operation_id": "op-1",
                "new_operation_id": "op-2",
                "install_component_ids": ["document_parsing"],
                "download_source_ids": ["pypi"],
            },
        )
        assert retried.status_code == 200

    assert control.execute_calls[0]["install_component_ids"] == ()
    assert control.execute_calls[0]["download_source_ids"] == ("pypi",)
    assert control.execute_calls[1]["install_component_ids"] is None
    assert control.execute_calls[1]["download_source_ids"] is None
    assert control.command_calls[0]["install_component_ids"] == ("document_parsing",)
    assert control.command_calls[0]["download_source_ids"] == ("pypi",)


def _host_request(**overrides: Any) -> dict[str, Any]:
    request: dict[str, Any] = {
        "protocol_version": 2,
        "product_root": "C:/product",
        "component_lock": "C:/product/component-lock.json",
        "runtime_manifest": "C:/product/runtime-manifest.json",
    }
    if overrides.pop("command_kind", None) is None:
        request["request_kind"] = "start"
        request["operation"] = "ensure"
    else:
        request["request_kind"] = "command"
        request.update(
            {
                "command": "retry",
                "command_id": "c1",
                "target_operation_id": "op-1",
                "new_operation_id": "op-2",
            }
        )
    request.update(overrides)
    return request


def test_runtime_host_request_accepts_selection_fields_on_start_and_retry() -> None:
    parsed = _request(
        _host_request(
            install_component_ids=[],
            download_source_ids=["pypi"],
        )
    )
    assert parsed["install_component_ids"] == []
    assert parsed["download_source_ids"] == ["pypi"]

    command = _request(
        _host_request(
            command_kind="retry",
            install_component_ids=["document_parsing"],
        )
    )
    assert command["install_component_ids"] == ["document_parsing"]


def test_runtime_host_request_rejects_selection_outside_ensure_and_retry() -> None:
    with pytest.raises(RuntimeInstallError, match="require operation ensure"):
        _request(_host_request(operation="inspect", install_component_ids=[]))
    with pytest.raises(RuntimeInstallError, match="require command retry"):
        _request(
            _host_request(
                command_kind="retry",
                command="cancel",
                install_component_ids=[],
            )
        )
    with pytest.raises(RuntimeInstallError, match="download_source_ids is invalid"):
        _request(_host_request(download_source_ids=[]))
