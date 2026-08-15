"""Check the real Backend route table against the formal Runtime API v2 spec."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from fastapi.routing import APIRoute

ROOT = Path(__file__).resolve().parents[1]

backend_source = ROOT / "packages/vibeocr-backend/src"
if backend_source.is_dir():
    sys.path.insert(0, str(backend_source))

runtime_contracts = importlib.import_module("vibeocr.runtime_contracts")
V2 = Path(runtime_contracts.__file__).resolve().parent

from vibeocr.backend.supervisor.app import create_app  # noqa: E402
from vibeocr.backend.supervisor.module import (  # noqa: E402
    SupervisorModule,
    SupervisorOptions,
)

HTTP_METHODS = {"GET", "PUT", "POST", "DELETE", "PATCH", "HEAD", "OPTIONS", "TRACE"}

STABLE_ENGINE_IDS = ["rapidocr", "windows", "paddleocr"]


def _contains(actual: object, expected: object) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and actual == expected
    return actual == expected


def formal_operations() -> dict[tuple[str, str], str]:
    spec = json.loads(V2.joinpath("openapi.yaml").read_text(encoding="utf-8"))
    return {
        (method.upper(), path): operation["operationId"]
        for path, path_item in spec["paths"].items()
        for method, operation in path_item.items()
        if method.upper() in HTTP_METHODS
    }


def backend_app():
    temp = tempfile.TemporaryDirectory(prefix="vibeocr-protocol-conformance-")
    module = SupervisorModule(
        options=SupervisorOptions(instance_id="protocol-conformance"),
        stager_root=Path(temp.name),
        executor=MagicMock(),
    )
    app = create_app(module, "0" * 64)
    app.state._conformance_temp = temp
    return app


def backend_operations() -> dict[tuple[str, str], str]:
    app = backend_app()
    return {
        (method, route.path): str(route.operation_id)
        for route in app.routes
        if isinstance(route, APIRoute) and route.path.startswith("/v2/")
        for method in route.methods
        if method in HTTP_METHODS
    }


def check_conformance() -> None:
    formal = formal_operations()
    backend = backend_operations()
    if backend != formal:
        missing = sorted(formal.keys() - backend.keys())
        extra = sorted(backend.keys() - formal.keys())
        changed = sorted(
            key for key in formal.keys() & backend.keys() if formal[key] != backend[key]
        )
        raise RuntimeError(
            "Backend route table differs from formal Protocol v2: "
            f"missing={missing}, extra={extra}, operation_ids={changed}"
        )
    app = backend_app()
    formal_spec = json.loads(V2.joinpath("openapi.yaml").read_text(encoding="utf-8"))
    generated_spec = app.state.generated_openapi
    for (method, path), operation_id in formal.items():
        formal_operation = formal_spec["paths"][path][method.lower()]
        generated_operation = generated_spec["paths"][path][method.lower()]
        if generated_operation["operationId"] != operation_id:
            raise RuntimeError(f"generated operationId differs for {method} {path}")
        for key in ("requestBody", "security"):
            if not _contains(
                generated_operation.get(key),
                formal_operation.get(key),
            ):
                raise RuntimeError(
                    f"Backend generated OpenAPI {key} differs for {method} {path}"
                )
        for status, formal_response in formal_operation["responses"].items():
            generated_response = generated_operation["responses"].get(status)
            expected = formal_response
            if status == "200" and "application/json" in formal_response.get(
                "content", {}
            ):
                expected = {"content": formal_response["content"]}
            if not _contains(generated_response, expected):
                raise RuntimeError(
                    f"Backend generated OpenAPI response {status} differs "
                    f"for {method} {path}"
                )
    if app.openapi() != formal_spec:
        raise RuntimeError(
            "Backend does not serve the committed formal OpenAPI document"
        )


def _engine_selection_app():
    """app + 确定性引擎目录 fixture：rapidocr ready / windows unavailable /
    paddleocr preparation_required。"""
    from vibeocr.backend.supervisor.inference.ocr_engines import (
        EngineAvailability,
        EngineDescriptor,
        OcrEngineRegistry,
    )
    from vibeocr.runtime_contracts.dtos import OcrEngine

    class _Engine:
        def __init__(self, engine_id, availability, required_component=None):
            self.engine_id = engine_id
            self._availability = availability
            self._required_component = required_component

        def descriptor(self):
            return EngineDescriptor(
                engine_id=self.engine_id,
                availability=self._availability,
                included_in_base=self.engine_id is not OcrEngine.PADDLEOCR,
                reason_code=None
                if self._availability is EngineAvailability.READY
                else "engine_not_installed",
                required_component=self._required_component,
            )

    registry = OcrEngineRegistry(
        [
            _Engine(OcrEngine.RAPIDOCR, EngineAvailability.READY),
            _Engine(OcrEngine.WINDOWS, EngineAvailability.UNAVAILABLE),
            _Engine(
                OcrEngine.PADDLEOCR,
                EngineAvailability.PREPARATION_REQUIRED,
                required_component="full-cpu",
            ),
        ]
    )
    temp = tempfile.TemporaryDirectory(prefix="vibeocr-engine-conformance-")
    module = SupervisorModule(
        options=SupervisorOptions(instance_id="engine-conformance"),
        stager_root=Path(temp.name),
        executor=MagicMock(),
        engine_registry=registry,
    )
    app = create_app(module, "0" * 64)
    app.state._conformance_temp = temp
    return app


def check_engine_selection_conformance() -> None:
    """Engine selection conformance：formal schema + catalog + submit fixtures。"""
    spec = json.loads(V2.joinpath("openapi.yaml").read_text(encoding="utf-8"))
    schemas = spec["components"]["schemas"]

    # 1. formal spec：PipelineSelection 允许 engine，OcrEngineId 是稳定枚举。
    selection_properties = schemas["PipelineSelection"]["properties"]
    if "engine" not in selection_properties:
        raise RuntimeError("formal PipelineSelection lacks the engine property")
    engine_ids = schemas["OcrEngineId"]["enum"]
    if engine_ids != STABLE_ENGINE_IDS:
        raise RuntimeError(f"formal OcrEngineId enum differs: {engine_ids}")

    import asyncio

    asyncio.run(_engine_selection_http_checks())


async def _engine_selection_http_checks() -> None:
    import httpx

    app = _engine_selection_app()
    token = "0" * 64
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1",
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        # 2. ready envelope：catalog 覆盖全部稳定 id，descriptor 形状合法。
        health = (await client.get("/v2/health")).json()
        descriptor = next(
            d
            for d in health["capability_descriptors"]
            if d["name"] == "ocr.engine-selection.v1"
        )
        catalog = descriptor.get("ocr_engine_catalog")
        if catalog is None:
            raise RuntimeError("engine-selection descriptor lacks ocr_engine_catalog")
        entries = catalog["engines"]
        if [entry["id"] for entry in entries] != STABLE_ENGINE_IDS:
            raise RuntimeError(f"engine catalog ids differ: {entries}")
        for entry in entries:
            missing_keys = {
                "id",
                "availability",
                "included_in_base",
                "reason_code",
                "required_component",
            } - set(entry)
            if missing_keys:
                raise RuntimeError(
                    f"engine descriptor missing keys {sorted(missing_keys)}"
                )
            if entry["availability"] not in {
                "ready",
                "preparation_required",
                "unavailable",
            }:
                raise RuntimeError(f"invalid availability: {entry['availability']}")

        # 3. submit fixtures：显式 engine 进入 job；未知值与跨 pipeline 拒绝。
        def _manifest(engine=None, pipeline="OCR"):
            pipeline_obj: dict[str, Any] = {
                "pipeline_id": pipeline,
                "options_version": 1,
                "options": {},
            }
            if engine is not None:
                pipeline_obj["engine"] = engine
            return json.dumps(
                {
                    "schema_version": 2,
                    "request_id": "conformance-engine",
                    "kind": "recognition",
                    "priority": "interactive",
                    "pipeline": pipeline_obj,
                    "items": [
                        {
                            "client_item_key": "k",
                            "ordinal": 0,
                            "display_name": "a.png",
                            "source": {"type": "upload.v1", "attachment": "f"},
                        }
                    ],
                }
            )

        async def _submit(manifest: str) -> httpx.Response:
            return await client.post(
                "/v2/jobs",
                data={"manifest": manifest},
                files={"f": ("a.png", b"png", "image/png")},
            )

        ok = await _submit(_manifest(engine="rapidocr"))
        if ok.status_code != 200:
            raise RuntimeError(
                f"engine submit fixture failed: {ok.status_code} {ok.text}"
            )

        unknown = await _submit(_manifest(engine="paddle"))
        if unknown.status_code != 400 or unknown.json()["code"] != "OCR_ENGINE_UNKNOWN":
            raise RuntimeError("unknown engine fixture did not fail closed")

        wrong_pipeline = await _submit(
            _manifest(engine="rapidocr", pipeline="TABLE_RECOGNITION")
        )
        if (
            wrong_pipeline.status_code != 400
            or wrong_pipeline.json()["code"] != "OCR_ENGINE_NOT_VALID_FOR_PIPELINE"
        ):
            raise RuntimeError("engine on non-OCR pipeline fixture did not reject")

        unavailable = await _submit(_manifest(engine="windows"))
        if (
            unavailable.status_code != 426
            or unavailable.json()["code"] != "OCR_ENGINE_UNAVAILABLE"
        ):
            raise RuntimeError("unavailable engine fixture did not return 426")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    check_conformance()
    check_engine_selection_conformance()
    print("Backend Runtime API v2 conformance: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
