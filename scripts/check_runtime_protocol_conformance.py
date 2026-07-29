"""Check the real Backend route table against the formal Runtime API v2 spec."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    check_conformance()
    print("Backend Runtime API v2 conformance: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
