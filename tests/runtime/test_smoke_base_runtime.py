from __future__ import annotations

from pathlib import Path

import pytest

from scripts import smoke_base_runtime
from scripts.smoke_base_runtime import (
    BaseRuntimeSmokeError,
    _base_ensure_request,
    _offline_environment,
)


def test_base_smoke_forces_offline_environment(monkeypatch) -> None:
    monkeypatch.setenv("PIP_INDEX_URL", "https://example.invalid/simple")
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://example.invalid/extra")

    environment = _offline_environment()

    assert environment["PIP_NO_INDEX"] == "1"
    assert environment["UV_OFFLINE"] == "1"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["HTTP_PROXY"] == "http://127.0.0.1:9"
    assert environment["NO_PROXY"] == "127.0.0.1,localhost"
    assert "PIP_INDEX_URL" not in environment
    assert "PIP_EXTRA_INDEX_URL" not in environment


def test_base_smoke_ensure_request_explicitly_selects_no_optional_components(
    tmp_path: Path,
) -> None:
    request = _base_ensure_request(
        product_root=tmp_path / "product",
        component_lock=tmp_path / "component-lock.json",
        runtime_manifest=tmp_path / "runtime-manifest.json",
    )

    assert request["operation"] == "ensure"
    assert request["accelerator"] == "cpu"
    assert request["install_component_ids"] == []


def test_rapidocr_timeout_reports_last_job_state(monkeypatch) -> None:
    clock = iter((0.0, 0.0, 121.0))
    monkeypatch.setattr(smoke_base_runtime.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(smoke_base_runtime.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        smoke_base_runtime,
        "_json_request",
        lambda *_args, **_kwargs: {
            "snapshot": {"state": "running"},
            "events": [
                {"sequence": 3, "event_type": "batch_plan"},
            ],
        },
    )

    with pytest.raises(BaseRuntimeSmokeError) as exc_info:
        smoke_base_runtime._wait_for_ocr("http://127.0.0.1:1", "token", "job")

    message = str(exc_info.value)
    assert '"state": "running"' in message
    assert '"event_type": "batch_plan"' in message
