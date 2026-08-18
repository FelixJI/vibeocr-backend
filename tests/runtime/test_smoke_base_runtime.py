from __future__ import annotations

from pathlib import Path

from scripts.smoke_base_runtime import _base_ensure_request, _offline_environment


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
