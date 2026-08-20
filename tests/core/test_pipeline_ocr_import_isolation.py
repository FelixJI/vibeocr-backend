from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from vibeocr.backend.core.pipelines.pipeline_ocr import (
    _create_ocr_pipeline,
    _isolate_unused_modelscope_import,
)
from vibeocr.backend.core.pipelines.pipeline_paddlocr_vl import (
    _create_paddlocr_vl_pipeline,
)
from vibeocr.backend.core.pipelines.pipeline_pp_structure import (
    _create_pp_structure_pipeline,
)


def test_create_ocr_pipeline_consumes_verified_local_model_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    binding = tmp_path / "resolved-models.json"
    binding.write_text(
        json.dumps(
            {"consumers": {"paddleocr": {"text_recognition_model_dir": str(model_dir)}}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VIBEOCR_RESOLVED_MODELS", str(binding))
    captured: dict[str, object] = {}

    def paddle_ocr(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        types.SimpleNamespace(PaddleOCR=paddle_ocr),
    )

    _create_ocr_pipeline("cpu")

    assert captured["device"] == "cpu"
    assert captured["text_recognition_model_dir"] == str(model_dir)


@pytest.mark.parametrize(
    ("factory", "consumer", "constructor_name", "binding_key"),
    [
        (
            _create_pp_structure_pipeline,
            "pp_structure",
            "PPStructureV3",
            "layout_detection_model_dir",
        ),
        (
            _create_paddlocr_vl_pipeline,
            "paddleocr_vl",
            "PaddleOCRVL",
            "vl_rec_model_dir",
        ),
    ],
)
def test_document_pipeline_factories_consume_verified_local_model_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory: object,
    consumer: str,
    constructor_name: str,
    binding_key: str,
) -> None:
    model_dir = tmp_path / consumer
    model_dir.mkdir()
    binding = tmp_path / "resolved-models.json"
    binding.write_text(
        json.dumps({"consumers": {consumer: {binding_key: str(model_dir)}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("VIBEOCR_RESOLVED_MODELS", str(binding))
    captured: dict[str, object] = {}

    def constructor(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        types.SimpleNamespace(**{constructor_name: constructor}),
    )

    assert callable(factory)
    factory("cpu")

    assert captured == {"device": "cpu", binding_key: str(model_dir)}


def test_non_modelscope_source_installs_non_torch_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PADDLE_PDX_MODEL_SOURCE", "huggingface")
    monkeypatch.delitem(sys.modules, "modelscope", raising=False)

    assert _isolate_unused_modelscope_import() is True

    placeholder = sys.modules["modelscope"]
    with pytest.raises(RuntimeError, match="disabled in the Paddle process"):
        placeholder.snapshot_download("unused")  # type: ignore[attr-defined]


def test_explicit_modelscope_source_is_never_stubbed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PADDLE_PDX_MODEL_SOURCE", "modelscope")
    monkeypatch.delitem(sys.modules, "modelscope", raising=False)

    assert _isolate_unused_modelscope_import() is False
    assert "modelscope" not in sys.modules


def test_existing_modelscope_module_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setenv("PADDLE_PDX_MODEL_SOURCE", "huggingface")
    monkeypatch.setitem(sys.modules, "modelscope", sentinel)  # type: ignore[arg-type]

    assert _isolate_unused_modelscope_import() is False
    assert sys.modules["modelscope"] is sentinel
