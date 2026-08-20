from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from vibeocr.backend.core.pipelines.pipeline_ocr import _create_ocr_pipeline
from vibeocr.backend.core.pipelines.pipeline_paddlocr_vl import (
    _create_paddlocr_vl_pipeline,
)
from vibeocr.backend.core.pipelines.pipeline_pp_structure import (
    _create_pp_structure_pipeline,
)


def test_create_ocr_pipeline_delegates_model_discovery_to_paddle(
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
    monkeypatch.setenv("PADDLE_PDX_MODEL_SOURCE", "huggingface")
    monkeypatch.delitem(sys.modules, "modelscope", raising=False)
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

    assert captured == {"device": "cpu"}
    assert "modelscope" not in sys.modules


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
def test_document_pipeline_factories_delegate_model_discovery_to_paddle(
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

    assert captured == {"device": "cpu"}
