from __future__ import annotations

import sys

import pytest
from vibeocr.backend.core.pipelines.pipeline_ocr import (
    _isolate_unused_modelscope_import,
)


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
