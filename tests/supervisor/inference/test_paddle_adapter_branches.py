"""Tests for the residency / capability / preload branches of PaddlePipelineAdapter.

Existing test_paddle_adapter.py covers the happy path; this file targets the
branches left behind: real-batch detection failure, preload raise + empty
selection, residency/release/configure/close with a real cache manager, and
the non-RGB image conversion path.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from PIL import Image
from vibeocr.backend.supervisor.inference.budgets import InputItem
from vibeocr.backend.supervisor.inference.paddle_adapter import PaddlePipelineAdapter
from vibeocr.runtime_contracts import PipelineSpec, SettingsSnapshot


def _png_bytes(mode: str = "RGB", color: tuple = (255, 0, 0)) -> bytes:
    import io

    img = Image.new(mode, (4, 4), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _FakeService:
    """Mirrors the OCRService surface; carries an optional cache_manager."""

    def __init__(
        self,
        *,
        cache_manager: Any | None = None,
        preload_results: dict[str, bool] | None = None,
        preload_raises: bool = False,
    ) -> None:
        self.calls: list[int] = []
        self.cache_manager = cache_manager
        self._preload_results = preload_results if preload_results is not None else {}
        self._preload_raises = preload_raises

    def recognize_batch(self, images, options=None):  # type: ignore[no-untyped-def]
        self.calls.append(len(images))
        return [{"text": "ok"} for _ in images]

    def preload_pipelines_sequential(self, pipelines):  # type: ignore[no-untyped-def]
        if self._preload_raises:
            raise RuntimeError("preload failed")
        return dict(self._preload_results)


def _raw_item(data: bytes) -> InputItem:
    return InputItem(
        item_id="it-0",
        data=data,
        encoded_bytes=len(data),
        decoded_pixels=16,
        estimated_pages=1,
    )


# ---------------------------------------------------------------------------
# Capability: real-batch detection failure
# ---------------------------------------------------------------------------


def test_capability_real_batch_detection_swallows_registry_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the pipeline registry raises, _pipeline_supports_real_batch returns False
    (lines 92-95)."""
    # Inject a fake registry module whose get_registry raises.
    import sys
    import types

    fake_pipelines = types.ModuleType("vibeocr.backend.core.pipelines")

    def raising_get_registry():  # type: ignore[no-untyped-def]
        raise RuntimeError("registry broken")

    fake_pipelines.get_registry = raising_get_registry  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vibeocr.backend.core.pipelines", fake_pipelines)

    adapter = PaddlePipelineAdapter(service=_FakeService(), pipeline_name="OCR")
    assert adapter._pipeline_supports_real_batch("OCR") is False


def test_capability_real_batch_detection_returns_true_when_registry_has_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the registry exposes a real recognize_batch, the probe returns True."""
    import sys
    import types

    class _Spec:
        recognize_batch = staticmethod(lambda images, options=None: [])

    class _Reg:
        def has(self, name):  # type: ignore[no-untyped-def]
            return True

        def get(self, name):  # type: ignore[no-untyped-def]
            return _Spec()

    fake_pipelines = types.ModuleType("vibeocr.backend.core.pipelines")
    fake_pipelines.get_registry = lambda: _Reg()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vibeocr.backend.core.pipelines", fake_pipelines)

    adapter = PaddlePipelineAdapter(service=_FakeService(), pipeline_name="OCR")
    # The cache must be bypassed: use a fresh pipeline name.
    cap = adapter.capabilities(
        __import__(
            "vibeocr.runtime_contracts", fromlist=["PipelineSelection"]
        ).PipelineSelection("OCR")
    )
    assert cap.real_batch is True


# ---------------------------------------------------------------------------
# Preload: raise + empty selection
# ---------------------------------------------------------------------------


def test_preload_logs_failure_when_service_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A preload failure logs an exception per pipeline and re-raises
    (lines 184-193)."""
    service = _FakeService(preload_raises=True)
    adapter = PaddlePipelineAdapter(service=service, pipeline_name="OCR")

    with (
        caplog.at_level(
            logging.ERROR,
            logger="vibeocr.backend.supervisor.inference.paddle_adapter",
        ),
        pytest.raises(RuntimeError, match="preload failed"),
    ):
        adapter.preload(("OCR",))

    assert any(
        "pipeline=OCR" in record.getMessage() and "result=failed" in record.getMessage()
        for record in caplog.records
    )


def test_preload_skips_when_no_paddle_pipeline_selected() -> None:
    """A preload request with no Paddle pipeline short-circuits (line 180 False)."""
    service = _FakeService(preload_results={"OCR": True})
    adapter = PaddlePipelineAdapter(service=service, pipeline_name="OCR")
    # MinerU is not a Paddle pipeline.
    status = adapter.preload(("MinerU",))
    # No preload call made; just the current residency returned.
    assert service.calls == []
    assert status.default_ttl_seconds == 300


# ---------------------------------------------------------------------------
# Residency status: cache_manager without ``status`` + with full status
# ---------------------------------------------------------------------------


def test_residency_status_when_cache_manager_lacks_status() -> None:
    """A cache_manager without a ``status`` method falls back to the settings
    snapshot (line 232->235)."""
    service = _FakeService(cache_manager=object())  # no status() attr
    adapter = PaddlePipelineAdapter(service=service, pipeline_name="OCR")
    status = adapter.residency_status()
    assert status.default_ttl_seconds == 300
    assert status.entries == ()


def test_release_idle_calls_release_when_manager_present_no_pipeline() -> None:
    """release_idle(pipeline=None) calls manager.release(heavy_only=False)
    (lines 254-260)."""
    released: list[dict] = []

    class _Manager:
        def release(self, *, heavy_only=False, force=False):  # type: ignore[no-untyped-def]
            released.append({"heavy_only": heavy_only, "force": force})

        def release_one(self, pipeline):  # type: ignore[no-untyped-def]
            released.append({"release_one": pipeline})

        def status(self):  # type: ignore[no-untyped-def]
            return {
                "loaded_pipelines": [],
                "pipeline_ttls": {},
                "active_counts": {},
                "pinned_pipelines": [],
                "last_used_unix_ms": {},
            }

        def shutdown(self) -> None:
            return

    service = _FakeService(cache_manager=_Manager())
    adapter = PaddlePipelineAdapter(service=service, pipeline_name="OCR")
    adapter.release_idle()  # pipeline=None
    assert released == [{"heavy_only": False, "force": False}]


def test_release_idle_calls_release_one_when_pipeline_given() -> None:
    """release_idle(pipeline='OCR') calls manager.release_one('OCR')."""
    released: list[Any] = []

    class _Manager:
        def release(self, *, heavy_only=False, force=False):  # type: ignore[no-untyped-def]
            released.append("release")

        def release_one(self, pipeline):  # type: ignore[no-untyped-def]
            released.append(f"release_one:{pipeline}")

        def status(self):  # type: ignore[no-untyped-def]
            return {
                "loaded_pipelines": [],
                "pipeline_ttls": {},
                "active_counts": {},
                "pinned_pipelines": [],
                "last_used_unix_ms": {},
            }

        def shutdown(self) -> None:
            return

    service = _FakeService(cache_manager=_Manager())
    adapter = PaddlePipelineAdapter(service=service, pipeline_name="OCR")
    adapter.release_idle("OCR")
    assert released == ["release_one:OCR"]


def test_configure_settings_propagates_to_manager_when_present() -> None:
    """configure_settings calls manager.configure_residency when present (line 266)."""
    configured: list[dict] = []

    class _Manager:
        def configure_residency(self, *, default_ttl_seconds, pipelines):  # type: ignore[no-untyped-def]
            configured.append(
                {"ttl": default_ttl_seconds, "pipelines": list(pipelines)}
            )

    service = _FakeService(cache_manager=_Manager())
    adapter = PaddlePipelineAdapter(service=service, pipeline_name="OCR")
    spec = PipelineSpec(name="OCR")
    snap = SettingsSnapshot(
        default_ttl_seconds=600,
        pipelines=(spec,),
    )
    adapter.configure_settings(snap)
    assert configured == [{"ttl": 600, "pipelines": [spec]}]


def test_close_releases_and_shuts_down_manager_when_present() -> None:
    """close() calls manager.release(force=True) + manager.shutdown() (lines 272-276)."""
    events: list[str] = []

    class _Manager:
        def release(self, *, heavy_only=False, force=False):  # type: ignore[no-untyped-def]
            events.append(f"release:force={force}")

        def shutdown(self) -> None:
            events.append("shutdown")

    service = _FakeService(cache_manager=_Manager())
    adapter = PaddlePipelineAdapter(service=service, pipeline_name="OCR")
    adapter.close()
    assert events == ["release:force=True", "shutdown"]


def test_close_is_noop_when_no_manager() -> None:
    service = _FakeService(cache_manager=None)
    adapter = PaddlePipelineAdapter(service=service, pipeline_name="OCR")
    adapter.close()  # must not raise


# ---------------------------------------------------------------------------
# Image conversion: non-RGB mode
# ---------------------------------------------------------------------------


def test_recognize_many_converts_non_rgb_image_to_rgb() -> None:
    """An RGBA (non-RGB) PNG triggers the convert-to-RGB branch (line 303)."""
    service = _FakeService()
    adapter = PaddlePipelineAdapter(service=service, pipeline_name="OCR")
    rgba_png = _png_bytes(mode="RGBA", color=(255, 0, 0, 255))
    results = adapter.recognize_many([_raw_item(rgba_png)])
    assert results == [{"text": "ok"}]
    assert service.calls == [1]  # one image processed


# ---------------------------------------------------------------------------
# residency_status: pinned + soft-ttl entries
# ---------------------------------------------------------------------------


def test_residency_status_marks_pinned_and_soft_ttl_entries() -> None:
    """A full status dict yields PINNED + SOFT_TTL entries with remaining TTL."""

    class _Manager:
        def status(self):  # type: ignore[no-untyped-def]
            return {
                "loaded_pipelines": ["OCR", "PP-StructureV3"],
                "pipeline_ttls": {"OCR": 0, "PP-StructureV3": 100},
                "active_counts": {"OCR": 1},
                "pinned_pipelines": ["OCR"],
                "last_used_unix_ms": {},
            }

    service = _FakeService(cache_manager=_Manager())
    adapter = PaddlePipelineAdapter(service=service, pipeline_name="OCR")
    status = adapter.residency_status()
    by_name = {e.pipeline: e for e in status.entries}
    from vibeocr.runtime_contracts import ResidencyKind

    assert by_name["OCR"].kind is ResidencyKind.PINNED
    assert by_name["OCR"].active_leases == 1
    assert by_name["OCR"].remaining_ttl_seconds is None  # pinned
    assert by_name["PP-StructureV3"].kind is ResidencyKind.SOFT_TTL
    assert by_name["PP-StructureV3"].remaining_ttl_seconds is not None
