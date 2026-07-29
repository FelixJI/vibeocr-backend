"""Tests for the composition root: build_supervisor + adapter builders.

The composition root wires the SupervisorModule with real or null executors.
Most ``_build_*`` helpers lazily import heavy services (OCRService,
MinerUService, PdfBackendClient); we mock those modules so the wiring logic
is covered without pulling paddle/torch/mineru.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

from vibeocr.backend.supervisor import composition
from vibeocr.backend.supervisor.composition import (
    _mineru_available,
    _MinerUServiceLifecycle,
    _NullExecutor,
    _paddle_available,
    build_supervisor,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# _NullExecutor
# ---------------------------------------------------------------------------


def _record_factory(monkeypatch, *, state) -> Any:
    """Build a minimal record-like object exposing the surface _NullExecutor uses."""
    calls: list[str] = []

    class _Rec:
        def __init__(self) -> None:
            self.state = state

        def transition(self, target):  # type: ignore[no-untyped-def]
            calls.append(f"transition:{target.value if hasattr(target, 'value') else target}")
            self.state = target

        def append_event(self, stage, *, detail=None):  # type: ignore[no-untyped-def]
            calls.append(f"event:{stage}")

    return _Rec(), calls


def test_null_executor_fails_non_terminal_job() -> None:
    from vibeocr.runtime_contracts import JobState

    rec, calls = _record_factory(None, state=JobState.QUEUED)
    _NullExecutor().execute(rec, [])
    assert any("transition:failed" in c for c in calls)
    assert any("event:no_backend" in c for c in calls)


def test_null_executor_skips_already_terminal_job() -> None:
    from vibeocr.runtime_contracts import JobState

    rec, calls = _record_factory(None, state=JobState.COMPLETED)
    _NullExecutor().execute(rec, [])
    assert calls == []


def test_null_executor_passthrough_methods_return_defaults() -> None:
    from vibeocr.runtime_contracts import CancelMode, ResidencyStatus, SettingsSnapshot

    ex = _NullExecutor()
    assert ex.cancel_mode_for(None) is CancelMode.COOPERATIVE
    assert ex.residency_status() == ResidencyStatus()
    assert ex.release_idle() == ResidencyStatus()
    assert ex.release_idle("OCR") == ResidencyStatus()
    assert ex.preload(("OCR",)) == ResidencyStatus()
    snap = SettingsSnapshot(default_ttl_seconds=600)
    configured = ex.configure_settings(snap)
    assert configured.default_ttl_seconds == 600
    ex.close()  # no-op, must not raise


def test_null_executor_swallows_transition_error(monkeypatch) -> None:
    """The defensive except in execute() must not leak a transition failure."""
    from vibeocr.runtime_contracts import JobState

    class _BrokenRec:
        state = JobState.QUEUED

        def transition(self, target):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

        def append_event(self, *a, **k):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    # Must not raise.
    _NullExecutor().execute(_BrokenRec(), [])


# ---------------------------------------------------------------------------
# build_supervisor — null path
# ---------------------------------------------------------------------------


def test_build_supervisor_uses_null_executor_without_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(composition, "_paddle_available", lambda: False)
    monkeypatch.setattr(composition, "_mineru_available", lambda: False)
    module, handle = build_supervisor(
        instance_id="comp-test", stager_root=tmp_path / "stage"
    )
    assert isinstance(module._executor, _NullExecutor)
    assert handle.token  # bootstrap handle should now carry a token


def test_build_supervisor_keeps_existing_bootstrap_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vibeocr.backend.supervisor.bootstrap import BootstrapHandle

    monkeypatch.setattr(composition, "_paddle_available", lambda: False)
    monkeypatch.setattr(composition, "_mineru_available", lambda: False)
    pre_handle = BootstrapHandle("pre-existing-token")
    _, handle = build_supervisor(
        instance_id="comp-test",
        stager_root=tmp_path / "stage",
        bootstrap_handle=pre_handle,
    )
    assert handle.token == "pre-existing-token"


def test_build_supervisor_explicit_executor_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Sentinel:
        def close(self) -> None:
            return

    sentinel = _Sentinel()
    monkeypatch.setattr(composition, "_paddle_available", lambda: True)
    module, _ = build_supervisor(
        instance_id="comp-test",
        stager_root=tmp_path / "stage",
        executor=sentinel,
    )
    assert module._executor is sentinel


def test_build_supervisor_picks_composite_when_paddle_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, bool] = {}

    def fake_composite(*, use_paddle, use_mineru):  # type: ignore[no-untyped-def]
        captured.update(use_paddle=use_paddle, use_mineru=use_mineru)
        return _NullExecutor()

    monkeypatch.setattr(composition, "_paddle_available", lambda: True)
    monkeypatch.setattr(composition, "_mineru_available", lambda: False)
    monkeypatch.setattr(composition, "_build_composite_executor", fake_composite)
    build_supervisor(instance_id="comp-test", stager_root=tmp_path / "stage")
    assert captured == {"use_paddle": True, "use_mineru": False}


def test_build_supervisor_picks_composite_when_mineru_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, bool] = {}

    def fake_composite(*, use_paddle, use_mineru):  # type: ignore[no-untyped-def]
        captured.update(use_paddle=use_paddle, use_mineru=use_mineru)
        return _NullExecutor()

    monkeypatch.setattr(composition, "_paddle_available", lambda: False)
    monkeypatch.setattr(composition, "_mineru_available", lambda: True)
    monkeypatch.setattr(composition, "_build_composite_executor", fake_composite)
    build_supervisor(
        instance_id="comp-test",
        stager_root=tmp_path / "stage",
        use_real_paddle=False,
    )
    assert captured == {"use_paddle": False, "use_mineru": True}


def test_build_supervisor_force_paddle_overrides_availability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """use_real_paddle=True forces the composite even if paddle is unavailable."""
    captured: dict[str, bool] = {}

    def fake_composite(*, use_paddle, use_mineru):  # type: ignore[no-untyped-def]
        captured.update(use_paddle=use_paddle, use_mineru=use_mineru)
        return _NullExecutor()

    monkeypatch.setattr(composition, "_paddle_available", lambda: False)
    monkeypatch.setattr(composition, "_mineru_available", lambda: False)
    monkeypatch.setattr(composition, "_build_composite_executor", fake_composite)
    build_supervisor(
        instance_id="comp-test",
        stager_root=tmp_path / "stage",
        use_real_paddle=True,
        use_mineru=False,
    )
    assert captured == {"use_paddle": True, "use_mineru": False}


# ---------------------------------------------------------------------------
# build_supervisor — pdf adapter wiring
# ---------------------------------------------------------------------------


def test_build_supervisor_attaches_pdf_adapter_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = object()

    def fake_pdf_adapter():  # type: ignore[no-untyped-def]
        return sentinel

    monkeypatch.setattr(composition, "_paddle_available", lambda: False)
    monkeypatch.setattr(composition, "_mineru_available", lambda: False)
    monkeypatch.setattr(composition, "_build_pdf_adapter", fake_pdf_adapter)
    module, _ = build_supervisor(
        instance_id="comp-test",
        stager_root=tmp_path / "stage",
        with_pdf_adapter=True,
    )
    assert module.pdf_adapter is sentinel


def test_build_pdf_adapter_wraps_lazy_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_build_pdf_adapter`` returns an adapter whose ``child_factory`` defers to PdfBackendClient."""
    from vibeocr.backend.supervisor.pdf.adapter import PdfProcessAdapter

    sentinel_client = object()

    fake_client_module = types.ModuleType("vibeocr.backend.services.pdf_backend_client")
    fake_client_module.PdfBackendClient = type(
        "PdfBackendClient",
        (),
        {"instance": staticmethod(lambda: sentinel_client)},
    )
    monkeypatch.setitem(sys.modules, "vibeocr.backend.services.pdf_backend_client", fake_client_module)

    adapter = composition._build_pdf_adapter()
    assert isinstance(adapter, PdfProcessAdapter)
    assert adapter.child_factory() is sentinel_client


# ---------------------------------------------------------------------------
# _build_paddle_executor / _build_mineru_executor — lazy import path
# ---------------------------------------------------------------------------


def test_build_paddle_executor_wires_factory_and_clear_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover the paddle executor builder and its inner factory + clear_cache."""
    sentinel_service = object()

    fake_ocr_module = types.ModuleType("vibeocr.backend.services.ocr_service")
    fake_ocr_module.OCRService = lambda: sentinel_service  # type: ignore[assignment]
    monkeypatch.setitem(sys.modules, "vibeocr.backend.services.ocr_service", fake_ocr_module)

    executor = composition._build_paddle_executor()
    # Calling adapter_factory() drives the lazy OCRService import + adapter wrap.
    adapter = executor._adapter_factory()
    assert adapter is not None
    # ``clear_cache`` is stored as a plain callable on the executor. It must
    # not raise regardless of whether paddle is importable / CUDA-compiled.
    assert callable(executor._clear_cache)
    executor._clear_cache()


def test_paddle_clear_cache_runs_empty_cache_when_cuda_compiled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force the CUDA branch of ``clear_cache`` so empty_cache() is invoked."""
    empty_calls: list[str] = []

    fake_paddle = types.ModuleType("paddle")
    fake_device = types.ModuleType("paddle.device")
    fake_cuda = types.ModuleType("paddle.device.cuda")

    fake_device.is_compiled_with_cuda = lambda: True
    fake_cuda.empty_cache = lambda: empty_calls.append("called")
    fake_device.cuda = fake_cuda
    fake_paddle.device = fake_device
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)
    monkeypatch.setitem(sys.modules, "paddle.device", fake_device)
    monkeypatch.setitem(sys.modules, "paddle.device.cuda", fake_cuda)

    sentinel_service = object()
    fake_ocr_module = types.ModuleType("vibeocr.backend.services.ocr_service")
    fake_ocr_module.OCRService = lambda: sentinel_service  # type: ignore[assignment]
    monkeypatch.setitem(sys.modules, "vibeocr.backend.services.ocr_service", fake_ocr_module)

    executor = composition._build_paddle_executor()
    executor._clear_cache()
    assert empty_calls == ["called"]


def test_paddle_clear_cache_swallows_paddle_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The except branch must swallow any error from the paddle import."""
    import builtins

    real_import = builtins.__import__

    def raise_importerror(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]  # noqa: A002 - mirrors builtins.__import__ signature
        if name == "paddle":
            raise ImportError("simulated paddle absence")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", raise_importerror)

    sentinel_service = object()
    fake_ocr_module = types.ModuleType("vibeocr.backend.services.ocr_service")
    fake_ocr_module.OCRService = lambda: sentinel_service  # type: ignore[assignment]
    monkeypatch.setitem(sys.modules, "vibeocr.backend.services.ocr_service", fake_ocr_module)

    executor = composition._build_paddle_executor()
    # Must not raise even though the paddle import fails.
    executor._clear_cache()


def test_paddle_clear_cache_skips_empty_cache_when_not_cuda_compiled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The False branch of ``is_compiled_with_cuda`` must skip empty_cache."""
    empty_calls: list[str] = []

    fake_paddle = types.ModuleType("paddle")
    fake_device = types.ModuleType("paddle.device")
    fake_cuda = types.ModuleType("paddle.device.cuda")

    fake_device.is_compiled_with_cuda = lambda: False
    fake_cuda.empty_cache = lambda: empty_calls.append("should-not-run")
    fake_device.cuda = fake_cuda
    fake_paddle.device = fake_device
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)
    monkeypatch.setitem(sys.modules, "paddle.device", fake_device)
    monkeypatch.setitem(sys.modules, "paddle.device.cuda", fake_cuda)

    fake_ocr_module = types.ModuleType("vibeocr.backend.services.ocr_service")
    fake_ocr_module.OCRService = lambda: object()
    monkeypatch.setitem(sys.modules, "vibeocr.backend.services.ocr_service", fake_ocr_module)

    executor = composition._build_paddle_executor()
    executor._clear_cache()
    assert empty_calls == []


def test_build_mineru_executor_wires_adapter_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover the MinerU executor builder and its inner adapter_factory."""
    sentinel_service = object()

    fake_mineru_module = types.ModuleType("vibeocr.backend.services.mineru_service")
    fake_mineru_module.MinerUService = type(
        "MinerUService", (), {"instance": staticmethod(lambda: sentinel_service)}
    )
    monkeypatch.setitem(sys.modules, "vibeocr.backend.services.mineru_service", fake_mineru_module)

    executor = composition._build_mineru_executor()
    # Calling adapter_factory() drives the lazy MinerUService import + adapter wrap.
    adapter = executor._adapter_factory()
    assert adapter is not None


def test_build_composite_executor_routes_to_each_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Composite builder assembles paddle+mineru children sharing one scheduler."""
    paddle_exec = _NullExecutor()
    mineru_exec = _NullExecutor()

    def fake_paddle(*, scheduler=None):  # type: ignore[no-untyped-def]
        return paddle_exec

    def fake_mineru(*, scheduler=None):  # type: ignore[no-untyped-def]
        return mineru_exec

    monkeypatch.setattr(composition, "_build_paddle_executor", fake_paddle)
    monkeypatch.setattr(composition, "_build_mineru_executor", fake_mineru)
    composite = composition._build_composite_executor(
        use_paddle=True, use_mineru=True
    )
    children = composite._children
    assert {id(c.executor) for c in children} == {id(paddle_exec), id(mineru_exec)}


def test_build_composite_executor_paddle_only(monkeypatch: pytest.MonkeyPatch) -> None:
    paddle_exec = _NullExecutor()
    monkeypatch.setattr(
        composition, "_build_paddle_executor", lambda *, scheduler=None: paddle_exec
    )
    composite = composition._build_composite_executor(
        use_paddle=True, use_mineru=False
    )
    assert len(composite._children) == 1


def test_build_composite_executor_mineru_only(monkeypatch: pytest.MonkeyPatch) -> None:
    mineru_exec = _NullExecutor()
    monkeypatch.setattr(
        composition, "_build_mineru_executor", lambda *, scheduler=None: mineru_exec
    )
    composite = composition._build_composite_executor(
        use_paddle=False, use_mineru=True
    )
    assert len(composite._children) == 1


# ---------------------------------------------------------------------------
# _MinerUServiceLifecycle
# ---------------------------------------------------------------------------


def test_mineru_lifecycle_start_invokes_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    fake_module = types.ModuleType("vibeocr.backend.services.mineru_service")
    fake_module.MinerUService = type(
        "MinerUService",
        (),
        {"instance": staticmethod(lambda: calls.append("instance") or object())},
    )
    monkeypatch.setitem(sys.modules, "vibeocr.backend.services.mineru_service", fake_module)

    _MinerUServiceLifecycle().start()
    assert calls == ["instance"]


def test_mineru_lifecycle_stop_invokes_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _Svc:
        def shutdown(self):  # type: ignore[no-untyped-def]
            calls.append("shutdown")

    fake_module = types.ModuleType("vibeocr.backend.services.mineru_service")
    fake_module.MinerUService = type(
        "MinerUService", (), {"instance": staticmethod(lambda: _Svc())}
    )
    monkeypatch.setitem(sys.modules, "vibeocr.backend.services.mineru_service", fake_module)

    _MinerUServiceLifecycle().stop()
    assert calls == ["shutdown"]


def test_mineru_lifecycle_stop_swallows_shutdown_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BrokenSvc:
        def shutdown(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("wedged")

    fake_module = types.ModuleType("vibeocr.backend.services.mineru_service")
    fake_module.MinerUService = type(
        "MinerUService", (), {"instance": staticmethod(lambda: _BrokenSvc())}
    )
    monkeypatch.setitem(sys.modules, "vibeocr.backend.services.mineru_service", fake_module)

    # Must not raise.
    _MinerUServiceLifecycle().stop()


# ---------------------------------------------------------------------------
# Availability probes
# ---------------------------------------------------------------------------


def test_paddle_available_detects_importable_module() -> None:
    """When paddle imports cleanly, the probe returns True."""
    # paddle may or may not be installed in the test environment; the probe's
    # contract is "True iff importable", so just assert it matches reality.
    try:
        __import__("paddle")
        expected = True
    except Exception:
        expected = False
    assert _paddle_available() is expected


def test_paddle_available_returns_false_on_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The except branch returns False when the import raises."""
    import builtins

    real_import = builtins.__import__

    def raise_importerror(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]  # noqa: A002 - mirrors builtins.__import__ signature
        if name == "paddle":
            raise ImportError("no paddle")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", raise_importerror)
    assert _paddle_available() is False


def test_mineru_available_detects_importable_module() -> None:
    try:
        import mineru  # type: ignore  # noqa: F401
        expected = True
    except Exception:
        expected = False
    assert _mineru_available() is expected


def test_mineru_available_returns_false_on_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The except branch returns False when mineru import raises."""
    import builtins

    real_import = builtins.__import__

    def raise_importerror(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]  # noqa: A002 - mirrors builtins.__import__ signature
        if name == "mineru":
            raise ImportError("no mineru")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", raise_importerror)
    assert _mineru_available() is False
