"""Composition root: assemble the SupervisorModule with its adapters.

In Phase 2 the only executor is a fake used by tests. Phase 4/5/6 will plug
the real Paddle/MinerU/PDF adapters here without changing the module or app
shape. The composition root is also where ``stager_root`` is chosen (a
session-scoped temp directory under the OS temp or a portable location).
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .bootstrap import BootstrapHandle, generate_session_token, new_instance_id
from .module import Executor, SupervisorModule, SupervisorOptions
from .settings_store import RuntimeSettings

if TYPE_CHECKING:
    from vibeocr.runtime_contracts import CancelMode, ResidencyStatus

    from .inference.ocr_engines import OcrEngineRegistry, OcrEngineResolver
    from .inference.recognition_modes import RecognitionModeRegistry
    from .jobs.registry import JobRecord


class _NullExecutor:
    """Default no-op executor used until real adapters are plugged.

    Real Phase 4/5/6 work replaces this; keeping a null default lets the
    bootstrap/app smoke tests run without OCR dependencies.
    """

    def execute(self, record, staged) -> None:  # type: ignore[no-untyped-def]
        # Immediately mark the job failed with a typed error: no backend.
        from vibeocr.runtime_contracts import JobState

        if record.state not in (
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
        ):
            try:
                record.transition(JobState.FAILED)
                record.append_event("no_backend", detail={"reason": "null-executor"})
            except Exception:  # pragma: no cover - defensive
                pass

    def cancel_mode_for(self, record: JobRecord) -> CancelMode:
        from vibeocr.runtime_contracts import CancelMode

        return CancelMode.COOPERATIVE

    def residency_status(self) -> ResidencyStatus:
        from vibeocr.runtime_contracts import ResidencyStatus

        return ResidencyStatus()

    def release_idle(self, pipeline: str | None = None) -> ResidencyStatus:
        from vibeocr.runtime_contracts import ResidencyStatus

        return ResidencyStatus()

    def preload(self, pipelines: tuple[str, ...]) -> ResidencyStatus:
        from vibeocr.runtime_contracts import ResidencyStatus

        del pipelines
        return ResidencyStatus()

    def configure_settings(self, snapshot) -> ResidencyStatus:  # type: ignore[no-untyped-def]
        from vibeocr.runtime_contracts import ResidencyStatus

        return ResidencyStatus(
            default_ttl_seconds=snapshot.default_ttl_seconds,
            pipelines=snapshot.pipelines,
        )

    def close(self) -> None:
        return


def _paddle_adapter_factory() -> Any:
    """惰性构建 Paddle adapter：导入 OCRService 延迟到首次使用。"""
    from vibeocr.backend.services.ocr_service import OCRService

    from .inference.paddle_adapter import PaddlePipelineAdapter

    return PaddlePipelineAdapter(service=OCRService())


def _build_ocr_engine_registry(
    paddle_adapter_factory: Callable[[], Any],
) -> OcrEngineRegistry:
    """构建引擎 registry：rapidocr / windows / paddleocr 全部注册。

    目录探测来自实际探针（importability/语言包），未安装的稳定 id 以
    unavailable descriptor 占位。Paddle 以 LazyEngineHandle 注册，保持
    重依赖导入惰性。
    """
    from vibeocr.runtime_contracts.dtos import OcrEngine

    from .inference.native_runtime import prepare_windows_ocr_native_runtime
    from .inference.ocr_engines import LazyEngineHandle, OcrEngineRegistry
    from .inference.rapidocr_engine import RapidOcrEngine
    from .inference.windows_ocr_engine import WindowsMediaOcrEngine

    prepare_windows_ocr_native_runtime()

    def paddle_descriptor() -> Any:
        from .inference.paddle_adapter import PaddlePipelineAdapter

        return PaddlePipelineAdapter._probe_descriptor()

    return OcrEngineRegistry(
        [
            RapidOcrEngine(),
            WindowsMediaOcrEngine(),
            LazyEngineHandle(
                engine_id=OcrEngine.PADDLEOCR,
                descriptor_probe=paddle_descriptor,
                factory=paddle_adapter_factory,
            ),
        ]
    )


def _build_paddle_executor(
    *,
    scheduler: Any = None,
    engine_registry: OcrEngineRegistry | None = None,
    engine_resolver: OcrEngineResolver | None = None,
    paddle_fallback: bool = True,
) -> Executor:
    """Construct the RECOGNITION executor backed by real OCR engines.

    The heavy model load is deferred: adapter factories are lazy, so the
    first ``recognize_many`` call (not import) pays the model-load cost.
    When ``engine_registry`` is provided the paddle adapter is wrapped in an
    :class:`OcrEngineRoutingAdapter`: plain-text ``OCR`` jobs route through
    the engine resolver (default rapidocr, fail closed), while every other
    Paddle pipeline keeps the direct adapter path.
    """
    from .inference.ocr_engine_router import OcrEngineRoutingAdapter
    from .inference.ocr_engines import OcrEngineResolver
    from .inference.paddle_executor import PaddleExecutor

    def clear_cache() -> None:
        try:
            import paddle

            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
        except Exception:
            pass

    adapter_factory: Any = _paddle_adapter_factory
    if engine_registry is not None:
        resolver = engine_resolver or OcrEngineResolver(registry=engine_registry)

        def routing_factory() -> Any:
            return OcrEngineRoutingAdapter(
                fallback_factory=(_paddle_adapter_factory if paddle_fallback else None),
                resolver=resolver,
            )

        adapter_factory = routing_factory
    return PaddleExecutor(
        adapter_factory=adapter_factory,
        scheduler=scheduler,
        clear_cache=clear_cache,
    )


def _paddle_available() -> bool:
    """Return True if a real Paddle backend is importable in this environment."""
    try:
        __import__("paddle")
    except Exception:
        return False
    return True


def _build_pdf_adapter() -> Any:
    """Construct a PdfProcessAdapter backed by the legacy PdfBackendClient.

    The child factory returns ``PdfBackendClient.instance()`` so the supervisor
    reuses the existing FastAPI PDF child process (``pdf_backend_process.py``)
    rather than reimplementing PyMuPDF. The supervisor becomes the sole owner
    of that child; the GUI no longer holds a ``PdfBackendClient`` reference.
    Import is lazy: the backend client pulls httpx + the process launcher, so
    we defer it to first use (see ``PdfProcessAdapter.ensure_started``).
    """
    from .pdf.adapter import PdfProcessAdapter

    def factory() -> Any:
        from vibeocr.backend.services.pdf_backend_client import PdfBackendClient

        return PdfBackendClient.instance()

    return PdfProcessAdapter(child_factory=factory)


class _MinerUServiceLifecycle:
    """Drives the mineru-api subprocess behind the MinerU singleton.

    ``MinerUService`` is a lazy singleton: its ``__init__`` blocks until the
    API is reachable, and ``shutdown()`` tears it down. The adapter's
    lifecycle seam calls these so the supervisor owns start/stop of the API
    subprocess exactly like it owns the Paddle model residency.
    """

    def start(self) -> None:
        from vibeocr.backend.services.mineru_service import MinerUService

        MinerUService.instance()  # blocks until API up

    def stop(self) -> None:
        from vibeocr.backend.services.mineru_service import MinerUService

        try:
            MinerUService.instance().shutdown()
        except Exception:  # pragma: no cover - defensive
            pass


def _build_mineru_executor(*, scheduler: Any = None) -> Executor:
    """Construct a real MinerUExecutor owning the MinerU API subprocess.

    The ``client_factory`` returns the singleton ``MinerUService``, whose
    ``file_parse`` issues one budgeted multi-file ``/file_parse`` request and
    returns ``{stem: payload}``. The lifecycle wrapper starts/stops the
    mineru-api subprocess; the heavy model download happens on first parse.
    """
    from .inference.mineru_adapter import MinerUProcessAdapter
    from .inference.mineru_executor import MinerUExecutor

    def adapter_factory() -> MinerUProcessAdapter:
        from vibeocr.backend.services.mineru_service import MinerUService

        return MinerUProcessAdapter(
            client_factory=lambda: MinerUService.instance(),
            lifecycle=_MinerUServiceLifecycle(),
        )

    return MinerUExecutor(adapter_factory=adapter_factory, scheduler=scheduler)


def _build_recognition_mode_registry(
    *,
    engine_resolver: OcrEngineResolver,
    use_paddle: bool,
    use_mineru: bool,
) -> RecognitionModeRegistry:
    """把实时 engine/component 探针投影成单一 Recognition Mode 目录。"""
    from .inference.recognition_modes import (
        ModeAvailability,
        RecognitionModeAvailability,
        RecognitionModeId,
        RecognitionModeRegistry,
    )

    accelerator = os.environ.get("VIBEOCR_RUNTIME_ACCELERATOR", "cpu")
    suffix = "cuda" if accelerator == "nvidia_cuda" else "cpu"

    def availability(definition) -> ModeAvailability:  # type: ignore[no-untyped-def]
        if definition.engine is not None:
            descriptors = {
                descriptor.engine_id.value: descriptor
                for descriptor in engine_resolver.probe_descriptors()
            }
            descriptor = descriptors[definition.engine]
            if definition.mode_id is RecognitionModeId.RAPID_TEXT:
                if descriptor.availability.value == "ready":
                    return ModeAvailability(RecognitionModeAvailability.READY)
                return ModeAvailability(
                    RecognitionModeAvailability.PREPARATION_REQUIRED,
                    reason_code=descriptor.reason_code,
                    required_component="rapidocr-base",
                )
            required_component = (
                f"paddleocr-{suffix}"
                if definition.mode_id is RecognitionModeId.PADDLE_TEXT
                else None
            )
            return ModeAvailability(
                descriptor.availability.value,
                reason_code=descriptor.reason_code,
                required_component=required_component,
            )
        is_mineru = definition.mode_id is RecognitionModeId.MINERU_DOCUMENT
        ready = use_mineru if is_mineru else use_paddle
        component = f"mineru-{suffix}" if is_mineru else f"paddleocr-{suffix}"
        return ModeAvailability(
            RecognitionModeAvailability.READY
            if ready
            else RecognitionModeAvailability.PREPARATION_REQUIRED,
            reason_code=None if ready else "runtime_component_missing",
            required_component=None if ready else component,
        )

    return RecognitionModeRegistry(availability_probe=availability)


def _mineru_available() -> bool:
    """Return True if a real MinerU backend is importable in this environment."""
    try:
        import mineru  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def _build_composite_executor(
    *,
    use_paddle: bool,
    use_mineru: bool,
    engine_registry: OcrEngineRegistry | None = None,
    engine_resolver: OcrEngineResolver | None = None,
) -> Executor:
    """Build a CompositeExecutor over whichever real backends are available.

    Paddle handles ``RECOGNITION`` jobs (optionally behind the OCR engine
    router); MinerU handles ``MINERU_PARSE`` jobs. If only one is available
    the composite still routes correctly; if neither is, the caller falls
    back to ``_NullExecutor``.
    """
    from vibeocr.runtime_contracts import JobKind

    from .inference.composite_executor import CompositeExecutor
    from .inference.scheduler import DeviceScheduler

    scheduler = DeviceScheduler(devices=["gpu:0"])
    children: list[tuple[Executor, frozenset]] = []
    # Base Runtime always owns RECOGNITION through RapidOCR/Windows. Paddle is
    # only the optional fallback for non-OCR recognition pipelines.
    children.append(
        (
            _build_paddle_executor(
                scheduler=scheduler,
                engine_registry=engine_registry,
                engine_resolver=engine_resolver,
                paddle_fallback=use_paddle,
            ),
            frozenset({JobKind.RECOGNITION}),
        )
    )
    if use_mineru:
        children.append(
            (
                _build_mineru_executor(scheduler=scheduler),
                frozenset({JobKind.MINERU_PARSE}),
            )
        )
    return CompositeExecutor(children)


def build_supervisor(
    *,
    instance_id: str | None = None,
    stager_root: Path | None = None,
    executor: Executor | None = None,
    options: SupervisorOptions | None = None,
    bootstrap_handle: BootstrapHandle | None = None,
    use_real_paddle: bool | None = None,
    use_mineru: bool | None = None,
    with_pdf_adapter: bool = False,
    engine_registry: OcrEngineRegistry | None = None,
    settings_store: RuntimeSettings | None = None,
) -> tuple[SupervisorModule, BootstrapHandle]:
    """Assemble a supervisor module + bootstrap handle (token out of band).

    When ``use_real_paddle`` is True (or left as None and a Paddle backend is
    importable), the supervisor is wired with a real
    :class:`~vibeocr.backend.supervisor.inference.paddle_executor.PaddleExecutor`
    backed by the singleton :class:`~vibeocr.backend.services.ocr_service.OCRService`,
    so recognition jobs actually run Paddle OCR.

    When ``use_mineru`` is True (or left as None and a MinerU backend is
    importable), a real
    :class:`~vibeocr.backend.supervisor.inference.mineru_executor.MinerUExecutor`
    (owning the mineru-api subprocess) is added alongside Paddle behind a
    :class:`~vibeocr.backend.supervisor.inference.composite_executor.CompositeExecutor`
    that routes by ``JobKind`` (RECOGNITION → Paddle, MINERU_PARSE → MinerU).

    Otherwise (or in lightweight test environments without paddle/mineru) the
    null executor is used so the job engine stays importable and unit-testable
    without model dependencies.

    When ``with_pdf_adapter`` is True, the module owns a
    :class:`~vibeocr.backend.supervisor.pdf.adapter.PdfProcessAdapter` whose
    ``child_factory`` returns the legacy
    :class:`~vibeocr.backend.services.pdf_backend_client.PdfBackendClient` singleton.
    The v2 PDF session routes then proxy through it instead of the GUI holding
    the client directly (plan §6 / ADR §"Transport"). The PDF child subprocess
    is spawned lazily on first ``open_session``; no cost at import.
    """
    iid = instance_id or new_instance_id()
    opts = options or SupervisorOptions(instance_id=iid)
    root = stager_root or Path(tempfile.mkdtemp(prefix=f"vibeocr-sup-{iid}-"))
    if executor is not None:
        exec_impl = executor
        engine_resolver = None
        recognition_mode_registry = None
    else:
        want_paddle = use_real_paddle is True or (
            use_real_paddle is None and _paddle_available()
        )
        want_mineru = use_mineru is True or (use_mineru is None and _mineru_available())
        if engine_registry is None:
            engine_registry = _build_ocr_engine_registry(_paddle_adapter_factory)
        from .inference.ocr_engines import OcrEngineResolver

        engine_resolver = OcrEngineResolver(registry=engine_registry)
        recognition_mode_registry = _build_recognition_mode_registry(
            engine_resolver=engine_resolver,
            use_paddle=want_paddle,
            use_mineru=want_mineru,
        )
        exec_impl = _build_composite_executor(
            use_paddle=want_paddle,
            use_mineru=want_mineru,
            engine_registry=engine_registry,
            engine_resolver=engine_resolver,
        )
    pdf_adapter = _build_pdf_adapter() if with_pdf_adapter else None
    module = SupervisorModule(
        options=opts,
        stager_root=root,
        executor=exec_impl,
        pdf_adapter=pdf_adapter,
        engine_registry=engine_registry,
        engine_resolver=engine_resolver,
        recognition_mode_registry=recognition_mode_registry,
        settings_store=settings_store,
    )
    # Clean stale staging left by a previous crashed instance (plan Phase 2).
    # At startup no jobs are known yet, so every existing dir is stale.
    module.stager.cleanup_stale(set())
    handle = bootstrap_handle or BootstrapHandle()
    # Always generate a new token unless the handle already has one set.
    # BootstrapHandle.token raises if unset, so we use a safe check.
    try:
        _ = handle.token  # type: ignore[attr-defined]
    except RuntimeError:
        handle.set_token(generate_session_token())
    return module, handle


__all__ = ["build_supervisor"]
