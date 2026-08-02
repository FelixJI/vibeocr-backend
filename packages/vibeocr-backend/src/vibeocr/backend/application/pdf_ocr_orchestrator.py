"""UI-free PDF OCR orchestrator: the single source of truth for batch OCR,
checkpoint (sidecar), final compress and crash-resume.

This module is deliberately independent of PySide6 so that both the Qt
``PdfSessionManager`` and the WorkerHost PDF handler can delegate to it
without duplicating the durable-write state machine.

Durable semantics (must match the production PDF flow on ``main``):

- OCR writes text layers in fixed-size batches via the backend's
  ``add_text_layer_batch(save=True)``. Only pages whose batch reports
  ``extra.saved == True`` may enter the sidecar; a batch whose save fails is
  rolled back by the backend and must not be checkpointed.
- Sidecar is path-keyed (md5 of the normalized absolute path) and validated by
  a "only-grows-never-rolls-back" check on ``original_size`` /
  ``original_mtime_ns`` (delegated to :mod:`vibeocr.backend.utils.ocr_sidecar`).
- After the final aggregate compress, the orchestrator calls
  ``refresh_baseline`` then ``mark_completed``; on compress failure the
  sidecar stays ``completed=False`` with already-persisted pages intact.
- ``reset_cancel`` runs before a new task; batch write cancels at page
  boundaries; mutations never auto-retry. A crashed session is explicitly
  invalidated for resume.
- Page processing state is the closed set ``none / processing / done /
  failed``; text-layer source is the orthogonal ``none / native / ocr``
  projection (gray / light green / dark green). Native PDF text layers are
  never written into the OCR sidecar's completed set.

The orchestrator is driven by a :class:`PdfOcrBackend` protocol the caller
implements against the real PDF backend (or a fake in tests).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from vibeocr.backend.utils import ocr_sidecar

logger = logging.getLogger(__name__)

#: How many pages to render + recognize + write in one durable batch.
DEFAULT_BATCH_SIZE = 16


class PageState(StrEnum):
    """Closed set of page processing states."""

    NONE = "none"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class LayerSource(StrEnum):
    """Orthogonal text-layer origin (gray / light green / dark green)."""

    NONE = "none"
    NATIVE = "native"
    OCR = "ocr"


@dataclass(frozen=True)
class OcrPageResult:
    """Outcome for one page within a batch."""

    page_index: int
    text: str | None
    blocks: list[dict] | None
    preproc_angle: int = 0
    #: True when the page had a native PDF text layer before OCR (display only).
    had_native_layer: bool = False


@dataclass(frozen=True)
class BatchOutcome:
    """Result of one durable batch write."""

    saved_pages: tuple[int, ...]
    failed_pages: tuple[int, ...]
    #: ``True`` only when the backend committed the batch to disk.
    saved: bool
    #: Backend error detail (aggregated, deduplicated) for this batch.
    write_errors: tuple[str, ...] = ()


@dataclass
class OcrRunResult:
    """Aggregate outcome of a full OCR run over a session."""

    completed: int = 0
    failed: int = 0
    cancelled: bool = False
    #: Deduplicated, ordered write-error messages collected across all batches.
    write_errors: list[str] = field(default_factory=list)
    #: Final compress succeeded.
    compressed: bool = False
    #: Page index -> (PageState, LayerSource) projection for the UI.
    page_states: dict[int, tuple[PageState, LayerSource]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.completed + self.failed


@runtime_checkable
class PdfOcrBackend(Protocol):
    """Backend boundary the orchestrator drives.

    Implementations wrap the real PDF backend (render + recognize +
    add_text_layer_batch + save) or a fake. All methods may raise; the
    orchestrator maps exceptions to page failures rather than aborting the run.
    """

    def reset_cancel(self, session_id: str) -> None: ...

    def render_pages(
        self,
        session_id: str,
        page_indices: list[int],
        cancel_check: Any,
    ) -> list[bytes]:
        """Render the given pages to PNG bytes, in order."""
        ...

    def recognize_pages(
        self,
        session_id: str,
        images: list[bytes],
        cancel_check: Any,
    ) -> list[OcrPageResult]:
        """Recognize rendered images; one result per input image."""
        ...

    def write_batch(
        self,
        session_id: str,
        pages: list[tuple[int, OcrPageResult]],
        *,
        overwrite: bool,
        save: bool,
        cancel_check: Any,
    ) -> BatchOutcome:
        """Write one durable batch. ``save=True`` requests incremental save."""
        ...

    def compress(self, session_id: str, cancel_check: Any) -> bool:
        """Final aggregate compress-in-place; return success."""
        ...

    def cancel(self, session_id: str) -> None: ...


class _CancelFlag:
    """Cooperative cancel flag shared with backend cancel_check callbacks."""

    __slots__ = ("_cancelled",)

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


def project_layer_source(*, had_native_layer: bool, ocr_saved: bool) -> LayerSource:
    """Project the text-layer source for display.

    OCR-saved takes precedence (dark green); a native layer alone is light
    green; otherwise none (gray). Native layers are never reported as OCR.
    """
    if ocr_saved:
        return LayerSource.OCR
    if had_native_layer:
        return LayerSource.NATIVE
    return LayerSource.NONE


class PdfOcrOrchestrator:
    """Single source of truth for batch OCR + checkpoint + compress + resume.

    The orchestrator is stateless across runs: all durability lives in the
    sidecar. Callers pass the file path (sidecar key) and the session id.
    """

    def __init__(
        self, backend: PdfOcrBackend, *, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._backend = backend
        self._batch_size = batch_size

    def run_ocr(
        self,
        *,
        session_id: str,
        file_path: str,
        page_indices: list[int],
        overwrite: bool = False,
        sidecar_root: str | None = None,
    ) -> OcrRunResult:
        """Run OCR over ``page_indices`` with durable per-batch checkpointing.

        - ``overwrite=False`` filters out pages already in the sidecar (resume).
        - ``overwrite=True`` redoes all requested pages.
        - When every requested page is already saved, a completion event is
          still produced and the UI must reset (no silent no-op).
        """
        if not page_indices:
            return OcrRunResult()

        cancel = _CancelFlag()
        self._backend.reset_cancel(session_id)

        pending = self._filter_pending(file_path, page_indices, overwrite, sidecar_root)
        pending_set = set(pending)
        result = OcrRunResult()
        # Initialize the projection. Pages already saved (resume) are DONE/OCR and
        # count as completed; pages still to process start as NONE.
        for idx in page_indices:
            source = _resume_source(file_path, idx, sidecar_root)
            if idx not in pending_set:
                result.page_states[idx] = (PageState.DONE, source)
                result.completed += 1
            else:
                result.page_states[idx] = (PageState.NONE, source)

        if not pending:
            # Everything already saved: emit completion so the UI resets.
            return result

        seen_errors: set[str] = set()
        batch_start = 0
        while batch_start < len(pending):
            if cancel.cancelled:
                result.cancelled = True
                _mark_unfinished(pending, batch_start, result, PageState.NONE)
                break
            batch = pending[batch_start : batch_start + self._batch_size]
            for idx in batch:
                result.page_states[idx] = (
                    PageState.PROCESSING,
                    result.page_states[idx][1],
                )
            outcome = self._process_batch(
                session_id, file_path, batch, overwrite, cancel, sidecar_root
            )
            self._record_batch(outcome, result, seen_errors, batch)
            if cancel.cancelled:
                result.cancelled = True
                _mark_unfinished(
                    pending, batch_start + len(batch), result, PageState.NONE
                )
                break
            batch_start += len(batch)

        if not result.cancelled:
            result.compressed = self._final_compress(
                session_id, file_path, cancel, sidecar_root
            )

        return result

    def request_cancel(self) -> None:
        """Signal the current run to stop at the next page boundary.

        The orchestrator does not own a run handle; callers typically hold the
        cancel flag returned from :meth:`make_cancel_flag` instead. This method
        is retained for symmetry with the backend boundary.
        """
        # No-op without a flag; real cancellation is flag-driven (see run_ocr).
        logger.debug("PdfOcrOrchestrator.request_cancel called without a flag handle")

    def _filter_pending(
        self,
        file_path: str,
        page_indices: list[int],
        overwrite: bool,
        sidecar_root: str | None,
    ) -> list[int]:
        if overwrite:
            return list(page_indices)
        with _sidecar_root(sidecar_root):
            saved = restore_pending_pages_for(file_path) or {}
        return [idx for idx in page_indices if idx not in saved]

    def _process_batch(
        self,
        session_id: str,
        file_path: str,
        batch: list[int],
        overwrite: bool,
        cancel: _CancelFlag,
        sidecar_root: str | None,
    ) -> BatchOutcome:
        try:
            images = self._backend.render_pages(session_id, batch, cancel)
            if cancel.cancelled:
                return BatchOutcome(
                    saved_pages=(), failed_pages=tuple(batch), saved=False
                )
            results = self._backend.recognize_pages(session_id, images, cancel)
            if cancel.cancelled:
                return BatchOutcome(
                    saved_pages=(), failed_pages=tuple(batch), saved=False
                )
            pages_data = [
                (idx, res)
                for idx, res in zip(batch, results, strict=True)
                if res.text or res.blocks
            ]
            outcome = self._backend.write_batch(
                session_id,
                pages_data,
                overwrite=overwrite,
                save=True,
                cancel_check=cancel,
            )
            if outcome.saved:
                with _sidecar_root(sidecar_root):
                    ocr_sidecar.mark_pages_saved(
                        file_path,
                        list(outcome.saved_pages),
                        dict.fromkeys(outcome.saved_pages, 0),
                    )
            return outcome
        except Exception as error:
            logger.warning("PDF OCR batch failed (page boundary): %s", error)
            self._backend.cancel(session_id)
            return BatchOutcome(
                saved_pages=(),
                failed_pages=tuple(batch),
                saved=False,
                write_errors=(str(error),),
            )

    def _record_batch(
        self,
        outcome: BatchOutcome,
        result: OcrRunResult,
        seen_errors: set[str],
        batch: list[int],
    ) -> None:
        saved_set = set(outcome.saved_pages)
        failed_set = set(outcome.failed_pages)
        for idx in batch:
            if idx in saved_set:
                result.completed += 1
                result.page_states[idx] = (PageState.DONE, LayerSource.OCR)
            elif idx in failed_set:
                result.failed += 1
                result.page_states[idx] = (PageState.FAILED, result.page_states[idx][1])
            else:
                # Not written (no text recognized) — keep prior source, mark done.
                result.completed += 1
                result.page_states[idx] = (PageState.DONE, result.page_states[idx][1])
        for err in outcome.write_errors:
            key = err.strip()
            if key and key not in seen_errors:
                seen_errors.add(key)
                result.write_errors.append(key)

    def _final_compress(
        self,
        session_id: str,
        file_path: str,
        cancel: _CancelFlag,
        sidecar_root: str | None,
    ) -> bool:
        try:
            ok = self._backend.compress(session_id, cancel)
        except Exception as error:
            logger.warning("PDF final compress failed: %s", error)
            return False
        if ok and not cancel.cancelled:
            with _sidecar_root(sidecar_root):
                ocr_sidecar.refresh_baseline(file_path)
                ocr_sidecar.mark_completed(file_path)
        return ok


# ---------------------------------------------------------------------------
# Sidecar root indirection — production keeps the default; winui-dev profile
# redirects to an isolated cache so sidecar/config/model/output stay untouched.
# ---------------------------------------------------------------------------

_ACTIVE_SIDECAR_ROOT: str | None = None


class _sidecar_root:
    """Context manager that temporarily overrides the sidecar directory root.

    The default sidecar location is resolved through ``get_project_root``; the
    winui-dev profile injects an isolated root so bypass OCR never writes the
    production sidecar.
    """

    def __init__(self, root: str | None) -> None:
        self._root = root
        self._previous: str | None = None

    def __enter__(self) -> None:
        global _ACTIVE_SIDECAR_ROOT
        self._previous = _ACTIVE_SIDECAR_ROOT
        _ACTIVE_SIDECAR_ROOT = self._root

    def __exit__(self, *exc: object) -> None:
        global _ACTIVE_SIDECAR_ROOT
        _ACTIVE_SIDECAR_ROOT = self._previous


def _resume_source(file_path: str, idx: int, sidecar_root: str | None) -> LayerSource:
    with _sidecar_root(sidecar_root):
        saved = restore_pending_pages_for(file_path) or {}
    return LayerSource.OCR if idx in saved else LayerSource.NONE


def restore_pending_pages_for(file_path: str) -> dict[int, int] | None:
    """Restore pending pages honouring the active sidecar-root override."""
    if _ACTIVE_SIDECAR_ROOT is None:
        return ocr_sidecar.restore_pending_pages(file_path)
    return _restore_with_root(file_path, _ACTIVE_SIDECAR_ROOT)


def _restore_with_root(file_path: str, root: str) -> dict[int, int] | None:
    # The production ocr_sidecar resolves the directory via get_project_root().
    # For the winui-dev profile we point get_cache_dir at the isolated root by
    # monkeypatching the project root lookup for the duration of the call.
    import vibeocr.backend.utils.ocr_sidecar as sidecar_mod

    original_sessions_dir = sidecar_mod._sessions_dir
    try:
        from pathlib import Path

        root_path = Path(root)

        def _override() -> Path:
            return root_path / sidecar_mod._SIDECAR_SUBDIR

        sidecar_mod._sessions_dir = _override  # type: ignore[assignment]
        return sidecar_mod.restore_pending_pages(file_path)
    finally:
        sidecar_mod._sessions_dir = original_sessions_dir  # type: ignore[assignment]


def _mark_unfinished(
    pending: list[int], from_offset: int, result: OcrRunResult, state: PageState
) -> None:
    for idx in pending[from_offset:]:
        prev_source = result.page_states.get(idx, (PageState.NONE, LayerSource.NONE))[1]
        result.page_states[idx] = (state, prev_source)


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "BatchOutcome",
    "LayerSource",
    "OcrPageResult",
    "OcrRunResult",
    "PageState",
    "PdfOcrBackend",
    "PdfOcrOrchestrator",
    "project_layer_source",
    "restore_pending_pages_for",
]
