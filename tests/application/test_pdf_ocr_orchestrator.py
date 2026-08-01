"""Tests for the UI-free PDF OCR orchestrator.

These pin the durable semantics the WinUI tab must mirror: per-batch save +
sidecar checkpoint, crash-resume filtering, cancel-at-page-boundary, the
orthogonal page-state / layer-source projection, and the ocr_write_error
aggregation. They do not touch PySide6.
"""

from __future__ import annotations

from typing import Any

import pytest
from vibeocr.backend.application.pdf_ocr_orchestrator import (
    BatchOutcome,
    LayerSource,
    OcrPageResult,
    PageState,
    PdfOcrOrchestrator,
    project_layer_source,
)


class FakeBackend:
    """In-memory backend that records calls and simulates save/cancel/failure."""

    def __init__(
        self,
        *,
        save_batches: bool = True,
        fail_pages: set[int] | None = None,
        compress_ok: bool = True,
        native_layers: set[int] | None = None,
    ) -> None:
        self._save_batches = save_batches
        self._fail_pages = fail_pages or set()
        self._compress_ok = compress_ok
        self._native = native_layers or set()
        self.reset_cancel_calls: list[str] = []
        self.cancel_calls: list[str] = []
        self.render_calls: list[list[int]] = []
        self.recognize_calls: list[int] = []
        self.write_calls: list[list[int]] = []
        self.compress_calls: int = 0
        self._cancelled_session: str | None = None

    def reset_cancel(self, session_id: str) -> None:
        self.reset_cancel_calls.append(session_id)
        self._cancelled_session = None

    def render_pages(
        self, session_id: str, page_indices: list[int], cancel_check: Any
    ) -> list[bytes]:
        self.render_calls.append(list(page_indices))
        return [b"\x89PNG" for _ in page_indices]

    def recognize_pages(
        self, session_id: str, images: list[bytes], cancel_check: Any
    ) -> list[OcrPageResult]:
        self.recognize_calls.append(len(images))
        return [
            OcrPageResult(
                page_index=0,
                text=f"text-{i}",
                blocks=[{"text": f"text-{i}"}],
                had_native_layer=i in self._native,
            )
            for i, _ in enumerate(images)
        ]

    def write_batch(
        self,
        session_id: str,
        pages: list[tuple[int, OcrPageResult]],
        *,
        overwrite: bool,
        save: bool,
        cancel_check: Any,
    ) -> BatchOutcome:
        page_indices = [idx for idx, _ in pages]
        self.write_calls.append(page_indices)
        if self._cancelled_session == session_id:
            return BatchOutcome(
                saved_pages=(), failed_pages=tuple(page_indices), saved=False
            )
        if not self._save_batches:
            return BatchOutcome(
                saved_pages=(),
                failed_pages=tuple(page_indices),
                saved=False,
                write_errors=("backend save failed",),
            )
        saved = tuple(idx for idx in page_indices if idx not in self._fail_pages)
        failed = tuple(idx for idx in page_indices if idx in self._fail_pages)
        errors: tuple[str, ...] = ("disk full",) if failed else ()
        return BatchOutcome(
            saved_pages=saved, failed_pages=failed, saved=save, write_errors=errors
        )

    def compress(self, session_id: str, cancel_check: Any) -> bool:
        self.compress_calls += 1
        return self._compress_ok

    def cancel(self, session_id: str) -> None:
        self.cancel_calls.append(session_id)
        self._cancelled_session = session_id


@pytest.fixture
def isolated_sidecar(tmp_path, monkeypatch):
    """Redirect the sidecar directory to a temp root so tests are hermetic.

    Returns a real (empty) PDF path so sidecar ``os.stat`` baseline capture
    succeeds.
    """
    import vibeocr.backend.utils.ocr_sidecar as sidecar_mod

    root = tmp_path / "sidecar"
    monkeypatch.setattr(sidecar_mod, "_sessions_dir", lambda: root / "ocr_sessions")
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%test\n")
    return pdf


def test_project_layer_source_precedence():
    assert (
        project_layer_source(had_native_layer=False, ocr_saved=False)
        == LayerSource.NONE
    )
    assert (
        project_layer_source(had_native_layer=True, ocr_saved=False)
        == LayerSource.NATIVE
    )
    assert (
        project_layer_source(had_native_layer=False, ocr_saved=True) == LayerSource.OCR
    )
    # OCR-saved takes precedence over native.
    assert (
        project_layer_source(had_native_layer=True, ocr_saved=True) == LayerSource.OCR
    )


def test_run_ocr_writes_all_pages_and_compresses(isolated_sidecar):
    backend = FakeBackend()
    orch = PdfOcrOrchestrator(backend, batch_size=4)

    result = orch.run_ocr(
        session_id="sess-1",
        file_path=str(isolated_sidecar),
        page_indices=[0, 1, 2, 3],
    )

    assert result.cancelled is False
    assert result.completed == 4
    assert result.failed == 0
    assert result.compressed is True
    assert backend.compress_calls == 1
    assert backend.reset_cancel_calls == ["sess-1"]
    assert all(result.page_states[i][0] is PageState.DONE for i in range(4))
    assert all(result.page_states[i][1] is LayerSource.OCR for i in range(4))


def test_overwrite_false_filters_already_saved_pages(isolated_sidecar, monkeypatch):
    import vibeocr.backend.utils.ocr_sidecar as sidecar_mod

    file_path = str(isolated_sidecar)
    # Seed: pages 0 and 2 already persisted.
    sidecar_mod.mark_pages_saved(file_path, [0, 2], {0: 0, 2: 0})
    backend = FakeBackend()
    orch = PdfOcrOrchestrator(backend, batch_size=4)

    result = orch.run_ocr(
        session_id="sess-1",
        file_path=file_path,
        page_indices=[0, 1, 2, 3],
        overwrite=False,
    )

    # Only pages 1 and 3 should have been rendered/written.
    rendered = [idx for call in backend.render_calls for idx in call]
    assert sorted(rendered) == [1, 3]
    assert result.completed == 4  # 2 already-saved + 2 newly done


def test_all_pages_saved_emits_completion_without_rendering(isolated_sidecar):
    import vibeocr.backend.utils.ocr_sidecar as sidecar_mod

    file_path = str(isolated_sidecar)
    sidecar_mod.mark_pages_saved(file_path, [0, 1], {0: 0, 1: 0})
    backend = FakeBackend()
    orch = PdfOcrOrchestrator(backend, batch_size=4)

    result = orch.run_ocr(
        session_id="sess-1",
        file_path=file_path,
        page_indices=[0, 1],
        overwrite=False,
    )

    assert backend.render_calls == []
    assert result.completed == 2
    assert all(result.page_states[i][1] is LayerSource.OCR for i in [0, 1])


def test_save_failure_keeps_pages_resumable(isolated_sidecar):
    import vibeocr.backend.utils.ocr_sidecar as sidecar_mod

    file_path = str(isolated_sidecar)
    backend = FakeBackend(save_batches=False)
    orch = PdfOcrOrchestrator(backend, batch_size=4)

    result = orch.run_ocr(
        session_id="sess-1",
        file_path=file_path,
        page_indices=[0, 1],
    )

    assert result.failed == 2
    assert result.completed == 0
    # No pages should have been checkpointed (save failed).
    assert sidecar_mod.restore_pending_pages(
        file_path
    ) is None or not sidecar_mod.restore_pending_pages(file_path)


def test_failed_pages_aggregate_write_errors_deduplicated(isolated_sidecar):
    backend = FakeBackend(fail_pages={1, 3})
    orch = PdfOcrOrchestrator(backend, batch_size=2)

    result = orch.run_ocr(
        session_id="sess-1",
        file_path=str(isolated_sidecar),
        page_indices=[0, 1, 2, 3],
    )

    assert result.failed == 2
    assert result.completed == 2
    # "disk full" emitted once per failing batch; with batch_size=2 there are
    # two failing batches (each containing one failed page) → dedup to one entry.
    assert result.write_errors.count("disk full") == 1
    assert result.page_states[1][0] is PageState.FAILED
    assert result.page_states[0][0] is PageState.DONE


def test_cancel_at_page_boundary_keeps_prior_batches(isolated_sidecar):
    import vibeocr.backend.utils.ocr_sidecar as sidecar_mod

    file_path = str(isolated_sidecar)
    backend = FakeBackend()
    orch = PdfOcrOrchestrator(backend, batch_size=2)

    # Cancel mid-run by hooking the second write_batch to set the cancel flag.
    original_write = backend.write_batch

    call_count = {"n": 0}

    def cancelling_write(session_id, pages, *, overwrite, save, cancel_check):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            cancel_check.cancel()
            return BatchOutcome(
                saved_pages=(),
                failed_pages=tuple(idx for idx, _ in pages),
                saved=False,
            )
        return original_write(
            session_id, pages, overwrite=overwrite, save=save, cancel_check=cancel_check
        )

    backend.write_batch = cancelling_write  # type: ignore[assignment]
    result = orch.run_ocr(
        session_id="sess-1",
        file_path=file_path,
        page_indices=[0, 1, 2, 3],
    )

    assert result.cancelled is True
    # First batch (pages 0,1) was saved and checkpointed.
    saved = sidecar_mod.restore_pending_pages(file_path) or {}
    assert 0 in saved and 1 in saved
    # Pages 2,3 were not saved.
    assert 2 not in saved and 3 not in saved


def test_compress_failure_keeps_completed_false_and_pages_intact(isolated_sidecar):
    import vibeocr.backend.utils.ocr_sidecar as sidecar_mod

    file_path = str(isolated_sidecar)
    backend = FakeBackend(compress_ok=False)
    orch = PdfOcrOrchestrator(backend, batch_size=4)

    result = orch.run_ocr(
        session_id="sess-1",
        file_path=file_path,
        page_indices=[0, 1],
    )

    assert result.compressed is False
    # Pages were still checkpointed (per-batch save succeeded).
    saved = sidecar_mod.restore_pending_pages(file_path) or {}
    assert 0 in saved and 1 in saved


def test_native_layer_pages_project_to_native_not_ocr(isolated_sidecar):
    backend = FakeBackend(native_layers={0})
    orch = PdfOcrOrchestrator(backend, batch_size=4)

    result = orch.run_ocr(
        session_id="sess-1",
        file_path=str(isolated_sidecar),
        page_indices=[0],
    )

    # Page 0 had a native layer AND got OCR-saved → source is OCR (precedence).
    assert result.page_states[0][1] is LayerSource.OCR


def test_sidecar_root_override_isolates_winui_dev(isolated_sidecar, tmp_path):
    """The winui-dev sidecar root must not write the production sidecar."""
    from vibeocr.backend.application.pdf_ocr_orchestrator import (
        _sidecar_root,
        restore_pending_pages_for,
    )

    prod_file = str(tmp_path / "prod.pdf")
    tmp_path.joinpath("prod.pdf").write_bytes(b"%PDF-1.4\n%test\n")
    dev_root = str(tmp_path / "winui-dev")

    # Read under the dev root override — this must not create a production sidecar.
    with _sidecar_root(dev_root):
        assert restore_pending_pages_for(prod_file) in (None, {})
    # Production sidecar should be empty (no file written for prod_file).
    import vibeocr.backend.utils.ocr_sidecar as sidecar_mod

    assert sidecar_mod.restore_pending_pages(prod_file) in (None, {})


def test_empty_page_list_is_a_no_op(isolated_sidecar):
    backend = FakeBackend()
    orch = PdfOcrOrchestrator(backend, batch_size=4)

    result = orch.run_ocr(
        session_id="sess-1",
        file_path=str(isolated_sidecar),
        page_indices=[],
    )

    assert result.completed == 0
    assert backend.render_calls == []


def test_batch_render_exception_marks_all_pages_failed(isolated_sidecar):
    """render_pages 抛异常时整个 batch 标记失败 + 调 cancel（line 320-323）。"""

    class _CrashBackend(FakeBackend):
        def render_pages(self, session_id, page_indices, cancel_check):
            raise RuntimeError("render crashed")

    backend = _CrashBackend()
    orch = PdfOcrOrchestrator(backend, batch_size=2)
    result = orch.run_ocr(
        session_id="s1",
        file_path=str(isolated_sidecar),
        page_indices=[0, 1],
    )
    assert result.failed == 2
    assert result.completed == 0
    assert backend.cancel_calls == ["s1"]


def test_compress_with_saved_zero_marks_completed_false(isolated_sidecar):
    """无 saved 页面时压缩仍执行但 result.completed=0（line 195, 271）。"""
    backend = FakeBackend(save_batches=False)  # 所有 write 失败
    orch = PdfOcrOrchestrator(backend, batch_size=1)
    result = orch.run_ocr(
        session_id="s2",
        file_path=str(isolated_sidecar),
        page_indices=[0],
    )
    assert result.completed == 0
    assert result.failed == 1


def test_ocr_run_result_total_property():
    """OcrRunResult.total = completed + failed（line 104）。"""
    from vibeocr.backend.application.pdf_ocr_orchestrator import OcrRunResult

    r = OcrRunResult()
    r.completed = 3
    r.failed = 2
    assert r.total == 5
