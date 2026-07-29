"""Shared fixtures for supervisor PDF route/client tests.

The fake adapter + null executor are reused by test_pdf_routes.py and
test_pdf_supervisor_client.py; centralising them here avoids fragile
cross-test-module imports and keeps each test file focused on assertions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from vibeocr.backend.ipc.schemas import (
    DetectTextLayersResponse,
    ModelDiff,
    MutateResponse,
    OpenResponse,
    PdfDocumentMirror,
    ProgressEvent,
    ProgressPhase,
    SaveResponse,
)
from vibeocr.backend.supervisor.app import create_app
from vibeocr.backend.supervisor.bootstrap import generate_session_token, new_instance_id
from vibeocr.backend.supervisor.module import SupervisorModule, SupervisorOptions

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from vibeocr.backend.supervisor.inference.mineru_adapter import MinerUProcessAdapter
    from vibeocr.runtime_contracts import CancelMode, ResidencyStatus, SettingsSnapshot


class FakePdfAdapter:
    """In-memory adapter recording every call; returns canned DTOs."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self._closed: set[str] = set()
        self._cancelled: set[str] = set()

    def _record(self, name: str, args: tuple, kwargs: dict) -> None:
        self.calls.append((name, args, kwargs))

    def open_session(self, path: str) -> OpenResponse:
        self._record("open_session", (path,), {})
        return OpenResponse(
            session_id="sid-1",
            model=PdfDocumentMirror(file_path=path, pages=[]),
        )

    def close_session(self, session_id: str) -> None:
        self._record("close_session", (session_id,), {})
        self._closed.add(session_id)

    def get_model(self, session_id: str) -> PdfDocumentMirror:
        self._record("get_model", (session_id,), {})
        return PdfDocumentMirror(file_path="doc.pdf", pages=[])

    def load_stream(self, session_id: str) -> Iterator[ProgressEvent]:
        self._record("load_stream", (session_id,), {})
        yield ProgressEvent(
            phase=ProgressPhase.LOAD, current=0, total=1, page_index=0, message="done"
        )

    def render_thumbnail(self, session_id: str, page: int, size: int = 160) -> bytes:
        self._record("render_thumbnail", (session_id, page), {"size": size})
        return b"\x89PNG\r\nthumb"

    def render_preview(self, session_id: str, page: int, dpi: int = 150) -> bytes:
        self._record("render_preview", (session_id, page), {"dpi": dpi})
        return b"\x89PNG\r\npreview"

    def detect_text_layers(self, session_id: str, page: int) -> DetectTextLayersResponse:
        self._record("detect_text_layers", (session_id, page), {})
        return DetectTextLayersResponse(text_layers=[])

    def rotate(self, session_id: str, pages: list[int], angle: int) -> MutateResponse:
        self._record("rotate", (session_id, pages, angle), {})
        return MutateResponse(diff=ModelDiff())

    def delete_pages(self, session_id: str, pages: list[int]) -> MutateResponse:
        self._record("delete_pages", (session_id, pages), {})
        return MutateResponse(diff=ModelDiff(structural_change=True))

    def insert_blank(
        self,
        session_id: str,
        after_index: int,
        width: float = 612.0,
        height: float = 792.0,
    ) -> MutateResponse:
        self._record(
            "insert_blank", (session_id, after_index), {"width": width, "height": height}
        )
        return MutateResponse(diff=ModelDiff(structural_change=True))

    def insert_from(
        self, session_id: str, source_path: str, after_index: int
    ) -> MutateResponse:
        self._record("insert_from", (session_id, source_path, after_index), {})
        return MutateResponse(diff=ModelDiff(structural_change=True))

    def move_page(
        self, session_id: str, from_index: int, to_index: int
    ) -> MutateResponse:
        self._record("move_page", (session_id, from_index, to_index), {})
        return MutateResponse(diff=ModelDiff(structural_change=True))

    def reorder(self, session_id: str, new_order: list[int]) -> MutateResponse:
        self._record("reorder", (session_id, new_order), {})
        return MutateResponse(diff=ModelDiff(structural_change=True))

    def add_text_layer(
        self,
        session_id: str,
        page: int,
        ocr_result: dict[str, Any],
        pdf_settings: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> MutateResponse:
        self._record(
            "add_text_layer",
            (session_id, page, ocr_result),
            {"pdf_settings": pdf_settings, "overwrite": overwrite},
        )
        return MutateResponse(diff=ModelDiff())

    def add_text_layer_batch(
        self,
        session_id: str,
        pages_data: list[dict[str, Any]],
        pdf_settings: dict[str, Any] | None = None,
        overwrite: bool = False,
        save: bool = False,
    ) -> MutateResponse:
        self._record(
            "add_text_layer_batch",
            (session_id, pages_data),
            {"pdf_settings": pdf_settings, "overwrite": overwrite, "save": save},
        )
        return MutateResponse(diff=ModelDiff(), extra={"saved": save})

    def rewrite_text_layer(
        self,
        session_id: str,
        page: int,
        text_blocks: list[Any],
        preproc_angle: int = 0,
        pdf_settings: dict[str, Any] | None = None,
    ) -> MutateResponse:
        self._record(
            "rewrite_text_layer",
            (session_id, page, text_blocks),
            {"preproc_angle": preproc_angle, "pdf_settings": pdf_settings},
        )
        return MutateResponse(diff=ModelDiff())

    def update_block_text(
        self, session_id: str, page: int, block_index: int, new_text: str
    ) -> MutateResponse:
        self._record(
            "update_block_text", (session_id, page, block_index, new_text), {}
        )
        return MutateResponse(diff=ModelDiff())

    def delete_text_layers_stream(
        self, session_id: str, pages: list[int]
    ) -> Iterator[ProgressEvent]:
        self._record("delete_text_layers_stream", (session_id, pages), {})
        yield ProgressEvent(
            phase=ProgressPhase.DELETE,
            current=1,
            total=1,
            page_payload={"residual_pages": []},
        )

    def save(
        self,
        session_id: str,
        path: str | None = None,
        pdf_settings: dict[str, Any] | None = None,
        *,
        rewrite_text_layers: bool = True,
    ) -> SaveResponse:
        self._record(
            "save",
            (session_id, path, pdf_settings),
            {"rewrite_text_layers": rewrite_text_layers},
        )
        return SaveResponse(path=path or "out.pdf", diff=ModelDiff())

    def save_transactional(self, session_id: str, target_path: str) -> str:
        # Fake: record the call and echo the target path (the real adapter
        # writes to a temp file + fsync + Path.replace onto target_path).
        self._record("save_transactional", (session_id, target_path), {})
        return target_path

    def cancel(self, session_id: str) -> None:
        self._record("cancel", (session_id,), {})
        self._cancelled.add(session_id)

    def reset_cancel(self, session_id: str) -> None:
        self._record("reset_cancel", (session_id,), {})
        self._cancelled.discard(session_id)

    def stop(self) -> None:
        self._record("stop", (), {})


class NullExecutor:
    """Minimal executor: fails every job immediately (PDF tests don't use jobs)."""

    def execute(self, record, staged) -> None:  # type: ignore[no-untyped-def]
        from vibeocr.runtime_contracts import JobState

        if record.state not in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
            record.transition(JobState.FAILED)

    def cancel_mode_for(self, record) -> CancelMode:  # type: ignore[no-untyped-def]
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

        return ResidencyStatus()

    def configure_settings(self, snapshot: SettingsSnapshot) -> ResidencyStatus:
        from vibeocr.runtime_contracts import ResidencyStatus

        return ResidencyStatus(
            default_ttl_seconds=snapshot.default_ttl_seconds,
            pipelines=snapshot.pipelines,
        )

    def close(self) -> None:
        return


@pytest.fixture()
def fake_pdf_adapter() -> FakePdfAdapter:
    return FakePdfAdapter()


@pytest.fixture()
def pdf_module(tmp_path: Path, fake_pdf_adapter: FakePdfAdapter) -> SupervisorModule:
    opts = SupervisorOptions(instance_id=new_instance_id())
    return SupervisorModule(
        options=opts,
        stager_root=tmp_path / "staging",
        executor=NullExecutor(),
        pdf_adapter=fake_pdf_adapter,
    )


@pytest.fixture()
def supervisor_token() -> str:
    return generate_session_token()


@pytest.fixture()
def pdf_app(pdf_module: SupervisorModule, supervisor_token: str):
    return create_app(pdf_module, supervisor_token)


@pytest.fixture(autouse=True)
def _stop_mineru_watchers_after_test(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Close leaked MinerU residency-watcher threads after each test.

    Several inference tests construct ``MinerUProcessAdapter`` instances (which
    spawn a resident ``_watch_residency`` daemon when ``_ensure_watcher_locked``
    runs) without calling ``.close()`` afterwards. Those daemon threads
    accumulate across the full pytest session and, combined with the
    pdf-supervisor-loop singleton and Qt widget creation in later view tests,
    have triggered ``Windows fatal exception: access violation`` crashes in the
    restricted CI session (run 30357452209, coverage job).

    This wraps ``_ensure_watcher_locked`` to track every adapter that starts a
    watcher and closes them after each test, so no watcher daemon outlives the
    test that created it. The lazy import keeps the backend module out of the
    client-side test path until a test actually touches MinerU.
    """
    from vibeocr.backend.supervisor.inference import mineru_adapter as _ma

    created: list = []
    original: Callable[..., None] = _ma.MinerUProcessAdapter._ensure_watcher_locked

    def _tracking(self: MinerUProcessAdapter) -> None:
        original(self)
        if self not in created:
            created.append(self)

    monkeypatch.setattr(
        _ma.MinerUProcessAdapter, "_ensure_watcher_locked", _tracking
    )
    yield
    for adapter in created:
        try:
            adapter.close()
        except Exception:
            # A test may have already torn the adapter down; never let cleanup
            # mask the real failure or break teardown.
            pass
