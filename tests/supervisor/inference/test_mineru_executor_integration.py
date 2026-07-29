"""Integration test: v2 supervisor runs REAL MinerU parse end-to-end.

Proves the plan's Phase 5 requirement that the supervisor can execute
``MINERU_PARSE`` jobs through the supervisor-owned MinerU API subprocess.

Heavy: starts the mineru-api subprocess (downloads models on first run,
~GBs). Skipped in CI and anywhere ``mineru`` is not importable. Run locally:

    python -m pytest tests/supervisor/inference/test_mineru_executor_integration.py -m slow

Note on multi-file: the real mineru-api's acceptance of a single multi-file
``/file_parse`` request has not been exercised elsewhere in the codebase. If
this test fails at the HTTP layer (not the model layer), treat it as evidence
that ``MinerUService.file_parse`` should fall back to a per-file loop (the
historical ``MinerUBatchService.batch_commit`` pattern). See the TODO in
``services/mineru_service.py::file_parse``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from vibeocr.backend.supervisor.composition import build_supervisor
from vibeocr.runtime_contracts import TERMINAL_JOB_STATES, JobKind, JobPriority

if TYPE_CHECKING:
    from pathlib import Path

_MINERU_AVAILABLE = True
try:
    import mineru  # type: ignore  # noqa: F401
except Exception:
    _MINERU_AVAILABLE = False

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _MINERU_AVAILABLE, reason="mineru not installed"),
]


def _minimal_pdf_bytes(text: str = "Hello MinerU") -> bytes:
    """A minimal valid single-page PDF containing ``text``.

    Hand-written so the test has no binary fixture dependency.
    """
    # Minimal PDF with one page and a text show operator. mineru-api parses
    # this; we only assert non-empty structured output, not the exact glyphs.
    stream = f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode()
    content = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
    return (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >> endobj\n"
        b"4 0 obj " + content + b"\n"
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"xref\n0 6\n0000000000 65535 f \n"
        b"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
    )


def _wait_for_terminal(module, job_id: str, *, timeout: float = 600.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = module.status(job_id)
        if snap.state in TERMINAL_JOB_STATES:
            return
        time.sleep(0.5)
    raise AssertionError(f"job {job_id} did not reach terminal within {timeout}s")


def test_submit_mineru_parse_returns_structured_payload(tmp_path: Path) -> None:
    module, _handle = build_supervisor(use_mineru=True, stager_root=tmp_path / "staging")
    pdf_bytes = _minimal_pdf_bytes("Hello MinerU")

    ref = module.submit(
        kind=JobKind.MINERU_PARSE,
        priority=JobPriority.INTERACTIVE,
        uploads=[("input.pdf", "application/pdf", pdf_bytes)],
    )

    _wait_for_terminal(module, ref.job_id, timeout=600.0)
    snap = module.status(ref.job_id)
    assert snap.state in TERMINAL_JOB_STATES, f"job not terminal: {snap.state}"
    results = module.result(ref.job_id)
    assert len(results) == 1
    payload = results[0].payload

    # Structural assertions: the MinerU path must produce structured output,
    # not the broken {"text": str(result)} fallback.
    assert "raw_text" in payload, f"missing raw_text; keys={list(payload.keys())}"
    assert payload.get("pipeline_type") == "MinerU", (
        f"expected pipeline_type=='MinerU', got {payload.get('pipeline_type')!r}"
    )
    assert isinstance(payload.get("content_list"), list)
    assert isinstance(payload.get("text_blocks"), list)
    # MinerU produced markdown from the PDF.
    assert (payload.get("markdown_text") or "").strip(), (
        f"expected non-empty markdown_text, got {payload.get('markdown_text')!r}"
    )
