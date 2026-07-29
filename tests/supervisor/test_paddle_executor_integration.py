"""Integration test: v2 supervisor runs REAL Paddle OCR end-to-end.

This proves the plan's Phase 4/8 requirement that the supervisor can actually
execute recognition jobs through the unified seam. It builds a supervisor with
the real ``PaddleExecutor`` (backed by the singleton ``OCRService``), submits
a one-element recognition job carrying a rendered-text image, polls until
terminal, and asserts the returned text contains the expected word.

Heavy: loads PaddlePaddle + the OCR model (~seconds, GBs). Skipped in CI and
on any environment without paddle; run locally with:

    $env:VIBEOCR_RUN_REAL_PADDLE_GATE = "1"
    python -m pytest tests/supervisor/test_paddle_executor_integration.py -m slow

The test is self-contained: it renders the input image with Pillow so there is
no binary fixture dependency.
"""

from __future__ import annotations

import io
import os
import time
from typing import TYPE_CHECKING

import pytest

from vibeocr.backend.supervisor.composition import build_supervisor
from vibeocr.runtime_contracts import TERMINAL_JOB_STATES, JobState

if TYPE_CHECKING:
    from pathlib import Path

_RUN_REAL_PADDLE_GATE = os.environ.get("VIBEOCR_RUN_REAL_PADDLE_GATE") == "1"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not _RUN_REAL_PADDLE_GATE,
        reason="set VIBEOCR_RUN_REAL_PADDLE_GATE=1 on the dedicated Paddle gate",
    ),
]


def _render_text_image(text: str) -> bytes:
    """Render ``text`` as a high-contrast PNG using Pillow (no font file needed)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (320, 80), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Default bitmap font is small; use a large bbox so OCR has clear glyphs.
    draw.text((20, 20), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _wait_for_terminal(module, job_id: str, *, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = module.status(job_id)
        if snap.state in TERMINAL_JOB_STATES:
            return
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not reach terminal within {timeout}s")


def test_submit_recognition_returns_real_text(tmp_path: Path) -> None:
    # Import only inside the explicitly selected real-hardware gate. Importing
    # Paddle and probing Torch during ordinary collection can crash the Windows
    # loader before pytest has a chance to deselect this slow test.
    pytest.importorskip("paddle")
    module, _handle = build_supervisor(use_real_paddle=True, stager_root=tmp_path / "staging")
    expected_word = "HELLO"
    image_bytes = _render_text_image(expected_word)

    ref = module.submit(
        kind=__import__("vibeocr.runtime_contracts", fromlist=["JobKind"]).JobKind.RECOGNITION,
        priority=__import__("vibeocr.runtime_contracts", fromlist=["JobPriority"]).JobPriority.INTERACTIVE,
        uploads=[("test.png", "image/png", image_bytes)],
    )

    _wait_for_terminal(module, ref.job_id, timeout=180.0)
    snap = module.status(ref.job_id)
    # The job should succeed (OCR ran, even if the bitmap font is imperfect).
    assert snap.state in (JobState.COMPLETED, JobState.COMPLETED_WITH_ERRORS), (
        f"job did not complete successfully: state={snap.state}"
    )
    results = module.result(ref.job_id)
    assert len(results) == 1
    payload = results[0].payload

    # --- Structural assertions (these are the real "OCR ran" proof) ---
    # The serializer must have produced the structured key set, NOT the old
    # broken fallback {"text": str(result)} which only carried a dataclass repr.
    # Guard against that regression explicitly: a real structured payload has
    # `raw_text` (not `text`) and `text_blocks` as a list.
    assert "raw_text" in payload, (
        f"payload missing 'raw_text' — serializer regression? payload keys: "
        f"{list(payload.keys())}"
    )
    # The old broken serializer produced only {"text": "<OCRResult repr>"}. If a
    # `text` key is present it must NOT be a dataclass repr (which always starts
    # with "OCRResult("), proving we did not regress to the str(result) fallback.
    if "text" in payload:
        assert not str(payload["text"]).startswith("OCRResult("), (
            "payload['text'] looks like the broken str(result) fallback"
        )
    assert payload.get("pipeline_type") == "OCR", (
        f"expected pipeline_type=='OCR', got {payload.get('pipeline_type')!r}"
    )
    text_blocks = payload.get("text_blocks")
    assert isinstance(text_blocks, list), (
        f"expected text_blocks list, got {type(text_blocks).__name__}"
    )
    # Paddle OCR must find at least one text block on a rendered glyph image.
    assert len(text_blocks) >= 1, (
        f"expected >=1 text block, got {len(text_blocks)}; raw_text={payload.get('raw_text')!r}"
    )
    block = text_blocks[0]
    for key in ("text", "score", "bbox"):
        assert key in block, f"text_block missing key {key!r}: {block}"
    assert isinstance(block["bbox"], list), (
        f"bbox must be JSON list (tuple->list conversion), got {type(block['bbox']).__name__}"
    )

    # Non-empty text proves the real Paddle predict actually decoded glyphs.
    text = (payload["raw_text"] or "").upper()
    assert text.strip(), f"expected non-empty OCR raw_text, got {text!r}"
