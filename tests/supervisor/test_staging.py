"""Tests for the InputStager: quotas, sanitisation, per-item isolation, cleanup."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vibeocr.backend.supervisor.jobs.staging import (
    InputStager,
    StagedInput,
    StagingQuotaError,
    _safe_stem,
)
from vibeocr.runtime_contracts import ItemState

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def stager(tmp_path: Path) -> InputStager:
    return InputStager(
        root=tmp_path / "staging",
        max_file_count=4,
        max_total_bytes=1024,
        max_per_file_bytes=512,
    )


def _uploads(*sizes: int) -> list[tuple[str, str | None, bytes]]:
    return [(f"file{i}.png", "image/png", b"x" * s) for i, s in enumerate(sizes)]


def test_stage_job_preserves_order_and_assigns_item_ids(stager: InputStager) -> None:
    staged = stager.stage_job("job-1", _uploads(10, 20, 30))
    assert [s.item_id for s in staged] == ["it-0000", "it-0001", "it-0002"]
    assert [s.size_bytes for s in staged] == [10, 20, 30]
    assert all(s.path.exists() for s in staged)


def test_stage_job_rejects_empty(stager: InputStager) -> None:
    with pytest.raises(StagingQuotaError):
        stager.stage_job("job-1", [])


def test_stage_job_rejects_too_many_files(stager: InputStager) -> None:
    with pytest.raises(StagingQuotaError, match="too many files"):
        stager.stage_job("job-1", _uploads(1, 1, 1, 1, 1))


def test_stage_job_rejects_total_too_large(stager: InputStager) -> None:
    # 4 files x 300 bytes = 1200 > 1024 total
    with pytest.raises(StagingQuotaError, match="request too large"):
        stager.stage_job("job-1", _uploads(300, 300, 300, 300))


def test_stage_job_rejects_oversized_single_file(stager: InputStager) -> None:
    with pytest.raises(StagingQuotaError, match="per-file limit"):
        stager.stage_job("job-1", _uploads(600))


def test_stage_with_item_errors_isolates_oversized(stager: InputStager) -> None:
    staged, items = stager.stage_job_with_item_errors(
        "job-1", [("ok.png", None, b"x" * 10), ("big.png", None, b"x" * 600)]
    )
    assert len(staged) == 1
    assert staged[0].item_id == "it-0000"
    assert items[0].state is ItemState.QUEUED
    assert items[1].state is ItemState.FAILED
    assert "QUOTA_EXCEEDED" in (items[1].error or "")


def test_safe_stem_strips_unsafe_characters() -> None:
    # basename is applied first; splitext removes the extension.
    assert _safe_stem("../../etc/passwd") == "passwd"
    assert _safe_stem("..\\..\\evil<>.exe") == "evil"  # ext stripped, <> -> _
    assert _safe_stem("") == "input"
    assert _safe_stem("a" * 200) == "a" * 64


def test_release_removes_job_dir(stager: InputStager) -> None:
    staged = stager.stage_job("job-1", _uploads(10))
    job_dir = staged[0].path.parent
    assert job_dir.exists()
    stager.release("job-1")
    assert not job_dir.exists()


def test_release_all_removes_everything(stager: InputStager) -> None:
    stager.stage_job("job-1", _uploads(10))
    stager.stage_job("job-2", _uploads(10))
    removed = stager.release_all()
    assert removed == 2
    assert not any(p.is_dir() for p in stager.root.iterdir()) if stager.root.exists() else True


def test_cleanup_stale_leaves_known_dirs(stager: InputStager) -> None:
    stager.stage_job("job-known", _uploads(10))
    # Simulate a stale dir from a previous crashed instance.
    (stager.root / "stale-orphan-dir").mkdir(parents=True)
    removed = stager.cleanup_stale({"job-known"})
    assert removed == 1
    assert (stager.root / _safe_dir("job-known")).exists()


def _safe_dir(job_id: str) -> str:
    from vibeocr.backend.supervisor.jobs.staging import _safe_dir as impl

    return impl(job_id)


def test_filenames_never_used_as_path_directly(stager: InputStager) -> None:
    # A malicious upload name must not escape the staging root.
    staged: list[StagedInput] = stager.stage_job(
        "job-1", [("..\\..\\windows\\system32\\evil.dll", None, b"MZ")]
    )
    target = staged[0].path.resolve()
    root = stager.root.resolve()
    assert root in target.parents
