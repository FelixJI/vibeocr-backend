"""Tests for the retry-clone + release branches of InputStager.

Existing test_staging.py covers stage/release; this file targets the
branches left behind: clone_for_retry (happy path + missing source),
has_staged_item, release_all, cleanup_stale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vibeocr.backend.supervisor.jobs.staging import (
    InputStager,
    StagingQuotaError,
)

if TYPE_CHECKING:
    from pathlib import Path


def _stager(tmp_path: Path, **kwargs) -> InputStager:  # type: ignore[no-untyped-def]
    return InputStager(root=tmp_path / "stage", **kwargs)


# ---------------------------------------------------------------------------
# clone_for_retry
# ---------------------------------------------------------------------------


def test_clone_for_retry_clones_retained_inputs(tmp_path: Path) -> None:
    """A successful clone copies each retained input into the retry job dir
    (lines 173-209)."""
    stager = _stager(tmp_path)
    staged = stager.stage_job("source-job", [("a.png", None, b"alpha")])
    source_id = staged[0].item_id
    cloned = stager.clone_for_retry(
        source_job_id="source-job",
        retry_job_id="retry-job",
        source_to_retry_item_ids=[(source_id, "new-it-0")],
    )
    assert len(cloned) == 1
    assert cloned[0].item_id == "new-it-0"
    assert cloned[0].path.exists()
    assert cloned[0].path.read_bytes() == b"alpha"


def test_clone_for_retry_raises_when_source_missing(tmp_path: Path) -> None:
    """A missing source raises StagingQuotaError (lines 175-183)."""
    stager = _stager(tmp_path)
    with pytest.raises(StagingQuotaError, match="retry input expired"):
        stager.clone_for_retry(
            source_job_id="never-staged",
            retry_job_id="retry-job",
            source_to_retry_item_ids=[("it-0", "new-it-0")],
        )


def test_clone_for_retry_raises_when_source_file_deleted(tmp_path: Path) -> None:
    """A source whose backing file was deleted raises StagingQuotaError."""
    stager = _stager(tmp_path)
    staged = stager.stage_job("source-job", [("a.png", None, b"x")])
    staged[0].path.unlink()  # delete the backing file
    with pytest.raises(StagingQuotaError, match="retry input expired"):
        stager.clone_for_retry(
            source_job_id="source-job",
            retry_job_id="retry-job",
            source_to_retry_item_ids=[(staged[0].item_id, "new-it-0")],
        )


def test_clone_for_retry_cleans_retry_dir_when_copy_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``shutil.copy2`` raises mid-clone, the retry dir is removed and the
    error re-raises (lines 203-205)."""
    import shutil

    stager = _stager(tmp_path)
    staged = stager.stage_job("source-job", [("a.png", None, b"x")])

    def boom_copy2(src, dst, *, follow_symlinks=True):  # type: ignore[no-untyped-def]
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copy2", boom_copy2)
    with pytest.raises(OSError, match="disk full"):
        stager.clone_for_retry(
            source_job_id="source-job",
            retry_job_id="retry-job",
            source_to_retry_item_ids=[(staged[0].item_id, "new-it-0")],
        )
    # The retry dir was cleaned up by the except handler.
    retry_dirs = [p for p in stager.root.iterdir() if p.name != _safe_dir_name("source-job")]
    assert retry_dirs == []


def _safe_dir_name(job_id: str) -> str:
    """Mirror _safe_dir for matching the source-job directory name."""
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "-", job_id)[:128]


# ---------------------------------------------------------------------------
# has_staged_item
# ---------------------------------------------------------------------------


def test_has_staged_item_returns_true_for_existing(tmp_path: Path) -> None:
    stager = _stager(tmp_path)
    staged = stager.stage_job("job-1", [("a.png", None, b"x")])
    assert stager.has_staged_item("job-1", staged[0].item_id) is True


def test_has_staged_item_returns_false_for_unknown(tmp_path: Path) -> None:
    stager = _stager(tmp_path)
    assert stager.has_staged_item("job-1", "it-0") is False


def test_has_staged_item_returns_false_when_backing_file_missing(
    tmp_path: Path,
) -> None:
    stager = _stager(tmp_path)
    staged = stager.stage_job("job-1", [("a.png", None, b"x")])
    staged[0].path.unlink()
    assert stager.has_staged_item("job-1", staged[0].item_id) is False


# ---------------------------------------------------------------------------
# release_all
# ---------------------------------------------------------------------------


def test_release_all_removes_every_directory(tmp_path: Path) -> None:
    stager = _stager(tmp_path)
    stager.stage_job("job-1", [("a.png", None, b"x")])
    stager.stage_job("job-2", [("b.png", None, b"y")])
    removed = stager.release_all()
    assert removed == 2
    assert not any(stager.root.iterdir())


def test_release_all_returns_zero_when_root_missing(tmp_path: Path) -> None:
    """A non-existent root yields 0 removed (line 231)."""
    stager = _stager(tmp_path)
    # Remove the root directory that __post_init__ created.
    import shutil

    shutil.rmtree(stager.root)
    assert stager.release_all() == 0


# ---------------------------------------------------------------------------
# cleanup_stale
# ---------------------------------------------------------------------------


def test_cleanup_stale_removes_unknown_job_dirs(tmp_path: Path) -> None:
    stager = _stager(tmp_path)
    stager.stage_job("job-1", [("a.png", None, b"x")])
    # Plant a stale directory directly under the root.
    (stager.root / "stale-orphan").mkdir()
    (stager.root / "stale-orphan" / "file").write_bytes(b"orphan")
    removed = stager.cleanup_stale({"job-1"})
    assert removed == 1
    assert not (stager.root / "stale-orphan").exists()


def test_cleanup_stale_returns_zero_when_root_missing(tmp_path: Path) -> None:
    """A non-existent root yields 0 removed (line 245)."""
    stager = _stager(tmp_path)
    import shutil

    shutil.rmtree(stager.root)
    assert stager.cleanup_stale(set()) == 0


def test_cleanup_stale_drops_unknown_internal_entries(tmp_path: Path) -> None:
    """Internal _staged_by_job entries for unknown jobs are pruned (line 252)."""
    stager = _stager(tmp_path)
    stager.stage_job("job-1", [("a.png", None, b"x")])
    assert "job-1" in stager._staged_by_job
    stager.cleanup_stale(set())  # job-1 is now stale
    assert "job-1" not in stager._staged_by_job


# ---------------------------------------------------------------------------
# release
# ---------------------------------------------------------------------------


def test_release_removes_job_directory(tmp_path: Path) -> None:
    stager = _stager(tmp_path)
    staged = stager.stage_job("job-1", [("a.png", None, b"x")])
    assert staged[0].path.exists()
    stager.release("job-1")
    assert not staged[0].path.exists()


def test_release_is_idempotent_for_unknown_job(tmp_path: Path) -> None:
    """Releasing a never-staged job is a no-op (line 262)."""
    stager = _stager(tmp_path)
    stager.release("never-staged")  # must not raise


# ---------------------------------------------------------------------------
# stage_job_with_item_errors: per-item isolation
# ---------------------------------------------------------------------------


def test_stage_job_with_item_errors_isolates_oversized_item(tmp_path: Path) -> None:
    """An oversized item is recorded as FAILED while the rest stage normally."""
    stager = InputStager(
        root=tmp_path / "stage",
        max_file_count=10,
        max_total_bytes=1000,
        max_per_file_bytes=5,
    )
    staged, items = stager.stage_job_with_item_errors(
        "job-1",
        [("small.png", None, b"x"), ("huge.png", None, b"way-too-big")],
    )
    # The small item staged; the huge item is a FAILED JobItem.
    assert len(staged) == 1
    from vibeocr.runtime_contracts import ItemState

    states = {item.item_id: item.state for item in items}
    assert states["it-0000"] is ItemState.QUEUED
    assert states["it-0001"] is ItemState.FAILED
