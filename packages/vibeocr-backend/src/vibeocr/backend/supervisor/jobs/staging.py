"""InputStager: spool uploaded files into job-private temp directories.

Responsibilities (plan §6 Phase 2):

* Move multipart ``UploadFile`` bytes into a job-private staging directory.
* Sanitise uploaded filenames — the server generates internal item ids and
  never trusts the upload filename as a path.
* Enforce request quotas: max file count, max total bytes, max per-file bytes.
* Support per-item independent failure (an oversized item fails alone).
* Release staging on job purge/shutdown with bounded Windows-friendly retry.
"""

from __future__ import annotations

import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from vibeocr.runtime_contracts import ItemState, JobItem


class StagingQuotaError(ValueError):
    """Raised when an upload violates the configured quota."""


class InputExpiredError(ValueError):
    """Raised when retry input retention has elapsed or become unavailable."""


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_stem(original: str) -> str:
    """Reduce an untrusted filename to a filesystem-safe stem.

    We never use the original name as the stored path; we only keep a
    display-friendly suffix. Collisions are avoided by appending a short
    unique token at write time.
    """
    base = Path(original or "").name
    name = Path(base).stem
    cleaned = _SAFE_NAME_RE.sub("_", name).strip("._-") or "input"
    return cleaned[:64]


@dataclass(slots=True)
class StagedInput:
    """A single staged item: internal path + display name + item id."""

    item_id: str
    display_name: str
    path: Path
    size_bytes: int
    content_type: str | None = None


@dataclass
class InputStager:
    """Spool uploaded files into job-private temp directories.

    The stager is transport-agnostic: callers pass already-read bytes. This
    keeps it unit-testable without spinning up Starlette ``UploadFile``.
    """

    root: Path
    max_file_count: int = 64
    max_total_bytes: int = 256 * 1024 * 1024
    max_per_file_bytes: int = 64 * 1024 * 1024
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _staged_by_job: dict[str, dict[str, StagedInput]] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage
    # ------------------------------------------------------------------

    def stage_job(
        self,
        job_id: str,
        uploads: list[tuple[str, str | None, bytes]],
    ) -> list[StagedInput]:
        """Stage ``uploads`` for ``job_id`` preserving input order."""
        with self._lock:
            staged, _items = self._stage_job_locked(
                job_id, uploads, per_item_errors=False
            )
            return staged

    def stage_job_with_item_errors(
        self,
        job_id: str,
        uploads: list[tuple[str, str | None, bytes]],
    ) -> tuple[list[StagedInput], list[JobItem]]:
        """Stage with per-item isolation: oversized items fail alone."""
        with self._lock:
            return self._stage_job_locked(job_id, uploads, per_item_errors=True)

    def _stage_job_locked(
        self,
        job_id: str,
        uploads: list[tuple[str, str | None, bytes]],
        *,
        per_item_errors: bool,
    ) -> tuple[list[StagedInput], list[JobItem]]:
        if len(uploads) == 0:
            raise StagingQuotaError("at least one file is required")
        if len(uploads) > self.max_file_count:
            raise StagingQuotaError(
                f"too many files: {len(uploads)} > {self.max_file_count}"
            )
        total = sum(len(data) for _, _, data in uploads)
        if total > self.max_total_bytes:
            raise StagingQuotaError(
                f"request too large: {total} > {self.max_total_bytes}"
            )
        job_dir = self.root / _safe_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=False)
        staged: list[StagedInput] = []
        items: list[JobItem] = []
        for idx, (display_name, content_type, data) in enumerate(uploads):
            item_id = f"it-{idx:04d}"
            stem = _safe_stem(display_name)
            if len(data) > self.max_per_file_bytes:
                if per_item_errors:
                    items.append(
                        JobItem(
                            item_id=item_id,
                            display_name=display_name or stem,
                            state=ItemState.FAILED,
                            error="QUOTA_EXCEEDED: per-file limit",
                        )
                    )
                    continue
                # Strict mode: roll back the whole job dir and fail.
                self._safe_rmtree(job_dir)
                raise StagingQuotaError(
                    f"item {item_id} exceeds per-file limit: "
                    f"{len(data)} > {self.max_per_file_bytes}"
                )
            unique = f"{idx:04d}-{stem}-{uuid.uuid4().hex[:8]}"
            target = job_dir / unique
            target.write_bytes(data)
            staged.append(
                StagedInput(
                    item_id=item_id,
                    display_name=display_name or stem,
                    path=target,
                    size_bytes=len(data),
                    content_type=content_type,
                )
            )
            if per_item_errors:
                items.append(
                    JobItem(
                        item_id=item_id,
                        display_name=display_name or stem,
                        state=ItemState.QUEUED,
                    )
                )
        self._staged_by_job[job_id] = {entry.item_id: entry for entry in staged}
        return staged, items

    def clone_for_retry(
        self,
        *,
        source_job_id: str,
        retry_job_id: str,
        source_to_retry_item_ids: list[tuple[str, str]],
    ) -> list[StagedInput]:
        """Clone retained source inputs into a retry job, keyed by item id."""
        with self._lock:
            source = self._staged_by_job.get(source_job_id, {})
            missing = [
                source_id
                for source_id, _retry_id in source_to_retry_item_ids
                if source_id not in source or not source[source_id].path.exists()
            ]
            if missing:
                raise StagingQuotaError(
                    "retry input expired or unavailable: " + ", ".join(missing)
                )
            retry_dir = self.root / _safe_dir(retry_job_id)
            retry_dir.mkdir(parents=True, exist_ok=False)
            cloned: list[StagedInput] = []
            try:
                for ordinal, (source_id, retry_id) in enumerate(
                    source_to_retry_item_ids
                ):
                    original = source[source_id]
                    target = retry_dir / f"{ordinal:04d}-{uuid.uuid4().hex[:8]}"
                    shutil.copy2(original.path, target)
                    cloned.append(
                        StagedInput(
                            item_id=retry_id,
                            display_name=original.display_name,
                            path=target,
                            size_bytes=original.size_bytes,
                            content_type=original.content_type,
                        )
                    )
            except Exception:
                self._safe_rmtree(retry_dir)
                raise
            self._staged_by_job[retry_job_id] = {
                entry.item_id: entry for entry in cloned
            }
            return cloned

    def has_staged_item(self, job_id: str, item_id: str) -> bool:
        with self._lock:
            entry = self._staged_by_job.get(job_id, {}).get(item_id)
            return bool(entry is not None and entry.path.exists())

    # ------------------------------------------------------------------
    # Release
    # ------------------------------------------------------------------

    def release(self, job_id: str) -> None:
        with self._lock:
            job_dir = self.root / _safe_dir(job_id)
            self._safe_rmtree(job_dir)
            self._staged_by_job.pop(job_id, None)

    def release_all(self) -> int:
        """Remove every staging directory. Returns count removed."""
        with self._lock:
            removed = 0
            if not self.root.exists():
                return 0
            for entry in self.root.iterdir():
                if entry.is_dir():
                    self._safe_rmtree(entry)
                    removed += 1
            self._staged_by_job.clear()
            return removed

    def cleanup_stale(self, known_job_ids: set[str]) -> int:
        """Remove staging directories not belonging to ``known_job_ids``."""
        with self._lock:
            known_dirs = {_safe_dir(jid) for jid in known_job_ids}
            removed = 0
            if not self.root.exists():
                return 0
            for entry in self.root.iterdir():
                if entry.is_dir() and entry.name not in known_dirs:
                    self._safe_rmtree(entry)
                    removed += 1
            for job_id in list(self._staged_by_job):
                if job_id not in known_job_ids:
                    self._staged_by_job.pop(job_id, None)
            return removed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _safe_rmtree(
        self, path: Path, *, attempts: int = 5, delay: float = 0.05
    ) -> None:
        """Remove a directory tree with bounded retry for Windows locks."""
        if not path.exists():
            return
        last_exc: Exception | None = None
        for _ in range(attempts):
            try:
                shutil.rmtree(path)
                return
            except OSError as exc:  # pragma: no cover - timing dependent
                last_exc = exc
                time.sleep(delay)
        if last_exc is not None and path.exists():  # pragma: no cover
            raise last_exc


def _safe_dir(job_id: str) -> str:
    return _SAFE_NAME_RE.sub("-", job_id)[:128]


__all__ = [
    "InputExpiredError",
    "InputStager",
    "StagedInput",
    "StagingQuotaError",
    "_safe_dir",
    "_safe_stem",
]
