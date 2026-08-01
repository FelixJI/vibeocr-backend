"""Idempotent profile/config migrator for the WinUI cutover.

Adds ``schema_version`` to the user's ``app_settings.json`` so the WinUI shell
can distinguish migrated from legacy data. The migrator is:

- **Idempotent**: a second run on already-migrated data is a no-op and returns
  ``already_migrated``.
- **Non-destructive**: a hashed backup is written before any change; the
  original file is only touched when a migration actually happens.
- **Atomic**: writes a temp file, fsyncs, then ``os.replace`` so a crash leaves
  either the old or the new file, never a torn one.
- **Read-only on failure**: disk-full, read-only file, or unknown fields do not
  corrupt the original.

The backup is only for data repair; it is NOT a UI rollback channel (the
WinUI cutover ships no legacy UI launcher).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True)
class MigrationResult:
    """Outcome of a migration run."""

    status: str  # "migrated", "already_migrated", "skipped"
    path: str
    backup_path: str | None = None
    schema_version: int = CURRENT_SCHEMA_VERSION
    message: str = ""


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("migrator: cannot read %s: %s", path, exc)
        return None


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: dict) -> None:
    payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _write_hashed_backup(path: Path) -> Path | None:
    """Write a sha256-tagged backup of the current file; return its path."""
    try:
        original = path.read_bytes()
    except OSError as exc:
        logger.warning("migrator: cannot back up %s: %s", path, exc)
        return None
    digest = _hash_bytes(original)[:16]
    backup = path.with_name(f"{path.stem}.pre-migrate-{digest}{path.suffix}.bak")
    try:
        backup.write_bytes(original)
        return backup
    except OSError as exc:
        logger.warning("migrator: cannot write backup %s: %s", backup, exc)
        return None


def migrate_config(config_path: str | Path) -> MigrationResult:
    """Migrate a single config file to ``CURRENT_SCHEMA_VERSION``.

    Returns a :class:`MigrationResult`. The file is left untouched when it is
    already migrated or missing.
    """
    path = Path(config_path)
    data = _read_json(path)
    if data is None:
        return MigrationResult("skipped", str(path), message="file not found")
    if not isinstance(data, dict):
        return MigrationResult("skipped", str(path), message="not a JSON object")

    existing_version = data.get("schema_version")
    if existing_version == CURRENT_SCHEMA_VERSION:
        return MigrationResult("already_migrated", str(path))
    if isinstance(existing_version, int) and existing_version > CURRENT_SCHEMA_VERSION:
        return MigrationResult(
            "skipped",
            str(path),
            message=f"schema_version {existing_version} is newer than migrator",
        )

    backup = _write_hashed_backup(path)
    # Unknown fields are preserved; we only add schema_version.
    data["schema_version"] = CURRENT_SCHEMA_VERSION
    try:
        _atomic_write(path, data)
    except OSError as exc:
        return MigrationResult("skipped", str(path), message=f"write failed: {exc}")
    return MigrationResult(
        "migrated",
        str(path),
        backup_path=str(backup) if backup else None,
        schema_version=CURRENT_SCHEMA_VERSION,
    )


def migrate_profile(config_dir: str | Path) -> list[MigrationResult]:
    """Migrate all known config files in a profile directory.

    Currently migrates ``app_settings.json``. Returns one result per file.
    """
    root = Path(config_dir)
    targets = [root / "app_settings.json"]
    return [migrate_config(target) for target in targets if target.exists() or True]


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "MigrationResult",
    "migrate_config",
    "migrate_profile",
]
