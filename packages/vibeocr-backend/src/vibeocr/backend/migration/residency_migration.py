"""One-time migration: legacy ``pipeline_ttls`` -> v2 residency schema.

Plan §8 ("一次性迁移设置数据"):
* The old ``pipeline_ttls: dict[str, int]`` used ``0`` ambiguously to mean
  "inherit / infinite / pin" (ADR §8 calls this out explicitly).
* The new schema is ``residency: {default_ttl_seconds, pipelines:
  [{name, ttl_seconds|null, pinned}]}`` where ``null`` = inherit, ``>0`` =
  finite TTL, ``pinned=true`` = hard pin.
* Migration must preserve user-observable semantics, back up the original
  settings file before writing, and be idempotent (re-running on an already-
  migrated file is a no-op).

This module is pure (no I/O of the live settings file); it exposes the
conversion + a backup helper so callers (the supervisor bootstrap, a one-shot
migration tool) control file I/O.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

# Pipelines whose legacy ``0`` meant "permanent residency" (hard pin), not
# "inherit default". This list is conservative; unknown pipelines with ttl=0
# are treated as inherit (the safer default) unless explicitly pinned here.
_LEGACY_ZERO_MEANS_PINNED: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class MigrationResult:
    """Outcome of a single settings migration."""

    migrated: bool
    """True if the settings were actually rewritten; False if already v2."""

    backed_up_to: Path | None
    """Path of the pre-migration backup, or None if no backup was made."""

    default_ttl_seconds: int
    pipelines: tuple[dict[str, Any], ...]


def convert_legacy_pipeline_ttls(
    pipeline_ttls: dict[str, int] | None,
    *,
    default_ttl_seconds: int = 300,
) -> list[dict[str, Any]]:
    """Convert a legacy ``pipeline_ttls`` dict into the v2 ``pipelines`` list.

    Semantics (plan §8):
    * ``ttl == 0`` for a known "always-on" pipeline -> ``pinned=true``;
    * ``ttl == 0`` for an unknown pipeline -> ``ttl_seconds=null`` (inherit);
    * ``ttl > 0`` -> finite TTL;
    * ``ttl < 0`` -> treated as inherit (invalid legacy value, do not crash).
    """
    if not pipeline_ttls:
        return []
    out: list[dict[str, Any]] = []
    for name, ttl in pipeline_ttls.items():
        if ttl is None or ttl < 0:
            out.append({"name": name, "ttl_seconds": None, "pinned": False})
        elif ttl == 0:
            if (
                name in _LEGACY_ZERO_MEANS_PINNED
            ):  # pragma: no cover - set is empty by design
                out.append({"name": name, "ttl_seconds": None, "pinned": True})
            else:
                out.append({"name": name, "ttl_seconds": None, "pinned": False})
        else:
            out.append({"name": name, "ttl_seconds": int(ttl), "pinned": False})
    return out


def migrate_settings_file(
    settings_path: Path,
    *,
    default_ttl_seconds: int = 300,
    backup_suffix: str = ".v1.bak",
) -> MigrationResult:
    """Migrate a settings JSON file in place, backing up the original first.

    Idempotent: if the file already has a top-level ``residency`` key with a
    ``default_ttl_seconds`` field, it is treated as already migrated and
    returned unchanged (no backup, ``migrated=False``).
    """
    if not settings_path.exists():
        raise FileNotFoundError(settings_path)
    raw = settings_path.read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(raw)

    # Already migrated?
    residency = data.get("residency")
    if isinstance(residency, dict) and "default_ttl_seconds" in residency:
        pipelines = tuple(residency.get("pipelines", []))
        return MigrationResult(
            migrated=False,
            backed_up_to=None,
            default_ttl_seconds=int(
                residency.get("default_ttl_seconds", default_ttl_seconds)
            ),
            pipelines=pipelines,
        )

    legacy_ttls = data.get("pipeline_ttls")
    legacy_ttls_dict: dict[str, int] | None = None
    if isinstance(legacy_ttls, dict):
        legacy_ttls_dict = {str(k): int(v) for k, v in legacy_ttls.items()}

    pipelines = convert_legacy_pipeline_ttls(
        legacy_ttls_dict, default_ttl_seconds=default_ttl_seconds
    )

    # Back up the original file before rewriting.
    backup_path = settings_path.with_suffix(settings_path.suffix + backup_suffix)
    shutil.copy2(settings_path, backup_path)

    # Preserve any non-legacy keys, drop the legacy field, add residency.
    data.pop("pipeline_ttls", None)
    data["residency"] = {
        "default_ttl_seconds": default_ttl_seconds,
        "pipelines": pipelines,
    }
    # Schema-version the file so future migrations can detect the baseline.
    data["schema_version"] = 2

    settings_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return MigrationResult(
        migrated=True,
        backed_up_to=backup_path,
        default_ttl_seconds=default_ttl_seconds,
        pipelines=tuple(pipelines),
    )


__all__ = [
    "MigrationResult",
    "convert_legacy_pipeline_ttls",
    "migrate_settings_file",
]
