"""Durable local storage for supervisor settings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from vibeocr.runtime_contracts import SettingsSnapshot, parse_pipeline_spec


class RuntimeSettingsStoreError(RuntimeError):
    """Raised when persisted supervisor settings cannot be read or replaced."""


class RuntimeSettings(Protocol):
    """The local-substitutable settings seam used by ``SupervisorModule``."""

    def load(self, default: SettingsSnapshot) -> SettingsSnapshot: ...

    def replace(self, snapshot: SettingsSnapshot) -> None: ...


class RuntimeSettingsStore:
    """Atomically persist one Protocol ``SettingsSnapshot`` at a local path."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self, default: SettingsSnapshot) -> SettingsSnapshot:
        if not self.path.exists():
            return default
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeSettingsStoreError("invalid supervisor settings file") from exc
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise RuntimeSettingsStoreError("unsupported supervisor settings schema")
        payload = value.get("settings")
        if not isinstance(payload, dict):
            raise RuntimeSettingsStoreError("invalid supervisor settings payload")
        return _snapshot_from_payload(payload)

    def replace(self, snapshot: SettingsSnapshot) -> None:
        payload = _snapshot_payload(snapshot)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(
                    {"schema_version": 1, "settings": payload},
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeSettingsStoreError(
                "could not replace supervisor settings"
            ) from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _snapshot_payload(snapshot: SettingsSnapshot) -> dict[str, Any]:
    payload = snapshot.to_payload()
    if not isinstance(payload, dict):
        raise RuntimeSettingsStoreError("invalid supervisor settings payload")
    return payload


def _snapshot_from_payload(payload: dict[str, Any]) -> SettingsSnapshot:
    try:
        if payload.get("schema_version") != SettingsSnapshot().schema_version:
            raise ValueError("settings schema")
        residency = payload["residency"]
        if not isinstance(residency, dict):
            raise ValueError("residency")
        ttl = residency["default_ttl_seconds"]
        pipelines = residency["pipelines"]
        extra = payload.get("extra", {})
        source_ids = payload.get("download_source_ids", [])
        if (
            type(ttl) is not int
            or ttl < 0
            or not isinstance(pipelines, list)
            or not isinstance(extra, dict)
            or not isinstance(source_ids, list)
            or any(
                not isinstance(source_id, str) or not source_id
                for source_id in source_ids
            )
            or len(set(source_ids)) != len(source_ids)
        ):
            raise ValueError("settings shape")
        return SettingsSnapshot(
            default_ttl_seconds=ttl,
            pipelines=tuple(parse_pipeline_spec(item) for item in pipelines),
            extra=extra,
            download_source_ids=tuple(source_ids),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeSettingsStoreError("invalid supervisor settings payload") from exc


__all__ = ["RuntimeSettings", "RuntimeSettingsStore", "RuntimeSettingsStoreError"]
