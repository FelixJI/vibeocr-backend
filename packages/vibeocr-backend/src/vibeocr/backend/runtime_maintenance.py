"""Transport-neutral Runtime maintenance state and profile descriptors."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from vibeocr.backend.runtime_manifest import (
    RuntimeComponent,
    RuntimeProfile,
    load_runtime_manifest,
)

EventSink = Callable[[dict[str, Any]], None]
MAINTENANCE_STATE_FILENAME = "runtime-maintenance.json"


def _timestamp() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class RuntimeProfileDescriptor:
    profile_id: str
    accelerator: str
    components: tuple[RuntimeComponent, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "accelerator": self.accelerator,
            "components": [component.to_payload() for component in self.components],
        }


def profile_descriptor(
    profile: RuntimeProfile,
    *,
    accelerator: str,
) -> RuntimeProfileDescriptor:
    return RuntimeProfileDescriptor(
        profile_id=profile.name,
        accelerator=accelerator,
        components=profile.components,
    )


class RuntimeMaintenanceReporter:
    """Own one operation's monotonic snapshots and optional event projection."""

    def __init__(
        self,
        *,
        state_root: Path,
        profile: RuntimeProfileDescriptor,
        event_sink: EventSink | None = None,
    ) -> None:
        self._state_root = state_root
        self._profile = profile
        self._event_sink = event_sink
        self._operation: str | None = None
        self._operation_id: str | None = None
        self._sequence = 0
        self._snapshot: dict[str, Any] | None = None
        self._message_code: str | None = None

    @property
    def profile(self) -> RuntimeProfileDescriptor:
        return self._profile

    @property
    def snapshot(self) -> dict[str, Any] | None:
        return dict(self._snapshot) if self._snapshot is not None else None

    def start(self, operation: str, *, total_steps: int) -> None:
        self._operation = operation
        self._operation_id = str(uuid4())
        self._sequence = 0
        self._snapshot = None
        self.advance(
            phase="validate_binding",
            current=1,
            total=total_steps,
            message_code="runtime.validate_binding",
        )

    def advance(
        self,
        *,
        phase: str,
        current: int,
        total: int | None,
        message_code: str,
        component_id: str | None = None,
    ) -> None:
        self._publish(
            event_type="progress",
            operation_state="running",
            phase=phase,
            progress={
                "unit": "steps",
                "current": current,
                **({"total": total} if total is not None else {}),
            },
            message_code=message_code,
            component_id=component_id,
        )

    def heartbeat(self, *, message_code: str) -> None:
        if self._snapshot is None:
            return
        self._publish(
            event_type="heartbeat",
            operation_state="running",
            phase=str(self._snapshot["phase"]),
            progress=self._snapshot.get("progress"),
            message_code=message_code,
            component_id=self._snapshot.get("component_id"),
        )

    def succeed(
        self,
        *,
        phase: str,
        current: int,
        total: int,
        message_code: str,
    ) -> None:
        self._publish(
            event_type="snapshot",
            operation_state="succeeded",
            phase=phase,
            progress={"unit": "steps", "current": current, "total": total},
            message_code=message_code,
        )

    def fail(self, *, message_code: str = "runtime.operation_failed") -> None:
        if self._operation is None or self._operation_id is None:
            return
        phase = str(self._snapshot["phase"]) if self._snapshot else "validate_binding"
        progress = self._snapshot.get("progress") if self._snapshot else None
        component_id = self._snapshot.get("component_id") if self._snapshot else None
        self._publish(
            event_type="snapshot",
            operation_state="failed",
            phase=phase,
            progress=progress,
            message_code=message_code,
            component_id=component_id,
        )

    def _publish(
        self,
        *,
        event_type: str,
        operation_state: str,
        phase: str,
        progress: dict[str, Any] | None,
        message_code: str,
        component_id: str | None = None,
    ) -> None:
        if self._operation is None or self._operation_id is None:
            raise RuntimeError("maintenance operation has not started")
        self._sequence += 1
        snapshot: dict[str, Any] = {
            "operation_id": self._operation_id,
            "sequence": self._sequence,
            "operation": self._operation,
            "operation_state": operation_state,
            "phase": phase,
            "profile_id": self._profile.profile_id,
            "updated_at": _timestamp(),
        }
        if component_id is not None:
            snapshot["component_id"] = component_id
        if progress is not None:
            snapshot["progress"] = progress
        self._snapshot = snapshot
        self._message_code = message_code
        self._persist()
        if self._event_sink is not None:
            self._event_sink(
                {
                    "protocol_version": 2,
                    "event_version": 1,
                    "event_type": event_type,
                    "operation": self._operation,
                    "snapshot": snapshot,
                    "message_code": message_code,
                }
            )

    def _persist(self) -> None:
        if self._snapshot is None:
            return
        self._state_root.mkdir(parents=True, exist_ok=True)
        target = self._state_root / MAINTENANCE_STATE_FILENAME
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "snapshot": self._snapshot,
                        "message_code": self._message_code,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def runtime_status_from_environment(
    instance_id: str,
    service_state: str,
) -> dict[str, Any]:
    """Build the HTTP status snapshot from installer-owned environment/state."""
    manifest_value = os.environ.get("VIBEOCR_RUNTIME_MANIFEST")
    accelerator = os.environ.get("VIBEOCR_RUNTIME_ACCELERATOR")
    state_root_value = os.environ.get("VIBEOCR_RUNTIME_STATE_ROOT")
    if not manifest_value or not accelerator or not state_root_value:
        raise RuntimeError("Runtime status environment is incomplete")
    # Runtime installation verified the full release asset set before launch.
    # Status polling only needs the immutable descriptor and must not rehash the
    # large Python/runtime archives on every HTTP request.
    manifest = load_runtime_manifest(manifest_value, verify_artifacts=False)
    plan = "win-x64-cpu" if accelerator == "cpu" else "win-x64-cu126"
    descriptor = profile_descriptor(manifest.profiles[plan], accelerator=accelerator)
    components = [
        {**component.to_payload(), "state": "ready"}
        for component in descriptor.components
    ]
    maintenance = None
    state_path = Path(state_root_value) / MAINTENANCE_STATE_FILENAME
    try:
        persisted = json.loads(state_path.read_text(encoding="utf-8"))
        snapshot = persisted["snapshot"]
        if isinstance(snapshot, dict):
            maintenance = dict(snapshot)
            message_code = persisted.get("message_code")
            if isinstance(message_code, str):
                maintenance["message_code"] = message_code
    except (OSError, ValueError, KeyError, TypeError):
        maintenance = None
    if maintenance is not None and maintenance.get("operation_state") in {
        "queued",
        "running",
    }:
        service_state = "maintenance"
    from vibeocr.backend import __version__

    return {
        "schema_version": 2,
        "instance_id": instance_id,
        "service_state": service_state,
        "backend_version": __version__,
        "profile": {
            "profile_id": descriptor.profile_id,
            "accelerator": descriptor.accelerator,
            "components": components,
        },
        "maintenance": maintenance,
    }


__all__ = [
    "EventSink",
    "MAINTENANCE_STATE_FILENAME",
    "RuntimeMaintenanceReporter",
    "RuntimeProfileDescriptor",
    "profile_descriptor",
    "runtime_status_from_environment",
]
