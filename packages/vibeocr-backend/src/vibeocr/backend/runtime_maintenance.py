"""Transport-neutral Runtime maintenance state and profile descriptors."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from vibeocr.backend.runtime_lock import RuntimeLockTimeout, RuntimeStoreLock
from vibeocr.backend.runtime_manifest import (
    ManifestError,
    RuntimeComponent,
    RuntimeProfile,
    load_runtime_manifest,
)

EventSink = Callable[[dict[str, Any]], None]
OPERATIONS_DIRECTORY = "runtime-operations"
COMMANDS_DIRECTORY = "runtime-commands"
TERMINAL_REPLAY_RETENTION = timedelta(days=7)
_COMPONENT_DISTRIBUTIONS = {
    "ocr_engine": "paddleocr",
    "document_parsing": "mineru",
    "pdf_document_tools": "pymupdf",
    "image_code_tools": "opencv-contrib-python",
    "runtime_host": "fastapi",
    "gpu_runtime": "torch",
}
_COMPONENT_IMPORTS = {
    "ocr_engine": "paddleocr",
    "document_parsing": "mineru",
    "pdf_document_tools": "fitz",
    "image_code_tools": "cv2",
    "runtime_host": "fastapi",
    "gpu_runtime": "torch",
}
_COMPONENT_PROBE_CACHE_TTL_SECONDS = 10.0
_component_probe_cache_lock = threading.Lock()
_component_probe_cache: dict[
    tuple[str, tuple[str, ...], int | None], tuple[float, dict[str, bool]]
] = {}


class RuntimeOperationError(RuntimeError):
    """Base error for durable Runtime maintenance control."""


class RuntimeOperationConflict(RuntimeOperationError):
    """An operation id is already bound to a different normalized intent."""


class RuntimeCapabilityError(RuntimeOperationError):
    """A requested Runtime capability is unavailable."""


class RuntimeInstallFailure(RuntimeOperationError):
    """Runtime installation failed after the request was accepted."""


class RuntimeCommandConflict(RuntimeOperationError):
    """A command id is already bound to a different command payload."""


class RuntimeOperationNotFound(RuntimeOperationError):
    """The requested operation is absent from the durable store."""


class RuntimeOperationCancelled(RuntimeOperationError):
    """Cooperative cancellation reached an operation checkpoint."""


class RuntimeOperationNotCancellable(RuntimeOperationError):
    """The target operation cannot be cancelled in its current state."""


class RuntimeOperationNotRetryable(RuntimeOperationError):
    """The target operation is not a failed/cancelled terminal operation."""


class RuntimeSourceIdentityMismatch(RuntimeOperationError):
    """A durable operation is bound to a different verified Runtime source."""


class RuntimeCursorExpired(RuntimeOperationError):
    def __init__(self, *, oldest_sequence: int, snapshot: dict[str, Any]) -> None:
        super().__init__("Runtime maintenance cursor has expired")
        self.oldest_sequence = oldest_sequence
        self.snapshot = dict(snapshot)


@dataclass(frozen=True, slots=True)
class RuntimeOperationStart:
    created: bool
    snapshot: dict[str, Any] | None
    event: dict[str, Any] | None = None


def _normalized(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _identity_path(root: Path, directory: str, value: str) -> Path:
    if not value:
        raise ValueError("Runtime operation and command ids must be non-empty")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return root / directory / digest


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _command_error(exc: Exception) -> dict[str, Any]:
    detail: dict[str, Any] = {}
    if isinstance(exc, RuntimeCursorExpired):
        detail = {
            "oldest_sequence": exc.oldest_sequence,
            "snapshot": exc.snapshot,
        }
    taxonomy: str | None = None
    if isinstance(exc, RuntimeLockTimeout):
        taxonomy = "busy"
    elif isinstance(exc, OSError):
        taxonomy = "io"
    elif isinstance(exc, RuntimeCapabilityError):
        taxonomy = "capability"
    elif isinstance(exc, (RuntimeSourceIdentityMismatch, ManifestError)):
        taxonomy = "identity"
    elif isinstance(exc, RuntimeInstallFailure):
        taxonomy = "install"
    return {
        "type": type(exc).__name__,
        "taxonomy": taxonomy,
        "message": str(exc),
        "detail": detail,
    }


def _raise_command_error(value: dict[str, Any]) -> None:
    message = str(value.get("message") or "Runtime command failed")
    taxonomy = value.get("taxonomy")
    taxonomy_types: dict[str, type[Exception]] = {
        "busy": RuntimeLockTimeout,
        "io": OSError,
        "capability": RuntimeCapabilityError,
        "identity": RuntimeSourceIdentityMismatch,
        "install": RuntimeInstallFailure,
    }
    if isinstance(taxonomy, str) and taxonomy in taxonomy_types:
        raise taxonomy_types[taxonomy](message)
    error_type = value.get("type")
    error_types: dict[str, type[Exception]] = {
        "RuntimeOperationConflict": RuntimeOperationConflict,
        "RuntimeCommandConflict": RuntimeCommandConflict,
        "RuntimeOperationNotFound": RuntimeOperationNotFound,
        "RuntimeOperationCancelled": RuntimeOperationCancelled,
        "RuntimeOperationNotCancellable": RuntimeOperationNotCancellable,
        "RuntimeOperationNotRetryable": RuntimeOperationNotRetryable,
        "RuntimeIdentityMismatch": RuntimeSourceIdentityMismatch,
        "RuntimeSourceIdentityMismatch": RuntimeSourceIdentityMismatch,
        "ValueError": ValueError,
    }
    exception_type = error_types.get(str(error_type), RuntimeOperationError)
    raise exception_type(message)


class RuntimeOperationStore:
    """Durable operation/command identity and append-only event journal."""

    def __init__(
        self,
        state_root: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._state_root = state_root
        self._lock = threading.RLock()
        self._clock = clock or (lambda: datetime.now(tz=UTC))

    def _operation_directory(self, operation_id: str) -> Path:
        return _identity_path(self._state_root, OPERATIONS_DIRECTORY, operation_id)

    def _metadata_path(self, operation_id: str) -> Path:
        return self._operation_directory(operation_id) / "metadata.json"

    def _events_path(self, operation_id: str) -> Path:
        return self._operation_directory(operation_id) / "events.jsonl"

    def _operation_lock(self, operation_id: str) -> RuntimeStoreLock:
        return RuntimeStoreLock(
            self._operation_directory(operation_id) / "ledger.lock",
            timeout=60.0,
        )

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeOperationNotFound(path.name) from exc
        if not isinstance(value, dict):
            raise RuntimeOperationError(f"invalid Runtime operation record: {path}")
        return value

    def _reconcile_locked(self, operation_id: str) -> dict[str, Any]:
        """Recover a durable journal tail that outran its metadata projection."""
        metadata_path = self._metadata_path(operation_id)
        metadata = self._read(metadata_path)
        events_path = self._events_path(operation_id)
        if not events_path.exists():
            if int(metadata.get("through_sequence", 0)) != 0:
                raise RuntimeOperationError("Runtime event journal is missing")
            return metadata
        raw = events_path.read_bytes()
        compaction = metadata.get("compaction_pending")
        if compaction is not None:
            if not isinstance(compaction, dict):
                raise RuntimeOperationError("Runtime compaction record is invalid")
            compacted_oldest = compaction.get("oldest_sequence")
            before_sha256 = compaction.get("before_sha256")
            after_sha256 = compaction.get("after_sha256")
            if (
                type(compacted_oldest) is not int
                or compacted_oldest < 1
                or not isinstance(before_sha256, str)
                or not isinstance(after_sha256, str)
            ):
                raise RuntimeOperationError("Runtime compaction record is invalid")
            digest = hashlib.sha256(raw).hexdigest()
            if digest == after_sha256:
                metadata["oldest_sequence"] = compacted_oldest
            elif digest != before_sha256:
                raise RuntimeOperationError("Runtime compaction journal is invalid")
            metadata.pop("compaction_pending", None)
            _atomic_json(metadata_path, metadata)
        complete_end = raw.rfind(b"\n") + 1
        if complete_end < len(raw):
            with events_path.open("r+b") as stream:
                stream.truncate(complete_end)
            raw = raw[:complete_end]
        try:
            events = [
                json.loads(line) for line in raw.decode("utf-8").splitlines() if line
            ]
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeOperationError("Runtime event journal is invalid") from exc
        if not events:
            if int(metadata.get("through_sequence", 0)) != 0:
                raise RuntimeOperationError("Runtime event journal is incomplete")
            return metadata
        sequences = [event.get("sequence") for event in events]
        oldest = int(metadata.get("oldest_sequence", 1))
        if (
            any(type(sequence) is not int for sequence in sequences)
            or sequences[0] != oldest
            or any(right != left + 1 for left, right in zip(sequences, sequences[1:]))
        ):
            raise RuntimeOperationError("Runtime event journal sequence is invalid")
        through = int(metadata.get("through_sequence", 0))
        tail = int(sequences[-1])
        if through > tail or (through and through < oldest - 1):
            raise RuntimeOperationError("Runtime event metadata outran its journal")
        recovered = [event for event in events if int(event["sequence"]) > through]
        if recovered and int(recovered[0]["sequence"]) != through + 1:
            raise RuntimeOperationError(
                "Runtime event journal has an unrecoverable gap"
            )
        if not recovered:
            return metadata
        for event in recovered:
            snapshot = event.get("snapshot")
            if not isinstance(snapshot, dict):
                raise RuntimeOperationError("Runtime event snapshot is invalid")
            metadata["through_sequence"] = int(event["sequence"])
            metadata["snapshot"] = snapshot
            metadata["message_code"] = event.get("message_code")
            projection = event.get("_projection")
            if isinstance(projection, dict):
                if projection.get("cancel_requested") is True:
                    metadata["cancel_requested"] = True
                failure = projection.get("failure")
                if isinstance(failure, dict):
                    metadata["failure"] = failure
        snapshot = metadata["snapshot"]
        metadata["terminal"] = snapshot.get("operation_state") in {
            "succeeded",
            "failed",
            "cancelled",
        }
        if metadata["terminal"] and metadata.get("replay_expires_at") is None:
            metadata["replay_expires_at"] = _timestamp(
                self._clock() + TERMINAL_REPLAY_RETENTION
            )
        _atomic_json(metadata_path, metadata)
        return metadata

    def latest_projection(self) -> dict[str, Any] | None:
        """Return the newest journal-backed status projection after recovery."""
        operations_root = self._state_root / OPERATIONS_DIRECTORY
        if not operations_root.exists():
            return None
        latest: tuple[bool, str, str, str, dict[str, Any]] | None = None
        with self._lock:
            for metadata_path in operations_root.glob("*/metadata.json"):
                metadata = self._read(metadata_path)
                operation_id = metadata.get("operation_id")
                if not isinstance(
                    operation_id, str
                ) or metadata_path != self._metadata_path(operation_id):
                    raise RuntimeOperationError(
                        "Runtime operation metadata identity is invalid"
                    )
                with self._operation_lock(operation_id):
                    metadata = self._reconcile_locked(operation_id)
                created_at = metadata.get("created_at")
                snapshot = metadata.get("snapshot")
                updated_at = (
                    snapshot.get("updated_at") if isinstance(snapshot, dict) else None
                )
                if (
                    not isinstance(created_at, str)
                    or not isinstance(snapshot, dict)
                    or not isinstance(updated_at, str)
                ):
                    raise RuntimeOperationError(
                        "Runtime operation status projection is invalid"
                    )
                candidate = (
                    snapshot.get("operation_state") in {"queued", "running"},
                    updated_at,
                    created_at,
                    operation_id,
                    metadata,
                )
                if latest is None or candidate[:4] > latest[:4]:
                    latest = candidate
        if latest is None:
            return None
        metadata = latest[4]
        projection = dict(metadata["snapshot"])
        message_code = metadata.get("message_code")
        if isinstance(message_code, str):
            projection["message_code"] = message_code
        return projection

    def start(
        self,
        operation_id: str,
        intent: dict[str, Any],
        *,
        source_operation_id: str | None = None,
        initial_snapshot: dict[str, Any] | None = None,
        initial_message_code: str = "runtime.validate_binding",
    ) -> RuntimeOperationStart:
        if initial_snapshot is None:
            initial_snapshot = {
                "operation_id": operation_id,
                "source_operation_id": source_operation_id,
                "sequence": 1,
                "operation": str(intent.get("operation", "inspect")),
                "operation_state": "running",
                "phase": "validate_binding",
                "profile_id": str(intent.get("profile_id", "unknown")),
                "updated_at": _timestamp(self._clock()),
            }
        metadata_path = self._metadata_path(operation_id)
        normalized = _normalized(intent)
        with self._lock, self._operation_lock(operation_id):
            if metadata_path.exists():
                metadata = self._reconcile_locked(operation_id)
                if metadata.get("normalized_intent") != normalized:
                    raise RuntimeOperationConflict(operation_id)
                snapshot = metadata.get("snapshot")
                if snapshot is None:
                    event = self._append_locked(
                        operation_id,
                        event_type="progress",
                        snapshot=initial_snapshot,
                        message_code=initial_message_code,
                    )
                    return RuntimeOperationStart(
                        created=True,
                        snapshot=dict(initial_snapshot),
                        event=event,
                    )
                return RuntimeOperationStart(
                    created=False,
                    snapshot=dict(snapshot) if isinstance(snapshot, dict) else None,
                )
            metadata = {
                "schema_version": 2,
                "operation_id": operation_id,
                "source_operation_id": source_operation_id,
                "intent": intent,
                "normalized_intent": normalized,
                "created_at": _timestamp(self._clock()),
                "oldest_sequence": 1,
                "through_sequence": 0,
                "snapshot": None,
                "message_code": None,
                "terminal": False,
                "cancel_requested": False,
                "replay_expires_at": None,
            }
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(metadata_path, metadata)
            event = self._append_locked(
                operation_id,
                event_type="progress",
                snapshot=initial_snapshot,
                message_code=initial_message_code,
            )
            return RuntimeOperationStart(
                created=True,
                snapshot=dict(initial_snapshot),
                event=event,
            )

    def append(
        self,
        operation_id: str,
        *,
        event_type: str,
        snapshot: dict[str, Any],
        message_code: str,
        message_args: dict[str, str] | None = None,
        fallback_message: str | None = None,
        failure: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock, self._operation_lock(operation_id):
            return self._append_locked(
                operation_id,
                event_type=event_type,
                snapshot=snapshot,
                message_code=message_code,
                message_args=message_args,
                fallback_message=fallback_message,
                failure=failure,
            )

    def _append_locked(
        self,
        operation_id: str,
        *,
        event_type: str,
        snapshot: dict[str, Any],
        message_code: str,
        message_args: dict[str, str] | None = None,
        fallback_message: str | None = None,
        cancel_requested: bool = False,
        failure: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata_path = self._metadata_path(operation_id)
        metadata = self._reconcile_locked(operation_id)
        expected = int(metadata["through_sequence"]) + 1
        sequence = snapshot.get("sequence")
        if snapshot.get("operation_id") != operation_id or sequence != expected:
            raise RuntimeOperationError(
                f"Runtime event sequence must be {expected} for {operation_id}"
            )
        event: dict[str, Any] = {
            "schema_version": 2,
            "protocol_version": 2,
            "event_version": 1,
            "event_type": event_type,
            "sequence": sequence,
            "operation": snapshot["operation"],
            "snapshot": snapshot,
            "message_code": message_code,
            "message_args": dict(message_args or {}),
        }
        if fallback_message is not None:
            event["fallback_message"] = fallback_message
        projection: dict[str, Any] = {}
        if cancel_requested:
            projection["cancel_requested"] = True
        if failure is not None:
            projection["failure"] = dict(failure)
        if projection:
            event["_projection"] = projection
        events_path = self._events_path(operation_id)
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(_normalized(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        metadata["through_sequence"] = sequence
        metadata["snapshot"] = snapshot
        metadata["message_code"] = message_code
        if cancel_requested:
            metadata["cancel_requested"] = True
        if failure is not None:
            metadata["failure"] = dict(failure)
        metadata["terminal"] = snapshot.get("operation_state") in {
            "succeeded",
            "failed",
            "cancelled",
        }
        if metadata["terminal"] and metadata.get("replay_expires_at") is None:
            metadata["replay_expires_at"] = _timestamp(
                self._clock() + TERMINAL_REPLAY_RETENTION
            )
        _atomic_json(metadata_path, metadata)
        return self._public_event(event)

    @staticmethod
    def _public_event(event: dict[str, Any]) -> dict[str, Any]:
        public = dict(event)
        public.pop("_projection", None)
        return public

    def snapshot(self, operation_id: str) -> dict[str, Any] | None:
        with self._lock, self._operation_lock(operation_id):
            metadata = self._reconcile_locked(operation_id)
            snapshot = metadata.get("snapshot")
            return dict(snapshot) if isinstance(snapshot, dict) else None

    def intent(self, operation_id: str) -> dict[str, Any]:
        """Return the immutable normalized intent bound to an operation id."""
        with self._lock, self._operation_lock(operation_id):
            metadata = self._reconcile_locked(operation_id)
            intent = metadata.get("intent")
            if not isinstance(intent, dict):
                raise RuntimeOperationError("Runtime operation has no intent")
            return dict(intent)

    def cancel_requested(self, operation_id: str) -> bool:
        with self._lock, self._operation_lock(operation_id):
            return bool(self._reconcile_locked(operation_id).get("cancel_requested"))

    def raise_replayed_failure(self, operation_id: str) -> None:
        with self._lock, self._operation_lock(operation_id):
            metadata = self._reconcile_locked(operation_id)
            failure = metadata.get("failure")
            if isinstance(failure, dict):
                _raise_command_error(failure)
            raise RuntimeInstallFailure(
                "Runtime operation previously failed before its error was persisted"
            )

    def request_cancel(
        self,
        operation_id: str,
        *,
        expected_sequence: int | None = None,
    ) -> dict[str, Any]:
        with self._lock, self._operation_lock(operation_id):
            metadata = self._reconcile_locked(operation_id)
            snapshot = metadata.get("snapshot")
            if not isinstance(snapshot, dict):
                raise RuntimeOperationError("Runtime operation has no snapshot")
            if metadata.get("cancel_requested"):
                return dict(snapshot)
            if (
                expected_sequence is not None
                and snapshot.get("sequence") != expected_sequence
            ):
                raise RuntimeOperationConflict("expected_sequence mismatch")
            if metadata.get("terminal"):
                raise RuntimeOperationNotCancellable(operation_id)
            cancel_snapshot = {
                **snapshot,
                "sequence": int(metadata["through_sequence"]) + 1,
                "updated_at": _timestamp(self._clock()),
            }
            self._append_locked(
                operation_id,
                event_type="snapshot",
                snapshot=cancel_snapshot,
                message_code="runtime.cancel_requested",
                cancel_requested=True,
            )
            return cancel_snapshot

    def observe(
        self, operation_id: str, *, after_sequence: int, limit: int
    ) -> dict[str, Any]:
        if after_sequence < 0 or limit < 1 or limit > 512:
            raise ValueError("invalid Runtime maintenance cursor page")
        with self._lock, self._operation_lock(operation_id):
            metadata = self._reconcile_locked(operation_id)
            snapshot = metadata.get("snapshot")
            if not isinstance(snapshot, dict):
                raise RuntimeOperationError("Runtime operation has no snapshot")
            oldest = int(metadata["oldest_sequence"])
            if after_sequence < oldest - 1:
                raise RuntimeCursorExpired(
                    oldest_sequence=oldest,
                    snapshot=snapshot,
                )
            retained: list[dict[str, Any]] = []
            events_path = self._events_path(operation_id)
            if events_path.exists():
                for line in events_path.read_text(encoding="utf-8").splitlines():
                    event = json.loads(line)
                    if int(event["sequence"]) > after_sequence:
                        retained.append(self._public_event(event))
            page = retained[:limit]
            through = (
                page[-1]["sequence"]
                if page
                else min(after_sequence, int(metadata["through_sequence"]))
            )
            return {
                "schema_version": 2,
                "operation_id": operation_id,
                "snapshot": snapshot,
                "events": page,
                "oldest_sequence": oldest,
                "through_sequence": through,
                "more": len(retained) > len(page),
                "replay_expires_at": metadata.get("replay_expires_at"),
            }

    def compact(self, operation_id: str, *, keep_last: int) -> None:
        if keep_last < 1:
            raise ValueError("keep_last must be positive")
        with self._lock, self._operation_lock(operation_id):
            metadata_path = self._metadata_path(operation_id)
            metadata = self._reconcile_locked(operation_id)
            if not metadata.get("terminal"):
                raise RuntimeOperationError(
                    "active Runtime operation cannot be compacted"
                )
            expires_at = metadata.get("replay_expires_at")
            if (
                not isinstance(expires_at, str)
                or _parse_timestamp(expires_at) > self._clock()
            ):
                raise RuntimeOperationError(
                    "terminal Runtime replay retention has not expired"
                )
            events_path = self._events_path(operation_id)
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]
            retained = events[-keep_last:]
            before_raw = events_path.read_bytes()
            after_raw = "".join(_normalized(event) + "\n" for event in retained).encode(
                "utf-8"
            )
            metadata["compaction_pending"] = {
                "oldest_sequence": retained[0]["sequence"],
                "before_sha256": hashlib.sha256(before_raw).hexdigest(),
                "after_sha256": hashlib.sha256(after_raw).hexdigest(),
            }
            _atomic_json(metadata_path, metadata)
            temporary = events_path.with_name(f".{events_path.name}.{uuid4().hex}.tmp")
            try:
                with temporary.open("wb") as stream:
                    stream.write(after_raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, events_path)
            finally:
                temporary.unlink(missing_ok=True)
            metadata["oldest_sequence"] = retained[0]["sequence"]
            metadata.pop("compaction_pending", None)
            _atomic_json(metadata_path, metadata)

    def apply_command(
        self,
        command_id: str,
        payload: dict[str, Any],
        apply: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        command_path = (
            _identity_path(self._state_root, COMMANDS_DIRECTORY, command_id)
            / "command.json"
        )
        normalized = _normalized(payload)
        command_lock = RuntimeStoreLock(
            command_path.with_name("command.lock"), timeout=3600.0
        )
        with self._lock, command_lock:
            if command_path.exists():
                command = self._read(command_path)
                if command.get("normalized_payload") != normalized:
                    raise RuntimeCommandConflict(command_id)
                outcome = command.get("outcome")
                if not isinstance(outcome, dict):
                    try:
                        result = apply()
                    except Exception as exc:
                        outcome = {"ok": False, "error": _command_error(exc)}
                    else:
                        outcome = {"ok": True, "result": result}
                    command["outcome"] = outcome
                    _atomic_json(command_path, command)
                if outcome.get("ok") is True and isinstance(
                    outcome.get("result"), dict
                ):
                    return dict(outcome["result"])
                error = outcome.get("error")
                if isinstance(error, dict):
                    _raise_command_error(error)
                raise RuntimeOperationError("Runtime command outcome is invalid")
            command_path.parent.mkdir(parents=True, exist_ok=True)
            reservation = {
                "schema_version": 1,
                "command_id": command_id,
                "payload": payload,
                "normalized_payload": normalized,
                "outcome": None,
            }
            _atomic_json(command_path, reservation)
            try:
                result = apply()
            except Exception as exc:
                reservation["outcome"] = {
                    "ok": False,
                    "error": _command_error(exc),
                }
                _atomic_json(command_path, reservation)
                raise
            reservation["outcome"] = {"ok": True, "result": result}
            _atomic_json(command_path, reservation)
            return dict(result)


def _timestamp(value: datetime | None = None) -> str:
    return (value or datetime.now(tz=UTC)).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


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


def runtime_source_identity(manifest: Any) -> dict[str, str]:
    return {
        "backend_version": manifest.backend_version,
        "backend_source_sha": manifest.source_commit,
        "runtime_manifest_sha256": manifest.sha256,
        "protocol_version": manifest.protocol_version,
        "protocol_manifest_sha256": manifest.protocol_manifest_sha256,
    }


def _distribution_versions(runtime_root: Path) -> dict[str, str]:
    site_packages = [runtime_root / "Lib" / "site-packages"]
    site_packages.extend((runtime_root / "lib").glob("python*/site-packages"))
    versions: dict[str, str] = {}
    for directory in site_packages:
        if not directory.is_dir():
            continue
        for distribution in importlib.metadata.Distribution.discover(
            path=[str(directory)]
        ):
            name = distribution.metadata.get("Name")
            if isinstance(name, str) and name:
                versions[re.sub(r"[-_.]+", "-", name).lower()] = distribution.version
    return versions


def probe_runtime_components(
    runtime_root: Path, component_ids: tuple[str, ...]
) -> dict[str, bool]:
    python = next(
        (
            candidate
            for candidate in (
                runtime_root / "Scripts" / "python.exe",
                runtime_root / "python.exe",
                runtime_root / "bin" / "python",
            )
            if candidate.is_file()
        ),
        runtime_root / "python.exe",
    )
    modules = {
        component_id: _COMPONENT_IMPORTS[component_id] for component_id in component_ids
    }
    script = (
        "import importlib,json,sys\n"
        "modules=json.loads(sys.argv[1])\n"
        "result={}\n"
        "for component_id,module in modules.items():\n"
        "  try:\n"
        "    importlib.import_module(module)\n"
        "  except BaseException:\n"
        "    result[component_id]=False\n"
        "  else:\n"
        "    result[component_id]=True\n"
        "print('VIBEOCR_COMPONENT_PROBE='+json.dumps(result,sort_keys=True))\n"
    )
    try:
        completed = subprocess.run(
            [str(python), "-I", "-c", script, json.dumps(modules, sort_keys=True)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {component_id: False for component_id in component_ids}
    prefix = "VIBEOCR_COMPONENT_PROBE="
    payload = next(
        (
            line[len(prefix) :]
            for line in reversed(completed.stdout.splitlines())
            if line.startswith(prefix)
        ),
        None,
    )
    if completed.returncode != 0 or payload is None:
        return {component_id: False for component_id in component_ids}
    try:
        value = json.loads(payload)
    except ValueError:
        return {component_id: False for component_id in component_ids}
    return {
        component_id: value.get(component_id) is True for component_id in component_ids
    }


def _cached_runtime_component_probe(
    runtime_root: Path, component_ids: tuple[str, ...]
) -> dict[str, bool]:
    marker = runtime_root / ".installed.json"
    try:
        marker_mtime = marker.stat().st_mtime_ns
    except OSError:
        marker_mtime = None
    key = (str(runtime_root.resolve()), component_ids, marker_mtime)
    now = time.monotonic()
    with _component_probe_cache_lock:
        cached = _component_probe_cache.get(key)
        if cached is not None and cached[0] > now:
            return dict(cached[1])
        result = probe_runtime_components(runtime_root, component_ids)
        expired = [
            existing_key
            for existing_key, (expires_at, _) in _component_probe_cache.items()
            if expires_at <= now
        ]
        for existing_key in expired:
            _component_probe_cache.pop(existing_key, None)
        _component_probe_cache[key] = (
            now + _COMPONENT_PROBE_CACHE_TTL_SECONDS,
            dict(result),
        )
        return result


def _component_statuses(
    descriptor: RuntimeProfileDescriptor,
    *,
    manifest: Any,
    runtime_root: Path | None,
    probe_results: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    marker: dict[str, Any] | None = None
    marker_path = runtime_root / ".installed.json" if runtime_root is not None else None
    if marker_path is not None:
        try:
            value = json.loads(marker_path.read_text(encoding="utf-8"))
            marker = value if isinstance(value, dict) else None
        except (OSError, ValueError):
            marker = None
    python_ready = runtime_root is not None and any(
        candidate.is_file()
        for candidate in (
            runtime_root / "Scripts" / "python.exe",
            runtime_root / "python.exe",
            runtime_root / "bin" / "python",
        )
    )
    expected_marker = {
        "schema_version": 1,
        "backend_version": manifest.backend_version,
        "manifest_sha256": manifest.sha256,
        "accelerator": descriptor.accelerator,
    }
    versions = _distribution_versions(runtime_root) if runtime_root is not None else {}
    statuses: list[dict[str, Any]] = []
    for component in descriptor.components:
        actual_version: str | None = None
        if marker is None or not python_ready:
            actual_state = "missing"
            drift_reason = "missing"
        elif marker != expected_marker:
            actual_state = "drifted"
            drift_reason = "identity_mismatch"
        elif component.version is None:
            actual_state = "ready"
            drift_reason = "none"
        else:
            distribution = _COMPONENT_DISTRIBUTIONS[component.component_id]
            actual_version = versions.get(re.sub(r"[-_.]+", "-", distribution).lower())
            if actual_version is None:
                actual_state = "missing"
                drift_reason = "missing"
            elif actual_version != component.version:
                actual_state = "drifted"
                drift_reason = "version_mismatch"
            else:
                actual_state = "ready"
                drift_reason = "none"
        if (
            actual_state == "ready"
            and probe_results is not None
            and not probe_results.get(component.component_id, False)
        ):
            actual_state = "drifted"
            drift_reason = "integrity_failed"
        statuses.append(
            {
                **component.to_payload(),
                "state": "ready" if actual_state == "ready" else "failed",
                "desired_state": "ready",
                "desired_version": component.version,
                "actual_state": actual_state,
                "actual_version": actual_version,
                "drift_reason": drift_reason,
                "repairable": actual_state != "ready",
            }
        )
    return statuses


def runtime_profile_status(
    manifest: Any,
    *,
    accelerator: str,
    runtime_root: Path | None,
    probe_results: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Project one desired/actual component view for every transport."""
    plan = "win-x64-cpu" if accelerator == "cpu" else "win-x64-cu126"
    descriptor = profile_descriptor(manifest.profiles[plan], accelerator=accelerator)
    if probe_results is None and runtime_root is not None:
        probe_results = _cached_runtime_component_probe(
            runtime_root,
            tuple(component.component_id for component in descriptor.components),
        )
    return {
        "profile_id": descriptor.profile_id,
        "accelerator": descriptor.accelerator,
        "components": _component_statuses(
            descriptor,
            manifest=manifest,
            runtime_root=runtime_root,
            probe_results=probe_results,
        ),
    }


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
        self._store = RuntimeOperationStore(state_root)
        self._profile = profile
        self._event_sink = event_sink
        self._operation: str | None = None
        self._operation_id: str | None = None
        self._sequence = 0
        self._snapshot: dict[str, Any] | None = None
        self._message_code: str | None = None
        self._requested_component_ids: tuple[str, ...] = ()
        self._effective_component_ids: tuple[str, ...] = ()
        self._source: dict[str, Any] | None = None
        self._source_operation_id: str | None = None

    @property
    def profile(self) -> RuntimeProfileDescriptor:
        return self._profile

    @property
    def snapshot(self) -> dict[str, Any] | None:
        return dict(self._snapshot) if self._snapshot is not None else None

    @property
    def operation_id(self) -> str | None:
        return self._operation_id

    def start(
        self,
        operation: str,
        *,
        total_steps: int,
        operation_id: str | None = None,
        component_ids: tuple[str, ...] = (),
        effective_component_ids: tuple[str, ...] = (),
        source: dict[str, Any] | None = None,
        source_operation_id: str | None = None,
        required_capabilities: tuple[str, ...] = (),
    ) -> bool:
        self._operation = operation
        self._operation_id = operation_id or str(uuid4())
        self._requested_component_ids = component_ids
        self._effective_component_ids = effective_component_ids
        self._source = dict(source) if source is not None else None
        self._source_operation_id = source_operation_id
        self._sequence = 0
        self._snapshot = None
        initial_snapshot: dict[str, Any] = {
            "operation_id": self._operation_id,
            "source_operation_id": self._source_operation_id,
            "sequence": 1,
            "operation": operation,
            "operation_state": "running",
            "phase": "validate_binding",
            "profile_id": self._profile.profile_id,
            "updated_at": _timestamp(),
            "progress": {"unit": "steps", "current": 1, "total": total_steps},
        }
        if self._requested_component_ids:
            initial_snapshot["requested_component_ids"] = list(
                self._requested_component_ids
            )
        if self._effective_component_ids:
            initial_snapshot["effective_component_ids"] = list(
                self._effective_component_ids
            )
        if self._source is not None:
            initial_snapshot["source"] = dict(self._source)
        started = self._store.start(
            self._operation_id,
            {
                "operation": operation,
                "profile_id": self._profile.profile_id,
                "component_ids": list(component_ids),
                "required_capabilities": list(required_capabilities),
                "source_identity": dict(source or {}),
                "source_operation_id": source_operation_id,
            },
            source_operation_id=source_operation_id,
            initial_snapshot=initial_snapshot,
            initial_message_code="runtime.validate_binding",
        )
        self._snapshot = started.snapshot
        self._sequence = (
            int(started.snapshot["sequence"]) if started.snapshot is not None else 0
        )
        self._message_code = "runtime.validate_binding"
        if (
            started.created
            and started.event is not None
            and self._event_sink is not None
        ):
            self._event_sink(started.event)
        if (
            not started.created
            and started.snapshot is not None
            and started.snapshot.get("operation_state") == "failed"
        ):
            self._store.raise_replayed_failure(self._operation_id)
        return started.created

    def advance(
        self,
        *,
        phase: str,
        current: int,
        total: int | None,
        message_code: str,
        component_id: str | None = None,
    ) -> None:
        self._abort_if_cancel_requested()
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
        self._abort_if_cancel_requested()
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
        self._abort_if_cancel_requested()
        self._publish(
            event_type="snapshot",
            operation_state="succeeded",
            phase=phase,
            progress={"unit": "steps", "current": current, "total": total},
            message_code=message_code,
        )

    def fail(
        self,
        error: Exception,
        *,
        message_code: str = "runtime.operation_failed",
    ) -> None:
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
            failure=_command_error(error),
        )

    def advance_measured(
        self,
        *,
        phase: str,
        unit: str,
        current: int,
        total: int,
        message_code: str,
        component_id: str | None = None,
        estimated_remaining_seconds: int | None = None,
    ) -> None:
        """Publish determinate progress only for a real items/bytes total."""
        if unit not in {"items", "bytes"}:
            raise ValueError("measured Runtime progress unit must be items or bytes")
        if total <= 0 or current < 0 or current > total:
            raise ValueError(
                "measured Runtime progress must have a positive real total"
            )
        if estimated_remaining_seconds is not None and estimated_remaining_seconds < 0:
            raise ValueError("estimated_remaining_seconds must be non-negative")
        self._abort_if_cancel_requested()
        progress: dict[str, Any] = {
            "unit": unit,
            "current": current,
            "total": total,
        }
        if estimated_remaining_seconds is not None:
            progress["estimated_remaining_seconds"] = estimated_remaining_seconds
        self._publish(
            event_type="progress",
            operation_state="running",
            phase=phase,
            progress=progress,
            message_code=message_code,
            component_id=component_id,
        )

    def cancel(self, *, message_code: str = "runtime.operation_cancelled") -> None:
        if self._snapshot is None:
            raise RuntimeOperationError("Runtime operation has no active snapshot")
        if self._snapshot.get("operation_state") in {
            "succeeded",
            "failed",
            "cancelled",
        }:
            raise RuntimeOperationError("Runtime operation is not cancellable")
        self._publish(
            event_type="snapshot",
            operation_state="cancelled",
            phase=str(self._snapshot["phase"]),
            progress=self._snapshot.get("progress"),
            message_code=message_code,
            component_id=self._snapshot.get("component_id"),
        )

    def _abort_if_cancel_requested(self) -> None:
        if self._operation_id is None or not self._store.cancel_requested(
            self._operation_id
        ):
            return
        latest = self._store.snapshot(self._operation_id)
        if latest is not None:
            self._snapshot = latest
            self._sequence = int(latest["sequence"])
        self.cancel()
        raise RuntimeOperationCancelled(self._operation_id)

    def observe(self, *, after_sequence: int, limit: int = 128) -> dict[str, Any]:
        if self._operation_id is None:
            raise RuntimeOperationError("maintenance operation has not started")
        return self._store.observe(
            self._operation_id,
            after_sequence=after_sequence,
            limit=limit,
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
        failure: dict[str, Any] | None = None,
    ) -> None:
        if self._operation is None or self._operation_id is None:
            raise RuntimeError("maintenance operation has not started")
        if self._snapshot is not None and self._snapshot.get("operation_state") in {
            "succeeded",
            "failed",
            "cancelled",
        }:
            raise RuntimeOperationError("terminal Runtime snapshot is immutable")
        self._sequence += 1
        snapshot: dict[str, Any] = {
            "operation_id": self._operation_id,
            "source_operation_id": self._source_operation_id,
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
        if self._requested_component_ids:
            snapshot["requested_component_ids"] = list(self._requested_component_ids)
        if self._effective_component_ids:
            snapshot["effective_component_ids"] = list(self._effective_component_ids)
        if self._source is not None:
            snapshot["source"] = dict(self._source)
        try:
            event = self._store.append(
                self._operation_id,
                event_type=event_type,
                snapshot=snapshot,
                message_code=message_code,
                failure=failure,
            )
        except RuntimeOperationError:
            if operation_state != "cancelled" and self._store.cancel_requested(
                self._operation_id
            ):
                latest = self._store.snapshot(self._operation_id)
                if latest is not None:
                    self._snapshot = latest
                    self._sequence = int(latest["sequence"])
                self.cancel()
                raise RuntimeOperationCancelled(self._operation_id) from None
            raise
        self._snapshot = snapshot
        self._message_code = message_code
        if self._event_sink is not None:
            self._event_sink(event)


def runtime_status_from_environment(
    instance_id: str,
    service_state: str,
) -> dict[str, Any]:
    """Build the HTTP status snapshot from installer-owned environment/state."""
    manifest_value = os.environ.get("VIBEOCR_RUNTIME_MANIFEST")
    accelerator = os.environ.get("VIBEOCR_RUNTIME_ACCELERATOR")
    state_root_value = os.environ.get("VIBEOCR_RUNTIME_STATE_ROOT")
    runtime_root_value = os.environ.get("VIBEOCR_RUNTIME_ROOT")
    if not manifest_value or not accelerator or not state_root_value:
        raise RuntimeError("Runtime status environment is incomplete")
    # Runtime installation verified the full release asset set before launch.
    # Status polling only needs the immutable descriptor and must not rehash the
    # large Python/runtime archives on every HTTP request.
    manifest = load_runtime_manifest(manifest_value, verify_artifacts=False)
    profile = runtime_profile_status(
        manifest,
        accelerator=accelerator,
        runtime_root=Path(runtime_root_value) if runtime_root_value else None,
    )
    if service_state != "maintenance" and any(
        component.get("actual_state") != "ready" for component in profile["components"]
    ):
        service_state = "degraded"
    maintenance = RuntimeOperationStore(Path(state_root_value)).latest_projection()
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
        "source": runtime_source_identity(manifest),
        "profile": profile,
        "maintenance": maintenance,
    }


__all__ = [
    "COMMANDS_DIRECTORY",
    "EventSink",
    "OPERATIONS_DIRECTORY",
    "RuntimeCommandConflict",
    "RuntimeCursorExpired",
    "RuntimeMaintenanceReporter",
    "RuntimeOperationConflict",
    "RuntimeOperationCancelled",
    "RuntimeOperationError",
    "RuntimeOperationNotFound",
    "RuntimeOperationNotCancellable",
    "RuntimeOperationNotRetryable",
    "RuntimeOperationStart",
    "RuntimeOperationStore",
    "RuntimeProfileDescriptor",
    "profile_descriptor",
    "runtime_source_identity",
    "runtime_profile_status",
    "runtime_status_from_environment",
]
