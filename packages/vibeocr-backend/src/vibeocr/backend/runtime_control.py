"""Single transport-neutral command/observe seam for Runtime maintenance."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vibeocr.backend.runtime_installer import (
    RuntimeIdentityMismatch,
    RuntimeInspection,
    RuntimeInstaller,
    RuntimeLaunch,
    RuntimeState,
)
from vibeocr.backend.runtime_lock import RuntimeLockTimeout
from vibeocr.backend.runtime_maintenance import (
    RuntimeOperationCancelled,
    RuntimeOperationConflict,
    RuntimeOperationNotFound,
    RuntimeOperationNotRetryable,
    RuntimeOperationStore,
    runtime_source_identity,
)


@dataclass(frozen=True, slots=True)
class RuntimeControlResult:
    receipt: dict[str, Any]
    state: RuntimeState
    profile: dict[str, Any]
    launch: RuntimeLaunch | None
    available_capabilities: tuple[str, ...]


class RuntimeControl:
    """Execute idempotent maintenance intents and observe their durable journal."""

    def __init__(
        self,
        *,
        product_root: str | Path,
        component_lock: str | Path,
        runtime_manifest: str | Path,
        accelerator: str | None = None,
    ) -> None:
        self._product_root = Path(product_root)
        self._component_lock = Path(component_lock)
        self._runtime_manifest = Path(runtime_manifest)
        self._accelerator = accelerator
        self._installer_factory: Callable[..., RuntimeInstaller] | None = None
        self._active_snapshot: dict[str, Any] | None = None
        probe = self._installer()
        self._state_root = probe.paths.state_root
        self._store = RuntimeOperationStore(self._state_root)

    @classmethod
    def from_installer_factory(
        cls,
        installer_factory: Callable[..., RuntimeInstaller],
    ) -> RuntimeControl:
        """Build the same control seam for layout-aware stdio adapters."""
        control = cls.__new__(cls)
        control._product_root = Path()
        control._component_lock = Path()
        control._runtime_manifest = Path()
        control._accelerator = None
        control._installer_factory = installer_factory
        control._active_snapshot = None
        probe = control._installer()
        control._state_root = probe.paths.state_root
        control._store = RuntimeOperationStore(control._state_root)
        return control

    @classmethod
    def from_environment(cls) -> RuntimeControl:
        required = {
            "product_root": os.environ.get("VIBEOCR_PRODUCT_ROOT"),
            "component_lock": os.environ.get("VIBEOCR_COMPONENT_LOCK"),
            "runtime_manifest": os.environ.get("VIBEOCR_RUNTIME_MANIFEST"),
        }
        if any(not value for value in required.values()):
            raise RuntimeError("Runtime control environment is incomplete")
        return cls(
            product_root=str(required["product_root"]),
            component_lock=str(required["component_lock"]),
            runtime_manifest=str(required["runtime_manifest"]),
            accelerator=os.environ.get("VIBEOCR_RUNTIME_ACCELERATOR"),
        )

    @property
    def state_root(self) -> Path:
        return self._state_root

    @property
    def maintenance_snapshot(self) -> dict[str, Any] | None:
        return (
            dict(self._active_snapshot) if self._active_snapshot is not None else None
        )

    def _installer(
        self,
        *,
        operation_id: str | None = None,
        source_operation_id: str | None = None,
        component_ids: tuple[str, ...] = (),
        required_capabilities: tuple[str, ...] = (),
    ) -> RuntimeInstaller:
        if self._installer_factory is not None:
            return self._installer_factory(
                operation_id=operation_id,
                source_operation_id=source_operation_id,
                component_ids=component_ids,
                required_capabilities=required_capabilities,
            )
        return RuntimeInstaller(
            product_root=self._product_root,
            component_lock=self._component_lock,
            runtime_manifest=self._runtime_manifest,
            accelerator=self._accelerator,
            operation_id=operation_id,
            source_operation_id=source_operation_id,
            component_ids=component_ids,
            required_capabilities=required_capabilities,
        )

    @staticmethod
    def _receipt(
        installer: RuntimeInstaller,
        negotiated_capabilities: tuple[str, ...],
    ) -> dict[str, Any]:
        snapshot = installer.maintenance_snapshot()
        if snapshot is None:
            raise RuntimeError("Runtime maintenance operation produced no snapshot")
        return {
            "schema_version": 2,
            "operation_id": snapshot["operation_id"],
            "snapshot": snapshot,
            "negotiated_capabilities": list(negotiated_capabilities),
        }

    def execute(
        self,
        *,
        operation: str,
        operation_id: str | None = None,
        component_ids: tuple[str, ...] = (),
        required_capabilities: tuple[str, ...] = (),
        source_operation_id: str | None = None,
        profile_id: str | None = None,
    ) -> dict[str, Any]:
        if operation not in {"inspect", "ensure", "repair"}:
            raise ValueError("invalid Runtime maintenance operation")
        result = self.execute_with_result(
            operation=operation,
            operation_id=operation_id,
            component_ids=component_ids,
            required_capabilities=required_capabilities,
            source_operation_id=source_operation_id,
            profile_id=profile_id,
        )
        return result.receipt

    def execute_with_result(
        self,
        *,
        operation: str,
        operation_id: str | None = None,
        component_ids: tuple[str, ...] = (),
        required_capabilities: tuple[str, ...] = (),
        source_operation_id: str | None = None,
        profile_id: str | None = None,
    ) -> RuntimeControlResult:
        """Execute once while exposing adapter-only launch projection."""
        if operation not in {"inspect", "ensure", "repair"}:
            raise ValueError("invalid Runtime maintenance operation")
        installer = self._installer(
            operation_id=operation_id,
            source_operation_id=source_operation_id,
            component_ids=component_ids,
            required_capabilities=required_capabilities,
        )
        if profile_id is not None and profile_id != installer.plan:
            raise ValueError("requested Runtime profile is unavailable")
        try:
            launch: RuntimeLaunch | None
            inspection: RuntimeInspection | None = None
            if operation == "inspect":
                inspection = installer.inspect_snapshot()
                launch = None
            else:
                launch = getattr(installer, operation)()
        finally:
            self._active_snapshot = installer.maintenance_snapshot()
        receipt = self._receipt(installer, required_capabilities)
        return self._project_result(
            installer,
            receipt,
            launch,
            inspection=inspection,
        )

    def project_receipt(
        self,
        receipt: dict[str, Any],
        *,
        include_launch: bool,
    ) -> RuntimeControlResult:
        """Project a replayed command receipt without leaking installer details."""
        installer = self._installer()
        inspection = installer.inspect_snapshot(emit=False)
        launch = (
            installer._launch()
            if include_launch and inspection.state.status == "ready"
            else None
        )
        return self._project_result(
            installer,
            receipt,
            launch,
            inspection=inspection,
        )

    @staticmethod
    def _project_result(
        installer: RuntimeInstaller,
        receipt: dict[str, Any],
        launch: RuntimeLaunch | None,
        *,
        inspection: RuntimeInspection | None = None,
    ) -> RuntimeControlResult:
        inspection = inspection or installer.inspect_snapshot(emit=False)
        return RuntimeControlResult(
            receipt=receipt,
            state=inspection.state,
            profile=inspection.profile,
            launch=launch,
            available_capabilities=tuple(installer.manifest.capabilities),
        )

    def command(
        self,
        *,
        command_id: str,
        command: str,
        target_operation_id: str,
        new_operation_id: str | None = None,
        expected_sequence: int | None = None,
    ) -> dict[str, Any]:
        payload = {
            "command": command,
            "target_operation_id": target_operation_id,
            "new_operation_id": new_operation_id,
            "expected_sequence": expected_sequence,
        }

        def apply() -> dict[str, Any]:
            target = self._store.snapshot(target_operation_id)
            if target is None:
                raise RuntimeOperationNotFound(target_operation_id)
            if command == "cancel":
                snapshot = self._store.request_cancel(
                    target_operation_id,
                    expected_sequence=expected_sequence,
                )
                return {
                    "schema_version": 2,
                    "operation_id": target_operation_id,
                    "snapshot": snapshot,
                    "negotiated_capabilities": [],
                }
            if command != "retry" or not new_operation_id:
                raise ValueError("invalid Runtime maintenance command")
            if (
                expected_sequence is not None
                and target["sequence"] != expected_sequence
            ):
                raise RuntimeOperationConflict("expected_sequence mismatch")
            if target["operation_state"] not in {"failed", "cancelled"}:
                raise RuntimeOperationNotRetryable(target_operation_id)
            intent = self._store.intent(target_operation_id)
            retry_installer = self._installer()
            if intent.get("source_identity") != runtime_source_identity(
                retry_installer.manifest
            ):
                raise RuntimeIdentityMismatch(
                    "retry Runtime source identity differs from the bound intent"
                )
            try:
                existing = self._store.snapshot(new_operation_id)
            except RuntimeOperationNotFound:
                existing = None
            if existing is not None:
                expected_retry_intent = {
                    "operation": str(target["operation"]),
                    "profile_id": str(intent["profile_id"]),
                    "component_ids": list(intent.get("component_ids", [])),
                    "required_capabilities": list(
                        intent.get("required_capabilities", [])
                    ),
                    "source_identity": dict(intent.get("source_identity", {})),
                    "source_operation_id": target_operation_id,
                }
                if self._store.intent(new_operation_id) != expected_retry_intent:
                    raise RuntimeOperationConflict(new_operation_id)
                state = existing.get("operation_state")
                if state == "succeeded":
                    return {
                        "schema_version": 2,
                        "operation_id": new_operation_id,
                        "snapshot": existing,
                        "negotiated_capabilities": list(
                            intent.get("required_capabilities", [])
                        ),
                    }
                if state == "failed":
                    self._store.raise_replayed_failure(new_operation_id)
                if state == "cancelled":
                    raise RuntimeOperationCancelled(new_operation_id)
                raise RuntimeLockTimeout(
                    "retry operation was interrupted or is still running"
                )
            return self.execute(
                operation=str(target["operation"]),
                operation_id=new_operation_id,
                component_ids=tuple(intent.get("component_ids", [])),
                required_capabilities=tuple(intent.get("required_capabilities", [])),
                source_operation_id=target_operation_id,
                profile_id=str(intent["profile_id"]),
            )

        return self._store.apply_command(command_id, payload, apply)

    def observe(
        self, operation_id: str, *, after_sequence: int = 0, limit: int = 128
    ) -> dict[str, Any]:
        return self._store.observe(
            operation_id,
            after_sequence=after_sequence,
            limit=limit,
        )


__all__ = ["RuntimeControl", "RuntimeControlResult"]
