"""Durable read-only previews, bound to semantic state rather than local hashes."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from vibeocr.backend.runtime_maintenance import (
    RuntimeInstallPlanBlocked,
    RuntimeInstallPlanStale,
    RuntimeOperationConflict,
    _atomic_json,
)
from vibeocr.backend.runtime_manifest import scoped_component_version
from vibeocr.backend.runtime_selection import ResolvedRuntimeSelection
from vibeocr.runtime_contracts.parser import parse_runtime_install_plan_response

CAPABILITY = "runtime.install-plan.v1"


def read_plan(state_root: Path, plan_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{32}", plan_id):
        raise RuntimeInstallPlanStale("unknown install plan; preview again")
    try:
        record = json.loads(
            (state_root / "install-plans" / f"{plan_id}.json").read_text(
                encoding="utf-8"
            )
        )
        parse_runtime_install_plan_response(
            {
                "schema_version": 2,
                "plan": record["plan"],
                "negotiated_capabilities": [CAPABILITY],
            }
        )
        if not isinstance(record["baseline"], dict):
            raise ValueError("invalid baseline")
        if record["plan"]["plan_id"] != plan_id:
            raise ValueError("plan identity mismatch")
        return record
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeInstallPlanStale(
            "install plan is unavailable; preview again"
        ) from exc


def bind_plan(state_root: Path, record: dict[str, Any], operation_id: str) -> None:
    """Bind once under the caller's runtime-store lock, before any execution."""
    bound_operation = record.get("operation_id")
    if bound_operation is not None and bound_operation != operation_id:
        raise RuntimeOperationConflict(
            "install plan already accepted by another operation"
        )
    _atomic_json(
        state_root / "install-plans" / f"{record['plan']['plan_id']}.json",
        {**record, "operation_id": operation_id},
    )


def validate_plan(
    record: dict[str, Any], baseline: dict[str, Any], source: dict[str, str]
) -> None:
    plan = record["plan"]
    if (
        datetime.fromisoformat(plan["expires_at"]) <= datetime.now(UTC)
        or record["baseline"] != baseline
        or plan["source"] != source
    ):
        raise RuntimeInstallPlanStale(
            "install baseline changed or plan expired; preview again"
        )
    if plan["blockers"]:
        raise RuntimeInstallPlanBlocked(
            "install plan has blockers; resolve and preview again"
        )


def create_plan(
    state_root: Path,
    selection: ResolvedRuntimeSelection,
    source: dict[str, str],
    baseline: dict[str, Any],
    blockers: list[dict[str, str]],
) -> dict[str, Any]:
    marker = baseline["marker"] or {}
    installed = marker.get("component_ids", [])
    current_binding = (
        marker.get("schema_version") == 1
        and marker.get("backend_version") == source["backend_version"]
        and marker.get("manifest_sha256") == source["runtime_manifest_sha256"]
        and marker.get("accelerator") == selection.accelerator
    )
    probes = baseline["probes"]
    versions = baseline.get("versions", {})
    expected_versions = {
        item.component_id: scoped_component_version(
            selection.profile, selection.install_scope, item
        )
        for item in selection.profile.components
    }
    components = []
    requested = selection.requested_component_ids
    for component_id in dict.fromkeys([*selection.effective_component_ids, *installed]):
        selected = component_id in selection.effective_component_ids
        present = component_id in installed
        expected = expected_versions.get(component_id)
        healthy = (
            present
            and current_binding
            and probes.get(component_id, False)
            and (expected is None or versions.get(component_id) == expected)
        )
        action = (
            "remove"
            if not selected
            else "retain"
            if healthy
            else "replace"
            if present
            else "install"
        )
        components.append(
            {
                "component_id": component_id,
                "action": action,
                "dependency_state": "satisfied" if healthy else "pending",
                "reason_codes": [
                    "removed_by_selection"
                    if not selected
                    else "requested"
                    if requested is None or component_id in requested
                    else "required_dependency"
                ],
            }
        )
    unchanged = all(item["action"] == "retain" for item in components)
    unknown_costs = (
        []
        if unchanged
        else ["artifact_resolution_required", "candidate_disk_usage_unknown"]
    )
    if any(
        item.startswith(("paddleocr-", "mineru-"))
        for item in selection.effective_component_ids
    ):
        unknown_costs.append("native_model_preparation_not_estimated")
    plan = {
        "plan_id": uuid4().hex,
        "expires_at": (datetime.now(UTC) + timedelta(minutes=10)).isoformat(),
        "accelerator": selection.accelerator,
        "profile_id": selection.profile.name,
        "requested_component_ids": None if requested is None else list(requested),
        "effective_component_ids": list(selection.effective_component_ids),
        "requested_download_source_ids": None
        if selection.requested_download_source_ids is None
        else list(selection.requested_download_source_ids),
        "effective_download_source_ids": list(selection.effective_download_source_ids),
        "source": source,
        "components": components,
        "blockers": blockers,
        "cost": {
            "download_bytes": 0 if unchanged else None,
            "additional_disk_bytes": 0 if unchanged else None,
            "unknown_reason_codes": unknown_costs,
        },
    }
    _atomic_json(
        state_root / "install-plans" / f"{plan['plan_id']}.json",
        {"plan": plan, "baseline": baseline},
    )
    return plan
