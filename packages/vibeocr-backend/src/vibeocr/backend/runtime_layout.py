"""Portable Backend runtime store layout."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LayoutError(ValueError):
    """Raised when a portable layout could escape or is ambiguous."""


@dataclass(frozen=True, slots=True)
class RuntimeStorePaths:
    product_root: Path
    store_root: Path
    models_root: Path
    locks_root: Path
    state_root: Path
    runtime_root: Path
    model_root: Path
    shared: bool


@dataclass(frozen=True, slots=True)
class AppPaths:
    """Read-only application paths derived from the active Python environment."""

    install_root: Path
    resources_root: Path
    changelog_path: Path | None


def _backend_repository_root(module_path: Path) -> Path | None:
    for candidate in module_path.parents:
        backend_source = candidate / "packages" / "vibeocr-backend" / "src"
        if not (candidate / "pyproject.toml").is_file() or not backend_source.is_dir():
            continue
        try:
            module_path.relative_to(backend_source)
        except ValueError:
            continue
        return candidate.resolve()
    return None


def resolve_app_paths() -> AppPaths:
    """Resolve install, bundled-resource and changelog paths once.

    Frozen onedir builds are anchored at the executable directory. Backend source
    checkouts are recognized by their package layout. A regular wheel is anchored
    at ``sys.prefix`` so runtime state stays with its active Python environment.
    """

    frozen = bool(getattr(sys, "frozen", False))
    executable_root = Path(sys.executable).resolve().parent
    if frozen:
        install_root = executable_root
    else:
        module_path = Path(__file__).resolve()
        install_root = (
            _backend_repository_root(module_path) or Path(sys.prefix).resolve()
        )

    meipass = getattr(sys, "_MEIPASS", None)
    resources_root = (
        Path(meipass).resolve() / "resources" if meipass else install_root / "resources"
    )
    changelog_candidates = (
        (Path(meipass).resolve() / "CHANGELOG.md", executable_root / "CHANGELOG.md")
        if meipass
        else (install_root / "CHANGELOG.md",)
    )
    changelog_path = next(
        (candidate for candidate in changelog_candidates if candidate.is_file()),
        None,
    )
    return AppPaths(
        install_root=install_root,
        resources_root=resources_root,
        changelog_path=changelog_path,
    )


def _safe_relative(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise LayoutError(f"{field} must be a non-empty relative path")
    candidate = PurePath(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise LayoutError(f"{field} must stay within the portable bundle")
    return Path(candidate)


def _contained(root: Path, relative: Path, *, field: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise LayoutError(f"{field} escapes portable root") from exc
    return candidate


def _load_explicit_shared_layout(
    layout_manifest: Path,
    *,
    product_root: Path,
    product_id: str,
) -> Path:
    """Validate an explicitly supplied co-location marker.

    The caller must pass ``layout_manifest``.  This function never scans the
    product's parents looking for one.
    """
    manifest_path = layout_manifest.resolve(strict=True)
    bundle_root = manifest_path.parent
    try:
        data: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LayoutError(f"invalid portable layout: {manifest_path}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise LayoutError("portable layout schema_version must be 1")
    shared = _safe_relative(data.get("shared_root"), field="shared_root")
    products = data.get("products")
    if not isinstance(products, dict) or product_id not in products:
        raise LayoutError(f"product {product_id!r} is not registered")
    record = products[product_id]
    if not isinstance(record, dict):
        raise LayoutError(f"product {product_id!r} record must be an object")
    registered_root = _contained(
        bundle_root,
        _safe_relative(record.get("root"), field=f"products.{product_id}.root"),
        field=f"products.{product_id}.root",
    )
    if registered_root != product_root.resolve():
        raise LayoutError("product_root does not match portable layout registration")
    component_lock = _safe_relative(
        record.get("component_lock", "component-lock.json"),
        field=f"products.{product_id}.component_lock",
    )
    _contained(registered_root, component_lock, field="component_lock")
    return _contained(bundle_root, shared, field="shared_root")


def resolve_runtime_store(
    product_root: str | Path,
    *,
    manifest_sha256: str,
    layout_manifest: str | Path | None = None,
    product_id: str | None = None,
) -> RuntimeStorePaths:
    """Resolve the single mutable runtime and persistent model store.

    With no explicit ``layout_manifest`` the store is entirely inside the
    product directory.  Shared storage is enabled only when both the manifest
    path and registered product id are supplied and validate.
    """
    product = Path(product_root).resolve()
    if not _SHA256_RE.fullmatch(manifest_sha256):
        raise LayoutError("manifest_sha256 must be lowercase SHA-256")
    shared = layout_manifest is not None
    if shared:
        if not product_id:
            raise LayoutError("product_id is required for shared layout")
        store = _load_explicit_shared_layout(
            Path(layout_manifest),
            product_root=product,
            product_id=product_id,
        )
    else:
        if product_id is not None:
            raise LayoutError("product_id requires an explicit layout_manifest")
        store = product

    models = store / "models"
    locks = store / "locks"
    state = store / "state"
    runtime = store / "runtime"
    model = models
    return RuntimeStorePaths(
        product_root=product,
        store_root=store,
        models_root=models,
        locks_root=locks,
        state_root=state,
        runtime_root=runtime,
        model_root=model,
        shared=shared,
    )


def resolve_model_path(
    paths: RuntimeStorePaths,
    *,
    provider: str,
    model: str,
    version: str,
    artifact_sha256: str,
) -> Path:
    """Return a content-addressed model directory."""
    for field, value in {
        "provider": provider,
        "model": model,
        "version": version,
    }.items():
        _safe_relative(value, field=field)
        if len(PurePath(value).parts) != 1:
            raise LayoutError(f"{field} must be one path component")
    if not _SHA256_RE.fullmatch(artifact_sha256):
        raise LayoutError("artifact_sha256 must be lowercase SHA-256")
    return paths.models_root / provider / f"{model}@{version}" / artifact_sha256


__all__ = [
    "AppPaths",
    "LayoutError",
    "RuntimeStorePaths",
    "resolve_app_paths",
    "resolve_model_path",
    "resolve_runtime_store",
]
