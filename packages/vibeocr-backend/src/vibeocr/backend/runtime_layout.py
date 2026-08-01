"""Portable, content-addressed Backend runtime store layout."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PROFILES = frozenset({"win-x64-cpu", "win-x64-cu126"})

# 物理 runtime 目录名 / runtime_id 仅取 manifest 哈希前 6 位，避免安装命中
# Windows MAX_PATH（完整 64 位哈希会让 onnxruntime 等深路径超出 260 字符）。
# 完整哈希仍保留在 component-lock 的 runtime_manifest_sha256 字段做密码学校验，
# 与本标识彻底解耦。前缀在单产品 store 内足以避免内容寻址冲突（16^6 ≈ 16M）。
_RUNTIME_PREFIX_LEN = 6
_RUNTIME_PREFIX_RE = re.compile(r"^[0-9a-f]{%d}$" % _RUNTIME_PREFIX_LEN)


def runtime_id_prefix(manifest_sha256: str) -> str:
    """runtime 标识前缀 = manifest 哈希前 6 位。

    用于物理目录名与 ``runtime_id``（日志、lock 文件名、GC 寻址）。完整 64 位
    哈希仅留在 component-lock 的 ``runtime_manifest_sha256`` 字段做密码学完整性
    校验，与本标识解耦。
    """
    if not _SHA256_RE.fullmatch(manifest_sha256):
        raise LayoutError("manifest_sha256 must be lowercase SHA-256")
    return manifest_sha256[:_RUNTIME_PREFIX_LEN]


class LayoutError(ValueError):
    """Raised when a portable layout could escape or is ambiguous."""


@dataclass(frozen=True, slots=True)
class RuntimeStorePaths:
    product_root: Path
    store_root: Path
    runtimes_root: Path
    models_root: Path
    locks_root: Path
    state_root: Path
    runtime_root: Path
    model_root: Path
    shared: bool


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
    profile: str,
    layout_manifest: str | Path | None = None,
    product_id: str | None = None,
) -> RuntimeStorePaths:
    """Resolve one immutable runtime and model store.

    With no explicit ``layout_manifest`` the store is entirely inside the
    product directory.  Shared storage is enabled only when both the manifest
    path and registered product id are supplied and validate.
    """
    product = Path(product_root).resolve()
    # 校验完整哈希（防篡改），但物理目录用 6 位前缀（见 runtime_id_prefix）。
    if not _SHA256_RE.fullmatch(manifest_sha256):
        raise LayoutError("manifest_sha256 must be lowercase SHA-256")
    if profile not in _PROFILES:
        raise LayoutError(f"unsupported runtime profile: {profile}")

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

    runtimes = store / "runtimes"
    models = store / "models"
    locks = store / "locks"
    state = store / "state"
    runtime = runtimes / runtime_id_prefix(manifest_sha256) / profile
    model = models
    return RuntimeStorePaths(
        product_root=product,
        store_root=store,
        runtimes_root=runtimes,
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
    "LayoutError",
    "RuntimeStorePaths",
    "resolve_model_path",
    "resolve_runtime_store",
    "runtime_id_prefix",
]
