"""Transport-neutral selection policy for optional components and download sources.

Settings、HTTP maintenance 与 Runtime Host 三条 adapter 共用本模块：省略、
空集、未知 id 与 accelerator 匹配的语义只在这里解释一次（计划 §4.2），
编排层（``runtime_control``/``runtime_installer``）只消费规范化结果。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vibeocr.backend.runtime_manifest import (
    ACCELERATOR_TO_PLAN,
    PROFILE_COMPONENTS,
)
from vibeocr.runtime_contracts import ErrorCode

DOWNLOAD_SOURCE_KIND_PACKAGE_INDEX = "package_index"
DOWNLOAD_SOURCE_KIND_MODEL_REGISTRY = "model_registry"

# 可选组件目录只描述 full 档位：base 档位的闭包是必备项，不可选择。
VARIANT_ACCELERATORS = ("cpu", "nvidia_cuda")
BASE_PROFILE = "win-x64-base"

# Backend 声明的缺省下载源。Protocol 不写死 “official”；新增源（镜像、
# 模型仓库）从这里扩展。每种 kind 至多声明一个默认源。
_DEFAULT_DOWNLOAD_SOURCES: tuple[dict[str, str], ...] = (
    {
        "kind": DOWNLOAD_SOURCE_KIND_PACKAGE_INDEX,
        "id": "pypi",
        "endpoint": "https://pypi.org/simple",
    },
)


class RuntimeSelectionError(ValueError):
    """Raised when a selection cannot be normalized against the catalogs."""

    def __init__(self, code: ErrorCode, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


def default_download_sources() -> tuple[dict[str, str], ...]:
    return _DEFAULT_DOWNLOAD_SOURCES


def download_source_catalog_payload(
    sources: Sequence[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build and validate the ``runtime.download-sources.v1`` catalog payload."""
    entries = list(_DEFAULT_DOWNLOAD_SOURCES if sources is None else sources)
    seen_ids: set[str] = set()
    seen_kinds: set[str] = set()
    for source in entries:
        source_id = source["id"]
        kind = source["kind"]
        # JSON Schema 的 uniqueItems 只能拒绝完全相同的对象；source id 跨
        # kind 唯一、每种 kind 至多一个必须由业务键在这里 fail closed。
        if source_id in seen_ids:
            raise RuntimeSelectionError(
                ErrorCode.VALIDATION_ERROR,
                f"duplicate download source id: {source_id}",
            )
        if kind in seen_kinds:
            raise RuntimeSelectionError(
                ErrorCode.VALIDATION_ERROR,
                f"multiple download sources for kind: {kind}",
            )
        seen_ids.add(source_id)
        seen_kinds.add(kind)
    return {"sources": [dict(source) for source in entries]}


def selectable_component_ids(accelerator: str) -> tuple[str, ...]:
    """可选组件 = 目标档位 profile 相对 base 闭包的新增组件。"""
    plan = ACCELERATOR_TO_PLAN[accelerator]
    base_ids = {component_id for component_id, _ in PROFILE_COMPONENTS[BASE_PROFILE]}
    return tuple(
        component_id
        for component_id, _ in PROFILE_COMPONENTS[plan]
        if component_id not in base_ids
    )


def component_variant_catalog_payload() -> dict[str, Any]:
    """Build and validate the ``runtime.component-selection.v1`` catalog payload."""
    variants: list[dict[str, str]] = []
    business_keys: set[tuple[str, str]] = set()
    for accelerator in VARIANT_ACCELERATORS:
        for component_id in selectable_component_ids(accelerator):
            key = (component_id, accelerator)
            if key in business_keys:
                raise RuntimeSelectionError(
                    ErrorCode.VALIDATION_ERROR,
                    f"duplicate component variant: {component_id}/{accelerator}",
                )
            business_keys.add(key)
            variants.append(
                {
                    "feature_id": component_id,
                    "accelerator": accelerator,
                    "component_id": component_id,
                }
            )
    return {"variants": variants}


def selectable_component_ids_across_catalog() -> frozenset[str]:
    """全部可选 component id（跨 accelerator），用于未知 id 判定。"""
    return frozenset(
        component_id
        for accelerator in VARIANT_ACCELERATORS
        for component_id in selectable_component_ids(accelerator)
    )


def normalize_download_source_ids(
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    """规范化源选择：省略/空 → Backend 缺省源；未知 id、同 kind 多选 fail closed。"""
    catalog = {
        source["id"]: source for source in download_source_catalog_payload()["sources"]
    }
    if not requested:
        return tuple(source["id"] for source in catalog.values())
    if len(set(requested)) != len(requested):
        raise RuntimeSelectionError(
            ErrorCode.VALIDATION_ERROR,
            "download_source_ids must not contain duplicates",
        )
    resolved_kinds: set[str] = set()
    for source_id in requested:
        source = catalog.get(source_id)
        if source is None:
            raise RuntimeSelectionError(
                ErrorCode.DOWNLOAD_SOURCE_UNKNOWN,
                f"unknown download source id: {source_id}",
            )
        if source["kind"] in resolved_kinds:
            raise RuntimeSelectionError(
                ErrorCode.VALIDATION_ERROR,
                f"multiple download sources for kind: {source['kind']}",
            )
        resolved_kinds.add(source["kind"])
    return tuple(requested)


def normalize_install_component_ids(
    install_component_ids: Sequence[str] | None,
    *,
    accelerator: str,
) -> tuple[str, ...] | None:
    """规范化可选组件安装范围。

    ``None``（省略）表示 Backend 缺省（目标档位完整闭包）；``[]`` 表示显式
    只安装 base；非空列表逐项校验：不在目录中返回
    ``RUNTIME_COMPONENT_UNKNOWN``，在目录中但属于其他 accelerator 返回
    校验错误——component selection 不得隐式切换 Runtime 档位。
    """
    if install_component_ids is None:
        return None
    selectable = selectable_component_ids(accelerator)
    catalog = selectable_component_ids_across_catalog()
    for component_id in install_component_ids:
        if component_id not in catalog:
            raise RuntimeSelectionError(
                ErrorCode.RUNTIME_COMPONENT_UNKNOWN,
                f"unknown install component id: {component_id}",
            )
        if component_id not in selectable:
            raise RuntimeSelectionError(
                ErrorCode.VALIDATION_ERROR,
                f"install component {component_id} requires a different accelerator",
            )
    return tuple(install_component_ids)


__all__ = [
    "BASE_PROFILE",
    "DOWNLOAD_SOURCE_KIND_MODEL_REGISTRY",
    "DOWNLOAD_SOURCE_KIND_PACKAGE_INDEX",
    "RuntimeSelectionError",
    "VARIANT_ACCELERATORS",
    "component_variant_catalog_payload",
    "default_download_sources",
    "download_source_catalog_payload",
    "normalize_download_source_ids",
    "normalize_install_component_ids",
    "selectable_component_ids",
    "selectable_component_ids_across_catalog",
]
