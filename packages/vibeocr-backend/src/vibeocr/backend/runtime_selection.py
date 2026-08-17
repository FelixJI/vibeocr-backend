"""Transport-neutral selection policy for optional components and download sources.

Settings、HTTP maintenance 与 Runtime Host 三条 adapter 共用本模块：省略、
空集、未知 id 与 accelerator 匹配的语义只在这里解释一次（计划 §4.2），
编排层（``runtime_control``/``runtime_installer``）只消费规范化结果。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vibeocr.backend.network_detector import get_pip_mirror
from vibeocr.backend.runtime_manifest import (
    ACCELERATOR_TO_PLAN,
    PROFILE_COMPONENTS,
    RuntimeInstallScope,
    RuntimeManifest,
    RuntimeProfile,
)
from vibeocr.runtime_contracts import ErrorCode

DOWNLOAD_SOURCE_KIND_PACKAGE_INDEX = "package_index"
DOWNLOAD_SOURCE_KIND_MODEL_REGISTRY = "model_registry"

# 可选组件目录只描述 full 档位：base 档位的闭包是必备项，不可选择。
VARIANT_ACCELERATORS = ("cpu", "nvidia_cuda")
BASE_PROFILE = "win-x64-base"

# Backend release 声明的候选源。Protocol 允许同 kind 多候选；单次选择
# 每种 kind 至多一个。TUNA 是发布/运行时默认 package index，官方 PyPI
# 保留为显式候选，不做静默 fallback。
_DOWNLOAD_SOURCES: tuple[dict[str, str], ...] = (
    {
        "kind": DOWNLOAD_SOURCE_KIND_PACKAGE_INDEX,
        "id": "tuna-pypi",
        "endpoint": get_pip_mirror("domestic"),
    },
    {
        "kind": DOWNLOAD_SOURCE_KIND_PACKAGE_INDEX,
        "id": "pypi",
        "endpoint": get_pip_mirror("international"),
    },
)
_DEFAULT_DOWNLOAD_SOURCE_IDS = ("tuna-pypi",)


class RuntimeSelectionError(ValueError):
    """Raised when a selection cannot be normalized against the catalogs."""

    def __init__(self, code: ErrorCode, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


@dataclass(frozen=True, slots=True)
class BoundDownloadSource:
    """Release-declared source selected for one install plan."""

    kind: str
    source_id: str
    endpoint: str


@dataclass(frozen=True, slots=True)
class ResolvedRuntimeSelection:
    """Immutable selection result consumed by orchestration and installation."""

    accelerator: str
    profile: RuntimeProfile
    install_scope: RuntimeInstallScope
    requested_component_ids: tuple[str, ...] | None
    effective_component_ids: tuple[str, ...]
    requested_download_source_ids: tuple[str, ...] | None
    effective_download_sources: tuple[BoundDownloadSource, ...]

    @property
    def effective_download_source_ids(self) -> tuple[str, ...]:
        return tuple(source.source_id for source in self.effective_download_sources)

    def durable_intent_fields(self) -> dict[str, list[str]]:
        return normalized_selection_fields(
            install_component_ids=self.requested_component_ids,
            download_source_ids=self.effective_download_source_ids,
        )


class RuntimeSelectionPolicy:
    """Resolve Protocol selection once against one verified release graph."""

    def __init__(
        self,
        *,
        profiles: Mapping[str, RuntimeProfile],
        sources: Sequence[Mapping[str, str]] | None = None,
        default_download_source_ids: Sequence[str] = _DEFAULT_DOWNLOAD_SOURCE_IDS,
    ) -> None:
        self._profiles = dict(profiles)
        catalog = download_source_catalog_payload(
            [dict(source) for source in sources] if sources is not None else None
        )["sources"]
        self._sources = tuple(
            BoundDownloadSource(
                kind=source["kind"],
                source_id=source["id"],
                endpoint=source["endpoint"],
            )
            for source in catalog
        )
        self._default_download_source_ids = self._resolve_source_ids(
            tuple(default_download_source_ids)
        )

    @classmethod
    def from_manifest(
        cls,
        manifest: RuntimeManifest,
        *,
        sources: Sequence[Mapping[str, str]] | None = None,
        default_download_source_ids: Sequence[str] = _DEFAULT_DOWNLOAD_SOURCE_IDS,
    ) -> RuntimeSelectionPolicy:
        return cls(
            profiles=manifest.profiles,
            sources=sources,
            default_download_source_ids=default_download_source_ids,
        )

    def plan_start(
        self,
        *,
        accelerator: str,
        install_component_ids: Sequence[str] | None,
        download_source_ids: Sequence[str] | None,
        default_download_source_ids: Sequence[str] = (),
    ) -> ResolvedRuntimeSelection:
        try:
            profile = self._profiles[ACCELERATOR_TO_PLAN[accelerator]]
            base_profile = self._profiles[BASE_PROFILE]
        except KeyError as exc:
            raise RuntimeSelectionError(
                ErrorCode.VALIDATION_ERROR,
                f"unsupported accelerator: {accelerator}",
            ) from exc

        requested_components = self._canonical_component_ids(
            install_component_ids,
            profile=profile,
            base_profile=base_profile,
        )
        effective_components = self._component_closure(
            requested_components,
            profile=profile,
            base_profile=base_profile,
        )
        install_scope = self._install_scope(
            effective_components,
            profile=profile,
            base_profile=base_profile,
        )

        requested_sources = (
            None
            if download_source_ids is None
            else self._resolve_source_ids(tuple(download_source_ids))
        )
        default_sources = (
            self._resolve_source_ids(tuple(default_download_source_ids))
            if default_download_source_ids
            else self._default_download_source_ids
        )
        effective_source_ids = (
            default_sources if requested_sources is None else requested_sources
        )
        selected = set(effective_source_ids)
        return ResolvedRuntimeSelection(
            accelerator=accelerator,
            profile=profile,
            install_scope=install_scope,
            requested_component_ids=requested_components,
            effective_component_ids=effective_components,
            requested_download_source_ids=requested_sources,
            effective_download_sources=tuple(
                source for source in self._sources if source.source_id in selected
            ),
        )

    def _resolve_source_ids(self, requested: tuple[str, ...]) -> tuple[str, ...]:
        if not requested:
            raise RuntimeSelectionError(
                ErrorCode.VALIDATION_ERROR,
                "download_source_ids must contain at least one source",
            )
        if len(set(requested)) != len(requested):
            raise RuntimeSelectionError(
                ErrorCode.VALIDATION_ERROR,
                "download_source_ids must not contain duplicates",
            )
        by_id = {source.source_id: source for source in self._sources}
        try:
            selected = {source_id: by_id[source_id] for source_id in requested}
        except KeyError as exc:
            raise RuntimeSelectionError(
                ErrorCode.DOWNLOAD_SOURCE_UNKNOWN,
                f"unknown download source id: {exc.args[0]}",
            ) from exc
        kinds = [source.kind for source in selected.values()]
        if len(set(kinds)) != len(kinds):
            raise RuntimeSelectionError(
                ErrorCode.VALIDATION_ERROR,
                "download_source_ids must select at most one source per kind",
            )
        requested_set = set(requested)
        return tuple(
            source.source_id
            for source in self._sources
            if source.source_id in requested_set
        )

    @staticmethod
    def _canonical_component_ids(
        requested: Sequence[str] | None,
        *,
        profile: RuntimeProfile,
        base_profile: RuntimeProfile,
    ) -> tuple[str, ...] | None:
        if requested is None:
            return None
        requested_tuple = tuple(requested)
        if len(set(requested_tuple)) != len(requested_tuple):
            raise RuntimeSelectionError(
                ErrorCode.VALIDATION_ERROR,
                "install_component_ids must not contain duplicates",
            )
        base_ids = {component.component_id for component in base_profile.components}
        selectable = tuple(
            component.component_id
            for component in profile.components
            if component.component_id not in base_ids
        )
        catalog = selectable_component_ids_across_catalog()
        for component_id in requested_tuple:
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
        selected = set(requested_tuple)
        return tuple(item for item in selectable if item in selected)

    @staticmethod
    def _component_closure(
        requested: tuple[str, ...] | None,
        *,
        profile: RuntimeProfile,
        base_profile: RuntimeProfile,
    ) -> tuple[str, ...]:
        base_ids = {component.component_id for component in base_profile.components}
        if requested is None:
            return tuple(component.component_id for component in profile.components)
        selected = set(base_ids)
        dependencies = {
            component.component_id: component.dependencies
            for component in profile.components
        }

        def include(component_id: str) -> None:
            if component_id in selected:
                return
            selected.add(component_id)
            for dependency in dependencies[component_id]:
                include(dependency)

        for component_id in requested:
            include(component_id)
        return tuple(
            component.component_id
            for component in profile.components
            if component.component_id in selected
        )

    @staticmethod
    def _install_scope(
        component_ids: tuple[str, ...],
        *,
        profile: RuntimeProfile,
        base_profile: RuntimeProfile,
    ) -> RuntimeInstallScope:
        desired = set(component_ids)
        base_ids = {component.component_id for component in base_profile.components}
        candidates = base_profile.scopes if desired == base_ids else profile.scopes
        for scope in candidates:
            if set(scope.component_ids) == desired:
                return scope
        raise RuntimeSelectionError(
            ErrorCode.VALIDATION_ERROR,
            "runtime release has no install scope for selected component closure",
        )


def default_download_sources() -> tuple[dict[str, str], ...]:
    defaults = set(_DEFAULT_DOWNLOAD_SOURCE_IDS)
    return tuple(source for source in _DOWNLOAD_SOURCES if source["id"] in defaults)


def download_source_catalog_payload(
    sources: Sequence[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build and validate the ``runtime.download-sources.v1`` catalog payload."""
    entries = list(_DOWNLOAD_SOURCES if sources is None else sources)
    seen_ids: set[str] = set()
    for source in entries:
        source_id = source["id"]
        # Protocol 只要求 source id 跨 kind 唯一；同 kind 可以声明多个候选，
        # “每种 kind 至多选一个”属于单次 selection 的约束。
        if source_id in seen_ids:
            raise RuntimeSelectionError(
                ErrorCode.VALIDATION_ERROR,
                f"duplicate download source id: {source_id}",
            )
        seen_ids.add(source_id)
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
        return _DEFAULT_DOWNLOAD_SOURCE_IDS
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


def normalized_selection_fields(
    *,
    install_component_ids: Sequence[str] | None,
    download_source_ids: Sequence[str] | None,
) -> dict[str, list[str]]:
    """Project unordered Protocol selections onto a stable durable identity."""
    fields: dict[str, list[str]] = {}
    if install_component_ids is not None:
        fields["install_component_ids"] = sorted(install_component_ids)
    if download_source_ids is not None:
        fields["download_source_ids"] = sorted(download_source_ids)
    return fields


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
    "BoundDownloadSource",
    "DOWNLOAD_SOURCE_KIND_MODEL_REGISTRY",
    "DOWNLOAD_SOURCE_KIND_PACKAGE_INDEX",
    "ResolvedRuntimeSelection",
    "RuntimeSelectionError",
    "RuntimeSelectionPolicy",
    "VARIANT_ACCELERATORS",
    "component_variant_catalog_payload",
    "default_download_sources",
    "download_source_catalog_payload",
    "normalize_download_source_ids",
    "normalize_install_component_ids",
    "normalized_selection_fields",
    "selectable_component_ids",
    "selectable_component_ids_across_catalog",
]
