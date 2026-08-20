from __future__ import annotations

from pathlib import Path

import pytest
from vibeocr.backend.runtime_manifest import (
    RuntimeInstallScope,
    RuntimeProfile,
    default_profile_components,
)
from vibeocr.backend.runtime_selection import (
    RuntimeSelectionError,
    RuntimeSelectionPolicy,
    component_variant_catalog_payload,
    default_download_sources,
    download_source_catalog_payload,
    normalize_download_source_ids,
    normalize_install_component_ids,
    selectable_component_ids,
)
from vibeocr.runtime_contracts import ErrorCode


def _profile(name: str, *scopes: RuntimeInstallScope) -> RuntimeProfile:
    return RuntimeProfile(
        name=name,
        lock_path=Path(f"{name}.lock"),
        sha256="0" * 64,
        runtime_pack=(),
        components=default_profile_components(name),
        scopes=scopes,
    )


def _scope(scope_id: str, component_ids: tuple[str, ...]) -> RuntimeInstallScope:
    return RuntimeInstallScope(
        scope_id=scope_id,
        component_ids=component_ids,
        lock_path=Path(f"{scope_id}.lock"),
        sha256="0" * 64,
        runtime_pack=(),
        runtime_pack_sha256=(),
    )


def _selection_profiles() -> dict[str, RuntimeProfile]:
    base_ids = tuple(
        component.component_id
        for component in default_profile_components("win-x64-base")
    )
    cpu_ids = tuple(
        component.component_id
        for component in default_profile_components("win-x64-cpu")
    )
    cuda_ids = tuple(
        component.component_id
        for component in default_profile_components("win-x64-cu126")
    )
    return {
        "win-x64-base": _profile("win-x64-base", _scope("default", base_ids)),
        "win-x64-cpu": _profile("win-x64-cpu", _scope("default", cpu_ids)),
        "win-x64-cu126": _profile(
            "win-x64-cu126",
            _scope("default", cuda_ids),
            _scope("gpu-runtime", (*base_ids, "gpu_runtime")),
        ),
    }


def _selection_policy() -> RuntimeSelectionPolicy:
    return RuntimeSelectionPolicy(profiles=_selection_profiles())


def test_download_source_catalog_declares_default_and_unique_business_keys() -> None:
    catalog = download_source_catalog_payload()

    sources = catalog["sources"]
    assert sources, "catalog must declare at least one Backend default source"
    ids = [source["id"] for source in sources]
    assert len(set(ids)) == len(ids)
    assert [source["id"] for source in default_download_sources()] == ["tuna-pypi"]
    assert {source["id"] for source in sources} == {
        "tuna-pypi",
        "pypi",
        "huggingface",
        "modelscope",
    }
    assert {source["kind"] for source in sources} == {
        "package_index",
        "model_registry",
    }
    for source in sources:
        assert set(source) == {"kind", "id", "endpoint"}
        assert source["endpoint"].startswith("https://")


def test_download_source_catalog_rejects_duplicate_ids_and_unknown_model_sources() -> (
    None
):
    duplicate_id = [
        {"kind": "package_index", "id": "pypi", "endpoint": "https://a"},
        {"kind": "package_index", "id": "pypi", "endpoint": "https://b"},
    ]
    with pytest.raises(RuntimeSelectionError) as excinfo:
        download_source_catalog_payload(duplicate_id)
    assert excinfo.value.code is ErrorCode.VALIDATION_ERROR

    unsupported_model_source = [
        {
            "kind": "model_registry",
            "id": "unknown-registry",
            "endpoint": "https://example.invalid",
        }
    ]
    with pytest.raises(RuntimeSelectionError) as excinfo:
        download_source_catalog_payload(unsupported_model_source)
    assert excinfo.value.code is ErrorCode.VALIDATION_ERROR

    same_kind_candidates = [
        {"kind": "package_index", "id": "pypi", "endpoint": "https://a"},
        {"kind": "package_index", "id": "mirror", "endpoint": "https://b"},
    ]
    assert download_source_catalog_payload(same_kind_candidates) == {
        "sources": same_kind_candidates
    }


def test_component_variant_catalog_lists_only_selectable_full_components() -> None:
    catalog = component_variant_catalog_payload()

    variants = catalog["variants"]
    assert variants, "catalog must list the optional full-closure components"
    business_keys = {(v["feature_id"], v["accelerator"]) for v in variants}
    assert len(business_keys) == len(variants)
    component_ids_by_accelerator: dict[str, set[str]] = {}
    for variant in variants:
        assert set(variant) == {"feature_id", "accelerator", "component_id"}
        assert variant["accelerator"] in {"cpu", "nvidia_cuda"}
        # 同一 accelerator 内 component_id 不得重复承载两个 feature。
        per_accelerator = component_ids_by_accelerator.setdefault(
            variant["accelerator"], set()
        )
        assert variant["component_id"] not in per_accelerator
        per_accelerator.add(variant["component_id"])
    # base 必备组件不进入可选目录。
    listed = {variant["component_id"] for variant in variants}
    assert listed == {"document_parsing", "gpu_runtime"}
    assert selectable_component_ids("cpu") == ("document_parsing",)
    assert set(selectable_component_ids("nvidia_cuda")) == {
        "document_parsing",
        "gpu_runtime",
    }


def test_selection_policy_resolves_exact_cuda_dependency_closure() -> None:
    policy = _selection_policy()

    gpu_only = policy.plan_start(
        accelerator="nvidia_cuda",
        install_component_ids=("gpu_runtime",),
        download_source_ids=("pypi",),
    )
    document_parsing = policy.plan_start(
        accelerator="nvidia_cuda",
        install_component_ids=("document_parsing",),
        download_source_ids=("pypi",),
    )

    assert gpu_only.requested_component_ids == ("gpu_runtime",)
    assert gpu_only.install_scope.scope_id == "gpu-runtime"
    assert gpu_only.effective_component_ids == (
        "ocr_engine",
        "pdf_document_tools",
        "image_code_tools",
        "runtime_host",
        "gpu_runtime",
    )
    assert document_parsing.install_scope.scope_id == "default"
    assert document_parsing.effective_component_ids == (
        "ocr_engine",
        "document_parsing",
        "pdf_document_tools",
        "image_code_tools",
        "runtime_host",
        "gpu_runtime",
    )


def test_selection_policy_canonicalizes_sources_and_rejects_same_kind() -> None:
    policy = RuntimeSelectionPolicy(
        profiles=_selection_profiles(),
        sources=(
            {
                "kind": "package_index",
                "id": "pypi",
                "endpoint": "https://pypi.org/simple",
            },
            {
                "kind": "package_index",
                "id": "mirror",
                "endpoint": "https://mirror.invalid/simple",
            },
            {
                "kind": "model_registry",
                "id": "huggingface",
                "endpoint": "https://huggingface.co",
            },
            {
                "kind": "model_registry",
                "id": "modelscope",
                "endpoint": "https://www.modelscope.cn",
            },
        ),
        default_download_source_ids=("pypi",),
    )

    resolved = policy.plan_start(
        accelerator="cpu",
        install_component_ids=None,
        download_source_ids=("pypi", "huggingface"),
    )
    assert resolved.requested_download_source_ids == ("pypi", "huggingface")
    assert resolved.effective_download_source_ids == ("pypi", "huggingface")

    with pytest.raises(RuntimeSelectionError) as excinfo:
        policy.plan_start(
            accelerator="cpu",
            install_component_ids=None,
            download_source_ids=("pypi", "huggingface", "modelscope"),
        )
    assert excinfo.value.code is ErrorCode.VALIDATION_ERROR


def test_selection_policy_overlays_explicit_sources_by_kind() -> None:
    policy = RuntimeSelectionPolicy(
        profiles=_selection_profiles(),
        sources=(
            {
                "kind": "package_index",
                "id": "tuna-pypi",
                "endpoint": "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/",
            },
            {
                "kind": "package_index",
                "id": "pypi",
                "endpoint": "https://pypi.org/simple",
            },
            {
                "kind": "model_registry",
                "id": "huggingface",
                "endpoint": "https://huggingface.co",
            },
            {
                "kind": "model_registry",
                "id": "modelscope",
                "endpoint": "https://www.modelscope.cn",
            },
        ),
        default_download_source_ids=("tuna-pypi", "modelscope"),
    )

    model_override = policy.plan_start(
        accelerator="cpu",
        install_component_ids=None,
        download_source_ids=("huggingface",),
    )
    package_override = policy.plan_start(
        accelerator="cpu",
        install_component_ids=None,
        download_source_ids=("pypi",),
    )

    assert model_override.requested_download_source_ids == ("huggingface",)
    assert model_override.effective_download_source_ids == (
        "tuna-pypi",
        "huggingface",
    )
    assert package_override.requested_download_source_ids == ("pypi",)
    assert package_override.effective_download_source_ids == ("pypi", "modelscope")


def test_normalize_download_source_ids_resolves_omission_to_backend_default() -> None:
    assert normalize_download_source_ids(None) == ("tuna-pypi",)
    assert normalize_download_source_ids(()) == ("tuna-pypi",)
    assert normalize_download_source_ids(["pypi"]) == ("pypi",)


def test_normalize_download_source_ids_fails_closed_on_unknown_or_conflict() -> None:
    with pytest.raises(RuntimeSelectionError) as excinfo:
        normalize_download_source_ids(["tsinghua-mirror"])
    assert excinfo.value.code is ErrorCode.DOWNLOAD_SOURCE_UNKNOWN

    with pytest.raises(RuntimeSelectionError) as excinfo:
        normalize_download_source_ids(["pypi", "pypi"])
    assert excinfo.value.code is ErrorCode.VALIDATION_ERROR


def test_normalize_install_component_ids_distinguishes_omission_and_empty() -> None:
    # None（省略）= Backend 缺省完整闭包；[]（显式空集）= base-only。
    assert normalize_install_component_ids(None, accelerator="cpu") is None
    assert normalize_install_component_ids((), accelerator="cpu") == ()
    assert normalize_install_component_ids((), accelerator="base") == ()
    assert normalize_install_component_ids(["document_parsing"], accelerator="cpu") == (
        "document_parsing",
    )


def test_normalize_install_component_ids_fails_closed() -> None:
    with pytest.raises(RuntimeSelectionError) as excinfo:
        normalize_install_component_ids(["not-a-component"], accelerator="cpu")
    assert excinfo.value.code is ErrorCode.RUNTIME_COMPONENT_UNKNOWN

    # gpu_runtime 在目录中，但只能装进 nvidia_cuda 档位：选择不得隐式换档。
    with pytest.raises(RuntimeSelectionError) as excinfo:
        normalize_install_component_ids(["gpu_runtime"], accelerator="cpu")
    assert excinfo.value.code is ErrorCode.VALIDATION_ERROR

    with pytest.raises(RuntimeSelectionError) as excinfo:
        normalize_install_component_ids(["document_parsing"], accelerator="base")
    assert excinfo.value.code is ErrorCode.VALIDATION_ERROR
