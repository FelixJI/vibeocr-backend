from __future__ import annotations

import pytest
from vibeocr.backend.runtime_selection import (
    RuntimeSelectionError,
    component_variant_catalog_payload,
    download_source_catalog_payload,
    normalize_download_source_ids,
    normalize_install_component_ids,
    selectable_component_ids,
)
from vibeocr.runtime_contracts import ErrorCode


def test_download_source_catalog_declares_default_and_unique_business_keys() -> None:
    catalog = download_source_catalog_payload()

    sources = catalog["sources"]
    assert sources, "catalog must declare at least one Backend default source"
    ids = [source["id"] for source in sources]
    kinds = [source["kind"] for source in sources]
    assert len(set(ids)) == len(ids)
    assert len(set(kinds)) == len(kinds)
    for source in sources:
        assert set(source) == {"kind", "id", "endpoint"}
        assert source["endpoint"].startswith("https://")


def test_download_source_catalog_rejects_duplicate_ids_and_kinds() -> None:
    duplicate_id = [
        {"kind": "package_index", "id": "pypi", "endpoint": "https://a"},
        {"kind": "model_registry", "id": "pypi", "endpoint": "https://b"},
    ]
    with pytest.raises(RuntimeSelectionError) as excinfo:
        download_source_catalog_payload(duplicate_id)
    assert excinfo.value.code is ErrorCode.VALIDATION_ERROR

    duplicate_kind = [
        {"kind": "package_index", "id": "pypi", "endpoint": "https://a"},
        {"kind": "package_index", "id": "mirror", "endpoint": "https://b"},
    ]
    with pytest.raises(RuntimeSelectionError):
        download_source_catalog_payload(duplicate_kind)


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


def test_normalize_download_source_ids_resolves_omission_to_backend_default() -> None:
    assert normalize_download_source_ids(None) == ("pypi",)
    assert normalize_download_source_ids(()) == ("pypi",)
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
