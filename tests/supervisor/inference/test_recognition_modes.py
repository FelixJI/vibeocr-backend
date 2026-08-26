"""Recognition Mode registry/resolver 的公开行为契约。"""

from __future__ import annotations

import pytest


def test_catalog_exposes_stable_semantics_without_runtime_state_ambiguity() -> None:
    from vibeocr.backend.supervisor.inference.recognition_modes import (
        RecognitionModeRegistry,
    )

    registry = RecognitionModeRegistry()

    catalog = registry.catalog_payload()
    modes = {
        item["id"]: (
            item["family"],
            item["pipeline_id"],
            item["engine"],
            item["provisioning"],
            item["lifecycle"],
        )
        for item in catalog["modes"]
    }

    assert modes == {
        "rapid_text": (
            "text",
            "OCR",
            "rapidocr",
            "base_runtime",
            {
                "kind": "unmanaged",
                "supports_preload": False,
                "supports_ttl": False,
                "supports_pinning": False,
                "supports_release": False,
            },
        ),
        "windows_text": (
            "text",
            "OCR",
            "windows",
            "operating_system",
            {
                "kind": "unmanaged",
                "supports_preload": False,
                "supports_ttl": False,
                "supports_pinning": False,
                "supports_release": False,
            },
        ),
        "paddle_text": (
            "text",
            "OCR",
            "paddleocr",
            "advanced_component",
            {
                "kind": "model_residency",
                "supports_preload": True,
                "supports_ttl": True,
                "supports_pinning": True,
                "supports_release": True,
            },
        ),
        "paddle_structure": (
            "document",
            "PP-StructureV3",
            None,
            "advanced_component",
            {
                "kind": "model_residency",
                "supports_preload": True,
                "supports_ttl": True,
                "supports_pinning": True,
                "supports_release": True,
            },
        ),
        "paddle_document_vl": (
            "document",
            "PaddleOCR-VL",
            None,
            "advanced_component",
            {
                "kind": "model_residency",
                "supports_preload": True,
                "supports_ttl": True,
                "supports_pinning": True,
                "supports_release": True,
            },
        ),
        "mineru_document": (
            "document",
            "MinerU",
            None,
            "advanced_component",
            {
                "kind": "process_keep_alive",
                "supports_preload": False,
                "supports_ttl": True,
                "supports_pinning": False,
                "supports_release": True,
            },
        ),
        "paddle_table": (
            "specialized",
            "TABLE_RECOGNITION",
            None,
            "advanced_component",
            {
                "kind": "model_residency",
                "supports_preload": True,
                "supports_ttl": True,
                "supports_pinning": True,
                "supports_release": True,
            },
        ),
        "paddle_formula": (
            "specialized",
            "FORMULA_RECOGNITION",
            None,
            "advanced_component",
            {
                "kind": "model_residency",
                "supports_preload": True,
                "supports_ttl": True,
                "supports_pinning": True,
                "supports_release": True,
            },
        ),
    }


def test_registry_sources_execution_semantics_from_formal_protocol_sdk() -> None:
    from vibeocr.backend.supervisor.inference.recognition_modes import (
        RecognitionModeRegistry,
    )
    from vibeocr.runtime_contracts.contracts.pipelines import (
        get_all_recognition_modes,
        get_recognition_mode_definition,
    )

    registry = RecognitionModeRegistry()
    for mode_id in get_all_recognition_modes():
        protocol = get_recognition_mode_definition(mode_id)
        backend = registry.definition(mode_id.value)

        assert (
            backend.mode_id,
            backend.family,
            backend.pipeline_id,
            backend.engine,
            backend.provisioning,
            backend.lifecycle,
            backend.supported_options,
        ) == (
            protocol.mode,
            protocol.family,
            protocol.pipeline.value,
            protocol.engine,
            protocol.provisioning,
            protocol.lifecycle,
            protocol.supported_options,
        )


def test_resolver_fail_closed_for_unknown_unavailable_and_mapping_mismatch() -> None:
    from vibeocr.backend.supervisor.inference.recognition_modes import (
        ModeAvailability,
        RecognitionModeError,
        RecognitionModeRegistry,
    )

    registry = RecognitionModeRegistry(
        availability_probe=lambda definition: ModeAvailability(
            availability=(
                "ready" if definition.mode_id.value == "rapid_text" else "unavailable"
            ),
            reason_code=(
                None
                if definition.mode_id.value == "rapid_text"
                else "runtime_component_missing"
            ),
            required_component=definition.required_component,
        )
    )

    resolved = registry.resolve_execution(
        "rapid_text", pipeline_id="OCR", engine="rapidocr"
    )
    assert resolved.mode_id.value == "rapid_text"

    with pytest.raises(RecognitionModeError) as unknown:
        registry.resolve_execution("typo", pipeline_id="OCR", engine="rapidocr")
    assert unknown.value.code_name == "RECOGNITION_MODE_UNKNOWN"

    with pytest.raises(RecognitionModeError) as unavailable:
        registry.resolve_execution("paddle_text", pipeline_id="OCR", engine="paddleocr")
    assert unavailable.value.code_name == "RECOGNITION_MODE_UNAVAILABLE"
    assert unavailable.value.detail == {
        "recognition_mode": "paddle_text",
        "reason_code": "runtime_component_missing",
        "required_component": "paddleocr-cpu",
    }

    with pytest.raises(RecognitionModeError) as mismatch:
        registry.resolve_execution("rapid_text", pipeline_id="OCR", engine="paddleocr")
    assert mismatch.value.code_name == "RECOGNITION_MODE_PIPELINE_MISMATCH"

    with pytest.raises(RecognitionModeError) as missing_engine:
        registry.resolve_execution("rapid_text", pipeline_id="OCR")
    assert missing_engine.value.code_name == "RECOGNITION_MODE_PIPELINE_MISMATCH"

    assert (
        registry.resolve_execution_fields(
            pipeline_id="OCR", engine="rapidocr"
        ).mode_id.value
        == "rapid_text"
    )
    with pytest.raises(RecognitionModeError) as inverse_unavailable:
        registry.resolve_execution_fields(pipeline_id="OCR", engine="paddleocr")
    assert inverse_unavailable.value.code_name == "RECOGNITION_MODE_UNAVAILABLE"


def test_lifecycle_validation_distinguishes_models_processes_and_unmanaged() -> None:
    from vibeocr.backend.supervisor.inference.recognition_modes import (
        RecognitionModeError,
        RecognitionModeRegistry,
    )

    registry = RecognitionModeRegistry()

    assert registry.validate_lifecycle("paddle_text", "preload").pipeline_id == "OCR"
    assert (
        registry.validate_lifecycle("mineru_document", "release").pipeline_id
        == "MinerU"
    )

    for mode, operation in (
        ("rapid_text", "preload"),
        ("windows_text", "release"),
        ("mineru_document", "preload"),
        ("mineru_document", "pinning"),
    ):
        with pytest.raises(RecognitionModeError) as unsupported:
            registry.validate_lifecycle(mode, operation)
        assert unsupported.value.code_name == "RECOGNITION_MODE_LIFECYCLE_UNSUPPORTED"

    assert (
        registry.validate_pipeline_spec(
            "mineru_document", pipeline_id="MinerU", pinned=False
        ).mode_id.value
        == "mineru_document"
    )
    with pytest.raises(RecognitionModeError) as mismatch:
        registry.validate_pipeline_spec(
            "paddle_text", pipeline_id="MinerU", pinned=False
        )
    assert mismatch.value.code_name == "RECOGNITION_MODE_PIPELINE_MISMATCH"


def test_preload_resolver_requires_mode_pipeline_agreement_and_rejects_legacy_ocr() -> (
    None
):
    from vibeocr.backend.supervisor.inference.recognition_modes import (
        ModeAvailability,
        RecognitionModeError,
        RecognitionModeRegistry,
    )

    registry = RecognitionModeRegistry(
        availability_probe=lambda definition: ModeAvailability("ready")
    )

    assert registry.resolve_preload(
        ("paddle_text", "paddle_structure"),
        ("OCR", "PP-StructureV3"),
    ) == ("OCR", "PP-StructureV3")
    assert registry.resolve_preload(None, ("PP-StructureV3",)) == ("PP-StructureV3",)

    with pytest.raises(RecognitionModeError) as mismatch:
        registry.resolve_preload(("paddle_text",), ("PP-StructureV3",))
    assert mismatch.value.code_name == "RECOGNITION_MODE_PIPELINE_MISMATCH"

    with pytest.raises(RecognitionModeError) as ambiguous:
        registry.resolve_preload(None, ("OCR",))
    assert ambiguous.value.code_name == "RECOGNITION_MODE_PIPELINE_MISMATCH"

    unavailable = RecognitionModeRegistry()
    with pytest.raises(RecognitionModeError) as missing:
        unavailable.resolve_preload(None, ("PP-StructureV3",))
    assert missing.value.code_name == "RECOGNITION_MODE_UNAVAILABLE"


def test_residency_payload_labels_model_and_process_resources_separately() -> None:
    from vibeocr.backend.supervisor.inference.recognition_modes import (
        RecognitionModeRegistry,
    )

    payload = RecognitionModeRegistry().annotate_residency_payload(
        {
            "entries": [
                {"pipeline": "OCR", "kind": "soft_ttl"},
                {"pipeline": "MinerU", "kind": "soft_ttl"},
            ]
        }
    )

    assert payload["entries"] == [
        {
            "pipeline": "OCR",
            "kind": "soft_ttl",
            "recognition_mode": "paddle_text",
            "resource_kind": "model",
            "resource_id": "paddle_text",
        },
        {
            "pipeline": "MinerU",
            "kind": "soft_ttl",
            "recognition_mode": "mineru_document",
            "resource_kind": "process",
            "resource_id": "mineru-api",
        },
    ]
