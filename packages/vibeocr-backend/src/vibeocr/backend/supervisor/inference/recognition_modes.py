"""Recognition Mode 的稳定领域目录与解析边界。

Recognition Mode 是用户选择的产品语义；pipeline/engine 只是执行映射。
领域规则保持 transport-neutral，由 HTTP adapter 使用正式 Protocol DTO、
capability 与错误枚举投影到 wire。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from vibeocr.runtime_contracts.contracts.pipelines import (
    RecognitionMode as RecognitionModeId,
)
from vibeocr.runtime_contracts.contracts.pipelines import (
    RecognitionModeFamily,
    RecognitionModeLifecycle,
    get_all_recognition_modes,
    get_recognition_mode_definition,
)
from vibeocr.runtime_contracts.contracts.pipelines import (
    RecognitionModeLifecycleKind as LifecycleKind,
)
from vibeocr.runtime_contracts.contracts.pipelines import (
    RecognitionModeProvisioning as ProvisioningKind,
)


class RecognitionModeAvailability(StrEnum):
    READY = "ready"
    PREPARATION_REQUIRED = "preparation_required"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ModeAvailability:
    availability: RecognitionModeAvailability | str
    reason_code: str | None = None
    required_component: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "availability",
            RecognitionModeAvailability(self.availability),
        )


@dataclass(frozen=True, slots=True)
class RecognitionModeDefinition:
    mode_id: RecognitionModeId
    family: RecognitionModeFamily
    pipeline_id: str
    engine: str | None
    provisioning: ProvisioningKind
    lifecycle: RecognitionModeLifecycle
    required_component: str | None = None
    supported_options: tuple[str, ...] = ()

    def to_payload(
        self, availability: ModeAvailability | None = None
    ) -> dict[str, Any]:
        payload = {
            "id": self.mode_id.value,
            "family": self.family.value,
            "pipeline_id": self.pipeline_id,
            "engine": self.engine,
            "provisioning": self.provisioning.value,
            "supported_options": list(self.supported_options),
            "lifecycle": {
                "kind": self.lifecycle.kind.value,
                "supports_preload": self.lifecycle.supports_preload,
                "supports_ttl": self.lifecycle.supports_ttl,
                "supports_pinning": self.lifecycle.supports_pinning,
                "supports_release": self.lifecycle.supports_release,
            },
        }
        if availability is not None:
            payload.update(
                {
                    "availability": availability.availability.value,
                    "reason_code": availability.reason_code,
                    "required_component": availability.required_component,
                }
            )
        return payload


class RecognitionModeError(ValueError):
    """Transport-neutral mode failure mapped by the HTTP adapter."""

    def __init__(
        self,
        code_name: str,
        *,
        recognition_mode: str | None,
        reason_code: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{code_name}: {reason_code}")
        self.code_name = code_name
        self.recognition_mode = recognition_mode
        self.reason_code = reason_code
        self.detail = {
            "recognition_mode": recognition_mode,
            "reason_code": reason_code,
            **(detail or {}),
        }


_DEFAULT_REQUIRED_COMPONENTS = {
    RecognitionModeId.PADDLE_TEXT: "paddleocr-cpu",
    RecognitionModeId.PADDLE_STRUCTURE: "paddleocr-cpu",
    RecognitionModeId.PADDLE_DOCUMENT_VL: "paddleocr-cpu",
    RecognitionModeId.MINERU_DOCUMENT: "mineru-cpu",
    RecognitionModeId.PADDLE_TABLE: "paddleocr-cpu",
    RecognitionModeId.PADDLE_FORMULA: "paddleocr-cpu",
}


def _definition_from_protocol(mode_id: RecognitionModeId) -> RecognitionModeDefinition:
    definition = get_recognition_mode_definition(mode_id)
    return RecognitionModeDefinition(
        mode_id=definition.mode,
        family=definition.family,
        pipeline_id=definition.pipeline.value,
        engine=definition.engine,
        provisioning=definition.provisioning,
        lifecycle=definition.lifecycle,
        required_component=_DEFAULT_REQUIRED_COMPONENTS.get(mode_id),
        supported_options=definition.supported_options,
    )


_DEFINITIONS = tuple(
    _definition_from_protocol(mode_id) for mode_id in get_all_recognition_modes()
)


class RecognitionModeRegistry:
    """所有 Recognition Mode 语义的单一真相源。"""

    def __init__(
        self,
        definitions: tuple[RecognitionModeDefinition, ...] = _DEFINITIONS,
        *,
        availability_probe: Callable[[RecognitionModeDefinition], ModeAvailability]
        | None = None,
    ) -> None:
        self._definitions = definitions
        self._by_id = {definition.mode_id: definition for definition in definitions}
        if len(self._by_id) != len(definitions):
            raise ValueError("duplicate recognition mode definition")
        self._availability_probe = availability_probe or self._default_availability

    def catalog_payload(self) -> dict[str, Any]:
        return {
            "modes": [
                definition.to_payload(self._availability_probe(definition))
                for definition in self._definitions
            ]
        }

    def definition(self, value: RecognitionModeId | str) -> RecognitionModeDefinition:
        try:
            mode_id = RecognitionModeId(value)
        except (TypeError, ValueError) as exc:
            raise RecognitionModeError(
                "RECOGNITION_MODE_UNKNOWN",
                recognition_mode=value if isinstance(value, str) else None,
                reason_code="recognition_mode_unknown",
            ) from exc
        return self._by_id[mode_id]

    def resolve_execution(
        self,
        recognition_mode: RecognitionModeId | str,
        *,
        pipeline_id: str,
        engine: str | None = None,
    ) -> RecognitionModeDefinition:
        definition = self.definition(recognition_mode)
        if pipeline_id != definition.pipeline_id or engine != definition.engine:
            raise RecognitionModeError(
                "RECOGNITION_MODE_PIPELINE_MISMATCH",
                recognition_mode=definition.mode_id.value,
                reason_code="recognition_mode_pipeline_mismatch",
                detail={
                    "expected_pipeline": definition.pipeline_id,
                    "expected_engine": definition.engine,
                    "actual_pipeline": pipeline_id,
                    "actual_engine": engine,
                },
            )
        self._ensure_available(definition)
        return definition

    def resolve_execution_fields(
        self, *, pipeline_id: str, engine: str | None
    ) -> RecognitionModeDefinition:
        """由 strict execution fields 逆向解析唯一 Recognition Mode。"""
        matches = tuple(
            definition
            for definition in self._definitions
            if definition.pipeline_id == pipeline_id and definition.engine == engine
        )
        if len(matches) != 1:
            raise ValueError(
                "pipeline_id and engine do not identify one recognition mode"
            )
        definition = matches[0]
        self._ensure_available(definition)
        return definition

    def _ensure_available(self, definition: RecognitionModeDefinition) -> None:
        availability = self._availability_probe(definition)
        if availability.availability is not RecognitionModeAvailability.READY:
            detail: dict[str, Any] = {}
            if availability.required_component is not None:
                detail["required_component"] = availability.required_component
            raise RecognitionModeError(
                "RECOGNITION_MODE_UNAVAILABLE",
                recognition_mode=definition.mode_id.value,
                reason_code=availability.reason_code or "recognition_mode_unavailable",
                detail=detail,
            )

    def validate_lifecycle(
        self, recognition_mode: RecognitionModeId | str, operation: str
    ) -> RecognitionModeDefinition:
        definition = self.definition(recognition_mode)
        supported = {
            "preload": definition.lifecycle.supports_preload,
            "ttl": definition.lifecycle.supports_ttl,
            "pinning": definition.lifecycle.supports_pinning,
            "release": definition.lifecycle.supports_release,
        }
        if operation not in supported:
            raise ValueError(
                f"unknown recognition mode lifecycle operation: {operation}"
            )
        if not supported[operation]:
            raise RecognitionModeError(
                "RECOGNITION_MODE_LIFECYCLE_UNSUPPORTED",
                recognition_mode=definition.mode_id.value,
                reason_code="recognition_mode_lifecycle_unsupported",
                detail={
                    "operation": operation,
                    "lifecycle_kind": definition.lifecycle.kind.value,
                },
            )
        return definition

    def validate_pipeline_spec(
        self,
        recognition_mode: RecognitionModeId | str,
        *,
        pipeline_id: str,
        pinned: bool,
    ) -> RecognitionModeDefinition:
        """Validate one capability-gated residency setting.

        A ``PipelineSpec`` always configures TTL residency (using its explicit
        TTL or the snapshot default), and may additionally request pinning.
        """
        definition = self.definition(recognition_mode)
        if pipeline_id != definition.pipeline_id:
            raise RecognitionModeError(
                "RECOGNITION_MODE_PIPELINE_MISMATCH",
                recognition_mode=definition.mode_id.value,
                reason_code="recognition_mode_pipeline_mismatch",
                detail={
                    "expected_pipeline": definition.pipeline_id,
                    "actual_pipeline": pipeline_id,
                },
            )
        self.validate_lifecycle(definition.mode_id, "ttl")
        if pinned:
            self.validate_lifecycle(definition.mode_id, "pinning")
        return definition

    def resolve_preload(
        self,
        recognition_modes: tuple[str, ...] | None,
        pipelines: tuple[str, ...],
    ) -> tuple[str, ...]:
        """校验新旧 preload 字段一致，并返回实际 Paddle model 管道。

        legacy ``OCR`` 同时映射三个产品模式，不能再被解释为默认引擎预热。
        MinerU 也不支持显式 preload；它的 child process 只能通过实际任务启动。
        """
        canonical_pipelines = tuple(dict.fromkeys(pipelines))
        if recognition_modes is None:
            ambiguous = [
                pipeline
                for pipeline in canonical_pipelines
                if pipeline in {"OCR", "MinerU"}
            ]
            if ambiguous:
                raise RecognitionModeError(
                    "RECOGNITION_MODE_PIPELINE_MISMATCH",
                    recognition_mode=None,
                    reason_code="legacy_pipeline_does_not_identify_preloadable_mode",
                    detail={"pipelines": ambiguous},
                )
            definitions = tuple(
                definition
                for pipeline in canonical_pipelines
                for definition in self._definitions
                if definition.pipeline_id == pipeline
            )
            if len(definitions) != len(canonical_pipelines):
                raise RecognitionModeError(
                    "RECOGNITION_MODE_PIPELINE_MISMATCH",
                    recognition_mode=None,
                    reason_code="legacy_pipeline_does_not_identify_preloadable_mode",
                    detail={"pipelines": list(canonical_pipelines)},
                )
            for definition in definitions:
                self.validate_lifecycle(definition.mode_id, "preload")
                self._ensure_available(definition)
            return canonical_pipelines

        definitions = tuple(self.definition(mode) for mode in recognition_modes)
        mapped = tuple(
            dict.fromkeys(definition.pipeline_id for definition in definitions)
        )
        if len(mapped) != len(canonical_pipelines) or set(mapped) != set(
            canonical_pipelines
        ):
            raise RecognitionModeError(
                "RECOGNITION_MODE_PIPELINE_MISMATCH",
                recognition_mode=None,
                reason_code="recognition_modes_do_not_match_pipelines",
                detail={
                    "recognition_modes": [
                        definition.mode_id.value for definition in definitions
                    ],
                    "expected_pipelines": list(mapped),
                    "actual_pipelines": list(canonical_pipelines),
                },
            )
        for definition in definitions:
            self.validate_lifecycle(definition.mode_id, "preload")
            self._ensure_available(definition)
        return mapped

    def annotate_residency_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """为 residency DTO 投影稳定的 model/process 资源身份。"""
        result = dict(payload)
        entries: list[dict[str, Any]] = []
        managed_by_pipeline = {
            definition.pipeline_id: definition
            for definition in self._definitions
            if definition.lifecycle.kind is not LifecycleKind.UNMANAGED
        }
        for raw_entry in payload.get("entries", []):
            entry = dict(raw_entry)
            definition = managed_by_pipeline.get(entry.get("pipeline"))
            if definition is not None:
                is_process = (
                    definition.lifecycle.kind is LifecycleKind.PROCESS_KEEP_ALIVE
                )
                entry.update(
                    {
                        "recognition_mode": definition.mode_id.value,
                        "resource_kind": "process" if is_process else "model",
                        "resource_id": (
                            "mineru-api" if is_process else definition.mode_id.value
                        ),
                    }
                )
            entries.append(entry)
        result["entries"] = entries
        return result

    @staticmethod
    def _default_availability(
        definition: RecognitionModeDefinition,
    ) -> ModeAvailability:
        if definition.provisioning is ProvisioningKind.ADVANCED_COMPONENT:
            return ModeAvailability(
                RecognitionModeAvailability.PREPARATION_REQUIRED,
                reason_code="runtime_component_missing",
                required_component=definition.required_component,
            )
        return ModeAvailability(RecognitionModeAvailability.READY)


__all__ = [
    "LifecycleKind",
    "ModeAvailability",
    "ProvisioningKind",
    "RecognitionModeAvailability",
    "RecognitionModeDefinition",
    "RecognitionModeError",
    "RecognitionModeFamily",
    "RecognitionModeId",
    "RecognitionModeLifecycle",
    "RecognitionModeRegistry",
]
