"""通用 OCR 引擎深模块测试：registry/resolver 语义与共享引擎契约。

覆盖计划 §3.2 的关键不变量：

* 目录覆盖全部稳定 engine id，且来自实际探针（含 descriptor 抛错降级）。
* resolver 只按显式 ID 选择；缺省用 rapidocr；不可用时返回结构化
  协议错误并回显可选引擎，绝不静默切换。
* 未知 engine 值 fail closed 为 OCR_ENGINE_UNKNOWN；engine 只对
  ``OCR`` pipeline 合法。
* 三个引擎 adapter（fake 驱动 rapidocr / windows / paddle）共享同一
  contract suite：descriptor/capabilities/recognize_many 顺序与形状。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from vibeocr.backend.supervisor.inference.budgets import AdapterCapability, InputItem
from vibeocr.backend.supervisor.inference.ocr_engines import (
    DEFAULT_OCR_ENGINE,
    REASON_ENGINE_INIT_FAILED,
    REASON_ENGINE_LANGUAGE_UNAVAILABLE,
    REASON_ENGINE_NOT_INSTALLED,
    EngineAvailability,
    EngineDescriptor,
    LazyEngineHandle,
    OcrEngineError,
    OcrEngineRegistry,
    OcrEngineResolver,
    ensure_engine_valid_for_pipeline,
    parse_wire_engine,
)
from vibeocr.runtime_contracts import ErrorCode
from vibeocr.runtime_contracts.dtos import OcrEngine

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeEngine:
    engine_id: OcrEngine
    availability: EngineAvailability = EngineAvailability.READY
    reason_code: str | None = None
    required_component: str | None = None
    included_in_base: bool = False
    recognize_calls: list[list[InputItem]] = field(default_factory=list)
    preload_calls: list[tuple[str, ...]] = field(default_factory=list)
    closed: bool = False

    def descriptor(self) -> EngineDescriptor:
        if (
            self.availability is EngineAvailability.UNAVAILABLE
            and self.reason_code == "boom"
        ):
            raise RuntimeError("probe exploded")
        return EngineDescriptor(
            engine_id=self.engine_id,
            availability=self.availability,
            included_in_base=self.included_in_base,
            reason_code=self.reason_code,
            required_component=self.required_component,
        )

    def capabilities(self, options: Any | None = None) -> AdapterCapability:
        del options
        return AdapterCapability(name="OCR", real_batch=False, max_compute_batch=1)

    def recognize_many(
        self,
        items: list[InputItem],
        *,
        options: Any | None = None,
        compute_batch: Any | None = None,
    ) -> list[dict[str, Any]]:
        self.recognize_calls.append(list(items))
        return [{"text": f"{self.engine_id.value}:{item.item_id}"} for item in items]

    def preload(self, pipelines: tuple[str, ...]) -> Any:
        self.preload_calls.append(tuple(pipelines))
        from vibeocr.runtime_contracts import ResidencyStatus

        return ResidencyStatus()

    def residency_status(self) -> Any:
        from vibeocr.runtime_contracts import ResidencyStatus

        return ResidencyStatus()

    def release_idle(self, pipeline: str | None = None) -> Any:
        return self.residency_status()

    def configure_settings(self, snapshot: Any) -> None:
        del snapshot

    def close(self) -> None:
        self.closed = True


def _input(item_id: str = "it-1", data: bytes = b"img") -> InputItem:
    return InputItem(
        item_id=item_id, encoded_bytes=len(data), decoded_pixels=10, data=data
    )


# ---------------------------------------------------------------------------
# parse_wire_engine / pipeline 校验
# ---------------------------------------------------------------------------


class TestParseWireEngine:
    def test_accepts_stable_ids(self) -> None:
        assert parse_wire_engine("rapidocr") is OcrEngine.RAPIDOCR
        assert parse_wire_engine("windows") is OcrEngine.WINDOWS
        assert parse_wire_engine("paddleocr") is OcrEngine.PADDLEOCR
        # enum 实例直接透传（服务端内部路径）。
        assert parse_wire_engine(OcrEngine.WINDOWS) is OcrEngine.WINDOWS

    @pytest.mark.parametrize("value", ["paddle", "", "PaddleOCR", 42, None])
    def test_unknown_values_fail_closed(self, value: Any) -> None:
        with pytest.raises(OcrEngineError) as excinfo:
            parse_wire_engine(value)
        assert excinfo.value.code is ErrorCode.OCR_ENGINE_UNKNOWN


class TestEngineValidForPipeline:
    def test_ocr_pipeline_accepts_any_engine(self) -> None:
        ensure_engine_valid_for_pipeline(OcrEngine.WINDOWS, "OCR")

    def test_engine_none_is_always_valid(self) -> None:
        ensure_engine_valid_for_pipeline(None, "TABLE_RECOGNITION")

    @pytest.mark.parametrize(
        "pipeline", ["TABLE_RECOGNITION", "FORMULA_RECOGNITION", "PP-StructureV3"]
    )
    def test_non_ocr_pipeline_rejects_engine(self, pipeline: str) -> None:
        with pytest.raises(OcrEngineError) as excinfo:
            ensure_engine_valid_for_pipeline(OcrEngine.RAPIDOCR, pipeline)
        assert excinfo.value.code is ErrorCode.OCR_ENGINE_NOT_VALID_FOR_PIPELINE


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestOcrEngineRegistry:
    def test_catalog_covers_all_stable_ids_when_empty(self) -> None:
        registry = OcrEngineRegistry([])
        payload = registry.catalog_payload()
        ids = [entry["id"] for entry in payload["engines"]]
        assert ids == ["rapidocr", "windows", "paddleocr"]
        assert all(
            entry["availability"] == "unavailable"
            and entry["reason_code"] == REASON_ENGINE_NOT_INSTALLED
            for entry in payload["engines"]
        )

    def test_catalog_reflects_registered_descriptors(self) -> None:
        rapid = FakeEngine(OcrEngine.RAPIDOCR, included_in_base=True)
        paddle = FakeEngine(
            OcrEngine.PADDLEOCR,
            availability=EngineAvailability.PREPARATION_REQUIRED,
            reason_code=REASON_ENGINE_NOT_INSTALLED,
            required_component="document_parsing",
        )
        registry = OcrEngineRegistry([rapid, paddle])
        entries = {e["id"]: e for e in registry.catalog_payload()["engines"]}
        assert entries["rapidocr"]["availability"] == "ready"
        assert entries["rapidocr"]["included_in_base"] is True
        assert entries["paddleocr"]["availability"] == "preparation_required"
        assert entries["paddleocr"]["required_component"] == "document_parsing"
        # 未注册的 windows 保持 unavailable 占位。
        assert entries["windows"]["availability"] == "unavailable"

    def test_duplicate_registration_rejected(self) -> None:
        registry = OcrEngineRegistry([FakeEngine(OcrEngine.RAPIDOCR)])
        with pytest.raises(ValueError, match="duplicate"):
            registry.register(FakeEngine(OcrEngine.RAPIDOCR))

    def test_invalid_engine_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="engine_id"):
            OcrEngineRegistry(["not-an-engine"])

    def test_descriptor_probe_exception_maps_to_unavailable(self) -> None:
        broken = FakeEngine(
            OcrEngine.WINDOWS,
            availability=EngineAvailability.UNAVAILABLE,
            reason_code="boom",
        )
        registry = OcrEngineRegistry([broken])
        entries = {e["id"]: e for e in registry.catalog_payload()["engines"]}
        assert entries["windows"]["availability"] == "unavailable"
        assert entries["windows"]["reason_code"] == REASON_ENGINE_INIT_FAILED

    def test_selectable_engine_ids_only_ready_in_protocol_order(self) -> None:
        registry = OcrEngineRegistry(
            [
                FakeEngine(OcrEngine.PADDLEOCR),
                FakeEngine(OcrEngine.RAPIDOCR),
                FakeEngine(
                    OcrEngine.WINDOWS,
                    availability=EngineAvailability.UNAVAILABLE,
                    reason_code=REASON_ENGINE_LANGUAGE_UNAVAILABLE,
                ),
            ]
        )
        assert registry.selectable_engine_ids() == ("rapidocr", "paddleocr")


# ---------------------------------------------------------------------------
# Resolver：无静默回退
# ---------------------------------------------------------------------------


class TestOcrEngineResolver:
    def test_default_engine_is_rapidocr(self) -> None:
        assert DEFAULT_OCR_ENGINE is OcrEngine.RAPIDOCR
        assert OcrEngineRegistry([]).default_engine is OcrEngine.RAPIDOCR

    def test_resolve_ready_engine(self) -> None:
        engine = FakeEngine(OcrEngine.RAPIDOCR, included_in_base=True)
        resolver = OcrEngineResolver(registry=OcrEngineRegistry([engine]))
        assert resolver.resolve(OcrEngine.RAPIDOCR) is engine
        # 缺省解析到同一引擎。
        assert resolver.resolve(None) is engine

    def test_unregistered_engine_fails_closed_without_fallback(self) -> None:
        paddle = FakeEngine(OcrEngine.PADDLEOCR)
        resolver = OcrEngineResolver(registry=OcrEngineRegistry([paddle]))
        with pytest.raises(OcrEngineError) as excinfo:
            resolver.resolve(OcrEngine.RAPIDOCR)
        assert excinfo.value.code is ErrorCode.OCR_ENGINE_UNAVAILABLE
        assert excinfo.value.reason_code == REASON_ENGINE_NOT_INSTALLED
        assert excinfo.value.engine == "rapidocr"
        # 可选回显只包含 ready 引擎，绝不返回 paddle 作为替代。
        assert excinfo.value.selectable_engines == ("paddleocr",)
        assert resolver.resolve(OcrEngine.PADDLEOCR) is paddle

    def test_unavailable_engine_fails_closed_without_fallback(self) -> None:
        rapid = FakeEngine(OcrEngine.RAPIDOCR)
        windows = FakeEngine(
            OcrEngine.WINDOWS,
            availability=EngineAvailability.UNAVAILABLE,
            reason_code=REASON_ENGINE_LANGUAGE_UNAVAILABLE,
        )
        resolver = OcrEngineResolver(registry=OcrEngineRegistry([rapid, windows]))
        with pytest.raises(OcrEngineError) as excinfo:
            resolver.resolve(OcrEngine.WINDOWS)
        assert excinfo.value.code is ErrorCode.OCR_ENGINE_UNAVAILABLE
        assert excinfo.value.reason_code == REASON_ENGINE_LANGUAGE_UNAVAILABLE
        assert excinfo.value.selectable_engines == ("rapidocr",)

    def test_preparation_required_maps_to_428_code(self) -> None:
        paddle = FakeEngine(
            OcrEngine.PADDLEOCR,
            availability=EngineAvailability.PREPARATION_REQUIRED,
            reason_code=REASON_ENGINE_NOT_INSTALLED,
            required_component="document_parsing",
        )
        resolver = OcrEngineResolver(registry=OcrEngineRegistry([paddle]))
        with pytest.raises(OcrEngineError) as excinfo:
            resolver.validate(OcrEngine.PADDLEOCR)
        assert excinfo.value.code is ErrorCode.OCR_ENGINE_PREPARATION_REQUIRED
        assert excinfo.value.detail == {"required_component": "document_parsing"}

    def test_default_engine_unavailable_fails_closed(self) -> None:
        # 缺省引擎不可用时同样 fail closed，而不是切到其他 ready 引擎。
        windows = FakeEngine(OcrEngine.WINDOWS)
        resolver = OcrEngineResolver(registry=OcrEngineRegistry([windows]))
        with pytest.raises(OcrEngineError) as excinfo:
            resolver.resolve(None)
        assert excinfo.value.code is ErrorCode.OCR_ENGINE_UNAVAILABLE
        assert excinfo.value.selectable_engines == ("windows",)

    def test_probe_cache_and_invalidation(self) -> None:
        engine = FakeEngine(OcrEngine.RAPIDOCR)
        resolver = OcrEngineResolver(registry=OcrEngineRegistry([engine]))
        resolver.validate(None)
        # 变为不可用后缓存仍放行——component 修复/安装需显式失效。
        engine.availability = EngineAvailability.UNAVAILABLE
        resolver.validate(None)
        resolver.invalidate_probe_cache(OcrEngine.RAPIDOCR)
        with pytest.raises(OcrEngineError):
            resolver.validate(None)


# ---------------------------------------------------------------------------
# LazyEngineHandle
# ---------------------------------------------------------------------------


class TestLazyEngineHandle:
    def test_descriptor_uses_probe_without_building_instance(self) -> None:
        built: list[bool] = []

        def factory() -> FakeEngine:
            built.append(True)
            return FakeEngine(OcrEngine.PADDLEOCR)

        handle = LazyEngineHandle(
            engine_id=OcrEngine.PADDLEOCR,
            descriptor_probe=lambda: EngineDescriptor(
                engine_id=OcrEngine.PADDLEOCR,
                availability=EngineAvailability.READY,
            ),
            factory=factory,
        )
        assert handle.descriptor().availability is EngineAvailability.READY
        assert built == []
        # 首次使用才构建，且复用同一实例。
        first = handle.instance()
        handle.recognize_many([_input()])
        assert built == [True]
        assert handle.instance() is first

    def test_lifecycle_delegates_only_when_built(self) -> None:
        engine = FakeEngine(OcrEngine.PADDLEOCR)
        handle = LazyEngineHandle(
            engine_id=OcrEngine.PADDLEOCR,
            descriptor_probe=lambda: EngineDescriptor(
                engine_id=OcrEngine.PADDLEOCR,
                availability=EngineAvailability.READY,
            ),
            factory=lambda: engine,
        )
        # 未构建时 residency 返回空快照，close 不触发构建。
        from vibeocr.runtime_contracts import ResidencyStatus

        assert isinstance(handle.residency_status(), ResidencyStatus)
        handle.close()
        assert engine.closed is False
        handle.instance()
        handle.close()
        assert engine.closed is True


# ---------------------------------------------------------------------------
# 共享引擎契约套件：所有 adapter 实现同形语义
# ---------------------------------------------------------------------------


def engine_contract_suite(engine: Any, *, expected_id: OcrEngine) -> None:
    """三 adapter 共享契约：descriptor/capabilities/顺序保持/lifecycle。"""
    assert engine.engine_id is expected_id
    descriptor = engine.descriptor()
    assert descriptor.engine_id is expected_id
    payload = descriptor.to_payload()
    assert payload["id"] == expected_id.value
    assert payload["availability"] in {
        "ready",
        "preparation_required",
        "unavailable",
    }

    capability = engine.capabilities(None)
    assert isinstance(capability, AdapterCapability)

    items = [_input("a"), _input("b")]
    payloads = engine.recognize_many(items)
    assert len(payloads) == len(items)
    assert all(isinstance(p, dict) and p for p in payloads)

    from vibeocr.runtime_contracts import ResidencyStatus

    assert isinstance(engine.residency_status(), ResidencyStatus)
    engine.close()


def test_fake_engines_satisfy_shared_contract() -> None:
    for engine in (
        FakeEngine(OcrEngine.RAPIDOCR),
        FakeEngine(OcrEngine.WINDOWS),
        FakeEngine(OcrEngine.PADDLEOCR),
    ):
        engine_contract_suite(engine, expected_id=engine.engine_id)
