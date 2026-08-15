"""OcrEngineRoutingAdapter 测试：OCR pipeline 按引擎路由、其余直达 fallback。

另覆盖 executor 对 OcrEngineError 的 fail-fast：确定性引擎错误按协议
错误码标记 item，不进入恢复重试，也不切换引擎。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from vibeocr.backend.supervisor.inference.budgets import AdapterCapability, InputItem
from vibeocr.backend.supervisor.inference.ocr_engine_router import (
    OcrEngineRoutingAdapter,
)
from vibeocr.backend.supervisor.inference.ocr_engines import (
    EngineAvailability,
    EngineDescriptor,
    OcrEngineError,
    OcrEngineRegistry,
    OcrEngineResolver,
)
from vibeocr.runtime_contracts import (
    ErrorCode,
    ItemState,
    JobKind,
    JobPriority,
    JobState,
)
from vibeocr.runtime_contracts.dtos import JobItem, OcrEngine, PipelineSelection


@dataclass
class RecordingEngine:
    engine_id: OcrEngine
    availability: EngineAvailability = EngineAvailability.READY
    recognize_calls: list[list[InputItem]] = field(default_factory=list)
    preload_calls: list[tuple[str, ...]] = field(default_factory=list)
    closed: bool = False

    def descriptor(self) -> EngineDescriptor:
        return EngineDescriptor(
            engine_id=self.engine_id, availability=self.availability
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
        return [{"engine": self.engine_id.value} for _ in items]

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


@dataclass
class RecordingFallback:
    recognize_calls: list[Any] = field(default_factory=list)
    capabilities_calls: list[Any] = field(default_factory=list)
    preload_calls: list[tuple[str, ...]] = field(default_factory=list)
    closed: bool = False

    def capabilities(self, options: Any | None = None) -> AdapterCapability:
        self.capabilities_calls.append(options)
        return AdapterCapability(name="fallback", real_batch=False, max_compute_batch=1)

    def recognize_many(
        self,
        items: list[InputItem],
        *,
        options: Any | None = None,
        compute_batch: Any | None = None,
    ) -> list[dict[str, Any]]:
        self.recognize_calls.append((list(items), options))
        return [{"engine": "fallback"} for _ in items]

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


def _input(item_id: str = "it-1") -> InputItem:
    return InputItem(item_id=item_id, encoded_bytes=4, decoded_pixels=10, data=b"png")


def _router(
    engines: list[RecordingEngine], fallback: RecordingFallback
) -> tuple[OcrEngineRoutingAdapter, OcrEngineRegistry]:
    registry = OcrEngineRegistry(engines)
    return (
        OcrEngineRoutingAdapter(
            fallback_factory=lambda: fallback, resolver=OcrEngineResolver(registry)
        ),
        registry,
    )


class TestRouting:
    def test_ocr_default_routes_to_rapidocr(self) -> None:
        rapid = RecordingEngine(OcrEngine.RAPIDOCR)
        paddle = RecordingEngine(OcrEngine.PADDLEOCR)
        fallback = RecordingFallback()
        router, _ = _router([rapid, paddle], fallback)
        options = PipelineSelection(pipeline_id="OCR", options={})
        payloads = router.recognize_many([_input()], options=options)
        assert payloads == [{"engine": "rapidocr"}]
        assert rapid.recognize_calls and not paddle.recognize_calls
        assert not fallback.recognize_calls

    def test_explicit_engine_selection_is_honoured(self) -> None:
        rapid = RecordingEngine(OcrEngine.RAPIDOCR)
        windows = RecordingEngine(OcrEngine.WINDOWS)
        fallback = RecordingFallback()
        router, _ = _router([rapid, windows], fallback)
        options = PipelineSelection(
            pipeline_id="OCR", options={}, engine=OcrEngine.WINDOWS
        )
        assert router.recognize_many([_input()], options=options) == [
            {"engine": "windows"}
        ]
        assert not rapid.recognize_calls

    def test_unavailable_engine_fails_closed_without_fallback(self) -> None:
        # paddle ready 但请求 windows（未注册）：错误，不落到 fallback/paddle。
        paddle = RecordingEngine(OcrEngine.PADDLEOCR)
        fallback = RecordingFallback()
        router, _ = _router([paddle], fallback)
        options = PipelineSelection(
            pipeline_id="OCR", options={}, engine=OcrEngine.WINDOWS
        )
        with pytest.raises(OcrEngineError) as excinfo:
            router.recognize_many([_input()], options=options)
        assert excinfo.value.code is ErrorCode.OCR_ENGINE_UNAVAILABLE
        assert not fallback.recognize_calls
        assert not paddle.recognize_calls

    def test_unavailable_engine_capabilities_raises(self) -> None:
        windows = RecordingEngine(
            OcrEngine.WINDOWS, availability=EngineAvailability.UNAVAILABLE
        )
        fallback = RecordingFallback()
        router, _ = _router([windows], fallback)
        options = PipelineSelection(
            pipeline_id="OCR", options={}, engine=OcrEngine.WINDOWS
        )
        with pytest.raises(OcrEngineError):
            router.capabilities(options)

    def test_non_ocr_pipeline_goes_to_fallback(self) -> None:
        rapid = RecordingEngine(OcrEngine.RAPIDOCR)
        fallback = RecordingFallback()
        router, _ = _router([rapid], fallback)
        options = PipelineSelection(pipeline_id="TABLE_RECOGNITION", options={})
        assert router.recognize_many([_input()], options=options) == [
            {"engine": "fallback"}
        ]
        assert not rapid.recognize_calls
        router.capabilities(options)
        assert fallback.capabilities_calls == [options]

    def test_ocr_capabilities_delegate_to_resolved_engine(self) -> None:
        rapid = RecordingEngine(OcrEngine.RAPIDOCR)
        fallback = RecordingFallback()
        router, _ = _router([rapid], fallback)
        options = PipelineSelection(pipeline_id="OCR", options={})
        capability = router.capabilities(options)
        assert capability.name == "OCR"
        assert not fallback.capabilities_calls


class TestLifecycle:
    def test_preload_splits_ocr_from_other_pipelines(self) -> None:
        rapid = RecordingEngine(OcrEngine.RAPIDOCR)
        fallback = RecordingFallback()
        router, _ = _router([rapid], fallback)
        router.preload(("OCR", "TABLE_RECOGNITION", "FORMULA_RECOGNITION"))
        # OCR 走默认引擎，其余管道直达 fallback。
        assert rapid.preload_calls == [("OCR",)]
        assert fallback.preload_calls == [("TABLE_RECOGNITION", "FORMULA_RECOGNITION")]

    def test_preload_non_ocr_only_skips_engines(self) -> None:
        rapid = RecordingEngine(OcrEngine.RAPIDOCR)
        fallback = RecordingFallback()
        router, _ = _router([rapid], fallback)
        router.preload(("MinerU",))
        assert rapid.preload_calls == []
        assert fallback.preload_calls == [("MinerU",)]

    def test_residency_and_settings_delegate_to_fallback(self) -> None:
        from vibeocr.runtime_contracts import ResidencyStatus, SettingsSnapshot

        rapid = RecordingEngine(OcrEngine.RAPIDOCR)
        fallback = RecordingFallback()
        router, _ = _router([rapid], fallback)
        assert isinstance(router.residency_status(), ResidencyStatus)
        assert isinstance(router.release_idle("OCR"), ResidencyStatus)
        router.configure_settings(SettingsSnapshot())

    def test_close_closes_engines_and_fallback(self) -> None:
        rapid = RecordingEngine(OcrEngine.RAPIDOCR)
        windows = RecordingEngine(OcrEngine.WINDOWS)
        fallback = RecordingFallback()
        router, _ = _router([rapid, windows], fallback)
        router.close()
        assert rapid.closed and windows.closed and fallback.closed


class TestExecutorFailFast:
    def test_engine_error_fails_items_without_recovery(self, tmp_path) -> None:
        """AdapterExecutor 集成：引擎错误按协议码标 item，无恢复事件循环。"""

        def _staged_pngs(items: list[JobItem], base: Any) -> list[Any]:
            import io

            from PIL import Image
            from vibeocr.backend.supervisor.jobs.staging import StagedInput

            buffer = io.BytesIO()
            Image.new("RGB", (1, 1)).save(buffer, format="PNG")
            png = buffer.getvalue()
            staged = []
            for index, item in enumerate(items):
                path = base / f"f{index}.png"
                path.write_bytes(png)
                staged.append(
                    StagedInput(
                        item_id=item.item_id,
                        display_name=item.display_name,
                        path=path,
                        size_bytes=len(png),
                    )
                )
            return staged

        from vibeocr.backend.supervisor.inference.paddle_executor import AdapterExecutor
        from vibeocr.backend.supervisor.jobs.registry import JobRegistry

        class _EngineErrorAdapter:
            def capabilities(self, options: Any | None = None) -> AdapterCapability:
                del options
                return AdapterCapability(
                    name="OCR", real_batch=False, max_compute_batch=1
                )

            def recognize_many(
                self,
                items: list[InputItem],
                *,
                options: Any | None = None,
                compute_batch: Any | None = None,
            ) -> list[dict[str, Any]]:
                raise OcrEngineError(
                    ErrorCode.OCR_ENGINE_UNAVAILABLE,
                    reason_code="engine_not_installed",
                    engine="rapidocr",
                )

            def residency_status(self) -> Any:
                from vibeocr.runtime_contracts import ResidencyStatus

                return ResidencyStatus()

            def release_idle(self, pipeline: str | None = None) -> Any:
                return self.residency_status()

        registry = JobRegistry(instance_id="t-router")
        record = registry.create(
            kind=JobKind.RECOGNITION,
            priority=JobPriority.INTERACTIVE,
            items=[
                JobItem(
                    item_id=f"it-{i}",
                    display_name=f"f{i}.png",
                    state=ItemState.QUEUED,
                )
                for i in range(2)
            ],
            progress_total=2,
            pipeline=PipelineSelection(pipeline_id="OCR", options={}),
        )
        record.transition(JobState.QUEUED)
        staged = _staged_pngs(record.items, tmp_path)
        executor = AdapterExecutor(adapter_factory=_EngineErrorAdapter, device="cpu:0")
        executor.execute(record, staged)
        errors = list(record.item_errors.values())
        assert errors == ["OCR_ENGINE_UNAVAILABLE", "OCR_ENGINE_UNAVAILABLE"]
        event_names = [event.stage for event in record.events]
        assert "ocr_engine_rejected" in event_names
        # 不进入恢复路径：没有 recovery_decision 事件。
        assert "recovery_decision" not in event_names
