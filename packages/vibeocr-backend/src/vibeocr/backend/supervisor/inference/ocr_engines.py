"""通用文本 OCR 引擎深模块：稳定接口、registry、resolver 与 fail-closed 错误。

Plan（docs/ocr-engine-runtime-profiles-execution-plan.md §3）约束：

* ``GeneralTextOcrEngine`` 是 application/core 边界上与具体库无关的接口；
  RapidOCR / Windows Media OCR / PaddleOCR 三个 adapter 在接口之后实现。
* ``OcrEngineRegistry`` 负责注册、探测和缓存；engine 目录来自实际探针和
  已安装 component，不从静态依赖列表猜测。
* ``OcrEngineResolver`` 只按 Protocol 的显式 engine ID 选择：缺省时使用
  Backend 默认引擎（rapidocr），用户选择的引擎不可用时返回结构化协议
  错误（OCR_ENGINE_UNAVAILABLE / OCR_ENGINE_PREPARATION_REQUIRED /
  OCR_ENGINE_LANGUAGE_UNAVAILABLE），绝不静默切换到其他引擎。
* 未知 engine 值 fail closed 为 OCR_ENGINE_UNKNOWN；engine 只对纯文本
  ``OCR`` pipeline 合法，其他 pipeline 拒绝为
  OCR_ENGINE_NOT_VALID_FOR_PIPELINE。

引擎接口刻意与 supervisor 的 adapter seam（``capabilities`` /
``recognize_many`` / residency / lifecycle）同形，这样现有
``PaddleExecutor`` 的调度、恢复与预算机制无需复制即可承载任意引擎。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from vibeocr.runtime_contracts import ErrorCode
from vibeocr.runtime_contracts.dtos import OcrEngine

logger = logging.getLogger(__name__)

# Protocol 固定：请求缺省 engine 时 Backend 默认 rapidocr。
DEFAULT_OCR_ENGINE = OcrEngine.RAPIDOCR

# catalog 必须覆盖全部稳定 engine id（每个 id 恰好一个 descriptor）。
_STABLE_ENGINE_ORDER: tuple[OcrEngine, ...] = (
    OcrEngine.RAPIDOCR,
    OcrEngine.WINDOWS,
    OcrEngine.PADDLEOCR,
)

OCR_PIPELINE_ID = "OCR"

# 稳定 reason_code（机器可读，前端自行本地化，Backend 不发展示文案）。
REASON_ENGINE_NOT_INSTALLED = "engine_not_installed"
REASON_ENGINE_INIT_FAILED = "engine_init_failed"
REASON_ENGINE_LANGUAGE_UNAVAILABLE = "engine_language_unavailable"


class EngineAvailability(StrEnum):
    """单个引擎在当前 runtime 中的可用性（wire: OcrEngineAvailability）。"""

    READY = "ready"
    PREPARATION_REQUIRED = "preparation_required"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class EngineDescriptor:
    """单个引擎的能力与状态（wire: OcrEngineDescriptor）。

    reason_code 仅在非 ready 时非空；required_component 是用户需要准备的
    runtime component id（runtime.component-repair.v1），无要求时为 None。
    """

    engine_id: OcrEngine
    availability: EngineAvailability
    included_in_base: bool = False
    reason_code: str | None = None
    required_component: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.engine_id.value,
            "availability": self.availability.value,
            "included_in_base": self.included_in_base,
            "reason_code": self.reason_code,
            "required_component": self.required_component,
        }


class OcrEngineError(Exception):
    """引擎选择失败，映射到协议错误码并携带结构化 detail。

    code 是 ``ErrorCode``；reason_code 是稳定机器可读原因；
    selectable_engines 是当前 ready 的 engine id（可选回显）。
    """

    def __init__(
        self,
        code: ErrorCode,
        *,
        reason_code: str,
        engine: str | None = None,
        selectable_engines: tuple[str, ...] = (),
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{code.value}: {reason_code}")
        self.code = code
        self.reason_code = reason_code
        self.engine = engine
        self.selectable_engines = tuple(selectable_engines)
        self.detail = dict(detail or {})


@runtime_checkable
class GeneralTextOcrEngine(Protocol):
    """纯文本 OCR 引擎的稳定接口。

    实现方：``RapidOcrEngine``、``WindowsMediaOcrEngine``、Paddle
    adapter（pipeline ``OCR`` 视角）。除 ``descriptor``/``engine_id`` 外，
    其余方法与 supervisor adapter seam 同形，可直接被 executor 消费。
    """

    @property
    def engine_id(self) -> OcrEngine: ...

    def descriptor(self) -> EngineDescriptor:
        """返回 ID、availability、reason 与所需 component 的实时探针。"""
        ...

    def capabilities(self, options: Any | None = None) -> Any:
        """返回该引擎的批处理能力（AdapterCapability）。"""
        ...

    def recognize_many(
        self,
        items: list[Any],
        *,
        options: Any | None = None,
        compute_batch: Any | None = None,
    ) -> list[dict[str, Any]]:
        """统一单图/批量入口；输出顺序与输入一致。"""
        ...

    def preload(self, pipelines: tuple[str, ...]) -> Any: ...

    def residency_status(self) -> Any: ...

    def release_idle(self, pipeline: str | None = None) -> Any: ...

    def configure_settings(self, snapshot: Any) -> None: ...

    def close(self) -> None: ...


def parse_wire_engine(value: Any) -> OcrEngine:
    """把 wire 上的 engine 值严格解析为稳定 OcrEngine。

    未知值 fail closed 为 OCR_ENGINE_UNKNOWN，绝不猜测或回退。
    """
    if isinstance(value, OcrEngine):
        return value
    if isinstance(value, str):
        try:
            return OcrEngine(value)
        except ValueError:
            pass
    raise OcrEngineError(
        ErrorCode.OCR_ENGINE_UNKNOWN,
        reason_code="engine_id_unknown",
        engine=value if isinstance(value, str) else None,
    )


def ensure_engine_valid_for_pipeline(
    engine: OcrEngine | None, pipeline_id: str | None
) -> None:
    """engine 仅对纯文本 ``OCR`` pipeline 合法，其余拒绝而非静默忽略。"""
    if (
        engine is not None
        and pipeline_id is not None
        and pipeline_id != OCR_PIPELINE_ID
    ):
        raise OcrEngineError(
            ErrorCode.OCR_ENGINE_NOT_VALID_FOR_PIPELINE,
            reason_code="engine_only_valid_for_ocr_pipeline",
            engine=engine.value,
        )


class OcrEngineRegistry:
    """注册、探测并缓存各引擎 adapter；目录来自实际探针。

    未注册实现的稳定 engine id 以 ``engine_not_installed`` 的
    unavailable descriptor 出现在目录中，保证 catalog 覆盖全部 id。
    """

    def __init__(
        self,
        engines: list[Any] | None = None,
        *,
        default_engine: OcrEngine = DEFAULT_OCR_ENGINE,
    ) -> None:
        self._engines: dict[OcrEngine, Any] = {}
        self._default_engine = default_engine
        for engine in engines or []:
            self.register(engine)

    @property
    def default_engine(self) -> OcrEngine:
        return self._default_engine

    def register(self, engine: Any) -> None:
        engine_id = getattr(engine, "engine_id", None)
        if not isinstance(engine_id, OcrEngine):
            raise ValueError(
                f"engine must expose an OcrEngine engine_id, got {engine!r}"
            )
        if engine_id in self._engines:
            raise ValueError(f"duplicate OCR engine registration: {engine_id.value}")
        self._engines[engine_id] = engine

    def get(self, engine_id: OcrEngine) -> Any | None:
        return self._engines.get(engine_id)

    def engines(self) -> tuple[Any, ...]:
        return tuple(
            self._engines[engine_id]
            for engine_id in _STABLE_ENGINE_ORDER
            if engine_id in self._engines
        )

    def probe_descriptors(self) -> list[EngineDescriptor]:
        """按协议顺序探测全部稳定 engine id（含未注册 id 的占位）。"""
        descriptors: list[EngineDescriptor] = []
        for engine_id in _STABLE_ENGINE_ORDER:
            engine = self._engines.get(engine_id)
            if engine is None:
                descriptors.append(
                    EngineDescriptor(
                        engine_id=engine_id,
                        availability=EngineAvailability.UNAVAILABLE,
                        reason_code=REASON_ENGINE_NOT_INSTALLED,
                    )
                )
                continue
            try:
                descriptor = engine.descriptor()
            except Exception:
                logger.exception(
                    "[Supervisor][OcrEngines] probe failed engine=%s",
                    engine_id.value,
                )
                descriptor = EngineDescriptor(
                    engine_id=engine_id,
                    availability=EngineAvailability.UNAVAILABLE,
                    reason_code=REASON_ENGINE_INIT_FAILED,
                )
            descriptors.append(descriptor)
        return descriptors

    def selectable_engine_ids(self) -> tuple[str, ...]:
        """当前 ready 的 engine id（按协议顺序）。"""
        return tuple(
            descriptor.engine_id.value
            for descriptor in self.probe_descriptors()
            if descriptor.availability is EngineAvailability.READY
        )

    def catalog_payload(self) -> dict[str, Any]:
        """OcrEngineCatalog wire payload：每个稳定 id 恰好一个 descriptor。"""
        return {
            "engines": [
                descriptor.to_payload() for descriptor in self.probe_descriptors()
            ]
        }


@dataclass
class OcrEngineResolver:
    """只按显式 engine ID 选择；缺省用 Backend 默认引擎；绝不静默切换。"""

    registry: OcrEngineRegistry
    _probe_cache: dict[OcrEngine, EngineDescriptor] = field(default_factory=dict)

    def resolve(self, engine: OcrEngine | None) -> Any:
        """解析引擎实例；不可用时抛 OcrEngineError（fail closed）。"""
        target = engine if engine is not None else self.registry.default_engine
        adapter = self.registry.get(target)
        if adapter is None:
            raise OcrEngineError(
                ErrorCode.OCR_ENGINE_UNAVAILABLE,
                reason_code=REASON_ENGINE_NOT_INSTALLED,
                engine=target.value,
                selectable_engines=self.registry.selectable_engine_ids(),
            )
        descriptor = self._descriptor_for(adapter)
        if descriptor.availability is EngineAvailability.PREPARATION_REQUIRED:
            raise OcrEngineError(
                ErrorCode.OCR_ENGINE_PREPARATION_REQUIRED,
                reason_code=descriptor.reason_code or REASON_ENGINE_NOT_INSTALLED,
                engine=target.value,
                selectable_engines=self.registry.selectable_engine_ids(),
                detail={"required_component": descriptor.required_component},
            )
        if descriptor.availability is not EngineAvailability.READY:
            raise OcrEngineError(
                ErrorCode.OCR_ENGINE_UNAVAILABLE,
                reason_code=descriptor.reason_code or REASON_ENGINE_INIT_FAILED,
                engine=target.value,
                selectable_engines=self.registry.selectable_engine_ids(),
            )
        return adapter

    def validate(self, engine: OcrEngine | None) -> None:
        """提交期校验：解析成功返回，失败抛 OcrEngineError。"""
        self.resolve(engine)

    def _descriptor_for(self, adapter: Any) -> EngineDescriptor:
        engine_id = adapter.engine_id
        cached = self._probe_cache.get(engine_id)
        if cached is not None:
            return cached
        descriptor = adapter.descriptor()
        self._probe_cache[engine_id] = descriptor
        return descriptor

    def invalidate_probe_cache(self, engine_id: OcrEngine | None = None) -> None:
        """component 安装/修复后使探针缓存失效，下一次解析重新探测。"""
        if engine_id is None:
            self._probe_cache.clear()
        else:
            self._probe_cache.pop(engine_id, None)


@dataclass
class LazyEngineHandle:
    """惰性引擎句柄：descriptor 走独立探针，首次使用才构建引擎实例。

    Paddle 这类重依赖的导入成本必须推迟到真正使用（与现有
    ``PaddleExecutor.adapter_factory`` 语义一致）；descriptor 探针只做
    importability 检查，不触发模型加载。
    """

    engine_id: OcrEngine
    descriptor_probe: Callable[[], EngineDescriptor]
    factory: Callable[[], Any]
    _instance: Any = field(default=None, init=False, repr=False)
    _instance_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def descriptor(self) -> EngineDescriptor:
        return self.descriptor_probe()

    def instance(self) -> Any:
        if self._instance is not None:
            return self._instance
        with self._instance_lock:
            if self._instance is None:
                self._instance = self.factory()
        return self._instance

    def capabilities(self, options: Any | None = None) -> Any:
        return self.instance().capabilities(options)

    def recognize_many(
        self,
        items: list[Any],
        *,
        options: Any | None = None,
        compute_batch: Any | None = None,
    ) -> list[dict[str, Any]]:
        return self.instance().recognize_many(
            items, options=options, compute_batch=compute_batch
        )

    def preload(self, pipelines: tuple[str, ...]) -> Any:
        return self.instance().preload(pipelines)

    def residency_status(self) -> Any:
        if self._instance is None:
            from vibeocr.runtime_contracts import ResidencyStatus

            return ResidencyStatus()
        return self._instance.residency_status()

    def release_idle(self, pipeline: str | None = None) -> Any:
        if self._instance is None:
            from vibeocr.runtime_contracts import ResidencyStatus

            return ResidencyStatus()
        return self._instance.release_idle(pipeline)

    def configure_settings(self, snapshot: Any) -> None:
        if self._instance is not None:
            self._instance.configure_settings(snapshot)

    def close(self) -> None:
        if self._instance is not None:
            self._instance.close()


__all__ = [
    "DEFAULT_OCR_ENGINE",
    "EngineAvailability",
    "EngineDescriptor",
    "GeneralTextOcrEngine",
    "LazyEngineHandle",
    "OCR_PIPELINE_ID",
    "OcrEngineError",
    "OcrEngineRegistry",
    "OcrEngineResolver",
    "REASON_ENGINE_INIT_FAILED",
    "REASON_ENGINE_LANGUAGE_UNAVAILABLE",
    "REASON_ENGINE_NOT_INSTALLED",
    "ensure_engine_valid_for_pipeline",
    "parse_wire_engine",
]
