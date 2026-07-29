"""PaddlePipelineAdapter: the unified ``recognize_many`` seam for Paddle.

Plan §4 Phase 4 goals:

* All Paddle pipelines enter the scheduler through one adapter.
* Single image is a one-element batch (no separate single implementation).
* The capability reports the *honest* real-batch support: only the OCR
  pipeline registers a true ``recognize_batch``; PP-StructureV3 / VL /
  formula / table currently fall back to per-item loops and must not be
  reported as real batch.
* Stable OCRResult / TextBlock / parsing logic is reused unchanged.

This adapter is deliberately thin: it injects the existing
:class:`OCRService`/registry and never re-implements result parsing.
"""

from __future__ import annotations

import contextlib
import io
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from vibeocr.runtime_contracts import (
    PipelineSelection,
    ResidencyEntry,
    ResidencyKind,
    ResidencyStatus,
    SettingsSnapshot,
)

from .budgets import AdapterCapability, ComputeBatch, InputItem

logger = logging.getLogger(__name__)


class _OCRServiceLike(Protocol):
    """Minimal slice of :class:`OCRService` we depend on."""

    def recognize_batch(self, images: list[Any], options: Any | None = ...) -> list[Any]: ...

    def preload_pipelines_sequential(self, pipelines: list[Any]) -> dict[str, bool]: ...


@dataclass
class PaddlePipelineAdapter:
    """Wraps an OCRService-like object behind ``recognize_many``."""

    service: _OCRServiceLike
    pipeline_name: str = "OCR"
    # Capability is cached per semantic pipeline, never globally.
    _capabilities: dict[str, AdapterCapability] = field(default_factory=dict)
    _settings: SettingsSnapshot = field(default_factory=SettingsSnapshot)

    # ------------------------------------------------------------------
    # Capability
    # ------------------------------------------------------------------

    def capabilities(
        self, options: PipelineSelection | None = None
    ) -> AdapterCapability:
        pipeline_name = (
            options.pipeline_id
            if isinstance(options, PipelineSelection)
            else self.pipeline_name
        )
        cached = self._capabilities.get(pipeline_name)
        if cached is not None:
            return cached
        real_batch = self._pipeline_supports_real_batch(pipeline_name)
        max_compute = 8 if real_batch else 1
        capability = AdapterCapability(
            name=pipeline_name,
            real_batch=real_batch,
            max_compute_batch=max_compute,
        )
        self._capabilities[pipeline_name] = capability
        return capability

    def _pipeline_supports_real_batch(self, pipeline_name: str) -> bool:
        """Return True only if the registry registers a real batch adapter."""
        try:
            from vibeocr.backend.core.pipelines import get_registry  # type: ignore

            registry = get_registry()
            spec = registry.get(pipeline_name) if registry.has(pipeline_name) else None
            return bool(spec is not None and getattr(spec, "recognize_batch", None) is not None)
        except Exception:
            # In test environments without the pipeline registry we report
            # conservatively: not a real batch.
            return False

    # ------------------------------------------------------------------
    # recognize_many — the unified entry point
    # ------------------------------------------------------------------

    def recognize_many(
        self,
        items: list[InputItem],
        *,
        options: Any | None = None,
        compute_batch: ComputeBatch | None = None,
    ) -> list[dict[str, Any]]:
        """Recognise a list of staged inputs.

        ``items`` must carry raw image bytes in ``encoded_bytes``-style
        payloads (we read bytes via the item's ``data`` attribute if present,
        else assume the caller pre-decoded). Returns one result dict per
        input, in identical order. Single-image callers pass a one-element
        list; there is no separate single path.
        """
        if not items:
            return []
        pipeline_name = (
            options.pipeline_id
            if isinstance(options, PipelineSelection)
            else self.pipeline_name
        )
        started = time.perf_counter()
        try:
            images = [self._to_ndarray(self._raw_bytes(it)) for it in items]
            service_options = self._service_options(options)
            manager = self._physical_cache_manager()
            lease = (
                manager.lease(pipeline_name)
                if manager is not None and hasattr(manager, "lease")
                else contextlib.nullcontext()
            )
            with lease:
                results = self.service.recognize_batch(images, service_options)
            payloads = [self._result_to_payload(r) for r in results]
        except Exception:
            logger.exception(
                "[Supervisor][Recognize] action=recognize pipeline=%s "
                "items=%d result=failed elapsed_ms=%.1f",
                pipeline_name,
                len(items),
                (time.perf_counter() - started) * 1000,
            )
            raise
        logger.info(
            "[Supervisor][Recognize] action=recognize pipeline=%s "
            "items=%d result=success elapsed_ms=%.1f",
            pipeline_name,
            len(items),
            (time.perf_counter() - started) * 1000,
        )
        return payloads

    def _service_options(self, options: Any | None) -> Any | None:
        if not isinstance(options, PipelineSelection):
            return options
        # Temporary backend boundary conversion while the stable pipeline
        # option implementation now lives in the Backend package. The wire
        # DTO itself remains contracts-owned and strictly validated.
        from vibeocr.backend.models.ocr_options import OCROptions

        return OCROptions.from_dict(
            {"pipeline": options.pipeline_id, **options.options}
        )

    # ------------------------------------------------------------------
    # Residency passthrough
    # ------------------------------------------------------------------

    def preload(self, pipelines: tuple[str, ...]) -> ResidencyStatus:
        """顺序加载本 adapter 拥有的 Paddle 管道并返回真实驻留快照。"""
        from vibeocr.runtime_contracts.contracts.pipelines import (
            OCRPipeline,
            get_paddle_pipelines,
        )

        paddle_pipelines = set(get_paddle_pipelines())
        selected = [
            pipeline
            for name in pipelines
            if (pipeline := OCRPipeline(name)) in paddle_pipelines
        ]
        if selected:
            started = time.perf_counter()
            try:
                results = self.service.preload_pipelines_sequential(selected)
            except Exception:
                elapsed_ms = (time.perf_counter() - started) * 1000
                for pipeline in selected:
                    logger.exception(
                        "[Supervisor][Preload] action=preload pipeline=%s "
                        "result=failed elapsed_ms=%.1f",
                        pipeline.value,
                        elapsed_ms,
                    )
                raise
            elapsed_ms = (time.perf_counter() - started) * 1000
            for pipeline in selected:
                succeeded = bool(results.get(pipeline.value, False))
                log = logger.info if succeeded else logger.error
                log(
                    "[Supervisor][Preload] action=preload pipeline=%s "
                    "result=%s elapsed_ms=%.1f",
                    pipeline.value,
                    "success" if succeeded else "failed",
                    elapsed_ms,
                )
            failed = [
                pipeline.value
                for pipeline in selected
                if not results.get(pipeline.value, False)
            ]
            if failed:
                raise RuntimeError(f"管道预加载失败: {', '.join(failed)}")
        return self.residency_status()

    def residency_status(self) -> ResidencyStatus:
        manager = self._physical_cache_manager()
        if manager is None or not hasattr(manager, "status"):
            return ResidencyStatus(
                default_ttl_seconds=self._settings.default_ttl_seconds,
                pipelines=self._settings.pipelines,
            )
        raw = manager.status()
        loaded = raw.get("loaded_pipelines", [])
        ttls = raw.get("pipeline_ttls", {})
        active = raw.get("active_counts", {})
        pinned = set(raw.get("pinned_pipelines", []))
        last_used = raw.get("last_used_unix_ms", {})
        now_ms = int(time.time() * 1000)
        entries = []
        for name in loaded:
            ttl = int(ttls.get(name, self._settings.default_ttl_seconds))
            remaining = None
            if name not in pinned and ttl > 0:
                used_ms = int(last_used.get(name, now_ms))
                remaining = max(0, ttl - max(0, now_ms - used_ms) // 1000)
            entries.append(
                ResidencyEntry(
                    pipeline=str(name),
                    kind=(
                        ResidencyKind.PINNED
                        if name in pinned
                        else ResidencyKind.SOFT_TTL
                    ),
                    active_leases=int(active.get(name, 0)),
                    remaining_ttl_seconds=remaining,
                )
            )
        return ResidencyStatus(
            default_ttl_seconds=self._settings.default_ttl_seconds,
            entries=tuple(entries),
            pipelines=self._settings.pipelines,
        )

    def release_idle(self, pipeline: str | None = None) -> ResidencyStatus:
        manager = self._physical_cache_manager()
        if manager is not None:
            if pipeline is None:
                manager.release(heavy_only=False)
            else:
                manager.release_one(pipeline)
        return self.residency_status()

    def configure_settings(self, snapshot: SettingsSnapshot) -> None:
        self._settings = snapshot
        manager = self._physical_cache_manager()
        if manager is not None and hasattr(manager, "configure_residency"):
            manager.configure_residency(
                default_ttl_seconds=snapshot.default_ttl_seconds,
                pipelines=list(snapshot.pipelines),
            )

    def close(self) -> None:
        manager = self._physical_cache_manager()
        if manager is None:
            return
        manager.release(heavy_only=False, force=True)
        manager.shutdown()

    def _physical_cache_manager(self) -> Any | None:
        try:
            return self.service.cache_manager  # type: ignore[attr-defined]
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _raw_bytes(item: InputItem) -> bytes:
        data = getattr(item, "data", b"")
        if isinstance(data, (bytes, bytearray)) and len(data) > 0:
            return bytes(data)
        raise ValueError(
            f"InputItem {item.item_id} has no raw bytes; decode before calling recognize_many"
        )

    @staticmethod
    def _to_ndarray(raw: bytes) -> np.ndarray:
        from PIL import Image as PILImage

        img = PILImage.open(io.BytesIO(raw))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)

    @staticmethod
    def _result_to_payload(result: Any) -> dict[str, Any]:
        """Convert an OCRResult to a JSON-native payload.

        Delegates to :func:`vibeocr.backend.models.ocr_result_serializer.ocr_result_to_payload`,
        which is the single source of truth for the wire shape and produces the
        key set consumed by downstream callers (``text_blocks``/``preproc_angle``
        for PDF text-layer writeback, ``raw_text``/``markdown_text``/
        ``html_text``/``content_list`` for export). ``dict`` inputs pass through
        unchanged so test fakes keep working.
        """
        # Imported lazily: the serializer lives in Backend domain models,
        # which the backend depends on via wheel. Lazy import keeps this module
        # importable in minimal test environments that monkeypatch the adapter.
        from vibeocr.backend.models import ocr_result_to_payload

        return ocr_result_to_payload(result)


__all__ = ["PaddlePipelineAdapter"]
