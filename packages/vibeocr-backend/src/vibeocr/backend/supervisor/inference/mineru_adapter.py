"""MinerUProcessAdapter: supervisor-owned MinerU API subprocess.

Plan §4 Phase 5 goals addressed by this seam:

* Supervisor owns the MinerU API subprocess lifecycle (start/health/stop).
* ``recognize_many`` issues ONE budgeted multi-file native parse job
  with unique internal stems; results map back to stable input order.
* Default backend does NOT promise cross-document compute batching — the
  capability reports ``real_batch=False`` so metrics can distinguish HTTP
  batching from compute batching.
* Idle release stops the API subprocess (disk models are not deleted).
* Cancel: cooperative stop of subsequent chunks; exclusive MinerU jobs may
  escalate to hard termination after a grace period.

The actual HTTP call to mineru-api and result parsing reuse
``services/mineru_service.py``; this adapter is the ownership seam and the
unique-stem mapper.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from vibeocr.backend.utils.mime_types import extension_to_mime, mime_to_extension
from vibeocr.runtime_contracts import ErrorCode, EvictionReason, PipelineSelection

from .budgets import AdapterCapability, InputItem

if TYPE_CHECKING:
    from collections.abc import Callable

    from vibeocr.runtime_contracts import ResidencyStatus

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
logger = logging.getLogger(__name__)


def unique_stem(original: str, index: int, content_type: str = "") -> str:
    """Build a unique, filesystem-safe stem for one input in a batch.

    Duplicate stems are disambiguated by an index + short random token so a
    single multi-file native parse job cannot collide on result keys.

    The stem must carry a MinerU-supported extension: mineru-api routes
    parsing (and PDF page ranges) by the upload filename. An extension the
    shared MIME map already recognises in ``original`` wins; otherwise the
    item's transport-declared ``content_type`` resolves one through the same
    map. There is deliberately no ``.pdf`` fallback here: without a trusted
    source we refuse the upload (fail closed) instead of guessing a type.
    """
    p = Path(original or "")
    name = p.stem
    ext = p.suffix
    if extension_to_mime(ext) is None:
        mime = content_type.split(";", 1)[0].strip().lower()
        resolved = mime_to_extension(mime) if mime else None
        if resolved is None:
            raise ValueError(
                f"unsupported file type for MinerU upload: no known extension "
                f"from display_name {original!r} or content_type {content_type!r}"
            )
        ext = resolved
    cleaned = _SAFE_RE.sub("_", name).strip("._-") or "input"
    cleaned = cleaned[:48]
    return f"{index:04d}-{cleaned}-{uuid.uuid4().hex[:6]}{ext}"


class _MinerUClientLike(Protocol):
    """Minimal slice of the MinerU API client we depend on."""

    def file_parse(self, files: list[tuple[str, bytes]], **kwargs: Any) -> Any: ...


class _MinerULifecycle(Protocol):
    """Owns the mineru-api subprocess start/stop.

    ``MinerUService`` is a singleton whose ``__init__`` blocks until the API is
    up, so in production the lifecycle just ensures the singleton exists (start)
    and tears it down (stop). Tests inject a no-op lifecycle or leave it None.
    """

    def start(self) -> None: ...

    def stop(self) -> None: ...


@dataclass
class MinerUProcessAdapter:
    """Supervisor-owned MinerU adapter."""

    client_factory: Callable[[], _MinerUClientLike]
    backend: str = "hybrid-engine"
    # Default does NOT promise cross-document compute batching.
    capability: AdapterCapability = field(
        default_factory=lambda: AdapterCapability(
            name="MinerU", real_batch=False, max_compute_batch=1
        )
    )
    # Optional real subprocess lifecycle. When None the adapter falls back to a
    # flag-flip (kept for unit tests that inject a fake client_factory without a
    # backing process). Production wiring injects a lifecycle over MinerUService.
    lifecycle: _MinerULifecycle | None = None
    _process_started: bool = False
    _process_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _settings: Any = None
    _active_leases: int = 0
    _last_used: float = field(default_factory=time.monotonic)
    _eviction_reason: EvictionReason = EvictionReason.NONE
    _watch_stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _watch_wakeup: threading.Event = field(default_factory=threading.Event, repr=False)
    _watch_thread: threading.Thread | None = field(default=None, repr=False)

    def capabilities(self, options: Any | None = None) -> AdapterCapability:
        del options
        return self.capability

    # ------------------------------------------------------------------
    # Process lifecycle (ownership seam)
    # ------------------------------------------------------------------

    def ensure_started(self) -> None:
        with self._process_lock:
            if self._remote_locked():
                # 远程连接由自部署服务器承担解析：本地不 spawn/拥有
                # mineru-api 子进程，也不承担进程驻留/TTL 生命周期。
                return
            if self._process_started:
                return
            # When a real lifecycle is injected, drive the mineru-api subprocess
            # through it; otherwise this is a no-op flag for test fakes.
            if self.lifecycle is not None:
                self.lifecycle.start()
            self._process_started = True
            self._last_used = time.monotonic()
            self._eviction_reason = EvictionReason.NONE
            self._ensure_watcher_locked()

    @staticmethod
    def _remote_locked() -> bool:
        """当前连接是否为远程（调用方必须已持有 ``_process_lock``）."""
        from vibeocr.backend.services.mineru_readiness import (
            CONNECTION_REMOTE,
            current_connection,
        )

        return current_connection().mode == CONNECTION_REMOTE

    def stop(self, *, reason: Any = None) -> None:
        """Stop the MinerU API subprocess; disk models are NOT deleted."""
        with self._process_lock:
            self._stop_locked(reason=reason)

    def _stop_locked(self, *, reason: Any = None) -> None:
        if self.lifecycle is not None and self._process_started:
            try:
                self.lifecycle.stop()
            except Exception:  # pragma: no cover - defensive
                pass
        if self._process_started and reason is not None:
            self._eviction_reason = reason
        self._process_started = False

    def _ensure_watcher_locked(self) -> None:
        thread = self._watch_thread
        if thread is not None and thread.is_alive():
            return
        self._watch_stop.clear()
        self._watch_thread = threading.Thread(
            target=self._watch_residency,
            name="vibeocr-mineru-residency",
            daemon=True,
        )
        self._watch_thread.start()

    def _policy_locked(self) -> tuple[bool, int]:
        default_ttl = int(getattr(self._settings, "default_ttl_seconds", 300))
        # MinerU 是 child-process keep-alive，不是可 pin 的模型驻留资源。
        pinned = False
        ttl = default_ttl
        for spec in getattr(self._settings, "pipelines", ()):
            if spec.name != "MinerU":
                continue
            if spec.ttl_seconds is not None:
                ttl = int(spec.ttl_seconds)
            break
        return pinned, max(0, ttl)

    def _watch_residency(self) -> None:
        while not self._watch_stop.is_set():
            self._watch_wakeup.wait(timeout=0.25)
            self._watch_wakeup.clear()
            with self._process_lock:
                pinned, ttl = self._policy_locked()
                if (
                    self._process_started
                    and self._active_leases == 0
                    and not pinned
                    and ttl > 0
                    and time.monotonic() - self._last_used >= ttl
                ):
                    self._stop_locked(reason=EvictionReason.TTL_EXPIRED)

    # ------------------------------------------------------------------
    # recognize_many — budgeted multi-file request
    # ------------------------------------------------------------------

    def _service_options(self, options: Any | None) -> Any | None:
        """Convert a wire ``PipelineSelection`` into backend ``OCROptions``.

        Same boundary conversion as the paddle adapter: the wire DTO stays
        contracts-owned and strictly validated, while the Backend package
        owns the option semantics (``lang_list``, ``backend``, page ranges,
        formula/table toggles, ...).
        """
        if not isinstance(options, PipelineSelection):
            return options
        from vibeocr.backend.services.mineru_config import resolve_mineru_config

        return resolve_mineru_config(options)

    def recognize_many(
        self,
        items: list[InputItem],
        *,
        options: Any | None = None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> list[dict[str, Any]]:
        if not items:
            return []
        started = time.perf_counter()
        lease_acquired = False
        try:
            service_options = self._service_options(options)
            self.ensure_started()
            with self._process_lock:
                self._active_leases += 1
                lease_acquired = True
            client = self.client_factory()
            # Build unique-stem file list preserving input order.
            files: list[tuple[str, bytes]] = []
            stem_to_index: dict[str, int] = {}
            for idx, item in enumerate(items):
                raw = getattr(item, "data", b"")
                if not isinstance(raw, (bytes, bytearray)) or len(raw) == 0:
                    raise ValueError(
                        f"InputItem {item.item_id} has no raw bytes for MinerU upload"
                    )
                display = getattr(item, "display_name", None) or f"input-{idx}"
                content_type = getattr(item, "content_type", "") or ""
                stem = unique_stem(display, idx, content_type=content_type)
                stem_to_index[stem] = idx
                stem_to_index[Path(stem).stem] = idx
                files.append((stem, bytes(raw)))
            # A request-provided backend (user-selected pipeline option) wins
            # over the adapter default: file_parse's ``backend`` kwarg would
            # override options.backend, so pass None when the request already
            # carries one.
            backend_override = (
                None if getattr(service_options, "backend", None) else self.backend
            )
            raw_results = client.file_parse(
                files,
                options=service_options,
                backend=backend_override,
                cancelled=cancelled,
            )
            payloads = self._map_results_back(raw_results, stem_to_index, len(items))
        except Exception:
            logger.exception(
                "[Supervisor][Recognize] action=recognize pipeline=MinerU "
                "items=%d result=failed elapsed_ms=%.1f",
                len(items),
                (time.perf_counter() - started) * 1000,
            )
            raise
        else:
            logger.info(
                "[Supervisor][Recognize] action=recognize pipeline=MinerU "
                "items=%d result=success elapsed_ms=%.1f",
                len(items),
                (time.perf_counter() - started) * 1000,
            )
            return payloads
        finally:
            if lease_acquired:
                with self._process_lock:
                    self._active_leases = max(0, self._active_leases - 1)
                    self._last_used = time.monotonic()
                self._watch_wakeup.set()

    @staticmethod
    def _map_results_back(
        raw_results: Any, stem_to_index: dict[str, int], expected: int
    ) -> list[dict[str, Any]]:
        """Restore stable input order from a MinerU multi-file result dict.

        MinerU returns results keyed by filename; we map each stem back to its
        original index. Missing stems yield an empty dict (the caller decides
        whether to mark the item failed).
        """
        out: list[dict[str, Any]] = [{} for _ in range(expected)]
        if isinstance(raw_results, dict):
            for key, value in raw_results.items():
                idx = stem_to_index.get(key)
                if idx is not None:
                    payload = value if isinstance(value, dict) else {"raw": value}
                    out[idx] = payload
        elif isinstance(raw_results, list):
            # If the API returned a list in the same order, use positional mapping.
            for i, value in enumerate(raw_results):
                if i < expected:
                    out[i] = value if isinstance(value, dict) else {"raw": value}
        return out

    # ------------------------------------------------------------------
    # Residency passthrough (idle release stops the subprocess)
    # ------------------------------------------------------------------

    def preload(self, pipelines: tuple[str, ...]) -> ResidencyStatus:
        if "MinerU" not in pipelines:
            return self.residency_status()
        started = time.perf_counter()
        try:
            self.ensure_started()
            with self._process_lock:
                self._active_leases += 1
            try:
                prepare = getattr(self.client_factory(), "prepare", None)
                if prepare is not None:
                    prepare()
            finally:
                with self._process_lock:
                    self._active_leases -= 1
                    self._last_used = time.monotonic()
            status = self.residency_status()
        except Exception:
            logger.exception(
                "[Supervisor][Preload] action=preload pipeline=MinerU "
                "result=failed elapsed_ms=%.1f",
                (time.perf_counter() - started) * 1000,
            )
            raise
        logger.info(
            "[Supervisor][Preload] action=preload pipeline=MinerU "
            "result=success elapsed_ms=%.1f",
            (time.perf_counter() - started) * 1000,
        )
        return status

    def residency_status(self) -> ResidencyStatus:
        from vibeocr.runtime_contracts import ResidencyEntry, ResidencyKind
        from vibeocr.runtime_contracts import ResidencyStatus as _RS

        # ``lifecycle.start()`` may spend a long time booting MinerU while it
        # owns this lock. Status is observational and must remain bounded, so
        # fall back to a best-effort snapshot instead of waiting indefinitely.
        acquired = self._process_lock.acquire(timeout=0.05)
        try:
            default_ttl = int(getattr(self._settings, "default_ttl_seconds", 300))
            pinned, ttl = self._policy_locked()
            # 远程连接不宣称本地进程驻留：local 会话遗留的子进程（等待 TTL
            # 回收）不得显示为当前 MinerU 资源的本地/远程驻留；本地清理路径
            # （TTL/release/close）照常回收它，远程服务器永不受影响。
            started = self._process_started and not self._remote_locked()
            kind = (
                ResidencyKind.PINNED
                if started and pinned
                else (ResidencyKind.SOFT_TTL if started else ResidencyKind.EVICTED)
            )
            remaining = (
                max(0, int(ttl - (time.monotonic() - self._last_used)))
                if started and not pinned and ttl > 0
                else None
            )
            active_leases = self._active_leases
            eviction_reason = self._eviction_reason
        finally:
            if acquired:
                self._process_lock.release()
        return _RS(
            default_ttl_seconds=default_ttl,
            entries=(
                ResidencyEntry(
                    pipeline="MinerU",
                    kind=kind,
                    active_leases=active_leases,
                    remaining_ttl_seconds=remaining,
                    eviction_reason=eviction_reason,
                ),
            ),
            pipelines=tuple(getattr(self._settings, "pipelines", ())),
        )

    def release_idle(self, pipeline: str | None = None) -> ResidencyStatus:
        with self._process_lock:
            pinned, _ttl = self._policy_locked()
            if (
                not pinned
                and self._active_leases == 0
                and (pipeline is None or pipeline == "MinerU")
            ):
                self._stop_locked(reason=EvictionReason.EXPLICIT_RELEASE)
        return self.residency_status()

    def configure_settings(self, snapshot: Any) -> None:
        from vibeocr.backend.services.mineru_config import MineruConfigError
        from vibeocr.backend.services.mineru_readiness import (
            configure_connection,
            connection_from_extra,
            current_connection,
        )

        # 非法连接配置 fail closed；合法变更在无活动 MinerU 任务时原子应用。
        connection = connection_from_extra(getattr(snapshot, "extra", None))
        with self._process_lock:
            if connection != current_connection() and self._active_leases > 0:
                # 在途解析固定原端点：活动任务期间拒绝改连接，避免串用/
                # 旧任务 mark_executed 污染新连接的 tier 观察集。
                raise MineruConfigError(
                    ErrorCode.VALIDATION_ERROR,
                    "mineru_connection_change_blocked_by_active_tasks",
                )
            configure_connection(connection)
            self._settings = snapshot
            self._ensure_watcher_locked()
        self._watch_wakeup.set()

    def close(self) -> None:
        self._watch_stop.set()
        self._watch_wakeup.set()
        thread = self._watch_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self.stop(reason=EvictionReason.SUPERVISOR_SHUTDOWN)


__all__ = ["MinerUProcessAdapter", "unique_stem"]
