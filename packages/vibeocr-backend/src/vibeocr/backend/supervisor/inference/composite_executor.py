"""CompositeExecutor: route a job to a child executor by ``JobKind``.

The :class:`~vibeocr.backend.supervisor.module.SupervisorModule` takes a single
``Executor``, but the supervisor must run more than one backend (Paddle for
``RECOGNITION`` jobs, MinerU for ``MINERU_PARSE`` jobs). This executor wraps a
list of children and dispatches each job to the first child whose
``handles(record)`` returns True.

It also aggregates ``residency_status`` / ``release_idle`` across children so
the v2 runtime endpoints reflect every loaded pipeline, and picks the
strongest cancel mode offered by any matching child (a queued-only child can
still be cancelled cooperatively if another child handles the kind).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vibeocr.runtime_contracts import CancelMode, ResidencyStatus, SettingsSnapshot

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vibeocr.runtime_contracts import JobKind

    from ..module import Executor


class _Routed:
    """A child executor plus the job kinds it accepts."""

    def __init__(self, executor: Executor, kinds: frozenset[JobKind]) -> None:
        self.executor = executor
        self.kinds = kinds


class CompositeExecutor:
    """Route jobs to children by ``record.kind``; aggregate runtime ops."""

    def __init__(self, children: Iterable[tuple[Executor, frozenset[JobKind]]]) -> None:
        self._children: list[_Routed] = [_Routed(ex, frozenset(ks)) for ex, ks in children]
        # Remember the last executor we dispatched to per job_id so cancel_mode_for
        # and residency/release hit the right backend without re-inspecting kind.
        self._dispatch: dict[str, _Routed] = {}

    def _routed_for(self, record: Any) -> _Routed | None:
        kind = getattr(record, "kind", None)
        for child in self._children:
            if kind in child.kinds:
                return child
        return None

    # ------------------------------------------------------------------
    # Executor protocol
    # ------------------------------------------------------------------

    def execute(self, record: Any, staged: Any) -> None:
        routed = self._routed_for(record)
        if routed is None:
            # No child handles this kind: fail the job with a typed reason
            # rather than silently leaving it non-terminal.
            from vibeocr.runtime_contracts import TERMINAL_JOB_STATES, JobState

            if record.state not in TERMINAL_JOB_STATES:
                try:
                    record.transition(JobState.FAILED)
                    record.append_event(
                        "no_backend_for_kind",
                        detail={"kind": str(getattr(record, "kind", None))},
                    )
                except Exception:  # pragma: no cover - defensive
                    pass
            return
        self._dispatch[getattr(record, "job_id", "")] = routed
        routed.executor.execute(record, staged)

    def cancel_mode_for(self, record: Any) -> CancelMode:
        routed = self._dispatch.get(getattr(record, "job_id", "")) or self._routed_for(record)
        if routed is not None:
            return routed.executor.cancel_mode_for(record)
        return CancelMode.COOPERATIVE

    def residency_status(self) -> ResidencyStatus:
        # Merge each child's entries into one status. We take the first child's
        # default_ttl_seconds as the representative default (children share the
        # same configured TTL in practice); entries are unioned.
        default_ttl = 300
        entries: list[Any] = []
        pipelines: dict[str, Any] = {}
        vram_total: int | None = None
        vram_used: int | None = None
        for child in self._children:
            try:
                status = child.executor.residency_status()
            except Exception:  # pragma: no cover - defensive
                continue
            try:
                default_ttl = int(getattr(status, "default_ttl_seconds", default_ttl))
            except (TypeError, ValueError):
                pass
            child_entries = getattr(status, "entries", ()) or ()
            entries.extend(child_entries)
            for spec in getattr(status, "pipelines", ()) or ():
                pipelines[spec.name] = spec
            child_total = getattr(status, "vram_total_mb", None)
            child_used = getattr(status, "vram_used_mb", None)
            if child_total is not None:
                vram_total = max(vram_total or 0, int(child_total))
            if child_used is not None:
                vram_used = max(vram_used or 0, int(child_used))
        try:
            return ResidencyStatus(
                default_ttl_seconds=default_ttl,
                entries=tuple(entries),
                pipelines=tuple(pipelines.values()),
                vram_total_mb=vram_total,
                vram_used_mb=vram_used,
            )
        except TypeError:
            return ResidencyStatus()

    def release_idle(self, pipeline: str | None = None) -> ResidencyStatus:
        for child in self._children:
            try:
                child.executor.release_idle(pipeline)
            except Exception:  # pragma: no cover - defensive
                pass
        return self.residency_status()

    def preload(self, pipelines: tuple[str, ...]) -> ResidencyStatus:
        for child in self._children:
            child.executor.preload(pipelines)
        return self.residency_status()

    def configure_settings(self, snapshot: SettingsSnapshot) -> ResidencyStatus:
        for child in self._children:
            child.executor.configure_settings(snapshot)
        return self.residency_status()

    def close(self) -> None:
        for child in self._children:
            try:
                child.executor.close()
            except Exception:  # pragma: no cover - defensive
                pass
        self._dispatch.clear()


__all__ = ["CompositeExecutor"]
