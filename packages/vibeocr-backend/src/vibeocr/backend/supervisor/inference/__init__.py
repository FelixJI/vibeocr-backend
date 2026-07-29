"""Inference adapters and scheduling.

The production composition combines scheduling, budget and recovery policy
with the physical Paddle cache and MinerU process residency owners. There is
no transport-level shadow residency model. Phase 2 only needs the
:class:`~vibeocr.backend.supervisor.module.Executor` seam (defined in module.py).
"""

from __future__ import annotations
