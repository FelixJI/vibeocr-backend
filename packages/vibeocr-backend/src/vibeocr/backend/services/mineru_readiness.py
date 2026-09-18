"""MinerU tier readiness: installed dependencies and observed execution differ."""

from __future__ import annotations

import threading
from importlib import metadata

from vibeocr.runtime_contracts import ErrorCode, MineruTier

from .mineru_config import LANGUAGES, MineruConfigError

_lock = threading.Lock()
_ready: set[MineruTier] = set()


def mark_executed(tier: MineruTier) -> None:
    with _lock:
        _ready.add(tier)


def mark_failed(tier: MineruTier) -> None:
    with _lock:
        _ready.discard(tier)


def catalog_payload(*, observe_runtime: bool = True) -> dict:
    try:
        installed = metadata.version("mineru").split(".")[0] == "4"
    except metadata.PackageNotFoundError:
        installed = False
    with _lock:
        ready = set(_ready) if observe_runtime else set()
    return {
        "default_tier": "basic",
        "languages": list(LANGUAGES),
        "tiers": [
            {
                "id": tier.value,
                "availability": "ready"
                if installed and tier in ready
                else "preparation_required",
                "reason_code": None
                if installed and tier in ready
                else (
                    "mineru_execution_preparation_required"
                    if not observe_runtime
                    else "mineru_dependencies_required"
                    if not installed
                    else "mineru_execution_preparation_required"
                ),
            }
            for tier in MineruTier
        ],
    }


def require_ready(tier: MineruTier) -> None:
    descriptor = next(
        item for item in catalog_payload()["tiers"] if item["id"] == tier.value
    )
    if descriptor["availability"] != "ready":
        raise MineruConfigError(
            ErrorCode.MINERU_TIER_PREPARATION_REQUIRED, descriptor["reason_code"]
        )
