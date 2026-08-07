from __future__ import annotations

import pytest
import vibeocr.backend.utils.system_memory as system_memory
from vibeocr.backend.utils.system_memory import (
    CPU_BATCH_CAP,
    FALLBACK_RAM_MB,
    estimate_cpu_batch_size,
    get_available_ram_mb,
)


def test_get_available_ram_returns_positive_probe_result(monkeypatch) -> None:
    monkeypatch.setattr(system_memory, "_read_available_ram", lambda: 4096.9)

    assert get_available_ram_mb() == 4096


@pytest.mark.parametrize("probe_result", [None, 0, -1])
def test_get_available_ram_falls_back_for_unavailable_probe(
    monkeypatch, probe_result: int | None
) -> None:
    monkeypatch.setattr(
        system_memory,
        "_read_available_ram",
        lambda: probe_result,
    )

    assert get_available_ram_mb() == FALLBACK_RAM_MB


def test_get_available_ram_falls_back_when_probe_raises(monkeypatch) -> None:
    def fail_probe() -> int:
        raise OSError("memory probe failed")

    monkeypatch.setattr(system_memory, "_read_available_ram", fail_probe)

    assert get_available_ram_mb() == FALLBACK_RAM_MB


@pytest.mark.parametrize(
    ("free_mb", "avg_pixels", "expected"),
    [
        (16_384, 1_000_000, CPU_BATCH_CAP),
        (512, 12_000_000, 1),
        (4_096, 10_000_000, 5),
        (0, 10_000_000, 1),
        (4_096, 0, 1),
    ],
)
def test_estimate_cpu_batch_size_respects_cap_floor_and_invalid_inputs(
    free_mb: int, avg_pixels: int, expected: int
) -> None:
    assert estimate_cpu_batch_size(free_mb, avg_pixels) == expected
