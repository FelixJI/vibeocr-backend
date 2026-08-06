"""Backend GPU memory status interface tests."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from vibeocr.backend.utils.gpu_memory_monitor import (
    GPU_BATCH_CAP,
    GPU_FALLBACK_BATCH_SIZE,
    GPUMemoryMonitor,
    estimate_gpu_batch_size,
)


def _install_fake_pynvml(monkeypatch) -> ModuleType:
    module = ModuleType("pynvml")
    module.nvmlInit = Mock()
    module.nvmlShutdown = Mock()
    module.nvmlDeviceGetHandleByIndex = Mock(return_value="gpu-handle")
    module.nvmlDeviceGetMemoryInfo = Mock(
        return_value=SimpleNamespace(
            total=8 * 1024 * 1024 * 1024,
            free=6 * 1024 * 1024 * 1024,
            used=2 * 1024 * 1024 * 1024,
        )
    )
    monkeypatch.setitem(sys.modules, "pynvml", module)
    return module


def test_get_status_is_unavailable_when_nvml_initialization_fails(monkeypatch) -> None:
    pynvml = _install_fake_pynvml(monkeypatch)
    pynvml.nvmlInit.side_effect = RuntimeError("NVML unavailable")

    status = GPUMemoryMonitor().get_status()

    assert status.available is False
    assert (status.total, status.free, status.used) == (0, 0, 0)


def test_get_status_reports_selected_device_memory_in_megabytes(monkeypatch) -> None:
    pynvml = _install_fake_pynvml(monkeypatch)

    status = GPUMemoryMonitor(device_id=2).get_status()

    pynvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(2)
    assert status.available is True
    assert (status.total, status.free, status.used) == (8192, 6144, 2048)


def test_get_status_is_unavailable_when_nvml_query_fails(monkeypatch) -> None:
    pynvml = _install_fake_pynvml(monkeypatch)
    pynvml.nvmlDeviceGetHandleByIndex.side_effect = RuntimeError("GPU disappeared")

    status = GPUMemoryMonitor().get_status()

    assert status.available is False
    assert (status.total, status.free, status.used) == (0, 0, 0)


def test_context_manager_releases_nvml(monkeypatch) -> None:
    pynvml = _install_fake_pynvml(monkeypatch)

    with GPUMemoryMonitor() as monitor:
        assert monitor.is_available() is True

    pynvml.nvmlShutdown.assert_called_once_with()


def test_close_suppresses_nvml_shutdown_failure(monkeypatch) -> None:
    pynvml = _install_fake_pynvml(monkeypatch)
    pynvml.nvmlShutdown.side_effect = RuntimeError("driver stopped")

    GPUMemoryMonitor().close()

    pynvml.nvmlShutdown.assert_called_once_with()


@pytest.mark.parametrize(
    ("free_mb", "avg_pixels", "expected"),
    [
        (24_576, 1_000_000, GPU_BATCH_CAP),
        (64, 10_000_000, 1),
        (0, 1_000_000, GPU_FALLBACK_BATCH_SIZE),
        (8_192, 0, 1),
    ],
)
def test_estimate_gpu_batch_size_respects_cap_floor_and_fallback(
    free_mb: int, avg_pixels: int, expected: int
) -> None:
    assert estimate_gpu_batch_size(free_mb, avg_pixels) == expected
