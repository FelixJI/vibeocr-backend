from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from vibeocr.backend import network_detector as network_detector_module
from vibeocr.backend.network_detector import NetworkDetector
from vibeocr.backend.runtime_state import CACHE_VERSION


def test_legacy_model_source_cache_is_invalidated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_cache = {
        "version": CACHE_VERSION,
        "machine_id": "machine",
        "network": {
            "last_detected": datetime.now().isoformat(),
            "paddlex_source": "bos",
            "mineru_source": "modelscope",
        },
    }
    monkeypatch.setattr(
        network_detector_module,
        "load_cache",
        lambda _root: legacy_cache,
    )
    monkeypatch.setattr(
        network_detector_module,
        "generate_machine_id",
        lambda: "machine",
    )
    detected: list[Path] = []

    def detect(detector: NetworkDetector) -> None:
        detected.append(detector._project_root)  # noqa: SLF001
        detector._network_type = "international"  # noqa: SLF001

    monkeypatch.setattr(NetworkDetector, "_detect", detect)

    detector = NetworkDetector(tmp_path)

    assert detected == [tmp_path]
    assert detector.network_type == "international"


def test_network_type_cache_round_trips_without_model_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored: dict[str, object] = {
        "version": CACHE_VERSION,
        "machine_id": "machine",
    }
    monkeypatch.setattr(
        network_detector_module,
        "load_cache",
        lambda _root: dict(stored),
    )
    monkeypatch.setattr(
        network_detector_module,
        "save_cache",
        lambda _root, payload: stored.update(payload),
    )
    monkeypatch.setattr(
        network_detector_module,
        "generate_machine_id",
        lambda: "machine",
    )

    writer = object.__new__(NetworkDetector)
    writer._project_root = tmp_path  # noqa: SLF001
    writer._network_type = "international"  # noqa: SLF001
    writer._save_to_cache()  # noqa: SLF001

    network = stored["network"]
    assert isinstance(network, dict)
    assert set(network) == {"last_detected", "network_type"}
    assert network["network_type"] == "international"

    monkeypatch.setattr(
        NetworkDetector,
        "_detect",
        lambda _detector: pytest.fail("fresh network cache should be reused"),
    )
    assert NetworkDetector(tmp_path).network_type == "international"
