from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.resolve_protocol_binding import resolve_protocol_binding

if TYPE_CHECKING:
    from pathlib import Path


def _write_package(path: Path, requirement: str) -> Path:
    path.write_text(
        "[project]\n"
        'name = "vibeocr-backend"\n'
        'version = "0.10.1"\n'
        f'dependencies = ["{requirement}"]\n',
        encoding="utf-8",
    )
    return path


def _write_lock(path: Path, version: str = "2.3.0") -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repository": "FelixJI/vibeocr-protocol",
                "version": version,
                "artifacts": {
                    f"vibeocr_runtime_contracts-{version}-py3-none-any.whl": {
                        "sha256": "a" * 64,
                        "size": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_resolves_exact_lock_inside_package_major_range(tmp_path: Path) -> None:
    lock = _write_lock(tmp_path / "protocol.lock.json")
    package = _write_package(
        tmp_path / "pyproject.toml",
        "vibeocr-runtime-contracts>=2.3.0,<3.0.0",
    )

    assert resolve_protocol_binding(lock, package) == "2.3.0"


@pytest.mark.parametrize(
    "requirement",
    [
        "vibeocr-runtime-contracts==2.3.0",
        "vibeocr-runtime-contracts>=2.4.0,<3.0.0",
        "vibeocr-runtime-contracts>=2.3.0,<4.0.0",
    ],
)
def test_rejects_requirement_that_does_not_cover_exact_lock_and_major(
    tmp_path: Path,
    requirement: str,
) -> None:
    lock = _write_lock(tmp_path / "protocol.lock.json")
    package = _write_package(tmp_path / "pyproject.toml", requirement)

    with pytest.raises(ValueError):
        resolve_protocol_binding(lock, package)


def test_rejects_lock_without_exact_contracts_wheel(tmp_path: Path) -> None:
    lock = _write_lock(tmp_path / "protocol.lock.json")
    value = json.loads(lock.read_text(encoding="utf-8"))
    value["artifacts"] = {}
    lock.write_text(json.dumps(value), encoding="utf-8")
    package = _write_package(
        tmp_path / "pyproject.toml",
        "vibeocr-runtime-contracts>=2.3.0,<3.0.0",
    )

    with pytest.raises(ValueError, match="exactly one contracts wheel"):
        resolve_protocol_binding(lock, package)
