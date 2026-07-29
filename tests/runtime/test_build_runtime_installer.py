from __future__ import annotations

import zipfile
from typing import TYPE_CHECKING

from scripts.build_runtime_installer import package_runtime_installer

if TYPE_CHECKING:
    from pathlib import Path


def test_runtime_installer_zip_is_deterministic_and_has_expected_layout(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "vibeocr-runtime-installer.exe"
    executable.write_bytes(b"installer")
    first = package_runtime_installer(
        executable,
        tmp_path / "first",
        backend_version="0.7.0",
    )
    second = package_runtime_installer(
        executable,
        tmp_path / "second",
        backend_version="0.7.0",
    )
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == [
            "runtime-installer/vibeocr-runtime-installer.exe"
        ]
        assert archive.read(archive.namelist()[0]) == b"installer"
