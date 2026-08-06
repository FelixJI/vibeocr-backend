from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pytest

from scripts.build_runtime_installer import (
    build_runtime_installer,
    package_runtime_installer,
)


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
        assert archive.namelist() == ["runtime-installer/vibeocr-runtime-installer.exe"]
        assert archive.read(archive.namelist()[0]) == b"installer"


def test_runtime_installer_collects_protocol_modules_and_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert cwd.is_dir()
        assert check is True
        dist = Path(command[command.index("--distpath") + 1])
        (dist / "vibeocr-runtime-installer.exe").write_bytes(b"installer")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    build_runtime_installer(
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        backend_version="0.10.0",
    )

    command = commands[0]
    assert command[command.index("--collect-submodules") + 1] == (
        "vibeocr.runtime_contracts"
    )
    assert command[command.index("--collect-data") + 1] == ("vibeocr.runtime_contracts")
