from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from scripts.build_release_checksums import (
    build_release_checksums,
    write_sidecar_checksum,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_checksums_are_sorted_and_exclude_the_index(tmp_path: Path) -> None:
    (tmp_path / "z.bin").write_bytes(b"z")
    (tmp_path / "a.bin").write_bytes(b"a")
    (tmp_path / "SHA256SUMS").write_text("stale", encoding="utf-8")

    output = build_release_checksums(tmp_path)

    assert output.read_text(encoding="utf-8").splitlines() == [
        f"{hashlib.sha256(b'a').hexdigest()}  a.bin",
        f"{hashlib.sha256(b'z').hexdigest()}  z.bin",
    ]


def test_checksums_reject_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        build_release_checksums(tmp_path)


def test_writes_updater_compatible_sidecar(tmp_path: Path) -> None:
    artifact = tmp_path / "VibeOCR.zip"
    artifact.write_bytes(b"zip")
    sidecar = write_sidecar_checksum(artifact)
    assert sidecar.read_text(encoding="utf-8") == (
        f"{hashlib.sha256(b'zip').hexdigest()}  VibeOCR.zip\n"
    )
