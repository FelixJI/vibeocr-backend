from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.verify_release_assets import ReleaseAssetError, verify_release_assets


def _write_candidate(root: Path, assets: dict[str, bytes]) -> None:
    for name, content in assets.items():
        (root / name).write_bytes(content)
    lines = [
        f"{hashlib.sha256(content).hexdigest()}  {name}"
        for name, content in sorted(assets.items())
    ]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_verify_release_assets_accepts_complete_candidate(tmp_path: Path) -> None:
    _write_candidate(
        tmp_path,
        {
            "SBOM.spdx.json": b"{}",
            "component-lock.json": b"{}",
            "VibeOCR-v1.0.0-win64.zip": b"zip",
        },
    )

    names = verify_release_assets(
        tmp_path,
        required=("SBOM.spdx.json", "component-lock.json"),
        require_one=("VibeOCR-v*-win64.zip",),
    )

    assert set(names) == {
        "SHA256SUMS",
        "SBOM.spdx.json",
        "VibeOCR-v1.0.0-win64.zip",
        "component-lock.json",
    }


def test_verify_release_assets_rejects_tampering(tmp_path: Path) -> None:
    _write_candidate(tmp_path, {"artifact.zip": b"original"})
    (tmp_path / "artifact.zip").write_bytes(b"tampered")

    with pytest.raises(ReleaseAssetError, match="SHA-256 mismatch"):
        verify_release_assets(tmp_path)


def test_verify_release_assets_rejects_unindexed_file(tmp_path: Path) -> None:
    _write_candidate(tmp_path, {"artifact.zip": b"zip"})
    (tmp_path / "unexpected.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(ReleaseAssetError, match="file set mismatch"):
        verify_release_assets(tmp_path)


def test_verify_release_assets_requires_exactly_one_pattern_match(
    tmp_path: Path,
) -> None:
    _write_candidate(
        tmp_path,
        {
            "first.zip": b"first",
            "second.zip": b"second",
        },
    )

    with pytest.raises(ReleaseAssetError, match="exactly one"):
        verify_release_assets(tmp_path, require_one=("*.zip",))


def test_verify_release_assets_rejects_unsafe_index_name(tmp_path: Path) -> None:
    digest = hashlib.sha256(b"escape").hexdigest()
    (tmp_path / "SHA256SUMS").write_text(
        f"{digest}  ../escape.zip\n",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseAssetError, match="unsafe asset name"):
        verify_release_assets(tmp_path)
