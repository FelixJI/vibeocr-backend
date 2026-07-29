from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from scripts.build_spdx_sbom import build_spdx_sbom

if TYPE_CHECKING:
    from pathlib import Path


def _artifacts(root: Path, *, reverse: bool = False) -> Path:
    root.mkdir()
    entries = [
        ("vibeocr_backend-0.7.0-py3-none-any.whl", b"backend-wheel"),
        ("runtime-manifest.json", b'{"schema_version":1}\n'),
        ("SHA256SUMS", b"checksums\n"),
    ]
    for name, content in reversed(entries) if reverse else entries:
        (root / name).write_bytes(content)
    return root


def test_spdx_sbom_is_deterministic_sorted_and_complete(tmp_path: Path) -> None:
    first = build_spdx_sbom(
        artifacts_dir=_artifacts(tmp_path / "first"),
        repository_name="FelixJI/vibeocr-backend",
        version="0.7.0",
    )
    second = build_spdx_sbom(
        artifacts_dir=_artifacts(tmp_path / "second", reverse=True),
        repository_name="FelixJI/vibeocr-backend",
        version="0.7.0",
    )

    assert first.read_bytes() == second.read_bytes()
    value = json.loads(first.read_text(encoding="utf-8"))
    assert value["spdxVersion"] == "SPDX-2.3"
    assert value["dataLicense"] == "CC0-1.0"
    assert value["packages"][0]["name"] == "FelixJI/vibeocr-backend"
    assert value["packages"][0]["versionInfo"] == "0.7.0"

    files = value["files"]
    names = [record["fileName"] for record in files]
    assert names == sorted(names)
    assert "SBOM.spdx.json" not in names
    for record in files:
        artifact = first.parent / record["fileName"]
        assert record["checksums"] == [
            {
                "algorithm": "SHA256",
                "checksumValue": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        ]
        assert record["comment"] == f"size={artifact.stat().st_size}"


def test_rebuild_excludes_existing_sbom_and_tracks_content_changes(
    tmp_path: Path,
) -> None:
    root = _artifacts(tmp_path / "artifacts")
    first = build_spdx_sbom(
        artifacts_dir=root,
        repository_name="vibeocr-protocol",
        version="2.0.0",
    )
    first_bytes = first.read_bytes()
    first_value = json.loads(first.read_text(encoding="utf-8"))

    rebuilt = build_spdx_sbom(
        artifacts_dir=root,
        repository_name="vibeocr-protocol",
        version="2.0.0",
    )
    assert rebuilt.read_bytes() == first_bytes
    assert all(
        record["fileName"] != "SBOM.spdx.json"
        for record in json.loads(rebuilt.read_text(encoding="utf-8"))["files"]
    )

    (root / "runtime-manifest.json").write_bytes(b"changed")
    changed = build_spdx_sbom(
        artifacts_dir=root,
        repository_name="vibeocr-protocol",
        version="2.0.0",
    )
    changed_value = json.loads(changed.read_text(encoding="utf-8"))
    assert changed_value["documentNamespace"] != first_value["documentNamespace"]


@pytest.mark.parametrize(
    ("repository_name", "version", "message"),
    [
        ("", "1.0.0", "repository_name"),
        ("vibeocr-backend", "", "version"),
    ],
)
def test_spdx_sbom_requires_repository_identity(
    tmp_path: Path,
    repository_name: str,
    version: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_spdx_sbom(
            artifacts_dir=_artifacts(tmp_path / "artifacts"),
            repository_name=repository_name,
            version=version,
        )


def test_spdx_sbom_rejects_empty_artifacts_directory(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    with pytest.raises(ValueError, match="at least one"):
        build_spdx_sbom(
            artifacts_dir=artifacts,
            repository_name="vibeocr-backend",
            version="0.7.0",
        )
