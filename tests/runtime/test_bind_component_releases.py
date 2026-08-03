from __future__ import annotations

import hashlib
import json
import zipfile
from typing import TYPE_CHECKING

import pytest

from scripts.bind_component_releases import (
    bind_product_releases,
    bind_protocol_release,
)

if TYPE_CHECKING:
    from pathlib import Path


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _protocol_release(root: Path, *, schema_version: int = 1) -> Path:
    root.mkdir()
    wheel = root / "vibeocr_runtime_contracts-2.0.0-py3-none-any.whl"
    wheel.write_bytes(b"protocol")
    manifest: dict[str, object] = {
        "artifacts": {
            wheel.name: {"sha256": _sha(b"protocol"), "size": len(b"protocol")}
        },
    }
    if schema_version == 1:
        manifest.update({"schema_version": 1, "protocol_version": "2.0.0"})
    else:
        manifest.update(
            {
                "schema_version": 2,
                "project": {"component": "protocol"},
                "protocol": {"version": "2.0.0"},
                "release": {"tag": "v2.0.0", "version": "2.0.0"},
            }
        )
    (root / "release-manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return root


def _backend_release(root: Path, protocol: Path) -> Path:
    root.mkdir()
    wheel = root / "vibeocr_backend-0.7.0-py3-none-any.whl"
    wheel.write_bytes(b"backend")
    protocol_wheel_source = next(protocol.glob("*.whl"))
    protocol_wheel = root / protocol_wheel_source.name
    protocol_wheel.write_bytes(protocol_wheel_source.read_bytes())
    protocol_manifest = root / "protocol-release-manifest.json"
    protocol_manifest.write_bytes((protocol / "release-manifest.json").read_bytes())
    python_archive = root / "python.tar.gz"
    python_archive.write_bytes(b"python")
    cpu_lock = root / "cpu.lock"
    cpu_lock.write_bytes(b"cpu")
    installer_archive = root / "installer.zip"
    with zipfile.ZipFile(installer_archive, "w") as archive:
        archive.writestr("runtime-installer/installer.exe", b"installer")
    manifest = {
        "backend_version": "0.7.0",
        "backend_wheel": wheel.name,
        "backend_sha256": _sha(b"backend"),
        "protocol_manifest": protocol_manifest.name,
        "protocol_manifest_sha256": _sha(protocol_manifest.read_bytes()),
        "protocol_wheel": protocol_wheel.name,
        "protocol_sha256": _sha(protocol_wheel.read_bytes()),
        "python": {
            "archive": python_archive.name,
            "sha256": _sha(b"python"),
        },
        "installer": {
            "archive": installer_archive.name,
            "sha256": _sha(installer_archive.read_bytes()),
            "executable_path": "runtime-installer/installer.exe",
            "executable_sha256": _sha(b"installer"),
        },
        "capabilities": ["ocr.recognition.v2", "pdf.edit.v2"],
        "profiles": {
            "win-x64-cpu": {
                "lock": cpu_lock.name,
                "sha256": _sha(b"cpu"),
            }
        },
    }
    manifest_path = root / "runtime-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{_sha(path.read_bytes())}  {path.name}\n"
            for path in sorted(root.iterdir(), key=lambda item: item.name)
            if path.is_file() and path.name != "SHA256SUMS"
        ),
        encoding="utf-8",
    )
    return root


def test_binds_verified_protocol_manifest(tmp_path: Path) -> None:
    release = _protocol_release(tmp_path / "protocol")
    output = bind_protocol_release(
        release_dir=release,
        repository="FelixJI/vibeocr-protocol",
        version="2.0.0",
        output=tmp_path / "protocol.lock.json",
    )
    lock = json.loads(output.read_text(encoding="utf-8"))
    assert lock["manifest_sha256"] == _sha(
        (release / "release-manifest.json").read_bytes()
    )


def test_binds_verified_protocol_v2_manifest(tmp_path: Path) -> None:
    release = _protocol_release(tmp_path / "protocol", schema_version=2)
    output = bind_protocol_release(
        release_dir=release,
        repository="FelixJI/vibeocr-protocol",
        version="2.0.0",
        output=tmp_path / "protocol.lock.json",
    )
    assert json.loads(output.read_text(encoding="utf-8"))["version"] == "2.0.0"


def test_rejects_inconsistent_protocol_v2_release_identity(tmp_path: Path) -> None:
    release = _protocol_release(tmp_path / "protocol", schema_version=2)
    manifest_path = release / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release"]["tag"] = "v2.0.1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="release identity mismatch"):
        bind_protocol_release(
            release_dir=release,
            repository="FelixJI/vibeocr-protocol",
            version="2.0.0",
            output=tmp_path / "protocol.lock.json",
        )


def test_rejects_tampered_protocol_asset(tmp_path: Path) -> None:
    release = _protocol_release(tmp_path / "protocol")
    next(release.glob("*.whl")).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        bind_protocol_release(
            release_dir=release,
            repository="FelixJI/vibeocr-protocol",
            version="2.0.0",
            output=tmp_path / "protocol.lock.json",
        )


def test_binds_product_to_exact_backend_and_protocol(tmp_path: Path) -> None:
    protocol = _protocol_release(tmp_path / "protocol")
    backend = _backend_release(tmp_path / "backend", protocol)
    output = bind_product_releases(
        protocol_release_dir=protocol,
        backend_release_dir=backend,
        protocol_repository="FelixJI/vibeocr-protocol",
        protocol_version="2.0.0",
        backend_repository="FelixJI/vibeocr-backend",
        backend_version="0.7.0",
        profile="win-x64-cpu",
        required_capabilities=("ocr.recognition.v2",),
        output=tmp_path / "component-lock.json",
    )
    lock = json.loads(output.read_text(encoding="utf-8"))
    assert lock["backend"]["artifact_sha256"] == _sha(b"backend")
    assert lock["required_capabilities"] == ["ocr.recognition.v2"]


def test_product_lock_rejects_missing_capability(tmp_path: Path) -> None:
    protocol = _protocol_release(tmp_path / "protocol")
    backend = _backend_release(tmp_path / "backend", protocol)
    with pytest.raises(ValueError, match="missing required capabilities"):
        bind_product_releases(
            protocol_release_dir=protocol,
            backend_release_dir=backend,
            protocol_repository="FelixJI/vibeocr-protocol",
            protocol_version="2.0.0",
            backend_repository="FelixJI/vibeocr-backend",
            backend_version="0.7.0",
            profile="win-x64-cpu",
            required_capabilities=("qrcode.v2",),
            output=tmp_path / "component-lock.json",
        )


def test_product_lock_rejects_tampered_runtime_closure(tmp_path: Path) -> None:
    protocol = _protocol_release(tmp_path / "protocol")
    backend = _backend_release(tmp_path / "backend", protocol)
    (backend / "python.tar.gz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Python archive hash mismatch"):
        bind_product_releases(
            protocol_release_dir=protocol,
            backend_release_dir=backend,
            protocol_repository="FelixJI/vibeocr-protocol",
            protocol_version="2.0.0",
            backend_repository="FelixJI/vibeocr-backend",
            backend_version="0.7.0",
            profile="win-x64-cpu",
            required_capabilities=("ocr.recognition.v2",),
            output=tmp_path / "component-lock.json",
        )
