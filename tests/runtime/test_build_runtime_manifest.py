from __future__ import annotations

import hashlib
import json
import zipfile
from typing import TYPE_CHECKING

import pytest
from vibeocr.backend.runtime_manifest import load_runtime_manifest

from scripts.build_runtime_manifest import build_runtime_manifest

if TYPE_CHECKING:
    from pathlib import Path


def _lock(profile: str) -> str:
    if profile == "win-x64-base":
        package = "\n".join(
            [
                f"rapidocr==3.9.2 \\\n    --hash=sha256:{'1' * 64}",
                f"onnxruntime==1.28.0 \\\n    --hash=sha256:{'1' * 64}",
                f"winrt-runtime==3.2.1 \\\n    --hash=sha256:{'1' * 64}",
                f"opencv-python==5.0.0.93 \\\n    --hash=sha256:{'1' * 64}",
            ]
        )
        return package + "\n"
    if profile == "win-x64-cpu":
        package = "paddlepaddle==3.3.1"
    else:
        package = (
            "paddlepaddle-gpu @ https://example.invalid/cu126/paddle.whl \\\n"
            f"    --hash=sha256:{'2' * 64}\n"
            "torch @ https://example.invalid/cu126/torch.whl \\\n"
            f"    --hash=sha256:{'3' * 64}\n"
            "torchvision @ https://example.invalid/cu126/torchvision.whl"
        )
    return f"{package} \\\n    --hash=sha256:{'1' * 64}\n"


def _inputs(root: Path) -> dict[str, Path]:
    root.mkdir()
    values = {
        "backend_wheel": root / "vibeocr_backend-0.7.0-py3-none-any.whl",
        "protocol_wheel": root / "vibeocr_runtime_contracts-2.0.0-py3-none-any.whl",
        "protocol_manifest": root / "release-manifest.json",
        "base_lock": root / "requirements-win-x64-base.lock",
        "cpu_lock": root / "requirements-win-x64-cpu.lock",
        "cu126_lock": root / "requirements-win-x64-cu126.lock",
        "python_archive": root / "cpython-3.13.12-win_amd64.tar.gz",
        "installer_archive": root / "vibeocr-runtime-installer-0.7.0.zip",
    }
    values["backend_wheel"].write_bytes(b"backend")
    values["protocol_wheel"].write_bytes(b"protocol")
    values["protocol_manifest"].write_text(
        json.dumps(
            {
                "schema_version": 2,
                "project": {"component": "protocol"},
                "protocol": {"version": "2.0.0"},
                "release": {"tag": "v2.0.0", "version": "2.0.0"},
                "artifacts": {
                    values["protocol_wheel"].name: {
                        "sha256": hashlib.sha256(b"protocol").hexdigest(),
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    values["base_lock"].write_text(_lock("win-x64-base"), encoding="utf-8")
    values["cpu_lock"].write_text(_lock("win-x64-cpu"), encoding="utf-8")
    values["cu126_lock"].write_text(_lock("win-x64-cu126"), encoding="utf-8")
    values["python_archive"].write_bytes(b"python-archive")
    with zipfile.ZipFile(values["installer_archive"], mode="w") as archive:
        archive.writestr(
            "runtime-installer/vibeocr-runtime-installer.exe",
            b"installer",
        )
    return values


def _build(root: Path, output: Path) -> Path:
    return build_runtime_manifest(
        **_inputs(root),
        backend_version="0.7.0",
        python_version="3.13.12",
        python_source_url=(
            "https://github.com/astral-sh/python-build-standalone/releases/"
            "download/20260325/"
            "cpython-3.13.12+20260325-x86_64-pc-windows-msvc"
            "-install_only.tar.gz"
        ),
        source_commit="a" * 40,
        build_workflow="tests/runtime-manifest",
        output_dir=output,
    )


def test_build_is_byte_deterministic_and_self_verifying(tmp_path: Path) -> None:
    first = _build(tmp_path / "first-input", tmp_path / "first-output")
    second = _build(tmp_path / "second-input", tmp_path / "second-output")
    assert first.read_bytes() == second.read_bytes()
    assert (first.parent / "SHA256SUMS").read_bytes() == (
        second.parent / "SHA256SUMS"
    ).read_bytes()
    manifest = load_runtime_manifest(first)
    assert manifest.protocol == ">=2.0.0,<3.0.0"
    assert manifest.protocol_version == "2.0.0"
    assert manifest.protocol_wheel.startswith("vibeocr_runtime_contracts-2.0.0-")
    assert set(manifest.capabilities) == {
        "export.document.v1",
        "ocr.recognition.v2",
        "pdf.edit.v2",
        "qrcode.v2",
        "runtime.settings.v2",
        "runtime.maintenance.v1",
        "runtime.maintenance.v2",
        "runtime.component-repair.v1",
        "runtime.capability-metadata.v1",
        "runtime.events.sse.v1",
        "runtime.events.ndjson.v1",
        "task.progress.v1",
    }
    assert [
        component.component_id
        for component in manifest.profiles["win-x64-cpu"].components
    ] == [
        "ocr_engine",
        "document_parsing",
        "pdf_document_tools",
        "image_code_tools",
        "runtime_host",
    ]
    assert manifest.profiles["win-x64-cpu"].components[0].version is None

    checksums = {}
    for line in (first.parent / "SHA256SUMS").read_text().splitlines():
        digest, filename = line.split("  ", 1)
        checksums[filename] = digest
    for filename, digest in checksums.items():
        assert (
            hashlib.sha256((first.parent / filename).read_bytes()).hexdigest() == digest
        )


def test_build_rejects_backend_wheel_version_mismatch(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "input")
    with pytest.raises(ValueError, match="does not match"):
        build_runtime_manifest(
            **inputs,
            backend_version="0.8.0",
            python_version="3.13.12",
            python_source_url=(
                "https://github.com/astral-sh/python-build-standalone/releases/"
                "download/20260325/"
                "cpython-3.13.12+20260325-x86_64-pc-windows-msvc"
                "-install_only.tar.gz"
            ),
            source_commit="a" * 40,
            build_workflow="tests",
            output_dir=tmp_path / "output",
        )


def test_manifest_json_has_no_self_hash(tmp_path: Path) -> None:
    manifest_path = _build(tmp_path / "input", tmp_path / "output")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "runtime_manifest_sha256" not in value


def test_build_binds_base_profile_and_runtime_pack(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "input")
    pack = tmp_path / "pack" / "vibeocr-runtime-pack-win-x64-base-0.7.0.zip"
    pack.parent.mkdir(parents=True)
    with zipfile.ZipFile(pack, mode="w") as archive:
        archive.writestr("rapidocr-3.9.2-py3-none-any.whl", b"rapidocr-wheel")
    manifest_path = build_runtime_manifest(
        **inputs,
        backend_version="0.7.0",
        python_version="3.13.12",
        python_source_url=(
            "https://github.com/astral-sh/python-build-standalone/releases/"
            "download/20260325/"
            "cpython-3.13.12+20260325-x86_64-pc-windows-msvc"
            "-install_only.tar.gz"
        ),
        source_commit="a" * 40,
        build_workflow="tests/runtime-manifest",
        output_dir=tmp_path / "output",
        base_runtime_pack=pack,
    )
    manifest = load_runtime_manifest(manifest_path)
    base = manifest.profiles["win-x64-base"]
    assert [c.component_id for c in base.components] == [
        "ocr_engine",
        "pdf_document_tools",
        "image_code_tools",
        "runtime_host",
    ]
    # base 的 ocr_engine 版本来自 rapidocr，image_code_tools 来自 opencv-python。
    versions = {c.component_id: c.version for c in base.components}
    assert versions["ocr_engine"] == "3.9.2"
    assert versions["image_code_tools"] == "5.0.0.93"
    assert base.runtime_pack == pack.name
    assert base.runtime_pack_sha256 == hashlib.sha256(pack.read_bytes()).hexdigest()
    # 篡改输出目录中绑定的 pack 副本后 loader fail closed。
    (manifest_path.parent / pack.name).write_bytes(b"tampered")
    from vibeocr.backend.runtime_manifest import ManifestError

    with pytest.raises(ManifestError, match="runtime pack SHA-256 mismatch"):
        load_runtime_manifest(manifest_path)


def test_loader_rejects_pack_sha_without_pack(tmp_path: Path) -> None:
    manifest_path = _build(tmp_path / "input", tmp_path / "output")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["profiles"]["win-x64-base"]["runtime_pack_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    from vibeocr.backend.runtime_manifest import ManifestError

    with pytest.raises(ManifestError, match="runtime_pack_sha256 requires"):
        load_runtime_manifest(manifest_path)
