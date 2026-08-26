from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from vibeocr.backend.runtime_manifest import (
    ManifestError,
    load_runtime_manifest,
    validate_requirements_lock,
)

from scripts.build_runtime_manifest import build_runtime_manifest


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
        "cu126_gpu_lock": root / "requirements-win-x64-cu126-gpu.lock",
        "python_archive": root / "cpython-3.13.12-win_amd64.tar.gz",
        "installer_archive": root / "vibeocr-runtime-installer-v0.7.0-win-x64.zip",
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
    values["cu126_gpu_lock"].write_text(_lock("win-x64-cu126"), encoding="utf-8")
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


def test_gpu_scope_input_is_source_neutral_and_excludes_document_parsing() -> None:
    content = (
        Path(__file__).parents[2]
        / "packages/vibeocr-backend/runtime-profiles/win-x64-cu126-gpu/requirements.in"
    ).read_text(encoding="utf-8")
    requirements = [
        line for line in content.splitlines() if line and not line.startswith("#")
    ]

    assert "uv 0.12.5" in content
    assert (
        "--no-config --no-sources --default-index "
        "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/"
    ) in content
    assert "--emit-index-url" not in content
    assert requirements == [
        "-r ../win-x64-base/requirements.in",
        (
            "paddlepaddle-gpu @ https://paddle-whl.bj.bcebos.com/stable/cu126/"
            "paddlepaddle-gpu/paddlepaddle_gpu-3.3.1-cp313-cp313-win_amd64.whl"
        ),
        (
            "torch @ https://download.pytorch.org/whl/cu126/"
            "torch-2.12.1%2Bcu126-cp313-cp313-win_amd64.whl"
        ),
        (
            "torchvision @ https://download.pytorch.org/whl/cu126/"
            "torchvision-0.27.1%2Bcu126-cp313-cp313-win_amd64.whl"
        ),
    ]


@pytest.mark.parametrize(
    ("directory", "profile"),
    [
        ("win-x64-base", "win-x64-base"),
        ("win-x64-cpu", "win-x64-cpu"),
        ("win-x64-cu126", "win-x64-cu126"),
        ("win-x64-cu126-gpu", "win-x64-cu126"),
    ],
)
def test_committed_runtime_locks_are_source_neutral(
    directory: str, profile: str
) -> None:
    lock = (
        Path(__file__).parents[2]
        / "packages/vibeocr-backend/runtime-profiles"
        / directory
        / f"requirements-{directory}.lock"
    )

    validate_requirements_lock(lock, profile=profile)


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
        "ocr.engine-selection.v1",
        "ocr.recognition-modes.v1",
        "runtime.download-sources.v1",
        "runtime.component-selection.v1",
    }
    assert [
        component.component_id
        for component in manifest.profiles["win-x64-cpu"].components
    ] == [
        "rapidocr-base",
        "paddleocr-cpu",
        "mineru-cpu",
        "pdf_document_tools",
        "image_code_tools",
        "runtime_host",
    ]
    assert manifest.profiles["win-x64-cpu"].components[0].version is None
    cpu_profile = manifest.profiles["win-x64-cpu"]
    assert [scope.scope_id for scope in cpu_profile.scopes] == ["default"]
    assert cpu_profile.scopes[0].component_ids == tuple(
        component.component_id for component in cpu_profile.components
    )
    assert cpu_profile.scopes[0].lock_path == cpu_profile.lock_path

    cuda_components = {
        component.component_id: component
        for component in manifest.profiles["win-x64-cu126"].components
    }
    assert cuda_components["paddleocr-cuda"].dependencies == ("gpu_runtime",)
    assert cuda_components["mineru-cuda"].dependencies == ("gpu_runtime",)
    assert cuda_components["gpu_runtime"].dependencies == ()
    assert "dependencies" not in cuda_components["mineru-cuda"].to_payload()

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


def test_build_emits_cuda_gpu_runtime_install_scope(tmp_path: Path) -> None:
    manifest_path = _build(tmp_path / "input", tmp_path / "output")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    cuda = raw["profiles"]["win-x64-cu126"]

    assert cuda["install_scopes"] == [
        {
            "scope_id": "gpu-runtime",
            "component_ids": [
                "rapidocr-base",
                "pdf_document_tools",
                "image_code_tools",
                "runtime_host",
                "gpu_runtime",
            ],
            "lock": "requirements-win-x64-cu126-gpu.lock",
            "runtime_pack": None,
            "sha256": hashlib.sha256(
                (
                    manifest_path.parent / "requirements-win-x64-cu126-gpu.lock"
                ).read_bytes()
            ).hexdigest(),
        }
    ]
    profile = load_runtime_manifest(manifest_path).profiles["win-x64-cu126"]
    assert [scope.scope_id for scope in profile.scopes] == [
        "default",
        "gpu-runtime",
    ]
    checksum_names = {
        line.split("  ", 1)[1]
        for line in (manifest_path.parent / "SHA256SUMS").read_text().splitlines()
    }
    assert "requirements-win-x64-cu126-gpu.lock" in checksum_names


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
        runtime_packs={"win-x64-base": [pack]},
    )
    manifest = load_runtime_manifest(manifest_path)
    base = manifest.profiles["win-x64-base"]
    assert [c.component_id for c in base.components] == [
        "rapidocr-base",
        "pdf_document_tools",
        "image_code_tools",
        "runtime_host",
    ]
    # RapidOCR 是 Base Runtime 固有、可探针修复的必备 component。
    versions = {c.component_id: c.version for c in base.components}
    assert versions["rapidocr-base"] == "3.9.2"
    assert versions["image_code_tools"] == "5.0.0.93"
    assert base.runtime_pack == (pack.name,)
    assert base.runtime_pack_sha256 == (hashlib.sha256(pack.read_bytes()).hexdigest(),)
    # 篡改输出目录中绑定的 pack 副本后 loader fail closed。
    (manifest_path.parent / pack.name).write_bytes(b"tampered")
    from vibeocr.backend.runtime_manifest import ManifestError

    with pytest.raises(ManifestError, match="runtime pack SHA-256 mismatch"):
        load_runtime_manifest(manifest_path)


def test_loader_rejects_legacy_string_pack(tmp_path: Path) -> None:
    """runtime_pack 必须是分片文件名列表:旧单字符串形态 fail closed。"""
    manifest_path = _build(tmp_path / "input", tmp_path / "output")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["profiles"]["win-x64-base"]["runtime_pack"] = "pack.zip"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    from vibeocr.backend.runtime_manifest import ManifestError

    with pytest.raises(ManifestError, match="must be a non-empty filename list"):
        load_runtime_manifest(manifest_path)


def test_loader_rejects_mismatched_pack_sha_length(tmp_path: Path) -> None:
    manifest_path = _build(tmp_path / "input", tmp_path / "output")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["profiles"]["win-x64-base"]["runtime_pack"] = ["p1.zip", "p2.zip"]
    raw["profiles"]["win-x64-base"]["runtime_pack_sha256"] = ["0" * 64]
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    from vibeocr.backend.runtime_manifest import ManifestError

    with pytest.raises(ManifestError, match="parallel runtime_pack"):
        load_runtime_manifest(manifest_path)


def test_loader_rejects_pack_sha_without_pack(tmp_path: Path) -> None:
    manifest_path = _build(tmp_path / "input", tmp_path / "output")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["profiles"]["win-x64-base"]["runtime_pack_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    from vibeocr.backend.runtime_manifest import ManifestError

    with pytest.raises(ManifestError, match="runtime_pack_sha256 requires"):
        load_runtime_manifest(manifest_path)


def test_loader_keeps_old_manifest_without_install_scopes_readable(
    tmp_path: Path,
) -> None:
    manifest_path = _build(tmp_path / "input", tmp_path / "output")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["profiles"]["win-x64-cu126"].pop("install_scopes")
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    profile = load_runtime_manifest(manifest_path).profiles["win-x64-cu126"]

    assert [scope.scope_id for scope in profile.scopes] == ["default"]
    assert profile.scopes[0].component_ids == tuple(
        component.component_id for component in profile.components
    )


def test_loader_parses_additional_install_scope(tmp_path: Path) -> None:
    manifest_path = _build(tmp_path / "input", tmp_path / "output")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    cuda = raw["profiles"]["win-x64-cu126"]
    cuda["install_scopes"] = [
        {
            "scope_id": "gpu-runtime",
            "component_ids": [
                "rapidocr-base",
                "pdf_document_tools",
                "image_code_tools",
                "runtime_host",
                "gpu_runtime",
            ],
            "lock": cuda["lock"],
            "sha256": cuda["sha256"],
            "runtime_pack": None,
        }
    ]
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    profile = load_runtime_manifest(manifest_path).profiles["win-x64-cu126"]

    assert [scope.scope_id for scope in profile.scopes] == [
        "default",
        "gpu-runtime",
    ]
    gpu_scope = profile.scopes[1]
    assert gpu_scope.component_ids[-1] == "gpu_runtime"
    assert gpu_scope.lock_path.name == cuda["lock"]


def _cuda_scope(raw: dict) -> dict:
    cuda = raw["profiles"]["win-x64-cu126"]
    return {
        "scope_id": "gpu-runtime",
        "component_ids": [
            "rapidocr-base",
            "pdf_document_tools",
            "image_code_tools",
            "runtime_host",
            "gpu_runtime",
        ],
        "lock": cuda["lock"],
        "sha256": cuda["sha256"],
        "runtime_pack": None,
    }


def test_loader_rejects_install_scope_without_base_components(tmp_path: Path) -> None:
    manifest_path = _build(tmp_path / "input", tmp_path / "output")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    scope = _cuda_scope(raw)
    scope["component_ids"].remove("runtime_host")
    raw["profiles"]["win-x64-cu126"]["install_scopes"] = [scope]
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ManifestError, match="must include base components"):
        load_runtime_manifest(manifest_path)


def test_loader_rejects_install_scope_that_is_not_dependency_closure(
    tmp_path: Path,
) -> None:
    manifest_path = _build(tmp_path / "input", tmp_path / "output")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    scope = _cuda_scope(raw)
    scope["component_ids"].remove("gpu_runtime")
    scope["component_ids"].append("paddleocr-cuda")
    raw["profiles"]["win-x64-cu126"]["install_scopes"] = [scope]
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ManifestError, match="must be a dependency closure"):
        load_runtime_manifest(manifest_path)


def test_loader_rejects_circular_component_dependencies(tmp_path: Path) -> None:
    manifest_path = _build(tmp_path / "input", tmp_path / "output")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    components = raw["profiles"]["win-x64-cu126"]["components"]
    by_id = {component["component_id"]: component for component in components}
    by_id["paddleocr-cuda"]["dependencies"] = ["gpu_runtime"]
    by_id["gpu_runtime"]["dependencies"] = ["paddleocr-cuda"]
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ManifestError, match="must be acyclic"):
        load_runtime_manifest(manifest_path)


@pytest.mark.parametrize(
    "dependencies",
    [
        ["gpu_runtime", "gpu_runtime"],
        ["not-in-profile"],
    ],
)
def test_loader_rejects_invalid_component_dependencies(
    tmp_path: Path,
    dependencies: list[str],
) -> None:
    manifest_path = _build(tmp_path / "input", tmp_path / "output")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    components = raw["profiles"]["win-x64-cu126"]["components"]
    next(
        component
        for component in components
        if component["component_id"] == "paddleocr-cuda"
    )["dependencies"] = dependencies
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ManifestError, match="components"):
        load_runtime_manifest(manifest_path)


def test_loader_rejects_duplicate_install_scope_component_set(tmp_path: Path) -> None:
    manifest_path = _build(tmp_path / "input", tmp_path / "output")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    cuda = raw["profiles"]["win-x64-cu126"]
    cuda["install_scopes"] = [
        {
            **_cuda_scope(raw),
            "component_ids": [
                component["component_id"] for component in cuda["components"]
            ],
        }
    ]
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ManifestError, match="component sets must be unique"):
        load_runtime_manifest(manifest_path)


def test_loader_rejects_install_scope_lock_sha_mismatch(tmp_path: Path) -> None:
    manifest_path = _build(tmp_path / "input", tmp_path / "output")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    scope = _cuda_scope(raw)
    scope["sha256"] = "0" * 64
    raw["profiles"]["win-x64-cu126"]["install_scopes"] = [scope]
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ManifestError, match="lock SHA-256 mismatch"):
        load_runtime_manifest(manifest_path)
