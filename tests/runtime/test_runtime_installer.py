from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import zipfile
from pathlib import Path

import pytest
from vibeocr.backend.runtime_installer import (
    RuntimeInstaller,
    RuntimeInstallError,
    _extract_python_archive,
)
from vibeocr.backend.runtime_layout import (
    LayoutError,
    resolve_runtime_store,
    runtime_id_prefix,
)
from vibeocr.backend.runtime_lock import RuntimeLockTimeout, RuntimeStoreLock
from vibeocr.backend.runtime_manifest import ManifestError, load_runtime_manifest


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _lock_text(profile: str) -> str:
    base = "fastapi==1.0.0 \\\n    --hash=sha256:" + "1" * 64 + "\n"
    if profile == "win-x64-cpu":
        return base + ("paddlepaddle==3.3.1 \\\n    --hash=sha256:" + "2" * 64 + "\n")
    return base + (
        "paddlepaddle-gpu @ https://example.invalid/cu126/paddle.whl \\\n"
        "    --hash=sha256:" + "3" * 64 + "\n"
        "torch @ https://example.invalid/cu126/torch.whl \\\n"
        "    --hash=sha256:" + "4" * 64 + "\n"
        "torchvision @ https://example.invalid/cu126/torchvision.whl \\\n"
        "    --hash=sha256:" + "5" * 64 + "\n"
    )


def _release(root: Path) -> tuple[Path, Path]:
    root.mkdir()
    wheel = root / "vibeocr_backend-0.7.0-py3-none-any.whl"
    wheel.write_bytes(b"backend-wheel")
    protocol_wheel = root / "vibeocr_runtime_contracts-2.0.0-py3-none-any.whl"
    protocol_wheel.write_bytes(b"protocol-wheel")
    protocol_manifest = root / "protocol-release-manifest.json"
    protocol_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_version": "2.0.0",
                "artifacts": {
                    protocol_wheel.name: {
                        "sha256": _sha(protocol_wheel.read_bytes()),
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    python_archive = root / "cpython-3.13.12-win_amd64-install_only.tar.gz"
    python_archive.write_bytes(b"python-archive")
    installer_archive = root / "vibeocr-runtime-installer-0.7.0.zip"
    with zipfile.ZipFile(installer_archive, mode="w") as archive:
        archive.writestr(
            "runtime-installer/vibeocr-runtime-installer.exe",
            b"installer",
        )
    profiles = {}
    for profile in ("win-x64-cpu", "win-x64-cu126"):
        lock = root / f"requirements-{profile}.lock"
        lock.write_text(_lock_text(profile), encoding="utf-8")
        profiles[profile] = {
            "lock": lock.name,
            "sha256": _sha(lock.read_bytes()),
            "runtime_pack": None,
        }
    manifest = {
        "schema_version": 1,
        "backend_version": "0.7.0",
        "backend_wheel": wheel.name,
        "backend_sha256": _sha(wheel.read_bytes()),
        "protocol": ">=2.0.0,<3.0.0",
        "protocol_manifest": protocol_manifest.name,
        "protocol_manifest_sha256": _sha(protocol_manifest.read_bytes()),
        "protocol_wheel": protocol_wheel.name,
        "protocol_sha256": _sha(protocol_wheel.read_bytes()),
        "python": {
            "version": "3.13.12",
            "abi": "cp313",
            "platform": "win_amd64",
            "source_url": (
                "https://github.com/astral-sh/python-build-standalone/releases/"
                "download/20260325/"
                "cpython-3.13.12+20260325-x86_64-pc-windows-msvc"
                "-install_only.tar.gz"
            ),
            "archive": python_archive.name,
            "sha256": _sha(python_archive.read_bytes()),
        },
        "installer": {
            "archive": installer_archive.name,
            "sha256": _sha(installer_archive.read_bytes()),
            "executable_path": "runtime-installer/vibeocr-runtime-installer.exe",
            "executable_sha256": _sha(b"installer"),
        },
        "profiles": profiles,
        "capabilities": ["ocr.recognition.v2"],
        "source_commit": "0" * 40,
        "build_workflow": "tests/runtime",
    }
    manifest_path = root / "runtime-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    component = {
        "schema_version": 1,
        "protocol": {
            "repository": "FelixJI/vibeocr-protocol",
            "version": "2.0.0",
            "manifest_sha256": manifest["protocol_manifest_sha256"],
        },
        "backend": {
            "repository": "FelixJI/vibeocr-backend",
            "version": "0.7.0",
            "artifact_sha256": manifest["backend_sha256"],
            "runtime_manifest_sha256": _sha(manifest_path.read_bytes()),
            "profile": "win-x64-cpu",
        },
        "required_capabilities": ["ocr.recognition.v2"],
    }
    component_path = root / "component-lock.json"
    component_path.write_text(
        json.dumps(component, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, component_path


def _fake_install(partial: Path, _manifest, _profile: str) -> Path:
    python = partial / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    return python


def _tar_with(path: Path, member_name: str, content: bytes = b"python") -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        info = tarfile.TarInfo(member_name)
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))


def test_python_archive_extracts_only_stripped_python_root(tmp_path: Path) -> None:
    archive = tmp_path / "python.tar.gz"
    _tar_with(archive, "python/python.exe")
    destination = tmp_path / "runtime"
    destination.mkdir()
    _extract_python_archive(archive, destination)
    assert (destination / "python.exe").read_bytes() == b"python"


def test_python_archive_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "python.tar.gz"
    _tar_with(archive, "python/../escape.exe")
    destination = tmp_path / "runtime"
    destination.mkdir()
    with pytest.raises(RuntimeInstallError, match="unsafe"):
        _extract_python_archive(archive, destination)
    assert not (tmp_path / "escape.exe").exists()


def test_runtime_id_prefix_is_six_chars() -> None:
    # 物理目录名 / runtime_id 用 6 位前缀，完整哈希仅留在 lock 校验字段。
    digest = "a" * 64
    assert runtime_id_prefix(digest) == "aaaaaa"
    # 非法哈希（校验职责）应拒绝，与完整哈希的防篡改校验保持一致。
    with pytest.raises(LayoutError, match="manifest_sha256"):
        runtime_id_prefix("short")


def test_manifest_verifies_raw_hash_and_bound_artifacts(tmp_path: Path) -> None:
    manifest_path, _ = _release(tmp_path / "release")
    manifest = load_runtime_manifest(manifest_path)
    assert manifest.backend_version == "0.7.0"
    assert manifest.protocol_wheel == "vibeocr_runtime_contracts-2.0.0-py3-none-any.whl"
    assert manifest.sha256 == _sha(manifest_path.read_bytes())
    assert set(manifest.profiles) == {"win-x64-cpu", "win-x64-cu126"}


def test_manifest_rejects_tampered_lock(tmp_path: Path) -> None:
    manifest_path, _ = _release(tmp_path / "release")
    (manifest_path.parent / "requirements-win-x64-cpu.lock").write_text(
        "tampered",
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="SHA-256 mismatch"):
        load_runtime_manifest(manifest_path)


def test_manifest_rejects_tampered_protocol_wheel(tmp_path: Path) -> None:
    manifest_path, _ = _release(tmp_path / "release")
    (
        manifest_path.parent / "vibeocr_runtime_contracts-2.0.0-py3-none-any.whl"
    ).write_bytes(b"tampered")
    with pytest.raises(ManifestError, match="Protocol wheel SHA-256 mismatch"):
        load_runtime_manifest(manifest_path)


def test_manifest_rejects_tampered_protocol_manifest(tmp_path: Path) -> None:
    manifest_path, _ = _release(tmp_path / "release")
    (manifest_path.parent / "protocol-release-manifest.json").write_bytes(b"tampered")
    with pytest.raises(ManifestError, match="Protocol manifest SHA-256 mismatch"):
        load_runtime_manifest(manifest_path)


def test_manifest_rejects_tampered_python_archive(tmp_path: Path) -> None:
    manifest_path, _ = _release(tmp_path / "release")
    (
        manifest_path.parent / "cpython-3.13.12-win_amd64-install_only.tar.gz"
    ).write_bytes(b"tampered")
    with pytest.raises(ManifestError, match="Python archive SHA-256 mismatch"):
        load_runtime_manifest(manifest_path)


def test_shared_layout_requires_explicit_valid_registration(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    product = bundle / "classic"
    product.mkdir(parents=True)
    marker = bundle / "portable-layout.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shared_root": "shared",
                "products": {
                    "classic": {
                        "root": "classic",
                        "component_lock": "component-lock.json",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    digest = "a" * 64
    local = resolve_runtime_store(
        product,
        manifest_sha256=digest,
        profile="win-x64-cpu",
    )
    assert local.store_root == product.resolve()
    shared = resolve_runtime_store(
        product,
        manifest_sha256=digest,
        profile="win-x64-cpu",
        layout_manifest=marker,
        product_id="classic",
    )
    assert shared.store_root == (bundle / "shared").resolve()
    # 物理目录名用 6 位前缀（见 runtime_id_prefix），完整哈希仅留在 lock 校验。
    assert (
        shared.runtime_root
        == (bundle / "shared" / "runtimes" / digest[:6] / "win-x64-cpu").resolve()
    )


def test_shared_layout_rejects_traversal(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    product = bundle / "classic"
    product.mkdir(parents=True)
    marker = bundle / "portable-layout.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shared_root": "../escape",
                "products": {"classic": {"root": "classic"}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(LayoutError):
        resolve_runtime_store(
            product,
            manifest_sha256="a" * 64,
            profile="win-x64-cpu",
            layout_manifest=marker,
            product_id="classic",
        )


def test_ensure_is_atomic_and_idempotent(tmp_path: Path) -> None:
    manifest, component = _release(tmp_path / "release")
    calls: list[Path] = []

    def install(partial: Path, runtime_manifest, profile: str) -> Path:
        calls.append(partial)
        return _fake_install(partial, runtime_manifest, profile)

    installer = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        profile="win-x64-cpu",
        install_runner=install,
    )
    first = installer.ensure()
    second = installer.ensure()
    assert first == second
    assert len(calls) == 1
    assert Path(first.python_executable).is_file()
    assert not list(installer.paths.runtime_root.parent.glob("*.partial-*"))
    assert installer.inspect().integrity == "verified"
    portable_path_keys = {
        "VIBEOCR_PRODUCT_ROOT",
        "VIBEOCR_RUNTIME_ROOT",
        "VIBEOCR_MODEL_ROOT",
        "PIP_CACHE_DIR",
        "UV_CACHE_DIR",
        "HF_HOME",
        "MODELSCOPE_CACHE",
        "TEMP",
        "TMP",
    }
    assert all(
        Path(first.environment[key]).is_relative_to(installer.paths.store_root)
        for key in portable_path_keys
    )
    assert first.environment["PIP_CONFIG_FILE"] == os.devnull
    assert first.environment["PYTHONNOUSERSITE"] == "1"


def test_failed_install_leaves_no_partial_or_final(tmp_path: Path) -> None:
    manifest, component = _release(tmp_path / "release")

    def fail(_partial: Path, _manifest, _profile: str) -> Path:
        raise RuntimeInstallError("boom")

    installer = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        profile="win-x64-cpu",
        install_runner=fail,
    )
    with pytest.raises(RuntimeInstallError, match="boom"):
        installer.ensure()
    assert not installer.paths.runtime_root.exists()
    assert not list(installer.paths.runtime_root.parent.glob("*.partial-*"))


def test_component_lock_capability_mismatch_is_rejected(tmp_path: Path) -> None:
    manifest, component = _release(tmp_path / "release")
    data = json.loads(component.read_text(encoding="utf-8"))
    data["required_capabilities"].append("unknown.feature")
    component.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeInstallError, match="missing required"):
        RuntimeInstaller(
            product_root=tmp_path / "product",
            component_lock=component,
            runtime_manifest=manifest,
            profile="win-x64-cpu",
            install_runner=_fake_install,
        )


def test_component_lock_protocol_manifest_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    manifest, component = _release(tmp_path / "release")
    data = json.loads(component.read_text(encoding="utf-8"))
    data["protocol"]["manifest_sha256"] = "f" * 64
    component.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeInstallError, match="Protocol manifest mismatch"):
        RuntimeInstaller(
            product_root=tmp_path / "product",
            component_lock=component,
            runtime_manifest=manifest,
            profile="win-x64-cpu",
            install_runner=_fake_install,
        )


def test_store_lock_is_cross_handle_exclusive(tmp_path: Path) -> None:
    first = RuntimeStoreLock(tmp_path / "locks" / "store.lock", timeout=0)
    second = RuntimeStoreLock(tmp_path / "locks" / "store.lock", timeout=0)
    first.acquire()
    try:
        with pytest.raises(RuntimeLockTimeout):
            second.acquire()
    finally:
        first.release()
    second.acquire()
    second.release()


def test_gc_fails_closed_on_invalid_component_lock(tmp_path: Path) -> None:
    manifest, component = _release(tmp_path / "release")
    installer = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        profile="win-x64-cpu",
        install_runner=_fake_install,
    )
    installer.ensure()
    invalid = tmp_path / "bad-component-lock.json"
    invalid.write_text("not json", encoding="utf-8")
    assert installer.gc(component_locks=[invalid], grace_seconds=0) == []
    assert installer.paths.runtime_root.is_dir()


def test_gc_still_matches_after_prefix_shorten(tmp_path: Path) -> None:
    """目录名改为 6 位前缀后，GC 仍能按目录名重建 runtime_id 并正确回收。

    自洽性：referenced（从 lock 算前缀）=== 目录名重建值，无需读 marker 拼回。
    """
    manifest, component = _release(tmp_path / "release")
    installer = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        profile="win-x64-cpu",
        install_runner=_fake_install,
    )
    installer.ensure()
    # 当前 runtime 的目录名应为 6 位前缀（与 runtime_id 一致）。
    digest_dir = installer.paths.runtime_root.parent
    assert len(digest_dir.name) == 6
    assert digest_dir.name == runtime_id_prefix(installer.manifest.sha256)

    # 未被任何 lock 引用的孤儿目录也应能被 GC 按 6 位前缀目录名回收。
    orphan = installer.paths.runtimes_root / "deadbeef" / "win-x64-cpu"
    orphan.mkdir(parents=True)
    (orphan / ".installed.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runtime_id": "deadbeef/win-x64-cpu",
                "backend_version": installer.manifest.backend_version,
                "profile": "win-x64-cpu",
            }
        ),
        encoding="utf-8",
    )
    removed = installer.gc(component_locks=[component], grace_seconds=0)
    assert "deadbeef/win-x64-cpu" in removed
    assert not orphan.exists()
    # 被引用的当前 runtime 保留。
    assert installer.paths.runtime_root.is_dir()
