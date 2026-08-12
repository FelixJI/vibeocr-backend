from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest
from vibeocr.backend import runtime_maintenance
from vibeocr.backend.runtime_control import RuntimeControl
from vibeocr.backend.runtime_installer import (
    RuntimeInstaller,
    RuntimeInstallError,
    _extract_python_archive,
    _run_install_command,
    main,
)
from vibeocr.backend.runtime_layout import (
    LayoutError,
    resolve_runtime_store,
)
from vibeocr.backend.runtime_lock import RuntimeLockTimeout, RuntimeStoreLock
from vibeocr.backend.runtime_maintenance import (
    RuntimeInstallFailure,
    RuntimeMaintenanceReporter,
    profile_descriptor,
    runtime_status_from_environment,
)
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
            "accelerator": "cpu",
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


def test_python_archive_reports_actual_uncompressed_bytes(tmp_path: Path) -> None:
    archive = tmp_path / "python.tar.gz"
    _tar_with(archive, "python/python.exe", b"python")
    destination = tmp_path / "runtime"
    destination.mkdir()
    samples: list[tuple[int, int]] = []

    _extract_python_archive(
        archive,
        destination,
        progress=lambda current, total: samples.append((current, total)),
    )

    assert samples == [(6, 6)]


def test_python_archive_coalesces_dense_progress_without_losing_completion(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "python.tar.gz"
    with tarfile.open(archive, mode="w:gz") as stream:
        for index in range(103):
            info = tarfile.TarInfo(f"python/Lib/module-{index}.py")
            info.size = 1
            stream.addfile(info, io.BytesIO(b"x"))
    destination = tmp_path / "runtime"
    destination.mkdir()
    samples: list[tuple[int, int]] = []

    _extract_python_archive(
        archive,
        destination,
        progress=lambda current, total: samples.append((current, total)),
    )

    assert samples[-1] == (103, 103)
    assert len(samples) <= 102


def test_python_archive_does_not_repeat_completion_for_empty_tail_files(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "python.tar.gz"
    with tarfile.open(archive, mode="w:gz") as stream:
        content = tarfile.TarInfo("python/python.exe")
        content.size = 1
        stream.addfile(content, io.BytesIO(b"x"))
        for index in range(3):
            empty = tarfile.TarInfo(f"python/empty-{index}.txt")
            empty.size = 0
            stream.addfile(empty, io.BytesIO())
    destination = tmp_path / "runtime"
    destination.mkdir()
    samples: list[tuple[int, int]] = []

    _extract_python_archive(
        archive,
        destination,
        progress=lambda current, total: samples.append((current, total)),
    )

    assert samples == [(1, 1)]


def test_python_archive_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "python.tar.gz"
    _tar_with(archive, "python/../escape.exe")
    destination = tmp_path / "runtime"
    destination.mkdir()
    with pytest.raises(RuntimeInstallError, match="unsafe"):
        _extract_python_archive(archive, destination)
    assert not (tmp_path / "escape.exe").exists()


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
    )
    assert local.store_root == product.resolve()
    shared = resolve_runtime_store(
        product,
        manifest_sha256=digest,
        layout_manifest=marker,
        product_id="classic",
    )
    assert shared.store_root == (bundle / "shared").resolve()
    assert shared.runtime_root == (bundle / "shared" / "runtime").resolve()


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
        accelerator="cpu",
        install_runner=install,
    )
    first = installer.ensure()
    second = installer.ensure()
    assert first == second
    assert len(calls) == 1
    assert calls == [installer.paths.runtime_root.with_name("runtime.installing")]
    assert Path(first.python_executable).is_file()
    assert not installer.paths.runtime_root.with_name("runtime.installing").exists()
    assert installer.inspect().integrity == "verified"
    assert installer.inspect().accelerator == "cpu"
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
    assert first.environment["VIBEOCR_RUNTIME_ACCELERATOR"] == "cpu"
    assert first.environment["VIBEOCR_USE_GPU"] == "false"


def test_gpu_launch_environment_is_derived_from_installer_profile(
    tmp_path: Path,
) -> None:
    manifest, component = _release(tmp_path / "release")
    installer = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="nvidia_cuda",
        install_runner=_fake_install,
    )

    launch = installer.ensure()

    assert launch is not None
    assert launch.environment["VIBEOCR_RUNTIME_ACCELERATOR"] == "nvidia_cuda"
    assert launch.environment["VIBEOCR_USE_GPU"] == "true"


def test_failed_install_leaves_no_partial_or_final(tmp_path: Path) -> None:
    manifest, component = _release(tmp_path / "release")

    def fail(_partial: Path, _manifest, _profile: str) -> Path:
        raise RuntimeInstallError("boom")

    installer = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=fail,
    )
    with pytest.raises(RuntimeInstallError, match="boom"):
        installer.ensure()
    assert not installer.paths.runtime_root.exists()
    assert not installer.paths.runtime_root.with_name("runtime.installing").exists()
    assert installer.maintenance_snapshot()["operation_state"] == "failed"


def test_failed_repair_preserves_previous_runtime_until_verified_commit(
    tmp_path: Path,
) -> None:
    manifest, component = _release(tmp_path / "release")
    initial = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=_fake_install,
    )
    launch = initial.ensure()
    assert launch is not None
    preserved = initial.paths.runtime_root / "preserved.txt"
    preserved.write_text("previous-runtime", encoding="utf-8")
    marker_before = (initial.paths.runtime_root / ".installed.json").read_bytes()

    def fail(_partial: Path, _manifest, _profile: str) -> Path:
        raise RuntimeInstallError("repair failed")

    repair = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=fail,
        component_probe=lambda _root, component_ids: {
            component_id: component_id != "ocr_engine" for component_id in component_ids
        },
        operation_id="failed-repair",
    )

    with pytest.raises(RuntimeInstallError, match="repair failed"):
        repair.repair()

    assert preserved.read_text(encoding="utf-8") == "previous-runtime"
    assert (
        initial.paths.runtime_root / ".installed.json"
    ).read_bytes() == marker_before
    assert Path(launch.python_executable).is_file()
    assert not initial.paths.runtime_root.with_name("runtime.installing").exists()
    assert not initial.paths.runtime_root.with_name("runtime.rollback").exists()


def test_component_import_probe_reports_integrity_failed_drift(
    tmp_path: Path,
) -> None:
    manifest, component = _release(tmp_path / "release")
    initial = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=_fake_install,
    )
    initial.ensure()
    inspected = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=_fake_install,
        component_probe=lambda _root, component_ids: {
            component_id: component_id != "ocr_engine" for component_id in component_ids
        },
    )

    component_status = inspected.profile_payload()["components"][0]
    inspection = inspected.inspect(emit=False)

    assert component_status["actual_state"] == "drifted"
    assert component_status["drift_reason"] == "integrity_failed"
    assert component_status["repairable"] is True
    assert inspection.status == "missing"
    assert inspection.integrity == "not-installed"


def test_runtime_control_inspect_probes_components_once(tmp_path: Path) -> None:
    manifest, component = _release(tmp_path / "release")
    initial = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=_fake_install,
    )
    initial.ensure()
    probe_calls: list[tuple[Path, tuple[str, ...]]] = []

    def probe(runtime_root: Path, component_ids: tuple[str, ...]) -> dict[str, bool]:
        probe_calls.append((runtime_root, component_ids))
        return {component_id: True for component_id in component_ids}

    inspected = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=_fake_install,
        component_probe=probe,
        operation_id="inspect-once",
    )
    control = object.__new__(RuntimeControl)
    control._installer_factory = lambda **_kwargs: inspected
    control._active_snapshot = None

    result = control.execute_with_result(operation="inspect")

    assert result.state.integrity == "verified"
    assert result.profile["profile_id"] == "win-x64-cpu"
    assert len(probe_calls) == 1
    assert probe_calls[0][0] == inspected.paths.runtime_root
    assert all(
        component["actual_state"] == "ready"
        for component in result.profile["components"]
    )


def test_repair_of_in_sync_component_succeeds_without_claiming_global_ready(
    tmp_path: Path,
) -> None:
    manifest, component = _release(tmp_path / "release")
    calls: list[Path] = []

    def install(partial: Path, runtime_manifest, profile: str) -> Path:
        calls.append(partial)
        return _fake_install(partial, runtime_manifest, profile)

    initial = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=install,
    )
    initial.ensure()
    repair = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=install,
        component_probe=lambda _root, component_ids: {
            component_id: component_id != "ocr_engine" for component_id in component_ids
        },
        component_ids=("runtime_host",),
        operation_id="repair-ready-component",
    )

    launch = repair.repair()

    assert launch is None
    assert len(calls) == 1
    snapshot = repair.maintenance_snapshot()
    assert snapshot is not None
    assert snapshot["operation_state"] == "succeeded"
    assert snapshot["requested_component_ids"] == ["runtime_host"]
    assert "effective_component_ids" not in snapshot
    assert repair.profile_payload()["components"][0]["actual_state"] == "drifted"


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
            accelerator="cpu",
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
            accelerator="cpu",
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


def test_repair_is_idempotent_when_runtime_has_no_drift(tmp_path: Path) -> None:
    manifest, component = _release(tmp_path / "release")
    calls: list[Path] = []

    def install(partial: Path, runtime_manifest, profile: str) -> Path:
        calls.append(partial)
        return _fake_install(partial, runtime_manifest, profile)

    installer = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=install,
    )
    installer.ensure()
    original_marker = installer.paths.runtime_root / "original.txt"
    original_marker.write_text("old", encoding="utf-8")

    installer.repair()

    assert calls == [installer.paths.runtime_root.with_name("runtime.installing")]
    assert original_marker.read_text(encoding="utf-8") == "old"


def test_failed_operation_id_replays_failure_without_reexecuting(
    tmp_path: Path,
) -> None:
    manifest, component = _release(tmp_path / "release")

    def fail_install(partial: Path, runtime_manifest, profile: str) -> Path:
        del partial, runtime_manifest, profile
        raise RuntimeInstallError("expected failure")

    failed = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=fail_install,
        operation_id="stable-operation",
    )
    with pytest.raises(RuntimeInstallError, match="expected failure"):
        failed.ensure()
    calls: list[Path] = []

    def unexpected_install(partial: Path, runtime_manifest, profile: str) -> Path:
        calls.append(partial)
        return _fake_install(partial, runtime_manifest, profile)

    replay = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=unexpected_install,
        operation_id="stable-operation",
    )

    with pytest.raises(RuntimeInstallFailure, match="expected failure"):
        replay.ensure()
    assert calls == []
    assert replay.maintenance_snapshot() == failed.maintenance_snapshot()


def test_component_repair_reports_requested_and_effective_scope(tmp_path: Path) -> None:
    manifest, component = _release(tmp_path / "release")
    calls: list[Path] = []

    def install(partial: Path, runtime_manifest, profile: str) -> Path:
        calls.append(partial)
        return _fake_install(partial, runtime_manifest, profile)

    initial = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=install,
    )
    initial.ensure()
    marker = initial.paths.runtime_root / ".installed.json"
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    marker_payload["manifest_sha256"] = "f" * 64
    marker.write_text(json.dumps(marker_payload), encoding="utf-8")

    repair = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=install,
        operation_id="repair-1",
        component_ids=("ocr_engine",),
    )
    repair.repair()

    assert len(calls) == 2
    snapshot = repair.maintenance_snapshot()
    assert snapshot is not None
    assert snapshot["requested_component_ids"] == ["ocr_engine"]
    assert set(snapshot["effective_component_ids"]) == {
        component.component_id
        for component in repair.manifest.profiles[repair.plan].components
    }
    assert repair.paths.runtime_root.is_dir()


def test_component_drift_uses_installed_distribution_and_selected_repair(
    tmp_path: Path,
) -> None:
    manifest, component_lock = _release(tmp_path / "release")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["profiles"]["win-x64-cpu"]["components"] = [
        {
            "component_id": "ocr_engine",
            "display_name": "OCR engine",
            "version": "3.7.0",
        },
        *[
            {"component_id": component_id, "display_name": display_name}
            for component_id, display_name in (
                ("document_parsing", "Document parsing"),
                ("pdf_document_tools", "PDF and document tools"),
                ("image_code_tools", "Image and code tools"),
                ("runtime_host", "Runtime HTTP host"),
            )
        ],
    ]
    manifest.write_text(json.dumps(manifest_payload) + "\n", encoding="utf-8")
    lock_payload = json.loads(component_lock.read_text(encoding="utf-8"))
    lock_payload["backend"]["runtime_manifest_sha256"] = _sha(manifest.read_bytes())
    component_lock.write_text(json.dumps(lock_payload) + "\n", encoding="utf-8")
    calls: list[Path] = []

    def install(partial: Path, runtime_manifest, profile: str) -> Path:
        calls.append(partial)
        python = _fake_install(partial, runtime_manifest, profile)
        metadata = partial / "Lib" / "site-packages" / "paddleocr-3.7.0.dist-info"
        metadata.mkdir(parents=True)
        (metadata / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: paddleocr\nVersion: 3.7.0\n",
            encoding="utf-8",
        )
        return python

    initial = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component_lock,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=install,
    )
    initial.ensure()
    metadata = (
        initial.paths.runtime_root
        / "Lib"
        / "site-packages"
        / "paddleocr-3.7.0.dist-info"
        / "METADATA"
    )
    metadata.write_text(
        "Metadata-Version: 2.1\nName: paddleocr\nVersion: 3.6.0\n",
        encoding="utf-8",
    )
    drifted = initial.profile_payload()["components"][0]
    assert drifted["actual_state"] == "drifted"
    assert drifted["actual_version"] == "3.6.0"
    assert drifted["drift_reason"] == "version_mismatch"
    metadata.unlink()
    missing = initial.profile_payload()["components"][0]
    assert missing["actual_state"] == "missing"

    repair = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component_lock,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=install,
        operation_id="repair-component",
        component_ids=("ocr_engine",),
    )
    repair.repair()

    assert len(calls) == 2
    assert repair.profile_payload()["components"][0]["actual_state"] == "ready"


def test_runtime_host_json_contract_selects_and_persists_accelerator(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, component = _release(tmp_path / "release")

    request = {
        "protocol_version": 2,
        "operation": "inspect",
        "product_root": str(tmp_path / "product"),
        "component_lock": str(component),
        "runtime_manifest": str(manifest),
        "accelerator": "nvidia_cuda",
    }
    assert main(["--request-json", json.dumps(request)]) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["protocol_version"] == 2
    assert envelope["ok"] is True
    assert envelope["operation"] == "inspect"
    assert envelope["state"]["accelerator"] == "nvidia_cuda"
    assert envelope["state"]["runtime_root"].endswith("runtime")
    assert "profile" not in envelope["state"]
    assert envelope["profile"]["profile_id"] == "win-x64-cu126"
    assert envelope["profile"]["components"][-1]["component_id"] == "gpu_runtime"
    assert envelope["maintenance"]["operation_state"] == "succeeded"
    descriptors = {item["name"]: item for item in envelope["capability_descriptors"]}
    recognition = descriptors["ocr.recognition.v2"]
    assert recognition == {
        "name": "ocr.recognition.v2",
        "lifecycle": "active",
        "introduced_in": "2.0.0",
        "deprecated_in": None,
        "sunset_at": None,
        "replacement": None,
    }


def test_runtime_host_emits_opt_in_ndjson_progress(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, component = _release(tmp_path / "release")
    request = {
        "protocol_version": 2,
        "operation": "inspect",
        "product_root": str(tmp_path / "product"),
        "component_lock": str(component),
        "runtime_manifest": str(manifest),
        "accelerator": "cpu",
        "accepted_event_streams": ["ndjson.v1"],
    }

    assert main(["--request-json", json.dumps(request)]) == 0

    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    events = messages[:-1]
    response = messages[-1]
    assert [event["event_type"] for event in events] == ["progress", "snapshot"]
    assert [event["snapshot"]["sequence"] for event in events] == [1, 2]
    assert all(event["operation"] == "inspect" for event in events)
    assert response["ok"] is True
    assert response["maintenance"] == events[-1]["snapshot"]


def test_runtime_host_rejects_unknown_event_stream(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, component = _release(tmp_path / "release")
    request = {
        "protocol_version": 2,
        "operation": "inspect",
        "product_root": str(tmp_path / "product"),
        "component_lock": str(component),
        "runtime_manifest": str(manifest),
        "accepted_event_streams": ["sse.v1"],
    }

    assert main(["--request-json", json.dumps(request)]) == 1
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["error"]["code"] == "invalid_request"


def test_install_progress_and_http_status_share_the_persisted_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, component = _release(tmp_path / "release")
    events: list[dict] = []
    installer = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=_fake_install,
        event_sink=events.append,
    )

    launch = installer.ensure()

    assert [event["snapshot"]["phase"] for event in events] == [
        "validate_binding",
        "wait_for_lock",
        "prepare_runtime",
        "install_profile",
        "verify_runtime",
        "commit_runtime",
        "commit_runtime",
    ]
    assert events[-1]["snapshot"]["operation_state"] == "succeeded"
    for key, value in launch.environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        "vibeocr.backend.runtime_maintenance.probe_runtime_components",
        lambda _root, component_ids: {
            component_id: True for component_id in component_ids
        },
    )
    runtime_maintenance._component_probe_cache.clear()
    status = runtime_status_from_environment("instance-test", "ready")
    assert status["maintenance"]["sequence"] == events[-1]["snapshot"]["sequence"]
    assert status["maintenance"]["message_code"] == "runtime.ensure_complete"
    assert status["profile"]["components"][0] == {
        "component_id": "ocr_engine",
        "display_name": "OCR engine",
        "state": "ready",
        "desired_state": "ready",
        "desired_version": None,
        "actual_state": "ready",
        "actual_version": None,
        "drift_reason": "none",
        "repairable": False,
    }
    assert status["source"]["backend_source_sha"] == "0" * 40

    monkeypatch.setattr(
        "vibeocr.backend.runtime_maintenance.probe_runtime_components",
        lambda _root, component_ids: {
            component_id: component_id != "ocr_engine" for component_id in component_ids
        },
    )
    runtime_maintenance._component_probe_cache.clear()
    degraded = runtime_status_from_environment("instance-test", "ready")
    assert degraded["service_state"] == "degraded"
    assert degraded["profile"]["components"][0]["drift_reason"] == ("integrity_failed")


def test_long_install_command_emits_heartbeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _ = _release(tmp_path / "release")
    manifest = load_runtime_manifest(manifest_path)
    events: list[dict] = []
    reporter = RuntimeMaintenanceReporter(
        state_root=tmp_path / "state",
        profile=profile_descriptor(
            manifest.profiles["win-x64-cpu"],
            accelerator="cpu",
        ),
        event_sink=events.append,
    )
    reporter.start("ensure", total_steps=7)
    reporter.advance(
        phase="install_profile",
        current=4,
        total=7,
        message_code="runtime.install_profile",
    )

    class FakeProcess:
        returncode = 0

        def __init__(self) -> None:
            self.communicate_calls = 0

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                raise subprocess.TimeoutExpired("pip", timeout)
            return "", ""

        def poll(self) -> int:
            return self.returncode

    process = FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)

    _run_install_command(
        ["python.exe", "-m", "pip"],
        timeout=60,
        env={},
        reporter=reporter,
        heartbeat_code="runtime.install_profile",
    )

    assert events[-1]["event_type"] == "heartbeat"
    assert events[-1]["snapshot"]["phase"] == "install_profile"
    assert events[-1]["message_code"] == "runtime.install_profile"


def test_runtime_host_rejects_legacy_profile_field(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, component = _release(tmp_path / "release")
    request = {
        "protocol_version": 2,
        "operation": "inspect",
        "product_root": str(tmp_path / "product"),
        "component_lock": str(component),
        "runtime_manifest": str(manifest),
        "profile": "win-x64-cpu",
    }
    assert main(["--request-json", json.dumps(request)]) == 1
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "invalid_request"
