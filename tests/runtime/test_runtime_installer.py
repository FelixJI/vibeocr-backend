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
from vibeocr.backend.model_registry import (
    ModelAcquisitionError,
    ResolvedModel,
    ResolvedModelSet,
)
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
    probe_runtime_components,
    profile_descriptor,
    runtime_profile_status,
    runtime_status_from_environment,
)
from vibeocr.backend.runtime_manifest import (
    ManifestError,
    load_runtime_manifest,
    runtime_component_binding,
    validate_requirements_lock,
)
from vibeocr.backend.runtime_selection import BoundDownloadSource


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _skip_model_acquisition(
    installer: RuntimeInstaller,
    **_kwargs: object,
) -> None:
    installer._resolved_models = ResolvedModelSet(  # noqa: SLF001
        installer._model_release_identity(),  # noqa: SLF001
        (),
    )


def _lock_text(profile: str) -> str:
    base = "fastapi==1.0.0 \\\n    --hash=sha256:" + "1" * 64 + "\n"
    if profile == "win-x64-base":
        return base + (
            "rapidocr==3.9.2 \\\n    --hash=sha256:" + "6" * 64 + "\n"
            "onnxruntime==1.28.0 \\\n    --hash=sha256:" + "6" * 64 + "\n"
            "winrt-runtime==3.2.1 \\\n    --hash=sha256:" + "6" * 64 + "\n"
            "opencv-python==5.0.0.93 \\\n    --hash=sha256:" + "6" * 64 + "\n"
        )
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


def _release(root: Path, *, with_base_pack: bool = False) -> tuple[Path, Path]:
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
    for profile in ("win-x64-base", "win-x64-cpu", "win-x64-cu126"):
        lock = root / f"requirements-{profile}.lock"
        lock.write_text(_lock_text(profile), encoding="utf-8")
        profiles[profile] = {
            "lock": lock.name,
            "sha256": _sha(lock.read_bytes()),
            "runtime_pack": None,
        }
    cu126_gpu_lock = root / "requirements-win-x64-cu126-gpu.lock"
    cu126_gpu_lock.write_text(_lock_text("win-x64-cu126"), encoding="utf-8")
    profiles["win-x64-cu126"]["install_scopes"] = [
        {
            "scope_id": "gpu-runtime",
            "component_ids": [
                "ocr_engine",
                "pdf_document_tools",
                "image_code_tools",
                "runtime_host",
                "gpu_runtime",
            ],
            "lock": cu126_gpu_lock.name,
            "sha256": _sha(cu126_gpu_lock.read_bytes()),
            "runtime_pack": None,
        }
    ]
    if with_base_pack:
        pack = root / "vibeocr-runtime-pack-win-x64-base-0.7.0.zip"
        with zipfile.ZipFile(pack, mode="w") as archive:
            archive.writestr("pack-requirements.txt", "rapidocr==3.9.2")
            archive.writestr("rapidocr-3.9.2-py3-none-any.whl", b"rapidocr-wheel")
            archive.writestr(
                "onnxruntime-1.28.0-cp313-cp313-win_amd64.whl", b"ort-wheel"
            )
        profiles["win-x64-base"]["runtime_pack"] = [pack.name]
        profiles["win-x64-base"]["runtime_pack_sha256"] = [_sha(pack.read_bytes())]
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


def _pypi_source() -> tuple[BoundDownloadSource, ...]:
    return (
        BoundDownloadSource(
            kind="package_index",
            source_id="pypi",
            endpoint="https://pypi.org/simple",
        ),
    )


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
    assert set(manifest.profiles) == {"win-x64-base", "win-x64-cpu", "win-x64-cu126"}


def test_manifest_rejects_tampered_lock(tmp_path: Path) -> None:
    manifest_path, _ = _release(tmp_path / "release")
    (manifest_path.parent / "requirements-win-x64-cpu.lock").write_text(
        "tampered",
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="SHA-256 mismatch"):
        load_runtime_manifest(manifest_path)


@pytest.mark.parametrize(
    "directive",
    ["--index-url https://pypi.org/simple", "--extra-index-url https://extra.invalid"],
)
def test_runtime_lock_rejects_embedded_source_directives(
    tmp_path: Path,
    directive: str,
) -> None:
    lock = tmp_path / "requirements-win-x64-cpu.lock"
    lock.write_text(f"{directive}\n{_lock_text('win-x64-cpu')}", encoding="utf-8")

    with pytest.raises(ManifestError, match="source-neutral"):
        validate_requirements_lock(lock, profile="win-x64-cpu")


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
        install_component_ids=(),
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
        install_component_ids=(),
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
        install_component_ids=(),
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
        component_probe=lambda _root, component_ids, _profile_id: {
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
        install_component_ids=(),
    )
    initial.ensure()
    inspected = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=_fake_install,
        install_component_ids=(),
        component_probe=lambda _root, component_ids, _profile_id: {
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


def test_base_component_probe_uses_rapidocr_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_root = tmp_path / "runtime"
    python = runtime_root / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")

    def run(command, **_kwargs):  # type: ignore[no-untyped-def]
        modules = json.loads(command[-1])
        result = {
            component_id: module == "rapidocr"
            for component_id, module in modules.items()
        }
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="VIBEOCR_COMPONENT_PROBE=" + json.dumps(result) + "\n",
            stderr="",
        )

    monkeypatch.setattr(runtime_maintenance.subprocess, "run", run)

    assert probe_runtime_components(
        runtime_root,
        ("ocr_engine",),
        profile_id="win-x64-base",
    ) == {"ocr_engine": True}


def test_runtime_control_inspect_probes_components_once(tmp_path: Path) -> None:
    manifest, component = _release(tmp_path / "release")
    initial = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=_fake_install,
        install_component_ids=(),
    )
    initial.ensure()
    probe_calls: list[tuple[Path, tuple[str, ...]]] = []

    def probe(
        runtime_root: Path, component_ids: tuple[str, ...], _profile_id: str
    ) -> dict[str, bool]:
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
        install_component_ids=(),
    )
    initial.ensure()
    repair = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=install,
        install_component_ids=(),
        component_probe=lambda _root, component_ids, _profile_id: {
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
        install_component_ids=(),
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


def test_component_repair_reports_requested_and_effective_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(RuntimeInstaller, "_acquire_models", _skip_model_acquisition)
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
    Path(initial._launch().python_executable).unlink()

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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RuntimeInstaller, "_acquire_models", _skip_model_acquisition)
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
        install_component_ids=(),
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
        lambda _root, component_ids, **_kwargs: {
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
        lambda _root, component_ids, **_kwargs: {
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
    events_before_cancel_checks = list(events)
    for _ in range(100):
        reporter.check_cancelled()
    assert events == events_before_cancel_checks

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


# ---------------------------------------------------------------------------
# base-offline 离线安装路径（计划 §4.2）
# ---------------------------------------------------------------------------


def _run_default_installer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_base_pack: bool,
    profile: str = "win-x64-base",
) -> tuple[list[list[str]], Path, Path]:
    """Run ``_default_install_runner`` with captured pip commands.

    Returns (captured_commands, partial_root, manifest_path). The Python
    archive extraction and every child process are faked so the test only
    asserts command construction and pack handling.
    """
    from vibeocr.backend import runtime_installer as installer

    manifest_path, _ = _release(tmp_path / "release", with_base_pack=with_base_pack)
    manifest = load_runtime_manifest(manifest_path)
    partial_root = tmp_path / "runtimes" / "runtime-0" / "partial"
    partial_root.mkdir(parents=True)
    commands: list[list[str]] = []

    def fake_run(command, *, timeout, env, reporter, heartbeat_code):  # type: ignore[no-untyped-def]
        commands.append(list(command))

    def fake_extract_python(archive_path, destination, *, progress=None):  # type: ignore[no-untyped-def]
        (destination / "python.exe").write_bytes(b"python")

    monkeypatch.setattr(installer, "_run_install_command", fake_run)
    monkeypatch.setattr(installer, "_extract_python_archive", fake_extract_python)
    installer._default_install_runner(
        partial_root,
        manifest,
        manifest.profiles[profile].scopes[0],
        _pypi_source(),
    )
    return commands, partial_root, manifest_path


class TestOfflineRuntimePack:
    def test_bound_pack_installs_with_no_index_and_find_links(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        commands, partial_root, manifest_path = _run_default_installer(
            tmp_path, monkeypatch, with_base_pack=True
        )
        profile_install = commands[0]
        assert "--no-index" in profile_install
        # 离线路径不做逐件 --require-hashes:pack 完整性由 manifest 绑定。
        assert "--require-hashes" not in profile_install
        find_links = profile_install[
            profile_install.index("--find-links") + 1  # type: ignore[arg-type]
        ]
        pack_dir = Path(find_links)
        # pack 已解压到 state 缓存且包含全部 wheel。
        assert (pack_dir / "rapidocr-3.9.2-py3-none-any.whl").is_file()
        assert (pack_dir / "onnxruntime-1.28.0-cp313-cp313-win_amd64.whl").is_file()
        assert (pack_dir / ".complete").is_file()
        # 完整标记存在时重复安装幂等复用，不重复解压。
        marker_before = (pack_dir / ".complete").stat().st_mtime_ns
        # 同一 partial 目录重复执行：完整标记存在时直接复用解压结果。
        from vibeocr.backend import runtime_installer as installer

        manifest = load_runtime_manifest(manifest_path)
        commands.clear()
        installer._default_install_runner(
            partial_root,
            manifest,
            manifest.profiles["win-x64-base"].scopes[0],
            _pypi_source(),
        )
        assert commands[0][commands[0].index("--find-links") + 1] == find_links
        assert (pack_dir / ".complete").stat().st_mtime_ns == marker_before

    def test_without_pack_installs_online_without_no_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        commands, _, _ = _run_default_installer(
            tmp_path, monkeypatch, with_base_pack=False
        )
        assert "--no-index" not in commands[0]
        assert "--find-links" not in commands[0]
        assert "--require-hashes" in commands[0]
        assert commands[0][-2:] == [
            "-r",
            str(tmp_path / "release" / "requirements-win-x64-base.lock"),
        ]

    def test_missing_bound_pack_fails_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vibeocr.backend import runtime_installer as installer

        manifest_path, _ = _release(tmp_path / "release", with_base_pack=True)
        (tmp_path / "release" / "vibeocr-runtime-pack-win-x64-base-0.7.0.zip").unlink()
        partial_root = tmp_path / "runtimes" / "runtime-0" / "partial"
        partial_root.mkdir(parents=True)
        manifest = load_runtime_manifest(manifest_path, verify_artifacts=False)

        def fake_run(command, *, timeout, env, reporter, heartbeat_code):  # type: ignore[no-untyped-def]
            raise AssertionError("install must not run when the pack is missing")

        monkeypatch.setattr(installer, "_run_install_command", fake_run)

        def fake_extract_python(archive_path, destination, *, progress=None):  # type: ignore[no-untyped-def]
            (destination / "python.exe").write_bytes(b"python")

        monkeypatch.setattr(installer, "_extract_python_archive", fake_extract_python)
        with pytest.raises(RuntimeInstallError, match="runtime pack is missing"):
            installer._default_install_runner(
                partial_root,
                manifest,
                manifest.profiles["win-x64-base"].scopes[0],
                _pypi_source(),
            )


@pytest.mark.parametrize(
    ("source_ids", "expected_endpoint"),
    [
        (None, "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/"),
        (("pypi",), "https://pypi.org/simple"),
    ],
)
def test_online_install_uses_selected_source_and_isolates_parent_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_ids: tuple[str, ...] | None,
    expected_endpoint: str,
) -> None:
    from vibeocr.backend import runtime_installer as installer_module

    manifest, component = _release(tmp_path / "release")
    captured: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command, *, timeout, env, reporter, heartbeat_code):  # type: ignore[no-untyped-def]
        captured.append((list(command), dict(env)))

    def fake_extract_python(archive_path, destination, *, progress=None):  # type: ignore[no-untyped-def]
        (destination / "python.exe").write_bytes(b"python")

    monkeypatch.setattr(installer_module, "_run_install_command", fake_run)
    monkeypatch.setattr(
        installer_module,
        "_extract_python_archive",
        fake_extract_python,
    )
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://extra.invalid/simple")
    monkeypatch.setenv("PIP_FIND_LINKS", "https://links.invalid")
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("UV_INDEX", "https://uv.invalid/simple")

    runtime = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        component_probe=lambda _root, ids, _profile_id: dict.fromkeys(ids, True),
        download_source_ids=source_ids,
        install_component_ids=(),
    )
    launch = runtime.ensure()

    profile_command, child_env = captured[0]
    assert profile_command[profile_command.index("--index-url") + 1] == (
        expected_endpoint
    )
    assert "--require-hashes" in profile_command
    assert "PIP_EXTRA_INDEX_URL" not in child_env
    assert "PIP_FIND_LINKS" not in child_env
    assert "PIP_NO_INDEX" not in child_env
    assert "UV_INDEX" not in child_env
    assert launch is not None
    assert launch.environment["PIP_INDEX_URL"] == expected_endpoint


def test_cuda_gpu_only_selection_uses_exact_install_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vibeocr.backend import runtime_installer as installer_module

    manifest, component = _release(tmp_path / "release")
    captured: list[list[str]] = []

    def fake_run(command, *, timeout, env, reporter, heartbeat_code):  # type: ignore[no-untyped-def]
        captured.append(list(command))

    def fake_extract_python(archive_path, destination, *, progress=None):  # type: ignore[no-untyped-def]
        (destination / "python.exe").write_bytes(b"python")

    monkeypatch.setattr(installer_module, "_run_install_command", fake_run)
    monkeypatch.setattr(
        installer_module,
        "_extract_python_archive",
        fake_extract_python,
    )

    runtime = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="nvidia_cuda",
        component_probe=lambda _root, ids, _profile_id: dict.fromkeys(ids, True),
        install_component_ids=("gpu_runtime",),
    )
    runtime.ensure()

    assert captured[0][-1] == str(
        tmp_path / "release" / "requirements-win-x64-cu126-gpu.lock"
    )
    marker = json.loads(
        (runtime.paths.runtime_root / ".installed.json").read_text(encoding="utf-8")
    )
    assert marker["component_ids"] == [
        "ocr_engine",
        "pdf_document_tools",
        "image_code_tools",
        "runtime_host",
        "gpu_runtime",
    ]


def test_control_installer_store_composition_retries_durable_selection(
    tmp_path: Path,
) -> None:
    manifest, component = _release(tmp_path / "release")
    product_root = tmp_path / "product"
    constructed: list[dict[str, object]] = []

    def fail_install(_partial: Path, _manifest, _profile: str) -> Path:
        raise RuntimeInstallError("synthetic install failure")

    def factory(**kwargs):  # type: ignore[no-untyped-def]
        constructed.append(dict(kwargs))
        install_runner = (
            fail_install
            if kwargs.get("operation_id") == "composition-failed"
            else _fake_install
        )
        return RuntimeInstaller(
            product_root=product_root,
            component_lock=component,
            runtime_manifest=manifest,
            accelerator="cpu",
            install_runner=install_runner,
            **kwargs,
        )

    control = RuntimeControl.from_installer_factory(factory)
    with pytest.raises(RuntimeInstallError, match="synthetic install failure"):
        control.execute(
            operation="ensure",
            operation_id="composition-failed",
            install_component_ids=(),
            download_source_ids=("pypi",),
        )

    receipt = control.command(
        command_id="composition-retry-command",
        command="retry",
        target_operation_id="composition-failed",
        new_operation_id="composition-retried",
    )
    assert receipt["snapshot"]["operation_state"] == "succeeded"
    retried = next(
        item
        for item in constructed
        if item.get("operation_id") == "composition-retried"
    )
    assert retried["install_component_ids"] == ()
    assert retried["download_source_ids"] == ("pypi",)

    intent = runtime_maintenance.RuntimeOperationStore(control.state_root).intent(
        "composition-retried"
    )
    assert intent["install_component_ids"] == []
    assert intent["download_source_ids"] == ["pypi"]
    restarted = RuntimeControl.from_installer_factory(factory)
    observation = restarted.observe("composition-retried")
    assert observation["snapshot"]["operation_state"] == "succeeded"
    assert observation["events"][-1]["snapshot"]["operation_state"] == "succeeded"

    default_receipt = restarted.execute(
        operation="ensure",
        operation_id="composition-default-source",
        install_component_ids=(),
    )
    assert default_receipt["snapshot"]["effective_download_source_ids"] == ["tuna-pypi"]
    default_intent = runtime_maintenance.RuntimeOperationStore(
        restarted.state_root
    ).intent("composition-default-source")
    assert default_intent["download_source_ids"] == ["tuna-pypi"]


def test_extract_runtime_pack_rejects_unsafe_members(tmp_path: Path) -> None:
    from vibeocr.backend import runtime_installer as installer

    pack = tmp_path / "pack.zip"
    with zipfile.ZipFile(pack, mode="w") as archive:
        archive.writestr("../evil.whl", b"evil")
    with pytest.raises(RuntimeInstallError, match="unsafe runtime pack member"):
        installer._extract_runtime_pack([pack], tmp_path / "cache")

    pack2 = tmp_path / "pack2.zip"
    with zipfile.ZipFile(pack2, mode="w") as archive:
        archive.writestr("payload.txt", b"not a wheel")
    with pytest.raises(RuntimeInstallError, match="unsafe runtime pack member"):
        installer._extract_runtime_pack([pack2], tmp_path / "cache2")


def test_base_accelerator_maps_to_base_profile() -> None:
    from vibeocr.backend.runtime_installer import ACCELERATOR_TO_PLAN

    assert ACCELERATOR_TO_PLAN["base"] == "win-x64-base"


def test_extract_runtime_pack_requires_pack_requirements(tmp_path: Path) -> None:
    from vibeocr.backend import runtime_installer as installer

    pack = tmp_path / "pack.zip"
    with zipfile.ZipFile(pack, mode="w") as archive:
        archive.writestr("rapidocr-3.9.2-py3-none-any.whl", b"wheel")
    with pytest.raises(RuntimeInstallError, match="lacks pack-requirements.txt"):
        installer._extract_runtime_pack([pack], tmp_path / "cache")


def test_full_profile_without_pack_falls_back_online(tmp_path: Path) -> None:
    """full 闭包绑定的 pack 未下载到位时回退在线安装,不 fail closed。"""
    from vibeocr.backend import runtime_installer as installer

    manifest_path, _ = _release(tmp_path / "release")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["profiles"]["win-x64-cpu"]["runtime_pack"] = ["not-yet-downloaded.zip"]
    raw["profiles"]["win-x64-cpu"]["runtime_pack_sha256"] = ["0" * 64]
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    # 绑定的 pack 文件不存在:component lock 的 manifest 哈希已失配,直接
    # 构造 manifest 对象驱动 runner(绕过 _validate_binding)。
    manifest = load_runtime_manifest(manifest_path, verify_artifacts=False)
    partial_root = tmp_path / "runtimes" / "runtime-0" / "partial"
    partial_root.mkdir(parents=True)
    commands: list[list[str]] = []

    def fake_run(command, *, timeout, env, reporter, heartbeat_code):  # type: ignore[no-untyped-def]
        commands.append(list(command))

    def fake_extract_python(archive_path, destination, *, progress=None):  # type: ignore[no-untyped-def]
        (destination / "python.exe").write_bytes(b"python")

    original_run = installer._run_install_command
    original_extract = installer._extract_python_archive
    installer._run_install_command = fake_run  # type: ignore[assignment]
    installer._extract_python_archive = fake_extract_python  # type: ignore[assignment]
    try:
        installer._default_install_runner(
            partial_root,
            manifest,
            manifest.profiles["win-x64-cpu"].scopes[0],
            _pypi_source(),
        )
    finally:
        installer._run_install_command = original_run  # type: ignore[assignment]
        installer._extract_python_archive = original_extract  # type: ignore[assignment]
    # 回退在线:require-hashes + lock,而非 --no-index。
    assert "--require-hashes" in commands[0]
    assert "--no-index" not in commands[0]
    assert commands[0][-1].endswith("requirements-win-x64-cpu.lock")


def test_multi_part_pack_extracts_into_one_directory(tmp_path: Path) -> None:
    from vibeocr.backend import runtime_installer as installer

    part1 = tmp_path / "vibeocr-runtime-pack-win-x64-cpu-0.7.0.part01.zip"
    part2 = tmp_path / "vibeocr-runtime-pack-win-x64-cpu-0.7.0.part02.zip"
    with zipfile.ZipFile(part1, mode="w") as archive:
        archive.writestr("pack-requirements.txt", "rapidocr==3.9.2\n")
        archive.writestr("rapidocr-3.9.2-py3-none-any.whl", b"rapidocr-wheel")
    with zipfile.ZipFile(part2, mode="w") as archive:
        archive.writestr("onnxruntime-1.28.0-cp313-cp313-win_amd64.whl", b"ort")

    pack_dir = installer._extract_runtime_pack([part1, part2], tmp_path / "cache")
    assert pack_dir.name == "vibeocr-runtime-pack-win-x64-cpu-0.7.0"
    assert (pack_dir / "pack-requirements.txt").is_file()
    assert (pack_dir / "rapidocr-3.9.2-py3-none-any.whl").is_file()
    assert (pack_dir / "onnxruntime-1.28.0-cp313-cp313-win_amd64.whl").is_file()
    assert (pack_dir / ".complete").is_file()
    # 幂等:完整标记存在时直接复用。
    again = installer._extract_runtime_pack([part1, part2], tmp_path / "cache")
    assert again == pack_dir


def test_ensure_with_explicit_base_only_scope_installs_base_lock(
    tmp_path: Path,
) -> None:
    manifest, component = _release(tmp_path / "release")
    seen_profiles: list[str] = []

    def install(partial: Path, runtime_manifest, profile: str) -> Path:
        seen_profiles.append(profile)
        return _fake_install(partial, runtime_manifest, profile)

    installer = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=install,
        install_component_ids=(),
    )
    installer.ensure()

    assert seen_profiles == ["win-x64-base"]
    marker = json.loads(
        (installer.paths.runtime_root / ".installed.json").read_text(encoding="utf-8")
    )
    assert "document_parsing" not in marker["component_ids"]
    snapshot = installer.maintenance_snapshot()
    # 显式空集回显 requested=[]（base-only），effective 为 base 闭包。
    assert snapshot["requested_component_ids"] == []
    assert "document_parsing" not in snapshot["effective_component_ids"]
    # base-only 缺 full 可选组件不算漂移：inspect 仍 ready。
    assert installer.inspect().integrity == "verified"


@pytest.mark.parametrize(
    ("install_component_ids", "expected_profile", "expected_ocr_import"),
    [
        ((), "win-x64-base", "rapidocr"),
        (("document_parsing",), "win-x64-cpu", "paddleocr"),
    ],
)
def test_install_probe_uses_actual_install_scope_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    install_component_ids: tuple[str, ...],
    expected_profile: str,
    expected_ocr_import: str,
) -> None:
    monkeypatch.setattr(RuntimeInstaller, "_acquire_models", _skip_model_acquisition)
    manifest, component = _release(tmp_path / "release")

    def probe(
        runtime_root: Path,
        component_ids: tuple[str, ...],
        profile_id: str,
    ) -> dict[str, bool]:
        if runtime_root.name != "runtime.installing":
            return dict.fromkeys(component_ids, True)
        return {
            component_id: (
                runtime_component_binding(profile_id, component_id).import_name
                == expected_ocr_import
                if component_id == "ocr_engine"
                else profile_id == expected_profile
            )
            for component_id in component_ids
        }

    installer = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=_fake_install,
        component_probe=probe,
        install_component_ids=install_component_ids,
    )

    assert installer.ensure() is not None


def test_repair_preserves_trusted_base_only_closure_when_runtime_is_incomplete(
    tmp_path: Path,
) -> None:
    manifest, component = _release(tmp_path / "release")
    installed_profiles: list[str] = []

    def install(partial: Path, runtime_manifest, profile: str) -> Path:
        installed_profiles.append(profile)
        return _fake_install(partial, runtime_manifest, profile)

    base_only = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=install,
        install_component_ids=(),
    )
    base_only.ensure()
    Path(base_only._launch().python_executable).unlink()

    repair = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=install,
    )
    repair.repair()

    assert installed_profiles == ["win-x64-base", "win-x64-base"]
    marker = json.loads(
        (repair.paths.runtime_root / ".installed.json").read_text(encoding="utf-8")
    )
    assert "document_parsing" not in marker["component_ids"]


def test_repair_fails_closed_when_installed_marker_is_untrusted(
    tmp_path: Path,
) -> None:
    manifest, component = _release(tmp_path / "release")
    initial = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=_fake_install,
        install_component_ids=(),
    )
    initial.ensure()
    marker_path = initial.paths.runtime_root / ".installed.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["manifest_sha256"] = "f" * 64
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    repair = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=_fake_install,
    )

    with pytest.raises(RuntimeInstallError, match="untrusted installed marker"):
        repair.repair()


def test_ensure_with_optional_components_reports_full_profile_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RuntimeInstaller, "_acquire_models", _skip_model_acquisition)
    manifest, component = _release(tmp_path / "release")
    seen_profiles: list[str] = []

    def install(partial: Path, runtime_manifest, profile: str) -> Path:
        seen_profiles.append(profile)
        return _fake_install(partial, runtime_manifest, profile)

    installer = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=install,
        install_component_ids=("document_parsing",),
    )
    installer.ensure()

    # per-profile lock 粒度：选择任一可选组件即安装目标档位完整闭包，
    # effective 诚实回显闭包而非请求子集。
    assert seen_profiles == ["win-x64-cpu"]
    snapshot = installer.maintenance_snapshot()
    assert snapshot["requested_component_ids"] == ["document_parsing"]
    assert snapshot["effective_component_ids"] == list(
        installer._profile_component_ids()
    )


def test_ensure_reinstalls_when_scope_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(RuntimeInstaller, "_acquire_models", _skip_model_acquisition)
    manifest, component = _release(tmp_path / "release")
    calls: list[Path] = []

    def install(partial: Path, runtime_manifest, profile: str) -> Path:
        calls.append(partial)
        return _fake_install(partial, runtime_manifest, profile)

    base_only = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=install,
        install_component_ids=(),
    )
    base_only.ensure()

    # 范围从 base-only 扩到缺省全量：marker 闭包不同 → 重装一次。
    default_scope = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=install,
    )
    default_scope.ensure()
    assert len(calls) == 2

    # 幂等：同范围再次 ensure 不重装。
    default_scope.ensure()
    assert len(calls) == 2


def test_unknown_install_component_fails_closed(tmp_path: Path) -> None:
    from vibeocr.backend.runtime_selection import RuntimeSelectionError
    from vibeocr.runtime_contracts import ErrorCode

    manifest, component = _release(tmp_path / "release")
    with pytest.raises(RuntimeSelectionError) as excinfo:
        RuntimeInstaller(
            product_root=tmp_path / "product",
            component_lock=component,
            runtime_manifest=manifest,
            accelerator="cpu",
            install_runner=_fake_install,
            install_component_ids=("not-a-component",),
        )
    assert excinfo.value.code is ErrorCode.RUNTIME_COMPONENT_UNKNOWN

    # gpu_runtime 只属于 nvidia_cuda 档位：component selection 不得隐式换档。
    with pytest.raises(RuntimeSelectionError):
        RuntimeInstaller(
            product_root=tmp_path / "product",
            component_lock=component,
            runtime_manifest=manifest,
            accelerator="cpu",
            install_runner=_fake_install,
            install_component_ids=("gpu_runtime",),
        )


def test_maintenance_snapshot_echoes_download_source_intent(
    tmp_path: Path,
) -> None:
    manifest, component = _release(tmp_path / "release")
    explicit = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=_fake_install,
        download_source_ids=("pypi",),
        install_component_ids=(),
    )
    explicit.ensure()
    snapshot = explicit.maintenance_snapshot()
    assert snapshot["requested_download_source_ids"] == ["pypi"]
    assert snapshot["effective_download_source_ids"] == ["pypi"]

    omitted = RuntimeInstaller(
        product_root=tmp_path / "product2",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_runner=_fake_install,
        install_component_ids=(),
    )
    omitted.ensure()
    snapshot = omitted.maintenance_snapshot()
    # 省略：requested 不出现，effective 解析为 Backend 缺省源。
    assert "requested_download_source_ids" not in snapshot
    assert snapshot["effective_download_source_ids"] == ["tuna-pypi"]


def test_base_only_ensure_probes_with_base_binding_not_plan_binding(
    tmp_path: Path,
) -> None:
    """base-only 安装的漂移探测必须用 base profile 的 import 绑定。

    回归：accelerator=cpu + 显式空安装范围时，已安装 ocr_engine 的绑定是
    RapidOCR；旧实现按 cpu plan 绑定 PaddleOCR 探测，会把刚装好的 base
    闭包整体判为漂移，ensure 最终以 "did not verify" 失败。
    """

    manifest, component = _release(tmp_path / "release")
    probe_profiles: list[str] = []

    def probe(
        _runtime_root: Path, component_ids: tuple[str, ...], profile_id: str
    ) -> dict[str, bool]:
        probe_profiles.append(profile_id)
        # 只按“绑定是否来自 base profile”判定：base 绑定（rapidocr）已装，
        # cpu plan 绑定（paddleocr）在 base-only 安装中不存在。
        return {
            component_id: profile_id == "win-x64-base" for component_id in component_ids
        }

    installer = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_component_ids=(),
        install_runner=_fake_install,
        component_probe=probe,
        operation_id="base-only-ensure",
    )

    launch = installer.ensure()

    assert launch is not None
    assert Path(launch.python_executable).is_file()
    assert probe_profiles
    # 漂移探测（非展示 payload）必须以 base 覆盖 profile 探测
    assert "win-x64-base" in probe_profiles
    assert installer._drifted_component_ids() == ()
    # 幂等复跑：base 闭包 ready，不触发重装
    assert installer.ensure() is not None


def test_full_scope_drift_still_probes_with_plan_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """full（含可选组件）安装的漂移探测仍使用 accelerator 的 plan 绑定。"""

    monkeypatch.setattr(RuntimeInstaller, "_acquire_models", _skip_model_acquisition)

    manifest, component = _release(tmp_path / "release")
    probe_profiles: list[str] = []

    def probe(
        _runtime_root: Path, component_ids: tuple[str, ...], profile_id: str
    ) -> dict[str, bool]:
        probe_profiles.append(profile_id)
        return dict.fromkeys(component_ids, True)

    installer = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        install_component_ids=("document_parsing",),
        install_runner=_fake_install,
        component_probe=probe,
        operation_id="full-ensure",
    )
    installer.ensure()

    assert installer._drifted_component_ids() == ()
    assert "win-x64-cpu" in probe_profiles


def test_drift_projection_uses_covering_profile_declared_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 单测没有真实 Python runtime：组件 import 探测统一视为通过，
    # 使断言聚焦在“声明版本/分布绑定按哪个 profile 投影”。
    monkeypatch.setattr(
        runtime_maintenance,
        "_cached_runtime_component_probe",
        lambda _root, component_ids, **_kwargs: dict.fromkeys(component_ids, True),
    )
    """base-only 安装的版本比对必须用 base 声明（rapidocr），不是 cpu plan。

    回归（v0.12.2 现场）：真实 manifest 的组件带声明版本；漂移投影仍按
    accelerator 的 plan descriptor 取 distribution（cpu 的 ocr_engine 绑定
    paddleocr），刚装好的 rapidocr 闭包被判 missing/version_mismatch，ensure
    以 "did not verify" 失败。``runtime_profile_status(profile_id=...)`` 与
    ``_drifted_component_ids`` 必须按已安装 scope 的覆盖 profile 投影。
    """

    manifest_path, component = _release(tmp_path / "release")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    # 声明组件必须与 loader 派生的稳定 id 顺序一致，仅注入版本差异
    document["profiles"]["win-x64-base"]["components"] = [
        {
            "component_id": "ocr_engine",
            "display_name": "Default offline OCR engine",
            "version": "3.9.2",
        },
        {
            "component_id": "pdf_document_tools",
            "display_name": "PDF and document tools",
        },
        {
            "component_id": "image_code_tools",
            "display_name": "Image, QR, and barcode tools",
        },
        {"component_id": "runtime_host", "display_name": "Runtime HTTP host"},
    ]
    document["profiles"]["win-x64-cpu"]["components"] = [
        {
            "component_id": "ocr_engine",
            "display_name": "OCR engine",
            "version": "3.7.0",
        },
        {"component_id": "document_parsing", "display_name": "Document parsing"},
        {
            "component_id": "pdf_document_tools",
            "display_name": "PDF and document tools",
        },
        {
            "component_id": "image_code_tools",
            "display_name": "Image, QR, and barcode tools",
        },
        {"component_id": "runtime_host", "display_name": "Runtime HTTP host"},
    ]
    manifest_path.write_text(
        json.dumps(document, sort_keys=True) + "\n", encoding="utf-8"
    )
    # 组件锁的 manifest 摘要必须跟随改写后的字节，保持绑定校验成立
    lock_document = json.loads(component.read_text(encoding="utf-8"))
    lock_document["backend"]["runtime_manifest_sha256"] = _sha(
        manifest_path.read_bytes()
    )
    component.write_text(
        json.dumps(lock_document, sort_keys=True) + "\n", encoding="utf-8"
    )
    loaded = load_runtime_manifest(manifest_path)

    runtime_root = tmp_path / "product" / "runtime"
    dist_info = runtime_root / "Lib" / "site-packages" / "rapidocr-3.9.2.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: rapidocr\nVersion: 3.9.2\n",
        encoding="utf-8",
    )
    (runtime_root / "python.exe").write_bytes(b"python")
    (runtime_root / ".installed.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend_version": loaded.backend_version,
                "manifest_sha256": loaded.sha256,
                "accelerator": "cpu",
                "component_ids": [
                    "ocr_engine",
                    "pdf_document_tools",
                    "image_code_tools",
                    "runtime_host",
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    base_view = runtime_profile_status(
        loaded,
        accelerator="cpu",
        runtime_root=runtime_root,
        profile_id="win-x64-base",
    )
    cpu_view = runtime_profile_status(
        loaded,
        accelerator="cpu",
        runtime_root=runtime_root,
    )
    base_states = {
        entry["component_id"]: entry["actual_state"]
        for entry in base_view["components"]
    }
    cpu_states = {
        entry["component_id"]: entry["actual_state"] for entry in cpu_view["components"]
    }

    assert base_states["ocr_engine"] == "ready"
    assert base_states["pdf_document_tools"] == "ready"
    # 默认 cpu 投影下同一安装是 missing（paddleocr 分布不存在）——证明
    # profile_id 覆盖确实改变了版本比对来源，而不是恒等于 plan。
    assert cpu_states["ocr_engine"] == "missing"

    installer = RuntimeInstaller(
        product_root=tmp_path / "product",
        component_lock=component,
        runtime_manifest=manifest_path,
        accelerator="cpu",
        install_component_ids=(),
        install_runner=_fake_install,
        component_probe=lambda _root, ids, _profile: dict.fromkeys(ids, True),
    )
    assert installer._drifted_component_ids() == ()


def test_document_parsing_ensure_acquires_models_with_selected_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """模型 acquisition 纳入 durable ensure,并把源映射进推理环境。"""
    from vibeocr.backend import model_registry as model_registry_module
    from vibeocr.backend import runtime_installer as installer_module

    manifest, component = _release(tmp_path / "release")

    def fake_run(command, *, timeout, env, reporter, heartbeat_code):  # type: ignore[no-untyped-def]
        return None

    def fake_extract_python(archive_path, destination, *, progress=None):  # type: ignore[no-untyped-def]
        (destination / "python.exe").write_bytes(b"python")

    monkeypatch.setattr(installer_module, "_run_install_command", fake_run)
    monkeypatch.setattr(
        installer_module, "_extract_python_archive", fake_extract_python
    )

    acquired: list[dict[str, object]] = []

    def fake_acquire(**kwargs: object) -> ResolvedModelSet:
        acquired.append(kwargs)
        (asset,) = kwargs["assets"]  # type: ignore[unreachable]
        target = kwargs["models_root"] / asset.engine / asset.target_dirname  # type: ignore[operator]
        target.mkdir(parents=True, exist_ok=True)
        (target / asset.files[0]).write_bytes(b"model")  # type: ignore[index]
        model_registry_module._write_ready_marker(  # noqa: SLF001
            target,
            asset,
            "0.7.0-" + "0" * 40,
        )
        return ResolvedModelSet(
            "0.7.0-" + "0" * 40,
            (
                ResolvedModel(
                    key=f"{asset.engine}/{asset.name}",
                    consumer=asset.consumer,
                    binding_key=asset.binding_key,
                    root=target,
                ),
            ),
        )

    monkeypatch.setattr(installer_module, "acquire_models", fake_acquire)

    missing_manifest = RuntimeInstaller(
        product_root=tmp_path / "product-missing-manifest",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        component_probe=lambda _root, ids, _profile_id: dict.fromkeys(ids, True),
        install_component_ids=("document_parsing",),
        download_source_ids=("tuna-pypi", "huggingface"),
        operation_id="document-model-ensure",
    )
    with pytest.raises(
        ModelAcquisitionError,
        match="document_parsing requires a model assets manifest",
    ):
        missing_manifest.ensure()

    product = tmp_path / "product"
    installer = RuntimeInstaller(
        product_root=product,
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        component_probe=lambda _root, ids, _profile_id: dict.fromkeys(ids, True),
        install_component_ids=("document_parsing",),
        download_source_ids=("tuna-pypi", "huggingface"),
        operation_id="document-model-ensure",
    )
    assets_file = product / "state" / "config" / "model-assets.json"
    assets_file.parent.mkdir(parents=True, exist_ok=True)
    assets_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_identity": "0.7.0-" + "0" * 40,
                "assets": [
                    {
                        "engine": "paddleocr",
                        "name": "PP-OCRv5-server",
                        "repository": "PaddlePaddle/PP-OCRv5-server",
                        "revision": "v1",
                        "files": [
                            {
                                "path": "inference.pdmodel",
                                "size": 5,
                                "sha256": "0" * 64,
                            }
                        ],
                        "consumer": "paddleocr",
                        "binding_key": "text_recognition_model_dir",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    launch = installer.ensure()

    assert launch is not None
    assert len(acquired) == 1
    assert acquired[0]["source_id"] == "huggingface"
    assert acquired[0]["endpoint"] == "https://huggingface.co"
    assert launch.environment["PADDLE_PDX_MODEL_SOURCE"] == "huggingface"
    assert launch.environment["MINERU_MODEL_SOURCE"] == "huggingface"
    assert Path(launch.environment["VIBEOCR_RESOLVED_MODELS"]).is_file()
    mineru_config = Path(launch.environment["MINERU_TOOLS_CONFIG_JSON"])
    assert mineru_config.is_file()
    assert json.loads(mineru_config.read_text(encoding="utf-8"))["models-dir"]

    def unexpected_acquire(**_kwargs: object) -> ResolvedModelSet:
        raise AssertionError("replayed ensure must load the persisted binding")

    monkeypatch.setattr(installer_module, "acquire_models", unexpected_acquire)
    replayed = RuntimeInstaller(
        product_root=product,
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        component_probe=lambda _root, ids, _profile_id: dict.fromkeys(ids, True),
        install_component_ids=("document_parsing",),
        download_source_ids=("tuna-pypi", "huggingface"),
        operation_id="document-model-ensure",
    ).ensure()
    assert replayed is not None
    assert (
        replayed.environment["VIBEOCR_RESOLVED_MODELS"]
        == launch.environment["VIBEOCR_RESOLVED_MODELS"]
    )

    # base-only 闭包不触发模型 acquisition(无 document_parsing)。
    monkeypatch.setattr(installer_module, "acquire_models", fake_acquire)
    acquired.clear()
    base_installer = RuntimeInstaller(
        product_root=tmp_path / "product-base",
        component_lock=component,
        runtime_manifest=manifest,
        accelerator="cpu",
        component_probe=lambda _root, ids, _profile_id: dict.fromkeys(ids, True),
        install_component_ids=(),
        download_source_ids=("tuna-pypi", "huggingface"),
    )
    base_launch = base_installer.ensure()
    assert base_launch is not None
    assert acquired == []
