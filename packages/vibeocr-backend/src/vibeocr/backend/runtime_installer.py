"""Backend-owned portable runtime installer.

The public surface is intentionally small: ``inspect``, ``ensure`` and ``repair``.
Frontends receive paths and integrity state,
never dependency names, index URLs or pip arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vibeocr.backend.runtime_layout import resolve_runtime_store
from vibeocr.backend.runtime_lock import RuntimeLockTimeout, RuntimeStoreLock
from vibeocr.backend.runtime_manifest import (
    ManifestError,
    RuntimeManifest,
    load_runtime_manifest,
)


class RuntimeInstallError(RuntimeError):
    pass


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


PROTOCOL_VERSION = 2
ACCELERATOR_TO_PLAN = {
    "cpu": "win-x64-cpu",
    "nvidia_cuda": "win-x64-cu126",
}


@dataclass(frozen=True, slots=True)
class RuntimeState:
    status: str
    runtime_root: str
    accelerator: str
    manifest_sha256: str
    backend_version: str
    integrity: str


@dataclass(frozen=True, slots=True)
class RuntimeLaunch:
    python_executable: str
    supervisor_module: str
    working_directory: str
    model_root: str
    environment: dict[str, str]


InstallRunner = Callable[[Path, RuntimeManifest, str], Path]


def _default_accelerator(component_lock: dict[str, Any]) -> str:
    accelerator = component_lock["backend"].get("accelerator")
    if accelerator not in ACCELERATOR_TO_PLAN:
        raise RuntimeInstallError(f"unsupported accelerator: {accelerator}")
    return accelerator


def _load_component_lock(path: Path) -> dict[str, Any]:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeInstallError(f"invalid component lock: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise RuntimeInstallError("component lock schema_version must be 1")
    protocol = value.get("protocol")
    backend = value.get("backend")
    if not isinstance(protocol, dict) or not isinstance(backend, dict):
        raise RuntimeInstallError("component lock requires protocol and backend")
    required = {
        "repository",
        "version",
        "artifact_sha256",
        "runtime_manifest_sha256",
        "accelerator",
    }
    if not required.issubset(backend):
        raise RuntimeInstallError("component lock backend binding is incomplete")
    if not {
        "repository",
        "version",
        "manifest_sha256",
    }.issubset(protocol):
        raise RuntimeInstallError("component lock Protocol binding is incomplete")
    if protocol["repository"] != "FelixJI/vibeocr-protocol":
        raise RuntimeInstallError("component lock Protocol repository is invalid")
    if backend["repository"] != "FelixJI/vibeocr-backend":
        raise RuntimeInstallError("component lock Backend repository is invalid")
    if not re.fullmatch(r"2\.\d+\.\d+", str(protocol["version"])):
        raise RuntimeInstallError("component lock Protocol version is invalid")
    if not re.fullmatch(r"\d+\.\d+\.\d+", str(backend["version"])):
        raise RuntimeInstallError("component lock Backend version is invalid")
    for record, field in (
        (protocol, "manifest_sha256"),
        (backend, "artifact_sha256"),
        (backend, "runtime_manifest_sha256"),
    ):
        if not _SHA256_RE.fullmatch(str(record[field])):
            raise RuntimeInstallError(f"component lock {field} is invalid")
    if backend.get("accelerator") not in ACCELERATOR_TO_PLAN:
        raise RuntimeInstallError("component lock Backend accelerator is invalid")
    capabilities = value.get("required_capabilities")
    if not isinstance(capabilities, list) or not all(
        isinstance(item, str) and item for item in capabilities
    ):
        raise RuntimeInstallError("component lock required_capabilities is invalid")
    return value


def _python_in(runtime_root: Path) -> Path:
    candidates = (
        runtime_root / "Scripts" / "python.exe",
        runtime_root / "python.exe",
        runtime_root / "bin" / "python",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def _default_install_runner(
    partial_root: Path,
    manifest: RuntimeManifest,
    profile: str,
) -> Path:
    """Extract bound Python and install only hash-locked dependencies.

    The installer may itself be a frozen EXE, so it must never depend on
    ``sys.executable -m venv``. Protocol and Backend wheels are installed from
    the manifest directory; no editable/local workspace source is accepted.
    """
    _extract_python_archive(manifest.python.archive_path, partial_root)
    python = _python_in(partial_root)
    if not python.is_file():
        raise RuntimeInstallError("Python archive has no python.exe")
    lock = manifest.profiles[profile].lock_path
    portable_env = os.environ.copy()
    cache = partial_root.parent.parent / "state" / "installer-cache"
    portable_env.update(
        {
            "PIP_CACHE_DIR": str(cache / "pip"),
            "UV_CACHE_DIR": str(cache / "uv"),
            "HF_HOME": str(cache / "huggingface"),
            "MODELSCOPE_CACHE": str(cache / "modelscope"),
            "TEMP": str(cache / "temp"),
            "TMP": str(cache / "temp"),
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
        }
    )
    for directory in portable_env["PIP_CACHE_DIR"], portable_env["TEMP"]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "--only-binary=:all:",
            "-r",
            str(lock),
        ],
        check=True,
        timeout=3600,
        env=portable_env,
    )
    artifact_root = manifest.path.parent
    protocol_wheel = artifact_root / manifest.protocol_wheel
    backend_wheel = artifact_root / manifest.backend_wheel
    if not protocol_wheel.is_file() or not backend_wheel.is_file():
        raise RuntimeInstallError(
            "release directory must contain the bound Protocol and Backend wheels"
        )
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            str(protocol_wheel),
            str(backend_wheel),
        ],
        check=True,
        timeout=600,
        env=portable_env,
    )
    subprocess.run(
        [
            str(python),
            "-c",
            (
                "import vibeocr.runtime_contracts, vibeocr.backend; "
                "import vibeocr.backend.supervisor.main"
            ),
        ],
        check=True,
        timeout=60,
        env=portable_env,
    )
    return python


def _extract_python_archive(archive_path: Path, destination: Path) -> None:
    """Safely strip the archive's ``python/`` prefix into the partial root."""
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise RuntimeInstallError("Python archive is empty")
        for member in members:
            parts = Path(member.name.replace("\\", "/")).parts
            if not parts or parts[0] != "python" or ".." in parts:
                raise RuntimeInstallError(
                    f"unsafe Python archive member: {member.name}"
                )
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeInstallError(
                    f"unsupported Python archive member: {member.name}"
                )
            relative = Path(*parts[1:])
            if not relative.parts:
                continue
            if relative.is_absolute() or any(":" in part for part in relative.parts):
                raise RuntimeInstallError(
                    f"unsafe Python archive member: {member.name}"
                )
            target = destination / relative
            try:
                target.resolve().relative_to(destination.resolve())
            except ValueError as exc:
                raise RuntimeInstallError(
                    f"unsafe Python archive member: {member.name}"
                ) from exc
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeInstallError(
                    f"unsupported Python archive member: {member.name}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeInstallError(
                    f"unreadable Python archive member: {member.name}"
                )
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


class RuntimeInstaller:
    def __init__(
        self,
        *,
        product_root: str | Path,
        component_lock: str | Path,
        runtime_manifest: str | Path,
        accelerator: str | None = None,
        layout_manifest: str | Path | None = None,
        product_id: str | None = None,
        install_runner: InstallRunner | None = None,
        lock_timeout: float = 60.0,
    ) -> None:
        self.product_root = Path(product_root).resolve()
        self.component_lock_path = Path(component_lock).resolve()
        self.manifest = load_runtime_manifest(runtime_manifest)
        self.component_lock = _load_component_lock(self.component_lock_path)
        self.paths = resolve_runtime_store(
            self.product_root,
            manifest_sha256=self.manifest.sha256,
            layout_manifest=layout_manifest,
            product_id=product_id,
        )
        self.accelerator = self._select_accelerator(accelerator)
        self.plan = ACCELERATOR_TO_PLAN[self.accelerator]
        self._install_runner = install_runner or _default_install_runner
        self._lock_timeout = lock_timeout
        self._validate_binding()

    def _preference_path(self) -> Path:
        return self.paths.state_root / "runtime-preference.json"

    def _select_accelerator(self, requested: str | None) -> str:
        if requested is not None:
            if requested not in ACCELERATOR_TO_PLAN:
                raise RuntimeInstallError(f"unsupported accelerator: {requested}")
            return requested
        try:
            value = json.loads(self._preference_path().read_text(encoding="utf-8"))
            preferred = (
                value.get("accelerator") if value.get("schema_version") == 1 else None
            )
        except (OSError, ValueError, AttributeError):
            preferred = None
        return (
            preferred
            if preferred in ACCELERATOR_TO_PLAN
            else _default_accelerator(self.component_lock)
        )

    def _save_preference(self) -> None:
        path = self._preference_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"schema_version": 1, "accelerator": self.accelerator}, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )

    def _validate_binding(self) -> None:
        protocol = self.component_lock["protocol"]
        backend = self.component_lock["backend"]
        protocol_version = protocol.get("version")
        if (
            not isinstance(protocol_version, str)
            or f"-{protocol_version}-" not in self.manifest.protocol_wheel
        ):
            raise RuntimeInstallError("component lock Protocol version mismatch")
        if protocol["manifest_sha256"] != self.manifest.protocol_manifest_sha256:
            raise RuntimeInstallError("component lock Protocol manifest mismatch")
        if backend["version"] != self.manifest.backend_version:
            raise RuntimeInstallError("component lock Backend version mismatch")
        if backend["artifact_sha256"] != self.manifest.backend_sha256:
            raise RuntimeInstallError("component lock Backend artifact mismatch")
        if backend["runtime_manifest_sha256"] != self.manifest.sha256:
            raise RuntimeInstallError("component lock runtime manifest mismatch")
        required = set(self.component_lock["required_capabilities"])
        missing = required.difference(self.manifest.capabilities)
        if missing:
            raise RuntimeInstallError(
                f"Backend is missing required capabilities: {sorted(missing)}"
            )

    def _marker(self) -> Path:
        return self.paths.runtime_root / ".installed.json"

    def _integrity_ok(self) -> bool:
        marker = self._marker()
        python = _python_in(self.paths.runtime_root)
        if not marker.is_file() or not python.is_file():
            return False
        try:
            value = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return value == {
            "schema_version": 1,
            "backend_version": self.manifest.backend_version,
            "manifest_sha256": self.manifest.sha256,
            "accelerator": self.accelerator,
        }

    def inspect(self) -> RuntimeState:
        ready = self._integrity_ok()
        return RuntimeState(
            status="ready" if ready else "missing",
            runtime_root=str(self.paths.runtime_root),
            accelerator=self.accelerator,
            manifest_sha256=self.manifest.sha256,
            backend_version=self.manifest.backend_version,
            integrity="verified" if ready else "not-installed",
        )

    def _environment(self) -> dict[str, str]:
        state = self.paths.state_root
        return {
            "VIBEOCR_PRODUCT_ROOT": str(self.product_root),
            "VIBEOCR_RUNTIME_ROOT": str(self.paths.runtime_root),
            "VIBEOCR_MODEL_ROOT": str(self.paths.models_root),
            "PIP_CACHE_DIR": str(state / "cache" / "pip"),
            "UV_CACHE_DIR": str(state / "cache" / "uv"),
            "HF_HOME": str(state / "cache" / "huggingface"),
            "MODELSCOPE_CACHE": str(state / "cache" / "modelscope"),
            "TEMP": str(state / "temp"),
            "TMP": str(state / "temp"),
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
        }

    def ensure(self) -> RuntimeLaunch:
        if not self._integrity_ok():
            lock = RuntimeStoreLock(
                self.paths.locks_root / "runtime-store.lock",
                timeout=self._lock_timeout,
            )
            with lock:
                if not self._integrity_ok():
                    self._install_locked()
        self._save_preference()
        return self._launch()

    def _install_locked(self) -> None:
        final = self.paths.runtime_root
        final.parent.mkdir(parents=True, exist_ok=True)
        partial = final.with_name("runtime.installing")
        if partial.exists():
            shutil.rmtree(partial)
        if final.exists():
            shutil.rmtree(final)
        partial.mkdir(parents=True)
        try:
            python = self._install_runner(partial, self.manifest, self.plan)
            try:
                python.relative_to(partial)
            except ValueError as exc:
                raise RuntimeInstallError(
                    "installer returned Python outside partial runtime"
                ) from exc
            if not python.is_file():
                raise RuntimeInstallError("installed runtime has no Python executable")
            marker = {
                "schema_version": 1,
                "backend_version": self.manifest.backend_version,
                "manifest_sha256": self.manifest.sha256,
                "accelerator": self.accelerator,
            }
            (partial / ".installed.json").write_text(
                json.dumps(marker, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            partial.replace(final)
        except Exception:
            shutil.rmtree(partial, ignore_errors=True)
            raise

    def repair(self) -> RuntimeLaunch:
        lock = RuntimeStoreLock(
            self.paths.locks_root / "runtime-store.lock",
            timeout=self._lock_timeout,
        )
        with lock:
            self._install_locked()
        self._save_preference()
        return self._launch()

    def _launch(self) -> RuntimeLaunch:
        if not self._integrity_ok():
            raise RuntimeInstallError("runtime installation did not verify")
        environment = self._environment()
        for directory in environment.values():
            path = Path(directory)
            if path.is_absolute() and directory.startswith(str(self.paths.store_root)):
                path.mkdir(parents=True, exist_ok=True)
        return RuntimeLaunch(
            python_executable=str(_python_in(self.paths.runtime_root)),
            supervisor_module="vibeocr.backend.supervisor.main",
            working_directory=str(self.product_root),
            model_root=str(self.paths.models_root),
            environment=environment,
        )

    def acquire_lease(self, *, timeout: float = 0.0) -> RuntimeStoreLock:
        lease = RuntimeStoreLock(
            self.paths.locks_root / "leases" / "runtime.lock",
            timeout=timeout,
        )
        lease.acquire()
        return lease


def _emit(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _request(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeInstallError("Runtime Host request must be a JSON object")
    required = {
        "protocol_version",
        "operation",
        "product_root",
        "component_lock",
        "runtime_manifest",
    }
    if set(value).difference(
        required | {"accelerator", "layout_manifest", "product_id"}
    ):
        raise RuntimeInstallError("Runtime Host request contains unknown fields")
    if not required.issubset(value) or value["protocol_version"] != PROTOCOL_VERSION:
        raise RuntimeInstallError(
            "Runtime Host request is incompatible with Protocol v2"
        )
    if value["operation"] not in ("inspect", "ensure", "repair"):
        raise RuntimeInstallError("Runtime Host operation is invalid")
    for field in ("product_root", "component_lock", "runtime_manifest"):
        if not isinstance(value[field], str) or not value[field]:
            raise RuntimeInstallError(f"Runtime Host {field} is invalid")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vibeocr-runtime-installer")
    parser.add_argument("--request-json")
    args = parser.parse_args(argv)
    operation: str | None = None
    try:
        raw_request = (
            args.request_json if args.request_json is not None else sys.stdin.read()
        )
        request = _request(json.loads(raw_request))
        operation = request["operation"]
        installer = RuntimeInstaller(
            product_root=request["product_root"],
            component_lock=request["component_lock"],
            runtime_manifest=request["runtime_manifest"],
            accelerator=request.get("accelerator"),
            layout_manifest=request.get("layout_manifest"),
            product_id=request.get("product_id"),
        )
        if operation == "inspect":
            state = installer.inspect()
            launch = None
        else:
            launch = getattr(installer, operation)()
            state = installer.inspect()
    except (
        json.JSONDecodeError,
        ManifestError,
        RuntimeInstallError,
        RuntimeLockTimeout,
        OSError,
    ) as exc:
        code = (
            "lock_timeout"
            if isinstance(exc, RuntimeLockTimeout)
            else "io_error"
            if isinstance(exc, OSError)
            else "invalid_request"
            if isinstance(exc, (json.JSONDecodeError, RuntimeInstallError))
            and operation is None
            else "invalid_binding"
            if isinstance(exc, ManifestError)
            else "install_failed"
        )
        _emit(
            {
                "protocol_version": PROTOCOL_VERSION,
                "ok": False,
                "operation": operation,
                "error": {
                    "code": code,
                    "message": str(exc),
                    "retryable": code in {"lock_timeout", "io_error"},
                },
            }
        )
        return 1
    _emit(
        {
            "protocol_version": PROTOCOL_VERSION,
            "ok": True,
            "operation": operation,
            "state": asdict(state),
            "launch": asdict(launch) if launch is not None else None,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RuntimeState",
    "RuntimeInstallError",
    "RuntimeInstaller",
    "RuntimeLaunch",
    "main",
]
