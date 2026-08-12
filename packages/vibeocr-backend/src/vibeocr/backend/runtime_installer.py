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
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from vibeocr.backend.runtime_layout import resolve_runtime_store
from vibeocr.backend.runtime_lock import RuntimeLockTimeout, RuntimeStoreLock
from vibeocr.backend.runtime_maintenance import (
    EventSink,
    RuntimeCapabilityError,
    RuntimeCommandConflict,
    RuntimeCursorExpired,
    RuntimeInstallFailure,
    RuntimeMaintenanceReporter,
    RuntimeOperationCancelled,
    RuntimeOperationConflict,
    RuntimeOperationError,
    RuntimeOperationNotCancellable,
    RuntimeOperationNotFound,
    RuntimeOperationNotRetryable,
    RuntimeSourceIdentityMismatch,
    probe_runtime_components,
    profile_descriptor,
    runtime_profile_status,
    runtime_source_identity,
)
from vibeocr.backend.runtime_manifest import (
    ManifestError,
    RuntimeManifest,
    load_runtime_manifest,
)


class RuntimeInstallError(RuntimeInstallFailure):
    pass


class RuntimeIdentityMismatch(RuntimeInstallError, RuntimeSourceIdentityMismatch):
    """The verified Runtime source identity differs from the bound intent."""


class RuntimeCapabilityUnavailable(RuntimeInstallError, RuntimeCapabilityError):
    """A requested capability is absent from the verified Runtime manifest."""


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
    source: dict[str, str]


@dataclass(frozen=True, slots=True)
class RuntimeInspection:
    state: RuntimeState
    profile: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RuntimeLaunch:
    python_executable: str
    supervisor_module: str
    working_directory: str
    model_root: str
    environment: dict[str, str]


InstallRunner = Callable[[Path, RuntimeManifest, str], Path]
ComponentProbe = Callable[[Path, tuple[str, ...]], dict[str, bool]]


def _default_component_probe(
    runtime_root: Path, component_ids: tuple[str, ...]
) -> dict[str, bool]:
    return probe_runtime_components(runtime_root, component_ids)


def _run_install_command(
    command: list[str],
    *,
    timeout: float,
    env: dict[str, str],
    reporter: RuntimeMaintenanceReporter | None,
    heartbeat_code: str,
) -> None:
    """Run a child without leaking package-manager output into NDJSON stdout."""
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.kill()
                process.communicate()
                raise RuntimeInstallError(f"{heartbeat_code} timed out")
            try:
                process.communicate(timeout=min(5.0, remaining))
                break
            except subprocess.TimeoutExpired:
                if reporter is not None:
                    reporter.heartbeat(message_code=heartbeat_code)
        if process.returncode != 0:
            raise RuntimeInstallError(
                f"{heartbeat_code} failed with exit code {process.returncode}"
            )
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


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
    reporter: RuntimeMaintenanceReporter | None = None,
) -> Path:
    """Extract bound Python and install only hash-locked dependencies.

    The installer may itself be a frozen EXE, so it must never depend on
    ``sys.executable -m venv``. Protocol and Backend wheels are installed from
    the manifest directory; no editable/local workspace source is accepted.
    """
    progress: Callable[[int, int], None] | None = None
    if reporter is not None:

        def report_extraction(current: int, total: int) -> None:
            reporter.advance_measured(
                phase="prepare_runtime",
                unit="bytes",
                current=current,
                total=total,
                message_code="runtime.extract_python",
                component_id="runtime_base",
            )

        progress = report_extraction
    _extract_python_archive(
        manifest.python.archive_path,
        partial_root,
        progress=progress,
    )
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
    if reporter is not None:
        reporter.advance(
            phase="install_profile",
            current=4,
            total=7,
            message_code="runtime.install_profile",
            component_id="ocr_engine",
        )
    _run_install_command(
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
        timeout=3600,
        env=portable_env,
        reporter=reporter,
        heartbeat_code="runtime.install_profile",
    )
    artifact_root = manifest.path.parent
    protocol_wheel = artifact_root / manifest.protocol_wheel
    backend_wheel = artifact_root / manifest.backend_wheel
    if not protocol_wheel.is_file() or not backend_wheel.is_file():
        raise RuntimeInstallError(
            "release directory must contain the bound Protocol and Backend wheels"
        )
    if reporter is not None:
        reporter.advance(
            phase="install_backend",
            current=5,
            total=7,
            message_code="runtime.install_backend",
            component_id="runtime_host",
        )
    _run_install_command(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            str(protocol_wheel),
            str(backend_wheel),
        ],
        timeout=600,
        env=portable_env,
        reporter=reporter,
        heartbeat_code="runtime.install_backend",
    )
    if reporter is not None:
        reporter.advance(
            phase="verify_runtime",
            current=6,
            total=7,
            message_code="runtime.verify_runtime",
        )
    _run_install_command(
        [
            str(python),
            "-c",
            (
                "import vibeocr.runtime_contracts, vibeocr.backend; "
                "import vibeocr.backend.supervisor.main"
            ),
        ],
        timeout=60,
        env=portable_env,
        reporter=reporter,
        heartbeat_code="runtime.verify_runtime",
    )
    return python


def _extract_python_archive(
    archive_path: Path,
    destination: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Safely strip the archive's ``python/`` prefix into the partial root."""
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise RuntimeInstallError("Python archive is empty")
        total_bytes = sum(member.size for member in members if member.isfile())
        extracted_bytes = 0
        reported_percent = -1
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
            extracted_bytes += member.size
            if progress is not None and total_bytes > 0:
                percent = min(100, extracted_bytes * 100 // total_bytes)
                if percent != reported_percent:
                    progress(extracted_bytes, total_bytes)
                    reported_percent = percent


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
        component_probe: ComponentProbe | None = None,
        event_sink: EventSink | None = None,
        lock_timeout: float = 60.0,
        operation_id: str | None = None,
        source_operation_id: str | None = None,
        component_ids: tuple[str, ...] = (),
        required_capabilities: tuple[str, ...] = (),
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
        available_components = {
            component.component_id
            for component in self.manifest.profiles[self.plan].components
        }
        if len(set(component_ids)) != len(component_ids) or not set(
            component_ids
        ).issubset(available_components):
            raise RuntimeInstallError("Runtime component_ids are invalid")
        missing_capabilities = set(required_capabilities).difference(
            self.manifest.capabilities
        )
        if missing_capabilities:
            raise RuntimeCapabilityUnavailable(
                f"required capabilities are unavailable: {sorted(missing_capabilities)}"
            )
        self._operation_id = operation_id
        self._source_operation_id = source_operation_id
        self._component_ids = component_ids
        self._required_capabilities = required_capabilities
        self._source = runtime_source_identity(self.manifest)
        self._runner_reports_phases = install_runner is None
        self._component_probe = component_probe or (
            _default_component_probe
            if install_runner is None
            else lambda _root, component_ids: {
                component_id: True for component_id in component_ids
            }
        )
        if install_runner is None:
            self._install_runner = lambda partial, manifest, profile: (
                _default_install_runner(
                    partial,
                    manifest,
                    profile,
                    self._reporter,
                )
            )
        else:
            self._install_runner = install_runner
        self._lock_timeout = lock_timeout
        self._validate_binding()
        self._reporter = RuntimeMaintenanceReporter(
            state_root=self.paths.state_root,
            profile=profile_descriptor(
                self.manifest.profiles[self.plan],
                accelerator=self.accelerator,
            ),
            event_sink=event_sink,
        )

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
            raise RuntimeIdentityMismatch("component lock Protocol version mismatch")
        if protocol["manifest_sha256"] != self.manifest.protocol_manifest_sha256:
            raise RuntimeIdentityMismatch("component lock Protocol manifest mismatch")
        if backend["version"] != self.manifest.backend_version:
            raise RuntimeIdentityMismatch("component lock Backend version mismatch")
        if backend["artifact_sha256"] != self.manifest.backend_sha256:
            raise RuntimeIdentityMismatch("component lock Backend artifact mismatch")
        if backend["runtime_manifest_sha256"] != self.manifest.sha256:
            raise RuntimeIdentityMismatch("component lock runtime manifest mismatch")
        required = set(self.component_lock["required_capabilities"])
        missing = required.difference(self.manifest.capabilities)
        if missing:
            raise RuntimeCapabilityUnavailable(
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

    def profile_payload(self) -> dict[str, Any]:
        component_ids = self._profile_component_ids()
        return runtime_profile_status(
            self.manifest,
            accelerator=self.accelerator,
            runtime_root=self.paths.runtime_root,
            probe_results=self._component_probe(self.paths.runtime_root, component_ids),
        )

    def maintenance_snapshot(self) -> dict[str, Any] | None:
        return self._reporter.snapshot

    def _profile_component_ids(self) -> tuple[str, ...]:
        return tuple(
            component.component_id
            for component in self.manifest.profiles[self.plan].components
        )

    def _drifted_component_ids(self) -> tuple[str, ...]:
        return self._drifted_component_ids_from(self.profile_payload())

    @staticmethod
    def _drifted_component_ids_from(profile: dict[str, Any]) -> tuple[str, ...]:
        return tuple(
            str(component["component_id"])
            for component in profile["components"]
            if component.get("actual_state") != "ready"
        )

    def _start_operation(
        self, operation: str, *, effective_component_ids: tuple[str, ...] = ()
    ) -> bool:
        return self._reporter.start(
            operation,
            total_steps=7 if operation != "inspect" else 2,
            operation_id=self._operation_id,
            component_ids=self._component_ids,
            effective_component_ids=effective_component_ids,
            source=self._source,
            source_operation_id=self._source_operation_id,
            required_capabilities=self._required_capabilities,
        )

    def inspect_snapshot(self, *, emit: bool = True) -> RuntimeInspection:
        started = self._start_operation("inspect") if emit else False
        profile = self.profile_payload()
        ready = self._integrity_ok() and not self._drifted_component_ids_from(profile)
        state = RuntimeState(
            status="ready" if ready else "missing",
            runtime_root=str(self.paths.runtime_root),
            accelerator=self.accelerator,
            manifest_sha256=self.manifest.sha256,
            backend_version=self.manifest.backend_version,
            integrity="verified" if ready else "not-installed",
            source=self._source,
        )
        if emit and started:
            self._reporter.succeed(
                phase="verify_runtime",
                current=2,
                total=2,
                message_code="runtime.inspect_complete",
            )
        return RuntimeInspection(state=state, profile=profile)

    def inspect(self, *, emit: bool = True) -> RuntimeState:
        return self.inspect_snapshot(emit=emit).state

    def _environment(self) -> dict[str, str]:
        state = self.paths.state_root
        return {
            "VIBEOCR_PRODUCT_ROOT": str(self.product_root),
            "VIBEOCR_RUNTIME_ROOT": str(self.paths.runtime_root),
            "VIBEOCR_MODEL_ROOT": str(self.paths.models_root),
            "VIBEOCR_RUNTIME_MANIFEST": str(self.manifest.path),
            "VIBEOCR_COMPONENT_LOCK": str(self.component_lock_path),
            "VIBEOCR_RUNTIME_ACCELERATOR": self.accelerator,
            # Backend inference/cache consumers still read this compatibility
            # flag.  Derive it from the Installer-owned profile instead of
            # asking a frontend shell to infer or inject the Runtime device.
            "VIBEOCR_USE_GPU": (
                "true" if self.accelerator == "nvidia_cuda" else "false"
            ),
            "VIBEOCR_RUNTIME_STATE_ROOT": str(state),
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

    def ensure(self) -> RuntimeLaunch | None:
        ready = self._integrity_ok() and not self._drifted_component_ids()
        started = self._start_operation(
            "ensure",
            effective_component_ids=() if ready else self._profile_component_ids(),
        )
        if not started:
            return self._launch() if ready else None
        try:
            if not ready:
                self._reporter.advance(
                    phase="wait_for_lock",
                    current=2,
                    total=7,
                    message_code="runtime.wait_for_lock",
                )
                lock = RuntimeStoreLock(
                    self.paths.locks_root / "runtime-store.lock",
                    timeout=self._lock_timeout,
                )
                with lock:
                    if not self._integrity_ok() or self._drifted_component_ids():
                        self._install_locked()
            else:
                self._reporter.advance(
                    phase="verify_runtime",
                    current=6,
                    total=7,
                    message_code="runtime.verify_runtime",
                )
            self._save_preference()
            launch = self._launch()
            self._reporter.succeed(
                phase="commit_runtime",
                current=7,
                total=7,
                message_code="runtime.ensure_complete",
            )
            return launch
        except RuntimeOperationCancelled:
            raise
        except Exception as exc:
            self._reporter.fail(exc)
            raise

    def _install_locked(self) -> None:
        self._reporter.advance(
            phase="prepare_runtime",
            current=3,
            total=7,
            message_code="runtime.prepare_runtime",
        )
        final = self.paths.runtime_root
        final.parent.mkdir(parents=True, exist_ok=True)
        partial = final.with_name("runtime.installing")
        rollback = final.with_name("runtime.rollback")
        if rollback.exists():
            if final.exists():
                shutil.rmtree(rollback)
            else:
                rollback.replace(final)
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        try:
            if not self._runner_reports_phases:
                self._reporter.advance(
                    phase="install_profile",
                    current=4,
                    total=7,
                    message_code="runtime.install_profile",
                    component_id="ocr_engine",
                )
            python = self._install_runner(partial, self.manifest, self.plan)
            if not self._runner_reports_phases:
                self._reporter.advance(
                    phase="verify_runtime",
                    current=6,
                    total=7,
                    message_code="runtime.verify_runtime",
                )
            try:
                python.relative_to(partial)
            except ValueError as exc:
                raise RuntimeInstallError(
                    "installer returned Python outside partial runtime"
                ) from exc
            if not python.is_file():
                raise RuntimeInstallError("installed runtime has no Python executable")
            probe_results = self._component_probe(
                partial, self._profile_component_ids()
            )
            failed_components = sorted(
                component_id
                for component_id, ready in probe_results.items()
                if not ready
            )
            if failed_components:
                raise RuntimeInstallError(
                    f"installed Runtime component imports failed: {failed_components}"
                )
            (partial / ".component-integrity.json").write_text(
                json.dumps(probe_results, sort_keys=True) + "\n",
                encoding="utf-8",
            )
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
            self._reporter.advance(
                phase="commit_runtime",
                current=7,
                total=7,
                message_code="runtime.commit_runtime",
            )
            if final.exists():
                final.replace(rollback)
            try:
                partial.replace(final)
            except Exception:
                if rollback.exists() and not final.exists():
                    rollback.replace(final)
                raise
            shutil.rmtree(rollback, ignore_errors=True)
        except RuntimeOperationCancelled:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        except Exception:
            shutil.rmtree(partial, ignore_errors=True)
            raise

    def repair(self) -> RuntimeLaunch | None:
        drifted = set(self._drifted_component_ids())
        requested = set(self._component_ids) if self._component_ids else drifted
        needs_repair = bool(requested.intersection(drifted))
        globally_ready = self._integrity_ok() and not drifted
        effective = () if not needs_repair else self._profile_component_ids()
        started = self._start_operation(
            "repair",
            effective_component_ids=effective,
        )
        if not started:
            return self._launch() if globally_ready else None
        try:
            if not needs_repair:
                if globally_ready:
                    self._save_preference()
                launch = self._launch() if globally_ready else None
                self._reporter.succeed(
                    phase="commit_runtime",
                    current=7,
                    total=7,
                    message_code="runtime.repair_complete",
                )
                return launch
            self._reporter.advance(
                phase="wait_for_lock",
                current=2,
                total=7,
                message_code="runtime.wait_for_lock",
            )
            lock = RuntimeStoreLock(
                self.paths.locks_root / "runtime-store.lock",
                timeout=self._lock_timeout,
            )
            with lock:
                self._install_locked()
            self._save_preference()
            launch = self._launch()
            self._reporter.succeed(
                phase="commit_runtime",
                current=7,
                total=7,
                message_code="runtime.repair_complete",
            )
            return launch
        except RuntimeOperationCancelled:
            raise
        except Exception as exc:
            self._reporter.fail(exc)
            raise

    def _launch(self) -> RuntimeLaunch:
        if not self._integrity_ok() or self._drifted_component_ids():
            raise RuntimeInstallError("runtime installation did not verify")
        environment = self._environment()
        directory_keys = {
            "VIBEOCR_PRODUCT_ROOT",
            "VIBEOCR_RUNTIME_ROOT",
            "VIBEOCR_MODEL_ROOT",
            "VIBEOCR_RUNTIME_STATE_ROOT",
            "PIP_CACHE_DIR",
            "UV_CACHE_DIR",
            "HF_HOME",
            "MODELSCOPE_CACHE",
            "TEMP",
            "TMP",
        }
        for key in directory_keys:
            directory = environment[key]
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
    binding_fields = {
        "protocol_version",
        "product_root",
        "component_lock",
        "runtime_manifest",
    }
    common_optional = {
        "request_kind",
        "accepted_event_streams",
    }
    request_kind = value.get("request_kind", "start")
    if request_kind == "command":
        required = binding_fields | {"command", "command_id", "target_operation_id"}
        allowed = (
            required
            | common_optional
            | {
                "new_operation_id",
                "expected_sequence",
                "accelerator",
                "layout_manifest",
                "product_id",
            }
        )
    elif request_kind == "observe":
        required = binding_fields | {"operation_id", "after_sequence"}
        allowed = (
            required
            | common_optional
            | {
                "limit",
                "accelerator",
                "layout_manifest",
                "product_id",
            }
        )
    elif request_kind == "start":
        required = binding_fields | {"operation"}
        allowed = (
            required
            | common_optional
            | {
                "accelerator",
                "layout_manifest",
                "product_id",
                "operation_id",
                "component_ids",
                "required_capabilities",
            }
        )
    else:
        raise RuntimeInstallError("Runtime Host request_kind is invalid")
    if set(value).difference(allowed):
        raise RuntimeInstallError("Runtime Host request contains unknown fields")
    if not required.issubset(value) or value["protocol_version"] != PROTOCOL_VERSION:
        raise RuntimeInstallError(
            "Runtime Host request is incompatible with Protocol v2"
        )
    if request_kind == "start" and value["operation"] not in (
        "inspect",
        "ensure",
        "repair",
    ):
        raise RuntimeInstallError("Runtime Host operation is invalid")
    for field in ("product_root", "component_lock", "runtime_manifest"):
        if not isinstance(value[field], str) or not value[field]:
            raise RuntimeInstallError(f"Runtime Host {field} is invalid")
    accepted_streams = value.get("accepted_event_streams", [])
    if (
        not isinstance(accepted_streams, list)
        or any(not isinstance(stream, str) for stream in accepted_streams)
        or len(set(accepted_streams)) != len(accepted_streams)
        or any(stream not in {"ndjson.v1", "ndjson.v2"} for stream in accepted_streams)
    ):
        raise RuntimeInstallError("Runtime Host accepted_event_streams is invalid")
    for field in ("component_ids", "required_capabilities"):
        items = value.get(field, [])
        if (
            not isinstance(items, list)
            or any(not isinstance(item, str) or not item for item in items)
            or len(set(items)) != len(items)
        ):
            raise RuntimeInstallError(f"Runtime Host {field} is invalid")
    for field in (
        "operation_id",
        "command_id",
        "target_operation_id",
        "new_operation_id",
    ):
        if field in value and (not isinstance(value[field], str) or not value[field]):
            raise RuntimeInstallError(f"Runtime Host {field} is invalid")
    if request_kind == "command":
        if value["command"] not in {"cancel", "retry"}:
            raise RuntimeInstallError("Runtime Host command is invalid")
        if value["command"] == "retry" and not value.get("new_operation_id"):
            raise RuntimeInstallError("Runtime Host retry requires new_operation_id")
    if request_kind == "observe":
        if type(value["after_sequence"]) is not int or value["after_sequence"] < 0:
            raise RuntimeInstallError("Runtime Host after_sequence is invalid")
        limit = value.get("limit", 128)
        if type(limit) is not int or limit < 1 or limit > 512:
            raise RuntimeInstallError("Runtime Host limit is invalid")
    return value


def _installer_from_request(
    request: dict[str, Any],
    *,
    event_sink: EventSink | None = None,
    operation_id: str | None = None,
    source_operation_id: str | None = None,
    component_ids: tuple[str, ...] | None = None,
    required_capabilities: tuple[str, ...] | None = None,
) -> RuntimeInstaller:
    return RuntimeInstaller(
        product_root=request["product_root"],
        component_lock=request["component_lock"],
        runtime_manifest=request["runtime_manifest"],
        accelerator=request.get("accelerator"),
        layout_manifest=request.get("layout_manifest"),
        product_id=request.get("product_id"),
        event_sink=event_sink,
        operation_id=operation_id
        if operation_id is not None
        else request.get("operation_id"),
        source_operation_id=source_operation_id,
        component_ids=(
            component_ids
            if component_ids is not None
            else tuple(request.get("component_ids", []))
        ),
        required_capabilities=(
            required_capabilities
            if required_capabilities is not None
            else tuple(request.get("required_capabilities", []))
        ),
    )


def _success_envelope(
    result: Any,
    *,
    operation: str,
) -> dict[str, Any]:
    capability_registry = json.loads(
        files("vibeocr.runtime_contracts")
        .joinpath("capabilities.json")
        .read_text(encoding="utf-8")
    )
    definitions = capability_registry["definitions"]
    capability_descriptors = [
        {
            "name": name,
            "lifecycle": definitions[name]["lifecycle"],
            "introduced_in": definitions[name]["introduced_in"],
            "deprecated_in": definitions[name]["deprecated_in"],
            "sunset_at": definitions[name]["sunset_at"],
            "replacement": definitions[name]["replacement"],
        }
        for name in result.available_capabilities
        if name in definitions
    ]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "ok": True,
        "operation": operation,
        "state": asdict(result.state),
        "launch": asdict(result.launch) if result.launch is not None else None,
        "profile": result.profile,
        "maintenance": result.receipt["snapshot"],
        "negotiated_capabilities": list(
            result.receipt.get("negotiated_capabilities", [])
        ),
        "capability_descriptors": capability_descriptors,
    }


def _runtime_control_from_request(
    request: dict[str, Any],
    *,
    event_sink: EventSink | None,
) -> Any:
    from vibeocr.backend.runtime_control import RuntimeControl

    def installer_factory(**kwargs: Any) -> RuntimeInstaller:
        return _installer_from_request(request, event_sink=event_sink, **kwargs)

    return RuntimeControl.from_installer_factory(installer_factory)


def _command_envelope(
    request: dict[str, Any],
    control: Any,
) -> dict[str, Any]:
    receipt = control.command(
        command_id=request["command_id"],
        command=request["command"],
        target_operation_id=request["target_operation_id"],
        new_operation_id=request.get("new_operation_id"),
        expected_sequence=request.get("expected_sequence"),
    )
    snapshot = receipt["snapshot"]
    operation = str(snapshot["operation"])
    include_launch = (
        request["command"] == "retry"
        and snapshot.get("operation_state") == "succeeded"
        and operation != "inspect"
    )
    return _success_envelope(
        control.project_receipt(receipt, include_launch=include_launch),
        operation=operation,
    )


def _failure_envelope(
    exc: Exception,
    *,
    operation: str | None,
    maintenance: dict[str, Any] | None,
) -> dict[str, Any]:
    legacy_code = "install_failed"
    canonical_code = "RUNTIME_INSTALL_FAILED"
    category = "backend_unavailable"
    retryable = False
    detail: dict[str, Any] = {}
    if isinstance(exc, RuntimeLockTimeout):
        legacy_code = "lock_timeout"
        canonical_code = "RUNTIME_BUSY"
        category = "transient"
        retryable = True
    elif isinstance(exc, OSError):
        legacy_code = "io_error"
        canonical_code = "RUNTIME_IO_ERROR"
        category = "transient"
        retryable = True
    elif isinstance(exc, RuntimeOperationNotFound):
        legacy_code = "invalid_request"
        canonical_code = "RUNTIME_OPERATION_NOT_FOUND"
        category = "not_found"
    elif isinstance(exc, RuntimeCursorExpired):
        legacy_code = "invalid_request"
        canonical_code = "RUNTIME_CURSOR_EXPIRED"
        category = "not_found"
        detail["oldest_sequence"] = exc.oldest_sequence
    elif isinstance(exc, RuntimeCommandConflict):
        legacy_code = "invalid_request"
        canonical_code = "RUNTIME_COMMAND_ID_CONFLICT"
        category = "conflict"
    elif isinstance(exc, RuntimeOperationNotCancellable):
        legacy_code = "invalid_request"
        canonical_code = "RUNTIME_OPERATION_NOT_CANCELLABLE"
        category = "conflict"
    elif isinstance(exc, RuntimeOperationNotRetryable):
        legacy_code = "invalid_request"
        canonical_code = "RUNTIME_OPERATION_NOT_RETRYABLE"
        category = "conflict"
    elif isinstance(exc, RuntimeOperationConflict):
        legacy_code = "invalid_request"
        canonical_code = "RUNTIME_OPERATION_ID_CONFLICT"
        category = "conflict"
    elif isinstance(exc, RuntimeCapabilityError):
        legacy_code = "invalid_request"
        canonical_code = "RUNTIME_CAPABILITY_UNAVAILABLE"
        category = "capability"
    elif isinstance(exc, RuntimeSourceIdentityMismatch):
        legacy_code = "invalid_binding"
        canonical_code = "RUNTIME_IDENTITY_MISMATCH"
        category = "identity"
    elif isinstance(exc, RuntimeOperationCancelled):
        canonical_code = "CANCELLED"
        category = "cancelled"
    elif isinstance(exc, ManifestError):
        legacy_code = "invalid_binding"
        canonical_code = "RUNTIME_IDENTITY_MISMATCH"
        category = "identity"
    elif (
        isinstance(exc, (json.JSONDecodeError, RuntimeInstallFailure))
        and operation is None
    ):
        legacy_code = "invalid_request"
        canonical_code = "VALIDATION_ERROR"
        category = "validation"
    error: dict[str, Any] = {
        "code": legacy_code,
        "canonical_code": canonical_code,
        "category": category,
        "message": str(exc),
        "message_code": "runtime.operation_failed",
        "retryable": retryable,
        "detail": detail,
    }
    if retryable:
        error["retry_after"] = 1
    envelope: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "ok": False,
        "operation": operation,
        "error": error,
    }
    if maintenance is not None:
        envelope["maintenance"] = maintenance
    if isinstance(exc, RuntimeCursorExpired):
        envelope["maintenance"] = exc.snapshot
    return envelope


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vibeocr-runtime-installer")
    parser.add_argument("--request-json")
    args = parser.parse_args(argv)
    operation: str | None = None
    control: Any | None = None
    try:
        raw_request = (
            args.request_json if args.request_json is not None else sys.stdin.read()
        )
        request = _request(json.loads(raw_request))
        request_kind = request.get("request_kind", "start")
        operation = request.get("operation")
        event_sink = (
            _emit
            if {"ndjson.v1", "ndjson.v2"}.intersection(
                request.get("accepted_event_streams", [])
            )
            else None
        )
        control = _runtime_control_from_request(
            request,
            event_sink=event_sink,
        )
        if request_kind == "observe":
            update = control.observe(
                request["operation_id"],
                after_sequence=request["after_sequence"],
                limit=request.get("limit", 128),
            )
            _emit(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "ok": True,
                    "request_kind": "observe",
                    **{
                        key: value
                        for key, value in update.items()
                        if key != "schema_version"
                    },
                }
            )
            return 0
        if request_kind == "command":
            _emit(_command_envelope(request, control))
            return 0
        assert operation is not None
        result = control.execute_with_result(
            operation=operation,
            operation_id=request.get("operation_id"),
            component_ids=tuple(request.get("component_ids", [])),
            required_capabilities=tuple(request.get("required_capabilities", [])),
            profile_id=request.get("profile_id"),
        )
    except (
        json.JSONDecodeError,
        ManifestError,
        RuntimeInstallError,
        RuntimeLockTimeout,
        RuntimeOperationError,
        OSError,
    ) as exc:
        _emit(
            _failure_envelope(
                exc,
                operation=operation,
                maintenance=(
                    control.maintenance_snapshot if control is not None else None
                ),
            )
        )
        return 1
    _emit(
        _success_envelope(
            result,
            operation=operation,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RuntimeState",
    "RuntimeCapabilityUnavailable",
    "RuntimeInstallError",
    "RuntimeInstaller",
    "RuntimeLaunch",
    "main",
]
