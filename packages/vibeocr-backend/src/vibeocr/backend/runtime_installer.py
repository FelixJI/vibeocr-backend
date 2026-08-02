"""Backend-owned portable runtime installer.

The public surface is intentionally small: ``inspect``, ``ensure``, ``repair``
and ``gc``.  Frontends receive paths and integrity state, never dependency
names, index URLs or pip arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vibeocr.backend.runtime_layout import resolve_runtime_store, runtime_id_prefix
from vibeocr.backend.runtime_lock import RuntimeLockTimeout, RuntimeStoreLock
from vibeocr.backend.runtime_manifest import (
    ManifestError,
    RuntimeManifest,
    load_runtime_manifest,
)


class RuntimeInstallError(RuntimeError):
    pass


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class RuntimeInspection:
    status: str
    runtime_id: str
    profile: str
    runtime_root: str
    manifest_sha256: str
    backend_version: str
    integrity: str


@dataclass(frozen=True, slots=True)
class RuntimeLaunch:
    runtime_id: str
    profile: str
    python_executable: str
    supervisor_module: str
    working_directory: str
    model_root: str
    environment: dict[str, str]


InstallRunner = Callable[[Path, RuntimeManifest, str], Path]


def _select_profile(requested: str | None, component_lock: dict[str, Any]) -> str:
    if requested in (None, "auto"):
        requested = component_lock["backend"].get("profile", "win-x64-cpu")
    if requested not in ("win-x64-cpu", "win-x64-cu126"):
        raise RuntimeInstallError(f"unsupported runtime profile: {requested}")
    return requested


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
    profile = backend.get("profile")
    if profile is not None and profile not in ("win-x64-cpu", "win-x64-cu126"):
        raise RuntimeInstallError("component lock Backend profile is invalid")
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
    cache = partial_root.parent.parent.parent / "state" / "installer-cache"
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
        profile: str | None = None,
        layout_manifest: str | Path | None = None,
        product_id: str | None = None,
        install_runner: InstallRunner | None = None,
        lock_timeout: float = 60.0,
    ) -> None:
        self.product_root = Path(product_root).resolve()
        self.component_lock_path = Path(component_lock).resolve()
        self.manifest = load_runtime_manifest(runtime_manifest)
        self.component_lock = _load_component_lock(self.component_lock_path)
        self.profile = _select_profile(profile, self.component_lock)
        self.paths = resolve_runtime_store(
            self.product_root,
            manifest_sha256=self.manifest.sha256,
            profile=self.profile,
            layout_manifest=layout_manifest,
            product_id=product_id,
        )
        self._install_runner = install_runner or _default_install_runner
        self._lock_timeout = lock_timeout
        self._validate_binding()

    @property
    def runtime_id(self) -> str:
        # 用 6 位前缀而非完整哈希，与物理目录名一致（见 runtime_id_prefix）。
        return f"{runtime_id_prefix(self.manifest.sha256)}/{self.profile}"

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
            "runtime_id": self.runtime_id,
            "backend_version": self.manifest.backend_version,
            "profile": self.profile,
        }

    def inspect(self) -> RuntimeInspection:
        ready = self._integrity_ok()
        return RuntimeInspection(
            status="ready" if ready else "missing",
            runtime_id=self.runtime_id,
            profile=self.profile,
            runtime_root=str(self.paths.runtime_root),
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
        return self._launch()

    def _install_locked(self) -> None:
        final = self.paths.runtime_root
        final.parent.mkdir(parents=True, exist_ok=True)
        partial = final.with_name(f"{final.name}.partial-{uuid.uuid4().hex}")
        partial.mkdir(parents=True)
        try:
            python = self._install_runner(partial, self.manifest, self.profile)
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
                "runtime_id": self.runtime_id,
                "backend_version": self.manifest.backend_version,
                "profile": self.profile,
            }
            (partial / ".installed.json").write_text(
                json.dumps(marker, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if final.exists():
                if self._integrity_ok():
                    shutil.rmtree(partial)
                    return
                raise RuntimeInstallError("invalid runtime already occupies final path")
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
            if self.paths.runtime_root.exists() and not self._integrity_ok():
                quarantine = self.paths.runtime_root.with_name(
                    f"{self.paths.runtime_root.name}.invalid-{uuid.uuid4().hex}"
                )
                self.paths.runtime_root.replace(quarantine)
            if not self._integrity_ok():
                self._install_locked()
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
            runtime_id=self.runtime_id,
            profile=self.profile,
            python_executable=str(_python_in(self.paths.runtime_root)),
            supervisor_module="vibeocr.backend.supervisor.main",
            working_directory=str(self.product_root),
            model_root=str(self.paths.models_root),
            environment=environment,
        )

    def acquire_lease(self, *, timeout: float = 0.0) -> RuntimeStoreLock:
        name = self.runtime_id.replace("/", "-")
        lease = RuntimeStoreLock(
            self.paths.locks_root / "leases" / f"{name}.lock",
            timeout=timeout,
        )
        lease.acquire()
        return lease

    def gc(
        self,
        *,
        component_locks: Iterable[str | Path],
        grace_seconds: float = 7 * 86400,
    ) -> list[str]:
        """Remove unreferenced runtimes, failing closed on any bad lock."""
        referenced: set[str] = set()
        try:
            all_locks = {
                self.component_lock_path,
                *(Path(item).resolve() for item in component_locks),
            }
            for lock_path in all_locks:
                value = _load_component_lock(Path(lock_path))
                # lock 存的是完整 64 位哈希（密码学校验用），GC 比对时取同样的
                # 6 位前缀，与物理目录名 / runtime_id 保持一致。
                digest = runtime_id_prefix(value["backend"]["runtime_manifest_sha256"])
                profile = value["backend"].get("profile")
                if profile is None:
                    referenced.update(
                        f"{digest}/{item}" for item in self.manifest.profiles
                    )
                elif profile in self.manifest.profiles:
                    referenced.add(f"{digest}/{profile}")
                else:
                    raise RuntimeInstallError("component lock profile is invalid")
        except RuntimeInstallError:
            return []

        removed: list[str] = []
        now = time.time()
        with RuntimeStoreLock(
            self.paths.locks_root / "runtime-store.lock",
            timeout=self._lock_timeout,
        ):
            if not self.paths.runtimes_root.is_dir():
                return []
            for digest_root in self.paths.runtimes_root.iterdir():
                if not digest_root.is_dir() or not digest_root.name.isalnum():
                    continue
                for profile_root in digest_root.iterdir():
                    runtime_id = f"{digest_root.name}/{profile_root.name}"
                    if runtime_id in referenced or not profile_root.is_dir():
                        continue
                    if now - profile_root.stat().st_mtime < grace_seconds:
                        continue
                    lease = RuntimeStoreLock(
                        self.paths.locks_root
                        / "leases"
                        / f"{runtime_id.replace('/', '-')}.lock",
                        timeout=0,
                    )
                    try:
                        lease.acquire()
                    except RuntimeLockTimeout:
                        continue
                    try:
                        shutil.rmtree(profile_root)
                        removed.append(runtime_id)
                    finally:
                        lease.release()
        return removed


def _emit(value: object) -> None:
    print(json.dumps(asdict(value), ensure_ascii=False, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vibeocr-runtime-installer")
    parser.add_argument("operation", choices=("inspect", "ensure", "repair", "gc"))
    parser.add_argument("--product-root", required=True)
    parser.add_argument("--component-lock", required=True)
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument(
        "--profile",
        choices=("auto", "win-x64-cpu", "win-x64-cu126"),
        default="auto",
    )
    parser.add_argument("--layout-manifest")
    parser.add_argument("--product-id")
    parser.add_argument(
        "--component-locks",
        nargs="*",
        help="registered component locks used by fail-closed garbage collection",
    )
    parser.add_argument(
        "--referenced-component-lock",
        action="append",
        dest="referenced_component_locks",
        help="repeatable cross-language alias for one registered component lock",
    )
    parser.add_argument("--grace-seconds", type=float, default=7 * 86400)
    parser.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        installer = RuntimeInstaller(
            product_root=args.product_root,
            component_lock=args.component_lock,
            runtime_manifest=args.runtime_manifest,
            profile=args.profile,
            layout_manifest=args.layout_manifest,
            product_id=args.product_id,
        )
        if args.operation == "gc":
            locks = (
                args.referenced_component_locks
                or args.component_locks
                or [args.component_lock]
            )
            result = installer.gc(
                component_locks=locks,
                grace_seconds=args.grace_seconds,
            )
        else:
            result = getattr(installer, args.operation)()
    except (ManifestError, RuntimeInstallError, RuntimeLockTimeout, OSError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}))
        return 1
    if isinstance(result, list):
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RuntimeInspection",
    "RuntimeInstallError",
    "RuntimeInstaller",
    "RuntimeLaunch",
    "main",
]
