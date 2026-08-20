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
import zipfile
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
    ACCELERATOR_TO_PLAN,
    ManifestError,
    RuntimeInstallScope,
    RuntimeManifest,
    load_runtime_manifest,
)
from vibeocr.backend.runtime_selection import (
    BASE_PROFILE,
    BoundDownloadSource,
    RuntimeSelectionError,
    RuntimeSelectionPolicy,
)


class RuntimeInstallError(RuntimeInstallFailure):
    pass


class RuntimeIdentityMismatch(RuntimeInstallError, RuntimeSourceIdentityMismatch):
    """The verified Runtime source identity differs from the bound intent."""


class RuntimeCapabilityUnavailable(RuntimeInstallError, RuntimeCapabilityError):
    """A requested capability is absent from the verified Runtime manifest."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


PROTOCOL_VERSION = 2


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
ComponentProbe = Callable[[Path, tuple[str, ...], str], dict[str, bool]]


def _default_component_probe(
    runtime_root: Path, component_ids: tuple[str, ...], profile_id: str
) -> dict[str, bool]:
    return probe_runtime_components(runtime_root, component_ids, profile_id=profile_id)


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
    install_scope: RuntimeInstallScope,
    download_sources: tuple[BoundDownloadSource, ...],
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
    lock = install_scope.lock_path
    runtime_pack = install_scope.runtime_pack
    portable_env = os.environ.copy()
    for name in tuple(portable_env):
        if name.upper().startswith(("PIP_", "UV_")):
            portable_env.pop(name)
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
    install_command = [
        str(python),
        "-m",
        "pip",
        "install",
        "--only-binary=:all:",
    ]
    pack_files = [manifest.path.parent / name for name in runtime_pack]
    pack_present = bool(pack_files) and all(path.is_file() for path in pack_files)
    base_ids = {
        component.component_id
        for component in manifest.profiles[BASE_PROFILE].components
    }
    is_base_scope = set(install_scope.component_ids) == base_ids
    if runtime_pack and not pack_present and is_base_scope:
        # base 是随 Portable 携带的必备闭包：缺失即产品不完整，fail closed
        # 而不是静默联网安装（计划 §4.1）。full 闭包的 pack 由用户按需下载，
        # 未到位时回退在线安装是合法状态（cu126 因 torch 单 wheel 超过
        # Release 资产上限，长期保持在线直链路径）。
        raise RuntimeInstallError(f"runtime pack is missing: {runtime_pack[0]}")
    if pack_present:
        # 离线路径（计划 §4.2）：manifest 绑定的 runtime pack 提供完整
        # wheel 闭包，安装禁止回退公网。pack 整体字节完整性由
        # runtime_pack_sha256 在 manifest 加载时验证；解析输入用 pack 自带
        # 的无哈希 requirements 清单——原 lock 的哈希行覆盖 sdist 工件，
        # 对 pack 内由 sdist 构建的 wheel 必然不匹配。
        pack_dir = _extract_runtime_pack(
            pack_files,
            cache / "runtime-packs",
            reporter=reporter,
        )
        requirements_file = pack_dir / "pack-requirements.txt"
        install_command += [
            "--no-index",
            "--find-links",
            str(pack_dir),
            "-r",
            str(requirements_file),
        ]
    else:
        package_indexes = tuple(
            source for source in download_sources if source.kind == "package_index"
        )
        if len(package_indexes) != 1:
            raise RuntimeInstallError(
                "online Runtime install requires one package_index source"
            )
        install_command += [
            "--index-url",
            package_indexes[0].endpoint,
            "--require-hashes",
            "-r",
            str(lock),
        ]
    _run_install_command(
        install_command,
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


def _extract_runtime_pack(
    archive_paths: list[Path],
    cache_root: Path,
    *,
    reporter: RuntimeMaintenanceReporter | None = None,
) -> Path:
    """Idempotently extract manifest-bound runtime pack parts into the cache.

    A pack is one or more flat zips of wheels; the first part also carries
    the hash-free requirements manifest (``pack-requirements.txt``). Each
    archive's SHA-256 has already been verified by ``load_runtime_manifest``;
    extraction is guarded against unsafe members and only re-runs when the
    destination marker is missing, so repeated ensure/repair installs stay
    offline and cheap.
    """
    if not archive_paths:
        raise RuntimeInstallError("runtime pack binding is empty")
    # 分片名形如 <pack>.part01.zip:缓存目录按去掉分片后缀的公共 stem。
    stem = re.sub(r"\.part\d+$", "", archive_paths[0].stem)
    destination = cache_root / stem
    marker = destination / ".complete"
    if marker.is_file():
        return destination
    for archive_path in archive_paths:
        if not archive_path.is_file():
            raise RuntimeInstallError(f"runtime pack is missing: {archive_path.name}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    try:
        for archive_path in archive_paths:
            with zipfile.ZipFile(archive_path) as archive:
                for name in archive.namelist():
                    parts = Path(name.replace("\\", "/")).parts
                    if (
                        not parts
                        or ".." in parts
                        or Path(name).is_absolute()
                        or len(parts) != 1
                        or not (
                            name.endswith(".whl") or name == "pack-requirements.txt"
                        )
                    ):
                        raise RuntimeInstallError(f"unsafe runtime pack member: {name}")
                    archive.extract(name, destination)
    except zipfile.BadZipFile as exc:
        raise RuntimeInstallError("runtime pack archive is invalid") from exc
    if not (destination / "pack-requirements.txt").is_file():
        raise RuntimeInstallError("runtime pack lacks pack-requirements.txt")
    marker.write_text("ok\n", encoding="utf-8")
    if reporter is not None:
        reporter.advance(
            phase="install_profile",
            current=4,
            total=7,
            message_code="runtime.extract_pack",
            component_id="ocr_engine",
        )
    return destination


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
        install_component_ids: tuple[str, ...] | None = None,
        download_source_ids: tuple[str, ...] | None = None,
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
        self._selection = RuntimeSelectionPolicy.from_manifest(
            self.manifest
        ).plan_start(
            accelerator=self.accelerator,
            install_component_ids=install_component_ids,
            download_source_ids=download_source_ids,
        )
        self._install_scope = self._selection.requested_component_ids
        self._requested_download_source_ids = (
            self._selection.requested_download_source_ids or ()
        )
        self._effective_download_source_ids = (
            self._selection.effective_download_source_ids
        )
        self._active_install_ids = self._selection.effective_component_ids
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
        if component_probe is not None:
            self._component_probe = component_probe
        elif install_runner is None:
            self._component_probe = _default_component_probe
        else:
            self._component_probe = lambda _root, component_ids, _profile_id: {
                component_id: True for component_id in component_ids
            }
        if install_runner is None:
            self._install_runner = lambda partial, manifest, profile: (
                _default_install_runner(
                    partial,
                    manifest,
                    self._scope_for_ids(self._active_install_ids),
                    self._selection.effective_download_sources,
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

    def _marker_value(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self._marker().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def _integrity_ok(self) -> bool:
        python = _python_in(self.paths.runtime_root)
        if not self._marker().is_file() or not python.is_file():
            return False
        return self._trusted_installed_scope_ids() is not None

    def _trusted_installed_scope_ids(self) -> tuple[str, ...] | None:
        """Read a bound, manifest-declared installed closure from the marker."""
        value = self._marker_value()
        if value is None or value.get("schema_version") != 1:
            return None
        if (
            value.get("backend_version") != self.manifest.backend_version
            or value.get("manifest_sha256") != self.manifest.sha256
            or value.get("accelerator") != self.accelerator
        ):
            return None
        installed = value.get("component_ids")
        if (
            not isinstance(installed, list)
            or not installed
            or any(not isinstance(item, str) for item in installed)
            or len(set(installed)) != len(installed)
            or not set(installed).issubset(self._profile_component_ids())
        ):
            return None
        try:
            scope = self._scope_for_ids(tuple(installed))
        except RuntimeInstallError:
            return None
        return scope.component_ids

    def _installed_scope_ids(self) -> tuple[str, ...]:
        """已安装闭包由可信 marker 提供，不依赖当前运行时完整。"""
        installed = self._trusted_installed_scope_ids()
        if installed is not None:
            return installed
        return self._desired_scope_ids()

    def _desired_scope_ids(self) -> tuple[str, ...]:
        """本次 intent 解析后的精确依赖闭包。"""
        return self._selection.effective_component_ids

    def _scope_for_ids(self, component_ids: tuple[str, ...]) -> RuntimeInstallScope:
        desired = set(component_ids)
        for profile_name in (BASE_PROFILE, self.plan):
            for scope in self.manifest.profiles[profile_name].scopes:
                if set(scope.component_ids) == desired:
                    return scope
        raise RuntimeInstallError("runtime manifest has no matching install scope")

    def _covering_profile(self, component_ids: tuple[str, ...]) -> str:
        base_ids = {
            component.component_id
            for component in self.manifest.profiles[BASE_PROFILE].components
        }
        return BASE_PROFILE if set(component_ids) <= base_ids else self.plan

    def profile_payload(self) -> dict[str, Any]:
        component_ids = self._profile_component_ids()
        return runtime_profile_status(
            self.manifest,
            accelerator=self.accelerator,
            runtime_root=self.paths.runtime_root,
            probe_results=self._component_probe(
                self.paths.runtime_root, component_ids, self.plan
            ),
        )

    def maintenance_snapshot(self) -> dict[str, Any] | None:
        return self._reporter.snapshot

    def _profile_component_ids(self) -> tuple[str, ...]:
        return tuple(
            component.component_id
            for component in self.manifest.profiles[self.plan].components
        )

    def _drifted_component_ids(self) -> tuple[str, ...]:
        installed = self._installed_scope_ids()
        if not installed:
            return ()
        # 已安装闭包可能来自 base profile（base-only 的 ocr_engine 绑定是
        # RapidOCR，而 cpu plan 的同名组件绑定 PaddleOCR）。漂移探测的
        # 组件集、声明版本与 import 绑定都必须按已安装 scope 的覆盖
        # profile 投影；否则成功的 base-only 安装会被 plan descriptor 的
        # 版本比对判为漂移，ensure 的 launch 校验随之失败。
        covering = self._covering_profile(installed)
        payload = runtime_profile_status(
            self.manifest,
            accelerator=self.accelerator,
            runtime_root=self.paths.runtime_root,
            probe_results=self._component_probe(
                self.paths.runtime_root,
                installed,
                covering,
            ),
            profile_id=covering,
        )
        return self._scope_drifted_from(payload)

    def _scope_drifted_from(self, profile: dict[str, Any]) -> tuple[str, ...]:
        # 漂移只对已安装闭包判定：base-only 安装缺 full 组件不是漂移。
        scope = set(self._installed_scope_ids())
        return tuple(
            component_id
            for component_id in self._drifted_component_ids_from(profile)
            if component_id in scope
        )

    @staticmethod
    def _drifted_component_ids_from(profile: dict[str, Any]) -> tuple[str, ...]:
        return tuple(
            str(component["component_id"])
            for component in profile["components"]
            if component.get("actual_state") != "ready"
        )

    def _start_operation(
        self,
        operation: str,
        *,
        effective_component_ids: tuple[str, ...] = (),
    ) -> bool:
        # ensure 的 requested 回显可选组件安装范围（None=缺省省略、
        # []=显式 base-only）；inspect/repair 保持 component_ids 语义，
        # 空集仍省略。
        requested: tuple[str, ...] | None
        if operation == "ensure":
            requested = self._install_scope
        else:
            requested = self._component_ids or None
        return self._reporter.start(
            operation,
            total_steps=7 if operation != "inspect" else 2,
            operation_id=self._operation_id,
            component_ids=requested,
            effective_component_ids=effective_component_ids,
            install_component_ids=self._install_scope
            if operation == "ensure"
            else None,
            download_source_ids=self._effective_download_source_ids,
            requested_download_source_ids=self._requested_download_source_ids,
            source=self._source,
            source_operation_id=self._source_operation_id,
            required_capabilities=self._required_capabilities,
        )

    def inspect_snapshot(self, *, emit: bool = True) -> RuntimeInspection:
        started = self._start_operation("inspect") if emit else False
        profile = self.profile_payload()
        # inspect 诚实反映“已安装闭包是否 ready”，不把缺 full 可选组件
        # 当作失败（base-only 是合法的安装状态）。
        ready = self._integrity_ok() and not self._scope_drifted_from(profile)
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
        environment = {
            "VIBEOCR_PRODUCT_ROOT": str(self.product_root),
            "VIBEOCR_RUNTIME_ROOT": str(self.paths.runtime_root),
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
            "VIBEOCR_SUPERVISOR_SETTINGS": str(
                self.product_root / "state" / "supervisor-settings.json"
            ),
            "PIP_CACHE_DIR": str(state / "cache" / "pip"),
            "UV_CACHE_DIR": str(state / "cache" / "uv"),
            "HF_HOME": str(state / "cache" / "huggingface"),
            "MODELSCOPE_CACHE": str(state / "cache" / "modelscope"),
            "PADDLE_PDX_CACHE_HOME": str(state / "cache" / "paddlex"),
            "MINERU_TOOLS_CONFIG_JSON": str(state / "config" / "mineru.json"),
            "TEMP": str(state / "temp"),
            "TMP": str(state / "temp"),
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
        }
        package_indexes = tuple(
            source
            for source in self._selection.effective_download_sources
            if source.kind == "package_index"
        )
        if package_indexes:
            environment["PIP_INDEX_URL"] = package_indexes[0].endpoint
        environment.update(self._selection.model_source_environment())
        return environment

    def ensure(self) -> RuntimeLaunch | None:
        # ready 额外要求已安装闭包等于期望闭包：从 base-only 扩到 full
        # （或反向）都会触发一次重装，而不是静默沿用旧范围。
        ready = (
            self._integrity_ok()
            and self._installed_scope_ids() == self._desired_scope_ids()
            and not self._drifted_component_ids()
        )
        started = self._start_operation(
            "ensure",
            effective_component_ids=() if ready else self._desired_scope_ids(),
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
                    if (
                        not self._integrity_ok()
                        or self._installed_scope_ids() != self._desired_scope_ids()
                        or self._drifted_component_ids()
                    ):
                        self._install_locked(self._desired_scope_ids())
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

    def _install_locked(self, target_ids: tuple[str, ...]) -> None:
        profile_name = self._covering_profile(target_ids)
        self._active_install_ids = target_ids
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
            python = self._install_runner(partial, self.manifest, profile_name)
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
            probe_results = self._component_probe(partial, target_ids, profile_name)
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
                "component_ids": list(target_ids),
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
        if self._marker().is_file() and self._trusted_installed_scope_ids() is None:
            error = RuntimeInstallError("untrusted installed marker")
            if self._start_operation("repair"):
                self._reporter.fail(error)
            raise error
        drifted = set(self._drifted_component_ids())
        requested = set(self._component_ids) if self._component_ids else drifted
        needs_repair = bool(requested.intersection(drifted))
        globally_ready = self._integrity_ok() and not drifted
        # repair 只重建已安装闭包：不得把 base-only 运行时顺带升级成 full。
        effective = () if not needs_repair else self._installed_scope_ids()
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
                self._install_locked(self._installed_scope_ids())
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
            "VIBEOCR_RUNTIME_STATE_ROOT",
            "PIP_CACHE_DIR",
            "UV_CACHE_DIR",
            "HF_HOME",
            "MODELSCOPE_CACHE",
            "PADDLE_PDX_CACHE_HOME",
            "TEMP",
            "TMP",
        }
        for key in directory_keys:
            directory = environment[key]
            path = Path(directory)
            if path.is_absolute() and directory.startswith(str(self.paths.store_root)):
                path.mkdir(parents=True, exist_ok=True)
        Path(environment["MINERU_TOOLS_CONFIG_JSON"]).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        model_root = Path(environment["VIBEOCR_RUNTIME_STATE_ROOT"]) / "models"
        model_root.mkdir(parents=True, exist_ok=True)
        return RuntimeLaunch(
            python_executable=str(_python_in(self.paths.runtime_root)),
            supervisor_module="vibeocr.backend.supervisor.main",
            working_directory=str(self.product_root),
            model_root=str(model_root),
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


def _configure_utf8_standard_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


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
                "install_component_ids",
                "download_source_ids",
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
                "install_component_ids",
                "download_source_ids",
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
    for field in ("install_component_ids", "download_source_ids"):
        if field not in value:
            continue
        items = value[field]
        if (
            not isinstance(items, list)
            or any(not isinstance(item, str) or not item for item in items)
            or len(set(items)) != len(items)
            or (field == "download_source_ids" and not items)
        ):
            raise RuntimeInstallError(f"Runtime Host {field} is invalid")
    # 选择字段只对 ensure / retry 合法（与正式 wire schema 的条件约束一致）。
    if "install_component_ids" in value or "download_source_ids" in value:
        if request_kind == "start" and value["operation"] != "ensure":
            raise RuntimeInstallError(
                "Runtime Host selection fields require operation ensure"
            )
        if request_kind == "command" and value["command"] != "retry":
            raise RuntimeInstallError(
                "Runtime Host selection fields require command retry"
            )
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
    install_component_ids: tuple[str, ...] | None = None,
    download_source_ids: tuple[str, ...] | None = None,
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
        install_component_ids=(
            install_component_ids
            if install_component_ids is not None
            else (
                tuple(request["install_component_ids"])
                if "install_component_ids" in request
                else None
            )
        ),
        download_source_ids=(
            download_source_ids
            if download_source_ids is not None
            else (
                tuple(request["download_source_ids"])
                if "download_source_ids" in request
                else None
            )
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
        install_component_ids=(
            tuple(request["install_component_ids"])
            if "install_component_ids" in request
            else None
        ),
        download_source_ids=(
            tuple(request["download_source_ids"])
            if "download_source_ids" in request
            else None
        ),
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
    elif isinstance(exc, RuntimeSelectionError):
        legacy_code = "invalid_request"
        canonical_code = exc.code.value
        category = "validation"
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
    _configure_utf8_standard_streams()
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
            install_component_ids=(
                tuple(request["install_component_ids"])
                if "install_component_ids" in request
                else None
            ),
            download_source_ids=(
                tuple(request["download_source_ids"])
                if "download_source_ids" in request
                else None
            ),
            profile_id=request.get("profile_id"),
        )
    except (
        json.JSONDecodeError,
        ManifestError,
        RuntimeInstallError,
        RuntimeLockTimeout,
        RuntimeOperationError,
        RuntimeSelectionError,
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
