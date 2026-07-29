"""Validation for released Backend runtime manifests and hash locks."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

PROFILE_NAMES = ("win-x64-cpu", "win-x64-cu126")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


class ManifestError(ValueError):
    """Raised when a release manifest or bound artifact is invalid."""


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    name: str
    lock_path: Path
    sha256: str
    runtime_pack: str | None


@dataclass(frozen=True, slots=True)
class PythonRuntime:
    version: str
    abi: str
    platform: str
    source_url: str
    archive_path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class InstallerArtifact:
    archive_path: Path
    sha256: str
    executable_path: str
    executable_sha256: str


@dataclass(frozen=True, slots=True)
class RuntimeManifest:
    path: Path
    sha256: str
    backend_version: str
    backend_wheel: str
    backend_sha256: str
    protocol: str
    protocol_manifest: str
    protocol_manifest_sha256: str
    protocol_wheel: str
    protocol_sha256: str
    python: PythonRuntime
    installer: InstallerArtifact
    profiles: dict[str, RuntimeProfile]
    capabilities: tuple[str, ...]
    source_commit: str
    build_workflow: str
    raw: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installer_executable_sha256(path: Path, executable_path: str) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if names != [executable_path]:
                raise ManifestError(
                    "installer archive must contain only the bound executable"
                )
            return hashlib.sha256(archive.read(executable_path)).hexdigest()
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ManifestError("installer archive is invalid") from exc


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ManifestError(f"{field} must be lowercase SHA-256")
    return value


def _relative_filename(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{field} must be a filename")
    path = PurePath(value)
    if path.is_absolute() or len(path.parts) != 1 or ".." in path.parts:
        raise ManifestError(f"{field} must be one relative filename")
    return value


def _validate_protocol_release_manifest(
    path: Path,
    *,
    protocol_version: str,
    protocol_wheel: str,
    protocol_sha256: str,
) -> None:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError("Protocol release manifest is invalid") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ManifestError("Protocol release manifest schema_version must be 1")
    if value.get("protocol_version") != protocol_version:
        raise ManifestError("Protocol release manifest version mismatch")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ManifestError("Protocol release manifest artifacts are invalid")
    artifact = artifacts.get(protocol_wheel)
    if not isinstance(artifact, dict) or artifact.get("sha256") != protocol_sha256:
        raise ManifestError("Protocol wheel is not bound by its release manifest")


def validate_requirements_lock(path: Path, *, profile: str) -> None:
    """Require an exact artifact and at least one hash for every entry."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ManifestError(f"runtime lock missing: {path}") from exc
    entries: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if not line or line.lstrip().startswith("#") or line.startswith("--"):
            continue
        if not line[0].isspace():
            if current:
                entries.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        entries.append(current)
    if not entries:
        raise ManifestError(f"runtime lock is empty: {path}")
    for entry in entries:
        declaration = entry[0]
        block = "\n".join(entry)
        if "==" not in declaration and " @ https://" not in declaration:
            raise ManifestError(f"runtime requirement is not exact: {declaration}")
        if "--hash=sha256:" not in block:
            raise ManifestError(f"runtime requirement has no SHA-256: {declaration}")

    lowered = "\n".join(lines).lower()
    if profile == "win-x64-cpu":
        if "paddlepaddle-gpu" in lowered or "cu126" in lowered:
            raise ManifestError("CPU lock contains a GPU/cu126 artifact")
    elif profile == "win-x64-cu126":
        for required in ("paddlepaddle-gpu", "torch", "torchvision", "cu126"):
            if required not in lowered:
                raise ManifestError(f"cu126 lock is missing {required}")
        if re.search(r"(?m)^paddlepaddle==", lowered):
            raise ManifestError("cu126 lock contains CPU paddlepaddle")
    else:
        raise ManifestError(f"unsupported profile: {profile}")


def load_runtime_manifest(
    path: str | Path,
    *,
    verify_artifacts: bool = True,
) -> RuntimeManifest:
    manifest_path = Path(path).resolve(strict=True)
    raw_bytes = manifest_path.read_bytes()
    try:
        data: Any = json.loads(raw_bytes)
    except (ValueError, TypeError) as exc:
        raise ManifestError("runtime manifest is invalid JSON") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ManifestError("runtime manifest schema_version must be 1")
    version = data.get("backend_version")
    if not isinstance(version, str) or not _VERSION_RE.fullmatch(version):
        raise ManifestError("backend_version must be stable SemVer")
    wheel = _relative_filename(data.get("backend_wheel"), field="backend_wheel")
    if not wheel.startswith(f"vibeocr_backend-{version}-"):
        raise ManifestError("backend_wheel does not match backend_version")
    backend_sha = _sha256(data.get("backend_sha256"), field="backend_sha256")
    protocol = data.get("protocol")
    if not isinstance(protocol, str) or protocol != ">=2.0.0,<3.0.0":
        raise ManifestError("protocol range must be >=2.0.0,<3.0.0")
    protocol_wheel = _relative_filename(
        data.get("protocol_wheel"),
        field="protocol_wheel",
    )
    protocol_wheel_match = re.fullmatch(
        r"vibeocr_runtime_contracts-(2\.\d+\.\d+)-py3-none-any\.whl",
        protocol_wheel,
    )
    if protocol_wheel_match is None:
        raise ManifestError("protocol_wheel must bind Protocol v2")
    protocol_version = protocol_wheel_match.group(1)
    protocol_sha = _sha256(data.get("protocol_sha256"), field="protocol_sha256")
    protocol_manifest = _relative_filename(
        data.get("protocol_manifest"),
        field="protocol_manifest",
    )
    if protocol_manifest != "protocol-release-manifest.json":
        raise ManifestError(
            "protocol_manifest must be protocol-release-manifest.json"
        )
    protocol_manifest_sha = _sha256(
        data.get("protocol_manifest_sha256"),
        field="protocol_manifest_sha256",
    )
    python_data = data.get("python")
    if not isinstance(python_data, dict):
        raise ManifestError("python runtime binding is required")
    python_version = python_data.get("version")
    if (
        not isinstance(python_version, str)
        or not re.fullmatch(r"3\.13\.\d+", python_version)
    ):
        raise ManifestError("python.version must pin CPython 3.13 patch version")
    if python_data.get("abi") != "cp313":
        raise ManifestError("python.abi must be cp313")
    if python_data.get("platform") != "win_amd64":
        raise ManifestError("python.platform must be win_amd64")
    python_source_url = python_data.get("source_url")
    expected_python_suffix = (
        f"cpython-{python_version}+20260325-x86_64-pc-windows-msvc"
        "-install_only.tar.gz"
    )
    if (
        not isinstance(python_source_url, str)
        or not python_source_url.startswith(
            "https://github.com/astral-sh/python-build-standalone/releases/"
            "download/20260325/"
        )
        or not python_source_url.endswith(expected_python_suffix)
    ):
        raise ManifestError("python.source_url must bind the approved CPython asset")
    python_archive = _relative_filename(
        python_data.get("archive"),
        field="python.archive",
    )
    python_sha = _sha256(python_data.get("sha256"), field="python.sha256")
    python_runtime = PythonRuntime(
        version=python_version,
        abi="cp313",
        platform="win_amd64",
        source_url=python_source_url,
        archive_path=manifest_path.parent / python_archive,
        sha256=python_sha,
    )
    installer_data = data.get("installer")
    if not isinstance(installer_data, dict):
        raise ManifestError("installer binding is required")
    installer_archive = _relative_filename(
        installer_data.get("archive"),
        field="installer.archive",
    )
    executable_path = installer_data.get("executable_path")
    if executable_path != "runtime-installer/vibeocr-runtime-installer.exe":
        raise ManifestError("installer.executable_path is invalid")
    installer_artifact = InstallerArtifact(
        archive_path=manifest_path.parent / installer_archive,
        sha256=_sha256(installer_data.get("sha256"), field="installer.sha256"),
        executable_path=executable_path,
        executable_sha256=_sha256(
            installer_data.get("executable_sha256"),
            field="installer.executable_sha256",
        ),
    )
    profiles_data = data.get("profiles")
    if not isinstance(profiles_data, dict) or set(profiles_data) != set(PROFILE_NAMES):
        raise ManifestError("runtime manifest must define CPU and cu126 profiles")
    profiles: dict[str, RuntimeProfile] = {}
    for name in PROFILE_NAMES:
        record = profiles_data[name]
        if not isinstance(record, dict):
            raise ManifestError(f"profiles.{name} must be an object")
        filename = _relative_filename(record.get("lock"), field=f"profiles.{name}.lock")
        lock_path = manifest_path.parent / filename
        lock_sha = _sha256(record.get("sha256"), field=f"profiles.{name}.sha256")
        runtime_pack = record.get("runtime_pack")
        if runtime_pack is not None:
            runtime_pack = _relative_filename(
                runtime_pack,
                field=f"profiles.{name}.runtime_pack",
            )
        if verify_artifacts:
            if sha256_file(lock_path) != lock_sha:
                raise ManifestError(f"{name} lock SHA-256 mismatch")
            validate_requirements_lock(lock_path, profile=name)
        profiles[name] = RuntimeProfile(name, lock_path, lock_sha, runtime_pack)
    capabilities = data.get("capabilities")
    if (
        not isinstance(capabilities, list)
        or not capabilities
        or not all(isinstance(item, str) and item for item in capabilities)
    ):
        raise ManifestError("capabilities must be a non-empty string list")
    source_commit = data.get("source_commit")
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ManifestError("source_commit must be a full Git SHA")
    workflow = data.get("build_workflow")
    if not isinstance(workflow, str) or not workflow.strip():
        raise ManifestError("build_workflow is required")
    if verify_artifacts:
        wheel_path = manifest_path.parent / wheel
        if sha256_file(wheel_path) != backend_sha:
            raise ManifestError("Backend wheel SHA-256 mismatch")
        protocol_path = manifest_path.parent / protocol_wheel
        if sha256_file(protocol_path) != protocol_sha:
            raise ManifestError("Protocol wheel SHA-256 mismatch")
        protocol_manifest_path = manifest_path.parent / protocol_manifest
        if sha256_file(protocol_manifest_path) != protocol_manifest_sha:
            raise ManifestError("Protocol manifest SHA-256 mismatch")
        _validate_protocol_release_manifest(
            protocol_manifest_path,
            protocol_version=protocol_version,
            protocol_wheel=protocol_wheel,
            protocol_sha256=protocol_sha,
        )
        if sha256_file(python_runtime.archive_path) != python_runtime.sha256:
            raise ManifestError("Python archive SHA-256 mismatch")
        if sha256_file(installer_artifact.archive_path) != installer_artifact.sha256:
            raise ManifestError("Installer archive SHA-256 mismatch")
        actual_executable_sha = installer_executable_sha256(
            installer_artifact.archive_path,
            installer_artifact.executable_path,
        )
        if actual_executable_sha != installer_artifact.executable_sha256:
            raise ManifestError("Installer executable SHA-256 mismatch")
    return RuntimeManifest(
        path=manifest_path,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        backend_version=version,
        backend_wheel=wheel,
        backend_sha256=backend_sha,
        protocol=protocol,
        protocol_manifest=protocol_manifest,
        protocol_manifest_sha256=protocol_manifest_sha,
        protocol_wheel=protocol_wheel,
        protocol_sha256=protocol_sha,
        python=python_runtime,
        installer=installer_artifact,
        profiles=profiles,
        capabilities=tuple(capabilities),
        source_commit=source_commit,
        build_workflow=workflow,
        raw=data,
    )


__all__ = [
    "PROFILE_NAMES",
    "InstallerArtifact",
    "ManifestError",
    "PythonRuntime",
    "RuntimeManifest",
    "RuntimeProfile",
    "installer_executable_sha256",
    "load_runtime_manifest",
    "sha256_file",
    "validate_requirements_lock",
]
