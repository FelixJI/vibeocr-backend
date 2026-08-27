"""Validation for released Backend runtime manifests and hash locks."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

PROFILE_NAMES = ("win-x64-base", "win-x64-cpu", "win-x64-cu126")
# base：随 Portable 携带的离线必备闭包（win-x64-base）；full 闭包由
# cpu / nvidia_cuda 显式选择（计划 §4.1）。accelerator → profile 的映射
# 是 manifest 域的单一事实来源，selection policy 与 installer 共同消费。
ACCELERATOR_TO_PLAN = {
    "base": "win-x64-base",
    "cpu": "win-x64-cpu",
    "nvidia_cuda": "win-x64-cu126",
}
PROFILE_COMPONENTS = {
    # base-offline 必备闭包：随 Portable 携带、禁网可安装（计划 §4.1）。
    # rapidocr-base 是 Base Runtime 的必备探针身份，不进入高级组件选择目录。
    "win-x64-base": (
        ("rapidocr-base", "RapidOCR base inference"),
        ("pdf_document_tools", "PDF and document tools"),
        ("image_code_tools", "Image, QR, and barcode tools"),
        ("runtime_host", "Runtime HTTP host"),
    ),
    "win-x64-cpu": (
        ("rapidocr-base", "RapidOCR base inference"),
        ("paddleocr-cpu", "PaddleOCR CPU inference"),
        ("mineru-cpu", "MinerU CPU document parsing"),
        ("pdf_document_tools", "PDF and document tools"),
        ("image_code_tools", "Image, QR, and barcode tools"),
        ("runtime_host", "Runtime HTTP host"),
    ),
    "win-x64-cu126": (
        ("rapidocr-base", "RapidOCR base inference"),
        ("paddleocr-cuda", "PaddleOCR CUDA inference"),
        ("mineru-cuda", "MinerU CUDA document parsing"),
        ("pdf_document_tools", "PDF and document tools"),
        ("image_code_tools", "Image, QR, and barcode tools"),
        ("runtime_host", "Runtime HTTP host"),
        ("gpu_runtime", "CUDA and Torch runtime"),
    ),
}
_DEFAULT_COMPONENT_DEPENDENCIES = {
    ("win-x64-cu126", "paddleocr-cuda"): ("gpu_runtime",),
    ("win-x64-cu126", "mineru-cuda"): ("gpu_runtime",),
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


class ManifestError(ValueError):
    """Raised when a release manifest or bound artifact is invalid."""


@dataclass(frozen=True, slots=True)
class RuntimeComponent:
    component_id: str
    display_name: str
    version: str | None = None
    dependencies: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, str]:
        payload = {
            "component_id": self.component_id,
            "display_name": self.display_name,
        }
        if self.version is not None:
            payload["version"] = self.version
        return payload


@dataclass(frozen=True, slots=True)
class RuntimeComponentBinding:
    """Profile-specific package metadata shared by build and runtime probes."""

    distribution: str
    import_name: str


_RUNTIME_COMPONENT_BINDINGS = {
    ("win-x64-base", "rapidocr-base"): RuntimeComponentBinding("rapidocr", "rapidocr"),
    ("win-x64-cpu", "rapidocr-base"): RuntimeComponentBinding("rapidocr", "rapidocr"),
    ("win-x64-cu126", "rapidocr-base"): RuntimeComponentBinding("rapidocr", "rapidocr"),
    ("win-x64-cpu", "paddleocr-cpu"): RuntimeComponentBinding("paddleocr", "paddleocr"),
    ("win-x64-cpu", "mineru-cpu"): RuntimeComponentBinding("mineru", "mineru"),
    ("win-x64-cu126", "paddleocr-cuda"): RuntimeComponentBinding(
        "paddleocr", "paddleocr"
    ),
    ("win-x64-cu126", "mineru-cuda"): RuntimeComponentBinding("mineru", "mineru"),
    ("win-x64-base", "image_code_tools"): RuntimeComponentBinding(
        "opencv-python", "cv2"
    ),
    ("win-x64-cpu", "image_code_tools"): RuntimeComponentBinding(
        "opencv-contrib-python", "cv2"
    ),
    ("win-x64-cu126", "image_code_tools"): RuntimeComponentBinding(
        "opencv-contrib-python", "cv2"
    ),
}
_SHARED_RUNTIME_COMPONENT_BINDINGS = {
    "pdf_document_tools": RuntimeComponentBinding("pymupdf", "fitz"),
    "runtime_host": RuntimeComponentBinding("fastapi", "fastapi"),
    "gpu_runtime": RuntimeComponentBinding("torch", "torch"),
}


def migrate_legacy_component_ids(
    profile_id: str, component_ids: tuple[str, ...]
) -> tuple[str, ...]:
    """把 2.7 的歧义 component id 迁移到一义的 accelerator variant。

    ``ocr_engine`` 在 base 曾表示 RapidOCR、在 full 同时承担 base closure 与
    PaddleOCR；因此 full 迁移必须同时保留两项。旧 ``document_parsing`` 仅表达
    MinerU 产品能力；底层原子 full install scope 的额外闭包由 selection policy
    决定，不能污染持久化的组件身份。
    """
    if profile_id == "win-x64-base":
        replacements = {
            "ocr_engine": ("rapidocr-base",),
            "document_parsing": ("mineru-cpu",),
        }
    elif profile_id == "win-x64-cpu":
        replacements = {
            "ocr_engine": ("rapidocr-base", "paddleocr-cpu"),
            "document_parsing": ("mineru-cpu",),
        }
    elif profile_id == "win-x64-cu126":
        replacements = {
            "ocr_engine": ("rapidocr-base", "paddleocr-cuda"),
            "document_parsing": ("mineru-cuda",),
        }
    else:
        raise ManifestError(f"unsupported profile: {profile_id}")
    migrated: list[str] = []
    for component_id in component_ids:
        for replacement in replacements.get(component_id, (component_id,)):
            if replacement not in migrated:
                migrated.append(replacement)
    return tuple(migrated)


def runtime_component_binding(
    profile_id: str, component_id: str
) -> RuntimeComponentBinding:
    """Return the authoritative package/import binding for one component."""
    binding = _RUNTIME_COMPONENT_BINDINGS.get((profile_id, component_id))
    if binding is None:
        binding = _SHARED_RUNTIME_COMPONENT_BINDINGS.get(component_id)
    if binding is None:
        raise ManifestError(
            f"runtime component binding is missing: {profile_id}/{component_id}"
        )
    return binding


@dataclass(frozen=True, slots=True)
class RuntimeInstallScope:
    scope_id: str
    component_ids: tuple[str, ...]
    lock_path: Path
    sha256: str
    runtime_pack: tuple[str, ...]
    runtime_pack_sha256: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    name: str
    lock_path: Path
    sha256: str
    runtime_pack: tuple[str, ...]
    components: tuple[RuntimeComponent, ...]
    runtime_pack_sha256: tuple[str, ...] = ()
    scopes: tuple[RuntimeInstallScope, ...] = ()


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
    protocol_version: str
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


def _runtime_pack_binding(
    record: dict[str, Any],
    *,
    field: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw_pack = record.get("runtime_pack")
    if raw_pack is None:
        if record.get("runtime_pack_sha256") is not None:
            raise ManifestError(f"{field}.runtime_pack_sha256 requires runtime_pack")
        return (), ()
    if (
        not isinstance(raw_pack, list)
        or not raw_pack
        or not all(isinstance(item, str) and item for item in raw_pack)
    ):
        raise ManifestError(f"{field}.runtime_pack must be a non-empty filename list")
    runtime_pack = tuple(
        _relative_filename(item, field=f"{field}.runtime_pack") for item in raw_pack
    )
    raw_shas = record.get("runtime_pack_sha256")
    if (
        not isinstance(raw_shas, list)
        or len(raw_shas) != len(runtime_pack)
        or not all(isinstance(item, str) and item for item in raw_shas)
    ):
        raise ManifestError(f"{field}.runtime_pack_sha256 must parallel runtime_pack")
    return runtime_pack, tuple(
        _sha256(item, field=f"{field}.runtime_pack_sha256") for item in raw_shas
    )


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
    if not isinstance(value, dict):
        raise ManifestError("Protocol release manifest must be an object")
    schema_version = value.get("schema_version")
    if schema_version == 1:
        manifest_version = value.get("protocol_version")
    elif schema_version == 2:
        project = value.get("project")
        protocol = value.get("protocol")
        release = value.get("release")
        if (
            not isinstance(project, dict)
            or project.get("component") != "protocol"
            or not isinstance(protocol, dict)
            or not isinstance(release, dict)
        ):
            raise ManifestError("Protocol v2 release identity is incomplete")
        manifest_version = protocol.get("version")
        if (
            not isinstance(manifest_version, str)
            or release.get("version") != manifest_version
            or release.get("tag") != f"v{manifest_version}"
        ):
            raise ManifestError("Protocol v2 release identity mismatch")
    else:
        raise ManifestError("unsupported Protocol release manifest schema_version")
    if manifest_version != protocol_version:
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
    source_directive = re.compile(
        r"^\s*--(?:index-url|extra-index-url|find-links|no-index)\b"
    )
    if any(source_directive.match(line) for line in lines):
        raise ManifestError("runtime lock must be source-neutral")
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
    if profile == "win-x64-base":
        # base 闭包必须自带缺省引擎（RapidOCR + 显式 ORT + Windows adapter），
        # 且不得混入 full 闭包的重型组件或第二套 OpenCV。winrt 投影的跨命名
        # 空间依赖只声明在 [all] extra 里，但 await 完成回调与集合投影在
        # 运行时导入 foundation 两个包；缺包时 Windows OCR 永久挂起。
        for required in (
            "rapidocr",
            "onnxruntime",
            "winrt-runtime",
            "winrt-windows-foundation",
            "winrt-windows-foundation-collections",
            "opencv-python",
        ):
            if required not in lowered:
                raise ManifestError(f"base lock is missing {required}")
        for forbidden in (
            "paddlepaddle",
            "paddleocr",
            "mineru",
            "torch",
            "cu126",
            "opencv-contrib-python",
        ):
            if forbidden in lowered:
                raise ManifestError(f"base lock contains {forbidden}")
    elif profile == "win-x64-cpu":
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


def default_profile_components(profile: str) -> tuple[RuntimeComponent, ...]:
    try:
        values = PROFILE_COMPONENTS[profile]
    except KeyError as exc:
        raise ManifestError(f"unsupported profile: {profile}") from exc
    return tuple(
        RuntimeComponent(
            component_id,
            display_name,
            dependencies=_DEFAULT_COMPONENT_DEPENDENCIES.get(
                (profile, component_id), ()
            ),
        )
        for component_id, display_name in values
    )


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
        raise ManifestError("protocol_manifest must be protocol-release-manifest.json")
    protocol_manifest_sha = _sha256(
        data.get("protocol_manifest_sha256"),
        field="protocol_manifest_sha256",
    )
    python_data = data.get("python")
    if not isinstance(python_data, dict):
        raise ManifestError("python runtime binding is required")
    python_version = python_data.get("version")
    if not isinstance(python_version, str) or not re.fullmatch(
        r"3\.13\.\d+", python_version
    ):
        raise ManifestError("python.version must pin CPython 3.13 patch version")
    if python_data.get("abi") != "cp313":
        raise ManifestError("python.abi must be cp313")
    if python_data.get("platform") != "win_amd64":
        raise ManifestError("python.platform must be win_amd64")
    python_source_url = python_data.get("source_url")
    expected_python_suffix = (
        f"cpython-{python_version}+20260807-x86_64-pc-windows-msvc-install_only.tar.gz"
    )
    if (
        not isinstance(python_source_url, str)
        or not python_source_url.startswith(
            "https://github.com/astral-sh/python-build-standalone/releases/"
            "download/20260807/"
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
        raise ManifestError(
            "runtime manifest must define base, CPU, and cu126 profiles"
        )
    profiles: dict[str, RuntimeProfile] = {}
    for name in PROFILE_NAMES:
        record = profiles_data[name]
        if not isinstance(record, dict):
            raise ManifestError(f"profiles.{name} must be an object")
        filename = _relative_filename(record.get("lock"), field=f"profiles.{name}.lock")
        lock_path = manifest_path.parent / filename
        lock_sha = _sha256(record.get("sha256"), field=f"profiles.{name}.sha256")
        # 离线 pack 是一个或多个分片 zip。每个分片必须绑定字节哈希：
        # installer 的禁网路径完全信任该闭包。
        runtime_pack, runtime_pack_shas = _runtime_pack_binding(
            record,
            field=f"profiles.{name}",
        )
        expected = default_profile_components(name)
        components_data = record.get("components")
        if components_data is None:
            components = expected
        else:
            if not isinstance(components_data, list):
                raise ManifestError(f"profiles.{name}.components must be an array")
            parsed_components: list[RuntimeComponent] = []
            for component in components_data:
                if not isinstance(component, dict):
                    raise ManifestError(f"profiles.{name}.components is invalid")
                component_id = component.get("component_id")
                display_name = component.get("display_name")
                version_value = component.get("version")
                default_dependencies = (
                    _DEFAULT_COMPONENT_DEPENDENCIES.get((name, component_id), ())
                    if isinstance(component_id, str)
                    else ()
                )
                dependencies_value = component.get(
                    "dependencies",
                    default_dependencies,
                )
                if (
                    not isinstance(component_id, str)
                    or not component_id
                    or not isinstance(display_name, str)
                    or not display_name
                    or (
                        version_value is not None and not isinstance(version_value, str)
                    )
                    or not isinstance(dependencies_value, (list, tuple))
                    or any(
                        not isinstance(dependency, str) or not dependency
                        for dependency in dependencies_value
                    )
                    or len(set(dependencies_value)) != len(dependencies_value)
                ):
                    raise ManifestError(f"profiles.{name}.components is invalid")
                parsed_components.append(
                    RuntimeComponent(
                        component_id,
                        display_name,
                        version_value,
                        tuple(dependencies_value),
                    )
                )
            if tuple(item.component_id for item in parsed_components) != tuple(
                item.component_id for item in expected
            ):
                raise ManifestError(f"profiles.{name}.components must use stable ids")
            components = tuple(parsed_components)
        component_ids = {component.component_id for component in components}
        for component in components:
            if not set(component.dependencies).issubset(component_ids):
                raise ManifestError(
                    f"profiles.{name}.components dependencies must stay in profile"
                )

        visiting: set[str] = set()
        visited: set[str] = set()
        dependencies_by_id = {
            component.component_id: component.dependencies for component in components
        }

        def visit(component_id: str) -> None:
            if component_id in visiting:
                raise ManifestError(
                    f"profiles.{name}.components dependencies must be acyclic"
                )
            if component_id in visited:
                return
            visiting.add(component_id)
            for dependency in dependencies_by_id[component_id]:
                visit(dependency)
            visiting.remove(component_id)
            visited.add(component_id)

        for component_id in dependencies_by_id:
            visit(component_id)
        if verify_artifacts:
            if sha256_file(lock_path) != lock_sha:
                raise ManifestError(f"{name} lock SHA-256 mismatch")
            validate_requirements_lock(lock_path, profile=name)
            for pack_name, pack_sha in zip(
                runtime_pack, runtime_pack_shas, strict=True
            ):
                if sha256_file(manifest_path.parent / pack_name) != pack_sha:
                    raise ManifestError(f"{name} runtime pack SHA-256 mismatch")
        default_scope = RuntimeInstallScope(
            scope_id="default",
            component_ids=tuple(component.component_id for component in components),
            lock_path=lock_path,
            sha256=lock_sha,
            runtime_pack=runtime_pack,
            runtime_pack_sha256=runtime_pack_shas,
        )
        scopes = [default_scope]
        scope_ids = {default_scope.scope_id}
        component_sets = {frozenset(default_scope.component_ids)}
        raw_scopes = record.get("install_scopes", [])
        if not isinstance(raw_scopes, list):
            raise ManifestError(f"profiles.{name}.install_scopes must be an array")
        base_component_ids = {
            component_id for component_id, _ in PROFILE_COMPONENTS["win-x64-base"]
        }
        for index, scope_record in enumerate(raw_scopes):
            scope_field = f"profiles.{name}.install_scopes[{index}]"
            if not isinstance(scope_record, dict):
                raise ManifestError(f"{scope_field} must be an object")
            scope_id = scope_record.get("scope_id")
            raw_component_ids = scope_record.get("component_ids")
            if not isinstance(scope_id, str) or not scope_id:
                raise ManifestError(f"{scope_field}.scope_id is invalid")
            if scope_id in scope_ids:
                raise ManifestError(f"profiles.{name}.install scope ids must be unique")
            if (
                not isinstance(raw_component_ids, list)
                or not raw_component_ids
                or any(
                    not isinstance(component_id, str) or not component_id
                    for component_id in raw_component_ids
                )
                or len(set(raw_component_ids)) != len(raw_component_ids)
            ):
                raise ManifestError(f"{scope_field}.component_ids is invalid")
            scope_component_ids = tuple(raw_component_ids)
            scope_component_set = frozenset(scope_component_ids)
            if not scope_component_set.issubset(component_ids):
                raise ManifestError(f"{scope_field}.component_ids must stay in profile")
            if not base_component_ids.issubset(scope_component_set):
                raise ManifestError(
                    f"{scope_field}.component_ids must include base components"
                )
            if scope_component_set in component_sets:
                raise ManifestError(
                    f"profiles.{name}.install scope component sets must be unique"
                )
            for component_id in scope_component_ids:
                if not set(dependencies_by_id[component_id]).issubset(
                    scope_component_set
                ):
                    raise ManifestError(
                        f"{scope_field}.component_ids must be a dependency closure"
                    )
            scope_lock_name = _relative_filename(
                scope_record.get("lock"),
                field=f"{scope_field}.lock",
            )
            scope_lock_path = manifest_path.parent / scope_lock_name
            scope_lock_sha = _sha256(
                scope_record.get("sha256"),
                field=f"{scope_field}.sha256",
            )
            scope_pack, scope_pack_shas = _runtime_pack_binding(
                scope_record,
                field=scope_field,
            )
            if verify_artifacts:
                if sha256_file(scope_lock_path) != scope_lock_sha:
                    raise ManifestError(f"{scope_field} lock SHA-256 mismatch")
                validate_requirements_lock(scope_lock_path, profile=name)
                for pack_name, pack_sha in zip(
                    scope_pack, scope_pack_shas, strict=True
                ):
                    if sha256_file(manifest_path.parent / pack_name) != pack_sha:
                        raise ManifestError(
                            f"{scope_field} runtime pack SHA-256 mismatch"
                        )
            scopes.append(
                RuntimeInstallScope(
                    scope_id=scope_id,
                    component_ids=scope_component_ids,
                    lock_path=scope_lock_path,
                    sha256=scope_lock_sha,
                    runtime_pack=scope_pack,
                    runtime_pack_sha256=scope_pack_shas,
                )
            )
            scope_ids.add(scope_id)
            component_sets.add(scope_component_set)
        profiles[name] = RuntimeProfile(
            name,
            lock_path,
            lock_sha,
            runtime_pack,
            components,
            runtime_pack_sha256=runtime_pack_shas,
            scopes=tuple(scopes),
        )
    capabilities = data.get("capabilities")
    if (
        not isinstance(capabilities, list)
        or not capabilities
        or not all(isinstance(item, str) and item for item in capabilities)
    ):
        raise ManifestError("capabilities must be a non-empty string list")
    source_commit = data.get("source_commit")
    if not isinstance(source_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", source_commit
    ):
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
        protocol_version=protocol_version,
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
    "ACCELERATOR_TO_PLAN",
    "PROFILE_NAMES",
    "PROFILE_COMPONENTS",
    "InstallerArtifact",
    "ManifestError",
    "PythonRuntime",
    "RuntimeManifest",
    "RuntimeComponent",
    "RuntimeComponentBinding",
    "RuntimeInstallScope",
    "RuntimeProfile",
    "installer_executable_sha256",
    "load_runtime_manifest",
    "migrate_legacy_component_ids",
    "runtime_component_binding",
    "sha256_file",
    "validate_requirements_lock",
    "default_profile_components",
]
