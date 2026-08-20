"""Build and verify a deterministic Backend runtime release manifest.

The manifest binds the exact Protocol/Backend wheels and both Windows runtime
profiles by raw-byte SHA-256.  Release automation should publish only the
files copied into ``--output-dir`` plus the generated manifest/checksum list.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[1]
for _source_root in (
    REPO_ROOT / "packages" / "vibeocr-backend" / "src",
    REPO_ROOT / "src",
):
    if _source_root.is_dir():
        sys.path.insert(0, str(_source_root))
        break

from vibeocr.backend.runtime_manifest import (  # noqa: E402
    PROFILE_NAMES,
    default_profile_components,
    installer_executable_sha256,
    load_runtime_manifest,
    runtime_component_binding,
    sha256_file,
    validate_requirements_lock,
)

_STABLE_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_CAPABILITIES = (
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
    "runtime.download-sources.v1",
    "runtime.component-selection.v1",
)


def _component_version_package(profile: str, component_id: str) -> str | None:
    return runtime_component_binding(profile, component_id).distribution


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _copy_exact(
    source: Path,
    output_dir: Path,
    *,
    destination_name: str | None = None,
) -> Path:
    source = source.resolve(strict=True)
    target = output_dir / (destination_name or source.name)
    if source != target:
        shutil.copyfile(source, target)
    return target


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _locked_version(path: Path, project: str) -> str | None:
    text = path.read_text(encoding="utf-8")
    exact = re.search(rf"(?mi)^{re.escape(project)}==([^\s\\]+)", text)
    if exact is not None:
        return exact.group(1)
    direct = re.search(rf"(?mi)^{re.escape(project)}\s+@\s+(\S+)", text)
    if direct is None:
        return None
    filename = unquote(direct.group(1).rsplit("/", 1)[-1])
    package_pattern = re.escape(project).replace(r"\-", "[-_]")
    artifact = re.search(
        rf"(?i)^{package_pattern}[-_](\d+(?:\.\d+)+(?:\+cu\d+)?)-",
        filename,
    )
    return artifact.group(1) if artifact is not None else None


def _profile_components(path: Path, profile: str) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for descriptor in default_profile_components(profile):
        version = _locked_version(
            path,
            _component_version_package(profile, descriptor.component_id),
        )
        result.append(
            {
                **descriptor.to_payload(),
                **({"version": version} if version is not None else {}),
            }
        )
    return result


def build_runtime_manifest(
    *,
    backend_wheel: Path,
    protocol_wheel: Path,
    protocol_manifest: Path,
    base_lock: Path,
    cpu_lock: Path,
    cu126_lock: Path,
    cu126_gpu_lock: Path,
    python_archive: Path,
    python_version: str,
    python_source_url: str,
    installer_archive: Path,
    backend_version: str,
    source_commit: str,
    build_workflow: str,
    output_dir: Path,
    capabilities: tuple[str, ...] = DEFAULT_CAPABILITIES,
    runtime_packs: dict[str, list[Path]] | None = None,
) -> Path:
    if not _STABLE_SEMVER.fullmatch(backend_version):
        raise ValueError("backend_version must be stable SemVer")
    if not _FULL_SHA.fullmatch(source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    if not build_workflow.strip():
        raise ValueError("build_workflow is required")
    if not capabilities or any(not item.strip() for item in capabilities):
        raise ValueError("at least one non-empty capability is required")

    backend_wheel = backend_wheel.resolve(strict=True)
    protocol_wheel = protocol_wheel.resolve(strict=True)
    protocol_manifest = protocol_manifest.resolve(strict=True)
    python_archive = python_archive.resolve(strict=True)
    installer_archive = installer_archive.resolve(strict=True)
    if not backend_wheel.name.startswith(f"vibeocr_backend-{backend_version}-"):
        raise ValueError("Backend wheel filename does not match backend_version")
    if not protocol_wheel.name.startswith("vibeocr_runtime_contracts-2."):
        raise ValueError("Protocol wheel must be a v2 runtime-contracts wheel")
    if protocol_manifest.name != "release-manifest.json":
        raise ValueError("Protocol manifest must be named release-manifest.json")

    profile_sources = {
        "win-x64-base": base_lock.resolve(strict=True),
        "win-x64-cpu": cpu_lock.resolve(strict=True),
        "win-x64-cu126": cu126_lock.resolve(strict=True),
    }
    for profile in PROFILE_NAMES:
        validate_requirements_lock(profile_sources[profile], profile=profile)
    cu126_gpu_lock = cu126_gpu_lock.resolve(strict=True)
    validate_requirements_lock(cu126_gpu_lock, profile="win-x64-cu126")

    output_dir.mkdir(parents=True, exist_ok=True)
    copied_backend = _copy_exact(backend_wheel, output_dir)
    copied_protocol = _copy_exact(protocol_wheel, output_dir)
    copied_protocol_manifest = _copy_exact(
        protocol_manifest,
        output_dir,
        destination_name="protocol-release-manifest.json",
    )
    copied_python = _copy_exact(python_archive, output_dir)
    copied_installer = _copy_exact(installer_archive, output_dir)
    pack_inputs = runtime_packs or {}
    copied_packs: dict[str, list[Path]] = {
        profile: [_copy_exact(pack, output_dir) for pack in packs]
        for profile, packs in pack_inputs.items()
        if packs
    }
    for profile in copied_packs:
        if profile not in PROFILE_NAMES:
            raise ValueError(f"unknown runtime pack profile: {profile}")
    copied_profiles = {
        profile: _copy_exact(profile_sources[profile], output_dir)
        for profile in PROFILE_NAMES
    }
    copied_cu126_gpu_lock = _copy_exact(cu126_gpu_lock, output_dir)
    base_component_ids = [
        component.component_id
        for component in default_profile_components("win-x64-base")
    ]

    manifest = {
        "schema_version": 1,
        "backend_version": backend_version,
        "backend_wheel": copied_backend.name,
        "backend_sha256": sha256_file(copied_backend),
        "protocol": ">=2.0.0,<3.0.0",
        "protocol_manifest": copied_protocol_manifest.name,
        "protocol_manifest_sha256": sha256_file(copied_protocol_manifest),
        "protocol_wheel": copied_protocol.name,
        "protocol_sha256": sha256_file(copied_protocol),
        "python": {
            "version": python_version,
            "abi": "cp313",
            "platform": "win_amd64",
            "source_url": python_source_url,
            "archive": copied_python.name,
            "sha256": sha256_file(copied_python),
        },
        "installer": {
            "archive": copied_installer.name,
            "sha256": sha256_file(copied_installer),
            "executable_path": "runtime-installer/vibeocr-runtime-installer.exe",
            "executable_sha256": installer_executable_sha256(
                copied_installer,
                "runtime-installer/vibeocr-runtime-installer.exe",
            ),
        },
        "profiles": {
            profile: {
                "lock": copied_profiles[profile].name,
                **(
                    {
                        "runtime_pack": [pack.name for pack in copied_packs[profile]],
                        "runtime_pack_sha256": [
                            sha256_file(pack) for pack in copied_packs[profile]
                        ],
                    }
                    if copied_packs.get(profile)
                    else {"runtime_pack": None}
                ),
                "sha256": sha256_file(copied_profiles[profile]),
                "components": _profile_components(copied_profiles[profile], profile),
                **(
                    {
                        "install_scopes": [
                            {
                                "scope_id": "gpu-runtime",
                                "component_ids": [
                                    *base_component_ids,
                                    "gpu_runtime",
                                ],
                                "lock": copied_cu126_gpu_lock.name,
                                "runtime_pack": None,
                                "sha256": sha256_file(copied_cu126_gpu_lock),
                            }
                        ]
                    }
                    if profile == "win-x64-cu126"
                    else {}
                ),
            }
            for profile in PROFILE_NAMES
        },
        "capabilities": sorted(set(capabilities)),
        "source_commit": source_commit,
        "build_workflow": build_workflow,
    }
    manifest_path = output_dir / "runtime-manifest.json"
    manifest_path.write_bytes(_canonical_json(manifest))
    load_runtime_manifest(manifest_path)

    checksum_targets = [
        copied_backend,
        copied_protocol,
        copied_protocol_manifest,
        copied_python,
        copied_installer,
        *(pack for packs in copied_packs.values() for pack in packs),
        *(copied_profiles[profile] for profile in PROFILE_NAMES),
        copied_cu126_gpu_lock,
        manifest_path,
    ]
    checksum_lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(checksum_targets, key=lambda item: item.name)
    ]
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-wheel", type=Path, required=True)
    parser.add_argument("--protocol-wheel", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--base-lock", type=Path, required=True)
    parser.add_argument("--cpu-lock", type=Path, required=True)
    parser.add_argument("--cu126-lock", type=Path, required=True)
    parser.add_argument("--cu126-gpu-lock", type=Path, required=True)
    parser.add_argument("--python-archive", type=Path, required=True)
    parser.add_argument("--python-version", default="3.13.12")
    parser.add_argument(
        "--python-source-url",
        default=(
            "https://github.com/astral-sh/python-build-standalone/releases/"
            "download/20260325/"
            "cpython-3.13.12+20260325-x86_64-pc-windows-msvc"
            "-install_only.tar.gz"
        ),
    )
    parser.add_argument("--installer-archive", type=Path, required=True)
    parser.add_argument(
        "--base-runtime-pack",
        type=Path,
        action="append",
        default=None,
        help="Offline wheel-closure archive bound to the base profile (repeatable).",
    )

    parser.add_argument("--backend-version", required=True)
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--build-workflow", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--capability",
        action="append",
        dest="capabilities",
    )
    args = parser.parse_args(argv)
    source_commit = args.source_commit or _git_sha()
    path = build_runtime_manifest(
        backend_wheel=args.backend_wheel,
        protocol_wheel=args.protocol_wheel,
        protocol_manifest=args.protocol_manifest,
        base_lock=args.base_lock,
        cpu_lock=args.cpu_lock,
        cu126_lock=args.cu126_lock,
        cu126_gpu_lock=args.cu126_gpu_lock,
        python_archive=args.python_archive,
        python_version=args.python_version,
        python_source_url=args.python_source_url,
        installer_archive=args.installer_archive,
        backend_version=args.backend_version,
        source_commit=source_commit,
        build_workflow=args.build_workflow,
        output_dir=args.output_dir,
        capabilities=tuple(args.capabilities or DEFAULT_CAPABILITIES),
        runtime_packs={"win-x64-base": args.base_runtime_pack or []},
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
