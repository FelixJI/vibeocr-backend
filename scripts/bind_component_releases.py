"""Verify immutable upstream release metadata and write committed lock files."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def verify_protocol_release(
    release_dir: Path,
    *,
    version: str,
) -> tuple[Path, dict[str, object]]:
    root = release_dir.resolve(strict=True)
    manifest_path = root / "release-manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("protocol_version") != version:
        raise ValueError("Protocol manifest version mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Protocol manifest artifacts must be an object")
    for name, record in artifacts.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            raise ValueError("invalid Protocol artifact record")
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if record.get("sha256") != _sha256(path):
            raise ValueError(f"Protocol artifact hash mismatch: {name}")
        if record.get("size") != path.stat().st_size:
            raise ValueError(f"Protocol artifact size mismatch: {name}")
    return manifest_path, manifest


def bind_protocol_release(
    *,
    release_dir: Path,
    repository: str,
    version: str,
    output: Path,
) -> Path:
    manifest_path, manifest = verify_protocol_release(release_dir, version=version)
    return _write_json(
        output,
        {
            "schema_version": 1,
            "repository": repository,
            "version": version,
            "manifest_sha256": _sha256(manifest_path),
            "artifacts": manifest["artifacts"],
        },
    )


def bind_product_releases(
    *,
    protocol_release_dir: Path,
    backend_release_dir: Path,
    protocol_repository: str,
    protocol_version: str,
    backend_repository: str,
    backend_version: str,
    profile: str,
    required_capabilities: tuple[str, ...],
    output: Path,
) -> Path:
    protocol_manifest_path, _ = verify_protocol_release(
        protocol_release_dir,
        version=protocol_version,
    )
    backend_root = backend_release_dir.resolve(strict=True)
    runtime_manifest_path = backend_root / "runtime-manifest.json"
    runtime_manifest = _load_json(runtime_manifest_path)
    if runtime_manifest.get("backend_version") != backend_version:
        raise ValueError("Backend manifest version mismatch")
    backend_wheel_name = runtime_manifest.get("backend_wheel")
    if not isinstance(backend_wheel_name, str):
        raise ValueError("Backend manifest wheel name is missing")
    backend_wheel = backend_root / backend_wheel_name
    if runtime_manifest.get("backend_sha256") != _sha256(backend_wheel):
        raise ValueError("Backend wheel hash mismatch")
    protocol_copy_name = runtime_manifest.get("protocol_manifest")
    protocol_wheel_name = runtime_manifest.get("protocol_wheel")
    if not isinstance(protocol_copy_name, str) or not isinstance(
        protocol_wheel_name,
        str,
    ):
        raise ValueError("Backend Protocol binding is incomplete")
    protocol_copy = backend_root / protocol_copy_name
    protocol_wheel = backend_root / protocol_wheel_name
    if protocol_copy.read_bytes() != protocol_manifest_path.read_bytes():
        raise ValueError("Backend Protocol manifest copy differs from Protocol release")
    source_protocol_wheel = protocol_release_dir.resolve(strict=True) / protocol_wheel_name
    if protocol_wheel.read_bytes() != source_protocol_wheel.read_bytes():
        raise ValueError("Backend Protocol wheel differs from Protocol release")
    if runtime_manifest.get("protocol_manifest_sha256") != _sha256(protocol_copy):
        raise ValueError("Backend Protocol manifest hash mismatch")
    if runtime_manifest.get("protocol_sha256") != _sha256(protocol_wheel):
        raise ValueError("Backend Protocol wheel hash mismatch")
    python = runtime_manifest.get("python")
    installer = runtime_manifest.get("installer")
    profiles = runtime_manifest.get("profiles")
    if not isinstance(python, dict) or not isinstance(installer, dict):
        raise ValueError("Backend runtime closure is incomplete")
    for label, record, name_key in (
        ("Python archive", python, "archive"),
        ("installer archive", installer, "archive"),
    ):
        name = record.get(name_key)
        if not isinstance(name, str):
            raise ValueError(f"{label} name is missing")
        path = backend_root / name
        if record.get("sha256") != _sha256(path):
            raise ValueError(f"{label} hash mismatch")
    if not isinstance(profiles, dict) or profile not in profiles:
        raise ValueError(f"Backend profile is missing: {profile}")
    for profile_name, record in profiles.items():
        if not isinstance(profile_name, str) or not isinstance(record, dict):
            raise ValueError("invalid Backend profile record")
        lock_name = record.get("lock")
        if not isinstance(lock_name, str):
            raise ValueError(f"Backend profile lock is missing: {profile_name}")
        if record.get("sha256") != _sha256(backend_root / lock_name):
            raise ValueError(f"Backend profile hash mismatch: {profile_name}")
    installer_archive = backend_root / str(installer["archive"])
    executable_path = installer.get("executable_path")
    executable_sha256 = installer.get("executable_sha256")
    if not isinstance(executable_path, str) or not isinstance(
        executable_sha256,
        str,
    ):
        raise ValueError("installer executable binding is incomplete")
    with zipfile.ZipFile(installer_archive) as archive:
        executable = archive.read(executable_path)
    if hashlib.sha256(executable).hexdigest() != executable_sha256:
        raise ValueError("installer executable hash mismatch")
    checksums = backend_root / "SHA256SUMS"
    checksum_records = {}
    for line in checksums.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        checksum_records[name] = digest
    expected_checksum_files = {
        path.name
        for path in backend_root.iterdir()
        if path.is_file() and path.name != checksums.name
    }
    if set(checksum_records) != expected_checksum_files:
        raise ValueError("Backend SHA256SUMS file set mismatch")
    for name, digest in checksum_records.items():
        if _sha256(backend_root / name) != digest:
            raise ValueError(f"Backend SHA256SUMS mismatch: {name}")
    capabilities = runtime_manifest.get("capabilities")
    if not isinstance(capabilities, list) or not all(
        isinstance(item, str) for item in capabilities
    ):
        raise ValueError("Backend capabilities must be a string array")
    missing = sorted(set(required_capabilities) - set(capabilities))
    if missing:
        raise ValueError(f"Backend is missing required capabilities: {missing}")
    return _write_json(
        output,
        {
            "schema_version": 1,
            "protocol": {
                "repository": protocol_repository,
                "version": protocol_version,
                "manifest_sha256": _sha256(protocol_manifest_path),
            },
            "backend": {
                "repository": backend_repository,
                "version": backend_version,
                "artifact_sha256": _sha256(backend_wheel),
                "runtime_manifest_sha256": _sha256(runtime_manifest_path),
                "profile": profile,
            },
            "required_capabilities": sorted(set(required_capabilities)),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    protocol = subparsers.add_parser("protocol-lock")
    protocol.add_argument("--release-dir", type=Path, required=True)
    protocol.add_argument("--repository", required=True)
    protocol.add_argument("--version", required=True)
    protocol.add_argument("--output", type=Path, required=True)
    product = subparsers.add_parser("product-lock")
    product.add_argument("--protocol-release-dir", type=Path, required=True)
    product.add_argument("--backend-release-dir", type=Path, required=True)
    product.add_argument("--protocol-repository", required=True)
    product.add_argument("--protocol-version", required=True)
    product.add_argument("--backend-repository", required=True)
    product.add_argument("--backend-version", required=True)
    product.add_argument("--profile", required=True)
    product.add_argument("--required-capability", action="append", required=True)
    product.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "protocol-lock":
        path = bind_protocol_release(
            release_dir=args.release_dir,
            repository=args.repository,
            version=args.version,
            output=args.output,
        )
    else:
        path = bind_product_releases(
            protocol_release_dir=args.protocol_release_dir,
            backend_release_dir=args.backend_release_dir,
            protocol_repository=args.protocol_repository,
            protocol_version=args.protocol_version,
            backend_repository=args.backend_repository,
            backend_version=args.backend_version,
            profile=args.profile,
            required_capabilities=tuple(args.required_capability),
            output=args.output,
        )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
