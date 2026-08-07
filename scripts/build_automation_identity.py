"""Bind Backend and resolved Protocol identities for canonical release automation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_version(runtime: dict[str, object], release: dict[str, object]) -> str:
    release_identity = release.get("release")
    if isinstance(release_identity, dict) and isinstance(
        release_identity.get("version"), str
    ):
        return release_identity["version"]
    legacy = release.get("protocol_version")
    if isinstance(legacy, str):
        return legacy
    wheel = runtime.get("protocol_wheel")
    match = re.fullmatch(
        r"vibeocr_runtime_contracts-(\d+\.\d+\.\d+)-.+\.whl", str(wheel)
    )
    if match is None:
        raise ValueError("unable to resolve actual Protocol version")
    return match.group(1)


def _project_policy() -> tuple[str, str, dict[str, object]]:
    config = json.loads((ROOT / ".ci/project.json").read_text(encoding="utf-8"))
    project = config.get("project")
    if not isinstance(project, dict):
        raise ValueError("project configuration is missing")
    component = project.get("component")
    repository = project.get("repository")
    compatibility = project.get("protocol_compatibility")
    if (
        not isinstance(component, str)
        or not isinstance(repository, str)
        or not isinstance(compatibility, dict)
        or not isinstance(compatibility.get("supported_majors"), list)
        or not all(isinstance(item, int) for item in compatibility["supported_majors"])
        or not isinstance(compatibility.get("minor_compatible"), bool)
    ):
        raise ValueError("project Protocol compatibility is invalid")
    return component, repository, compatibility


def build_identity(
    artifacts_dir: Path,
    *,
    version: str,
    source_sha: str,
) -> Path:
    artifacts = artifacts_dir.resolve(strict=True)
    runtime_path = artifacts / "runtime-manifest.json"
    protocol_path = artifacts / "protocol-release-manifest.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    protocol_release = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        runtime.get("backend_version") != version
        or runtime.get("source_commit") != source_sha
    ):
        raise ValueError("runtime manifest does not match Backend version/source")
    protocol_version = _protocol_version(runtime, protocol_release)
    protocol_source = protocol_release.get("source")
    protocol_source_sha = (
        protocol_source.get("sha") if isinstance(protocol_source, dict) else None
    )
    component, repository, compatibility = _project_policy()
    identity = {
        "schema_version": 1,
        "project": {
            "component": component,
            "repository": repository,
            "version": version,
            "source_sha": source_sha,
        },
        "backend": {
            "version": version,
            "runtime_manifest_sha256": _sha256(runtime_path),
            "wheel_sha256": runtime["backend_sha256"],
        },
        "protocol": {
            "repository": "FelixJI/vibeocr-protocol",
            "version": protocol_version,
            "source_sha": protocol_source_sha,
            "release_manifest_sha256": _sha256(protocol_path),
            "wheel_sha256": runtime["protocol_sha256"],
            "compatibility": compatibility,
        },
    }
    output = artifacts / "build-identity.json"
    output.write_text(
        json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args(argv)
    print(
        build_identity(
            args.artifacts_dir, version=args.version, source_sha=args.source_sha
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
