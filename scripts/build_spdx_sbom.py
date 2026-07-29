"""Build a deterministic SPDX 2.3 JSON SBOM for release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

_OUTPUT_FILENAME = "SBOM.spdx.json"
_CREATED = "1970-01-01T00:00:00Z"
_CREATOR = "Tool: build_spdx_sbom.py"


@dataclass(frozen=True)
class _Artifact:
    name: str
    sha1: str
    sha256: str
    size: int


def _artifact(path: Path) -> _Artifact:
    # SPDX 2.3 mandates SHA-1 for the package verification code.
    sha1 = hashlib.sha1()
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha1.update(chunk)
            sha256.update(chunk)
            size += len(chunk)
    return _Artifact(path.name, sha1.hexdigest(), sha256.hexdigest(), size)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _identifier(value: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9.-]+", "-", value).strip("-.")
    return identifier or "release"


def _package_verification_code(artifacts: tuple[_Artifact, ...]) -> str:
    # SPDX 2.3 defines this as SHA-1 over the sorted per-file SHA-1 values.
    digest = hashlib.sha1()
    digest.update("".join(sorted(item.sha1 for item in artifacts)).encode("ascii"))
    return digest.hexdigest()


def _document_namespace(
    repository_name: str,
    version: str,
    artifacts: tuple[_Artifact, ...],
) -> str:
    identity = {
        "repository_name": repository_name,
        "version": version,
        "artifacts": [
            {"name": item.name, "sha256": item.sha256, "size": item.size}
            for item in artifacts
        ],
    }
    digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
    return (
        "https://spdx.org/spdxdocs/"
        f"{_identifier(repository_name)}-{_identifier(version)}-{digest}"
    )


def build_spdx_sbom(
    *,
    artifacts_dir: Path,
    repository_name: str,
    version: str,
) -> Path:
    """Write ``SBOM.spdx.json`` for the regular files in ``artifacts_dir``."""

    if not repository_name.strip():
        raise ValueError("repository_name is required")
    if not version.strip():
        raise ValueError("version is required")

    root = artifacts_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("artifacts_dir must be a directory")

    paths = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file() and path.name != _OUTPUT_FILENAME
        ),
        key=lambda path: path.name,
    )
    if not paths:
        raise ValueError("artifacts_dir must contain at least one artifact")

    artifacts = tuple(_artifact(path) for path in paths)
    files = [
        {
            "SPDXID": f"SPDXRef-File-{index:04d}",
            "checksums": [
                {
                    "algorithm": "SHA256",
                    "checksumValue": artifact.sha256,
                }
            ],
            "comment": f"size={artifact.size}",
            "copyrightText": "NOASSERTION",
            "fileName": artifact.name,
            "licenseConcluded": "NOASSERTION",
            "licenseInfoInFiles": ["NOASSERTION"],
        }
        for index, artifact in enumerate(artifacts, start=1)
    ]
    package_id = "SPDXRef-Package"
    document = {
        "SPDXID": "SPDXRef-DOCUMENT",
        "creationInfo": {
            "created": _CREATED,
            "creators": [_CREATOR],
        },
        "dataLicense": "CC0-1.0",
        "documentNamespace": _document_namespace(
            repository_name,
            version,
            artifacts,
        ),
        "files": files,
        "name": f"{repository_name}-{version}-release-artifacts",
        "packages": [
            {
                "SPDXID": package_id,
                "copyrightText": "NOASSERTION",
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": True,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "name": repository_name,
                "packageVerificationCode": {
                    "packageVerificationCodeValue": (
                        _package_verification_code(artifacts)
                    )
                },
                "versionInfo": version,
            }
        ],
        "relationships": [
            {
                "relatedSpdxElement": package_id,
                "relationshipType": "DESCRIBES",
                "spdxElementId": "SPDXRef-DOCUMENT",
            },
            *[
                {
                    "relatedSpdxElement": file["SPDXID"],
                    "relationshipType": "CONTAINS",
                    "spdxElementId": package_id,
                }
                for file in files
            ],
        ],
        "spdxVersion": "SPDX-2.3",
    }
    output = root / _OUTPUT_FILENAME
    output.write_bytes(_canonical_json(document))
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--repository-name", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args(argv)
    output = build_spdx_sbom(
        artifacts_dir=args.artifacts_dir,
        repository_name=args.repository_name,
        version=args.version,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
