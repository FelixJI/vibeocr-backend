from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_automation_identity import build_identity


def _runtime(source_sha: str, protocol_version: str) -> dict[str, str]:
    return {
        "backend_version": "0.7.2",
        "source_commit": source_sha,
        "backend_sha256": "b" * 64,
        "protocol_wheel": (
            f"vibeocr_runtime_contracts-{protocol_version}-py3-none-any.whl"
        ),
        "protocol_sha256": "c" * 64,
    }


def test_build_identity_supports_legacy_protocol_manifest(tmp_path: Path) -> None:
    source_sha = "a" * 40
    (tmp_path / "runtime-manifest.json").write_text(
        json.dumps(_runtime(source_sha, "2.3.4")), encoding="utf-8"
    )
    protocol_path = tmp_path / "protocol-release-manifest.json"
    protocol_path.write_text(
        json.dumps({"schema_version": 1, "protocol_version": "2.3.4"}),
        encoding="utf-8",
    )

    output = build_identity(tmp_path, version="0.7.2", source_sha=source_sha)

    identity = json.loads(output.read_text(encoding="utf-8"))
    assert identity["project"] == {
        "component": "backend",
        "repository": "FelixJI/vibeocr-backend",
        "version": "0.7.2",
        "source_sha": source_sha,
    }
    assert identity["protocol"] == {
        "repository": "FelixJI/vibeocr-protocol",
        "version": "2.3.4",
        "source_sha": None,
        "release_manifest_sha256": hashlib.sha256(
            protocol_path.read_bytes()
        ).hexdigest(),
        "wheel_sha256": "c" * 64,
        "compatibility": {
            "supported_majors": [2],
            "minor_compatible": True,
        },
    }


def test_build_identity_reads_canonical_protocol_source(tmp_path: Path) -> None:
    source_sha = "a" * 40
    protocol_sha = "d" * 40
    (tmp_path / "runtime-manifest.json").write_text(
        json.dumps(_runtime(source_sha, "2.4.0")), encoding="utf-8"
    )
    (tmp_path / "protocol-release-manifest.json").write_text(
        json.dumps(
            {
                "release": {"version": "2.4.0"},
                "source": {"sha": protocol_sha},
            }
        ),
        encoding="utf-8",
    )

    identity = json.loads(
        build_identity(tmp_path, version="0.7.2", source_sha=source_sha).read_text(
            encoding="utf-8"
        )
    )
    assert identity["protocol"]["version"] == "2.4.0"
    assert identity["protocol"]["source_sha"] == protocol_sha
    assert identity["protocol"]["compatibility"] == {
        "supported_majors": [2],
        "minor_compatible": True,
    }
