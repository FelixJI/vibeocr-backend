"""Execute the frozen Runtime Installer against its bound release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

_PROTOCOL_WHEEL_RE = re.compile(
    r"^vibeocr_runtime_contracts-(?P<version>\d+\.\d+\.\d+)-py3-none-any\.whl$"
)


class RuntimeInstallerArtifactError(RuntimeError):
    """Raised when the frozen Runtime Installer cannot inspect its release."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeInstallerArtifactError("runtime manifest must be a JSON object")
    return value


def _component_lock(
    manifest: dict[str, Any],
    *,
    manifest_sha256: str,
) -> dict[str, Any]:
    protocol_wheel = manifest.get("protocol_wheel")
    if not isinstance(protocol_wheel, str):
        raise RuntimeInstallerArtifactError(
            "runtime manifest protocol wheel is invalid"
        )
    match = _PROTOCOL_WHEEL_RE.fullmatch(protocol_wheel)
    if match is None:
        raise RuntimeInstallerArtifactError(
            "runtime manifest protocol wheel does not contain a stable version"
        )
    return {
        "schema_version": 1,
        "protocol": {
            "repository": "FelixJI/vibeocr-protocol",
            "version": match.group("version"),
            "manifest_sha256": manifest["protocol_manifest_sha256"],
        },
        "backend": {
            "repository": "FelixJI/vibeocr-backend",
            "version": manifest["backend_version"],
            "artifact_sha256": manifest["backend_sha256"],
            "runtime_manifest_sha256": manifest_sha256,
            "accelerator": "cpu",
        },
        "required_capabilities": manifest["capabilities"],
    }


def verify_runtime_installer_artifact(
    artifacts_dir: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    root = artifacts_dir.resolve(strict=True)
    manifest_path = root / "runtime-manifest.json"
    manifest = _load_manifest(manifest_path)
    installer = manifest.get("installer")
    if not isinstance(installer, dict):
        raise RuntimeInstallerArtifactError("runtime manifest installer is invalid")
    archive_name = installer.get("archive")
    member_name = installer.get("executable_path")
    if not isinstance(archive_name, str) or Path(archive_name).name != archive_name:
        raise RuntimeInstallerArtifactError("runtime installer archive name is unsafe")
    if not isinstance(member_name, str) or not member_name:
        raise RuntimeInstallerArtifactError(
            "runtime installer executable path is invalid"
        )

    archive_path = root / archive_name
    with tempfile.TemporaryDirectory(prefix="vibeocr-installer-smoke-") as temporary:
        smoke_root = Path(temporary)
        executable = smoke_root / "vibeocr-runtime-installer.exe"
        with zipfile.ZipFile(archive_path) as archive:
            try:
                executable.write_bytes(archive.read(member_name))
            except KeyError as exc:
                raise RuntimeInstallerArtifactError(
                    f"runtime installer archive is missing {member_name}"
                ) from exc

        component_lock = smoke_root / "component-lock.json"
        component_lock.write_text(
            json.dumps(
                _component_lock(manifest, manifest_sha256=_sha256(manifest_path)),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        request = {
            "protocol_version": 2,
            "operation": "inspect",
            "product_root": str(smoke_root / "product"),
            "component_lock": str(component_lock),
            "runtime_manifest": str(manifest_path),
            "accelerator": "cpu",
        }
        completed = runner(
            [
                str(executable),
                "--request-json",
                json.dumps(request, separators=(",", ":"), sort_keys=True),
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeInstallerArtifactError(
                f"runtime installer inspect failed ({completed.returncode}): {detail}"
            )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeInstallerArtifactError(
                "runtime installer inspect produced no response"
            )
        try:
            envelope = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise RuntimeInstallerArtifactError(
                "runtime installer inspect response is not JSON"
            ) from exc
        if (
            not isinstance(envelope, dict)
            or envelope.get("ok") is not True
            or envelope.get("operation") != "inspect"
            or not isinstance(envelope.get("capability_descriptors"), list)
        ):
            raise RuntimeInstallerArtifactError(
                "runtime installer inspect response violates the success contract"
            )
        return envelope


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts_dir", type=Path)
    args = parser.parse_args()
    envelope = verify_runtime_installer_artifact(args.artifacts_dir)
    print(json.dumps(envelope, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
