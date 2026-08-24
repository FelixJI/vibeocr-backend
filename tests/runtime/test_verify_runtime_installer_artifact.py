from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from scripts.verify_runtime_installer_artifact import (
    verify_runtime_installer_artifact,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_frozen_runtime_installer_inspects_bound_release(tmp_path: Path) -> None:
    executable_bytes = b"frozen-installer"
    archive_name = "vibeocr-runtime-installer-v0.10.0-win-x64.zip"
    member_name = "runtime-installer/vibeocr-runtime-installer.exe"
    with zipfile.ZipFile(tmp_path / archive_name, mode="w") as archive:
        archive.writestr(member_name, executable_bytes)
    manifest = {
        "backend_version": "0.10.0",
        "backend_sha256": "1" * 64,
        "protocol_wheel": "vibeocr_runtime_contracts-2.3.0-py3-none-any.whl",
        "protocol_manifest_sha256": "2" * 64,
        "capabilities": ["ocr.recognition.v2"],
        "installer": {
            "archive": archive_name,
            "executable_path": member_name,
        },
    }
    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    observed: dict[str, Any] = {}

    def fake_run(
        command: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        executable = Path(command[0])
        assert executable.read_bytes() == executable_bytes
        request = json.loads(command[2])
        component_lock = json.loads(
            Path(request["component_lock"]).read_text(encoding="utf-8")
        )
        observed.update(request=request, component_lock=component_lock, kwargs=kwargs)
        envelope = {
            "protocol_version": 2,
            "ok": True,
            "operation": "inspect",
            "capability_descriptors": [
                {"name": "ocr.recognition.v2", "lifecycle": "active"}
            ],
        }
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(envelope) + "\n",
            stderr="",
        )

    envelope = verify_runtime_installer_artifact(tmp_path, runner=fake_run)

    assert envelope["ok"] is True
    assert observed["request"]["operation"] == "inspect"
    assert observed["component_lock"] == {
        "schema_version": 1,
        "protocol": {
            "repository": "FelixJI/vibeocr-protocol",
            "version": "2.3.0",
            "manifest_sha256": "2" * 64,
        },
        "backend": {
            "repository": "FelixJI/vibeocr-backend",
            "version": "0.10.0",
            "artifact_sha256": "1" * 64,
            "runtime_manifest_sha256": _sha(manifest_path.read_bytes()),
            "accelerator": "cpu",
        },
        "required_capabilities": ["ocr.recognition.v2"],
    }
    assert observed["kwargs"]["timeout"] == 60
