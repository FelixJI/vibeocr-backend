"""Run the released base Runtime through offline ensure and real inference.

This gate executes the frozen installer and the installed Supervisor.  It is
not a static archive inspection: the test performs RapidOCR and PDF work, then
proves a second offline ensure reuses the same installed marker.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import secrets
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

if __package__:
    from scripts.verify_runtime_installer_artifact import (
        _component_lock,
        _load_manifest,
        _sha256,
    )
else:
    from verify_runtime_installer_artifact import (  # type: ignore[import-not-found]
        _component_lock,
        _load_manifest,
        _sha256,
    )


class BaseRuntimeSmokeError(RuntimeError):
    """The packaged base Runtime failed a real offline smoke."""


def _offline_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PIP_NO_INDEX": "1",
            "UV_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "ALL_PROXY": "http://127.0.0.1:9",
            "NO_PROXY": "127.0.0.1,localhost",
        }
    )
    environment.pop("PIP_INDEX_URL", None)
    environment.pop("PIP_EXTRA_INDEX_URL", None)
    return environment


def _last_json_line(output: str, *, label: str) -> dict[str, Any]:
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        raise BaseRuntimeSmokeError(f"{label} produced no JSON response")
    try:
        value: Any = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise BaseRuntimeSmokeError(f"{label} response is not JSON") from exc
    if not isinstance(value, dict):
        raise BaseRuntimeSmokeError(f"{label} response must be an object")
    return value


def _run_installer(
    executable: Path,
    request: dict[str, Any],
    *,
    cwd: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(executable),
            "--request-json",
            json.dumps(request, separators=(",", ":"), sort_keys=True),
        ],
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    response = _last_json_line(completed.stdout, label=request["operation"])
    if completed.returncode != 0 or response.get("ok") is not True:
        detail = completed.stderr.strip() or response
        raise BaseRuntimeSmokeError(
            f"installer {request['operation']} failed: {detail}"
        )
    return response


def _base_ensure_request(
    *, product_root: Path, component_lock: Path, runtime_manifest: Path
) -> dict[str, Any]:
    """Build the explicit base-only request used by the release gate."""
    return {
        "protocol_version": 2,
        "operation": "ensure",
        "product_root": str(product_root),
        "component_lock": str(component_lock),
        "runtime_manifest": str(runtime_manifest),
        "accelerator": "cpu",
        "install_component_ids": [],
    }


def _json_request(
    url: str,
    token: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    content_type: str = "application/json",
) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": content_type,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            value: Any = json.loads(response.read())
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise BaseRuntimeSmokeError(f"Supervisor request failed: {url}") from exc
    if not isinstance(value, dict):
        raise BaseRuntimeSmokeError(f"Supervisor response is invalid: {url}")
    return value


def _submit_ocr(url: str, token: str, image: Path) -> str:
    boundary = f"vibeocr-{secrets.token_hex(12)}"
    manifest = json.dumps(
        {
            "schema_version": 2,
            "request_id": "release-base-smoke",
            "kind": "recognition",
            "priority": "interactive",
            "pipeline": {
                "pipeline_id": "OCR",
                "options_version": 1,
                "options": {},
                "engine": "rapidocr",
            },
            "items": [
                {
                    "client_item_key": "sample",
                    "ordinal": 0,
                    "display_name": image.name,
                    "source": {"type": "upload.v1", "attachment": "image"},
                }
            ],
        },
        separators=(",", ":"),
    )
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="manifest"\r\n\r\n{manifest}\r\n'.encode(),
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
            f'filename="{image.name}"\r\nContent-Type: image/png\r\n\r\n'
        ).encode(),
        image.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    request = urllib.request.Request(
        f"{url}/v2/jobs",
        data=b"".join(parts),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            payload: Any = json.loads(response.read())
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise BaseRuntimeSmokeError("RapidOCR submission failed") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("job_id"), str):
        raise BaseRuntimeSmokeError("RapidOCR submission returned no job_id")
    return payload["job_id"]


def _wait_for_ocr(url: str, token: str, job_id: str) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        update = _json_request(f"{url}/v2/jobs/{job_id}/observe", token)
        snapshot = update.get("snapshot")
        state = snapshot.get("state") if isinstance(snapshot, dict) else None
        if state == "succeeded":
            outcomes = update.get("outcomes")
            if not isinstance(outcomes, list) or not outcomes:
                raise BaseRuntimeSmokeError("RapidOCR succeeded without an outcome")
            payload = outcomes[0].get("payload")
            raw_text = payload.get("raw_text") if isinstance(payload, dict) else None
            if not isinstance(raw_text, str) or not raw_text.strip():
                raise BaseRuntimeSmokeError("RapidOCR returned no recognized text")
            return
        if state in {"failed", "cancelled"}:
            raise BaseRuntimeSmokeError(f"RapidOCR terminal state: {state}")
        time.sleep(0.1)
    raise BaseRuntimeSmokeError("RapidOCR job timed out")


def _read_ready_line(process: subprocess.Popen[str]) -> dict[str, Any]:
    result: queue.Queue[str] = queue.Queue(maxsize=1)
    threading.Thread(
        target=lambda: result.put(process.stdout.readline() if process.stdout else ""),
        daemon=True,
    ).start()
    try:
        line = result.get(timeout=60)
        value: Any = json.loads(line)
    except (queue.Empty, ValueError) as exc:
        raise BaseRuntimeSmokeError("Supervisor did not emit a ready envelope") from exc
    if not isinstance(value, dict) or value.get("ready") is not True:
        raise BaseRuntimeSmokeError("Supervisor ready envelope is invalid")
    return value


def verify_base_runtime(artifacts_dir: Path) -> dict[str, Any]:
    root = artifacts_dir.resolve(strict=True)
    manifest_path = root / "runtime-manifest.json"
    manifest = _load_manifest(manifest_path)
    installer = manifest["installer"]
    offline = _offline_environment()
    with tempfile.TemporaryDirectory(prefix="vibeocr-base-smoke-") as temporary:
        smoke_root = Path(temporary)
        executable = smoke_root / "vibeocr-runtime-installer.exe"
        with zipfile.ZipFile(root / installer["archive"]) as archive:
            executable.write_bytes(archive.read(installer["executable_path"]))
        component_lock = smoke_root / "component-lock.json"
        component_lock.write_text(
            json.dumps(
                _component_lock(manifest, manifest_sha256=_sha256(manifest_path)),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        product = smoke_root / "product"
        request = _base_ensure_request(
            product_root=product,
            component_lock=component_lock,
            runtime_manifest=manifest_path,
        )
        first = _run_installer(executable, request, cwd=root, environment=offline)
        launch = first.get("launch")
        if not isinstance(launch, dict):
            raise BaseRuntimeSmokeError("base ensure returned no launch contract")
        runtime_root = Path(launch["environment"]["VIBEOCR_RUNTIME_ROOT"])
        installed_marker = runtime_root / ".installed.json"
        marker_before = installed_marker.read_bytes()
        mtime_before = installed_marker.stat().st_mtime_ns
        second = _run_installer(executable, request, cwd=root, environment=offline)
        if second.get("launch") is None:
            raise BaseRuntimeSmokeError("idempotent base ensure returned no launch")
        if (
            installed_marker.read_bytes() != marker_before
            or installed_marker.stat().st_mtime_ns != mtime_before
        ):
            raise BaseRuntimeSmokeError("idempotent base ensure rewrote its marker")

        runtime_python = Path(launch["python_executable"])
        sample_png = smoke_root / "rapidocr.png"
        sample_pdf = smoke_root / "sample.pdf"
        prepare = subprocess.run(
            [
                str(runtime_python),
                "-c",
                (
                    "from PIL import Image,ImageDraw,ImageFont;import fitz,sys;"
                    "im=Image.new('RGB',(900,220),'white');"
                    "f=ImageFont.truetype(r'C:\\\\Windows\\\\Fonts\\\\arial.ttf',96);"
                    "ImageDraw.Draw(im).text((30,50),'VibeOCR 123',fill='black',font=f);"
                    "im.save(sys.argv[1]);d=fitz.open();d.new_page();d.save(sys.argv[2])"
                ),
                str(sample_png),
                str(sample_pdf),
            ],
            env={**offline, **launch["environment"]},
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if prepare.returncode != 0:
            raise BaseRuntimeSmokeError(
                f"base fixture creation failed: {prepare.stderr}"
            )

        token = secrets.token_urlsafe(32)
        supervisor_env = {
            **offline,
            **launch["environment"],
            "VIBEOCR_SUP_TOKEN": token,
            "VIBEOCR_SUP_ROOT": str(product / "state"),
        }
        process = subprocess.Popen(
            [str(runtime_python), "-m", "vibeocr.backend.supervisor.main"],
            env=supervisor_env,
            cwd=runtime_root,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            ready = _read_ready_line(process)
            url = f"http://127.0.0.1:{ready['port']}"
            health = _json_request(f"{url}/v2/health", token)
            if health.get("ready") is not True:
                raise BaseRuntimeSmokeError("Supervisor health is not ready")
            job_id = _submit_ocr(url, token, sample_png)
            _wait_for_ocr(url, token, job_id)
            opened = _json_request(
                f"{url}/v2/pdf/sessions/open",
                token,
                method="POST",
                body={"path": str(sample_pdf)},
            )
            session_id = opened.get("session_id")
            if not isinstance(session_id, str):
                raise BaseRuntimeSmokeError("PDF open returned no session")
            _json_request(
                f"{url}/v2/pdf/sessions/{session_id}/model",
                token,
                method="POST",
                body={},
            )
            _json_request(
                f"{url}/v2/pdf/sessions/{session_id}/close",
                token,
                method="POST",
                body={},
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        return {"base_ensure": True, "rapidocr": True, "pdf": True, "reuse": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts_dir", type=Path)
    args = parser.parse_args(argv)
    result = verify_base_runtime(args.artifacts_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
