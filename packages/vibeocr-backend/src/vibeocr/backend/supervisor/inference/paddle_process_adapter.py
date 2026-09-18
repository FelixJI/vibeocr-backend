"""Paddle adapter proxy; only the dedicated worker imports Paddle or its cv2."""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import subprocess
import sys
import threading
from dataclasses import asdict
from pathlib import Path

from vibeocr.backend.utils.job_object import JobObjectGuard
from vibeocr.runtime_contracts import (
    PipelineSelection,
    ResidencyStatus,
    SettingsSnapshot,
    parse_pipeline_spec,
    parse_residency_entry,
)
from vibeocr.runtime_contracts.dtos import OcrEngine

from .budgets import AdapterCapability, InputItem
from .ocr_engines import (
    REASON_ENGINE_NOT_INSTALLED,
    EngineAvailability,
    EngineDescriptor,
)
from .paddle_adapter import PADDLE_REQUIRED_COMPONENT

logger = logging.getLogger(__name__)


def paddle_python() -> Path | None:
    root = Path(sys.prefix) / "engines" / "paddle"
    return next(
        (
            p
            for p in (
                root / "python.exe",
                root / "Scripts/python.exe",
                root / "bin/python",
            )
            if p.is_file()
        ),
        None,
    )


class PaddleProcessAdapter:
    engine_id = OcrEngine.PADDLEOCR
    included_in_base = False

    def __init__(self, python: Path) -> None:
        self.python = python
        self._process: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._responses: queue.Queue = queue.Queue()
        self._guard: JobObjectGuard | None = None
        self._status = ResidencyStatus()
        self._settings = SettingsSnapshot()

    @staticmethod
    def probe_descriptor() -> EngineDescriptor:
        python = paddle_python()
        available = False
        if python is not None:
            try:
                result = subprocess.run(
                    [str(python), "-I", "-c", "import paddle, paddleocr"],
                    capture_output=True,
                    timeout=60,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                available = result.returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                pass
        return EngineDescriptor(
            engine_id=OcrEngine.PADDLEOCR,
            availability=EngineAvailability.READY
            if available
            else EngineAvailability.PREPARATION_REQUIRED,
            included_in_base=False,
            reason_code=None if available else REASON_ENGINE_NOT_INSTALLED,
            required_component=None if available else PADDLE_REQUIRED_COMPONENT,
        )

    def descriptor(self) -> EngineDescriptor:
        return self.probe_descriptor()

    def _start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._stop()
        responses = queue.Queue()
        self._responses = responses
        env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONUTF8="1")
        env.pop("PYTHONPATH", None)
        process = subprocess.Popen(
            [
                str(self.python),
                "-I",
                "-m",
                "vibeocr.backend.supervisor.inference.paddle_worker",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self._process = process
        self._guard = JobObjectGuard(name="vibeocr_paddle")
        self._guard.assign_from_popen(process)

        def read_replies() -> None:
            try:
                for line in process.stdout:
                    responses.put(line)
            finally:
                responses.put(None)

        def read_logs() -> None:
            for line in process.stderr:
                logger.debug("[Paddle worker] %s", line.rstrip())

        threading.Thread(target=read_replies, daemon=True).start()
        threading.Thread(target=read_logs, daemon=True).start()
        # Restore settings after a native crash, without replaying the failed job.
        self._exchange(
            {"operation": "settings", "settings": self._settings.to_payload()}
        )

    def _exchange(self, request: dict, *, timeout: float = 3600) -> object:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("Paddle worker is not running")
        try:
            process.stdin.write(json.dumps(request, ensure_ascii=True) + "\n")
            process.stdin.flush()
        except (OSError, ValueError):
            self._stop()
            raise
        try:
            line = self._responses.get(timeout=timeout)
        except queue.Empty as exc:
            self._stop()
            raise TimeoutError("Paddle worker deadline exceeded") from exc
        if line is None:
            self._stop()
            raise RuntimeError("Paddle worker exited before replying")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]

    def _request(self, operation: str, **payload: object) -> object:
        with self._lock:
            self._start()
            return self._exchange({"operation": operation, **payload})

    def capabilities(
        self, options: PipelineSelection | None = None
    ) -> AdapterCapability:
        result = self._request(
            "capabilities", options=(options or PipelineSelection("OCR")).to_payload()
        )
        return AdapterCapability(**result)

    def recognize_many(
        self,
        items: list[InputItem],
        *,
        options: PipelineSelection | None = None,
        compute_batch: object | None = None,
    ) -> list[dict]:
        del compute_batch  # The native Paddle adapter schedules its own batches.
        wire_items = [
            {**asdict(item), "data": base64.b64encode(item.data).decode("ascii")}
            for item in items
        ]
        result = self._request(
            "recognize",
            items=wire_items,
            options=(options or PipelineSelection("OCR")).to_payload(),
        )
        if not isinstance(result, list):
            raise RuntimeError("Invalid Paddle worker result")
        return result

    def _record_status(self, result: dict) -> ResidencyStatus:
        self._status = ResidencyStatus(
            default_ttl_seconds=result["default_ttl_seconds"],
            entries=tuple(parse_residency_entry(p) for p in result["entries"]),
            pipelines=tuple(
                parse_pipeline_spec(p) for p in result.get("pipelines", [])
            ),
        )
        return self._status

    def residency_status(self) -> ResidencyStatus:
        # Observing status must not queue behind inference/model downloads.
        if not self._lock.acquire(timeout=0.05):
            return self._status
        try:
            if self._process is None:
                return self._status
            return self._record_status(
                self._exchange({"operation": "status"}, timeout=1)
            )
        finally:
            self._lock.release()

    def preload(self, pipelines: tuple[str, ...]) -> ResidencyStatus:
        return self._record_status(self._request("preload", pipelines=pipelines))

    def release_idle(self, pipeline: str | None = None) -> ResidencyStatus:
        if not self._lock.acquire(blocking=False):
            return self._status
        try:
            if self._process is None:
                return self._status
            return self._record_status(
                self._exchange({"operation": "release", "pipeline": pipeline})
            )
        finally:
            self._lock.release()

    def configure_settings(self, snapshot: SettingsSnapshot) -> None:
        with self._lock:
            self._settings = snapshot
            if self._process is not None:
                self._record_status(
                    self._exchange(
                        {"operation": "settings", "settings": snapshot.to_payload()}
                    )
                )

    def _stop(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        if self._guard is not None:
            self._guard.close()
            self._guard = None
        self._status = ResidencyStatus(
            default_ttl_seconds=self._settings.default_ttl_seconds,
            pipelines=self._settings.pipelines,
        )

    def close(self) -> None:
        with self._lock:
            self._stop()
