"""MinerU 4 local REST boundary: uploads, cancelable jobs and owned outputs."""

from __future__ import annotations

import logging
import mimetypes
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import quote

import httpx
from vibeocr.runtime_contracts import MineruConfig

logger = logging.getLogger(__name__)


class MineruApiError(RuntimeError):
    """An upstream failure; never a reason to select another quality tier."""


class MineruCancelled(MineruApiError):
    """The caller or upstream canceled this job."""


def object_value(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise MineruApiError(f"Invalid MinerU {label}: expected object")
    return value


def string_value(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MineruApiError(f"Invalid MinerU {label}: expected nonempty string")
    return value


@dataclass(frozen=True)
class MineruDocument:
    """Downloaded outputs survive cleanup of the upstream job's files."""

    markdown: str
    structured_content: dict[str, object]
    middle_json: dict[str, object]
    archive: bytes


class MineruApiClient:
    def __init__(
        self,
        url: str,
        *,
        timeout: float = 3600,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.transport = transport

    def parse(
        self,
        files: list[tuple[str, bytes]],
        config: MineruConfig,
        *,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> dict[str, MineruDocument | MineruApiError]:
        if not files:
            return {}
        if len({name for name, _ in files}) != len(files):
            raise ValueError("MinerU input names must be unique")
        deadline = time.monotonic() + self.timeout
        owned_files: set[str] = set()
        pending_uploads: set[str] = set()
        job_id: str | None = None
        terminal = False
        with httpx.Client(
            base_url=self.url,
            trust_env=False,
            transport=self.transport,
            timeout=httpx.Timeout(15, connect=5),
        ) as client:

            def request(method: str, path: str, **kwargs: object) -> httpx.Response:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("MinerU job deadline exceeded")
                response = client.request(
                    method, path, timeout=min(15, remaining), **kwargs
                )
                if response.is_error:
                    raise MineruApiError(
                        f"MinerU {method} {path}: HTTP {response.status_code} "
                        f"{response.text[:500]}"
                    )
                return response

            def json_request(
                method: str, path: str, **kwargs: object
            ) -> dict[str, object]:
                response = request(method, path, **kwargs)
                try:
                    return object_value(response.json(), "response")
                except ValueError as exc:
                    raise MineruApiError("MinerU returned invalid JSON") from exc

            def check_cancelled() -> None:
                if cancelled():
                    raise MineruCancelled("MinerU job canceled by caller")

            try:
                sources: list[dict[str, object]] = []
                for name, data in files:
                    check_cancelled()
                    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
                    upload = json_request(
                        "POST",
                        "/v1/uploads",
                        json={"filename": name, "bytes": len(data), "mime_type": mime},
                    )
                    upload_id = string_value(upload.get("id"), "upload id")
                    pending_uploads.add(upload_id)
                    path = "/v1/uploads/" + quote(upload_id, safe="")
                    # Never follow an upstream-provided arbitrary upload URL.
                    request("PUT", path + "/content", content=data)
                    completed = json_request("POST", path + "/complete", json={})
                    file = object_value(completed.get("file"), "completed upload file")
                    file_id = string_value(file.get("id"), "file id")
                    owned_files.add(file_id)
                    pending_uploads.remove(upload_id)
                    entry: dict[str, object] = {
                        "source": {"type": "file_id", "file_id": file_id}
                    }
                    if name.lower().endswith(".pdf"):
                        entry["page_range"] = config.page_range
                    elif config.page_range != "all":
                        raise MineruApiError("Page ranges require a PDF input")
                    sources.append(entry)
                check_cancelled()
                job = json_request(
                    "POST",
                    "/v1/parse/jobs",
                    json={
                        "files": sources,
                        "tier": config.tier.value,
                        "ocr_mode": config.ocr_mode.value,
                        "output_formats": [
                            "markdown",
                            "middle_json",
                            "structured_content",
                            "zip",
                        ],
                    },
                )
                job_id = string_value(job.get("job_id"), "job id")
                path = "/v1/parse/jobs/" + quote(job_id, safe="")
                while job.get("status") in ("queued", "running"):
                    check_cancelled()
                    time.sleep(min(0.25, max(0, deadline - time.monotonic())))
                    job = json_request("GET", path)
                status = job.get("status")
                if status not in ("completed", "partial", "failed", "canceled"):
                    raise MineruApiError(f"Unknown MinerU job status: {status}")
                terminal = True
                if job.get("tier") != config.tier.value:
                    raise MineruApiError("MinerU returned a different tier")
                if status == "canceled":
                    raise MineruCancelled("MinerU job canceled upstream")
                results = job.get("files")
                if not isinstance(results, list):
                    raise MineruApiError("MinerU job has no file results")
                by_name: dict[str, MineruDocument | MineruApiError] = {}
                expected = {name for name, _ in files}
                for raw in results:
                    result = object_value(raw, "file result")
                    name = string_value(result.get("name"), "file name")
                    if name not in expected or name in by_name:
                        raise MineruApiError("MinerU result names do not match inputs")
                    outputs = object_value(result.get("output_files") or {}, "outputs")
                    for ref in outputs.values():
                        if ref is not None:
                            owned_files.add(
                                string_value(
                                    object_value(ref, "output reference").get(
                                        "file_id"
                                    ),
                                    "output file id",
                                )
                            )
                    if result.get("status") != "completed":
                        by_name[name] = MineruApiError(
                            f"MinerU file failed: {result.get('error')}"
                        )
                        continue

                    def download(kind: str) -> httpx.Response:
                        check_cancelled()
                        ref = object_value(outputs.get(kind), kind + " reference")
                        file_id = string_value(ref.get("file_id"), "output file id")
                        response = request(
                            "GET", "/v1/files/" + quote(file_id, safe="") + "/content"
                        )
                        if ref.get("bytes") != len(response.content):
                            raise MineruApiError("MinerU output length mismatch")
                        return response

                    by_name[name] = MineruDocument(
                        markdown=download("markdown").text,
                        middle_json=object_value(
                            download("middle_json").json(), "middle JSON"
                        ),
                        structured_content=object_value(
                            download("structured_content").json(), "structured content"
                        ),
                        archive=download("zip").content,
                    )
                for name in expected - by_name.keys():
                    by_name[name] = MineruApiError("MinerU omitted an input result")
                return by_name
            finally:
                # Only resources created by this call are eligible for cleanup.
                cleanup_files = job_id is None or terminal
                if job_id is not None and not terminal:
                    try:
                        response = client.delete(
                            "/v1/parse/jobs/" + quote(job_id, safe="")
                        )
                        response.raise_for_status()
                        cleanup_files = response.json().get("status") in (
                            "completed",
                            "partial",
                            "failed",
                            "canceled",
                        )
                    except (httpx.HTTPError, ValueError, AttributeError):
                        logger.warning("MinerU job cancellation could not be confirmed")
                for upload_id in pending_uploads:
                    try:
                        client.post(
                            "/v1/uploads/" + quote(upload_id, safe="") + "/cancel"
                        ).raise_for_status()
                    except httpx.HTTPError:
                        logger.debug("MinerU upload cleanup failed", exc_info=True)
                # Pending cancellation must retain inputs while the worker runs.
                # MinerU owns their expiry if it cannot confirm a terminal state.
                for file_id in owned_files if cleanup_files else ():
                    try:
                        client.delete(
                            "/v1/files/" + quote(file_id, safe="")
                        ).raise_for_status()
                    except httpx.HTTPError:
                        logger.debug("MinerU output cleanup failed", exc_info=True)
