"""Backend-owned portable runtime installer.

The public surface is intentionally small: ``inspect``, ``ensure`` and ``repair``.
Frontends receive paths and integrity state,
never dependency names, index URLs or pip arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import queue
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from importlib.resources import files
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

from vibeocr.backend.runtime_install_plan import (
    CAPABILITY,
    create_plan,
    read_plan,
    validate_plan,
)
from vibeocr.backend.runtime_layout import resolve_runtime_store
from vibeocr.backend.runtime_lock import RuntimeLockTimeout, RuntimeStoreLock
from vibeocr.backend.runtime_maintenance import (
    EventSink,
    RuntimeCapabilityError,
    RuntimeCommandConflict,
    RuntimeCursorExpired,
    RuntimeInstallFailure,
    RuntimeInstallPlanBlocked,
    RuntimeInstallPlanStale,
    RuntimeMaintenanceReporter,
    RuntimeOperationCancelled,
    RuntimeOperationConflict,
    RuntimeOperationError,
    RuntimeOperationNotCancellable,
    RuntimeOperationNotFound,
    RuntimeOperationNotRetryable,
    RuntimeOperationStore,
    RuntimeSourceIdentityMismatch,
    declared_installed_closure,
    probe_runtime_components,
    profile_descriptor,
    runtime_profile_status,
    runtime_source_identity,
)
from vibeocr.backend.runtime_manifest import (
    ACCELERATOR_TO_PLAN,
    ManifestError,
    RuntimeInstallScope,
    RuntimeManifest,
    covering_profile_id,
    load_runtime_manifest,
    migrate_legacy_component_ids,
)
from vibeocr.backend.runtime_selection import (
    BASE_PROFILE,
    BoundDownloadSource,
    RuntimeSelectionError,
    RuntimeSelectionPolicy,
)
from vibeocr.backend.utils.job_object import JobObjectGuard


class RuntimeInstallError(RuntimeInstallFailure):
    pass


class RuntimeIdentityMismatch(RuntimeInstallError, RuntimeSourceIdentityMismatch):
    """The verified Runtime source identity differs from the bound intent."""


class RuntimeCapabilityUnavailable(RuntimeInstallError, RuntimeCapabilityError):
    """A requested capability is absent from the verified Runtime manifest."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


PROTOCOL_VERSION = 2


@dataclass(frozen=True, slots=True)
class RuntimeState:
    status: str
    runtime_root: str
    accelerator: str
    manifest_sha256: str
    backend_version: str
    integrity: str
    source: dict[str, str]


@dataclass(frozen=True, slots=True)
class RuntimeInspection:
    state: RuntimeState
    profile: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RuntimeLaunch:
    python_executable: str
    supervisor_module: str
    working_directory: str
    model_root: str
    environment: dict[str, str]


InstallRunner = Callable[[Path, RuntimeManifest, str], Path]
ComponentProbe = Callable[[Path, tuple[str, ...], str], dict[str, bool]]


def _default_component_probe(
    runtime_root: Path, component_ids: tuple[str, ...], profile_id: str
) -> dict[str, bool]:
    return probe_runtime_components(runtime_root, component_ids, profile_id=profile_id)


_OUTPUT_TAIL_MAX_LINES = 20
_OUTPUT_TAIL_MAX_CHARS = 4000


def _child_output_tail(*streams: str) -> str:
    """Return the last non-empty lines of the first informative stream.

    pip 把错误写进 stderr、下载进度写进 stdout，失败排障需要这份输出；
    优先 stderr，全部为空时返回空串。tail 只拼进异常消息（随 journal 与
    错误信封持久化），绝不写入 NDJSON stdout。
    """

    for stream in streams:
        lines = [line.strip() for line in stream.splitlines() if line.strip()]
        if not lines:
            continue
        tail = "\n".join(
            _safe_child_text(line) for line in lines[-_OUTPUT_TAIL_MAX_LINES:]
        )
        if len(tail) > _OUTPUT_TAIL_MAX_CHARS:
            tail = tail[-_OUTPUT_TAIL_MAX_CHARS:]
        return "\n" + tail
    return ""


_CHILD_HEARTBEAT_INTERVAL_SECONDS = 5.0
_RESOLVE_NETWORK_TIMEOUT_SECONDS = 30
_RESOLVE_NETWORK_RETRIES = 2
_RESOLVE_IDLE_TIMEOUT_SECONDS = 300
_RESOLVE_TOTAL_TIMEOUT_SECONDS = 1800
_CHILD_DETAIL_MAX_CHARS = 160
# pip 在非交互管道下按行输出解析/下载/安装状态；进度条等噪声行不匹配
# 前缀，自然被忽略。明细行只经 maintenance 事件的 ``fallback_message``
# 回报，绝不把包管理器的原始输出 bulk 写进 NDJSON stdout。
_CHILD_STATUS_PREFIXES = (
    "Collecting ",
    "Downloading ",
    "Using cached ",
    "Installing collected packages",
    "Successfully installed",
    "Preparing metadata",
    "Installing build dependencies",
    "Getting requirements to build",
    "Building wheel",
    "Would install ",
    "Progress ",
)


def _child_status_detail(line: str) -> str | None:
    """Return one displayable package-manager status line, if any."""

    text = _safe_child_text(line.strip())
    if text.startswith(_CHILD_STATUS_PREFIXES):
        return text[:_CHILD_DETAIL_MAX_CHARS]
    return None


def _safe_child_text(text: str) -> str:
    """Remove locations and credentials before output reaches events or errors."""
    text = re.sub(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s<>\"']+", "[url]", text)
    text = re.sub(
        r"(?i)\b(?:authorization|password|token|api[_-]?key)\s*[:=]\s*(?:Bearer\s+)?\S+",
        "[credential]",
        text,
    )
    text = re.sub(r"(?i)(?:[a-z]:[\\/]|\\\\)[^\r\n\"']+", "[path]", text)
    text = re.sub(r"(?<![\w])/(?:[^\s/]+/)*[^\s,;)]+", "[path]", text)
    return text


def _drain_child_lines(
    stream: TextIO,
    lines: deque[str],
    line_queue: queue.Queue[str],
    done: threading.Event,
) -> None:
    # Bound both memory and individual lines. Never publish fragments of a long
    # line: a credential could straddle the read boundary.
    dropping = False
    try:
        while line := stream.readline(_OUTPUT_TAIL_MAX_CHARS + 1):
            too_long = len(line) > _OUTPUT_TAIL_MAX_CHARS
            if dropping or too_long:
                dropping = not line.endswith("\n")
                continue
            safe = _safe_child_text(line)
            lines.append(safe)
            try:
                line_queue.put_nowait(safe)
            except queue.Full:
                # Stale display output may be discarded, but draining must never
                # block cancellation. The independent stderr/stdout tails remain.
                try:
                    line_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    line_queue.put_nowait(safe)
                except queue.Full:
                    pass
    finally:
        stream.close()
        done.set()


def _run_install_command(
    command: list[str],
    *,
    timeout: float,
    env: dict[str, str],
    reporter: RuntimeMaintenanceReporter | None,
    heartbeat_code: str,
    idle_timeout: float | None = None,
) -> None:
    """Supervise one Python command with bounded diagnostics and owned cleanup.

    On Windows a Python gate waits on stdin until its anonymous Job is assigned.
    Only then can pip or its build children start. Closing the Job also kills
    descendants whose parent has already exited; unrelated operations are safe.
    """
    started = time.monotonic()
    last_activity = started
    last_detail: str | None = None
    snapshot = reporter.snapshot if reporter is not None else None
    source_ids = (snapshot or {}).get("effective_download_source_ids", [])
    operation_id = reporter.operation_id if reporter is not None else None
    resolver = heartbeat_code == "runtime.resolve_packages"
    tails = [deque[str](maxlen=_OUTPUT_TAIL_MAX_LINES) for _ in range(2)]

    def diagnostic(reason: str) -> str:
        now = time.monotonic()
        return (
            f"{heartbeat_code} reason={reason}; elapsed={now - started:.2f}s; "
            f"last_activity={now - last_activity:.2f}s ago; "
            f"operation={operation_id or 'unbound'}; "
            f"sources={','.join(source_ids) or 'unbound'}; "
            f"detail={last_detail or 'no effective activity'}"
        )

    if reporter is not None:
        reporter.check_cancelled(fallback_message=diagnostic("cancelled"))
    guard = JobObjectGuard(allow_breakaway=False)
    launched = command
    if os.name == "nt":
        launched = [
            command[0],
            "-c",
            (
                "import subprocess,sys; "
                "gate=sys.stdin.buffer.read(1); "
                "sys.exit(subprocess.call(sys.argv[1:], stdin=subprocess.DEVNULL) "
                "if gate == b'1' else 125)"
            ),
            *command,
        ]
    try:
        process = subprocess.Popen(
            launched,
            env=env,
            stdin=subprocess.PIPE if os.name == "nt" else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=os.name != "nt",
        )
    except OSError:
        if reporter is not None:
            reporter.clear_cancellation_detail()
        guard.close()
        raise RuntimeInstallError(diagnostic("spawn_failed")) from None
    readers: list[threading.Thread] = []
    done = [threading.Event(), threading.Event()]
    line_queue: queue.Queue[str] = queue.Queue(maxsize=64)
    try:
        if os.name == "nt":
            if not guard.assign_from_popen(process):
                raise RuntimeInstallError(diagnostic("process_containment_failed"))
            assert process.stdin is not None
            process.stdin.write("1")
            process.stdin.close()
        assert process.stdout is not None and process.stderr is not None
        for index, stream in enumerate((process.stdout, process.stderr)):
            reader = threading.Thread(
                target=_drain_child_lines,
                args=(stream, tails[index], line_queue, done[index]),
                daemon=True,
            )
            readers.append(reader)
            reader.start()
        last_heartbeat = started
        if resolver and reporter is not None:
            reporter.child_status_detail(
                message_code=heartbeat_code, fallback_message=diagnostic("started")
            )
        while True:
            now = time.monotonic()
            if reporter is not None:
                reporter.check_cancelled(
                    fallback_message=diagnostic("cancelled")
                    + _child_output_tail("".join(tails[1]), "".join(tails[0]))
                )
            if (
                all(event.is_set() for event in done)
                and line_queue.empty()
                and process.poll() is not None
            ):
                break
            reason = None
            if now - started >= timeout:
                reason = "total_timeout"
            elif idle_timeout is not None and now - last_activity >= idle_timeout:
                reason = "idle_timeout"
            if reason is not None:
                raise RuntimeInstallError(diagnostic(reason))
            try:
                line = line_queue.get(
                    timeout=min(0.1, max(0.001, timeout - (now - started)))
                )
            except queue.Empty:
                line = ""
            detail = _child_status_detail(line)
            if detail is not None and detail != last_detail:
                last_detail = detail
                last_activity = time.monotonic()
                if reporter is not None:
                    reporter.child_status_detail(
                        message_code=heartbeat_code,
                        fallback_message=diagnostic("activity") if resolver else detail,
                    )
            now = time.monotonic()
            if (
                reporter is not None
                and now - last_heartbeat >= _CHILD_HEARTBEAT_INTERVAL_SECONDS
            ):
                reporter.heartbeat(
                    message_code=heartbeat_code,
                    fallback_message=diagnostic("running") if resolver else None,
                )
                last_heartbeat = now
        if process.returncode != 0:
            raise RuntimeInstallError(
                f"{heartbeat_code} failed with exit code {process.returncode}; "
                + diagnostic("exit_nonzero")
            )
        if resolver and reporter is not None:
            reporter.child_status_detail(
                message_code=heartbeat_code, fallback_message=diagnostic("succeeded")
            )
    except RuntimeInstallError as exc:
        raise RuntimeInstallError(
            str(exc) + _child_output_tail("".join(tails[1]), "".join(tails[0]))
        ) from None
    finally:
        if reporter is not None:
            reporter.clear_cancellation_detail()
        guard.close()
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        for reader in readers:
            reader.join(timeout=5)
        if not readers:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
        if process.stdin is not None:
            process.stdin.close()


def _default_accelerator(component_lock: dict[str, Any]) -> str:
    accelerator = component_lock["backend"].get("accelerator")
    if accelerator not in ACCELERATOR_TO_PLAN:
        raise RuntimeInstallError(f"unsupported accelerator: {accelerator}")
    return accelerator


def _load_component_lock(path: Path) -> dict[str, Any]:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeInstallError(f"invalid component lock: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise RuntimeInstallError("component lock schema_version must be 1")
    protocol = value.get("protocol")
    backend = value.get("backend")
    if not isinstance(protocol, dict) or not isinstance(backend, dict):
        raise RuntimeInstallError("component lock requires protocol and backend")
    required = {
        "repository",
        "version",
        "artifact_sha256",
        "runtime_manifest_sha256",
        "accelerator",
    }
    if not required.issubset(backend):
        raise RuntimeInstallError("component lock backend binding is incomplete")
    if not {
        "repository",
        "version",
        "manifest_sha256",
    }.issubset(protocol):
        raise RuntimeInstallError("component lock Protocol binding is incomplete")
    if protocol["repository"] != "FelixJI/vibeocr-protocol":
        raise RuntimeInstallError("component lock Protocol repository is invalid")
    if backend["repository"] != "FelixJI/vibeocr-backend":
        raise RuntimeInstallError("component lock Backend repository is invalid")
    if not re.fullmatch(r"2\.\d+\.\d+", str(protocol["version"])):
        raise RuntimeInstallError("component lock Protocol version is invalid")
    if not re.fullmatch(r"\d+\.\d+\.\d+", str(backend["version"])):
        raise RuntimeInstallError("component lock Backend version is invalid")
    for record, field in (
        (protocol, "manifest_sha256"),
        (backend, "artifact_sha256"),
        (backend, "runtime_manifest_sha256"),
    ):
        if not _SHA256_RE.fullmatch(str(record[field])):
            raise RuntimeInstallError(f"component lock {field} is invalid")
    if backend.get("accelerator") not in ACCELERATOR_TO_PLAN:
        raise RuntimeInstallError("component lock Backend accelerator is invalid")
    capabilities = value.get("required_capabilities")
    if not isinstance(capabilities, list) or not all(
        isinstance(item, str) and item for item in capabilities
    ):
        raise RuntimeInstallError("component lock required_capabilities is invalid")
    return value


def _python_in(runtime_root: Path) -> Path:
    candidates = (
        runtime_root / "Scripts" / "python.exe",
        runtime_root / "python.exe",
        runtime_root / "bin" / "python",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


_LOCK_DECLARATION_RE = re.compile(r"(?m)^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:==|@)")
_LOCK_HASH_RE = re.compile(r"--hash=sha256:([0-9a-f]{64})")
_DOWNLOAD_EVENT_MIN_BYTES = 4 * 1024 * 1024
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_DOWNLOAD_HTTP_TIMEOUT_SECONDS = 60.0


def _normalize_dist_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True, slots=True)
class _ResolvedArtifact:
    name: str
    url: str
    sha256: str | None
    filename: str


def _lock_allowed_hashes(lock_path: Path) -> dict[str, set[str]]:
    """Parse the per-distribution sha256 allow-list from an exact lock."""
    allowed: dict[str, set[str]] = {}
    block: list[str] = []
    for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line.endswith("\\"):
            block.append(line[:-1])
            continue
        block.append(line)
        joined = " ".join(block)
        block = []
        match = _LOCK_DECLARATION_RE.match(joined)
        if match is None:
            continue
        allowed.setdefault(_normalize_dist_name(match.group(1)), set()).update(
            _LOCK_HASH_RE.findall(joined)
        )
    return allowed


def _parse_resolve_report(
    report_path: Path, download_root: Path | None = None
) -> tuple[_ResolvedArtifact, ...]:
    try:
        document = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeInstallError("pip resolve report is unreadable") from exc
    items = document.get("install") if isinstance(document, dict) else None
    if not isinstance(items, list) or not items:
        raise RuntimeInstallError("pip resolve report has no artifacts")
    artifacts: list[_ResolvedArtifact] = []
    for item in items:
        if not isinstance(item, dict):
            raise RuntimeInstallError("pip resolve report artifact is invalid")
        metadata = item.get("metadata")
        download = item.get("download_info")
        archive = download.get("archive_info") if isinstance(download, dict) else None
        hashes = archive.get("hashes") if isinstance(archive, dict) else None
        sha256 = hashes.get("sha256") if isinstance(hashes, dict) else None
        url = download.get("url") if isinstance(download, dict) else None
        name = metadata.get("name") if isinstance(metadata, dict) else None
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(url, str)
            or not url.startswith(("http://", "https://", "file://"))
            or (sha256 is not None and not isinstance(sha256, str))
        ):
            raise RuntimeInstallError("pip resolve report artifact is invalid")
        if url.startswith("file://"):
            parsed = urllib.parse.urlsplit(url)
            local = Path(urllib.request.url2pathname(parsed.path)).resolve()
            if (
                download_root is None
                or parsed.netloc not in {"", "localhost"}
                or local.parent != download_root.resolve()
            ):
                raise RuntimeInstallError("resolve artifact is outside download cache")
        filename = urllib.parse.unquote(url.rsplit("/", 1)[-1])
        if (
            not filename
            or "/" in filename
            or "\\" in filename
            or ":" in filename
            or ".." in filename
        ):
            raise RuntimeInstallError(f"pip resolve artifact url is unsafe: {url}")
        artifacts.append(_ResolvedArtifact(name, url, sha256, filename))
    return tuple(artifacts)


def _remote_content_length(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(
            request, timeout=_DOWNLOAD_HTTP_TIMEOUT_SECONDS
        ) as response:
            if response.status != 200:
                return None
            value = response.headers.get("Content-Length")
    except (OSError, ValueError):
        return None
    try:
        return int(value) if value else None
    except (TypeError, ValueError):
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_resolved_artifacts(
    artifacts: tuple[_ResolvedArtifact, ...],
    allowed_hashes: dict[str, set[str]],
    download_root: Path,
    reporter: RuntimeMaintenanceReporter | None,
) -> Path:
    """Resolve the exact lock closure and download it with byte progress.

    The download reuses previously verified files so a retry after a network
    failure only refetches the missing artifacts. Every completed file is
    verified against the lock allow-list before the offline install consumes
    it.
    """
    download_root.mkdir(parents=True, exist_ok=True)
    cached: dict[int, int] = {}
    for index, artifact in enumerate(artifacts):
        destination = download_root / artifact.filename
        if destination.is_file() and _file_sha256(destination) in allowed_hashes.get(
            _normalize_dist_name(artifact.name), set()
        ):
            cached[index] = destination.stat().st_size
    sizes = {
        index: cached.get(index)
        if index in cached
        else _remote_content_length(artifact.url)
        for index, artifact in enumerate(artifacts)
    }
    total_bytes = sum(size for size in sizes.values() if size is not None)
    known_bytes = all(size is not None for size in sizes.values())

    def report_progress(current: int, total: int, unit: str) -> None:
        if reporter is None:
            return
        reporter.advance_measured(
            phase="install_profile",
            unit=unit,  # type: ignore[arg-type]
            current=current,
            total=total,
            message_code="runtime.download_package",
        )

    if known_bytes:
        report_progress(sum(cached.values()), total_bytes, "bytes")
    else:
        report_progress(len(cached), len(artifacts), "items")

    completed_bytes = sum(cached.values())
    completed_items = len(cached)
    last_event_bytes = completed_bytes
    for index, artifact in enumerate(artifacts):
        if index in cached:
            continue
        destination = download_root / artifact.filename
        temporary = destination.with_suffix(destination.suffix + ".part")
        expected = allowed_hashes.get(_normalize_dist_name(artifact.name), set())
        digest = hashlib.sha256()
        request = urllib.request.Request(artifact.url)
        with (
            urllib.request.urlopen(
                request, timeout=_DOWNLOAD_HTTP_TIMEOUT_SECONDS
            ) as response,
            temporary.open("wb") as stream,
        ):
            while True:
                chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                stream.write(chunk)
                digest.update(chunk)
                if known_bytes:
                    completed_bytes += len(chunk)
                    if completed_bytes - last_event_bytes >= _DOWNLOAD_EVENT_MIN_BYTES:
                        report_progress(completed_bytes, total_bytes, "bytes")
                        last_event_bytes = completed_bytes
        if digest.hexdigest() not in expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeInstallError(
                f"downloaded artifact hash mismatch: {artifact.name}"
            )
        os.replace(temporary, destination)
        completed_items += 1
        if not known_bytes:
            report_progress(completed_items, len(artifacts), "items")
        elif completed_bytes > last_event_bytes:
            report_progress(completed_bytes, total_bytes, "bytes")
    return download_root


def _resolve_online_report(
    python: Path,
    lock: Path,
    endpoint: str,
    cache: Path,
    reporter: RuntimeMaintenanceReporter | None,
    env: dict[str, str],
) -> Path:
    """Resolve the lock closure with ``pip --dry-run --report``.

    ``--require-hashes`` keeps pip's resolver pinned to the exact lock
    artifacts, so the report's URLs are precisely what a direct install
    would download.
    """
    report_path = cache / "resolve" / f"{lock.stem}-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    inputs_path = report_path.with_suffix(".inputs.json")
    inputs = {"lock": lock.read_text(encoding="utf-8"), "endpoint": endpoint}
    try:
        if json.loads(inputs_path.read_text(encoding="utf-8")) == inputs:
            artifacts = _parse_resolve_report(
                report_path, cache / "downloads/artifacts"
            )
            allowed = _lock_allowed_hashes(lock)
            for artifact in artifacts:
                if artifact.url.startswith("file://"):
                    local = cache / "downloads/artifacts" / artifact.filename
                    if not local.is_file() or _file_sha256(local) not in allowed.get(
                        _normalize_dist_name(artifact.name), set()
                    ):
                        local.unlink(missing_ok=True)
                        break
            else:
                return report_path
    except (OSError, ValueError, RuntimeInstallError):
        pass
    _run_install_command(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--no-input",
            "--disable-pip-version-check",
            "--progress-bar",
            "raw",
            "--timeout",
            str(_RESOLVE_NETWORK_TIMEOUT_SECONDS),
            "--retries",
            str(_RESOLVE_NETWORK_RETRIES),
            "--resume-retries",
            str(_RESOLVE_NETWORK_RETRIES),
            "--report",
            str(report_path),
            "--find-links",
            str(cache / "downloads" / "artifacts"),
            "--index-url",
            endpoint,
            "--require-hashes",
            "-r",
            str(lock),
        ],
        # A hard ceiling still bounds repeated resolver/backtracking activity.
        # Network reads have a shorter pip bound; only effective status changes
        # reset the idle budget, never our heartbeat or arbitrary stderr noise.
        timeout=_RESOLVE_TOTAL_TIMEOUT_SECONDS,
        idle_timeout=_RESOLVE_IDLE_TIMEOUT_SECONDS,
        env={**env, "PYTHONUNBUFFERED": "1", "PIP_NO_INPUT": "1"},
        reporter=reporter,
        heartbeat_code="runtime.resolve_packages",
    )
    _parse_resolve_report(report_path, cache / "downloads/artifacts")
    inputs_path.write_text(json.dumps(inputs, sort_keys=True), encoding="utf-8")
    return report_path


def _prepare_online_artifacts(
    python: Path,
    lock: Path,
    endpoint: str,
    cache: Path,
    reporter: RuntimeMaintenanceReporter | None,
    env: dict[str, str],
) -> Path:
    """Resolve and download the online lock closure; return the artifact dir."""
    allowed = _lock_allowed_hashes(lock)
    downloads = cache / "downloads" / "artifacts"
    downloads.mkdir(parents=True, exist_ok=True)
    # Reuse only original wheels whose bytes the target lock already accepts.
    # Wheels built from sdists remain bound to their original offline pack.
    for wheel in (cache / "runtime-packs").glob("*/*.whl"):
        name = _normalize_dist_name(wheel.name.split("-", 1)[0])
        if (
            name in allowed
            and not (downloads / wheel.name).exists()
            and _file_sha256(wheel) in allowed[name]
        ):
            shutil.copyfile(wheel, downloads / wheel.name)
    return _download_resolved_artifacts(
        _parse_resolve_report(
            _resolve_online_report(python, lock, endpoint, cache, reporter, env),
            downloads,
        ),
        allowed,
        downloads,
        reporter,
    )


def _default_install_runner(
    partial_root: Path,
    manifest: RuntimeManifest,
    install_scope: RuntimeInstallScope,
    download_sources: tuple[BoundDownloadSource, ...],
    reporter: RuntimeMaintenanceReporter | None = None,
    *,
    cache_root: Path | None = None,
) -> Path:
    """Extract bound Python and install only hash-locked dependencies.

    The installer may itself be a frozen EXE, so it must never depend on
    ``sys.executable -m venv``. Protocol and Backend wheels are installed from
    the manifest directory; no editable/local workspace source is accepted.
    """
    progress: Callable[[int, int], None] | None = None
    if reporter is not None:

        def report_extraction(current: int, total: int) -> None:
            reporter.advance_measured(
                phase="prepare_runtime",
                unit="bytes",
                current=current,
                total=total,
                message_code="runtime.extract_python",
                component_id="runtime_base",
            )

        progress = report_extraction
    _extract_python_archive(
        manifest.python.archive_path,
        partial_root,
        progress=progress,
    )
    python = _python_in(partial_root)
    if not python.is_file():
        raise RuntimeInstallError("Python archive has no python.exe")
    lock = install_scope.lock_path
    runtime_pack = install_scope.runtime_pack
    portable_env = os.environ.copy()
    for name in tuple(portable_env):
        if name.upper().startswith(("PIP_", "UV_")):
            portable_env.pop(name)
    cache = cache_root or partial_root.parent.parent / "state" / "installer-cache"
    portable_env.update(
        {
            "PIP_CACHE_DIR": str(cache / "pip"),
            "UV_CACHE_DIR": str(cache / "uv"),
            "HF_HOME": str(cache / "huggingface"),
            "MODELSCOPE_CACHE": str(cache / "modelscope"),
            "TEMP": str(cache / "temp"),
            "TMP": str(cache / "temp"),
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
        }
    )
    for directory in portable_env["PIP_CACHE_DIR"], portable_env["TEMP"]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    if reporter is not None:
        reporter.advance(
            phase="install_profile",
            current=4,
            total=7,
            message_code="runtime.install_profile",
        )
    install_command = [
        str(python),
        "-m",
        "pip",
        "install",
    ]
    pack_files = [manifest.path.parent / name for name in runtime_pack]
    pack_present = bool(pack_files) and all(path.is_file() for path in pack_files)
    base_ids = {
        component.component_id
        for component in manifest.profiles[BASE_PROFILE].components
    }
    is_base_scope = set(install_scope.component_ids) == base_ids
    if runtime_pack and not pack_present and is_base_scope:
        # base 是随 Portable 携带的必备闭包：缺失即产品不完整，fail closed
        # 而不是静默联网安装（计划 §4.1）。full 闭包的 pack 由用户按需下载，
        # 未到位时回退在线安装是合法状态（cu126 因 torch 单 wheel 超过
        # Release 资产上限，长期保持在线直链路径）。
        raise RuntimeInstallError(f"runtime pack is missing: {runtime_pack[0]}")
    if pack_present:
        # 离线路径（计划 §4.2）：manifest 绑定的 runtime pack 提供完整
        # wheel 闭包，安装禁止回退公网。pack 整体字节完整性由
        # runtime_pack_sha256 在 manifest 加载时验证；解析输入用 pack 自带
        # 的无哈希 requirements 清单——原 lock 的哈希行覆盖 sdist 工件，
        # 对 pack 内由 sdist 构建的 wheel 必然不匹配。
        pack_dir = _extract_runtime_pack(
            pack_files,
            cache / "runtime-packs",
            reporter=reporter,
        )
        requirements_file = pack_dir / "pack-requirements.txt"
        install_command += [
            "--no-index",
            "--find-links",
            str(pack_dir),
            # pack 是纯 wheel 闭包，only-binary 保证离线安装永不触发本机构建。
            "--only-binary=:all:",
            "-r",
            str(requirements_file),
        ]
    else:
        package_indexes = tuple(
            source for source in download_sources if source.kind == "package_index"
        )
        if len(package_indexes) != 1:
            raise RuntimeInstallError(
                "online Runtime install requires one package_index source"
            )
        # 在线路径两步走：pip dry-run 解析出 lock 的精确工件清单，自管下载
        # 逐件校验哈希并推送字节级下载进度；最终安装把已验证工件作为
        # --find-links 输入，pip 直接复用本地文件不重复下载。不加
        # --only-binary：lock 的哈希行覆盖 index 上无 wheel 的 sdist 工件
        # （如经 omegaconf 传递的 antlr4-python3-runtime==4.9.3 只发
        # sdist），禁止 sdist 会直接解析失败。工件字节仍由 --require-hashes
        # 锁定，与 build_runtime_pack 阶段 1 的下载语义一致。
        endpoint = package_indexes[0].endpoint
        download_root = _prepare_online_artifacts(
            python, lock, endpoint, cache, reporter, portable_env
        )
        install_command += [
            "--index-url",
            endpoint,
            "--find-links",
            str(download_root),
            "--require-hashes",
            "-r",
            str(lock),
        ]
    _run_install_command(
        install_command,
        timeout=3600,
        env=portable_env,
        reporter=reporter,
        heartbeat_code="runtime.install_profile",
    )
    artifact_root = manifest.path.parent
    protocol_wheel = artifact_root / manifest.protocol_wheel
    backend_wheel = artifact_root / manifest.backend_wheel
    if not protocol_wheel.is_file() or not backend_wheel.is_file():
        raise RuntimeInstallError(
            "release directory must contain the bound Protocol and Backend wheels"
        )
    if reporter is not None:
        reporter.advance(
            phase="install_backend",
            current=5,
            total=7,
            message_code="runtime.install_backend",
            component_id="runtime_host",
        )
    _run_install_command(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            str(protocol_wheel),
            str(backend_wheel),
        ],
        timeout=600,
        env=portable_env,
        reporter=reporter,
        heartbeat_code="runtime.install_backend",
    )
    if reporter is not None:
        reporter.advance(
            phase="verify_runtime",
            current=6,
            total=7,
            message_code="runtime.verify_runtime",
        )
    _run_install_command(
        [
            str(python),
            "-c",
            (
                "import vibeocr.runtime_contracts, vibeocr.backend; "
                "import vibeocr.backend.supervisor.main"
            ),
        ],
        timeout=60,
        env=portable_env,
        reporter=reporter,
        heartbeat_code="runtime.verify_runtime",
    )
    _run_install_command(
        [str(python), "-m", "pip", "check"],
        timeout=60,
        env=portable_env,
        reporter=reporter,
        heartbeat_code="runtime.verify_runtime",
    )
    paddle_environment = install_scope.paddle_environment
    if paddle_environment is not None:
        # Both interpreters stay inside the unactivated candidate. A failure
        # leaves the old runtime active; existing activation commits them together.
        paddle_scope = replace(
            install_scope,
            scope_id="paddle-environment",
            component_ids=tuple(
                x for x in install_scope.component_ids if x.startswith("paddleocr-")
            ),
            lock_path=paddle_environment.lock_path,
            sha256=paddle_environment.sha256,
            runtime_pack=(),
            runtime_pack_sha256=(),
            paddle_environment=None,
        )
        _default_install_runner(
            partial_root / "engines" / "paddle",
            manifest,
            paddle_scope,
            download_sources,
            reporter,
            cache_root=cache,
        )
    return python


def _extract_python_archive(
    archive_path: Path,
    destination: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Safely strip the archive's ``python/`` prefix into the partial root."""
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise RuntimeInstallError("Python archive is empty")
        total_bytes = sum(member.size for member in members if member.isfile())
        extracted_bytes = 0
        reported_percent = -1
        for member in members:
            parts = Path(member.name.replace("\\", "/")).parts
            if not parts or parts[0] != "python" or ".." in parts:
                raise RuntimeInstallError(
                    f"unsafe Python archive member: {member.name}"
                )
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeInstallError(
                    f"unsupported Python archive member: {member.name}"
                )
            relative = Path(*parts[1:])
            if not relative.parts:
                continue
            if relative.is_absolute() or any(":" in part for part in relative.parts):
                raise RuntimeInstallError(
                    f"unsafe Python archive member: {member.name}"
                )
            target = destination / relative
            try:
                target.resolve().relative_to(destination.resolve())
            except ValueError as exc:
                raise RuntimeInstallError(
                    f"unsafe Python archive member: {member.name}"
                ) from exc
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeInstallError(
                    f"unsupported Python archive member: {member.name}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeInstallError(
                    f"unreadable Python archive member: {member.name}"
                )
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted_bytes += member.size
            if progress is not None and total_bytes > 0:
                percent = min(100, extracted_bytes * 100 // total_bytes)
                if percent != reported_percent:
                    progress(extracted_bytes, total_bytes)
                    reported_percent = percent


def _extract_runtime_pack(
    archive_paths: list[Path],
    cache_root: Path,
    *,
    reporter: RuntimeMaintenanceReporter | None = None,
) -> Path:
    """Idempotently extract manifest-bound runtime pack parts into the cache.

    A pack is one or more flat zips of wheels; the first part also carries
    the hash-free requirements manifest (``pack-requirements.txt``). Each
    archive's SHA-256 has already been verified by ``load_runtime_manifest``;
    extraction is guarded against unsafe members and only re-runs when the
    destination marker is missing, so repeated ensure/repair installs stay
    offline and cheap.
    """
    if not archive_paths:
        raise RuntimeInstallError("runtime pack binding is empty")
    # 分片名形如 <pack>.part01.zip:缓存目录按去掉分片后缀的公共 stem。
    stem = re.sub(r"\.part\d+$", "", archive_paths[0].stem)
    destination = cache_root / stem
    marker = destination / ".complete"
    if marker.is_file():
        return destination
    for archive_path in archive_paths:
        if not archive_path.is_file():
            raise RuntimeInstallError(f"runtime pack is missing: {archive_path.name}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    try:
        for archive_path in archive_paths:
            with zipfile.ZipFile(archive_path) as archive:
                for name in archive.namelist():
                    parts = Path(name.replace("\\", "/")).parts
                    if (
                        not parts
                        or ".." in parts
                        or Path(name).is_absolute()
                        or len(parts) != 1
                        or not (
                            name.endswith(".whl") or name == "pack-requirements.txt"
                        )
                    ):
                        raise RuntimeInstallError(f"unsafe runtime pack member: {name}")
                    archive.extract(name, destination)
    except zipfile.BadZipFile as exc:
        raise RuntimeInstallError("runtime pack archive is invalid") from exc
    if not (destination / "pack-requirements.txt").is_file():
        raise RuntimeInstallError("runtime pack lacks pack-requirements.txt")
    marker.write_text("ok\n", encoding="utf-8")
    if reporter is not None:
        reporter.advance(
            phase="install_profile",
            current=4,
            total=7,
            message_code="runtime.extract_pack",
        )
    return destination


class RuntimeInstaller:
    def __init__(
        self,
        *,
        product_root: str | Path,
        component_lock: str | Path,
        runtime_manifest: str | Path,
        accelerator: str | None = None,
        layout_manifest: str | Path | None = None,
        product_id: str | None = None,
        install_runner: InstallRunner | None = None,
        component_probe: ComponentProbe | None = None,
        event_sink: EventSink | None = None,
        lock_timeout: float = 60.0,
        operation_id: str | None = None,
        source_operation_id: str | None = None,
        component_ids: tuple[str, ...] = (),
        required_capabilities: tuple[str, ...] = (),
        install_component_ids: tuple[str, ...] | None = None,
        download_source_ids: tuple[str, ...] | None = None,
        plan_id: str | None = None,
    ) -> None:
        self.product_root = Path(product_root).resolve()
        self.component_lock_path = Path(component_lock).resolve()
        self.manifest = load_runtime_manifest(runtime_manifest)
        self.component_lock = _load_component_lock(self.component_lock_path)
        self.paths = resolve_runtime_store(
            self.product_root,
            manifest_sha256=self.manifest.sha256,
            layout_manifest=layout_manifest,
            product_id=product_id,
        )
        self._plan_id = plan_id
        self._plan_record = (
            read_plan(self.paths.state_root, plan_id) if plan_id else None
        )
        if self._plan_record is not None:
            if not operation_id or CAPABILITY not in required_capabilities:
                raise RuntimeInstallError(
                    "plan confirmation requires operation_id and capability"
                )
            if install_component_ids is not None or download_source_ids is not None:
                raise RuntimeInstallError("plan confirmation cannot override selection")
            frozen = self._plan_record["plan"]
            accelerator = frozen["accelerator"]
            base_ids = {
                item.component_id
                for item in self.manifest.profiles[BASE_PROFILE].components
            }
            install_component_ids = tuple(
                item
                for item in frozen["effective_component_ids"]
                if item not in base_ids
            )
            download_source_ids = tuple(frozen["effective_download_source_ids"])
        self.accelerator = self._select_accelerator(accelerator)
        self.plan = ACCELERATOR_TO_PLAN[self.accelerator]
        requested_components = install_component_ids
        marker = self._marker_value()
        if install_component_ids is None and marker is not None:
            known_ids = {
                item.component_id
                for item in self.manifest.profiles[self.plan].components
            }
            base_ids = {
                item.component_id
                for item in self.manifest.profiles[BASE_PROFILE].components
            }
            suffix = "cuda" if self.accelerator == "nvidia_cuda" else "cpu"
            previous_ids = migrate_legacy_component_ids(
                ACCELERATOR_TO_PLAN.get(marker.get("accelerator"), self.plan),
                tuple(marker.get("component_ids", [])),
            )
            mapped = [
                re.sub(r"-(cpu|cuda)$", f"-{suffix}", item)
                for item in previous_ids
                if isinstance(item, str)
            ]
            install_component_ids = tuple(
                item for item in mapped if item in known_ids - base_ids
            )
        self._selection = RuntimeSelectionPolicy.from_manifest(
            self.manifest
        ).plan_start(
            accelerator=self.accelerator,
            install_component_ids=install_component_ids,
            download_source_ids=download_source_ids,
        )
        if self._plan_record is not None:
            frozen = self._plan_record["plan"]
            self._selection = replace(
                self._selection,
                requested_component_ids=None
                if frozen["requested_component_ids"] is None
                else tuple(frozen["requested_component_ids"]),
                requested_download_source_ids=None
                if frozen["requested_download_source_ids"] is None
                else tuple(frozen["requested_download_source_ids"]),
            )
        elif requested_components is None:
            self._selection = replace(self._selection, requested_component_ids=None)
        self._install_scope = self._selection.requested_component_ids
        self._requested_download_source_ids = (
            self._selection.requested_download_source_ids
        )
        self._effective_download_source_ids = (
            self._selection.effective_download_source_ids
        )
        self._active_install_ids = self._selection.effective_component_ids
        available_components = {
            component.component_id
            for component in self.manifest.profiles[self.plan].components
        }
        if len(set(component_ids)) != len(component_ids) or not set(
            component_ids
        ).issubset(available_components):
            raise RuntimeInstallError("Runtime component_ids are invalid")
        missing_capabilities = set(required_capabilities).difference(
            self.manifest.capabilities
        )
        if missing_capabilities:
            raise RuntimeCapabilityUnavailable(
                f"required capabilities are unavailable: {sorted(missing_capabilities)}"
            )
        self._operation_id = operation_id
        self._source_operation_id = source_operation_id
        self._component_ids = component_ids
        self._required_capabilities = required_capabilities
        self._source = runtime_source_identity(self.manifest)
        self._runner_reports_phases = install_runner is None
        if component_probe is not None:
            self._component_probe = component_probe
        elif install_runner is None:
            self._component_probe = _default_component_probe
        else:
            self._component_probe = lambda _root, component_ids, _profile_id: {
                component_id: True for component_id in component_ids
            }
        if install_runner is None:
            self._install_runner = lambda partial, manifest, profile: (
                _default_install_runner(
                    partial,
                    manifest,
                    self._scope_for_ids(self._active_install_ids),
                    self._selection.effective_download_sources,
                    self._reporter,
                )
            )
        else:
            self._install_runner = install_runner
        self._lock_timeout = lock_timeout
        self._validate_binding()
        self._reporter = RuntimeMaintenanceReporter(
            state_root=self.paths.state_root,
            profile=profile_descriptor(
                self.manifest.profiles[self.plan],
                accelerator=self.accelerator,
            ),
            event_sink=event_sink,
        )

    def _preference_path(self) -> Path:
        return self.paths.state_root / "runtime-preference.json"

    def _select_accelerator(self, requested: str | None) -> str:
        if requested is not None:
            if requested not in ACCELERATOR_TO_PLAN:
                raise RuntimeInstallError(f"unsupported accelerator: {requested}")
            return requested
        marker = self._marker_value()
        if marker and marker.get("accelerator") in ACCELERATOR_TO_PLAN:
            return marker["accelerator"]
        try:
            value = json.loads(self._preference_path().read_text(encoding="utf-8"))
            preferred = (
                value.get("accelerator") if value.get("schema_version") == 1 else None
            )
        except (OSError, ValueError, AttributeError):
            preferred = None
        return (
            preferred
            if preferred in ACCELERATOR_TO_PLAN
            else _default_accelerator(self.component_lock)
        )

    def _validate_binding(self) -> None:
        protocol = self.component_lock["protocol"]
        backend = self.component_lock["backend"]
        protocol_version = protocol.get("version")
        if (
            not isinstance(protocol_version, str)
            or f"-{protocol_version}-" not in self.manifest.protocol_wheel
        ):
            raise RuntimeIdentityMismatch("component lock Protocol version mismatch")
        if protocol["manifest_sha256"] != self.manifest.protocol_manifest_sha256:
            raise RuntimeIdentityMismatch("component lock Protocol manifest mismatch")
        if backend["version"] != self.manifest.backend_version:
            raise RuntimeIdentityMismatch("component lock Backend version mismatch")
        if backend["artifact_sha256"] != self.manifest.backend_sha256:
            raise RuntimeIdentityMismatch("component lock Backend artifact mismatch")
        if backend["runtime_manifest_sha256"] != self.manifest.sha256:
            raise RuntimeIdentityMismatch("component lock runtime manifest mismatch")
        required = set(self.component_lock["required_capabilities"])
        missing = required.difference(self.manifest.capabilities)
        if missing:
            raise RuntimeCapabilityUnavailable(
                f"Backend is missing required capabilities: {sorted(missing)}"
            )

    def _marker(self) -> Path:
        return self.paths.runtime_root / ".installed.json"

    def _marker_value(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self._marker().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def _integrity_ok(self) -> bool:
        python = _python_in(self.paths.runtime_root)
        if not self._marker().is_file() or not python.is_file():
            return False
        return self._trusted_installed_scope_ids() is not None

    def _trusted_installed_scope_ids(self) -> tuple[str, ...] | None:
        """Read a bound, manifest-declared installed closure from the marker."""
        value = self._marker_value()
        if value is None or value.get("schema_version") != 1:
            return None
        if (
            value.get("backend_version") != self.manifest.backend_version
            or value.get("manifest_sha256") != self.manifest.sha256
            or value.get("accelerator") != self.accelerator
        ):
            return None
        installed = value.get("component_ids")
        if (
            not isinstance(installed, list)
            or not installed
            or any(not isinstance(item, str) for item in installed)
            or len(set(installed)) != len(installed)
            or not set(installed).issubset(self._profile_component_ids())
        ):
            return None
        try:
            scope = self._scope_for_ids(tuple(installed))
        except RuntimeInstallError:
            return None
        return tuple(
            item
            for item in self._profile_component_ids()
            if item in scope.component_ids
        )

    def _installed_scope_ids(self) -> tuple[str, ...]:
        """已安装闭包由可信 marker 提供，不依赖当前运行时完整。"""
        installed = self._trusted_installed_scope_ids()
        if installed is not None:
            return installed
        return self._desired_scope_ids()

    def _desired_scope_ids(self) -> tuple[str, ...]:
        """本次 intent 解析后的精确依赖闭包。"""
        return self._selection.effective_component_ids

    def _scope_for_ids(self, component_ids: tuple[str, ...]) -> RuntimeInstallScope:
        desired = set(component_ids)
        for profile_name in (BASE_PROFILE, self.plan):
            for scope in self.manifest.profiles[profile_name].scopes:
                if set(scope.component_ids) == desired:
                    return scope
        raise RuntimeInstallError("runtime manifest has no matching install scope")

    def _covering_profile(self, component_ids: tuple[str, ...]) -> str:
        return covering_profile_id(
            self.manifest,
            accelerator=self.accelerator,
            component_ids=component_ids,
        )

    def _projection_profile_id(self) -> str:
        """展示投影的 profile：可信已安装闭包用其覆盖 profile，否则 plan。"""
        closure = declared_installed_closure(
            self._marker_value(),
            manifest=self.manifest,
            accelerator=self.accelerator,
        )
        if closure is None:
            return self.plan
        return covering_profile_id(
            self.manifest,
            accelerator=self.accelerator,
            component_ids=closure,
        )

    def profile_payload(
        self, *, probe_results: dict[str, bool] | None = None
    ) -> dict[str, Any]:
        selected = self._projection_profile_id()
        component_ids = tuple(
            component.component_id
            for component in self.manifest.profiles[selected].components
        )
        return runtime_profile_status(
            self.manifest,
            accelerator=self.accelerator,
            runtime_root=self.paths.runtime_root,
            probe_results=(
                self._component_probe(self.paths.runtime_root, component_ids, selected)
                if probe_results is None
                else probe_results
            ),
            profile_id=selected,
        )

    def maintenance_snapshot(self) -> dict[str, Any] | None:
        return self._reporter.snapshot

    def _profile_component_ids(self) -> tuple[str, ...]:
        return tuple(
            component.component_id
            for component in self.manifest.profiles[self.plan].components
        )

    def _drifted_component_ids(
        self, *, probe_results: dict[str, bool] | None = None
    ) -> tuple[str, ...]:
        installed = self._installed_scope_ids()
        if not installed:
            return ()
        # 漂移探测按已安装 scope 的覆盖 profile 投影；base-only 缺少
        # Paddle/MinerU advanced components 是合法状态，不是漂移。
        covering = self._covering_profile(installed)
        payload = runtime_profile_status(
            self.manifest,
            accelerator=self.accelerator,
            runtime_root=self.paths.runtime_root,
            probe_results=(
                self._component_probe(
                    self.paths.runtime_root,
                    installed,
                    covering,
                )
                if probe_results is None
                else probe_results
            ),
            profile_id=covering,
        )
        return self._scope_drifted_from(payload)

    def _scope_drifted_from(self, profile: dict[str, Any]) -> tuple[str, ...]:
        # 漂移只对已安装闭包判定：base-only 安装缺 full 组件不是漂移。
        scope = set(self._installed_scope_ids())
        return tuple(
            component_id
            for component_id in self._drifted_component_ids_from(profile)
            if component_id in scope
        )

    @staticmethod
    def _drifted_component_ids_from(profile: dict[str, Any]) -> tuple[str, ...]:
        return tuple(
            str(component["component_id"])
            for component in profile["components"]
            if component.get("actual_state") != "ready"
        )

    def _start_operation(
        self,
        operation: str,
        *,
        effective_component_ids: tuple[str, ...] = (),
    ) -> bool:
        # ensure 的 requested 回显可选组件安装范围（None=缺省省略、
        # []=显式 base-only）；inspect/repair 保持 component_ids 语义，
        # 空集仍省略。
        requested: tuple[str, ...] | None
        if operation == "ensure":
            requested = self._install_scope
        else:
            requested = self._component_ids or None
        return self._reporter.start(
            operation,
            total_steps=7 if operation != "inspect" else 2,
            operation_id=self._operation_id,
            component_ids=requested,
            effective_component_ids=effective_component_ids,
            install_component_ids=self._install_scope
            if operation == "ensure"
            else None,
            download_source_ids=self._effective_download_source_ids,
            requested_download_source_ids=self._requested_download_source_ids,
            source=self._source,
            source_operation_id=self._source_operation_id,
            required_capabilities=self._required_capabilities,
            plan_id=self._plan_id if operation == "ensure" else None,
        )

    def inspect_snapshot(self, *, emit: bool = True) -> RuntimeInspection:
        started = self._start_operation("inspect") if emit else False
        # 单次组件探测同时供展示投影与漂移判定复用；inspect 位于产品
        # 启动关键路径，探测是真实子进程，不得加倍。探测范围与展示投影
        # 一致：已安装闭包的覆盖 profile（无可信闭包时为 accelerator
        # plan），漂移判定按同一闭包消费这份结果。
        selected = self._projection_profile_id()
        component_ids = tuple(
            component.component_id
            for component in self.manifest.profiles[selected].components
        )
        probe_results = self._component_probe(
            self.paths.runtime_root, component_ids, selected
        )
        profile = self.profile_payload(probe_results=probe_results)
        # inspect 诚实反映“已安装闭包是否 ready”，不把缺 full 可选组件
        # 当作失败（base-only 是合法的安装状态）。漂移必须按已安装闭包的
        # 覆盖 profile 投影（``_drifted_component_ids``）：plan 视角的组件
        # 集与绑定并不描述已安装的运行时，会把闭包外组件误判为 missing。
        ready = self._integrity_ok() and not self._drifted_component_ids(
            probe_results=probe_results
        )
        state = RuntimeState(
            status="ready" if ready else "missing",
            runtime_root=str(self.paths.runtime_root),
            accelerator=self.accelerator,
            manifest_sha256=self.manifest.sha256,
            backend_version=self.manifest.backend_version,
            integrity="verified" if ready else "not-installed",
            source=self._source,
        )
        if emit and started:
            self._reporter.succeed(
                phase="verify_runtime",
                current=2,
                total=2,
                message_code="runtime.inspect_complete",
            )
        return RuntimeInspection(state=state, profile=profile)

    def durable_selection_fields(self) -> dict[str, list[str] | None]:
        return self._selection.durable_intent_fields()

    def inspect(self, *, emit: bool = True) -> RuntimeState:
        return self.inspect_snapshot(emit=emit).state

    def _environment(self) -> dict[str, str]:
        state = self.paths.state_root
        environment = {
            "VIBEOCR_PRODUCT_ROOT": str(self.product_root),
            "VIBEOCR_RUNTIME_ROOT": str(self.paths.runtime_root),
            "VIBEOCR_RUNTIME_MANIFEST": str(self.manifest.path),
            "VIBEOCR_COMPONENT_LOCK": str(self.component_lock_path),
            "VIBEOCR_RUNTIME_ACCELERATOR": self.accelerator,
            # Backend inference/cache consumers still read this compatibility
            # flag.  Derive it from the Installer-owned profile instead of
            # asking a frontend shell to infer or inject the Runtime device.
            "VIBEOCR_USE_GPU": (
                "true" if self.accelerator == "nvidia_cuda" else "false"
            ),
            "VIBEOCR_RUNTIME_STATE_ROOT": str(state),
            "VIBEOCR_SUPERVISOR_SETTINGS": str(
                self.product_root / "state" / "supervisor-settings.json"
            ),
            "PIP_CACHE_DIR": str(state / "cache" / "pip"),
            "UV_CACHE_DIR": str(state / "cache" / "uv"),
            "HF_HOME": str(state / "cache" / "huggingface"),
            "MODELSCOPE_CACHE": str(state / "cache" / "modelscope"),
            "PADDLE_PDX_CACHE_HOME": str(state / "cache" / "paddlex"),
            "MINERU_HOME": str(state / "mineru4"),
            "TEMP": str(state / "temp"),
            "TMP": str(state / "temp"),
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
        }
        package_indexes = tuple(
            source
            for source in self._selection.effective_download_sources
            if source.kind == "package_index"
        )
        if package_indexes:
            environment["PIP_INDEX_URL"] = package_indexes[0].endpoint
        environment.update(self._selection.model_source_environment())
        return environment

    def _plan_baseline(self) -> dict[str, Any]:
        marker = self._marker_value()
        components = tuple((marker or {}).get("component_ids", []))
        profile = ACCELERATOR_TO_PLAN.get((marker or {}).get("accelerator"), self.plan)
        probes = (
            self._component_probe(self.paths.runtime_root, components, profile)
            if components
            else {}
        )

        def read_state(path: Path) -> object:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None

        return {
            "marker": marker,
            "probes": probes,
            "preference": read_state(self._preference_path()),
            "settings": read_state(
                self.product_root / "state" / "supervisor-settings.json"
            ),
        }

    def _installation_blockers(self) -> list[dict[str, str]]:
        blockers: list[dict[str, str]] = []
        if self._runner_reports_phases:
            if os.name != "nt" or platform.machine().lower() not in {"amd64", "x86_64"}:
                blockers.append(
                    {"code": "platform_unsupported", "next_action": "use_windows_x64"}
                )
            if (
                self.accelerator == "nvidia_cuda"
                and self._desired_scope_ids()
                != self.manifest.profiles[BASE_PROFILE].scopes[0].component_ids
            ):
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=driver_version",
                            "--format=csv,noheader",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    driver_available = result.returncode == 0 and bool(
                        result.stdout.strip()
                    )
                except (OSError, subprocess.TimeoutExpired):
                    driver_available = False
                if not driver_available:
                    blockers.append(
                        {
                            "code": "nvidia_driver_unavailable",
                            "next_action": "install_supported_driver",
                        }
                    )
            existing_parent = next(
                path
                for path in (self.product_root, *self.product_root.parents)
                if path.exists()
            )
            if not os.access(existing_parent, os.W_OK):
                blockers.append(
                    {
                        "code": "runtime_not_writable",
                        "next_action": "choose_writable_location",
                    }
                )
            if (
                shutil.disk_usage(existing_parent).free
                < self.manifest.python.archive_path.stat().st_size
            ):
                blockers.append(
                    {
                        "code": "insufficient_disk_space",
                        "next_action": "free_disk_space",
                    }
                )
        return blockers

    def preview_install_plan(self) -> dict[str, Any]:
        if CAPABILITY not in self._required_capabilities:
            raise RuntimeCapabilityUnavailable(
                "preview requires runtime.install-plan.v1"
            )
        with RuntimeStoreLock(self.paths.locks_root / "runtime-store.lock", timeout=0):
            return create_plan(
                self.paths.state_root,
                self._selection,
                self._source,
                self._plan_baseline(),
                self._installation_blockers(),
            )

    def ensure(self) -> RuntimeLaunch | None:
        with RuntimeStoreLock(
            self.paths.locks_root / "runtime-store.lock", timeout=self._lock_timeout
        ):
            if self._plan_record is not None:
                try:
                    existing = RuntimeOperationStore(self.paths.state_root).snapshot(
                        self._operation_id
                    )
                except RuntimeOperationNotFound:
                    existing = None
                if existing is None:
                    validate_plan(
                        self._plan_record, self._plan_baseline(), self._source
                    )
                    if self._installation_blockers():
                        raise RuntimeInstallPlanBlocked(
                            "preflight changed; resolve blockers and preview again"
                        )
            return self._ensure_locked()

    def _ensure_locked(self) -> RuntimeLaunch | None:
        # ready 额外要求已安装闭包等于期望闭包：从 base-only 扩到 full
        # （或反向）都会触发一次重装，而不是静默沿用旧范围。
        # 漂移探测要真实导入已安装组件，是启动固定成本；结果在 ready
        # 判定与 _launch 校验间复用，就绪路径每次 ensure 只探测一次。
        # 重装路径不传结果，_launch 对新运行时重新探测。
        installed = self._installed_scope_ids()
        startup_probe = (
            self._component_probe(
                self.paths.runtime_root,
                installed,
                self._covering_profile(installed),
            )
            if installed
            else None
        )
        ready = (
            self._integrity_ok()
            and installed == self._desired_scope_ids()
            and not self._drifted_component_ids(probe_results=startup_probe)
        )
        started = self._start_operation(
            "ensure",
            effective_component_ids=() if ready else self._desired_scope_ids(),
        )
        if not started:
            return self._launch(startup_probe) if ready else None
        try:
            if not ready:
                self._reporter.advance(
                    phase="wait_for_lock",
                    current=2,
                    total=7,
                    message_code="runtime.wait_for_lock",
                )
                if (
                    not self._integrity_ok()
                    or self._installed_scope_ids() != self._desired_scope_ids()
                    or self._drifted_component_ids()
                ):
                    self._install_locked(self._desired_scope_ids())
            else:
                self._reporter.advance(
                    phase="verify_runtime",
                    current=6,
                    total=7,
                    message_code="runtime.verify_runtime",
                )
            launch = self._launch(startup_probe if ready else None)
            self._reporter.succeed(
                phase="commit_runtime",
                current=7,
                total=7,
                message_code="runtime.ensure_complete",
            )
            return launch
        except RuntimeOperationCancelled:
            raise
        except Exception as exc:
            self._reporter.fail(exc)
            raise

    def _install_locked(self, target_ids: tuple[str, ...]) -> None:
        profile_name = self._covering_profile(target_ids)
        self._active_install_ids = target_ids
        self._reporter.advance(
            phase="prepare_runtime",
            current=3,
            total=7,
            message_code="runtime.prepare_runtime",
        )
        final = self.paths.runtime_root
        final.parent.mkdir(parents=True, exist_ok=True)
        partial = final.with_name("runtime.installing")
        rollback = final.with_name("runtime.rollback")
        if rollback.exists():
            if final.exists():
                rollback.replace(final.with_name(f"runtime.previous-{uuid4().hex}"))
            else:
                rollback.replace(final)
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        try:
            if not self._runner_reports_phases:
                self._reporter.advance(
                    phase="install_profile",
                    current=4,
                    total=7,
                    message_code="runtime.install_profile",
                )
            python = self._install_runner(partial, self.manifest, profile_name)
            if not self._runner_reports_phases:
                self._reporter.advance(
                    phase="verify_runtime",
                    current=6,
                    total=7,
                    message_code="runtime.verify_runtime",
                )
            try:
                python.relative_to(partial)
            except ValueError as exc:
                raise RuntimeInstallError(
                    "installer returned Python outside partial runtime"
                ) from exc
            if not python.is_file():
                raise RuntimeInstallError("installed runtime has no Python executable")
            probe_results = self._component_probe(partial, target_ids, profile_name)
            failed_components = sorted(
                component_id
                for component_id, ready in probe_results.items()
                if not ready
            )
            if failed_components:
                raise RuntimeInstallError(
                    f"installed Runtime component imports failed: {failed_components}"
                )
            (partial / ".component-integrity.json").write_text(
                json.dumps(probe_results, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            marker = {
                "schema_version": 1,
                "backend_version": self.manifest.backend_version,
                "manifest_sha256": self.manifest.sha256,
                "accelerator": self.accelerator,
                "component_ids": list(target_ids),
                "generation": uuid4().hex,
                "requested_component_ids": None
                if self._install_scope is None
                else list(self._install_scope),
                "download_source_ids": list(self._effective_download_source_ids),
            }
            (partial / ".installed.json").write_text(
                json.dumps(marker, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._reporter.advance(
                phase="commit_runtime",
                current=7,
                total=7,
                message_code="runtime.commit_runtime",
            )
            if final.exists():
                final.replace(rollback)
            try:
                partial.replace(final)
            except Exception:
                if rollback.exists() and not final.exists():
                    rollback.replace(final)
                raise
            # Keep the fixed recovery slot after commit. Archive it before the
            # next installation, so no fallible archive step follows activation.
        except RuntimeOperationCancelled:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        except Exception:
            shutil.rmtree(partial, ignore_errors=True)
            raise

    def repair(self) -> RuntimeLaunch | None:
        with RuntimeStoreLock(
            self.paths.locks_root / "runtime-store.lock", timeout=self._lock_timeout
        ):
            return self._repair_locked()

    def _repair_locked(self) -> RuntimeLaunch | None:
        if self._marker().is_file() and self._trusted_installed_scope_ids() is None:
            error = RuntimeInstallError("untrusted installed marker")
            if self._start_operation("repair"):
                self._reporter.fail(error)
            raise error
        drifted = set(self._drifted_component_ids())
        requested = set(self._component_ids) if self._component_ids else drifted
        needs_repair = bool(requested.intersection(drifted))
        globally_ready = self._integrity_ok() and not drifted
        # repair 只重建已安装闭包：不得把 base-only 运行时顺带升级成 full。
        effective = () if not needs_repair else self._installed_scope_ids()
        started = self._start_operation(
            "repair",
            effective_component_ids=effective,
        )
        if not started:
            return self._launch() if globally_ready else None
        try:
            if not needs_repair:
                launch = self._launch() if globally_ready else None
                self._reporter.succeed(
                    phase="commit_runtime",
                    current=7,
                    total=7,
                    message_code="runtime.repair_complete",
                )
                return launch
            self._reporter.advance(
                phase="wait_for_lock",
                current=2,
                total=7,
                message_code="runtime.wait_for_lock",
            )
            self._install_locked(self._installed_scope_ids())
            launch = self._launch()
            self._reporter.succeed(
                phase="commit_runtime",
                current=7,
                total=7,
                message_code="runtime.repair_complete",
            )
            return launch
        except RuntimeOperationCancelled:
            raise
        except Exception as exc:
            self._reporter.fail(exc)
            raise

    def _launch(self, probe_results: dict[str, bool] | None = None) -> RuntimeLaunch:
        if not self._integrity_ok() or self._drifted_component_ids(
            probe_results=probe_results
        ):
            raise RuntimeInstallError("runtime installation did not verify")
        environment = self._environment()
        directory_keys = {
            "VIBEOCR_PRODUCT_ROOT",
            "VIBEOCR_RUNTIME_ROOT",
            "VIBEOCR_RUNTIME_STATE_ROOT",
            "PIP_CACHE_DIR",
            "UV_CACHE_DIR",
            "HF_HOME",
            "MODELSCOPE_CACHE",
            "PADDLE_PDX_CACHE_HOME",
            "TEMP",
            "TMP",
        }
        for key in directory_keys:
            directory = environment[key]
            path = Path(directory)
            if path.is_absolute() and directory.startswith(str(self.paths.store_root)):
                path.mkdir(parents=True, exist_ok=True)
        Path(environment["MINERU_HOME"]).mkdir(
            parents=True,
            exist_ok=True,
        )
        model_root = Path(environment["VIBEOCR_RUNTIME_STATE_ROOT"]) / "models"
        model_root.mkdir(parents=True, exist_ok=True)
        return RuntimeLaunch(
            python_executable=str(_python_in(self.paths.runtime_root)),
            supervisor_module="vibeocr.backend.supervisor.main",
            working_directory=str(self.product_root),
            model_root=str(model_root),
            environment=environment,
        )

    def acquire_lease(self, *, timeout: float = 0.0) -> RuntimeStoreLock:
        lease = RuntimeStoreLock(
            self.paths.locks_root / "leases" / "runtime.lock",
            timeout=timeout,
        )
        lease.acquire()
        return lease


def _emit(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _configure_utf8_standard_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


def _request(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeInstallError("Runtime Host request must be a JSON object")
    binding_fields = {
        "protocol_version",
        "product_root",
        "component_lock",
        "runtime_manifest",
    }
    common_optional = {
        "request_kind",
        "accepted_event_streams",
    }
    request_kind = value.get("request_kind", "start")
    if request_kind == "command":
        required = binding_fields | {"command", "command_id", "target_operation_id"}
        allowed = (
            required
            | common_optional
            | {
                "plan_id",
                "required_capabilities",
                "new_operation_id",
                "expected_sequence",
                "accelerator",
                "layout_manifest",
                "product_id",
                "install_component_ids",
                "download_source_ids",
            }
        )
    elif request_kind == "install_plan":
        required = binding_fields | {"request_kind", "required_capabilities"}
        allowed = required | {
            "accelerator",
            "layout_manifest",
            "product_id",
            "install_component_ids",
            "download_source_ids",
        }
    elif request_kind == "observe":
        required = binding_fields | {"operation_id", "after_sequence"}
        allowed = (
            required
            | common_optional
            | {
                "limit",
                "accelerator",
                "layout_manifest",
                "product_id",
            }
        )
    elif request_kind == "start":
        required = binding_fields | {"operation"}
        allowed = (
            required
            | common_optional
            | {
                "accelerator",
                "layout_manifest",
                "product_id",
                "operation_id",
                "plan_id",
                "component_ids",
                "required_capabilities",
                "install_component_ids",
                "download_source_ids",
            }
        )
    else:
        raise RuntimeInstallError("Runtime Host request_kind is invalid")
    if request_kind == "install_plan" or "plan_id" in value:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import ValidationError

        schema = json.loads(
            files("vibeocr.runtime_contracts")
            .joinpath("runtime-host.schema.json")
            .read_text(encoding="utf-8")
        )
        definition = {
            "install_plan": "RuntimeInstallPlanRequest",
            "start": "RuntimeHostRequest",
            "command": "RuntimeMaintenanceCommandRequest",
        }[request_kind]
        try:
            Draft202012Validator(
                {"$ref": f"#/$defs/{definition}", "$defs": schema["$defs"]}
            ).validate(
                {
                    key: item
                    for key, item in value.items()
                    if not (request_kind == "start" and key == "request_kind")
                }
            )
        except ValidationError as exc:
            raise RuntimeInstallError(
                f"invalid install plan request: {exc.message}"
            ) from exc
    if set(value).difference(allowed):
        raise RuntimeInstallError("Runtime Host request contains unknown fields")
    if not required.issubset(value) or value["protocol_version"] != PROTOCOL_VERSION:
        raise RuntimeInstallError(
            "Runtime Host request is incompatible with Protocol v2"
        )
    if request_kind == "start" and value["operation"] not in (
        "inspect",
        "ensure",
        "repair",
    ):
        raise RuntimeInstallError("Runtime Host operation is invalid")
    for field in ("product_root", "component_lock", "runtime_manifest"):
        if not isinstance(value[field], str) or not value[field]:
            raise RuntimeInstallError(f"Runtime Host {field} is invalid")
    accepted_streams = value.get("accepted_event_streams", [])
    if (
        not isinstance(accepted_streams, list)
        or any(not isinstance(stream, str) for stream in accepted_streams)
        or len(set(accepted_streams)) != len(accepted_streams)
        or any(stream not in {"ndjson.v1", "ndjson.v2"} for stream in accepted_streams)
    ):
        raise RuntimeInstallError("Runtime Host accepted_event_streams is invalid")
    for field in ("component_ids", "required_capabilities"):
        items = value.get(field, [])
        if (
            not isinstance(items, list)
            or any(not isinstance(item, str) or not item for item in items)
            or len(set(items)) != len(items)
        ):
            raise RuntimeInstallError(f"Runtime Host {field} is invalid")
    for field in ("install_component_ids", "download_source_ids"):
        if field not in value:
            continue
        items = value[field]
        if (
            not isinstance(items, list)
            or any(not isinstance(item, str) or not item for item in items)
            or len(set(items)) != len(items)
            or (field == "download_source_ids" and not items)
        ):
            raise RuntimeInstallError(f"Runtime Host {field} is invalid")
    # 选择字段只对 ensure / retry 合法（与正式 wire schema 的条件约束一致）。
    if "install_component_ids" in value or "download_source_ids" in value:
        if request_kind == "start" and value["operation"] != "ensure":
            raise RuntimeInstallError(
                "Runtime Host selection fields require operation ensure"
            )
        if request_kind == "command" and value["command"] != "retry":
            raise RuntimeInstallError(
                "Runtime Host selection fields require command retry"
            )
    for field in (
        "operation_id",
        "command_id",
        "target_operation_id",
        "new_operation_id",
    ):
        if field in value and (not isinstance(value[field], str) or not value[field]):
            raise RuntimeInstallError(f"Runtime Host {field} is invalid")
    if request_kind == "command":
        if value["command"] not in {"cancel", "retry"}:
            raise RuntimeInstallError("Runtime Host command is invalid")
        if value["command"] == "retry" and not value.get("new_operation_id"):
            raise RuntimeInstallError("Runtime Host retry requires new_operation_id")
    if request_kind == "observe":
        if type(value["after_sequence"]) is not int or value["after_sequence"] < 0:
            raise RuntimeInstallError("Runtime Host after_sequence is invalid")
        limit = value.get("limit", 128)
        if type(limit) is not int or limit < 1 or limit > 512:
            raise RuntimeInstallError("Runtime Host limit is invalid")
    return value


def _installer_from_request(
    request: dict[str, Any],
    *,
    event_sink: EventSink | None = None,
    operation_id: str | None = None,
    source_operation_id: str | None = None,
    component_ids: tuple[str, ...] | None = None,
    required_capabilities: tuple[str, ...] | None = None,
    install_component_ids: tuple[str, ...] | None = None,
    download_source_ids: tuple[str, ...] | None = None,
    plan_id: str | None = None,
    accelerator: str | None = None,
) -> RuntimeInstaller:
    return RuntimeInstaller(
        product_root=request["product_root"],
        component_lock=request["component_lock"],
        runtime_manifest=request["runtime_manifest"],
        accelerator=accelerator
        if accelerator is not None
        else request.get("accelerator"),
        plan_id=plan_id,
        layout_manifest=request.get("layout_manifest"),
        product_id=request.get("product_id"),
        event_sink=event_sink,
        operation_id=operation_id
        if operation_id is not None
        else request.get("operation_id"),
        source_operation_id=source_operation_id,
        component_ids=(
            component_ids
            if component_ids is not None
            else tuple(request.get("component_ids", []))
        ),
        required_capabilities=(
            required_capabilities
            if required_capabilities is not None
            else tuple(request.get("required_capabilities", []))
        ),
        install_component_ids=(
            install_component_ids
            if install_component_ids is not None
            else (
                tuple(request["install_component_ids"])
                if "install_component_ids" in request
                else None
            )
        ),
        download_source_ids=(
            download_source_ids
            if download_source_ids is not None
            else (
                tuple(request["download_source_ids"])
                if "download_source_ids" in request
                else None
            )
        ),
    )


def _success_envelope(
    result: Any,
    *,
    operation: str,
) -> dict[str, Any]:
    capability_registry = json.loads(
        files("vibeocr.runtime_contracts")
        .joinpath("capabilities.json")
        .read_text(encoding="utf-8")
    )
    definitions = capability_registry["definitions"]
    capability_descriptors = [
        {
            "name": name,
            "lifecycle": definitions[name]["lifecycle"],
            "introduced_in": definitions[name]["introduced_in"],
            "deprecated_in": definitions[name]["deprecated_in"],
            "sunset_at": definitions[name]["sunset_at"],
            "replacement": definitions[name]["replacement"],
        }
        for name in result.available_capabilities
        if name in definitions
    ]
    from .services.mineru_readiness import catalog_payload

    for descriptor in capability_descriptors:
        if descriptor["name"] == "ocr.mineru-config.v1":
            # An installer cannot attest that the Supervisor executed a tier.
            descriptor["mineru_config_catalog"] = catalog_payload(observe_runtime=False)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "ok": True,
        "operation": operation,
        "state": asdict(result.state),
        "launch": asdict(result.launch) if result.launch is not None else None,
        "profile": result.profile,
        "maintenance": result.receipt["snapshot"],
        "negotiated_capabilities": list(
            result.receipt.get("negotiated_capabilities", [])
        ),
        "capability_descriptors": capability_descriptors,
    }


def _runtime_control_from_request(
    request: dict[str, Any],
    *,
    event_sink: EventSink | None,
) -> Any:
    from vibeocr.backend.runtime_control import RuntimeControl

    def installer_factory(**kwargs: Any) -> RuntimeInstaller:
        return _installer_from_request(request, event_sink=event_sink, **kwargs)

    return RuntimeControl.from_installer_factory(installer_factory)


def _command_envelope(
    request: dict[str, Any],
    control: Any,
) -> dict[str, Any]:
    receipt = control.command(
        command_id=request["command_id"],
        command=request["command"],
        target_operation_id=request["target_operation_id"],
        new_operation_id=request.get("new_operation_id"),
        expected_sequence=request.get("expected_sequence"),
        **(
            {
                "plan_id": request["plan_id"],
                "required_capabilities": tuple(request["required_capabilities"]),
            }
            if "plan_id" in request
            else {}
        ),
        install_component_ids=(
            tuple(request["install_component_ids"])
            if "install_component_ids" in request
            else None
        ),
        download_source_ids=(
            tuple(request["download_source_ids"])
            if "download_source_ids" in request
            else None
        ),
    )
    snapshot = receipt["snapshot"]
    operation = str(snapshot["operation"])
    include_launch = (
        request["command"] == "retry"
        and snapshot.get("operation_state") == "succeeded"
        and operation != "inspect"
    )
    return _success_envelope(
        control.project_receipt(receipt, include_launch=include_launch),
        operation=operation,
    )


def _failure_envelope(
    exc: Exception,
    *,
    operation: str | None,
    maintenance: dict[str, Any] | None,
) -> dict[str, Any]:
    legacy_code = "install_failed"
    canonical_code = "RUNTIME_INSTALL_FAILED"
    category = "backend_unavailable"
    retryable = False
    detail: dict[str, Any] = {}
    if isinstance(exc, (RuntimeInstallPlanStale, RuntimeInstallPlanBlocked)):
        legacy_code = "invalid_request"
        canonical_code = (
            "RUNTIME_INSTALL_PLAN_STALE"
            if isinstance(exc, RuntimeInstallPlanStale)
            else "RUNTIME_INSTALL_PLAN_BLOCKED"
        )
        category = "conflict"
    elif isinstance(exc, RuntimeLockTimeout):
        legacy_code = "lock_timeout"
        canonical_code = "RUNTIME_BUSY"
        category = "transient"
        retryable = True
    elif isinstance(exc, OSError):
        legacy_code = "io_error"
        canonical_code = "RUNTIME_IO_ERROR"
        category = "transient"
        retryable = True
    elif isinstance(exc, RuntimeOperationNotFound):
        legacy_code = "invalid_request"
        canonical_code = "RUNTIME_OPERATION_NOT_FOUND"
        category = "not_found"
    elif isinstance(exc, RuntimeCursorExpired):
        legacy_code = "invalid_request"
        canonical_code = "RUNTIME_CURSOR_EXPIRED"
        category = "not_found"
        detail["oldest_sequence"] = exc.oldest_sequence
    elif isinstance(exc, RuntimeCommandConflict):
        legacy_code = "invalid_request"
        canonical_code = "RUNTIME_COMMAND_ID_CONFLICT"
        category = "conflict"
    elif isinstance(exc, RuntimeOperationNotCancellable):
        legacy_code = "invalid_request"
        canonical_code = "RUNTIME_OPERATION_NOT_CANCELLABLE"
        category = "conflict"
    elif isinstance(exc, RuntimeOperationNotRetryable):
        legacy_code = "invalid_request"
        canonical_code = "RUNTIME_OPERATION_NOT_RETRYABLE"
        category = "conflict"
    elif isinstance(exc, RuntimeOperationConflict):
        legacy_code = "invalid_request"
        canonical_code = "RUNTIME_OPERATION_ID_CONFLICT"
        category = "conflict"
    elif isinstance(exc, RuntimeCapabilityError):
        legacy_code = "invalid_request"
        canonical_code = "RUNTIME_CAPABILITY_UNAVAILABLE"
        category = "capability"
    elif isinstance(exc, RuntimeSourceIdentityMismatch):
        legacy_code = "invalid_binding"
        canonical_code = "RUNTIME_IDENTITY_MISMATCH"
        category = "identity"
    elif isinstance(exc, RuntimeOperationCancelled):
        canonical_code = "CANCELLED"
        category = "cancelled"
    elif isinstance(exc, ManifestError):
        legacy_code = "invalid_binding"
        canonical_code = "RUNTIME_IDENTITY_MISMATCH"
        category = "identity"
    elif isinstance(exc, RuntimeSelectionError):
        legacy_code = "invalid_request"
        canonical_code = exc.code.value
        category = "validation"
    elif (
        isinstance(exc, (json.JSONDecodeError, RuntimeInstallFailure))
        and operation is None
    ):
        legacy_code = "invalid_request"
        canonical_code = "VALIDATION_ERROR"
        category = "validation"
    error: dict[str, Any] = {
        "code": legacy_code,
        "canonical_code": canonical_code,
        "category": category,
        "message": str(exc),
        "message_code": "runtime.operation_failed",
        "retryable": retryable,
        "detail": detail,
    }
    if retryable:
        error["retry_after"] = 1
    envelope: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "ok": False,
        "operation": operation,
        "error": error,
    }
    if maintenance is not None:
        envelope["maintenance"] = maintenance
    if isinstance(exc, RuntimeCursorExpired):
        envelope["maintenance"] = exc.snapshot
    return envelope


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_standard_streams()
    parser = argparse.ArgumentParser(prog="vibeocr-runtime-installer")
    parser.add_argument("--request-json")
    args = parser.parse_args(argv)
    operation: str | None = None
    control: Any | None = None
    try:
        raw_request = (
            args.request_json if args.request_json is not None else sys.stdin.read()
        )
        request = _request(json.loads(raw_request))
        request_kind = request.get("request_kind", "start")
        operation = request.get("operation")
        event_sink = (
            _emit
            if {"ndjson.v1", "ndjson.v2"}.intersection(
                request.get("accepted_event_streams", [])
            )
            else None
        )
        control = _runtime_control_from_request(
            request,
            event_sink=event_sink,
        )
        if request_kind == "install_plan":
            preview = control.preview_install_plan(
                accelerator=request.get("accelerator"),
                install_component_ids=tuple(request["install_component_ids"])
                if "install_component_ids" in request
                else None,
                download_source_ids=tuple(request["download_source_ids"])
                if "download_source_ids" in request
                else None,
                required_capabilities=tuple(request["required_capabilities"]),
            )
            _emit(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "response_kind": "install_plan",
                    "plan": preview["plan"],
                    "negotiated_capabilities": preview["negotiated_capabilities"],
                }
            )
            return 0
        if request_kind == "observe":
            update = control.observe(
                request["operation_id"],
                after_sequence=request["after_sequence"],
                limit=request.get("limit", 128),
            )
            _emit(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "ok": True,
                    "request_kind": "observe",
                    **{
                        key: value
                        for key, value in update.items()
                        if key != "schema_version"
                    },
                }
            )
            return 0
        if request_kind == "command":
            _emit(_command_envelope(request, control))
            return 0
        assert operation is not None
        result = control.execute_with_result(
            operation=operation,
            operation_id=request.get("operation_id"),
            component_ids=tuple(request.get("component_ids", [])),
            required_capabilities=tuple(request.get("required_capabilities", [])),
            install_component_ids=(
                tuple(request["install_component_ids"])
                if "install_component_ids" in request
                else None
            ),
            download_source_ids=(
                tuple(request["download_source_ids"])
                if "download_source_ids" in request
                else None
            ),
            profile_id=request.get("profile_id"),
            **({"plan_id": request["plan_id"]} if "plan_id" in request else {}),
        )
    except (
        json.JSONDecodeError,
        ManifestError,
        RuntimeInstallError,
        RuntimeLockTimeout,
        RuntimeOperationError,
        RuntimeSelectionError,
        OSError,
    ) as exc:
        _emit(
            _failure_envelope(
                exc,
                operation=operation,
                maintenance=(
                    control.maintenance_snapshot if control is not None else None
                ),
            )
        )
        return 1
    _emit(
        _success_envelope(
            result,
            operation=operation,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RuntimeState",
    "RuntimeCapabilityUnavailable",
    "RuntimeInstallError",
    "RuntimeInstaller",
    "RuntimeLaunch",
    "main",
]
