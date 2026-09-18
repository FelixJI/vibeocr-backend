from __future__ import annotations

import ctypes
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import zipfile
from ctypes import wintypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from vibeocr.backend import runtime_installer as installer
from vibeocr.backend.runtime_installer import RuntimeInstallError, _run_install_command
from vibeocr.backend.runtime_maintenance import (
    RuntimeMaintenanceReporter,
    RuntimeOperationCancelled,
    RuntimeOperationStore,
    RuntimeProfileDescriptor,
)


def test_resolver_timeout_preserves_stage_and_elapsed() -> None:
    with pytest.raises(RuntimeInstallError) as failure:
        _run_install_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=0.3,
            env=dict(os.environ),
            reporter=None,
            heartbeat_code="runtime.resolve_packages",
        )
    detail = str(failure.value)
    assert "runtime.resolve_packages" in detail
    assert "reason=total_timeout" in detail
    assert "elapsed=" in detail
    assert "last_activity=" in detail


@pytest.mark.parametrize("active", [False, True])
def test_resolver_idle_timeout_distinguishes_real_progress(active: bool) -> None:
    script = (
        "import time\n"
        "for i in range(8):\n"
        + (
            " print(f'Collecting package-{i}', flush=True)\n"
            if active
            else " print('noise', flush=True)\n"
        )
        + " time.sleep(0.1)\n"
    )
    if active:
        _run_install_command(
            [sys.executable, "-c", script],
            timeout=4,
            idle_timeout=0.4,
            env=dict(os.environ),
            reporter=None,
            heartbeat_code="runtime.resolve_packages",
        )
    else:
        with pytest.raises(RuntimeInstallError, match="reason=idle_timeout"):
            _run_install_command(
                [sys.executable, "-c", script],
                timeout=4,
                idle_timeout=0.4,
                env=dict(os.environ),
                reporter=None,
                heartbeat_code="runtime.resolve_packages",
            )


def test_resolver_failure_redacts_output() -> None:
    script = (
        "import sys\n"
        "print('ERROR: https://user:secret@example.test/private?token=secret '"
        " + r'C:\\Users\\private-user\\cache\\wheel.whl token=secret', file=sys.stderr)\n"
        "sys.exit(1)\n"
    )
    with pytest.raises(RuntimeInstallError) as failure:
        _run_install_command(
            [sys.executable, "-c", script],
            timeout=4,
            env=dict(os.environ),
            reporter=None,
            heartbeat_code="runtime.resolve_packages",
        )
    assert "secret" not in str(failure.value)
    assert "private-user" not in str(failure.value)
    assert "reason=exit_nonzero" in str(failure.value)


def _reporter(tmp_path: Path, sink=None) -> RuntimeMaintenanceReporter:
    reporter = RuntimeMaintenanceReporter(
        state_root=tmp_path,
        profile=RuntimeProfileDescriptor("win-x64-cpu", "cpu", ()),
        event_sink=sink,
    )
    reporter.start(
        "ensure",
        total_steps=7,
        operation_id="resolve-test",
        download_source_ids=("local-test-index",),
    )
    reporter.advance(
        phase="install_profile",
        current=4,
        total=7,
        message_code="runtime.install_profile",
    )
    return reporter


def _windows_process_running(pid: int) -> bool:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        assert ctypes.get_last_error() == 87  # PID no longer exists.
        return False
    try:
        return kernel.WaitForSingleObject(handle, 2000) == 258
    finally:
        kernel.CloseHandle(handle)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object integration")
@pytest.mark.parametrize("cancel", [False, True])
def test_resolver_cleans_descendants_even_after_parent_exit(
    tmp_path: Path, cancel: bool
) -> None:
    pid_path = tmp_path / "owned-pid.txt"
    store = RuntimeOperationStore(tmp_path / "state")

    def on_event(event: dict) -> None:
        if cancel and "Collecting owned-child" in event.get("fallback_message", ""):
            store.request_cancel("resolve-test")

    reporter = _reporter(tmp_path / "state", on_event)
    script = (
        "import subprocess,sys,pathlib\n"
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])\n"
        f"pathlib.Path({str(pid_path)!r}).write_text(str(p.pid))\n"
        "print('Collecting owned-child',flush=True)\n"
        # Exit while the child retains stdout/stderr handles.
    )
    outsider = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        expected = RuntimeOperationCancelled if cancel else RuntimeInstallError
        with pytest.raises(expected):
            _run_install_command(
                [sys.executable, "-c", script],
                timeout=2,
                env=dict(os.environ),
                reporter=reporter,
                heartbeat_code="runtime.resolve_packages",
            )
        assert not _windows_process_running(int(pid_path.read_text()))
        assert outsider.poll() is None
        if cancel:
            events = reporter.observe(after_sequence=0)["events"]
            terminal = events[-1]
            assert terminal["snapshot"]["operation_state"] == "cancelled"
            detail = terminal["fallback_message"]
            assert "reason=cancelled" in detail
            assert "operation=resolve-test" in detail
            assert "sources=local-test-index" in detail
            assert "elapsed=" in detail and "last_activity=" in detail
    finally:
        outsider.kill()
        outsider.wait(timeout=5)


def test_resolver_heartbeats_do_not_prevent_idle_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(installer, "_CHILD_HEARTBEAT_INTERVAL_SECONDS", 0.05)
    events = []
    reporter = _reporter(tmp_path, events.append)
    with pytest.raises(RuntimeInstallError, match="reason=idle_timeout"):
        _run_install_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=3,
            idle_timeout=0.4,
            env=dict(os.environ),
            reporter=reporter,
            heartbeat_code="runtime.resolve_packages",
        )
    assert any(event["event_type"] == "heartbeat" for event in events)
    assert all(
        "reason=activity" not in event.get("fallback_message", "") for event in events
    )


def test_continuous_activity_still_has_total_deadline() -> None:
    script = "import time\nfor i in range(100):\n print(f'Collecting pkg-{i}',flush=True)\n time.sleep(.05)\n"
    with pytest.raises(RuntimeInstallError, match="reason=total_timeout"):
        _run_install_command(
            [sys.executable, "-c", script],
            timeout=0.7,
            idle_timeout=0.4,
            env=dict(os.environ),
            reporter=None,
            heartbeat_code="runtime.resolve_packages",
        )


def test_resolver_output_is_bounded_before_reporting() -> None:
    script = "import sys\nprint('x'*200000+'token=private-token')\nprint('ERROR: final reason',file=sys.stderr)\nsys.exit(1)"
    with pytest.raises(RuntimeInstallError) as failure:
        _run_install_command(
            [sys.executable, "-c", script],
            timeout=5,
            env=dict(os.environ),
            reporter=None,
            heartbeat_code="runtime.resolve_packages",
        )
    assert len(str(failure.value)) < 4500
    assert "private-token" not in str(failure.value)
    assert "ERROR: final reason" in str(failure.value)


@pytest.mark.parametrize("scenario", ["success", "conflict", "network_timeout"])
def test_real_pip_resolves_hash_locked_wheel_with_observable_activity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    # This fixture is the producer of the lock digest; real pip is the consumer.
    # A mismatched transfer must fail --require-hashes, just like a release lock.
    wheel_name = "vibeocr_resolver_fixture-1.0-py3-none-any.whl"
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as wheel:
        wheel.writestr("vibeocr_resolver_fixture.py", "VALUE = 1\n")
        wheel.writestr(
            "vibeocr_resolver_fixture-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: vibeocr-resolver-fixture\nVersion: 1.0\n",
        )
        wheel.writestr(
            "vibeocr_resolver_fixture-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        wheel.writestr("vibeocr_resolver_fixture-1.0.dist-info/RECORD", "")
    payload = data.getvalue()
    transfers = []
    finish = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if scenario == "network_timeout":
                finish.wait(3)
                return
            if self.path.endswith(".whl"):
                transfers.append(len(payload))
                body = payload
                kind = "application/octet-stream"
            else:
                body = f'<a href="/{wheel_name}">{wheel_name}</a>'.encode()
                kind = "text/html"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    lock = tmp_path / "requirements.lock"
    lock.write_text(
        "vibeocr-resolver-fixture=="
        + ("2.0" if scenario == "conflict" else "1.0")
        + " --hash=sha256:"
        + hashlib.sha256(payload).hexdigest()
        + "\n"
    )
    events = []
    reporter = _reporter(tmp_path / "state", events.append)
    monkeypatch.setattr(installer, "_RESOLVE_NETWORK_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(installer, "_RESOLVE_NETWORK_RETRIES", 1)
    monkeypatch.setattr(installer, "_RESOLVE_TOTAL_TIMEOUT_SECONDS", 10)

    def resolve() -> Path:
        return installer._resolve_online_report(
            Path(sys.executable),
            lock,
            f"http://127.0.0.1:{server.server_port}/simple",
            tmp_path / "cache",
            reporter,
            {
                **os.environ,
                "PIP_CONFIG_FILE": os.devnull,
                "PIP_CACHE_DIR": str(tmp_path / "pip-cache"),
            },
        )

    try:
        if scenario != "success":
            with pytest.raises(RuntimeInstallError) as failure:
                resolve()
            detail = str(failure.value)
            assert "reason=exit_nonzero" in detail
            assert (
                "ReadTimeoutError" in detail
                if scenario == "network_timeout"
                else "No matching distribution" in detail
            )
            assert not (tmp_path / "cache/resolve/requirements-report.json").exists()
            return
        report = resolve()
        parsed = json.loads(report.read_text())
        assert parsed["install"][0]["metadata"]["name"] == "vibeocr-resolver-fixture"
        assert transfers == [len(payload)]  # This dry-run actually fetched the wheel.
        details = [event.get("fallback_message", "") for event in events]
        assert any("reason=started" in detail for detail in details)
        assert any(
            "Collecting vibeocr-resolver-fixture" in detail for detail in details
        )
        assert any("reason=succeeded" in detail for detail in details)
        assert all("http://127.0.0.1" not in detail for detail in details)
    finally:
        finish.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.skipif(os.name != "nt", reason="Windows containment boundary")
def test_uncontained_command_is_never_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "unexpected.txt"
    monkeypatch.setattr(
        installer.JobObjectGuard, "assign_from_popen", lambda self, process: False
    )
    with pytest.raises(RuntimeInstallError, match="process_containment_failed"):
        _run_install_command(
            [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).touch()",
            ],
            timeout=2,
            env=dict(os.environ),
            reporter=None,
            heartbeat_code="runtime.resolve_packages",
        )
    assert not marker.exists()


def test_finished_resolver_diagnostic_does_not_leak_into_download_cancel(
    tmp_path: Path,
) -> None:
    reporter = _reporter(tmp_path / "state")
    _run_install_command(
        [sys.executable, "-c", "print('Collecting finished', flush=True)"],
        timeout=4,
        env=dict(os.environ),
        reporter=reporter,
        heartbeat_code="runtime.resolve_packages",
    )
    RuntimeOperationStore(tmp_path / "state").request_cancel("resolve-test")
    with pytest.raises(RuntimeOperationCancelled):
        reporter.advance_measured(
            phase="download_artifacts",
            message_code="runtime.download_artifacts",
            current=1,
            total=10,
            unit="bytes",
        )
    terminal = reporter.observe(after_sequence=0)["events"][-1]
    assert terminal["snapshot"]["operation_state"] == "cancelled"
    assert "runtime.resolve_packages" not in terminal.get("fallback_message", "")
    assert "Collecting finished" not in terminal.get("fallback_message", "")
