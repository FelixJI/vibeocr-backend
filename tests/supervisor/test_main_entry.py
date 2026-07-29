"""Tests for the testable seams of ``vibeocr.backend.supervisor.main``.

The bulk of ``main`` is a genuine uvicorn entry point: it binds a loopback
socket, emits the ready envelope on stdout, then serves the FastAPI app
until interrupted. That code path is environment-dependent and is marked
``# pragma: no cover - entry point``. These tests cover the *seams* the
entry point delegates to: the self-test result writer, the missing-token
early-exit, the soak-crash env gate, and the uvicorn config builder.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from vibeocr.backend.supervisor import main as supervisor_main

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture()
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip supervisor env vars so each test starts from a known baseline."""
    for var in (
        "VIBEOCR_SUP_TOKEN",
        "VIBEOCR_SUP_ROOT",
        "VIBEOCR_SELF_TEST_SMOKE",
        "VIBEOCR_SELF_TEST_RESULT",
        "VIBEOCR_SUPERVISOR_SOAK_CRASH_AFTER_READY",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


# ---------------------------------------------------------------------------
# _write_self_test_result
# ---------------------------------------------------------------------------


def test_self_test_noop_when_smoke_env_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("VIBEOCR_SELF_TEST_SMOKE", raising=False)
    target = tmp_path / "result.json"
    monkeypatch.setenv("VIBEOCR_SELF_TEST_RESULT", str(target))
    supervisor_main._write_self_test_result()
    assert not target.exists()


def test_self_test_noop_when_result_unset(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VIBEOCR_SELF_TEST_SMOKE", "t6")
    monkeypatch.delenv("VIBEOCR_SELF_TEST_RESULT", raising=False)
    # Must not raise even though the result path is missing.
    supervisor_main._write_self_test_result()


def test_self_test_writes_payload_when_both_envs_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "result.json"
    monkeypatch.setenv("VIBEOCR_SELF_TEST_SMOKE", "t6")
    monkeypatch.setenv("VIBEOCR_SELF_TEST_RESULT", str(target))
    supervisor_main._write_self_test_result()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["supervisor_ready"] is True
    assert payload["module_file"].endswith("main.py")
    assert payload["python_executable"]


# ---------------------------------------------------------------------------
# _missing_token_exit
# ---------------------------------------------------------------------------


def test_missing_token_exit_returns_2_and_writes_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("VIBEOCR_SUP_TOKEN", raising=False)
    code = supervisor_main._missing_token_exit()
    assert code == 2
    err = capsys.readouterr().err
    assert "missing VIBEOCR_SUP_TOKEN" in err


def test_missing_token_exit_returns_0_when_token_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIBEOCR_SUP_TOKEN", "abc")
    assert supervisor_main._missing_token_exit() == 0


# ---------------------------------------------------------------------------
# _soak_crash_enabled
# ---------------------------------------------------------------------------


def test_soak_crash_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VIBEOCR_SUPERVISOR_SOAK_CRASH_AFTER_READY", raising=False)
    assert supervisor_main._soak_crash_enabled() is False


def test_soak_crash_enabled_only_when_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIBEOCR_SUPERVISOR_SOAK_CRASH_AFTER_READY", "1")
    assert supervisor_main._soak_crash_enabled() is True


# ---------------------------------------------------------------------------
# _build_uvicorn_config
# ---------------------------------------------------------------------------


def test_uvicorn_config_disables_access_log_and_binds_port() -> None:
    class _FakeUvicorn:
        class Config:
            def __init__(self, app, host, port, log_level, access_log, workers) -> None:
                self.app = app
                self.host = host
                self.port = port
                self.log_level = log_level
                self.access_log = access_log
                self.workers = workers

    sentinel_app = object()
    cfg = supervisor_main._build_uvicorn_config(_FakeUvicorn, sentinel_app, 9999)
    assert cfg.app is sentinel_app
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 9999
    assert cfg.access_log is False
    assert cfg.workers == 1
