"""Tests for auth/loopback checks, bootstrap token handling and retention."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vibeocr.backend.supervisor.auth import (
    check_bearer_token,
    check_loopback,
    is_bootstrap_path,
)
from vibeocr.backend.supervisor.bootstrap import (
    BootstrapHandle,
    ReadyEnvelope,
    bind_loopback_socket,
    emit_ready,
    generate_session_token,
    new_instance_id,
    token_from_environment,
)
from vibeocr.backend.supervisor.jobs.registry import JobRegistry
from vibeocr.backend.supervisor.jobs.retention import RetentionPolicy
from vibeocr.runtime_contracts import ErrorCode, JobKind, JobPriority, JobState
from vibeocr.runtime_contracts.generated.capabilities import OCR_RECOGNITION_V2

INSTANCE = "sup-test"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_check_bearer_token_accepts_valid() -> None:
    decision = check_bearer_token("Bearer abc123", "abc123", instance_id=INSTANCE)
    assert decision.ok
    assert decision.error is None


def test_check_bearer_token_rejects_missing_header() -> None:
    decision = check_bearer_token(None, "abc123", instance_id=INSTANCE)
    assert not decision.ok
    assert decision.error is not None
    assert decision.error.code is ErrorCode.UNAUTHORIZED


def test_check_bearer_token_rejects_wrong_scheme() -> None:
    decision = check_bearer_token("Basic abc123", "abc123", instance_id=INSTANCE)
    assert not decision.ok
    assert decision.error is not None
    assert decision.error.code is ErrorCode.UNAUTHORIZED


def test_check_bearer_token_rejects_wrong_token() -> None:
    decision = check_bearer_token("Bearer wrong", "abc123", instance_id=INSTANCE)
    assert not decision.ok
    assert decision.error is not None
    assert decision.error.code is ErrorCode.UNAUTHORIZED


def test_check_loopback_accepts_127() -> None:
    assert check_loopback("127.0.0.1", instance_id=INSTANCE).ok
    assert check_loopback("::1", instance_id=INSTANCE).ok


def test_check_loopback_rejects_external() -> None:
    decision = check_loopback("10.0.0.5", instance_id=INSTANCE)
    assert not decision.ok
    assert decision.error is not None
    assert decision.error.code is ErrorCode.FORBIDDEN_LOOPBACK


def test_check_loopback_rejects_unknown() -> None:
    decision = check_loopback(None, instance_id=INSTANCE)
    assert not decision.ok
    assert decision.error is not None
    assert decision.error.code is ErrorCode.FORBIDDEN_LOOPBACK


def test_is_bootstrap_path_only_health() -> None:
    assert is_bootstrap_path("/v2/health")
    assert not is_bootstrap_path("/v2/jobs/recognition")


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_generate_session_token_is_256_bits_and_unique() -> None:
    a = generate_session_token()
    b = generate_session_token()
    assert a != b
    # token_urlsafe(32) -> ~43 chars base64 of 32 bytes
    assert len(a) >= 32


def test_new_instance_id_has_prefix() -> None:
    assert new_instance_id().startswith("sup-")


def test_bind_loopback_socket_assigns_ephemeral_port() -> None:
    sock = bind_loopback_socket()
    try:
        host, port = sock.getsockname()
        assert host == "127.0.0.1"
        assert port != 0
    finally:
        sock.close()


def test_bootstrap_handle_round_trip() -> None:
    handle = BootstrapHandle()
    handle.set_token("xyz")
    assert handle.token == "xyz"


def test_bootstrap_handle_raises_if_unset() -> None:
    handle = BootstrapHandle()
    with pytest.raises(RuntimeError):
        _ = handle.token


def test_token_from_environment_reads_var() -> None:
    env = {"VIBEOCR_SUP_TOKEN": "from-env"}
    assert token_from_environment(env) == "from-env"
    assert token_from_environment({}) is None


def test_emit_ready_writes_single_json_line(tmp_path: Path) -> None:
    import io

    buf = io.StringIO()
    env = ReadyEnvelope(
        ready=True,
        pid=1234,
        port=54321,
        instance_id="sup-abc",
        protocol_version=2,
        schema_version=2,
        capabilities=[OCR_RECOGNITION_V2],
    )
    emit_ready(env, stream=buf)
    lines = buf.getvalue().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["port"] == 54321
    assert parsed["instance_id"] == "sup-abc"
    assert "token" not in parsed  # token must never appear on stdout
    assert "VIBEOCR_SUP_TOKEN" not in lines[0]


def test_supervisor_uvicorn_config_disables_access_log() -> None:
    from types import SimpleNamespace

    import vibeocr.backend.supervisor.main as main_module

    captured: dict[str, object] = {}

    def config_factory(app, **kwargs):
        captured["app"] = app
        captured.update(kwargs)
        return object()

    fake_uvicorn = SimpleNamespace(Config=config_factory)
    app = object()

    config = main_module._build_uvicorn_config(fake_uvicorn, app, 54321)

    assert config is not None
    assert captured["app"] is app
    assert captured["access_log"] is False


def test_t6_self_test_result_records_actual_supervisor_source(
    monkeypatch, tmp_path: Path
) -> None:
    import sys

    import vibeocr.backend.supervisor.main as main_module

    result = tmp_path / "supervisor-result.json"
    monkeypatch.setenv("VIBEOCR_SELF_TEST_SMOKE", "t6")
    monkeypatch.setenv("VIBEOCR_SELF_TEST_RESULT", str(result))

    main_module._write_self_test_result()

    evidence = json.loads(result.read_text(encoding="utf-8"))
    assert evidence["supervisor_ready"] is True
    assert (
        Path(evidence["module_file"]).resolve() == Path(main_module.__file__).resolve()
    )
    assert (
        Path(evidence["python_executable"]).resolve() == Path(sys.executable).resolve()
    )


def test_production_supervisor_does_not_write_self_test_result(
    monkeypatch, tmp_path: Path
) -> None:
    import vibeocr.backend.supervisor.main as main_module

    result = tmp_path / "supervisor-result.json"
    monkeypatch.delenv("VIBEOCR_SELF_TEST_SMOKE", raising=False)
    monkeypatch.setenv("VIBEOCR_SELF_TEST_RESULT", str(result))

    main_module._write_self_test_result()

    assert not result.exists()


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


def _registry_with_terminal_job() -> tuple[JobRegistry, str, float]:
    reg = JobRegistry(instance_id=INSTANCE)
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=[],
        progress_total=0,
    )
    rec.transition(JobState.QUEUED)
    rec.transition(JobState.FAILED)
    return reg, rec.job_id, 0.0


def test_retention_purges_after_ttl() -> None:
    reg, job_id, _ = _registry_with_terminal_job()
    clock_values = [0.0, 100.0]
    policy = RetentionPolicy(reg, retention_seconds=10.0, clock=lambda: clock_values[0])
    # mark terminal at t=0
    policy.mark_terminal(reg.get(job_id))
    # advance clock and purge
    clock_values[0] = 100.0
    purged = policy.purge_expired()
    assert purged == [job_id]
    # second purge is a no-op
    assert policy.purge_expired() == []


def test_retention_keeps_jobs_within_ttl() -> None:
    reg, job_id, _ = _registry_with_terminal_job()
    t = [0.0]
    policy = RetentionPolicy(reg, retention_seconds=60.0, clock=lambda: t[0])
    policy.mark_terminal(reg.get(job_id))
    t[0] = 5.0
    assert policy.purge_expired() == []
    # job still retrievable
    assert reg.get(job_id).state is JobState.FAILED
