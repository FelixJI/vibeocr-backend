"""Paddle worker failures cannot replay inference or return stale responses."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from vibeocr.backend.supervisor.inference.paddle_process_adapter import (
    PaddleProcessAdapter,
)


def test_worker_timeout_terminates_process_and_discards_late_reply():
    adapter = PaddleProcessAdapter(Path(sys.executable))
    process = MagicMock()
    process.poll.return_value = None
    adapter._process = process
    with pytest.raises(TimeoutError, match="deadline"):
        adapter._exchange({"operation": "status"}, timeout=0.001)
    process.terminate.assert_called_once()
    assert adapter._process is None


def test_worker_exit_is_failure_not_replayed():
    adapter = PaddleProcessAdapter(Path(sys.executable))
    process = MagicMock()
    process.poll.return_value = 1
    adapter._process = process
    adapter._responses.put(None)
    with pytest.raises(RuntimeError, match="exited"):
        adapter._exchange({"operation": "recognize"})
    process.stdin.write.assert_called_once()
    assert adapter._process is None


def test_worker_model_error_is_not_reported_as_payload():
    adapter = PaddleProcessAdapter(Path(sys.executable))
    process = MagicMock()
    adapter._process = process
    adapter._responses.put('{"error":"model unavailable","error_type":"RuntimeError"}')
    with pytest.raises(RuntimeError, match="model unavailable"):
        adapter._exchange({"operation": "recognize"})
    process.stdin.write.assert_called_once()


def test_shutdown_escalates_only_when_worker_does_not_exit():
    adapter = PaddleProcessAdapter(Path(sys.executable))
    process = MagicMock()
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("worker", 5), 0]
    adapter._process = process
    adapter.close()
    process.kill.assert_called_once()
    assert adapter._process is None


def test_routing_batch_argument_reaches_isolated_worker(monkeypatch):
    adapter = PaddleProcessAdapter(Path(sys.executable))
    request = MagicMock(return_value=[])
    monkeypatch.setattr(adapter, "_request", request)
    assert adapter.recognize_many([], compute_batch=4) == []
    assert request.call_args.args == ("recognize",)


def test_broken_worker_pipe_discards_process():
    adapter = PaddleProcessAdapter(Path(sys.executable))
    process = MagicMock()
    process.poll.return_value = 1
    process.stdin.write.side_effect = BrokenPipeError()
    adapter._process = process
    with pytest.raises(BrokenPipeError):
        adapter._exchange({"operation": "recognize"})
    assert adapter._process is None
