"""cutover_sequence 切换序列编排的边缘用例测试。

使用实现 CutoverBoundary 协议的 fake boundary 验证成功/失败/修复路径。
"""

from __future__ import annotations

import hashlib

import pytest
from vibeocr.backend.migration.cutover_sequence import (
    CutoverBoundary,
    CutoverError,
    CutoverPlan,
    run_cutover,
    verify_sha256,
)


class _FakeBoundary:
    """记录调用顺序的 CutoverBoundary 实现。

    通过 fail_at 设置在某步骤触发异常，并记录是否进入 repair 模式。
    """

    def __init__(self, *, fail_at: str | None = None, repair_fails: bool = False):
        self.calls: list[str] = []
        self.fail_at = fail_at
        self.repair_fails = repair_fails
        self.repair_reasons: list[str] = []

    def _record(self, name: str) -> None:
        self.calls.append(name)
        if self.fail_at == name:
            raise RuntimeError(f"injected failure at {name}")

    def verify_archive(self, archive_path: str, expected_sha256: str) -> None:
        self._record("verify archive")

    def stop_old_processes(self) -> None:
        self._record("stop old processes")

    def atomic_replace(self, archive_path: str) -> None:
        self._record("atomic replace")

    def migrate_config(self) -> None:
        self._record("migrate config")

    def check_prerequisites(self) -> None:
        self._record("check prerequisites")

    def winui_health_handshake(self, timeout_seconds: float) -> None:
        self._record("winui health handshake")

    def launch_winui(self) -> None:
        self._record("launch winui")

    def enter_repair_mode(self, reason: str) -> None:
        self.repair_reasons.append(reason)
        if self.repair_fails:
            raise RuntimeError("repair itself failed")


def _plan() -> CutoverPlan:
    return CutoverPlan(
        archive_path="/tmp/update.zip",
        expected_sha256="abc",
        health_timeout_seconds=5.0,
    )


class TestRunCutover:
    """run_cutover 序列执行。"""

    def test_full_success_returns_launched(self):
        """所有步骤成功时返回 'launched' 且不进入 repair 模式。"""
        boundary = _FakeBoundary()

        result = run_cutover(boundary, _plan())

        assert result == "launched"
        assert boundary.calls == [
            "verify archive",
            "stop old processes",
            "atomic replace",
            "migrate config",
            "check prerequisites",
            "winui health handshake",
            "launch winui",
        ]
        assert boundary.repair_reasons == []

    def test_health_timeout_passed_through(self):
        """plan.health_timeout_seconds 被传给 winui_health_handshake。"""
        captured: dict[str, float] = {}

        class _Boundary(_FakeBoundary):
            def winui_health_handshake(self, timeout_seconds: float) -> None:
                captured["timeout"] = timeout_seconds
                super().winui_health_handshake(timeout_seconds)

        run_cutover(_Boundary(), _plan())
        assert captured["timeout"] == 5.0

    @pytest.mark.parametrize(
        "fail_step, expected_prefix",
        [
            ("verify archive", "verify archive failed"),
            ("atomic replace", "atomic replace failed"),
            ("launch winui", "launch winui failed"),
        ],
    )
    def test_step_failure_raises_cutover_error(
        self, fail_step: str, expected_prefix: str
    ):
        """任意步骤失败时抛 CutoverError，消息包含步骤名。"""
        boundary = _FakeBoundary(fail_at=fail_step)

        with pytest.raises(CutoverError) as exc_info:
            run_cutover(boundary, _plan())

        assert str(exc_info.value).startswith(expected_prefix)
        assert boundary.repair_reasons, "应进入 repair 模式"
        assert fail_step in boundary.repair_reasons[0]

    def test_repair_mode_failure_does_not_swallow_cutover_error(self):
        """enter_repair_mode 自身失败时，CutoverError 仍向上抛出。"""
        boundary = _FakeBoundary(fail_at="stop old processes", repair_fails=True)

        with pytest.raises(CutoverError) as exc_info:
            run_cutover(boundary, _plan())

        assert "stop old processes failed" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None

    def test_steps_after_failure_not_executed(self):
        """失败后的步骤不被执行。"""
        boundary = _FakeBoundary(fail_at="migrate config")

        with pytest.raises(CutoverError):
            run_cutover(boundary, _plan())

        assert "check prerequisites" not in boundary.calls
        assert "launch winui" not in boundary.calls

    def test_boundary_protocol_is_runtime_checkable(self):
        """CutoverBoundary 是 runtime_checkable Protocol，fake 通过 isinstance。"""
        boundary = _FakeBoundary()
        assert isinstance(boundary, CutoverBoundary)


class TestVerifySha256:
    """verify_sha256 哈希校验辅助函数。"""

    def test_matching_hash_passes(self):
        """输入与期望哈希一致时不抛异常。"""
        data = b"payload"
        digest = hashlib.sha256(data).hexdigest()

        verify_sha256(data, digest)

    def test_mismatch_raises_cutover_error(self):
        """哈希不匹配时抛 CutoverError。"""
        with pytest.raises(CutoverError) as exc_info:
            verify_sha256(b"a", "0" * 64)

        assert "sha256 mismatch" in str(exc_info.value)

    def test_case_insensitive_comparison(self):
        """期望哈希大小写无关（uppercase expected 仍匹配）。"""
        data = b"payload"
        digest = hashlib.sha256(data).hexdigest()

        verify_sha256(data, digest.upper())


class TestCutoverPlan:
    """CutoverPlan 数据类默认值。"""

    def test_default_health_timeout(self):
        """未指定健康握手超时时使用默认 30 秒。"""
        plan = CutoverPlan(archive_path="/tmp/x", expected_sha256="abc")
        assert plan.health_timeout_seconds == 30.0
