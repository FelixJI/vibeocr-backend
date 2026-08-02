"""dependency_bootstrap CLI 入口的边缘用例测试。

通过 monkeypatch 隔离 GPU 检测与依赖安装内核，验证 profile 解析与退出码语义。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from vibeocr.backend import dependency_bootstrap


class TestResolveProfile:
    """_resolve_profile 把 'auto' 解析为具体 profile。"""

    def test_explicit_cpu_passthrough(self):
        """显式 cpu 直接返回。"""
        assert dependency_bootstrap._resolve_profile("cpu") == "cpu"

    def test_explicit_gpu_passthrough(self):
        """显式 gpu-cu126 直接返回。"""
        assert dependency_bootstrap._resolve_profile("gpu-cu126") == "gpu-cu126"

    def test_auto_with_gpu_detected(self, monkeypatch: pytest.MonkeyPatch):
        """auto 且检测到 GPU 时解析为 gpu-cu126。"""
        monkeypatch.setattr(dependency_bootstrap, "detect_gpu", lambda: (True, "cu126"))

        assert dependency_bootstrap._resolve_profile("auto") == "gpu-cu126"

    def test_auto_without_gpu_resolves_to_cpu(self, monkeypatch: pytest.MonkeyPatch):
        """auto 且无 GPU 时解析为 cpu。"""
        monkeypatch.setattr(dependency_bootstrap, "detect_gpu", lambda: (False, None))

        assert dependency_bootstrap._resolve_profile("auto") == "cpu"


class TestMainArgv:
    """main() 的 argparse 与退出码语义。"""

    def test_success_returns_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        """install 返回成功时 main 退出码为 0，消息进 stdout。"""
        monkeypatch.setattr(
            dependency_bootstrap,
            "install_backend_dependencies",
            lambda *a, **k: (True, "installed ok"),
        )

        code = dependency_bootstrap.main(["--profile", "cpu", "--network", "domestic"])

        assert code == 0
        captured = capsys.readouterr()
        assert "installed ok" in captured.out

    def test_failure_returns_one_and_writes_stderr(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        """install 返回失败时 main 退出码为 1，消息进 stderr。"""
        monkeypatch.setattr(
            dependency_bootstrap,
            "install_backend_dependencies",
            lambda *a, **k: (False, "boom"),
        )

        code = dependency_bootstrap.main(
            ["--profile", "cpu", "--network", "international"]
        )

        assert code == 1
        captured = capsys.readouterr()
        assert "boom" in captured.err

    def test_progress_callback_reports_stages(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        """main 注入的 report 回调把阶段消息打印到 stdout。"""
        captured_stages: list[str] = []

        def fake_install(python_exe, *, profile, network_type, progress_callback):
            progress_callback("resolve", "starting")
            progress_callback("install", "done")
            captured_stages.append(profile)
            return True, "all good"

        monkeypatch.setattr(
            dependency_bootstrap, "install_backend_dependencies", fake_install
        )

        code = dependency_bootstrap.main(["--profile", "gpu-cu126"])

        assert code == 0
        out = capsys.readouterr().out
        assert "[resolve] starting" in out
        assert "[install] done" in out
        assert captured_stages == ["gpu-cu126"]

    def test_explicit_python_path_forwarded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """--python 指定的解释器路径被透传给安装内核。"""
        seen_python: dict[str, Path] = {}
        target = tmp_path / "custompython"

        def fake_install(python_exe, *, profile, network_type, progress_callback):
            seen_python["python"] = python_exe
            return True, "ok"

        monkeypatch.setattr(
            dependency_bootstrap, "install_backend_dependencies", fake_install
        )

        dependency_bootstrap.main(["--profile", "cpu", "--python", str(target)])

        assert seen_python["python"] == target

    def test_auto_profile_resolves_before_install(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        """auto profile 经 _resolve_profile 解析后传入 install。"""
        monkeypatch.setattr(dependency_bootstrap, "detect_gpu", lambda: (True, "cu126"))
        seen_profile: list[str] = []

        def fake_install(python_exe, *, profile, network_type, progress_callback):
            seen_profile.append(profile)
            return True, "ok"

        monkeypatch.setattr(
            dependency_bootstrap, "install_backend_dependencies", fake_install
        )

        code = dependency_bootstrap.main(["--profile", "auto"])

        assert code == 0
        assert seen_profile == ["gpu-cu126"]

    def test_invalid_profile_choice_rejected(self, monkeypatch: pytest.MonkeyPatch):
        """argparse 拒绝非法 profile 选项（SystemExit）。"""
        monkeypatch.setattr(
            dependency_bootstrap,
            "install_backend_dependencies",
            lambda *a, **k: (True, ""),
        )

        with pytest.raises(SystemExit):
            dependency_bootstrap.main(["--profile", "bogus"])
