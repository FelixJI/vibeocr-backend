"""Tests for scripts/build_runtime_pack.py(离线 wheel 闭包打包)。

pip download / pip wheel 通过 monkeypatch 注入 fake 子进程,验证:
两阶段命令构造、hash 复核 fail closed、sdist→wheel 离线转化、
wheel 覆盖校验、确定性 zip 输出与 lock 校验前置。
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Any

import pytest

from scripts.build_runtime_pack import build_runtime_pack

_WHEEL_BYTES = b"deterministic-wheel-bytes"
_LOCK_HASH = hashlib.sha256(_WHEEL_BYTES).hexdigest()


def _base_lock(tmp_path: Path) -> Path:
    lock = tmp_path / "requirements-win-x64-base.lock"
    lock.write_text(
        "--index-url https://pypi.org/simple\n"
        f"rapidocr==3.9.2 \\\n    --hash=sha256:{_LOCK_HASH}\n"
        f"onnxruntime==1.28.0 \\\n    --hash=sha256:{_LOCK_HASH}\n"
        f"winrt-runtime==3.2.1 \\\n    --hash=sha256:{_LOCK_HASH}\n"
        f"opencv-python==5.0.0.93 \\\n    --hash=sha256:{_LOCK_HASH}\n",
        encoding="utf-8",
    )
    return lock


class _FakeCompleted:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.stdout = ""
        self.stderr = ""


def _install_fake_pip(
    monkeypatch: pytest.MonkeyPatch,
    downloads: dict[str, bytes] | None = None,
    *,
    returncode: int = 0,
) -> list[list[str]]:
    """Fake pip download + pip wheel.

    download 阶段把给定文件写进 -d 目录;wheel 阶段对每个 ``name==ver``
    参数在 -w 目录产出 ``name-ver-py3-none-any.whl``(模拟直接复用 wheel
    或从 sdist 构建)。
    """
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> _FakeCompleted:
        calls.append(list(command))
        if "download" in command:
            target = Path(command[command.index("-d") + 1])
            target.mkdir(parents=True, exist_ok=True)
            for name, payload in (downloads or {}).items():
                (target / name).write_bytes(payload)
            return _FakeCompleted(returncode)
        if "wheel" in command:
            target = Path(command[command.index("-w") + 1])
            target.mkdir(parents=True, exist_ok=True)
            for argument in command[command.index("-w") + 2 :]:
                if "==" not in argument:
                    continue
                name, version = argument.split("==", 1)
                wheel_name = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
                (target / wheel_name).write_bytes(_WHEEL_BYTES)
            return _FakeCompleted(returncode)
        raise AssertionError(f"unexpected command: {command[:4]}")

    monkeypatch.setattr("scripts.build_runtime_pack.subprocess.run", fake_run)
    return calls


def test_pack_build_two_phase_download_wheel_and_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # antlr 只有 sdist(经 omegaconf 进入闭包):download 阶段出现 sdist。
    calls = _install_fake_pip(
        monkeypatch,
        {
            "rapidocr-3.9.2-py3-none-any.whl": _WHEEL_BYTES,
            "antlr4-python3-runtime-4.9.3.tar.gz": _WHEEL_BYTES,
        },
    )
    lock = tmp_path / "requirements-win-x64-base.lock"
    lock.write_text(
        f"rapidocr==3.9.2 \\\n    --hash=sha256:{_LOCK_HASH}\n"
        f"antlr4-python3-runtime==4.9.3 \\\n    --hash=sha256:{_LOCK_HASH}\n",
        encoding="utf-8",
    )
    # base 规则要求的其余包不需要出现:这里直接用 cpu profile 校验路径。
    output = tmp_path / "out" / "pack.zip"

    first = build_runtime_pack(
        lock=lock,
        profile="win-x64-cpu",
        work_dir=tmp_path / "work",
        output=output,
    )
    assert first == output
    download_command = calls[0]
    assert "--require-hashes" in download_command
    assert str(lock) in download_command
    wheel_command = calls[1]
    # sdist→wheel 必须离线、免隔离构建(依赖运行环境 setuptools)。
    assert "--no-index" in wheel_command
    assert "--no-build-isolation" in wheel_command
    assert "--require-hashes" not in wheel_command
    assert "antlr4-python3-runtime==4.9.3" in wheel_command
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert names == [
            "pack-requirements.txt",
            "antlr4_python3_runtime-4.9.3-py3-none-any.whl",
            "rapidocr-3.9.2-py3-none-any.whl",
        ]
        assert archive.read("pack-requirements.txt").decode("utf-8") == (
            "antlr4-python3-runtime==4.9.3" + chr(10) + "rapidocr==3.9.2" + chr(10)
        )
        assert archive.read(names[2]) == _WHEEL_BYTES
        # 确定性:固定 zip 时间戳。
        info = archive.getinfo(names[1])
        assert tuple(info.date_time) == (1980, 1, 1, 0, 0, 0)

    # 第二次构建字节一致(确定性)。
    _install_fake_pip(
        monkeypatch,
        {
            "rapidocr-3.9.2-py3-none-any.whl": _WHEEL_BYTES,
            "antlr4-python3-runtime-4.9.3.tar.gz": _WHEEL_BYTES,
        },
    )
    second = tmp_path / "out2" / "pack.zip"
    build_runtime_pack(
        lock=lock,
        profile="win-x64-cpu",
        work_dir=tmp_path / "work2",
        output=second,
    )
    assert (
        hashlib.sha256(output.read_bytes()).hexdigest()
        == hashlib.sha256(second.read_bytes()).hexdigest()
    )


def test_pack_build_fails_when_wheel_coverage_is_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # download 成功但 wheel 阶段漏掉一个包(比如 sdist 构建失败被吞):
    # 覆盖校验必须 fail closed。
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> _FakeCompleted:
        calls.append(list(command))
        if "download" in command:
            target = Path(command[command.index("-d") + 1])
            target.mkdir(parents=True, exist_ok=True)
            (target / "rapidocr-3.9.2-py3-none-any.whl").write_bytes(_WHEEL_BYTES)
            return _FakeCompleted(0)
        if "wheel" in command:
            target = Path(command[command.index("-w") + 1])
            target.mkdir(parents=True, exist_ok=True)
            # 只产出 rapidocr,漏掉 winrt-runtime。
            (target / "rapidocr-3.9.2-py3-none-any.whl").write_bytes(_WHEEL_BYTES)
            return _FakeCompleted(0)
        raise AssertionError(f"unexpected command: {command[:4]}")

    monkeypatch.setattr("scripts.build_runtime_pack.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match=r"missing wheels for: .*'winrt-runtime'"):
        build_runtime_pack(
            lock=_base_lock(tmp_path),
            profile="win-x64-base",
            work_dir=tmp_path / "work",
            output=tmp_path / "pack.zip",
        )


def test_pack_build_rejects_artifact_not_bound_by_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unknown = b"unbound-wheel"
    assert hashlib.sha256(unknown).hexdigest() != _LOCK_HASH
    _install_fake_pip(monkeypatch, {"mystery-1.0-py3-none-any.whl": unknown})
    with pytest.raises(RuntimeError, match="not bound by lock"):
        build_runtime_pack(
            lock=_base_lock(tmp_path),
            profile="win-x64-base",
            work_dir=tmp_path / "work",
            output=tmp_path / "pack.zip",
        )


def test_pack_build_fails_closed_when_download_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_pip(monkeypatch, {}, returncode=1)
    with pytest.raises(RuntimeError, match="command failed"):
        build_runtime_pack(
            lock=_base_lock(tmp_path),
            profile="win-x64-base",
            work_dir=tmp_path / "work",
            output=tmp_path / "pack.zip",
        )


def test_pack_build_validates_profile_lock_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vibeocr.backend.runtime_manifest import ManifestError

    bad_lock = tmp_path / "requirements-win-x64-base.lock"
    bad_lock.write_text(
        f"paddlepaddle==3.3.1 \\\n    --hash=sha256:{_LOCK_HASH}\n",
        encoding="utf-8",
    )
    calls = _install_fake_pip(monkeypatch, {})
    with pytest.raises(ManifestError, match="base lock is missing"):
        build_runtime_pack(
            lock=bad_lock,
            profile="win-x64-base",
            work_dir=tmp_path / "work",
            output=tmp_path / "pack.zip",
        )
    # lock 校验失败时不发起下载。
    assert calls == []


def test_pack_build_requires_zip_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_pip(monkeypatch, {})
    with pytest.raises(ValueError, match="must be a .zip"):
        build_runtime_pack(
            lock=_base_lock(tmp_path),
            profile="win-x64-base",
            work_dir=tmp_path / "work",
            output=tmp_path / "pack.tar",
        )


def test_pack_cli_reports_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.build_runtime_pack import main

    _install_fake_pip(monkeypatch, {}, returncode=1)
    code = main(
        [
            "--lock",
            str(_base_lock(tmp_path)),
            "--profile",
            "win-x64-base",
            "--work-dir",
            str(tmp_path / "work"),
            "--output",
            str(tmp_path / "pack.zip"),
        ]
    )
    assert code == 1
    assert "runtime pack build failed" in capsys.readouterr().err
