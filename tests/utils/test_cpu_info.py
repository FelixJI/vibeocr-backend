"""cpu_info 线程数、版本判定与 oneDNN 安全决策的边缘用例测试。"""

from __future__ import annotations

import pytest
from vibeocr.backend.utils import cpu_info


class TestGetCpuThreadCount:
    """get_cpu_thread_count 边缘用例。"""

    def test_env_override_used(self, monkeypatch: pytest.MonkeyPatch):
        """合法 VIBEOCR_CPU_THREADS 优先于 os.cpu_count。"""
        monkeypatch.setenv("VIBEOCR_CPU_THREADS", "7")
        assert cpu_info.get_cpu_thread_count() == 7

    def test_env_override_with_whitespace(self, monkeypatch: pytest.MonkeyPatch):
        """带空白的覆盖值被 strip 后采用。"""
        monkeypatch.setenv("VIBEOCR_CPU_THREADS", "  3  ")
        assert cpu_info.get_cpu_thread_count() == 3

    def test_env_override_zero_ignored(self, monkeypatch: pytest.MonkeyPatch):
        """非正值忽略，回退 os.cpu_count。"""
        monkeypatch.setenv("VIBEOCR_CPU_THREADS", "0")
        assert cpu_info.get_cpu_thread_count() == cpu_info.get_cpu_thread_count()

    def test_env_override_negative_ignored(self, monkeypatch: pytest.MonkeyPatch):
        """负值忽略，回退。"""
        monkeypatch.setenv("VIBEOCR_CPU_THREADS", "-4")
        # 至少为正
        assert cpu_info.get_cpu_thread_count() >= 1

    def test_env_override_non_int_ignored(self, monkeypatch: pytest.MonkeyPatch):
        """非整数值忽略并回退。"""
        monkeypatch.setenv("VIBEOCR_CPU_THREADS", "abc")
        assert cpu_info.get_cpu_thread_count() >= 1

    def test_cpu_count_capped(self, monkeypatch: pytest.MonkeyPatch):
        """os.cpu_count 超过上限时裁剪到 CPU_THREADS_CAP。"""
        monkeypatch.delenv("VIBEOCR_CPU_THREADS", raising=False)
        monkeypatch.setattr(cpu_info.os, "cpu_count", lambda: 1024)
        assert cpu_info.get_cpu_thread_count() == cpu_info.CPU_THREADS_CAP

    def test_result_within_valid_range(self, monkeypatch: pytest.MonkeyPatch):
        """无覆盖时结果落在 [1, CAP]。"""
        monkeypatch.delenv("VIBEOCR_CPU_THREADS", raising=False)
        n = cpu_info.get_cpu_thread_count()
        assert 1 <= n <= cpu_info.CPU_THREADS_CAP

    def test_probe_failure_falls_back(self, monkeypatch: pytest.MonkeyPatch):
        """系统 CPU 数探针失败时使用保守回退值。"""
        monkeypatch.delenv("VIBEOCR_CPU_THREADS", raising=False)

        def raise_probe_error() -> int:
            raise OSError("probe failed")

        monkeypatch.setattr(cpu_info.os, "cpu_count", raise_probe_error)
        assert cpu_info.get_cpu_thread_count() == cpu_info.FALLBACK_CPU_THREADS


class TestVerTuple:
    """_ver_tuple 版本解析。"""

    def test_plain_version(self):
        """标准三段版本解析。"""
        assert cpu_info._ver_tuple("3.3.1") == (3, 3, 1)

    def test_local_suffix_stripped(self):
        """+cu126 本地后缀被剥离。"""
        assert cpu_info._ver_tuple("3.3.1+cu126") == (3, 3, 1)

    def test_tilde_suffix_stripped(self):
        """~ 分隔的本地后缀被剥离。"""
        assert cpu_info._ver_tuple("3.3.1~rc1") == (3, 3, 1)

    def test_two_segments(self):
        """两段版本补齐。"""
        assert cpu_info._ver_tuple("3.3") == (3, 3)


class TestVersionInRange:
    """_version_in_range 边界判定。"""

    def test_inside_range(self):
        """区间内命中。"""
        assert cpu_info._version_in_range("3.3.1", "3.3.0", "3.3.99") is True

    def test_lower_boundary_inclusive(self):
        """下界闭区间。"""
        assert cpu_info._version_in_range("3.3.0", "3.3.0", "3.3.99") is True

    def test_upper_boundary_inclusive(self):
        """上界闭区间。"""
        assert cpu_info._version_in_range("3.3.99", "3.3.0", "3.3.99") is True

    def test_below_range(self):
        """低于区间不命中。"""
        assert cpu_info._version_in_range("3.2.5", "3.3.0", "3.3.99") is False

    def test_above_range(self):
        """高于区间不命中。"""
        assert cpu_info._version_in_range("3.4.0", "3.3.0", "3.3.99") is False

    def test_invalid_version_returns_false(self):
        """非法版本号返回 False 而非抛异常。"""
        assert cpu_info._version_in_range("not-a-version", "3.3.0", "3.3.99") is False


class TestGetPaddleVersion:
    """_get_paddle_version 延迟 import 容错。"""

    def test_paddle_not_installed_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        """paddle 未安装/import 失败时返回 None。"""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "paddle":
                raise ImportError("no paddle")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert cpu_info._get_paddle_version() is None

    def test_installed_version_returned(self, monkeypatch: pytest.MonkeyPatch):
        """已安装 paddle 时返回其 __version__。"""

        class FakePaddle:
            __version__ = "3.3.1"

        # 直接构造已 import 的模块对象注入到 sys.modules。
        import sys

        monkeypatch.setitem(sys.modules, "paddle", FakePaddle)
        assert cpu_info._get_paddle_version() == "3.3.1"


class TestCanSafelyEnableOnednn:
    """can_safely_enable_onednn 决策矩阵。"""

    def test_force_enable(self, monkeypatch: pytest.MonkeyPatch):
        """VIBEOCR_FORCE_ONEDNN=1 强制启用。"""
        monkeypatch.setenv("VIBEOCR_FORCE_ONEDNN", "1")
        safe, reason = cpu_info.can_safely_enable_onednn()
        assert safe is True
        assert "强制启用" in reason

    def test_force_disable(self, monkeypatch: pytest.MonkeyPatch):
        """VIBEOCR_FORCE_ONEDNN=0 强制禁用。"""
        monkeypatch.setenv("VIBEOCR_FORCE_ONEDNN", "0")
        safe, reason = cpu_info.can_safely_enable_onednn()
        assert safe is False
        assert "强制禁用" in reason

    def test_no_avx2_rejected(self, monkeypatch: pytest.MonkeyPatch):
        """无 AVX2 指令集门槛拒绝。"""
        monkeypatch.delenv("VIBEOCR_FORCE_ONEDNN", raising=False)
        monkeypatch.setattr(
            cpu_info,
            "detect_cpu_features",
            lambda: {
                "avx": True,
                "avx2": False,
                "avx512": False,
                "fma": True,
                "amx": False,
            },
        )
        safe, reason = cpu_info.can_safely_enable_onednn()
        assert safe is False
        assert "AVX2" in reason

    def test_unknown_paddle_version_fail_closed(self, monkeypatch: pytest.MonkeyPatch):
        """paddle 版本未知时 fail-closed。"""
        monkeypatch.delenv("VIBEOCR_FORCE_ONEDNN", raising=False)
        monkeypatch.setattr(
            cpu_info,
            "detect_cpu_features",
            lambda: {
                "avx": True,
                "avx2": True,
                "avx512": False,
                "fma": True,
                "amx": False,
            },
        )
        monkeypatch.setattr(cpu_info, "_get_paddle_version", lambda: None)
        safe, reason = cpu_info.can_safely_enable_onednn()
        assert safe is False
        assert "paddle" in reason or "无法" in reason

    def test_blacklisted_paddle_rejected(self, monkeypatch: pytest.MonkeyPatch):
        """黑名单 3.3.x（PIR 与 oneDNN 不兼容）被拒绝。"""
        monkeypatch.delenv("VIBEOCR_FORCE_ONEDNN", raising=False)
        monkeypatch.setattr(
            cpu_info,
            "detect_cpu_features",
            lambda: {
                "avx": True,
                "avx2": True,
                "avx512": False,
                "fma": True,
                "amx": False,
            },
        )
        monkeypatch.setattr(cpu_info, "_get_paddle_version", lambda: "3.3.5")
        safe, reason = cpu_info.can_safely_enable_onednn()
        assert safe is False
        assert "不兼容" in reason

    def test_unvalidated_version_fail_closed(self, monkeypatch: pytest.MonkeyPatch):
        """已安装但未通过项目验证的版本 fail-closed。"""
        monkeypatch.delenv("VIBEOCR_FORCE_ONEDNN", raising=False)
        monkeypatch.setattr(
            cpu_info,
            "detect_cpu_features",
            lambda: {
                "avx": True,
                "avx2": True,
                "avx512": False,
                "fma": True,
                "amx": False,
            },
        )
        monkeypatch.setattr(cpu_info, "_get_paddle_version", lambda: "3.4.0")
        safe, reason = cpu_info.can_safely_enable_onednn()
        assert safe is False
        assert "尚未通过" in reason or "验证" in reason

    def test_validated_version_with_avx2_allowed(self, monkeypatch: pytest.MonkeyPatch):
        """只有显式验证过且支持 AVX2 的 Paddle 版本默认启用 oneDNN。"""
        monkeypatch.delenv("VIBEOCR_FORCE_ONEDNN", raising=False)
        monkeypatch.setattr(cpu_info, "_get_paddle_version", lambda: "3.4.1")
        monkeypatch.setattr(
            cpu_info,
            "_ONEDNN_VALIDATED_SAFE_PADDLE_RANGES",
            [("3.4.0", "3.4.2")],
        )
        monkeypatch.setattr(
            cpu_info,
            "detect_cpu_features",
            lambda: {
                "avx": True,
                "avx2": True,
                "avx512": True,
                "fma": True,
                "amx": False,
            },
        )

        safe, reason = cpu_info.can_safely_enable_onednn()

        assert safe is True
        assert "已通过验证" in reason


class TestDetectCpuFeatures:
    """detect_cpu_features 边缘用例。"""

    def test_empty_flags_all_false(self, monkeypatch: pytest.MonkeyPatch):
        """无任何 flags 文本时全部 False。"""
        monkeypatch.setattr(cpu_info, "_read_cpu_flags_text", lambda: "")
        feats = cpu_info.detect_cpu_features()
        assert feats == {
            "avx": False,
            "avx2": False,
            "avx512": False,
            "fma": False,
            "amx": False,
        }

    def test_flags_parsed(self, monkeypatch: pytest.MonkeyPatch):
        """包含 avx2/avx512f/fma 的 flags 文本被正确解析。"""
        monkeypatch.setattr(
            cpu_info, "_read_cpu_flags_text", lambda: "sse avx avx2 fma avx512f amx"
        )
        feats = cpu_info.detect_cpu_features()
        assert feats["avx"] is True
        assert feats["avx2"] is True
        assert feats["avx512"] is True
        assert feats["fma"] is True
        assert feats["amx"] is True

    def test_avx512_subsets_matched(self, monkeypatch: pytest.MonkeyPatch):
        """avx512 家族子集（avx512_vnni 等）视为 avx512 支持。"""
        monkeypatch.setattr(
            cpu_info, "_read_cpu_flags_text", lambda: "avx avx2 avx512_vnni"
        )
        feats = cpu_info.detect_cpu_features()
        assert feats["avx512"] is True
