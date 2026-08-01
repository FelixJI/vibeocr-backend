"""Tests for OCRService CUDA DLL path setup.

Verifies that ``_setup_cuda_dll_path`` and ``_register_dll_directories`` use
``torch/lib`` only as a fallback when Paddle does not bundle a CUDA runtime.

Background: torch wheels ship a complete CUDA 12 + cuDNN 9 runtime under
``torch/lib``. Without registering that directory, paddle falls back to CPU with
``error code 126`` when the system has no CUDA Toolkit and only ``nvidia/cu13``
(which provides ``cublas64_13.dll``, ABI-incompatible with paddle's CUDA 12 build).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from vibeocr.backend.services.ocr_service import OCRService


def _site_packages() -> Path | None:
    return next((Path(p) for p in sys.path if "site-packages" in p), None)


@pytest.mark.skipif(sys.platform != "win32", reason="DLL path setup is Windows-only")
class TestSetupCudaDllPathIncludesTorchLib:
    """``_setup_cuda_dll_path`` must add ``torch/lib`` to PATH."""

    def setup_method(self) -> None:
        # Reset the idempotency flag so each test re-runs the setup.
        OCRService._cuda_dll_registered = False

    def teardown_method(self) -> None:
        OCRService._cuda_dll_registered = False

    def test_torch_lib_follows_paddle_runtime_policy(self) -> None:
        """Do not mix Torch CUDA DLLs into a Paddle-bundled CUDA runtime."""
        site = _site_packages()
        if site is None or not (site / "torch" / "lib").is_dir():
            pytest.skip("torch/lib not present in this environment")

        torch_lib = str(site / "torch" / "lib")
        # Ensure clean baseline: remove torch/lib from PATH if already there.
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        path_parts = [p for p in path_parts if p != torch_lib]
        with patch.dict(os.environ, {"PATH": os.pathsep.join(path_parts)}):
            assert torch_lib not in os.environ["PATH"].split(os.pathsep)

            OCRService._setup_cuda_dll_path()

            import paddle.version

            paddle_has_cuda = bool(paddle.version.cuda())
            assert (torch_lib in os.environ["PATH"].split(os.pathsep)) is (
                not paddle_has_cuda
            )

    def test_idempotent(self) -> None:
        """Calling twice does not duplicate entries."""
        OCRService._setup_cuda_dll_path()
        first = os.environ.get("PATH", "")
        OCRService._setup_cuda_dll_path()
        second = os.environ.get("PATH", "")
        assert first == second

    def test_nvidia_dirs_still_present(self) -> None:
        """nvidia/* dirs are still registered (no regression)."""
        site = _site_packages()
        if site is None or not (site / "nvidia").is_dir():
            pytest.skip("nvidia/* not present in this environment")

        OCRService._setup_cuda_dll_path()

        # At least one nvidia subdir path should be in PATH.
        nv_base = site / "nvidia"
        path_parts = set(os.environ["PATH"].split(os.pathsep))
        found = any(
            str(p).startswith(str(nv_base)) for p in nv_base.iterdir() if p.is_dir()
        ) or any(part.startswith(str(nv_base)) for part in path_parts)
        assert found, "no nvidia/* path in PATH after setup"


@pytest.mark.skipif(sys.platform != "win32", reason="DLL path setup is Windows-only")
@pytest.mark.skipif(
    not hasattr(os, "add_dll_directory"), reason="os.add_dll_directory unavailable"
)
class TestRegisterDllDirectoriesIncludesTorchLib:
    """``add_dll_directory`` follows the same Paddle/Torch runtime policy."""

    def test_torch_lib_registration_follows_paddle_runtime_policy(self) -> None:
        site = _site_packages()
        if site is None or not (site / "torch" / "lib").is_dir():
            pytest.skip("torch/lib not present in this environment")

        torch_lib = str(site / "torch" / "lib")
        seen: list[str] = []
        with patch("os.add_dll_directory", side_effect=lambda d: seen.append(d)):
            OCRService._register_dll_directories()

        import paddle.version

        assert (torch_lib in seen) is (not bool(paddle.version.cuda()))
