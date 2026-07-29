"""Fix the paddle+torch same-process DLL conflict on Windows.

Root cause: torch and paddle both bundle cuDNN/CUDA DLLs under different
``nvidia/`` and ``torch/lib/`` directories. When both are imported in the
same process, whichever loads second finds the other's DLLs already in
memory, causing ``WinError 127`` (wrong CUDA runtime version).

The plan's architecture already isolates them by design: Paddle lives in
the supervisor process, MinerU (which needs torch) runs as a **separate
child subprocess**. So in production they never share a process. However,
test environments and dev setups may import both in one process.

This module adds all CUDA DLL directories to the Windows DLL search path
BEFORE either paddle or torch is imported, so Windows finds a consistent
set. Call :func:`ensure_dll_paths` at the very start of any entry point
that may import both.

Usage::

    from vibeocr.backend.supervisor.dll_path_fix import ensure_dll_paths
    ensure_dll_paths()  # before importing paddle or torch
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_dll_paths() -> None:
    """Add all CUDA/cuDNN DLL directories to the Windows search path.

    Must be called before importing paddle or torch. On non-Windows or when
    the directories do not exist, this is a no-op.
    """
    if sys.platform != "win32":
        return

    # Locate site-packages relative to the venv's python.exe.
    site = (Path(sys.executable).parent.parent / "Lib" / "site-packages").resolve()
    if not site.is_dir():
        # Fallback: use sys.prefix.
        site = (Path(sys.prefix) / "Lib" / "site-packages").resolve()
        if not site.is_dir():
            return

    # All directories that may contain CUDA/cuDNN DLLs.
    candidates = [
        "nvidia/cuda_runtime/bin",
        "nvidia/cublas/bin",
        "nvidia/cudnn/bin",
        "nvidia/cufft/bin",
        "nvidia/curand/bin",
        "nvidia/cusolver/bin",
        "nvidia/cusparse/bin",
        "nvidia/nvjitlink/bin",
        "torch/lib",
    ]

    for rel in candidates:
        path = site / rel
        if path.is_dir():
            try:
                import os

                os.add_dll_directory(str(path))
            except (OSError, AttributeError):
                # os.add_dll_directory may not be available (Python < 3.8) or
                # the dir may already be registered. Best-effort.
                pass


__all__ = ["ensure_dll_paths"]
