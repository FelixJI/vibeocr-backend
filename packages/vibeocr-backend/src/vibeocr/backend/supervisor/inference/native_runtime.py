"""Prepare process-wide Windows native dependencies for OCR adapters.

PyWinRT 3.2.1 ships an app-local ``msvcp140.dll``.  If WinRT is imported
before ONNX Runtime, Windows can bind ONNX Runtime to that older STL runtime
and fail its module initializer.  Load ONNX Runtime first, without creating a
RapidOCR instance or loading any model, so both adapters share a compatible
process-wide runtime.
"""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def prepare_windows_ocr_native_runtime(
    import_module: Callable[[str], Any] = importlib.import_module,
) -> None:
    """Load ONNX Runtime before any PyWinRT projection on Windows.

    Missing or intrinsically broken ONNX Runtime must not prevent the
    Supervisor from serving the independent Windows OCR adapter.  The
    RapidOCR descriptor will report that failure through its normal probe.
    """
    if sys.platform != "win32":
        return
    try:
        import_module("onnxruntime")
    except Exception:
        logger.exception("[Supervisor][NativeRuntime] ONNX Runtime preload failed")


__all__ = ["prepare_windows_ocr_native_runtime"]
