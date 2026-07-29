"""可选的真实 PP-OCR CPU/oneDNN 冒烟门禁。

默认跳过，避免 CI 下载约 140MB 模型。发布或升级 Paddle 前，在纯 CPU 环境中
通过环境变量提供已经下载的检测/识别模型目录后运行本文件。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _smoke_model_dirs() -> tuple[Path, Path]:
    if os.environ.get("VIBEOCR_RUN_ONEDNN_SMOKE") != "1":
        pytest.skip("设置 VIBEOCR_RUN_ONEDNN_SMOKE=1 才运行真实 CPU 模型冒烟")

    det_value = os.environ.get("VIBEOCR_ONEDNN_DET_MODEL_DIR", "").strip()
    rec_value = os.environ.get("VIBEOCR_ONEDNN_REC_MODEL_DIR", "").strip()
    if not det_value or not rec_value:
        pytest.fail("必须提供 VIBEOCR_ONEDNN_DET_MODEL_DIR/REC_MODEL_DIR")

    det = Path(det_value)
    rec = Path(rec_value)
    if not det.is_dir() or not rec.is_dir():
        pytest.fail("必须提供有效的 VIBEOCR_ONEDNN_DET_MODEL_DIR/REC_MODEL_DIR")
    return det, rec


def test_smoke_model_dirs_require_explicit_values(monkeypatch):
    """启用真实门禁后，缺少模型变量应给出明确错误而非把 cwd 当模型目录。"""
    monkeypatch.setenv("VIBEOCR_RUN_ONEDNN_SMOKE", "1")
    monkeypatch.delenv("VIBEOCR_ONEDNN_DET_MODEL_DIR", raising=False)
    monkeypatch.delenv("VIBEOCR_ONEDNN_REC_MODEL_DIR", raising=False)

    with pytest.raises(pytest.fail.Exception, match="必须提供"):
        _smoke_model_dirs()


def _predict_with_real_models(*, enable_mkldnn: bool) -> int:
    det, rec = _smoke_model_dirs()
    np = pytest.importorskip("numpy")
    paddle = pytest.importorskip("paddle")
    paddleocr = pytest.importorskip("paddleocr")

    paddle.device.set_device("cpu")
    pipeline = paddleocr.PaddleOCR(
        device="cpu",
        enable_mkldnn=enable_mkldnn,
        cpu_threads=4,
        text_detection_model_dir=str(det),
        text_recognition_model_dir=str(rec),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    image = np.full((128, 384, 3), 255, dtype=np.uint8)
    return len(list(pipeline.predict(image)))


def test_real_cpu_models_work_with_onednn_disabled():
    """标准 CPU 路径必须始终能完成真实 PP-OCR 推理。"""
    assert _predict_with_real_models(enable_mkldnn=False) == 1


def test_candidate_paddle_version_is_safe_with_onednn_enabled():
    """候选 Paddle 版本只有通过本项后才能加入安全版本范围。"""
    if os.environ.get("VIBEOCR_VALIDATE_ONEDNN_CANDIDATE") != "1":
        pytest.skip(
            "仅在验证候选 Paddle 版本时设置 VIBEOCR_VALIDATE_ONEDNN_CANDIDATE=1"
        )
    assert _predict_with_real_models(enable_mkldnn=True) == 1
