# tests/core/test_preproc_channel_order.py
"""回归测试：预处理图 PNG 的颜色通道顺序。

背景：PaddleOCR 3.7 的 doc_preprocessor_res['output_img'] 实际为 RGB。
旧代码误以为是 BGR 并做了 [:, :, ::-1] 翻转，导致 R/B 对调——
表现为 bbox 预览界面「黄色文件夹变青蓝色」。

本测试直接验证 pipeline_ocr._extract_preproc_info 的输出，
不依赖 PaddleOCR / GPU，可在任意 CI 环境运行。
复现脚本见 scripts/verify_preproc_channels.py。

pipeline_formula / pipeline_pp_structure / ocr_service 中的内联逻辑
与 _extract_preproc_info 完全一致（同样从 output_img 直接转 RGB），
由本测试 + 代码审查共同覆盖。
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from vibeocr.backend.core.pipelines.pipeline_ocr import _extract_preproc_info


class _FakeRes:
    """模拟 PaddleOCR 结果项（仅需 get('doc_preprocessor_res')）。"""

    def __init__(self, dp_res: dict | None) -> None:
        self._dp = dp_res

    def get(self, key, default=None):
        if key == "doc_preprocessor_res":
            return self._dp
        return default


def _build_rgb_blocks() -> np.ndarray:
    """构造 4x2 色块图（RGB），每块 50x50，总 200x100。

    含红/绿/蓝/黄/青/品红/白/黑——足以区分 R/B 是否被翻转。
    """
    block = 50
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 255),
        (0, 0, 0),
    ]
    img = np.zeros((2 * block, 4 * block, 3), dtype=np.uint8)
    for i, c in enumerate(colors):
        r, cc = divmod(i, 4)
        img[r * block : (r + 1) * block, cc * block : (cc + 1) * block] = c
    return img


def _block_pixel(arr: np.ndarray, idx: int, block: int = 50) -> tuple[int, int, int]:
    r, cc = divmod(idx, 4)
    y = r * block + block // 2
    x = cc * block + block // 2
    px = arr[y, x][:3]
    return int(px[0]), int(px[1]), int(px[2])


EXPECTED = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 255),
    (0, 0, 0),
]
NAMES = ["红", "绿", "蓝", "黄", "青", "品红", "白", "黑"]


def _assert_png_matches_rgb(png: bytes, src: np.ndarray) -> None:
    """解码 PNG，逐色块断言像素 == 源 RGB（不允许 R/B 翻转）。"""
    decoded = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
    assert decoded.shape[:2] == src.shape[:2], (
        f"PNG 尺寸 {decoded.shape[:2]} 与源 {src.shape[:2]} 不符"
    )
    mismatches = []
    for i, (name, exp) in enumerate(zip(NAMES, EXPECTED, strict=True)):
        got = _block_pixel(decoded, i)
        if got != exp:
            flipped = (got[2], got[1], got[0])
            tag = "（疑似 R/B 被翻转）" if flipped == exp else ""
            mismatches.append(f"{name}: 期望{exp} 实际{got} {tag}")
    assert not mismatches, "颜色通道异常:\n  " + "\n  ".join(mismatches)


def _make_res(src: np.ndarray) -> _FakeRes:
    return _FakeRes({"angle": 0, "output_img": src.copy()})


def test_extract_preproc_info_no_channel_swap():
    """_extract_preproc_info 输出 PNG 颜色不应被翻转（8 色块全匹配）。"""
    src = _build_rgb_blocks()
    _angle, png, w, h = _extract_preproc_info(_make_res(src))
    assert png is not None, "preprocessed_png 为空"
    assert (w, h) == (src.shape[1], src.shape[0])
    _assert_png_matches_rgb(png, src)


def test_yellow_block_not_flipped_to_cyan():
    """针对性回归：黄色 (255,255,0) 不能变成青蓝 (0,255,255)。

    这是用户报告的「黄色文件夹变蓝色」的直接断言。
    """
    src = np.full((10, 10, 3), (255, 255, 0), dtype=np.uint8)
    _angle, png, _w, _h = _extract_preproc_info(_make_res(src))
    assert png is not None
    decoded = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
    px = tuple(int(v) for v in decoded[5, 5])
    assert px == (255, 255, 0), f"黄色块被错误翻转成 {px}（期望 255,255,0）"


def test_missing_doc_preprocessor_res_returns_none():
    """无 doc_preprocessor_res 时 preprocessed_png 应为 None。"""
    _angle, png, _w, _h = _extract_preproc_info(_FakeRes(None))
    assert png is None
    assert _angle == 0


def test_readonly_array_handled():
    """output_img 可能是不可写视图，转换不应报错（取了 .copy()）。"""
    base = _build_rgb_blocks()
    readonly = np.asarray(base)  # asarray 可能返回只读视图
    readonly.setflags(write=False)
    _angle, png, _w, _h = _extract_preproc_info(_make_res(readonly))
    assert png is not None
    _assert_png_matches_rgb(png, base)


def test_extract_preproc_info_can_skip_png_but_preserve_geometry():
    """批量 PDF 路径不编码大 PNG，但方向与尺寸仍须用于 bbox 回写。"""
    src = _build_rgb_blocks()
    res = _FakeRes({"angle": 90, "output_img": src})

    angle, png, width, height = _extract_preproc_info(res, include_image=False)

    assert angle == 90
    assert png is None
    assert (width, height) == (src.shape[1], src.shape[0])
