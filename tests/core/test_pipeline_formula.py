# tests/core/test_pipeline_formula.py
from vibeocr.backend.core.pipelines.pipeline_formula import (
    FORMULA_RECOGNITION_SPEC,
    FormulaRecognitionOptions,
    _recognize_formula,
)


def test_formula_options_defaults():
    opts = FormulaRecognitionOptions()
    assert opts.pipeline == "FORMULA_RECOGNITION"
    assert opts.formula_recognition_batch_size == 1
    assert opts.formula_recognition_model_name is None


def test_formula_options_to_dict():
    opts = FormulaRecognitionOptions(formula_recognition_batch_size=4)
    d = opts.to_dict()
    assert d["formula_recognition_batch_size"] == 4
    assert d["pipeline"] == "FORMULA_RECOGNITION"


def test_formula_spec():
    assert FORMULA_RECOGNITION_SPEC.name == "FORMULA_RECOGNITION"
    assert FORMULA_RECOGNITION_SPEC.display_name == "公式识别"
    assert FORMULA_RECOGNITION_SPEC.options_class is FormulaRecognitionOptions


class _DictResult(dict):
    """模拟 PaddleX 结果：dict 子类，parsing_res_list 是 dict key。"""


class _LayoutBlock:
    """模拟 paddlex LayoutBlock（普通对象，属性访问）。"""

    def __init__(self, label, content, bbox=None, order_index=-1):
        self.label = label
        self.content = content
        self.bbox = bbox or [1, 2, 3, 4]
        self.order_index = order_index


class _FakePipeline:
    def __init__(self, result_list):
        self._result_list = result_list

    def predict(self, input, **kwargs):  # noqa: A002 — 模拟 PaddleOCR API（input 关键字参数）
        return list(self._result_list)


class _FakeService:
    def __init__(self, result_list):
        self._pipeline = _FakePipeline(result_list)

    def get_or_create_pipeline(self, name):
        return self._pipeline


def test_recognize_formula_extracts_from_dict_result():
    """回归：parsing_res_list 必须用下标访问（dict 子类）。

    修复前 getattr(res, "parsing_res_list", []) 对 dict 子类恒返回 []，
    导致公式识别永远返回空。
    """
    res = _DictResult(
        {
            "doc_preprocessor_res": None,
            "parsing_res_list": [
                _LayoutBlock(label="formula", content=r"a^2 + b^2"),
                _LayoutBlock(label="text", content="not a formula"),
            ],
        }
    )
    service = _FakeService([res])
    result = _recognize_formula(
        service, image=None, options=FormulaRecognitionOptions()
    )

    assert result.pipeline_type == "FORMULA_RECOGNITION"
    # 仅提取 label=="formula" 的块
    assert len(result.text_blocks) == 1
    assert result.text_blocks[0].label == "formula"
    assert "a^2 + b^2" in result.raw_text
    assert "$$" in result.markdown_text



def test_recognize_formula_with_preprocessed_output_img():
    """doc_preprocessor_res 含 output_img 时提取预处理图与角度（line 92-103）。"""
    import numpy as np

    # 构造一个 2x2 RGB 数组
    rgb_arr = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb_arr[..., 0] = 255  # R 通道
    res = _DictResult(
        {
            "doc_preprocessor_res": {
                "angle": 90,
                "output_img": rgb_arr,
            },
            "parsing_res_list": [
                _LayoutBlock(label="formula", content=r"x = 1"),
            ],
        }
    )
    service = _FakeService([res])
    result = _recognize_formula(
        service, image=None, options=FormulaRecognitionOptions()
    )
    assert result.preproc_angle == 90
    assert result.preproc_img_w == 2
    assert result.preproc_img_h == 2
    assert result.preprocessed_image  # 预处理 PNG 存入 preprocessed_image 字段
    assert result.preprocessed_image.startswith(b"\x89PNG")


def test_recognize_formula_with_batch_size_option():
    """formula_recognition_batch_size != 1 时透传给 predict（line 73）。"""
    res = _DictResult(
        {
            "doc_preprocessor_res": None,
            "parsing_res_list": [],
        }
    )
    pipeline = _FakePipeline([res])
    captured = {}

    orig_predict = pipeline.predict

    def capturing_predict(input, **kwargs):  # noqa: A002
        captured.update(kwargs)
        return orig_predict(input=input, **kwargs)

    pipeline.predict = capturing_predict
    service = _FakeService([res])
    service._pipeline = pipeline

    opts = FormulaRecognitionOptions(formula_recognition_batch_size=4)
    _recognize_formula(service, image=None, options=opts)
    assert captured.get("formula_recognition_batch_size") == 4


def test_recognize_formula_empty_output():
    """predict 返回空列表时不崩溃（line 88->105 分支）。"""
    service = _FakeService([])
    result = _recognize_formula(
        service, image=None, options=FormulaRecognitionOptions()
    )
    assert result.raw_text == ""
    assert result.text_blocks == []
