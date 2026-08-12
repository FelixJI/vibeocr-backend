import sys
from types import ModuleType

from vibeocr.backend.core.pipelines.pipeline_table import (
    TABLE_RECOGNITION_SPEC,
    TableRecognitionOptions,
    _recognize_table,
)


def _install_fake_paddlex_deps(monkeypatch):
    """注入最小 PaddleX deps 模块，避免单测依赖可选 OCR 运行时。"""
    paddlex = ModuleType("paddlex")
    utils = ModuleType("paddlex.utils")
    deps = ModuleType("paddlex.utils.deps")
    deps.EXTRAS = {
        "ocr": ["paddleocr", "opencv-contrib-python", "pyclipper"],
    }
    deps.is_extra_available = lambda extra: True
    deps.is_dep_available = lambda dep: True
    paddlex.utils = utils
    utils.deps = deps
    monkeypatch.setitem(sys.modules, "paddlex", paddlex)
    monkeypatch.setitem(sys.modules, "paddlex.utils", utils)
    monkeypatch.setitem(sys.modules, "paddlex.utils.deps", deps)
    return deps


def test_table_options_defaults():
    opts = TableRecognitionOptions()
    assert opts.pipeline == "TABLE_RECOGNITION"
    assert opts.use_wireless_table is True
    assert opts.formula_recognition_batch_size == 1


def test_table_options_thresholds_default_none():
    """检测/识别阈值默认 None，沿用 PaddleX 模型默认值。

    回归保护：曾误把这些默认值改成截图友好的固定值（0.2/2.5），但会全局
    引入误检（表格线/底纹被当文字），且漏字真因在 IoU 匹配不在检测参数。
    正确做法是保留 None 让 PaddleX 默认值生效，极端场景由用户显式覆盖。
    """
    opts = TableRecognitionOptions()
    assert opts.text_det_thresh is None
    assert opts.text_det_box_thresh is None
    assert opts.text_det_unclip_ratio is None
    assert opts.text_rec_score_thresh is None
    assert opts.text_det_limit_side_len is None


def test_table_options_to_dict():
    opts = TableRecognitionOptions(use_wireless_table=False)
    d = opts.to_dict()
    assert d["use_wireless_table"] is False
    assert d["pipeline"] == "TABLE_RECOGNITION"


def test_table_options_from_dict():
    d = {"pipeline": "TABLE_RECOGNITION", "use_wireless_table": False}
    opts = TableRecognitionOptions.from_dict(d)
    assert opts.use_wireless_table is False


def test_table_spec():
    assert TABLE_RECOGNITION_SPEC.name == "TABLE_RECOGNITION"
    assert TABLE_RECOGNITION_SPEC.display_name == "表格识别"
    assert TABLE_RECOGNITION_SPEC.options_class is TableRecognitionOptions


class TestCheckTableDeps:
    """_check_table_deps / TableDependencyError 单测。

    验证管道创建前的依赖探测能拦截 paddlex[ocr] 依赖缺失，
    报告具体缺失发行版名，而非让 PaddleX 抛无信息的 RuntimeError。

    关键：_check_table_deps 复用 PaddleX 的 is_extra_available / is_dep_available
    （与 @pipeline_requires_extra("ocr") 同一判定路径），故测试 mock paddlex.deps
    模块，而非 importlib.util.find_spec。
    """

    def test_all_present_passes(self, monkeypatch):
        """is_extra_available('ocr') 返回 True 时不抛错。"""
        from vibeocr.backend.core.pipelines import pipeline_table

        pdx_deps = _install_fake_paddlex_deps(monkeypatch)
        monkeypatch.setattr(pdx_deps, "is_extra_available", lambda extra: True)
        monkeypatch.setattr(pdx_deps, "is_dep_available", lambda dep: True)
        # 不抛异常即通过
        pipeline_table._check_table_deps()

    def test_missing_raises_with_package_names(self, monkeypatch):
        """is_extra_available 返回 False 时抛错，且消息含具体缺失发行版名。"""
        from vibeocr.backend.core.pipelines import pipeline_table

        pdx_deps = _install_fake_paddlex_deps(monkeypatch)
        ocr_deps = list(pdx_deps.EXTRAS.get("ocr", []))
        missing_pkg = ocr_deps[0]
        monkeypatch.setattr(pdx_deps, "is_extra_available", lambda extra: False)
        monkeypatch.setattr(
            pdx_deps,
            "is_dep_available",
            lambda dep: dep != missing_pkg,
        )
        try:
            pipeline_table._check_table_deps()
        except pipeline_table.TableDependencyError as e:
            msg = str(e)
            assert missing_pkg in msg, f"错误消息应含缺失包名 {missing_pkg}: {msg}"
            assert "设置" in msg or "重装" in msg, f"应引导用户重装: {msg}"
        else:
            raise AssertionError("缺失时应抛 TableDependencyError")

    def test_multiple_missing_all_listed(self, monkeypatch):
        """多个包缺失时全部列在错误消息里。"""
        from vibeocr.backend.core.pipelines import pipeline_table

        pdx_deps = _install_fake_paddlex_deps(monkeypatch)
        # 让 ocr extra 前 3 个包不可用
        missing_pkgs = list(pdx_deps.EXTRAS.get("ocr", []))[:3]
        monkeypatch.setattr(pdx_deps, "is_extra_available", lambda extra: False)
        monkeypatch.setattr(
            pdx_deps,
            "is_dep_available",
            lambda dep: dep not in missing_pkgs,
        )
        try:
            pipeline_table._check_table_deps()
        except pipeline_table.TableDependencyError as e:
            msg = str(e)
            for pkg in missing_pkgs:
                assert pkg in msg, f"错误消息应含 {pkg}: {msg}"
        else:
            raise AssertionError("缺失时应抛 TableDependencyError")

    def test_paddlex_import_error_falls_back_to_find_spec(self, monkeypatch):
        """paddlex 不可导入时，回退到 find_spec 兜底探测仍能拦截缺失。

        覆盖极端残缺环境（paddlex 本身未装）：_check_table_deps 不能因 paddlex
        导入失败而静默放过，应回退到本项目 leaf 清单的 find_spec 探测。
        """
        # 让 from paddlex.utils.deps import ... 抛 ImportError
        import builtins

        from vibeocr.backend.core.pipelines import pipeline_table
        from vibeocr.backend.dependency_spec import OCR_CHECK_LEAF_MODULES

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("paddlex.utils.deps"):
                raise ImportError("simulated: paddlex not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        # find_spec 让第一个 leaf 缺失
        missing_mod = next(iter(OCR_CHECK_LEAF_MODULES.keys()))
        import importlib.util

        def fake_find_spec(name, package=None):
            return None if name == missing_mod else object()

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        try:
            pipeline_table._check_table_deps()
        except pipeline_table.TableDependencyError as e:
            assert "设置" in str(e) or "重装" in str(e)
        else:
            raise AssertionError("回退路径缺失时也应抛 TableDependencyError")


class _DictResult(dict):
    """模拟 PaddleX TableRecognitionResult：dict 子类，键需下标访问。

    真实结构 keys: input_path, page_index, doc_preprocessor_res, layout_det_res,
    overall_ocr_res, table_res_list, model_settings。
    表格内容在 table_res_list[i].pred_html，不是 parsing_res_list。
    """


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


def _make_table_result(
    pred_html="<html><body><table><tr><td>Name</td><td>Age</td></tr></table></body></html>",
    ocr_texts=None,
    cell_box_list=None,
    ocr_polys=None,
):
    """构造模拟结果。

    cell_box_list 对应 PaddleX 真实字段：每个单元格的 [x1,y1,x2,y2]
    （原图坐标系）。PaddleX 的 SingleTableRecognitionResult 不含
    ``table_bbox`` 字段，表格整体框需从 cell_box_list 的并集推导。
    """
    table_entry = {"pred_html": pred_html, "table_region_id": 1}
    if cell_box_list is not None:
        table_entry["cell_box_list"] = cell_box_list
    res: dict[str, object] = {
        "table_res_list": [table_entry],
    }
    if ocr_texts is not None:
        ocr_entry = {
            "rec_texts": ocr_texts,
            "rec_scores": [0.9] * len(ocr_texts),
        }
        if ocr_polys is not None:
            ocr_entry["rec_polys"] = ocr_polys
        res["overall_ocr_res"] = ocr_entry
    return _DictResult(res)


def test_recognize_table_extracts_pred_html_from_table_res_list():
    """回归：表格内容在 table_res_list[].pred_html，不是 parsing_res_list。

    修复前代码读 parsing_res_list（表格管道无此字段），永远返回空，
    表现为"未识别到文字"。
    """
    service = _FakeService([_make_table_result()])
    result = _recognize_table(service, image=None, options=TableRecognitionOptions())

    assert result.pipeline_type == "TABLE_RECOGNITION"
    assert len(result.text_blocks) == 1
    assert result.text_blocks[0].label == "table"
    assert result.raw_text == "Name\tAge"
    assert len(result.content_list) == 1
    assert result.content_list[0]["type"] == "table"
    assert result.content_list[0]["table_body"].startswith("<table")
    # markdown 表格已转换
    assert "Name" in result.markdown_text and "Age" in result.markdown_text


def test_recognize_table_extracts_surrounding_ocr_text():
    """overall_ocr_res 中的文字（表格外）也应被提取。"""
    res = _make_table_result(ocr_texts=["标题", "脚注"])
    service = _FakeService([res])
    result = _recognize_table(service, image=None, options=TableRecognitionOptions())

    texts = [b.text for b in result.text_blocks]
    assert "标题" in texts
    assert "脚注" in texts
    # 仍含表格块
    assert any(b.label == "table" for b in result.text_blocks)
    for text_block in result.text_blocks:
        content = result.content_list[text_block.content_index]
        assert content["block_id"] == text_block.content_id


def test_recognize_table_empty_result():
    """无表格块时返回空但不报错。"""
    service = _FakeService([_DictResult({"table_res_list": []})])
    result = _recognize_table(service, image=None, options=TableRecognitionOptions())
    assert result.raw_text == ""
    assert result.content_list == []


# ---- 空单元格回填（soft fallback）相关测试 ----
#
# 根因：PaddleX 内部 match_table_and_ocr 要求 IoU>0.7 才把 OCR 文字填进
# 单元格，失配时输出空 <td></td>；但那些文字其实躺在 overall_ocr_res 里。
# 我们原本的去重逻辑把中心落在表格 bbox 内的 overall 文本全部丢弃 →
# 上游漏填的字被二次抹杀。修复：把落在"空单元格"内的 overall 文本回填
# 进单元格；未被吸收但落在表内的文本也不再丢弃，保留为独立 text 块。


class TestBackfillEmptyTableCells:
    """_backfill_empty_table_cells 单元测试。

    不依赖 paddle / Qt，纯函数测试。验证回填几何匹配、不覆盖已填单元格、
    返回被消费的 OCR 索引集合。
    """

    def test_backfills_text_into_empty_cell(self):
        """空单元格（<td></td>）中心落有 OCR 文本 → 回填进该单元格。"""
        from vibeocr.backend.core.pipelines.pipeline_table import (
            _backfill_empty_table_cells,
        )

        # 2x1 表格：第一格空，第二格已填
        table_html = "<table><tr><td></td><td>已填</td></tr></table>"
        cell_box_list = [
            [10.0, 10.0, 100.0, 50.0],  # 空单元格 idx=0
            [100.0, 10.0, 200.0, 50.0],  # 已填单元格 idx=1
        ]
        # OCR 文本中心 (55, 30) 落在第一个（空）单元格内
        ocr_items = [
            {"text": "漏掉的文字", "center": (55.0, 30.0)},
        ]
        new_html, consumed = _backfill_empty_table_cells(
            table_html, cell_box_list, ocr_items
        )
        assert "漏掉的文字" in new_html
        assert consumed == {0}
        # 已填单元格不被覆盖
        assert "已填" in new_html

    def test_absorbs_conflicting_text_in_filled_cell(self):
        """已填单元格吸收额外 OCR，保留文字且不再生成独立文本框。"""
        from vibeocr.backend.core.pipelines.pipeline_table import (
            _backfill_empty_table_cells,
        )

        table_html = "<table><tr><td>原有</td></tr></table>"
        cell_box_list = [[10.0, 10.0, 100.0, 50.0]]  # 已填单元格
        ocr_items = [{"text": "不该回填", "center": (50.0, 30.0)}]
        new_html, consumed = _backfill_empty_table_cells(
            table_html, cell_box_list, ocr_items
        )
        assert "原有" in new_html
        assert "不该回填" in new_html
        assert consumed == {0}

    def test_filled_cells_absorb_unmatched_ocr(self):
        """没有空单元格时仍吸收表内 OCR，避免独立文本框重复。"""
        from vibeocr.backend.core.pipelines.pipeline_table import (
            _backfill_empty_table_cells,
        )

        table_html = "<table><tr><td>A</td><td>B</td></tr></table>"
        cell_box_list = [[10, 10, 50, 50], [50, 10, 100, 50]]
        ocr_items = [{"text": "X", "center": (30.0, 30.0)}]
        new_html, consumed = _backfill_empty_table_cells(
            table_html, cell_box_list, ocr_items
        )
        assert "A<br>X" in new_html
        assert consumed == {0}

    def test_none_cell_box_list_returns_unchanged(self):
        """无 cell_box_list 时安全降级：不回填、不崩。"""
        from vibeocr.backend.core.pipelines.pipeline_table import (
            _backfill_empty_table_cells,
        )

        table_html = "<table><tr><td></td></tr></table>"
        ocr_items = [{"text": "X", "center": (30.0, 30.0)}]
        new_html, consumed = _backfill_empty_table_cells(table_html, None, ocr_items)
        assert new_html == table_html
        assert consumed == set()

    def test_none_center_ocr_skipped(self):
        """OCR 项无 center（缺 poly）时不参与回填，不崩。"""
        from vibeocr.backend.core.pipelines.pipeline_table import (
            _backfill_empty_table_cells,
        )

        table_html = "<table><tr><td></td></tr></table>"
        cell_box_list = [[10.0, 10.0, 100.0, 50.0]]
        ocr_items = [{"text": "无坐标", "center": None}]
        new_html, consumed = _backfill_empty_table_cells(
            table_html, cell_box_list, ocr_items
        )
        assert new_html == table_html
        assert consumed == set()

    def test_multiple_empty_cells_distribute_by_geometry(self):
        """多个空单元格时，OCR 按几何落点分配到对应单元格（不靠位置序号）。"""
        from vibeocr.backend.core.pipelines.pipeline_table import (
            _backfill_empty_table_cells,
        )

        # 一行两空单元格
        table_html = "<table><tr><td></td><td></td></tr></table>"
        cell_box_list = [
            [10.0, 10.0, 100.0, 50.0],
            [100.0, 10.0, 200.0, 50.0],
        ]
        ocr_items = [
            {"text": "左字", "center": (50.0, 30.0)},
            {"text": "右字", "center": (150.0, 30.0)},
        ]
        new_html, consumed = _backfill_empty_table_cells(
            table_html, cell_box_list, ocr_items
        )
        # 两个都被回填
        assert "左字" in new_html and "右字" in new_html
        assert consumed == {0, 1}
        # 左字在第一个单元格、右字在第二个（顺序保留）
        left_pos = new_html.index("左字")
        right_pos = new_html.index("右字")
        assert left_pos < right_pos

    def test_unordered_cell_boxes_are_matched_in_visual_order(self):
        from vibeocr.backend.core.pipelines.pipeline_table import (
            _backfill_empty_table_cells,
        )

        table_html = "<table><tr><td></td><td></td></tr></table>"
        cell_box_list = [
            [100.0, 10.0, 200.0, 50.0],
            [10.0, 10.0, 100.0, 50.0],
        ]
        ocr_items = [{"text": "左字", "center": (50.0, 30.0)}]

        new_html, consumed = _backfill_empty_table_cells(
            table_html, cell_box_list, ocr_items
        )

        assert "<td>左字</td><td></td>" in new_html
        assert consumed == {0}

    def test_same_row_y_jitter_does_not_override_left_to_right_order(self):
        from vibeocr.backend.core.pipelines.pipeline_table import (
            _backfill_empty_table_cells,
        )

        table_html = "<table><tr><td></td><td></td></tr></table>"
        cell_box_list = [
            [100.0, 9.0, 200.0, 51.0],
            [10.0, 10.0, 100.0, 50.0],
        ]
        ocr_items = [{"text": "左字", "center": (50.0, 30.0)}]

        new_html, _ = _backfill_empty_table_cells(table_html, cell_box_list, ocr_items)

        assert "<td>左字</td><td></td>" in new_html

    def test_middle_column_rowspan_maps_boxes_to_logical_cells(self):
        from vibeocr.backend.core.pipelines.pipeline_table import (
            _backfill_empty_table_cells,
        )

        table_html = (
            "<table>"
            "<tr><td></td><td rowspan='2'></td><td></td></tr>"
            "<tr><td></td><td></td></tr>"
            "</table>"
        )
        cell_box_list = [
            [20.0, 10.0, 30.0, 20.0],
            [10.0, 0.0, 20.0, 20.0],
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 0.0, 30.0, 10.0],
            [0.0, 10.0, 10.0, 20.0],
        ]
        ocr_items = [
            {"text": "D", "center": (5.0, 15.0)},
            {"text": "B", "center": (15.0, 15.0)},
            {"text": "C", "center": (25.0, 5.0)},
        ]

        new_html, consumed = _backfill_empty_table_cells(
            table_html, cell_box_list, ocr_items
        )

        assert (
            "<tr><td></td><td rowspan='2'>B</td><td>C</td></tr>"
            "<tr><td>D</td><td></td></tr>"
        ) in new_html
        assert consumed == {0, 1, 2}

    def test_combined_colspan_rowspan_maps_by_logical_grid(self):
        from vibeocr.backend.core.pipelines.pipeline_table import (
            _backfill_empty_table_cells,
        )

        table_html = (
            "<table>"
            "<tr><td rowspan='2' colspan='2'></td><td></td></tr>"
            "<tr><td></td></tr>"
            "<tr><td></td><td></td><td></td></tr>"
            "</table>"
        )
        cell_box_list = [
            [20.0, 10.0, 30.0, 20.0],
            [10.0, 20.0, 20.0, 30.0],
            [0.0, 0.0, 20.0, 20.0],
            [20.0, 0.0, 30.0, 10.0],
            [0.0, 20.0, 10.0, 30.0],
            [20.0, 20.0, 30.0, 30.0],
        ]
        ocr_items = [
            {"text": "MERGED", "center": (15.0, 15.0)},
            {"text": "BOTTOM", "center": (15.0, 25.0)},
        ]

        new_html, consumed = _backfill_empty_table_cells(
            table_html, cell_box_list, ocr_items
        )

        assert "rowspan='2' colspan='2'>MERGED</td>" in new_html
        assert "<td></td><td>BOTTOM</td><td></td>" in new_html
        assert consumed == {0, 1}

    def test_missing_box_skips_unprovable_mapping(self, caplog):
        from vibeocr.backend.core.pipelines.pipeline_table import (
            _backfill_empty_table_cells,
        )

        table_html = "<table><tr><td></td><td></td></tr></table>"
        ocr_items = [{"text": "RIGHT", "center": (15.0, 5.0)}]

        new_html, consumed = _backfill_empty_table_cells(
            table_html,
            [[10.0, 0.0, 20.0, 10.0]],
            ocr_items,
        )

        assert new_html == table_html
        assert consumed == set()
        assert "无法从 cell_box_list 证明" in caplog.text

    def test_extra_valid_box_skips_entire_table_mapping(self):
        from vibeocr.backend.core.pipelines.pipeline_table import (
            _backfill_empty_table_cells,
        )

        table_html = "<table><tr><td></td></tr></table>"
        mapping: dict = {}
        warnings: list[str] = []

        new_html, consumed = _backfill_empty_table_cells(
            table_html,
            [
                [0.0, 0.0, 10.0, 10.0],
                [0.0, 0.0, 10.0, 10.0],
            ],
            [{"text": "X", "center": (5.0, 5.0)}],
            table_id="table-extra",
            mapping_out=mapping,
            warnings_out=warnings,
        )

        assert new_html == table_html
        assert consumed == set()
        assert mapping == {}
        assert warnings == ["table-extra:cell-all:cell-box-count-mismatch"]

    def test_oversized_rowspan_is_rejected_before_grid_expansion(self):
        from vibeocr.backend.core.pipelines.pipeline_table import (
            _backfill_empty_table_cells,
        )

        table_html = "<table><tr><td rowspan='10001'></td></tr></table>"
        warnings: list[str] = []

        new_html, consumed = _backfill_empty_table_cells(
            table_html,
            [[0.0, 0.0, 10.0, 10.0]],
            [{"text": "X", "center": (5.0, 5.0)}],
            table_id="table-limit",
            warnings_out=warnings,
        )

        assert new_html == table_html
        assert consumed == set()
        assert warnings == ["table-limit:cell-0:span-limit-exceeded"]

    def test_oversized_span_coverage_is_rejected_before_grid_expansion(self):
        from vibeocr.backend.core.pipelines.pipeline_table import (
            _backfill_empty_table_cells,
        )

        table_html = "<table><tr><td rowspan='1001' colspan='1000'></td></tr></table>"
        warnings: list[str] = []

        new_html, consumed = _backfill_empty_table_cells(
            table_html,
            [[0.0, 0.0, 10.0, 10.0]],
            [{"text": "X", "center": (5.0, 5.0)}],
            table_id="table-coverage",
            warnings_out=warnings,
        )

        assert new_html == table_html
        assert consumed == set()
        assert warnings == ["table-coverage:cell-0:span-limit-exceeded"]

    def test_overlapping_cell_boxes_choose_smallest_containing_box(self):
        from vibeocr.backend.core.pipelines.pipeline_table import (
            _backfill_empty_table_cells,
        )

        table_html = "<table><tr><td></td><td></td></tr></table>"
        cell_box_list = [
            [0.0, 0.0, 200.0, 50.0],
            [100.0, 0.0, 200.0, 50.0],
        ]
        ocr_items = [{"text": "右字", "center": (150.0, 25.0)}]

        new_html, _ = _backfill_empty_table_cells(table_html, cell_box_list, ocr_items)

        assert "<td></td><td>右字</td>" in new_html

    def test_invalid_cell_box_does_not_displace_valid_geometry(self):
        from vibeocr.backend.core.pipelines.pipeline_table import (
            _backfill_empty_table_cells,
        )

        table_html = "<table><tr><td></td></tr></table>"
        cell_box_list = [
            [False, 0.0, 20.0, 50.0],
            [100.0, 0.0, 200.0, 50.0],
        ]
        ocr_items = [{"text": "有效", "center": (150.0, 25.0)}]

        new_html, consumed = _backfill_empty_table_cells(
            table_html, cell_box_list, ocr_items
        )

        assert "<td>有效</td>" in new_html
        assert consumed == {0}


def test_recognize_table_backfills_empty_cell_from_ocr():
    """端到端：pred_html 含空 <td></td>，overall_ocr_res 的文字回填进该单元格。

    回归根因：PaddleX IoU 失配输出空单元格，旧逻辑又把兜底的 overall 文本
    丢弃 → 漏字。修复后该文字出现在表格 HTML 与 markdown 中，且不再作为
    独立 text 块重复展示。
    """
    import numpy as np

    pred_html = "<table><tr><td></td><td>已填</td></tr></table>"
    cell_box_list = [
        [10.0, 10.0, 100.0, 50.0],  # 空
        [100.0, 10.0, 200.0, 50.0],  # 已填
    ]
    # OCR 文本中心 (55,30) 落在空单元格内
    poly = np.array([[40, 20], [70, 20], [70, 40], [40, 40]], dtype=float)
    res = _make_table_result(
        pred_html=pred_html,
        ocr_texts=["漏掉的字"],
        cell_box_list=cell_box_list,
        ocr_polys=[poly],
    )
    service = _FakeService([res])
    result = _recognize_table(service, image=None, options=TableRecognitionOptions())

    table_blocks = [b for b in result.content_list if b.get("type") == "table"]
    assert table_blocks, "应含表格块"
    table_body = table_blocks[0]["table_body"]
    assert "漏掉的字" in table_body, f"空单元格应被回填: {table_body!r}"
    assert "已填" in table_body
    assert "漏掉的字" not in table_blocks[0]["source"]["source_html"]
    # markdown 同步
    assert "漏掉的字" in result.markdown_text
    # 已被回填的 OCR 不再作为独立 text 块重复
    text_blocks = [b for b in result.content_list if b.get("type") == "text"]
    assert not any(b.get("text") == "漏掉的字" for b in text_blocks)


def test_recognize_table_consumes_each_ocr_item_at_most_once():
    import numpy as np

    empty_table = "<table><tr><td></td></tr></table>"
    cell_box = [[0.0, 0.0, 100.0, 50.0]]
    poly = np.array([[40, 10], [60, 10], [60, 30], [40, 30]], dtype=float)
    res = _DictResult(
        {
            "table_res_list": [
                {"pred_html": empty_table, "cell_box_list": cell_box},
                {"pred_html": empty_table, "cell_box_list": cell_box},
            ],
            "overall_ocr_res": {
                "rec_texts": ["ONLY_ONCE"],
                "rec_scores": [0.9],
                "rec_polys": [poly],
            },
        }
    )

    result = _recognize_table(
        _FakeService([res]), image=None, options=TableRecognitionOptions()
    )

    table_bodies = [block["table_body"] for block in result.content_list]
    assert sum("ONLY_ONCE" in body for body in table_bodies) == 1


def test_recognize_table_absorbs_unmatched_in_table_text():
    """表内 OCR 被吸收到对应单元格，不再产生独立文本框。"""
    import numpy as np

    # 两个单元格都已填，无空格可回填
    pred_html = "<table><tr><td>A</td><td>B</td></tr></table>"
    cell_box_list = [
        [10.0, 10.0, 100.0, 50.0],
        [100.0, 10.0, 200.0, 50.0],
    ]
    # OCR 文本中心 (55,30) 落在表内（已填单元格 A）
    inside_poly = np.array([[40, 20], [70, 20], [70, 40], [40, 40]], dtype=float)
    res = _make_table_result(
        pred_html=pred_html,
        ocr_texts=["表内未吸收"],
        cell_box_list=cell_box_list,
        ocr_polys=[inside_poly],
    )
    service = _FakeService([res])
    result = _recognize_table(service, image=None, options=TableRecognitionOptions())

    text_blocks = [b for b in result.content_list if b.get("type") == "text"]
    assert not any(b.get("text") == "表内未吸收" for b in text_blocks)
    assert "表内未吸收" in result.content_list[0]["table_body"]


def test_recognize_table_filters_text_inside_table_bbox():
    """表格外文字保留，表格内重复文字不重复展示（回填或保留，而非丢弃）。

    语义更新：原测试断言"表格内文字"被彻底丢弃（not any）。修复后表格内
    文本不再无条件丢弃——若对应单元格已填（无空格可回填），改为保留为独立
    text 块。本测试用默认 pred_html（两单元格均已填），"表格内文字"落点
    对应已填单元格，故应保留为独立块。
    """
    import numpy as np

    # 默认 pred_html: <table><tr><td>Name</td><td>Age</td></tr></table>（均已填）
    # 单元格覆盖 [10,10]-[500,200] 区域
    cell_box_list = [
        [10.0, 10.0, 250.0, 100.0],
        [250.0, 10.0, 500.0, 100.0],
        [10.0, 100.0, 250.0, 200.0],
        [250.0, 100.0, 500.0, 200.0],
    ]
    # 两个文本：一个在表格内（中心点 100,50），一个在表格外（中心点 300,300）
    inside_poly = np.array([[80, 40], [120, 40], [120, 60], [80, 60]], dtype=float)
    outside_poly = np.array(
        [[280, 290], [320, 290], [320, 310], [280, 310]], dtype=float
    )

    res = _make_table_result(
        ocr_texts=["表格内文字", "表格外文字"],
        cell_box_list=cell_box_list,
        ocr_polys=[inside_poly, outside_poly],
    )
    service = _FakeService([res])
    result = _recognize_table(service, image=None, options=TableRecognitionOptions())

    # 表格块
    assert any(b["type"] == "table" for b in result.content_list)
    # 表格外文字应保留
    assert any(b.get("text") == "表格外文字" for b in result.content_list)
    # HTML 只有一行两格、provider 却给出两行四框，逻辑映射不可证明；
    # 宁可保留独立文本，也不能把文字吸收到错误单元格。
    assert any(b.get("text") == "表格内文字" for b in result.content_list)
    table = next(b for b in result.content_list if b.get("type") == "table")
    assert "表格内文字" not in table["table_body"]
    warnings = table["table"]["provenance"]["warnings"]
    assert any(
        table["block_id"] in warning and "cell-mapping-unproven" in warning
        for warning in warnings
    )


def test_recognize_table_backfill_safe_without_cell_box_list():
    """无 cell_box_list 时回填安全降级，不崩，表格内容不丢。"""
    res = _make_table_result(
        pred_html="<table><tr><td></td><td>有</td></tr></table>",
        ocr_texts=["一些文字"],
        # 不传 cell_box_list，也不传 polys（中心无法计算）
    )
    service = _FakeService([res])
    result = _recognize_table(service, image=None, options=TableRecognitionOptions())
    # 不崩，表格块存在
    assert any(b["type"] == "table" for b in result.content_list)
    # OCR 文本作为独立块保留（无坐标 → 不在表内过滤逻辑里）
    assert "有" in result.raw_text


def test_recognize_table_assigns_bbox_from_cell_box_union():
    """表格块的 bbox 应从 cell_box_list 并集推导，而非 None。

    修复前 TextBlock.bbox 和 content_list[].bbox 均写死为 None，导致左侧
    画布无法绘制表格 bbox。修复后二者均应携带外接框（原图像素坐标）。
    """
    cell_box_list = [
        [10.0, 20.0, 250.0, 100.0],
        [250.0, 20.0, 500.0, 100.0],
        [10.0, 100.0, 250.0, 200.0],
        [250.0, 100.0, 500.0, 200.0],
    ]
    res = _make_table_result(cell_box_list=cell_box_list)
    service = _FakeService([res])
    result = _recognize_table(service, image=None, options=TableRecognitionOptions())

    # TextBlock 的 bbox 应为单元格并集 [10,20,500,200]
    table_block = result.text_blocks[0]
    assert table_block.label == "table"
    assert table_block.bbox is not None
    x0, y0, x1, y1 = table_block.bbox
    assert (x0, y0, x1, y1) == (10.0, 20.0, 500.0, 200.0)

    # content_list 的 table 项也应携带 bbox
    cl = result.content_list[0]
    assert cl["type"] == "table"
    assert cl["bbox"] is not None
    assert cl["bbox"] == [10.0, 20.0, 500.0, 200.0]


def test_recognize_table_bbox_none_when_no_cell_box_list():
    """cell_box_list 缺失时表格 bbox 应为 None，但仍正常生成表格块。

    验证修复不会因缺少 cell_box_list 而崩溃或丢失表格内容。
    """
    res = _make_table_result()  # 不传 cell_box_list
    service = _FakeService([res])
    result = _recognize_table(service, image=None, options=TableRecognitionOptions())

    assert len(result.text_blocks) == 1
    assert result.text_blocks[0].bbox is None
    assert result.content_list[0]["bbox"] is None
    assert result.text_blocks[0].text == "Name\tAge"
    assert result.content_list[0]["table_body"].startswith("<table")


def test_recognize_table_bbox_multiple_tables_no_misalignment():
    """多表格场景：每个表格 bbox 独立正确，不因列表错位而串框。

    回归保护：修复前 table_bboxes 只在 cell_box_list 有效时 append，
    与无条件创建的 text_blocks 会错位。修复后用局部变量即时赋值，
    即使某表格缺 cell_box_list 也不影响其他表格的 bbox。
    """
    # 两个表格：第一个有 cell_box_list，第二个没有
    res1 = _make_table_result(
        pred_html="<table><tr><td>A</td></tr></table>",
        cell_box_list=[[5.0, 5.0, 100.0, 50.0]],
    )
    res2 = _make_table_result(pred_html="<table><tr><td>B</td></tr></table>")
    service = _FakeService([res1, res2])
    result = _recognize_table(service, image=None, options=TableRecognitionOptions())

    table_blocks = [b for b in result.text_blocks if b.label == "table"]
    assert len(table_blocks) == 2
    # 第一个表格有 bbox
    assert table_blocks[0].bbox == (5.0, 5.0, 100.0, 50.0)
    # 第二个表格 bbox 为 None（无 cell_box_list），但内容正常
    assert table_blocks[1].bbox is None
    assert "B" in table_blocks[1].text


def test_recognize_table_bbox_ignores_invalid_provider_cell_boxes():
    result = _recognize_table(
        _FakeService(
            [
                _make_table_result(
                    pred_html="<table><tr><td>A</td></tr></table>",
                    cell_box_list=[
                        [5.0, 5.0, 100.0, 50.0],
                        [False, 0.0, 20.0, 20.0],
                        [float("nan"), 0.0, 20.0, 20.0],
                        [500.0, 500.0, 100.0, 100.0],
                    ],
                )
            ]
        ),
        image=None,
        options=TableRecognitionOptions(),
    )

    block = result.content_list[0]
    assert block["bbox"] == [5.0, 5.0, 100.0, 50.0]
    assert result.text_blocks[0].bbox == (5.0, 5.0, 100.0, 50.0)
    assert block["table"]["coordinate_space"] == "pixel"
    assert block["table"]["cells"][0]["bbox"] == [5.0, 5.0, 100.0, 50.0]


def test_recognize_table_emits_canonical_table_with_stable_ids():
    service = _FakeService(
        [
            _make_table_result(
                cell_box_list=[
                    [10.0, 20.0, 100.0, 80.0],
                    [100.0, 20.0, 200.0, 80.0],
                ]
            )
        ]
    )

    first = _recognize_table(
        service, image=None, options=TableRecognitionOptions()
    ).content_list[0]
    second = _recognize_table(
        service, image=None, options=TableRecognitionOptions()
    ).content_list[0]

    assert first["block_id"] == second["block_id"]
    assert first["table"]["table_id"] == first["block_id"]
    assert first["table"]["coordinate_space"] == "pixel"
    assert first["table"]["cells"][0]["bbox"] == [10.0, 20.0, 100.0, 80.0]
    assert first["table"]["cells"][0]["source_refs"] == [
        "paddlex:table_region:1:cell_box:0"
    ]
    assert first["table"]["provenance"]["provider_schema"] == "paddlex-table-res"
    first_result = _recognize_table(
        service, image=None, options=TableRecognitionOptions()
    )
    assert first_result.text_blocks[0].content_id == first["block_id"]
    assert first_result.text_blocks[0].text == "Name\tAge"
    assert first_result.text_with_scores[0] == ("Name\tAge", 0.9)


def test_recognize_table_html_projection_preserves_rowspan():
    pred_html = (
        "<table><tr><td rowspan='2'>A</td><td>B</td></tr><tr><td>C</td></tr></table>"
    )
    result = _recognize_table(
        _FakeService([_make_table_result(pred_html=pred_html)]),
        image=None,
        options=TableRecognitionOptions(),
    )

    assert 'rowspan="2"' in result.html_text


# ---- 纯函数覆盖：_point_in_box / _parse_cell_box / _normalize_match_text ----


class TestTablePureHelpers:
    def test_point_in_box_inside(self):
        from vibeocr.backend.core.pipelines.pipeline_table import _point_in_box

        assert _point_in_box(5, 5, (0, 0, 10, 10)) is True

    def test_point_in_box_on_boundary(self):
        from vibeocr.backend.core.pipelines.pipeline_table import _point_in_box

        assert _point_in_box(0, 0, (0, 0, 10, 10)) is True
        assert _point_in_box(10, 10, (0, 0, 10, 10)) is True

    def test_point_in_box_outside(self):
        from vibeocr.backend.core.pipelines.pipeline_table import _point_in_box

        assert _point_in_box(11, 5, (0, 0, 10, 10)) is False
        assert _point_in_box(5, -1, (0, 0, 10, 10)) is False

    def test_parse_cell_box_valid(self):
        from vibeocr.backend.core.pipelines.pipeline_table import _parse_cell_box

        assert _parse_cell_box([1, 2, 3, 4]) == (1.0, 2.0, 3.0, 4.0)

    def test_parse_cell_box_degenerate_returns_none(self):
        """退化框（x0>=x2 或 y0>=y3）返回 None。"""
        from vibeocr.backend.core.pipelines.pipeline_table import _parse_cell_box

        assert _parse_cell_box([5, 2, 3, 4]) is None  # x0 > x2
        assert _parse_cell_box([1, 5, 3, 4]) is None  # y0 > y3

    def test_parse_cell_box_non_finite_returns_none(self):
        from vibeocr.backend.core.pipelines.pipeline_table import _parse_cell_box

        assert _parse_cell_box([1, 2, float("inf"), 4]) is None
        assert _parse_cell_box([1, 2, 3, float("nan")]) is None

    def test_parse_cell_box_invalid_type_returns_none(self):
        from vibeocr.backend.core.pipelines.pipeline_table import _parse_cell_box

        assert _parse_cell_box(None) is None
        assert _parse_cell_box("not-a-box") is None
        assert _parse_cell_box([1, 2]) is None  # 太短
        assert _parse_cell_box([True, 2, 3, 4]) is None  # bool 不算 Real

    def test_parse_cell_box_with_tolist(self):
        from vibeocr.backend.core.pipelines.pipeline_table import _parse_cell_box

        class _FakeArray:
            def tolist(self):
                return [1, 2, 3, 4]

        assert _parse_cell_box(_FakeArray()) == (1.0, 2.0, 3.0, 4.0)

    def test_normalize_match_text(self):
        from vibeocr.backend.core.pipelines.pipeline_table import _normalize_match_text

        assert _normalize_match_text("Hello World") == "helloworld"
        assert _normalize_match_text("  a\tb\n") == "ab"
        assert _normalize_match_text("") == ""
