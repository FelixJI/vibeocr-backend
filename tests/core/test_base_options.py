from vibeocr.backend.core.pipelines.base_options import BasePipelineOptions


def test_base_options_has_pipeline_name():
    opts = BasePipelineOptions()
    assert opts.pipeline == ""


def test_base_options_to_dict_contains_pipeline():
    opts = BasePipelineOptions()
    d = opts.to_dict()
    assert d["pipeline"] == ""


def test_base_options_from_dict():
    d = {"pipeline": "OCR"}
    opts = BasePipelineOptions.from_dict(d)
    assert opts.pipeline == "OCR"


def test_base_options_copy():
    opts = BasePipelineOptions(pipeline="OCR")
    copied = opts.copy(pipeline="PP-StructureV3")
    assert copied.pipeline == "PP-StructureV3"
    assert opts.pipeline == "OCR"
