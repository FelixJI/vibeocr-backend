# tests/core/test_pipeline_registry.py
import pytest
from vibeocr.backend.core.pipelines.base_options import BasePipelineOptions
from vibeocr.backend.core.pipelines.registry import PipelineRegistry, PipelineSpec


class DummyOptions(BasePipelineOptions):
    pipeline: str = "DUMMY"
    foo: bool = True


def _create_dummy(device):
    return "dummy_pipeline"


def _recognize_dummy(service, image, options):
    return None


DUMMY_SPEC = PipelineSpec(
    name="DUMMY",
    display_name="Dummy Pipeline",
    description="For testing",
    options_class=DummyOptions,
    create_pipeline=_create_dummy,
    recognize=_recognize_dummy,
)


def test_spec_fields():
    assert DUMMY_SPEC.name == "DUMMY"
    assert DUMMY_SPEC.options_class is DummyOptions


def test_register_and_get():
    reg = PipelineRegistry()
    reg.register(DUMMY_SPEC)
    spec = reg.get("DUMMY")
    assert spec is DUMMY_SPEC


def test_get_unknown_raises():
    reg = PipelineRegistry()
    with pytest.raises(KeyError):
        reg.get("UNKNOWN")


def test_list_all():
    reg = PipelineRegistry()
    reg.register(DUMMY_SPEC)
    all_specs = reg.list_all()
    assert len(all_specs) == 1
    assert all_specs[0].name == "DUMMY"


def test_list_display_names():
    reg = PipelineRegistry()
    reg.register(DUMMY_SPEC)
    names = reg.list_display_names()
    assert names == ["Dummy Pipeline"]
