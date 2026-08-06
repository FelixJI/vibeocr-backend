import os
import tempfile
from unittest.mock import Mock

import pytest
from vibeocr.backend.utils.cjk_font_resolver import CjkFontResolver


def test_resolve_empty_characters_without_creating_subset(monkeypatch) -> None:
    resolver = CjkFontResolver()
    subset = Mock()
    monkeypatch.setattr(resolver, "_subset", subset)

    assert resolver.resolve("") is None
    subset.assert_not_called()


def test_resolve_returns_none_without_system_font(monkeypatch) -> None:
    resolver = CjkFontResolver()
    monkeypatch.setattr(resolver, "_find_system_font", lambda: None)

    assert resolver.resolve("中文") is None


def test_resolve_returns_none_when_subsetting_fails(monkeypatch) -> None:
    resolver = CjkFontResolver()
    monkeypatch.setattr(resolver, "_find_system_font", lambda: "fake.ttf")

    def fail_subsetting(_font_path: str, _chars: str) -> str:
        raise RuntimeError("subset failed")

    monkeypatch.setattr(resolver, "_subset", fail_subsetting)

    assert resolver.resolve("中文") is None


def test_resolve_reuses_subset_for_same_character_set(monkeypatch) -> None:
    resolver = CjkFontResolver()
    subset = Mock(return_value="subset.ttf")
    monkeypatch.setattr(resolver, "_find_system_font", lambda: "fake.ttf")
    monkeypatch.setattr(resolver, "_subset", subset)

    assert resolver.resolve("中文") == "subset.ttf"
    assert resolver.resolve("文中") == "subset.ttf"
    subset.assert_called_once_with("fake.ttf", "中文")


def test_resolve_probes_system_font_once(monkeypatch, tmp_path) -> None:
    resolver = CjkFontResolver()
    missing = tmp_path / "missing.ttf"
    available = tmp_path / "available.ttf"
    available.write_bytes(b"font")
    candidates = Mock(return_value=[str(missing), str(available)])
    subset = Mock(side_effect=["subset-1.ttf", "subset-2.ttf"])
    monkeypatch.setattr(resolver, "_get_candidates", candidates)
    monkeypatch.setattr(resolver, "_subset", subset)

    assert resolver.resolve("中") == "subset-1.ttf"
    available.unlink()
    assert resolver.resolve("文") == "subset-2.ttf"

    candidates.assert_called_once_with()
    assert [call.args[0] for call in subset.call_args_list] == [
        str(available),
        str(available),
    ]


def test_cleanup_removes_resolved_subset_files_and_is_idempotent(
    monkeypatch, tmp_path
) -> None:
    first = tmp_path / "first.ttf"
    second = tmp_path / "second.ttf"
    first.write_bytes(b"font")
    second.write_bytes(b"font")
    resolver = CjkFontResolver()
    monkeypatch.setattr(resolver, "_find_system_font", lambda: "source.ttf")
    monkeypatch.setattr(
        resolver,
        "_subset",
        Mock(side_effect=[str(first), str(second)]),
    )
    assert resolver.resolve("中") == str(first)
    assert resolver.resolve("文") == str(second)

    resolver.cleanup()
    resolver.cleanup()

    assert not first.exists()
    assert not second.exists()


def test_subset_save_failure_removes_temporary_file(monkeypatch, tmp_path) -> None:
    from fontTools import subset, ttLib

    temporary = tmp_path / "failed-subset.ttf"
    descriptor = os.open(temporary, os.O_CREAT | os.O_RDWR)
    font = Mock()
    font.save.side_effect = OSError("save failed")
    monkeypatch.setattr(ttLib, "TTFont", Mock(return_value=font))
    monkeypatch.setattr(subset, "Subsetter", Mock(return_value=Mock()))
    monkeypatch.setattr(
        tempfile,
        "mkstemp",
        lambda **_kwargs: (descriptor, str(temporary)),
    )

    with pytest.raises(OSError, match="save failed"):
        CjkFontResolver._subset("source.ttf", "中文")

    assert not temporary.exists()
