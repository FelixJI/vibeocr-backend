# tests/models/test_text_block_options.py
"""TextBlockOptions 数据结构测试"""

from vibeocr.backend.models.text_block_options import (
    LINE_MODE_KEEP,
    LINE_MODE_MERGE,
    LINE_MODE_SMART,
    TextBlockOptions,
)


class TestTextBlockOptionsRoundTrip:
    def test_defaults(self):
        opts = TextBlockOptions()
        assert opts.line_mode == LINE_MODE_MERGE
        assert opts.block_join_space is False
        assert opts.chinese_indent is False
        assert opts.drop_blank_blocks is True

    def test_round_trip_merge(self):
        opts = TextBlockOptions(
            line_mode=LINE_MODE_MERGE,
            block_join_space=True,
            chinese_indent=True,
            drop_blank_blocks=False,
        )
        restored = TextBlockOptions.from_dict(opts.to_dict())
        assert restored == opts

    def test_round_trip_smart(self):
        opts = TextBlockOptions(line_mode=LINE_MODE_SMART, block_join_space=True)
        restored = TextBlockOptions.from_dict(opts.to_dict())
        assert restored == opts

    def test_round_trip_keep(self):
        opts = TextBlockOptions(line_mode=LINE_MODE_KEEP, drop_blank_blocks=False)
        restored = TextBlockOptions.from_dict(opts.to_dict())
        assert restored == opts


class TestFromDictDefaults:
    def test_empty_dict_returns_defaults(self):
        opts = TextBlockOptions.from_dict({})
        assert opts == TextBlockOptions()

    def test_none_returns_defaults(self):
        opts = TextBlockOptions.from_dict(None)
        assert opts == TextBlockOptions()

    def test_partial_fields_use_defaults(self):
        opts = TextBlockOptions.from_dict({"line_mode": LINE_MODE_KEEP})
        assert opts.line_mode == LINE_MODE_KEEP
        assert opts.block_join_space is False
        assert opts.chinese_indent is False
        assert opts.drop_blank_blocks is True

    def test_unknown_fields_ignored(self):
        opts = TextBlockOptions.from_dict(
            {"line_mode": LINE_MODE_SMART, "unknown_key": 123, "extra": "x"}
        )
        assert opts.line_mode == LINE_MODE_SMART
        assert opts.drop_blank_blocks is True

    def test_invalid_line_mode_falls_back_to_merge(self):
        opts = TextBlockOptions.from_dict({"line_mode": "bogus"})
        assert opts.line_mode == LINE_MODE_MERGE

    def test_non_bool_coerced_to_bool(self):
        opts = TextBlockOptions.from_dict(
            {"block_join_space": 0, "chinese_indent": 1}
        )
        assert opts.block_join_space is False
        assert opts.chinese_indent is True
