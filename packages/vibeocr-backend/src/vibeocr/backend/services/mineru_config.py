"""Resolve only proven MinerU 3 equivalents; keep native MinerU 4 choices."""

from __future__ import annotations

from collections.abc import Mapping

from vibeocr.runtime_contracts import (
    ErrorCode,
    MineruConfig,
    MineruOcrMode,
    MineruTier,
    PipelineSelection,
)

LANGUAGES = (
    "ch",
    "ch_server",
    "korean",
    "ta",
    "te",
    "ka",
    "th",
    "el",
    "arabic",
    "east_slavic",
    "cyrillic",
    "devanagari",
)
_LEGACY_KEYS = frozenset(
    {
        "backend",
        "effort",
        "parse_method",
        "enable_formula",
        "enable_table",
        "lang_list",
        "start_page_id",
        "end_page_id",
    }
)


class MineruConfigError(ValueError):
    def __init__(self, code: ErrorCode, reason: str) -> None:
        self.code = code
        self.reason = reason
        super().__init__(reason)


def resolve_mineru_config(selection: PipelineSelection) -> MineruConfig:
    if selection.mineru is not None:
        if selection.options or selection.engine is not None:
            raise MineruConfigError(
                ErrorCode.VALIDATION_ERROR, "mixed_mineru_configuration"
            )
        result = selection.mineru
    else:
        result = migrate_legacy_options(selection.options)
    if result.language not in LANGUAGES:
        raise MineruConfigError(
            ErrorCode.MINERU_CONFIG_UNAVAILABLE, "unsupported_language"
        )
    return result


def migrate_legacy_options(options: Mapping[str, object]) -> MineruConfig:
    def reselect() -> None:
        raise MineruConfigError(
            ErrorCode.MINERU_CONFIG_MIGRATION_REQUIRED, "legacy_options_not_equivalent"
        )

    if options.keys() - _LEGACY_KEYS:
        reselect()
    if options.get("backend", "hybrid-engine") != "hybrid-engine":
        reselect()
    effort = options.get("effort", "medium")
    if effort not in ("medium", "high"):
        reselect()
    if (
        options.get("enable_formula", True) is not True
        or options.get("enable_table", True) is not True
    ):
        reselect()
    languages = options.get("lang_list", [])
    if not isinstance(languages, list) or len(languages) > 1:
        reselect()
    language = languages[0] if languages else "ch"
    if language not in LANGUAGES:
        reselect()
    start = options.get("start_page_id", 0)
    end = options.get("end_page_id")
    if type(start) is not int or start < 0:
        reselect()
    if end is not None and (type(end) is not int or end < start):
        reselect()
    page_range = (
        "all"
        if start == 0 and end is None
        else f"{start + 1}-{end + 1 if end is not None else 'r1'}"
    )
    try:
        mode = MineruOcrMode(options.get("parse_method", "auto"))
    except ValueError:
        reselect()
    return MineruConfig(
        tier=MineruTier.BASIC if effort == "medium" else MineruTier.STANDARD,
        ocr_mode=mode,
        page_range=page_range,
        language=language,
    )
