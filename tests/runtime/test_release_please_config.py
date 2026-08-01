import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EXPECTED_CHANGELOG_SECTIONS = [
    {"type": "feat", "section": "Features"},
    {"type": "fix", "section": "Bug Fixes"},
    {"type": "perf", "section": "Performance Improvements"},
    {"type": "security", "section": "Security"},
    {"type": "deps", "section": "Dependencies"},
    {"type": "build", "section": "Build and Packaging"},
    {"type": "revert", "section": "Reverts"},
    {"type": "docs", "section": "Documentation", "hidden": True},
    {"type": "refactor", "section": "Code Refactoring", "hidden": True},
    {"type": "test", "section": "Tests", "hidden": True},
    {"type": "ci", "section": "Continuous Integration", "hidden": True},
    {"type": "style", "section": "Styles", "hidden": True},
    {"type": "chore", "section": "Miscellaneous Chores", "hidden": True},
]


def test_release_please_uses_the_shared_changelog_filter() -> None:
    config = json.loads(
        (ROOT / "release-please-config.json").read_text(encoding="utf-8")
    )

    assert config["changelog-sections"] == EXPECTED_CHANGELOG_SECTIONS
