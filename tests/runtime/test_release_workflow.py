from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_workflows_are_thin_automation_cli_callers() -> None:
    workflows = sorted((ROOT / ".github/workflows").glob("*.yml"))

    assert [path.name for path in workflows] == ["cd.yml", "ci.yml"]
    for workflow in workflows:
        text = workflow.read_text(encoding="utf-8")
        assert "python scripts/automation.py" in text
        assert "release-please" not in text
