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


def test_release_build_hashes_runtime_without_shell_module_autoloading() -> None:
    script = (ROOT / "scripts/build-release.ps1").read_text(encoding="utf-8")

    assert "System.Security.Cryptography.SHA256" in script
    assert "ComputeHash" in script
    assert "Get-FileHash" not in script
