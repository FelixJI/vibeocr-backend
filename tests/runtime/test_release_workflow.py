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


def test_publish_checkout_keeps_job_token_for_git_tag_push() -> None:
    workflow = (ROOT / ".github/workflows/cd.yml").read_text(encoding="utf-8")
    publish_job = workflow.split("\n  publish:\n", maxsplit=1)[1]
    permissions = publish_job.split("\n    permissions:\n", maxsplit=1)[1].split(
        "\n    steps:\n", maxsplit=1
    )[0]
    checkout = publish_job.split("- uses: actions/checkout@", maxsplit=1)[1].split(
        "- uses: actions/setup-python@", maxsplit=1
    )[0]

    assert {line.strip() for line in permissions.splitlines() if line.strip()} == {
        "actions: read",
        "attestations: write",
        "contents: write",
        "id-token: write",
    }
    assert "persist-credentials: true" in checkout
    assert "persist-credentials: false" not in checkout
    assert "token:" not in checkout


def test_release_build_hashes_runtime_without_shell_module_autoloading() -> None:
    script = (ROOT / "scripts/build-release.ps1").read_text(encoding="utf-8")

    assert "System.Security.Cryptography.SHA256" in script
    assert "ComputeHash" in script
    assert "Get-FileHash" not in script


def test_release_build_fails_immediately_when_build_tool_installation_fails() -> None:
    script = (ROOT / "scripts/build-release.ps1").read_text(encoding="utf-8")

    install = script.index("python -m pip install build==")
    failure = script.index("Release build dependency installation failed")
    build = script.index("python -m build --wheel")

    assert install < failure < build
