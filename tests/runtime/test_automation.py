from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.automation_core import Automation, AutomationError, SemVer

ROOT = Path(__file__).resolve().parents[2]


def test_backend_config_uses_canonical_lanes_and_identity_asset() -> None:
    config = json.loads((ROOT / ".ci/project.json").read_text(encoding="utf-8"))

    assert list(config["ci"]) == [
        "bootstrap",
        "quality",
        "e2e",
        "release_build",
        "release_smoke",
    ]
    assert config["project"]["protocol_compatibility"] == {
        "supported_majors": [2],
        "minor_compatible": True,
    }
    assert config["release"]["identity_asset"] == "build-identity.json"
    assert "release-manifest.json" not in config["release"]["required_assets"]
    assert "SHA256SUMS" not in config["release"]["required_assets"]


def test_backend_versions_are_consistent_under_canonical_adapter() -> None:
    automation = Automation.for_repository()
    versions = automation.current_versions()

    assert automation.component == "backend"
    assert len(versions) == 4
    assert len(set(versions)) == 1
    assert automation.current_version() == versions[0]


def test_semver_project_parser_ignores_prerelease_for_manual_bump() -> None:
    assert SemVer.parse_project("0.7.2-preview.4").bump("minor") == SemVer(0, 8, 0)


def test_invalid_ci_source_sha_fails_closed() -> None:
    automation = Automation.for_repository()
    with pytest.raises(AutomationError, match="full lowercase Git SHA"):
        automation.ci(event="pull_request", source_sha="bad")
