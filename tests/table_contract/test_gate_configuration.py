"""Backend 项目适配器与统一 CI/CD 深接口的静态回归测试。"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]


def test_ci_runs_the_unified_full_quality_gate() -> None:
    workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    config = json.loads((REPO_ROOT / ".ci/project.json").read_text(encoding="utf-8"))
    bootstrap = (REPO_ROOT / "scripts/bootstrap-ci.ps1").read_text(encoding="utf-8")
    script = (REPO_ROOT / "scripts/check-quality.ps1").read_text(encoding="utf-8")

    assert workflow.count("python scripts/automation.py") == 1
    assert "python -m pip install" not in workflow
    assert config["ci"]["bootstrap"] == [
        [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            "scripts/bootstrap-ci.ps1",
        ]
    ]
    assert "python -m pip install --group dev" in bootstrap
    assert config["ci"]["quality"][0][-1] == "scripts/check-quality.ps1"
    assert "python -m ruff check packages/vibeocr-backend scripts tests" in script
    assert (
        "python -m ruff format --check packages/vibeocr-backend scripts tests" in script
    )
    assert "python -m pytest" in script
    assert "tests/runtime tests/application tests/core tests/models" not in script


def test_release_verifies_runtime_candidate_after_build() -> None:
    workflow = (REPO_ROOT / ".github/workflows/cd.yml").read_text(encoding="utf-8")
    config = json.loads((REPO_ROOT / ".ci/project.json").read_text(encoding="utf-8"))
    build_script = (REPO_ROOT / "scripts/build-release.ps1").read_text(encoding="utf-8")

    download = workflow.index("name: Download exact CI candidate")
    stage = workflow.index("name: Stage release")
    publish = workflow.index("name: Publish and reconcile release")
    assert download < stage < publish
    assert config["ci"]["release_build"][0][-4:] == [
        "-Version",
        "{version}",
        "-ArtifactsDir",
        "{artifacts_dir}",
    ]
    assert config["release"]["identity_asset"] == "build-identity.json"
    assert "runtime-manifest.json" in config["release"]["required_assets"]
    assert "vibeocr-runtime-installer-*.zip" in config["release"]["required_assets"]
    assert "requirements-win-x64-cpu.lock" in build_script
