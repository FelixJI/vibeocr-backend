"""拆仓后 Backend CI 与 Runtime 发布门禁的静态回归测试。"""

from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]


def test_ci_runs_the_unified_full_quality_gate() -> None:
    workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    script = (REPO_ROOT / "scripts/check-quality.ps1").read_text(encoding="utf-8")

    assert "python -m pip install --group dev" in workflow
    assert "./scripts/check-quality.ps1" in workflow
    assert "python -m ruff check packages/vibeocr-backend scripts tests" in script
    assert (
        "python -m ruff format --check packages/vibeocr-backend scripts tests" in script
    )
    assert "python -m pytest" in script
    assert "tests/runtime tests/application tests/core tests/models" not in script


def test_release_verifies_runtime_candidate_after_build() -> None:
    workflow = (REPO_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    build = workflow.index("- name: Build release assets")
    verify = workflow.index("- name: Verify release candidate assets")
    upload = workflow.index("- name: Attach assets to draft Release")

    assert build < verify < upload
    assert "runtime-manifest.json" in workflow
    assert "vibeocr-runtime-installer-*.zip" in workflow
    assert "requirements-win-x64-cpu.lock" in workflow
