from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_release_workflow_publishes_only_after_verified_uploads() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    build = workflow.index("- name: Build release assets")
    verify_local = workflow.index("- name: Verify release candidate assets")
    upload_artifact = workflow.index("uses: actions/upload-artifact@v4")
    upload_release = workflow.index("- name: Attach assets to draft Release")
    verify_remote = workflow.index("- name: Verify uploaded Release assets")
    publish = workflow.index("- name: Publish verified Release")

    assert (
        build
        < verify_local
        < upload_artifact
        < upload_release
        < verify_remote
        < publish
    )
    assert "github.event.workflow_run.conclusion == 'success'" in workflow
    assert "github.event_name == 'workflow_run'" in workflow
    assert "gh release edit $env:RELEASE_TAG --draft=false" in workflow
    assert "already has assets; skipping build" not in workflow
    assert "runtime-manifest.json" in workflow
    assert "requirements-win-x64-cpu.lock" in workflow
    assert "vibeocr-runtime-installer-*.zip" in workflow


def test_cleanup_uses_rest_release_ids_and_preserves_tags() -> None:
    workflow = (ROOT / ".github/workflows/cleanup-releases.yml").read_text(
        encoding="utf-8"
    )

    assert (
        'gh api --paginate "repos/${GITHUB_REPOSITORY}/releases?per_page=100"'
        in workflow
    )
    assert ".id, .tag_name" in workflow
    assert "databaseId" not in workflow
    assert "releases/${release_id}" in workflow
    assert "git push --delete" not in workflow
