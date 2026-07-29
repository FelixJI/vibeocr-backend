"""表格语义门禁配置的静态回归测试。"""

from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]


def test_main_table_regression_covers_cross_layer_surfaces_and_has_ratchet():
    workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    for path in (
        "tests/services/test_export_service_extra.py",
        "tests/supervisor/test_export_route_tables.py",
        "tests/utils/test_table_model_reducer.py",
        "tests/views/tabs/test_base_tab.py",
        "tests/widgets/test_result_view_widget.py",
    ):
        assert path in workflow
    assert "--cov=vibeocr.backend.tables" in workflow
    assert "--cov-branch" in workflow
    assert "--cov-fail-under=85" in workflow


def test_release_runs_table_artifact_gate_after_builds():
    workflow = (REPO_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    # The release workflow builds both installer variants, then runs a single
    # table-semantics artifact gate. The gate must come after the builds so it
    # inspects finished artifacts, and it covers whichever variant was built.
    pyside_build = workflow.index("Build PySide6 Classic (PyInstaller)")
    winui_build = workflow.index("Build WinUI Next (.NET publish)")
    artifact_gate = workflow.index("Verify table semantics in release artifacts")
    assert pyside_build < artifact_gate
    assert winui_build < artifact_gate
    assert "scripts/verify_table_artifact.py" in workflow
    assert "reports/table-release-contract" in workflow


def test_release_table_gate_runs_per_selected_variant():
    workflow = (REPO_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    # The gate is conditional on the resolved build variants, and verifies each
    # built artifact (Classic zip / WinUI zip) against the table contract.
    assert "steps.variants.outputs.build_pyside" in workflow
    assert "steps.variants.outputs.build_winui" in workflow
    assert "VibeOCR-Classic-" in workflow
    assert "VibeOCR-Next-" in workflow
    assert "reports/table-artifact/classic" in workflow
    assert "reports/table-artifact/winui" in workflow
