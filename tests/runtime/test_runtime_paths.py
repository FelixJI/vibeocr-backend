from __future__ import annotations

import sys
from pathlib import Path

from vibeocr.backend import env_manager, runtime_layout
from vibeocr.backend.services import env_config


def test_app_paths_resolve_backend_repository_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "backend"
    module_path = (
        repository
        / "packages"
        / "vibeocr-backend"
        / "src"
        / "vibeocr"
        / "backend"
        / "runtime_layout.py"
    )
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# source marker\n", encoding="utf-8")
    (repository / "pyproject.toml").write_text(
        "[dependency-groups]\n", encoding="utf-8"
    )
    (repository / "resources").mkdir()
    changelog = repository / "CHANGELOG.md"
    changelog.write_text("# Changes\n", encoding="utf-8")

    monkeypatch.setattr(runtime_layout, "__file__", str(module_path))
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "environment"))

    paths = runtime_layout.resolve_app_paths()

    assert paths.install_root == repository.resolve()
    assert paths.resources_root == repository.resolve() / "resources"
    assert paths.changelog_path == changelog.resolve()


def test_legacy_root_interfaces_share_runtime_layout_result() -> None:
    install_root = runtime_layout.resolve_app_paths().install_root

    assert env_config.get_project_root() == install_root
    assert env_manager.get_project_root() == install_root


def test_app_paths_anchor_regular_wheel_at_python_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    environment = tmp_path / "environment"
    module_path = environment / "Lib" / "site-packages" / "vibeocr" / "backend.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# installed wheel\n", encoding="utf-8")
    changelog = environment / "CHANGELOG.md"
    changelog.write_text("# Installed changes\n", encoding="utf-8")

    monkeypatch.setattr(runtime_layout, "__file__", str(module_path))
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.setattr(sys, "prefix", str(environment))

    paths = runtime_layout.resolve_app_paths()

    assert paths.install_root == environment.resolve()
    assert paths.resources_root == environment.resolve() / "resources"
    assert paths.changelog_path == changelog.resolve()


def test_app_paths_use_frozen_executable_and_meipass(
    monkeypatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "product" / "vibeocr.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"")
    meipass = executable.parent / "_internal"
    meipass.mkdir()
    bundled_changelog = meipass / "CHANGELOG.md"
    bundled_changelog.write_text("# Bundled changes\n", encoding="utf-8")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)

    paths = runtime_layout.resolve_app_paths()

    assert paths.install_root == executable.parent.resolve()
    assert paths.resources_root == meipass.resolve() / "resources"
    assert paths.changelog_path == bundled_changelog.resolve()
