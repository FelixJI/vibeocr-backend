"""Repository-level import direction contracts."""

from __future__ import annotations

import ast
from pathlib import Path

BACKEND_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "vibeocr-backend"
    / "src"
    / "vibeocr"
    / "backend"
)
FORBIDDEN_CORE_PREFIXES = (
    "vibeocr.backend.application",
    "vibeocr.backend.services",
    "vibeocr.backend.supervisor",
)


def _forbidden_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        imports.extend(
            (node.lineno, module)
            for module in modules
            if module.startswith(FORBIDDEN_CORE_PREFIXES)
        )
    return imports


def test_core_does_not_import_upper_layers() -> None:
    violations = [
        f"{path.relative_to(BACKEND_SOURCE)}:{line} imports {module}"
        for path in sorted((BACKEND_SOURCE / "core").rglob("*.py"))
        for line, module in _forbidden_imports(path)
    ]

    assert violations == []


def test_environment_config_does_not_import_environment_manager() -> None:
    config_path = BACKEND_SOURCE / "services" / "env_config.py"
    tree = ast.parse(
        config_path.read_text(encoding="utf-8"),
        filename=str(config_path),
    )
    imports = [
        (node.lineno, node.module)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "vibeocr.backend.env_manager"
    ]

    assert imports == []
