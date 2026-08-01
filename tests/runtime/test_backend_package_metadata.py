import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_backend_declares_multipart_form_parser_dependency() -> None:
    metadata = tomllib.loads(
        (ROOT / "packages" / "vibeocr-backend" / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )

    dependencies = metadata["project"]["dependencies"]
    assert any(item.startswith("python-multipart") for item in dependencies)
