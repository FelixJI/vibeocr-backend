import email
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]


def test_backend_declares_multipart_form_parser_dependency() -> None:
    metadata = tomllib.loads(
        (ROOT / "packages" / "vibeocr-backend" / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )

    dependencies = metadata["project"]["dependencies"]
    assert any(item.startswith("python-multipart") for item in dependencies)


def test_backend_declares_protocol_same_major_package_range() -> None:
    metadata = tomllib.loads(
        (ROOT / "packages" / "vibeocr-backend" / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )

    assert (
        "vibeocr-runtime-contracts>=2.8.0,<3.0.0" in metadata["project"]["dependencies"]
    )


def test_backend_wheel_metadata_preserves_protocol_package_range(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--python",
            sys.executable,
            str(ROOT / "packages" / "vibeocr-backend"),
            "--out-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")
    wheel = next(tmp_path.glob("vibeocr_backend-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = email.message_from_bytes(archive.read(metadata_name))
    protocol = [
        Requirement(item)
        for item in metadata.get_all("Requires-Dist", [])
        if Requirement(item).name == "vibeocr-runtime-contracts"
    ]

    assert len(protocol) == 1
    assert str(protocol[0].specifier) == "<3.0.0,>=2.8.0"
