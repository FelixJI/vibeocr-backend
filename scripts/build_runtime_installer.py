"""Build and deterministically package the standalone Runtime Installer EXE."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def package_runtime_installer(
    executable: Path,
    output_dir: Path,
    *,
    backend_version: str,
) -> Path:
    executable = executable.resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"vibeocr-runtime-installer-{backend_version}.zip"
    info = zipfile.ZipInfo(
        "runtime-installer/vibeocr-runtime-installer.exe",
        date_time=FIXED_ZIP_TIME,
    )
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o755 << 16
    with zipfile.ZipFile(target, mode="w", compresslevel=9) as archive:
        archive.writestr(info, executable.read_bytes())
    return target


def build_runtime_installer(
    *,
    output_dir: Path,
    work_dir: Path,
    backend_version: str,
) -> tuple[Path, Path]:
    dist = work_dir / "dist"
    build = work_dir / "build"
    spec = work_dir / "spec"
    for directory in (dist, build, spec):
        directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--onefile",
            "--name",
            "vibeocr-runtime-installer",
            "--paths",
            str(ROOT / "packages" / "vibeocr-backend" / "src"),
            "--distpath",
            str(dist),
            "--workpath",
            str(build),
            "--specpath",
            str(spec),
            str(ROOT / "scripts" / "runtime_installer_entry.py"),
        ],
        cwd=ROOT,
        check=True,
    )
    executable = dist / "vibeocr-runtime-installer.exe"
    archive = package_runtime_installer(
        executable,
        output_dir,
        backend_version=backend_version,
    )
    return executable, archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--backend-version", default="0.7.0")
    args = parser.parse_args(argv)
    executable, archive = build_runtime_installer(
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        backend_version=args.backend_version,
    )
    for path in (executable, archive):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{digest}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
