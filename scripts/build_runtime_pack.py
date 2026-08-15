"""Build a manifest-bound offline runtime pack from an exact hash lock.

Two-phase flow (plan §4.2):

1. ``pip download --require-hashes`` pulls the closure — wheels where the
   index has them, hash-verified sdists where it does not (e.g.
   ``antlr4-python3-runtime==4.9.3`` via omegaconf ships an sdist only).
   Every downloaded artifact is re-verified byte-for-byte against the lock.
2. ``pip wheel --no-index --find-links <downloads>`` turns the verified
   closure into a pure wheel directory offline — direct wheels are reused,
   sdists are built with the running environment's setuptools
   (``--no-build-isolation``; callers must provide setuptools, e.g. the
   release workflow installs it alongside ``build``).

The final pack is a flat, deterministic zip of every wheel. Release
automation binds it into ``runtime-manifest.json`` via
``build_runtime_manifest.py --base-runtime-pack``; the installer then
installs with ``--no-index --find-links`` so a blocked network can never
degrade an offline install.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _source_root in (
    REPO_ROOT / "packages" / "vibeocr-backend" / "src",
    REPO_ROOT / "src",
):
    if _source_root.is_dir():
        sys.path.insert(0, str(_source_root))
        break

from vibeocr.backend.runtime_manifest import (  # noqa: E402
    ManifestError,
    validate_requirements_lock,
)

# Deterministic zip metadata: identical inputs produce identical bytes.
_FIXED_ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
_HASH_ENTRY_RE = re.compile(r"--hash=sha256:([0-9a-f]{64})")
_DECLARATION_RE = re.compile(r"(?m)^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)")
_WHEEL_FILENAME_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)-(?P<version>\d[^-]*)-(?P<rest>.+)\.whl$"
)


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _lock_hashes(path: Path) -> set[str]:
    return set(_HASH_ENTRY_RE.findall(path.read_text(encoding="utf-8")))


def _lock_requirements(path: Path) -> set[tuple[str, str]]:
    return {
        (_normalize(match.group(1)), match.group(2))
        for match in _DECLARATION_RE.finditer(path.read_text(encoding="utf-8"))
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> None:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({' '.join(command[1:4])}...):\n"
            f"{result.stdout}\n{result.stderr}"
        )


def build_runtime_pack(
    *,
    lock: Path,
    profile: str,
    work_dir: Path,
    output: Path,
) -> Path:
    lock = lock.resolve(strict=True)
    validate_requirements_lock(lock, profile=profile)
    if output.suffix != ".zip":
        raise ValueError("runtime pack output must be a .zip archive")

    downloads = work_dir / "downloads"
    wheels = work_dir / "wheels"
    for directory in (downloads, wheels):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)

    # Phase 1: hash-verified closure download (wheels preferred, sdists kept).
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--require-hashes",
            "-d",
            str(downloads),
            "-r",
            str(lock),
        ]
    )
    known_hashes = _lock_hashes(lock)
    artifacts = sorted(downloads.iterdir())
    if not artifacts:
        raise RuntimeError("runtime pack closure is empty")
    for artifact in artifacts:
        if artifact.suffix not in (".whl", ".gz", ".zip"):
            raise RuntimeError(
                f"runtime pack download is neither wheel nor sdist: {artifact.name}"
            )
        if _sha256_file(artifact) not in known_hashes:
            raise RuntimeError(f"runtime pack artifact not bound by lock: {artifact}")

    # Phase 2: offline wheelification of the verified closure. Build
    # isolation is disabled on purpose: --no-index forbids the isolated
    # build env from fetching setuptools, so the running env provides it.
    requirement_set = _lock_requirements(lock)
    if not requirement_set:
        raise RuntimeError("runtime lock declares no requirements")
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-index",
            "--no-deps",
            "--no-build-isolation",
            "--find-links",
            str(downloads),
            "-w",
            str(wheels),
            *(f"{name}=={version}" for name, version in sorted(requirement_set)),
        ]
    )

    wheel_files = sorted(wheels.glob("*.whl"))
    if not wheel_files:
        raise RuntimeError("runtime pack produced no wheels")
    packed: set[tuple[str, str]] = set()
    for wheel in wheel_files:
        match = _WHEEL_FILENAME_RE.match(wheel.name)
        if match is None:
            raise RuntimeError(f"unparsable wheel filename: {wheel.name}")
        packed.add((_normalize(match.group("name")), match.group("version")))
    missing = sorted(requirement_set.difference(packed))
    if missing:
        raise RuntimeError(f"runtime pack is missing wheels for: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    # pack 内附无哈希需求清单:离线安装以它为解析输入(原 lock 含 sdist
    # 哈希行,pip 会对 sdist 构建的 wheel 报哈希不匹配);pack 整体字节
    # 完整性由 manifest 的 runtime_pack_sha256 绑定。
    pack_requirements = (
        "\n".join(f"{name}=={version}" for name, version in sorted(requirement_set))
        + "\n"
    )
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        requirements_info = zipfile.ZipInfo(
            "pack-requirements.txt", date_time=_FIXED_ZIP_DATE_TIME
        )
        requirements_info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(requirements_info, pack_requirements.encode("utf-8"))
        for wheel in wheel_files:
            info = zipfile.ZipInfo(wheel.name, date_time=_FIXED_ZIP_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, wheel.read_bytes())
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        path = build_runtime_pack(
            lock=args.lock,
            profile=args.profile,
            work_dir=args.work_dir,
            output=args.output,
        )
    except (ManifestError, ValueError, RuntimeError) as exc:
        print(f"runtime pack build failed: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
