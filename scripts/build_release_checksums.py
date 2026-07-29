"""Write a deterministic SHA256SUMS index for release assets."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_release_checksums(artifacts_dir: Path) -> Path:
    artifacts_dir = artifacts_dir.resolve(strict=True)
    output = artifacts_dir / "SHA256SUMS"
    files = sorted(
        (
            path
            for path in artifacts_dir.iterdir()
            if path.is_file() and path.name != output.name
        ),
        key=lambda path: path.name,
    )
    if not files:
        raise ValueError("release artifacts directory is empty")
    output.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
        newline="\n",
    )
    return output


def write_sidecar_checksum(path: Path) -> Path:
    path = path.resolve(strict=True)
    output = path.with_name(path.name + ".sha256")
    output.write_text(
        f"{sha256_file(path)}  {path.name}\n",
        encoding="utf-8",
        newline="\n",
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts_dir", type=Path)
    parser.add_argument("--sidecar-for", type=Path)
    args = parser.parse_args(argv)
    if args.sidecar_for is not None:
        print(write_sidecar_checksum(args.sidecar_for))
    print(build_release_checksums(args.artifacts_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
