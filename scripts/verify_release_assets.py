"""Verify a release candidate's checksum index and required assets."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

_CHECKSUM_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


class ReleaseAssetError(ValueError):
    """Raised when release candidate assets are incomplete or inconsistent."""


def _asset_name(value: str) -> str:
    if not value or Path(value).name != value or "/" in value or "\\" in value:
        raise ReleaseAssetError(f"unsafe asset name in SHA256SUMS: {value!r}")
    return value


def verify_release_assets(
    artifacts_dir: Path,
    *,
    required: Iterable[str] = (),
    require_one: Iterable[str] = (),
) -> tuple[str, ...]:
    """Verify checksums, the indexed file set, and repository asset requirements."""
    root = artifacts_dir.resolve(strict=True)
    if not root.is_dir():
        raise ReleaseAssetError("artifacts_dir must be a directory")

    checksum_path = root / "SHA256SUMS"
    if not checksum_path.is_file():
        raise ReleaseAssetError("missing required asset: SHA256SUMS")

    checksums: dict[str, str] = {}
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise ReleaseAssetError(f"invalid SHA256SUMS line {line_number}")
        digest, raw_name = match.groups()
        name = _asset_name(raw_name)
        if name in checksums:
            raise ReleaseAssetError(f"duplicate SHA256SUMS entry: {name}")
        checksums[name] = digest

    indexed = set(checksums)
    actual = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != checksum_path.name
    }
    if indexed != actual:
        missing = sorted(indexed - actual)
        unindexed = sorted(actual - indexed)
        raise ReleaseAssetError(
            f"SHA256SUMS file set mismatch; missing={missing}, unindexed={unindexed}"
        )

    for name, expected in checksums.items():
        actual_digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual_digest != expected:
            raise ReleaseAssetError(f"SHA-256 mismatch: {name}")

    for name in required:
        safe_name = _asset_name(name)
        if safe_name not in actual:
            raise ReleaseAssetError(f"missing required asset: {safe_name}")

    for pattern in require_one:
        matches = sorted(name for name in actual if fnmatch.fnmatchcase(name, pattern))
        if len(matches) != 1:
            raise ReleaseAssetError(
                f"required pattern must match exactly one asset: {pattern!r}; "
                f"matches={matches}"
            )

    return tuple(sorted(actual | {checksum_path.name}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts_dir", type=Path)
    parser.add_argument("--require", action="append", default=[])
    parser.add_argument("--require-one", action="append", default=[])
    args = parser.parse_args()
    for name in verify_release_assets(
        args.artifacts_dir,
        required=args.require,
        require_one=args.require_one,
    ):
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
