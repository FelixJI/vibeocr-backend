"""Resolve and validate the Backend's exact Protocol build binding."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

_STABLE_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_CONTRACTS_REQUIREMENT = re.compile(
    r"^vibeocr-runtime-contracts>=(\d+\.\d+\.\d+),<(\d+\.\d+\.\d+)$"
)


@dataclass(frozen=True, order=True, slots=True)
class StableVersion:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: object, *, field: str) -> StableVersion:
        if (
            not isinstance(value, str)
            or (match := _STABLE_SEMVER.fullmatch(value)) is None
        ):
            raise ValueError(f"{field} must be stable SemVer")
        return cls(*(int(part) for part in match.groups()))


def resolve_protocol_binding(lock_path: Path, package_path: Path) -> str:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(lock, dict) or lock.get("schema_version") != 1:
        raise ValueError("Protocol lock schema_version must be 1")
    if lock.get("repository") != "FelixJI/vibeocr-protocol":
        raise ValueError("Protocol lock repository is invalid")
    raw_version = lock.get("version")
    locked = StableVersion.parse(raw_version, field="Protocol lock version")

    project = tomllib.loads(package_path.read_text(encoding="utf-8"))["project"]
    dependencies = project.get("dependencies")
    if not isinstance(dependencies, list):
        raise ValueError("Backend project dependencies must be an array")
    requirements = [
        item
        for item in dependencies
        if isinstance(item, str) and item.startswith("vibeocr-runtime-contracts")
    ]
    if len(requirements) != 1:
        raise ValueError("Backend must declare exactly one Protocol requirement")
    match = _CONTRACTS_REQUIREMENT.fullmatch(requirements[0])
    if match is None:
        raise ValueError(
            "Backend Protocol requirement must use >=lower,<next-major stable SemVer"
        )
    lower = StableVersion.parse(match.group(1), field="Protocol lower bound")
    upper = StableVersion.parse(match.group(2), field="Protocol upper bound")
    if upper != StableVersion(lower.major + 1, 0, 0):
        raise ValueError("Backend Protocol requirement must stop at the next major")
    if not lower <= locked < upper:
        raise ValueError("Protocol lock version is outside the Backend package range")

    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Protocol lock artifacts must be an object")
    expected_prefix = f"vibeocr_runtime_contracts-{raw_version}-"
    matching_wheels = [
        name
        for name in artifacts
        if isinstance(name, str)
        and name.startswith(expected_prefix)
        and name.endswith(".whl")
    ]
    if len(matching_wheels) != 1:
        raise ValueError("Protocol lock must bind exactly one contracts wheel")
    return str(raw_version)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args(argv)
    print(resolve_protocol_binding(args.lock, args.package))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
