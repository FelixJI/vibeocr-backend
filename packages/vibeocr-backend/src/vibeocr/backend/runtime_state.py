"""Backend-owned portable runtime state.

This module deliberately does not import Classic.  Its JSON file is separate
from Classic's product cache so a clean Backend wheel can inspect and repair a
runtime without any frontend package installed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock

CACHE_VERSION = 1
CACHE_TTL_DAYS = 7
_STATE_DIR = "data/backend"
_STATE_FILE = "runtime-state.json"
_machine_id: str | None = None
_machine_id_lock = Lock()


def generate_machine_id() -> str:
    """Return a stable, non-secret machine fingerprint."""
    global _machine_id
    if _machine_id is not None:
        return _machine_id
    with _machine_id_lock:
        if _machine_id is None:
            material = f"{platform.node()}|{uuid.getnode():012x}"
            _machine_id = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return _machine_id


def get_cache_dir(project_root: Path) -> Path:
    """Return the Backend-owned state directory under the portable root."""
    return Path(project_root).resolve() / _STATE_DIR


def get_cache_path(project_root: Path) -> Path:
    return get_cache_dir(project_root) / _STATE_FILE


def load_cache(project_root: Path) -> dict | None:
    path = get_cache_path(project_root)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def save_cache(project_root: Path, data: dict) -> bool:
    path = get_cache_path(project_root)
    temporary = path.with_suffix(".json.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return True
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def is_cache_valid(project_root: Path) -> tuple[bool, dict | None]:
    data = load_cache(project_root)
    if data is None:
        return False, None
    if data.get("version") != CACHE_VERSION:
        return False, None
    if data.get("machine_id") != generate_machine_id():
        return False, None
    return True, data


def create_cache_entry(
    project_root: Path,
    dependencies: dict,
    hardware_info: dict,
) -> dict | None:
    import sys

    data = {
        "version": CACHE_VERSION,
        "machine_id": generate_machine_id(),
        "last_check_time": datetime.now().isoformat(),
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
        "dependencies": dependencies,
        "hardware_info": hardware_info,
    }
    return data if save_cache(project_root, data) else None


def update_cache_field(project_root: Path, key: str, value: object) -> bool:
    valid, data = is_cache_valid(project_root)
    if not valid or data is None:
        return False
    updated = dict(data)
    updated[key] = value
    return save_cache(project_root, updated)


def get_cache_age_seconds(project_root: Path) -> float | None:
    data = load_cache(project_root)
    if data is None:
        return None
    raw = data.get("last_check_time")
    if not isinstance(raw, str):
        return None
    try:
        return (datetime.now() - datetime.fromisoformat(raw)).total_seconds()
    except ValueError:
        return None


__all__ = [
    "CACHE_TTL_DAYS",
    "CACHE_VERSION",
    "create_cache_entry",
    "generate_machine_id",
    "get_cache_age_seconds",
    "get_cache_dir",
    "get_cache_path",
    "is_cache_valid",
    "load_cache",
    "save_cache",
    "update_cache_field",
]
