"""Supervisor bootstrap: session token, ready envelope, port-0 socket bind.

Phase 2 exit criteria:

* bind a pre-created ``127.0.0.1:0`` socket to eliminate port-selection races;
* emit a stdout ready envelope with port/PID/instance id/protocol/capabilities;
* generate a 256-bit session token delivered **out of band** (env or a
  controlled bootstrap handle), never on stdout/logs/command line.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import sys
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vibeocr.runtime_contracts.generated import (
    ALL_CAPABILITIES,
    PROTOCOL_VERSION,
    READY_ENVELOPE_VERSION,
    SCHEMA_VERSION,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def generate_session_token() -> str:
    """Generate a 256-bit URL-safe session token."""
    return secrets.token_urlsafe(32)


def new_instance_id() -> str:
    """Short, human-readable instance id for log correlation."""
    return f"sup-{secrets.token_hex(5)}"


def bind_loopback_socket() -> socket.socket:
    """Bind a ``127.0.0.1:0`` socket and return it (un-listened).

    The caller passes the bound socket to the server so the OS-assigned port
    is known *before* the server starts listening — this removes the
    port-selection race the old client had.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    return sock


@dataclass(slots=True)
class ReadyEnvelope:
    """Emitted as the first stdout line; subsequent stdout is logs only."""

    ready: bool
    pid: int
    port: int
    instance_id: str
    protocol_version: int
    schema_version: int
    capabilities: list[str]
    ready_version: int = READY_ENVELOPE_VERSION

    def __post_init__(self) -> None:
        if not self.ready or self.pid <= 0 or not 1 <= self.port <= 65535:
            raise ValueError("invalid ready envelope identity")
        if not self.instance_id:
            raise ValueError("instance_id is required")
        if (
            self.protocol_version != PROTOCOL_VERSION
            or self.schema_version != SCHEMA_VERSION
            or self.ready_version != READY_ENVELOPE_VERSION
        ):
            raise ValueError("incompatible ready envelope version")
        if len(self.capabilities) != len(set(self.capabilities)):
            raise ValueError("ready capabilities must be unique")
        unknown = set(self.capabilities) - set(ALL_CAPABILITIES)
        if unknown:
            raise ValueError(f"unknown ready capabilities: {sorted(unknown)}")

    def to_json(self) -> str:
        return json.dumps(
            {
                "ready": self.ready,
                "pid": self.pid,
                "port": self.port,
                "instance_id": self.instance_id,
                "protocol_version": self.protocol_version,
                "schema_version": self.schema_version,
                "capabilities": self.capabilities,
                "ready_version": self.ready_version,
            },
            separators=(",", ":"),
        )


class BootstrapHandle:
    """Controlled channel carrying the session token.

    The token is placed here by the parent process *before* the supervisor
    reads its ready envelope, and is never written to stdout, argv or logs.
    In production this is an inherited env var or a named handle; in tests it
    is set directly on the instance.
    """

    __slots__ = ("_lock", "_token")

    def __init__(self, token: str | None = None) -> None:
        self._token = token
        self._lock = threading.Lock()

    def set_token(self, token: str) -> None:
        with self._lock:
            self._token = token

    @property
    def token(self) -> str:
        with self._lock:
            if self._token is None:
                raise RuntimeError("session token not set on bootstrap handle")
            return self._token


def token_from_environment(env: Mapping[str, str] | None = None) -> str | None:
    """Read a session token from the process environment, if present.

    The env var name is intentionally obfuscated to discourage logging.
    """
    source = os.environ if env is None else env
    return source.get("VIBEOCR_SUP_TOKEN")


def emit_ready(envelope: ReadyEnvelope, *, stream: Any = None) -> None:
    """Write the ready envelope as a single line on stdout.

    After this call, the supervisor MUST NOT emit structured JSON on stdout;
    all further output is plain log text.
    """
    stream = stream or sys.stdout
    stream.write(envelope.to_json() + "\n")
    stream.flush()


__all__ = [
    "READY_ENVELOPE_VERSION",
    "BootstrapHandle",
    "ReadyEnvelope",
    "bind_loopback_socket",
    "emit_ready",
    "generate_session_token",
    "new_instance_id",
    "token_from_environment",
]
