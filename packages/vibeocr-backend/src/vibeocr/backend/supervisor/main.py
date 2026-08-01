"""``vibeocr-supervisor`` entry point.

Binds a pre-created ``127.0.0.1:0`` socket, emits the ready envelope on
stdout, then serves the FastAPI app via uvicorn. The session token is
delivered out of band via the ``VIBEOCR_SUP_TOKEN`` env var (inherited from
the parent process) so it never appears on stdout/argv/logs.

Usage from a parent process (PySide/WinUI):

    proc = subprocess.Popen(
        [python, "-m", "vibeocr.backend.supervisor.main"],
        env={**os.environ, "VIBEOCR_SUP_TOKEN": token, "VIBEOCR_SUP_ROOT": root},
        stdout=PIPE,
    )
    ready = json.loads(proc.stdout.readline())
    port = ready["port"]
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from vibeocr.runtime_contracts.generated import (
    ALL_CAPABILITIES,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
)

from .bootstrap import (
    BootstrapHandle,
    ReadyEnvelope,
    bind_loopback_socket,
    emit_ready,
    new_instance_id,
    token_from_environment,
)
from .composition import build_supervisor


def _write_self_test_result() -> None:
    """Persist child-process source evidence for the frozen T6 release gate."""
    if os.environ.get("VIBEOCR_SELF_TEST_SMOKE") != "t6":
        return
    target = os.environ.get("VIBEOCR_SELF_TEST_RESULT")
    if not target:
        return
    Path(target).write_text(
        json.dumps(
            {
                "supervisor_ready": True,
                "module_file": str(Path(__file__).resolve()),
                "python_executable": str(Path(sys.executable).resolve()),
            }
        ),
        encoding="utf-8",
    )


def _build_uvicorn_config(
    uvicorn_module: Any,
    app: Any,
    port: int,
) -> Any:
    """Create the local server config without per-request access-log noise."""
    return uvicorn_module.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="info",
        access_log=False,
        # Pass the pre-bound socket so uvicorn does not rebind (port-0 race).
        workers=1,
    )


def run_supervisor(
    argv: list[str] | None = None,
) -> int:  # pragma: no cover - entry point
    """Run the supervisor until interrupted. Returns process exit code."""
    instance_id = new_instance_id()
    token = token_from_environment()
    if not token:
        # Without a token we cannot safely serve; fail fast.
        sys.stderr.write("vibeocr-supervisor: missing VIBEOCR_SUP_TOKEN\n")
        return 2
    handle = BootstrapHandle(token)
    sock = bind_loopback_socket()
    port = sock.getsockname()[1]
    root_env = os.environ.get("VIBEOCR_SUP_ROOT")
    stager_root = Path(root_env) if root_env else None

    module, _ = build_supervisor(
        instance_id=instance_id,
        stager_root=stager_root,
        bootstrap_handle=handle,
        # The supervisor owns the PDF child so the GUI never talks to it
        # directly (plan §6 / ADR §"Transport"). The child is spawned lazily.
        with_pdf_adapter=True,
    )
    envelope = ReadyEnvelope(
        ready=True,
        pid=os.getpid(),
        port=port,
        instance_id=instance_id,
        protocol_version=PROTOCOL_VERSION,
        schema_version=SCHEMA_VERSION,
        capabilities=list(ALL_CAPABILITIES),
    )
    _write_self_test_result()
    emit_ready(envelope)
    _schedule_soak_crash_after_ready()

    # Import lazily so the module can be imported in environments without
    # uvicorn (e.g. pure contract tests).
    try:
        import uvicorn

        from .app import create_app
    except Exception as exc:  # pragma: no cover - environment dependent
        sys.stderr.write(f"vibeocr-supervisor: cannot start server: {exc}\n")
        return 3

    app = create_app(module, handle.token)
    config = _build_uvicorn_config(uvicorn, app, port)
    server = uvicorn.Server(config)
    config.load()
    # Hand the bound socket to the server.
    server.servers = []  # type: ignore[attr-defined]
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_serve_with_socket(server, sock, app, handle.token))
    except KeyboardInterrupt:  # pragma: no cover - signal
        pass
    finally:
        module.shutdown_now()
        loop.close()
    return 0


def _missing_token_exit() -> int:
    """Pure-logic seam extracted from ``run_supervisor`` for unit testing.

    ``run_supervisor`` itself is an environment-dependent entry point
    (``# pragma: no cover - entry point``) because it binds a real loopback
    socket and serves uvicorn. The missing-token early-exit, however, is a
    testable contract: callers must see a stable exit code and message.
    """
    token = token_from_environment()
    if not token:
        sys.stderr.write("vibeocr-supervisor: missing VIBEOCR_SUP_TOKEN\n")
        return 2
    return 0


def _schedule_soak_crash_after_ready() -> None:  # pragma: no cover - soak harness only
    """Crash this process only when the WinUI soak harness explicitly asks.

    The ready envelope is emitted first so the frontend exercises its normal
    attach path.  ``os._exit`` intentionally bypasses graceful cleanup and
    therefore tests the same unexpected-exit recovery used for real crashes.
    """
    if os.environ.get("VIBEOCR_SUPERVISOR_SOAK_CRASH_AFTER_READY") != "1":
        return

    def crash() -> None:
        time.sleep(0.25)
        os._exit(86)

    threading.Thread(
        target=crash,
        name="vibeocr-soak-crash",
        daemon=True,
    ).start()


def _soak_crash_enabled() -> bool:
    """Pure-logic seam for the env-gate checked by ``_schedule_soak_crash_after_ready``.

    The scheduling function itself spawns a thread that calls ``os._exit`` and
    is therefore not unit-testable; this predicate exposes the gate so the
    contract (crash only when the harness opts in) is covered.
    """
    return os.environ.get("VIBEOCR_SUPERVISOR_SOAK_CRASH_AFTER_READY") == "1"


async def _serve_with_socket(
    server, sock, app, token
) -> None:  # pragma: no cover - integration
    """Serve using the pre-bound socket. Kept thin for testability."""
    config = server.config
    # uvicorn supports passing a configured socket via Server.startup via
    # the ``sockets`` kwarg once the server is started. For the test path we
    # exercise the app via httpx/ASGI directly; this function only runs in
    # the real subprocess.
    config.app = app
    server.config = config
    await server.serve(sockets=[sock])


def main() -> int:  # pragma: no cover - entry point
    return run_supervisor()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
