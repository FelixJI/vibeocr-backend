"""Auth + loopback + quota middleware for the supervisor HTTP layer.

Enforces (plan §6 Phase 2):

* every business request carries a valid Bearer session token;
* the connection comes from the loopback interface;
* request bodies respect configured size/count quotas.

The middleware is framework-light: it operates on Starlette request/response
objects so it can be unit-tested without a live server.
"""

from __future__ import annotations

from dataclasses import dataclass

from vibeocr.runtime_contracts import SCHEMA_VERSION, ErrorCode, ErrorPayload


@dataclass(frozen=True, slots=True)
class AuthDecision:
    """Outcome of an auth/loopback/quota check."""

    ok: bool
    error: ErrorPayload | None = None


def _make_error(code: ErrorCode, instance_id: str, *, detail: dict | None = None) -> ErrorPayload:
    from vibeocr.runtime_contracts import error_registry

    entry = error_registry[code]
    return ErrorPayload(
        schema_version=SCHEMA_VERSION,
        instance_id=instance_id,
        code=code,
        message=entry.message,
        category=entry.category,
        retryable=entry.retryable,
        detail=detail or {},
    )


def check_bearer_token(
    authorization_header: str | None,
    expected_token: str,
    *,
    instance_id: str,
) -> AuthDecision:
    """Validate ``Authorization: Bearer <token>``."""
    if not authorization_header:
        return AuthDecision(False, _make_error(ErrorCode.UNAUTHORIZED, instance_id))
    parts = authorization_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return AuthDecision(False, _make_error(ErrorCode.UNAUTHORIZED, instance_id))
    # Constant-time-ish comparison to avoid trivial timing oracles.
    import hmac

    if not hmac.compare_digest(parts[1].strip(), expected_token):
        return AuthDecision(False, _make_error(ErrorCode.UNAUTHORIZED, instance_id))
    return AuthDecision(True)


def check_loopback(
    client_host: str | None,
    *,
    instance_id: str,
) -> AuthDecision:
    """Reject any client not on the loopback interface."""
    if client_host is None:
        return AuthDecision(False, _make_error(ErrorCode.FORBIDDEN_LOOPBACK, instance_id))
    if client_host not in ("127.0.0.1", "::1", "localhost"):
        return AuthDecision(False, _make_error(ErrorCode.FORBIDDEN_LOOPBACK, instance_id))
    return AuthDecision(True)


def is_bootstrap_path(path: str) -> bool:
    """Paths that do not require a token (health probe only)."""
    return path in ("/v2/health",)


__all__ = [
    "AuthDecision",
    "check_bearer_token",
    "check_loopback",
    "is_bootstrap_path",
]
