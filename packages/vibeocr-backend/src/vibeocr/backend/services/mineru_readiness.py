"""MinerU tier readiness: installed dependencies and observed execution differ."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import metadata
from urllib.parse import urlsplit

from vibeocr.runtime_contracts import ErrorCode, MineruTier

from .mineru_config import LANGUAGES, MineruConfigError

_lock = threading.Lock()
# tier → 观察到该 tier 实际解析成功的连接。旧连接的观察对新连接无效：
# 即使清观察集后仍有旧结果迟到落地，也不得污染新连接的就绪声明。
_observations: dict[MineruTier, MineruConnection] = {}

# 连接模式：local 由 Supervisor 管理本地 mineru-api 子进程；
# remote 使用自部署 MinerU 4 服务器的 V1 HTTP 接口。
CONNECTION_LOCAL = "local"
CONNECTION_REMOTE = "remote"
_CONNECTION_FIELDS = frozenset({"mode", "api_url", "api_key"})


@dataclass(frozen=True)
class MineruConnection:
    """已验证的 MinerU 传输目标：本地子进程或自部署远程服务器。"""

    mode: str = CONNECTION_LOCAL
    api_url: str = ""
    # 凭据不进入 repr：避免日志/异常消息意外泄漏 key。
    api_key: str = field(default="", repr=False)


_default_connection = MineruConnection()
_connection: MineruConnection = _default_connection


def _reject(reason: str) -> None:
    raise MineruConfigError(ErrorCode.VALIDATION_ERROR, reason)


def _validated_api_url(raw: object) -> str:
    if not isinstance(raw, str) or not raw:
        _reject("mineru_connection_api_url_missing")
    assert isinstance(raw, str)
    if any(character.isspace() for character in raw):
        _reject("mineru_connection_api_url_whitespace")
    try:
        parts = urlsplit(raw)
        # 无效 IPv6 字面量/端口在这里抛 ValueError；统一拒绝且错误码
        # 不携带用户输入，避免把 URL 回显到错误消息、日志或异常链。
        parts.port
    except ValueError:
        raise MineruConfigError(
            ErrorCode.VALIDATION_ERROR, "mineru_connection_api_url_invalid"
        ) from None
    if parts.scheme not in ("http", "https"):
        _reject("mineru_connection_api_url_scheme")
    if not parts.hostname:
        _reject("mineru_connection_api_url_host")
    if parts.username is not None or parts.password is not None:
        _reject("mineru_connection_api_url_userinfo")
    if parts.query or parts.fragment:
        _reject("mineru_connection_api_url_query_fragment")
    # 反向代理路径合法；仅去掉尾部斜杠以便配置变更比较语义稳定。
    return raw.rstrip("/")


def _validated_api_key(raw: object) -> str:
    if raw is None:
        return ""
    if not isinstance(raw, str):
        _reject("mineru_connection_api_key_type")
    assert isinstance(raw, str)
    if "\r" in raw or "\n" in raw:
        _reject("mineru_connection_api_key_control_characters")
    return raw


def connection_from_extra(extra: Mapping[str, object] | None) -> MineruConnection:
    """Resolve ``SettingsSnapshot.extra['mineru_connection']``; default local.

    非法配置 fail closed：未知字段、非法 mode、local 携带远程字段、
    URL 含 userinfo/query/fragment 或非 http(s) scheme、key 含 CRLF
    均抛出 :class:`MineruConfigError`，不静默回退 local。
    """
    if extra is None:
        return _default_connection
    raw = extra.get("mineru_connection")
    if raw is None:
        return _default_connection
    if not isinstance(raw, Mapping):
        _reject("mineru_connection_shape")
    unknown = set(raw) - _CONNECTION_FIELDS
    if unknown:
        _reject("mineru_connection_unknown_fields")
    mode = raw.get("mode", CONNECTION_LOCAL)
    if mode not in (CONNECTION_LOCAL, CONNECTION_REMOTE):
        _reject("mineru_connection_mode")
    if mode == CONNECTION_LOCAL:
        if raw.get("api_url") or raw.get("api_key"):
            _reject("mineru_connection_local_fields")
        return _default_connection
    return MineruConnection(
        mode=CONNECTION_REMOTE,
        api_url=_validated_api_url(raw.get("api_url")),
        api_key=_validated_api_key(raw.get("api_key", "")),
    )


def current_connection() -> MineruConnection:
    with _lock:
        return _connection


def configure_connection(connection: MineruConnection) -> None:
    """应用已验证的连接；连接变更清空 tier 观察集。

    观察到的 tier 就绪属于具体连接（远程服务器的模型/版本与本地不同），
    旧连接的执行结果不得污染新连接。清空之外，每条观察还携带连接身份
    （见 ``_observations``），迟到落地的旧结果仍不会计入新连接。
    """
    global _connection
    with _lock:
        if connection != _connection:
            _observations.clear()
            _connection = connection


def _mineru4_installed() -> bool:
    try:
        return metadata.version("mineru").split(".")[0] == "4"
    except metadata.PackageNotFoundError:
        return False


def mark_executed(
    tier: MineruTier, *, connection: MineruConnection | None = None
) -> None:
    """Record an observed execution against the connection that served it.

    ``connection`` 缺省为当前连接；调用方持有入口快照时（如 prepare 逐
    tier 解析）必须显式传入，避免切换后迟到的旧结果记到新连接名下。
    """
    with _lock:
        _observations[tier] = _connection if connection is None else connection


def mark_failed(tier: MineruTier) -> None:
    # 无条件移除：宁可保守要求重新准备，也不保留可能过时的就绪观察。
    with _lock:
        _observations.pop(tier, None)


def catalog_payload(*, observe_runtime: bool = True) -> dict:
    with _lock:
        connection = _connection
        remote = connection.mode == CONNECTION_REMOTE
        ready = (
            {tier for tier, observed in _observations.items() if observed == connection}
            if observe_runtime
            else set()
        )
    # 远程模式由自部署服务器承担解析，本地 mineru 包不是就绪前提；
    # 仍要求当前进程实际完成该 tier 的准备解析后才标记 ready。
    installed = remote or _mineru4_installed()
    return {
        "default_tier": "basic",
        "languages": list(LANGUAGES),
        "tiers": [
            {
                "id": tier.value,
                "availability": "ready"
                if installed and tier in ready
                else "preparation_required",
                "reason_code": None
                if installed and tier in ready
                else (
                    "mineru_execution_preparation_required"
                    if not observe_runtime
                    else "mineru_dependencies_required"
                    if not installed
                    else "mineru_execution_preparation_required"
                ),
            }
            for tier in MineruTier
        ],
    }


def require_ready(tier: MineruTier) -> None:
    descriptor = next(
        item for item in catalog_payload()["tiers"] if item["id"] == tier.value
    )
    if descriptor["availability"] != "ready":
        raise MineruConfigError(
            ErrorCode.MINERU_TIER_PREPARATION_REQUIRED, descriptor["reason_code"]
        )
