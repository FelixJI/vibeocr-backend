"""远程自部署 MinerU 4 连接（Backend #113）回归。

旧实现会在以下场景失败：无 mineru 包时直接构造 MinerUService 会启动本地
子进程、MineruApiClient 无 Bearer/重定向防护、连接配置无解析与门禁。
"""

from __future__ import annotations

import json
from importlib import metadata
from unittest.mock import patch

import httpx
import pytest
from vibeocr.backend.services import mineru_readiness as readiness
from vibeocr.backend.services.mineru_api import (
    MineruApiClient,
    MineruApiError,
    MineruDocument,
)
from vibeocr.backend.services.mineru_config import MineruConfigError
from vibeocr.backend.services.mineru_service import MinerUService
from vibeocr.runtime_contracts import ErrorCode, MineruConfig, MineruTier


@pytest.fixture()
def isolated_connection(monkeypatch: pytest.MonkeyPatch):
    """重置模块级连接与观察集，测试间互不污染。"""
    monkeypatch.setattr(readiness, "_connection", readiness.MineruConnection())
    monkeypatch.setattr(readiness, "_observations", {})
    yield readiness
    monkeypatch.setattr(readiness, "_connection", readiness.MineruConnection())


@pytest.fixture()
def not_installed_mineru(monkeypatch: pytest.MonkeyPatch):
    """本地未安装 mineru 包（Base Runtime）——远程模式仍必须可用。"""

    def raise_not_found(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(readiness.metadata, "version", raise_not_found)


REMOTE_URL = "https://mineru.example.com/selfhosted"
REMOTE_KEY = "secret-token-1"


def _remote_connection(url: str = REMOTE_URL, key: str = REMOTE_KEY):
    return readiness.connection_from_extra(
        {"mineru_connection": {"mode": "remote", "api_url": url, "api_key": key}}
    )


# ---------------------------------------------------------------------------
# connection_from_extra：解析与 fail-closed 校验
# ---------------------------------------------------------------------------


class TestConnectionParsing:
    def test_default_is_local(self, isolated_connection):
        assert readiness.connection_from_extra(None) == readiness.MineruConnection()
        assert readiness.connection_from_extra({}) == readiness.MineruConnection()
        assert readiness.connection_from_extra({"other": 1}).mode == "local"

    def test_remote_with_reverse_proxy_path_and_trailing_slash(self):
        connection = _remote_connection("https://host.example/internal/mineru/")
        assert connection.mode == "remote"
        assert connection.api_url == "https://host.example/internal/mineru"
        assert connection.api_key == REMOTE_KEY

    def test_remote_without_key_is_allowed(self):
        connection = readiness.connection_from_extra(
            {"mineru_connection": {"mode": "remote", "api_url": REMOTE_URL}}
        )
        assert connection.api_key == ""

    @pytest.mark.parametrize(
        "value",
        [
            {"mode": "remote"},  # 缺 api_url
            {"mode": "remote", "api_url": "ftp://host"},  # 非 http(s)
            {"mode": "remote", "api_url": "file:///etc/passwd"},
            {"mode": "remote", "api_url": "https://user:pw@host"},
            {"mode": "remote", "api_url": "https://user@host"},
            {"mode": "remote", "api_url": f"{REMOTE_URL}?token=1"},
            {"mode": "remote", "api_url": f"{REMOTE_URL}#frag"},
            {"mode": "remote", "api_url": "https:///path"},
            {"mode": "remote", "api_url": ""},
            {"mode": "remote", "api_url": 42},
            {"mode": "remote", "api_url": REMOTE_URL, "api_key": "x\r\nHost: evil"},
            {"mode": "remote", "api_url": REMOTE_URL, "api_key": "x\n"},
            {"mode": "remote", "api_url": REMOTE_URL, "api_key": 7},
            {"mode": "wan"},  # 未知 mode
            {"mode": "local", "api_url": REMOTE_URL},  # local 携带远程字段
            {"mode": "local", "api_key": "x"},
            {"mode": "remote", "api_url": REMOTE_URL, "extra_field": 1},  # 未知字段
        ],
    )
    def test_invalid_config_fails_closed(self, value):
        with pytest.raises(MineruConfigError) as caught:
            readiness.connection_from_extra({"mineru_connection": value})
        assert caught.value.code == ErrorCode.VALIDATION_ERROR
        # 错误原因不得回显 URL/key。
        assert REMOTE_URL not in str(caught.value)
        assert REMOTE_KEY not in str(caught.value)

    def test_non_object_connection_fails_closed(self):
        with pytest.raises(MineruConfigError):
            readiness.connection_from_extra({"mineru_connection": "remote"})

    @pytest.mark.parametrize(
        "url",
        [
            "http://[::1",  # 无效 IPv6 字面量：urlsplit 自身抛 ValueError
            "https://host.example:notaport",  # 无效端口
        ],
    )
    def test_invalid_ipv6_or_port_uses_sanitized_error_code(self, url):
        with pytest.raises(MineruConfigError) as caught:
            readiness.connection_from_extra(
                {"mineru_connection": {"mode": "remote", "api_url": url}}
            )
        assert caught.value.reason == "mineru_connection_api_url_invalid"
        assert url not in str(caught.value)

    def test_api_key_never_appears_in_connection_repr(self):
        connection = _remote_connection()
        assert connection.api_key == REMOTE_KEY
        assert REMOTE_KEY not in repr(connection)
        explicit = readiness.MineruConnection(mode="remote", api_key=REMOTE_KEY)
        assert REMOTE_KEY not in repr(explicit)


# ---------------------------------------------------------------------------
# configure_connection：变更清观察集
# ---------------------------------------------------------------------------


class TestConnectionChangeClearsObservations:
    def test_change_clears_ready_set(self, isolated_connection):
        remote = _remote_connection()
        readiness.configure_connection(remote)
        readiness.mark_executed(MineruTier.BASIC)
        assert readiness.require_ready(MineruTier.BASIC) is None
        # 切回 local：观察集被清空，重新需要准备。
        readiness.configure_connection(readiness.MineruConnection())
        with pytest.raises(MineruConfigError):
            readiness.require_ready(MineruTier.BASIC)

    def test_same_connection_keeps_observations(self, isolated_connection):
        remote = _remote_connection()
        readiness.configure_connection(remote)
        readiness.mark_executed(MineruTier.BASIC)
        readiness.configure_connection(remote)
        readiness.require_ready(MineruTier.BASIC)  # 不抛


class TestObservationIdentity:
    """观察归属实际服务解析的连接：迟到的旧结果不得污染新连接。"""

    @staticmethod
    def _remote(url: str):
        return readiness.connection_from_extra(
            {"mineru_connection": {"mode": "remote", "api_url": url}}
        )

    def test_stale_mark_executed_after_switch_does_not_pollute(
        self, isolated_connection
    ):
        remote_a = self._remote("https://a.example.com")
        remote_b = self._remote("https://b.example.com")
        readiness.configure_connection(remote_a)
        readiness.mark_executed(MineruTier.BASIC, connection=remote_a)
        readiness.require_ready(MineruTier.BASIC)  # A 端观察对 A 生效

        readiness.configure_connection(remote_b)  # 变更清观察集
        with pytest.raises(MineruConfigError):
            readiness.require_ready(MineruTier.BASIC)
        # 清空后才落地的旧 A 端结果（无租约直接服务路径的竞态）不计入 B。
        readiness.mark_executed(MineruTier.BASIC, connection=remote_a)
        with pytest.raises(MineruConfigError):
            readiness.require_ready(MineruTier.BASIC)
        # B 端自己的观察立即生效。
        readiness.mark_executed(MineruTier.BASIC, connection=remote_b)
        readiness.require_ready(MineruTier.BASIC)

    def test_default_connection_argument_uses_current(self, isolated_connection):
        remote = _remote_connection()
        readiness.configure_connection(remote)
        readiness.mark_executed(MineruTier.FLASH)
        assert readiness._observations[MineruTier.FLASH] == remote


# ---------------------------------------------------------------------------
# tier 目录门禁：remote 无本地依赖可用，local 无依赖仍拦截
# ---------------------------------------------------------------------------


class TestTierCatalogGating:
    def test_remote_catalog_does_not_require_local_package(
        self, isolated_connection, not_installed_mineru
    ):
        readiness.configure_connection(_remote_connection())
        payload = readiness.catalog_payload()
        assert all(
            tier["availability"] == "preparation_required" for tier in payload["tiers"]
        )
        assert all(
            tier["reason_code"] == "mineru_execution_preparation_required"
            for tier in payload["tiers"]
        )
        readiness.mark_executed(MineruTier.BASIC)
        basic = next(
            tier
            for tier in readiness.catalog_payload()["tiers"]
            if tier["id"] == "basic"
        )
        assert basic["availability"] == "ready"

    def test_local_catalog_without_package_stays_blocked(
        self, isolated_connection, not_installed_mineru
    ):
        readiness.configure_connection(_remote_connection())
        readiness.mark_executed(MineruTier.BASIC)
        # 切回 local：观察集被清空，且本地无依赖仍拦截。
        readiness.configure_connection(readiness.MineruConnection())
        payload = readiness.catalog_payload()
        assert all(
            tier["availability"] == "preparation_required" for tier in payload["tiers"]
        )
        assert all(
            tier["reason_code"] == "mineru_dependencies_required"
            for tier in payload["tiers"]
        )
        with pytest.raises(MineruConfigError) as caught:
            readiness.require_ready(MineruTier.BASIC)
        assert caught.value.code == ErrorCode.MINERU_TIER_PREPARATION_REQUIRED


# ---------------------------------------------------------------------------
# MinerUService：remote 不启动本地、失败不回退
# ---------------------------------------------------------------------------


class _RecordingClient:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.url = ""
        self.api_key = None
        self.calls = []

    def parse(self, files, config, *, cancelled=lambda: False):
        self.calls.append((self.url, self.api_key, tuple(name for name, _ in files)))
        if self.error is not None:
            raise self.error
        return {name: self.result for name, _ in files}


class TestMinerUServiceRemote:
    @pytest.fixture(autouse=True)
    def _reset_singleton(self):
        from vibeocr.backend.core.singleton_meta import SingletonMeta

        SingletonMeta._instances.pop(MinerUService, None)
        MinerUService._initialized = False
        MinerUService._api_process = None
        MinerUService._api_url = ""
        MinerUService._job_guard = None
        yield
        SingletonMeta._instances.pop(MinerUService, None)
        MinerUService._initialized = False
        MinerUService._api_process = None
        MinerUService._api_url = ""
        MinerUService._job_guard = None

    def test_remote_init_does_not_start_local_api(self, isolated_connection):
        readiness.configure_connection(_remote_connection())
        with (
            patch.object(MinerUService, "_ensure_api_running") as ensure,
            patch.object(MinerUService, "_start_api") as start,
        ):
            MinerUService()
        ensure.assert_not_called()
        start.assert_not_called()

    def test_remote_call_uses_configured_endpoint_and_key(
        self, isolated_connection, monkeypatch
    ):
        middle = {"schema": "docvortex.middle", "schema_version": "2.0", "pages": []}
        recording = _RecordingClient(
            result=MineruDocument(
                markdown="",
                structured_content={"pages": []},
                middle_json=middle,
                archive=b"",
            )
        )
        created = []

        def fake_client(url, *, timeout, api_key=None, transport=None):
            recording.url, recording.api_key = url, api_key
            created.append((url, api_key))
            return recording

        monkeypatch.setattr(
            "vibeocr.backend.services.mineru_service.MineruApiClient", fake_client
        )
        readiness.configure_connection(_remote_connection())
        service = MinerUService.__new__(MinerUService)
        service._call_api(b"pdf", "input.pdf", MineruConfig(tier=MineruTier.BASIC))
        assert created == [(REMOTE_URL, REMOTE_KEY)]
        assert recording.api_key == REMOTE_KEY

    def test_remote_failure_never_falls_back_to_local(
        self, isolated_connection, monkeypatch
    ):
        error = MineruApiError("MinerU GET /v1/uploads: HTTP 503")
        recording = _RecordingClient(error=error)

        def fake_client(url, *, timeout, api_key=None, transport=None):
            return recording

        monkeypatch.setattr(
            "vibeocr.backend.services.mineru_service.MineruApiClient", fake_client
        )
        readiness.configure_connection(_remote_connection())
        service = MinerUService.__new__(MinerUService)
        with (
            patch.object(MinerUService, "_ensure_api_running") as ensure,
            patch.object(MinerUService, "shutdown") as shutdown,
            pytest.raises(MineruApiError),
        ):
            service._call_api(b"pdf", "input.pdf", MineruConfig(tier=MineruTier.BASIC))
        ensure.assert_not_called()
        shutdown.assert_not_called()
        assert recording.calls  # 远程端点确实被调用过一次后失败

    def test_local_mode_still_ensures_local_api(self, isolated_connection):
        with patch.object(MinerUService, "_ensure_api_running") as ensure:
            MinerUService()
        ensure.assert_called_once()

    @staticmethod
    def _readiness_document() -> MineruDocument:
        """构造能通过 project_document 页/块身份校验的最小真实输出。

        middle（source）与 structured 两侧的 page_idx、block 数量、
        block type 必须一一对应；text 块正文取自 middle 的 content。
        """
        structured = {
            "pages": [
                {
                    "page_idx": 0,
                    "blocks": [
                        {
                            "type": "text",
                            "bbox": [0.1, 0.1, 0.9, 0.2],
                            "content": "remote readiness text",
                        }
                    ],
                }
            ]
        }
        middle = {
            "schema": "docvortex.middle",
            "schema_version": "2.0",
            "pages": [
                {
                    "page_idx": 0,
                    "blocks": [{"type": "text", "content": "remote readiness text"}],
                }
            ],
        }
        return MineruDocument("", structured, middle, b"")

    def test_remote_prepare_validates_each_tier_through_remote(
        self, isolated_connection, monkeypatch
    ):
        document = self._readiness_document()
        calls = []

        def fake_call_api(
            data, filename, options=None, *, files=None, cancelled=lambda: False
        ):
            calls.append((options.tier, options.ocr_mode))
            return {filename: document}

        readiness.configure_connection(_remote_connection())
        service = MinerUService.__new__(MinerUService)
        monkeypatch.setattr(service, "_call_api", fake_call_api)
        service.prepare()
        assert {tier for tier, _ in calls} == set(MineruTier)
        # prepare 只验证真实解析，不能仅凭 health 200 宣称就绪。
        for tier in MineruTier:
            readiness.require_ready(tier)

    def test_remote_prepare_failure_marks_only_failing_tier(
        self, isolated_connection, monkeypatch
    ):
        document = self._readiness_document()

        def fake_call_api(
            data, filename, options=None, *, files=None, cancelled=lambda: False
        ):
            if options.tier is MineruTier.BASIC:
                raise MineruApiError("MinerU POST /v1/parse/jobs: HTTP 500")
            return {filename: document}

        readiness.configure_connection(_remote_connection())
        service = MinerUService.__new__(MinerUService)
        monkeypatch.setattr(service, "_call_api", fake_call_api)
        with pytest.raises(MineruApiError):
            service.prepare()
        readiness.require_ready(MineruTier.FLASH)
        with pytest.raises(MineruConfigError):
            readiness.require_ready(MineruTier.BASIC)

    def test_prepare_attributes_observations_to_parsing_connection(
        self, isolated_connection, monkeypatch
    ):
        """无租约直接服务路径下的中途切换：观察只归属旧端点，不污染新连接。

        生产路径由 adapter 租约拒绝切换期间的连接变更；本回归锁定即使
        缺少租约（直接调用 prepare），迟到的旧结果也不得记入新连接。
        """
        remote_a = readiness.connection_from_extra(
            {
                "mineru_connection": {
                    "mode": "remote",
                    "api_url": "https://a.example.com",
                }
            }
        )
        remote_b = readiness.connection_from_extra(
            {
                "mineru_connection": {
                    "mode": "remote",
                    "api_url": "https://b.example.com",
                }
            }
        )
        document = self._readiness_document()
        real_mark = readiness.mark_executed

        def flip_on_flash(tier, *, connection=None):
            if tier is MineruTier.FLASH:
                # 模拟 FLASH 观察落地前连接被切到 B。
                readiness.configure_connection(remote_b)
            real_mark(tier, connection=connection)

        readiness.configure_connection(remote_a)
        monkeypatch.setattr(readiness, "mark_executed", flip_on_flash)
        service = MinerUService.__new__(MinerUService)
        monkeypatch.setattr(
            service,
            "_call_api",
            lambda data, filename, options=None, **kwargs: {filename: document},
        )
        service.prepare()
        # FLASH 的观察归属 A（解析时的入口快照），对当前 B 不生效。
        with pytest.raises(MineruConfigError):
            readiness.require_ready(MineruTier.FLASH)
        # 切换之后开始解析的 tier 归属 B，正常计入。
        readiness.require_ready(MineruTier.BASIC)
        readiness.require_ready(MineruTier.STANDARD)


# ---------------------------------------------------------------------------
# MineruApiClient：Bearer、重定向、错误脱敏
# ---------------------------------------------------------------------------


class RemoteServer:
    """复用 mineru4 V1 上传/jobs/files 流程的最小自部署服务器假件。

    REMOTE_URL 带反代路径前缀 ``/selfhosted``：假件先剥离该前缀再匹配
    路由，从而锁定客户端把前缀保留在每个请求上（反代部署契约）。
    """

    _PREFIX = "/selfhosted"

    def __init__(self, *, redirect_to=None, fail_with=None):
        self.redirect_to = redirect_to
        self.fail_with = fail_with
        self.requests = []
        self.middle = {
            "schema": "docvortex.middle",
            "schema_version": "2.0",
            "pages": [],
        }
        self.artifacts = {
            "markdown": b"",
            "middle_json": json.dumps(self.middle).encode(),
            "structured_content": b'{"pages": []}',
            "zip": b"zip",
        }

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        assert path.startswith(self._PREFIX), path
        path = path[len(self._PREFIX) :]
        if self.redirect_to is not None and path == "/v1/uploads":
            return httpx.Response(302, headers={"Location": self.redirect_to})
        if self.fail_with is not None and path == "/v1/uploads":
            return httpx.Response(
                self.fail_with, text=f"echo {request.headers.get('authorization')}"
            )
        if request.method == "DELETE" or path.endswith("/cancel"):
            return httpx.Response(200, json={})
        if path == "/v1/uploads":
            return httpx.Response(200, json={"id": "upload-1"})
        if path == "/v1/uploads/upload-1/content":
            return httpx.Response(200, json={})
        if path == "/v1/uploads/upload-1/complete":
            return httpx.Response(200, json={"file": {"id": "input-1"}})
        if path == "/v1/parse/jobs":
            return httpx.Response(
                200, json={"job_id": "job-1", "status": "queued", "tier": "basic"}
            )
        if path == "/v1/parse/jobs/job-1":
            return httpx.Response(
                200,
                json={
                    "job_id": "job-1",
                    "status": "completed",
                    "tier": "basic",
                    "files": [
                        {
                            "name": "sample.pdf",
                            "status": "completed",
                            "output_files": {
                                kind: {"file_id": kind, "bytes": len(data)}
                                for kind, data in self.artifacts.items()
                            },
                        }
                    ],
                },
            )
        if path.startswith("/v1/files/"):
            return httpx.Response(200, content=self.artifacts[path.split("/")[3]])
        raise AssertionError(path)


class TestRemoteTransportSecurity:
    def test_bearer_sent_on_every_request(self):
        server = RemoteServer()
        result = MineruApiClient(
            REMOTE_URL,
            transport=httpx.MockTransport(server),
            api_key=REMOTE_KEY,
        ).parse([("sample.pdf", b"pdf")], MineruConfig(tier=MineruTier.BASIC))
        assert isinstance(result["sample.pdf"], MineruDocument)
        assert server.requests, "requests were issued"
        assert all(
            request.headers.get("authorization") == f"Bearer {REMOTE_KEY}"
            for request in server.requests
        )

    def test_empty_key_sends_no_authorization(self):
        server = RemoteServer()
        MineruApiClient(REMOTE_URL, transport=httpx.MockTransport(server)).parse(
            [("sample.pdf", b"pdf")], MineruConfig(tier=MineruTier.BASIC)
        )
        assert all(
            request.headers.get("authorization") is None for request in server.requests
        )

    def test_redirect_is_never_followed_even_same_origin(self):
        server = RemoteServer(redirect_to=f"{REMOTE_URL}/elsewhere")
        with pytest.raises(MineruApiError, match="redirect rejected"):
            MineruApiClient(
                REMOTE_URL,
                transport=httpx.MockTransport(server),
                api_key=REMOTE_KEY,
            ).parse([("sample.pdf", b"pdf")], MineruConfig(tier=MineruTier.BASIC))
        # 只发出了被拒的初始请求；没有任何后续请求被重定向牵引。
        assert len(server.requests) == 1

    def test_cross_origin_redirect_is_rejected_without_location_leak(self):
        server = RemoteServer(redirect_to="https://attacker.example/steal")
        with pytest.raises(MineruApiError) as caught:
            MineruApiClient(
                REMOTE_URL, transport=httpx.MockTransport(server), api_key=REMOTE_KEY
            ).parse([("sample.pdf", b"pdf")], MineruConfig(tier=MineruTier.BASIC))
        message = str(caught.value)
        assert "attacker.example" not in message
        assert REMOTE_URL not in message

    def test_error_message_does_not_echo_url_key_or_server_body(self):
        server = RemoteServer(fail_with=500)
        with pytest.raises(MineruApiError) as caught:
            MineruApiClient(
                REMOTE_URL, transport=httpx.MockTransport(server), api_key=REMOTE_KEY
            ).parse([("sample.pdf", b"pdf")], MineruConfig(tier=MineruTier.BASIC))
        message = str(caught.value)
        assert "HTTP 500" in message
        assert REMOTE_KEY not in message
        assert "Bearer" not in message
        assert "echo" not in message
