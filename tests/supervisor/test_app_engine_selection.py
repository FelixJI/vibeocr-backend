"""OCR engine selection 的 HTTP 边界测试（submit seam + ready catalog）。

覆盖计划 §B1/§3.2 的 wire 语义：

* engine 只在 OCR pipeline 上合法，未知值 fail closed。
* 不可用/需准备返回 426/428 协议错误并回显可选引擎，绝不静默切换。
* ready envelope 的 ocr.engine-selection.v1 descriptor 携带真实探针目录，
  无真实推理后端时目录诚实标记全部不可用。
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from vibeocr.backend.supervisor.app import create_app
from vibeocr.backend.supervisor.bootstrap import generate_session_token, new_instance_id
from vibeocr.backend.supervisor.composition import build_supervisor
from vibeocr.backend.supervisor.inference.budgets import AdapterCapability
from vibeocr.backend.supervisor.inference.ocr_engines import (
    EngineAvailability,
    EngineDescriptor,
    OcrEngineRegistry,
)
from vibeocr.backend.supervisor.module import SupervisorModule, SupervisorOptions
from vibeocr.runtime_contracts.dtos import OcrEngine


class _FakeOcrEngine:
    def __init__(
        self,
        engine_id: OcrEngine,
        availability: EngineAvailability = EngineAvailability.READY,
        *,
        required_component: str | None = None,
    ) -> None:
        self.engine_id = engine_id
        self.availability = availability
        self.required_component = required_component

    def descriptor(self) -> EngineDescriptor:
        return EngineDescriptor(
            engine_id=self.engine_id,
            availability=self.availability,
            included_in_base=self.engine_id is not OcrEngine.PADDLEOCR,
            required_component=self.required_component,
            reason_code=None
            if self.availability is EngineAvailability.READY
            else "engine_not_installed",
        )

    def capabilities(self, options: Any | None = None) -> AdapterCapability:
        del options
        return AdapterCapability(name="OCR", real_batch=False, max_compute_batch=1)

    def recognize_many(
        self,
        items: list[Any],
        *,
        options: Any | None = None,
        compute_batch: Any | None = None,
    ) -> list[dict[str, Any]]:
        return [{"engine": self.engine_id.value} for _ in items]

    def configure_settings(self, snapshot: Any) -> None:
        del snapshot


class _CapturingExecutor:
    """记录 job pipeline（含 engine）并把 job 立即置为完成。"""

    def __init__(self) -> None:
        self.pipelines: list[Any] = []

    def execute(self, record: Any, staged: Any) -> None:
        from vibeocr.runtime_contracts import ItemState, JobState

        self.pipelines.append(record.pipeline)
        record.transition(JobState.RUNNING)
        for item in record.items:
            record.commit_item_success(
                item.item_id, payload_type="ocr.v1", payload={"text": "ok"}
            )
        record.transition(JobState.COMPLETED)
        del ItemState

    def cancel_mode_for(self, record: Any) -> Any:
        from vibeocr.runtime_contracts import CancelMode

        return CancelMode.COOPERATIVE

    def residency_status(self) -> Any:
        from vibeocr.runtime_contracts import ResidencyStatus

        return ResidencyStatus()

    def release_idle(self, pipeline: str | None = None) -> Any:
        return self.residency_status()

    def preload(self, pipelines: tuple[str, ...]) -> Any:
        return self.residency_status()

    def configure_settings(self, snapshot: Any) -> None:
        del snapshot

    def close(self) -> None:
        return


def _module(
    tmp_path: Any, engines: list[_FakeOcrEngine] | None
) -> tuple[SupervisorModule, _CapturingExecutor]:
    executor = _CapturingExecutor()
    module = SupervisorModule(
        options=SupervisorOptions(instance_id=new_instance_id()),
        stager_root=tmp_path / "staging",
        executor=executor,
        engine_registry=OcrEngineRegistry(engines) if engines is not None else None,
    )
    return module, executor


def _client(app: Any, token: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1",
        headers={"Authorization": f"Bearer {token}"},
    )


def _manifest(
    engine: str | None = None,
    pipeline: str = "OCR",
) -> str:
    pipeline_obj: dict[str, Any] = {
        "pipeline_id": pipeline,
        "options_version": 1,
        "options": {},
    }
    if engine is not None:
        pipeline_obj["engine"] = engine
    return json.dumps(
        {
            "schema_version": 2,
            "request_id": "r-1",
            "kind": "recognition",
            "priority": "interactive",
            "pipeline": pipeline_obj,
            "items": [
                {
                    "client_item_key": "k",
                    "ordinal": 0,
                    "display_name": "a.png",
                    "source": {"type": "upload.v1", "attachment": "f"},
                }
            ],
        }
    )


@pytest.fixture()
def engines() -> list[_FakeOcrEngine]:
    return [
        _FakeOcrEngine(OcrEngine.RAPIDOCR),
        _FakeOcrEngine(OcrEngine.WINDOWS, availability=EngineAvailability.UNAVAILABLE),
        _FakeOcrEngine(
            OcrEngine.PADDLEOCR,
            availability=EngineAvailability.PREPARATION_REQUIRED,
            required_component="paddleocr-cpu",
        ),
    ]


async def _submit(http: httpx.AsyncClient, manifest: str) -> httpx.Response:
    return await http.post(
        "/v2/jobs",
        data={"manifest": manifest},
        files={"f": ("a.png", b"png-bytes", "image/png")},
    )


class TestSubmitEngineSelection:
    async def test_explicit_engine_flows_into_job(
        self, tmp_path: Any, engines: list[_FakeOcrEngine]
    ) -> None:
        module, executor = _module(tmp_path, engines)
        token = generate_session_token()
        async with _client(create_app(module, token), token) as http:
            resp = await _submit(http, _manifest(engine="rapidocr"))
        assert resp.status_code == 200
        assert executor.pipelines[0].engine is OcrEngine.RAPIDOCR

    async def test_omitted_engine_defaults_to_none_on_dto(
        self, tmp_path: Any, engines: list[_FakeOcrEngine]
    ) -> None:
        module, executor = _module(tmp_path, engines)
        token = generate_session_token()
        async with _client(create_app(module, token), token) as http:
            resp = await _submit(http, _manifest())
        assert resp.status_code == 200
        assert executor.pipelines[0].engine is None

    async def test_unknown_engine_value_fails_closed_400(
        self, tmp_path: Any, engines: list[_FakeOcrEngine]
    ) -> None:
        module, _ = _module(tmp_path, engines)
        token = generate_session_token()
        async with _client(create_app(module, token), token) as http:
            resp = await _submit(http, _manifest(engine="paddle"))
        assert resp.status_code == 400
        body = resp.json()
        assert body["code"] == "OCR_ENGINE_UNKNOWN"
        assert body["detail"]["engine"] == "paddle"

    async def test_engine_on_non_ocr_pipeline_rejected_400(
        self, tmp_path: Any, engines: list[_FakeOcrEngine]
    ) -> None:
        module, _ = _module(tmp_path, engines)
        token = generate_session_token()
        async with _client(create_app(module, token), token) as http:
            resp = await _submit(
                http, _manifest(engine="rapidocr", pipeline="TABLE_RECOGNITION")
            )
        assert resp.status_code == 400
        assert resp.json()["code"] == "OCR_ENGINE_NOT_VALID_FOR_PIPELINE"

    async def test_unavailable_engine_rejected_426_with_selectable(
        self, tmp_path: Any, engines: list[_FakeOcrEngine]
    ) -> None:
        module, executor = _module(tmp_path, engines)
        token = generate_session_token()
        async with _client(create_app(module, token), token) as http:
            resp = await _submit(http, _manifest(engine="windows"))
        assert resp.status_code == 426, resp.text
        body = resp.json()
        assert body["code"] == "OCR_ENGINE_UNAVAILABLE"
        assert body["detail"]["selectable_engines"] == ["rapidocr"]
        # 未静默切换：job 从未进入 executor。
        assert executor.pipelines == []

    async def test_preparation_required_rejected_428(
        self, tmp_path: Any, engines: list[_FakeOcrEngine]
    ) -> None:
        module, _ = _module(tmp_path, engines)
        token = generate_session_token()
        async with _client(create_app(module, token), token) as http:
            resp = await _submit(http, _manifest(engine="paddleocr"))
        assert resp.status_code == 428
        body = resp.json()
        assert body["code"] == "OCR_ENGINE_PREPARATION_REQUIRED"
        assert body["detail"]["required_component"] == "paddleocr-cpu"

    async def test_job_recognition_mode_extension_is_strictly_rejected(
        self, tmp_path: Any, engines: list[_FakeOcrEngine]
    ) -> None:
        module, executor = _module(tmp_path, engines)
        token = generate_session_token()
        payload = json.loads(_manifest(engine="rapidocr"))
        payload["pipeline"]["recognition_mode"] = "rapid_text"
        async with _client(create_app(module, token), token) as http:
            rejected = await _submit(http, json.dumps(payload))
            accepted = await _submit(http, _manifest(engine="rapidocr"))

        assert rejected.status_code == 400
        assert rejected.json()["code"] == "VALIDATION_ERROR"
        assert accepted.status_code == 200
        assert executor.pipelines[0].engine is OcrEngine.RAPIDOCR

    async def test_default_engine_unavailable_fails_closed_426(
        self, tmp_path: Any
    ) -> None:
        # rapidocr 缺席时缺省请求同样 fail closed，不落到其他 ready 引擎。
        engines = [
            _FakeOcrEngine(
                OcrEngine.RAPIDOCR, availability=EngineAvailability.UNAVAILABLE
            ),
            _FakeOcrEngine(OcrEngine.WINDOWS),
        ]
        module, executor = _module(tmp_path, engines)
        token = generate_session_token()
        async with _client(create_app(module, token), token) as http:
            resp = await _submit(http, _manifest())
        assert resp.status_code == 426
        assert resp.json()["code"] == "OCR_ENGINE_UNAVAILABLE"
        assert resp.json()["detail"]["engine"] == "rapidocr"
        assert executor.pipelines == []

    async def test_specialized_pipeline_uses_mode_registry_availability(
        self, tmp_path: Any, engines: list[_FakeOcrEngine]
    ) -> None:
        module, executor = _module(tmp_path, engines)
        token = generate_session_token()
        async with _client(create_app(module, token), token) as http:
            resp = await _submit(http, _manifest(pipeline="TABLE_RECOGNITION"))

        assert resp.status_code == 426, resp.text
        assert resp.json()["code"] == "RECOGNITION_MODE_UNAVAILABLE"
        assert resp.json()["detail"]["recognition_mode"] == "paddle_table"
        assert executor.pipelines == []

    async def test_no_registry_value_checks_still_apply(self, tmp_path: Any) -> None:
        # 无真实推理后端（Null/fake executor）：engine 值级校验仍然生效。
        module, _ = _module(tmp_path, None)
        token = generate_session_token()
        async with _client(create_app(module, token), token) as http:
            resp = await _submit(http, _manifest(engine="bogus"))
        assert resp.status_code == 400
        assert resp.json()["code"] == "OCR_ENGINE_UNKNOWN"


class TestReadyEngineCatalog:
    async def test_catalog_reflects_live_probes(
        self, tmp_path: Any, engines: list[_FakeOcrEngine]
    ) -> None:
        module, _ = _module(tmp_path, engines)
        token = generate_session_token()
        async with _client(create_app(module, token), token) as http:
            resp = await http.get("/v2/health")
        assert resp.status_code == 200
        descriptors = resp.json()["capability_descriptors"]
        engine_descriptor = next(
            d for d in descriptors if d["name"] == "ocr.engine-selection.v1"
        )
        catalog = engine_descriptor["ocr_engine_catalog"]
        entries = {entry["id"]: entry for entry in catalog["engines"]}
        assert set(entries) == {"rapidocr", "windows", "paddleocr"}
        assert entries["rapidocr"]["availability"] == "ready"
        assert entries["rapidocr"]["included_in_base"] is True
        assert entries["windows"]["availability"] == "unavailable"
        assert entries["paddleocr"]["availability"] == "preparation_required"
        assert entries["paddleocr"]["required_component"] == "paddleocr-cpu"

    async def test_catalog_without_registry_is_honestly_unavailable(
        self, tmp_path: Any
    ) -> None:
        module, _ = _module(tmp_path, None)
        token = generate_session_token()
        async with _client(create_app(module, token), token) as http:
            resp = await http.get("/v2/health")
        engine_descriptor = next(
            d
            for d in resp.json()["capability_descriptors"]
            if d["name"] == "ocr.engine-selection.v1"
        )
        entries = {
            entry["id"]: entry
            for entry in engine_descriptor["ocr_engine_catalog"]["engines"]
        }
        assert set(entries) == {"rapidocr", "windows", "paddleocr"}
        assert all(entry["availability"] == "unavailable" for entry in entries.values())

    async def test_successful_maintenance_refreshes_cached_engine_catalog(
        self, tmp_path: Any, engines: list[_FakeOcrEngine]
    ) -> None:
        class _SuccessfulMaintenance:
            def execute(self, **kwargs: Any) -> dict[str, Any]:
                return {
                    "schema_version": 2,
                    "operation_id": "ensure-1",
                    "snapshot": {
                        "operation_id": "ensure-1",
                        "sequence": 1,
                        "operation": "ensure",
                        "operation_state": "succeeded",
                        "phase": "commit_runtime",
                        "profile_id": "win-x64-cpu",
                        "updated_at": "2026-08-17T00:00:00Z",
                    },
                    "negotiated_capabilities": list(kwargs["required_capabilities"]),
                }

        module, _ = _module(tmp_path, engines)
        paddle = next(
            engine for engine in engines if engine.engine_id is OcrEngine.PADDLEOCR
        )
        token = generate_session_token()
        app = create_app(
            module,
            token,
            runtime_control=_SuccessfulMaintenance(),  # type: ignore[arg-type]
        )

        async with _client(app, token) as http:
            before = await http.get("/v2/health")
            paddle.availability = EngineAvailability.READY
            cached = await http.get("/v2/health")
            maintained = await http.post(
                "/v2/runtime/maintenance",
                json={
                    "operation_id": "ensure-1",
                    "operation": "ensure",
                    "profile_id": "win-x64-cpu",
                    "component_ids": [],
                    "required_capabilities": [],
                },
            )
            refreshed = await http.get("/v2/health")

        def paddle_availability(response: httpx.Response) -> str:
            descriptor = next(
                item
                for item in response.json()["capability_descriptors"]
                if item["name"] == "ocr.engine-selection.v1"
            )
            entry = next(
                item
                for item in descriptor["ocr_engine_catalog"]["engines"]
                if item["id"] == "paddleocr"
            )
            return str(entry["availability"])

        assert paddle_availability(before) == "preparation_required"
        assert paddle_availability(cached) == "preparation_required"
        assert maintained.status_code == 200
        assert paddle_availability(refreshed) == "ready"


async def test_legacy_ocr_preload_is_rejected_instead_of_warming_default_engine(
    tmp_path: Any,
) -> None:
    rapid = _FakeOcrEngine(OcrEngine.RAPIDOCR)
    module, handle = build_supervisor(
        instance_id=new_instance_id(),
        stager_root=tmp_path / "staging",
        engine_registry=OcrEngineRegistry([rapid]),
        use_real_paddle=False,
        use_mineru=False,
    )
    async with _client(create_app(module, handle.token), handle.token) as http:
        response = await http.post("/v2/runtime/preload", json={"pipelines": ["OCR"]})
    assert response.status_code == 400, response.text
    assert response.json()["code"] == "RECOGNITION_MODE_PIPELINE_MISMATCH"


async def test_preload_and_release_enforce_recognition_mode_lifecycle(
    tmp_path: Any,
) -> None:
    module, executor = _module(tmp_path, None)
    token = generate_session_token()
    app = create_app(module, token)
    async with _client(app, token) as http:
        paddle = await http.post(
            "/v2/runtime/preload",
            json={"pipelines": ["OCR"], "recognition_modes": ["paddle_text"]},
        )
        mineru = await http.post(
            "/v2/runtime/preload",
            json={"pipelines": ["MinerU"], "recognition_modes": ["mineru_document"]},
        )
        rapid_release = await http.post(
            "/v2/runtime/release",
            json={"pipeline": "OCR", "recognition_mode": "rapid_text"},
        )

    assert paddle.status_code == 426, paddle.text
    assert paddle.json()["code"] == "RECOGNITION_MODE_UNAVAILABLE"
    assert mineru.status_code == 400
    assert mineru.json()["code"] == "RECOGNITION_MODE_LIFECYCLE_UNSUPPORTED"
    assert rapid_release.status_code == 400
    assert rapid_release.json()["code"] == ("RECOGNITION_MODE_LIFECYCLE_UNSUPPORTED")
    del executor


async def test_residency_response_keeps_protocol_resource_identity(
    tmp_path: Any,
) -> None:
    from vibeocr.runtime_contracts import ResidencyEntry, ResidencyKind, ResidencyStatus

    module, executor = _module(tmp_path, None)
    executor.residency_status = lambda: ResidencyStatus(  # type: ignore[method-assign]
        entries=(ResidencyEntry(pipeline="MinerU", kind=ResidencyKind.SOFT_TTL),)
    )
    token = generate_session_token()

    async with _client(create_app(module, token), token) as http:
        response = await http.get("/v2/runtime/residency")

    assert response.status_code == 200, response.text
    entry = response.json()["entries"][0]
    assert entry["recognition_mode"] == "mineru_document"
    assert entry["resource_kind"] == "process"
    assert entry["resource_id"] == "mineru-api"
