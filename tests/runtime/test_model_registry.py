"""Model registry:N0 工作包测试(真实 HTTP 下载、中断、retry、断网复用)。"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from vibeocr.backend.model_registry import (
    HuggingFaceAdapter,
    LocalDirectoryAdapter,
    ModelAcquisitionError,
    ModelAsset,
    ModelScopeAdapter,
    acquire_models,
    ensure_mineru_tools_config,
    load_model_assets,
    model_assets_config_path,
    model_source_environment,
)
from vibeocr.backend.runtime_selection import (
    download_source_catalog_payload,
    normalize_download_source_ids,
)

MODEL_ASSET = ModelAsset(
    engine="paddleocr",
    name="PaddlePaddle/PP-OCRv5_server_rec",
    revision="v1",
    files=("inference.pdmodel",),
)


class _FixtureServer:
    """本地真实 HTTP 服务:可注入中断(连接半途关闭)。"""

    def __init__(self, payload: bytes, *, interrupt_after: int | None = None) -> None:
        handler = _make_handler(payload, interrupt_after=interrupt_after)
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self.thread.join(timeout=5)


def _make_handler(payload: bytes, *, interrupt_after: int | None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if interrupt_after is None:
                self.wfile.write(payload)
                return
            self.wfile.write(payload[:interrupt_after])
            self.wfile.flush()
            self.close_connection = True
            self.connection.close()

        def log_message(self, *args: object) -> None:
            return

    return Handler


def test_catalog_declares_model_registry_sources() -> None:
    payload = download_source_catalog_payload()
    by_id = {source["id"]: source for source in payload["sources"]}
    assert by_id["huggingface"] == {
        "kind": "model_registry",
        "id": "huggingface",
        "endpoint": "https://huggingface.co",
    }
    assert by_id["modelscope"]["kind"] == "model_registry"
    ids = normalize_download_source_ids(["tuna-pypi", "huggingface"])
    assert set(ids) == {"tuna-pypi", "huggingface"}


def test_settings_source_selection_still_fails_closed_per_kind() -> None:
    with pytest.raises(Exception, match="kind"):
        normalize_download_source_ids(["huggingface", "modelscope"])
    with pytest.raises(Exception):
        normalize_download_source_ids(["hf-mirror"])


def test_model_assets_manifest_is_explicit_and_validated(tmp_path: Path) -> None:
    assert load_model_assets(None) == ()
    assert load_model_assets(tmp_path / "missing.json") == ()

    manifest = tmp_path / "model-assets.json"
    manifest.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "engine": "mineru",
                        "name": "opendatalab/PDF-Extract-Kit",
                        "revision": "v1",
                        "files": ["models.json"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (assets,) = load_model_assets(manifest)
    assert assets.engine == "mineru"
    assert assets.files == ("models.json",)

    manifest.write_text(
        json.dumps({"assets": [{"engine": "tesseract", "name": "x"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ModelAcquisitionError):
        load_model_assets(manifest)


def test_huggingface_and_modelscope_adapters_download_over_real_http(
    tmp_path: Path,
) -> None:
    payload = b"model-bytes-0123456789"
    server = _FixtureServer(payload)
    try:
        destination = tmp_path / "inference.pdmodel"
        progress_log: list[tuple[int, int]] = []

        class Progress:
            def __init__(self) -> None:
                self.current = 0
                self.total = 0
                self.file_name = ""

            def report(self) -> None:
                progress_log.append((self.current, self.total))

        progress = Progress()
        HuggingFaceAdapter().fetch_file(
            source_id="huggingface",
            endpoint=server.endpoint,
            asset=ModelAsset(
                engine="paddleocr",
                name="PaddlePaddle/PP-OCRv5_server_rec",
                revision="v1",
                files=("inference.pdmodel",),
            ),
            file_name="inference.pdmodel",
            destination=destination,
            progress=progress,  # type: ignore[arg-type]
        )
        assert destination.read_bytes() == payload
        assert progress_log[-1][0] == len(payload)
        assert progress_log[-1][1] == len(payload)

        scope_destination = tmp_path / "models.json"
        ModelScopeAdapter().fetch_file(
            source_id="modelscope",
            endpoint=server.endpoint,
            asset=ModelAsset(
                engine="mineru",
                name="opendatalab/PDF-Extract-Kit",
                revision="v1",
                files=("models.json",),
            ),
            file_name="models.json",
            destination=scope_destination,
            progress=Progress(),  # type: ignore[arg-type]
        )
        assert scope_destination.read_bytes() == payload
    finally:
        server.stop()


def test_interrupted_download_keeps_base_and_staging_clean(tmp_path: Path) -> None:
    payload = b"x" * 4096
    server = _FixtureServer(payload, interrupt_after=512)
    try:
        with pytest.raises(ModelAcquisitionError):
            acquire_models(
                assets=(MODEL_ASSET,),
                source_id="huggingface",
                endpoint=server.endpoint,
                models_root=tmp_path,
            )
        staging = tmp_path / "downloads"
        assert not staging.exists() or not any(
            path.is_file() for path in staging.rglob("*")
        )
        assert not (tmp_path / "paddleocr").exists()
    finally:
        server.stop()


def test_retry_after_failure_succeeds_and_is_atomic(tmp_path: Path) -> None:
    payload = b"y" * 2048
    broken = _FixtureServer(payload, interrupt_after=100)
    good = _FixtureServer(payload)
    try:
        with pytest.raises(ModelAcquisitionError):
            acquire_models(
                assets=(MODEL_ASSET,),
                source_id="huggingface",
                endpoint=broken.endpoint,
                models_root=tmp_path,
            )
        acquired = acquire_models(
            assets=(MODEL_ASSET,),
            source_id="huggingface",
            endpoint=good.endpoint,
            models_root=tmp_path,
        )
        target = acquired["paddleocr/PaddlePaddle/PP-OCRv5_server_rec"]
        assert (target / "inference.pdmodel").read_bytes() == payload
    finally:
        broken.stop()
        good.stop()


def test_offline_reuse_of_existing_models_skips_network(tmp_path: Path) -> None:
    target = tmp_path / "paddleocr" / f"{MODEL_ASSET.name}-{MODEL_ASSET.revision}"
    target.mkdir(parents=True)
    (target / "inference.pdmodel").write_bytes(b"local")

    class FailingAdapter:
        def fetch_file(self, **kwargs: object) -> None:
            raise AssertionError("network must not be touched for local models")

    acquired = acquire_models(
        assets=(MODEL_ASSET,),
        source_id="huggingface",
        endpoint="https://definitely.invalid",
        models_root=tmp_path,
        adapter=FailingAdapter(),  # type: ignore[arg-type]
    )
    assert acquired["paddleocr/PaddlePaddle/PP-OCRv5_server_rec"] == target


def test_local_directory_adapter_serves_fixtures(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    source_root = fixture / "huggingface" / MODEL_ASSET.name / MODEL_ASSET.revision
    source_root.mkdir(parents=True)
    (source_root / "inference.pdmodel").write_bytes(b"fixture-bytes")

    models = tmp_path / "models"
    acquired = acquire_models(
        assets=(MODEL_ASSET,),
        source_id="huggingface",
        endpoint="ignored",
        models_root=models,
        adapter=LocalDirectoryAdapter(fixture),
    )
    target = acquired["paddleocr/PaddlePaddle/PP-OCRv5_server_rec"]
    assert (target / "inference.pdmodel").read_bytes() == b"fixture-bytes"


def test_mineru_config_and_source_environment_are_state_scoped(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    models = tmp_path / "state" / "models"
    environment = model_source_environment(
        source_id="modelscope", state_root=state, models_root=models
    )
    assert environment["PADDLE_PDX_MODEL_SOURCE"] == "modelscope"
    assert environment["MINERU_MODEL_SOURCE"] == "modelscope"
    config_path = Path(environment["MINERU_TOOLS_CONFIG_JSON"])
    assert config_path == state / "config" / "mineru.json"
    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "models-dir": str(models / "mineru")
    }
    # 幂等:已存在的配置不被覆盖。
    config_path.write_text("{}", encoding="utf-8")
    again = model_source_environment(
        source_id=None, state_root=state, models_root=models
    )
    assert "PADDLE_PDX_MODEL_SOURCE" not in again
    assert Path(again["MINERU_TOOLS_CONFIG_JSON"]).read_text(encoding="utf-8") == "{}"
    assert model_assets_config_path(state) == state / "config" / "model-assets.json"
    ensure_mineru_tools_config(tmp_path / "fresh", tmp_path / "fresh-models")
