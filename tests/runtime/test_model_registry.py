"""Model registry:N0 工作包测试(真实 HTTP 下载、中断、retry、断网复用)。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from vibeocr.backend import model_registry as model_registry_module
from vibeocr.backend.model_registry import (
    MODEL_READY_FILENAME,
    HuggingFaceAdapter,
    LocalDirectoryAdapter,
    ModelAcquisitionError,
    ModelAsset,
    ModelFileIntegrity,
    ModelScopeAdapter,
    ResolvedModel,
    ResolvedModelSet,
    acquire_models,
    ensure_mineru_tools_config,
    load_model_assets,
    load_resolved_models,
    local_model_kwargs,
    model_assets_config_path,
    model_source_environment,
)
from vibeocr.backend.runtime_selection import (
    download_source_catalog_payload,
    normalize_download_source_ids,
)

MODEL_ASSET = ModelAsset(
    engine="paddleocr",
    name="pp-ocrv5-server-rec",
    repository="PaddlePaddle/PP-OCRv5_server_rec",
    revision="v1",
    files=("inference.pdmodel",),
    consumer="paddleocr",
    binding_key="text_recognition_model_dir",
)


def _integrity(payload: bytes, path: str = "inference.pdmodel") -> ModelFileIntegrity:
    return ModelFileIntegrity(
        path=path,
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _verified_asset(payload: bytes) -> ModelAsset:
    return replace(MODEL_ASSET, file_integrity=(_integrity(payload),))


class _FixtureServer:
    """本地真实 HTTP 服务:可注入中断(连接半途关闭)。"""

    def __init__(
        self,
        payload: bytes,
        *,
        interrupt_after: int | None = None,
        content_length: bool = True,
    ) -> None:
        handler = _make_handler(
            payload,
            interrupt_after=interrupt_after,
            content_length=content_length,
        )
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


def _make_handler(
    payload: bytes,
    *,
    interrupt_after: int | None,
    content_length: bool,
):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            if content_length:
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
                "schema_version": 1,
                "release_identity": "backend-release",
                "assets": [
                    {
                        "engine": "mineru",
                        "name": "pdf-extract-kit-pipeline",
                        "repository": "opendatalab/PDF-Extract-Kit",
                        "revision": "v1",
                        "files": [
                            {"path": "models.json", "size": 1, "sha256": "0" * 64}
                        ],
                        "consumer": "mineru",
                        "binding_key": "pipeline",
                    },
                    {
                        "engine": "mineru",
                        "name": "mineru-vlm",
                        "repository": "opendatalab/MinerU2.5-2509-1.2B",
                        "revision": "v1",
                        "files": [
                            {"path": "config.json", "size": 1, "sha256": "1" * 64}
                        ],
                        "consumer": "mineru",
                        "binding_key": "vlm",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    assets = load_model_assets(manifest)
    assert {asset.binding_key for asset in assets} == {"pipeline", "vlm"}
    assert assets[0].files == ("models.json",)

    complete_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    missing_vlm = dict(complete_manifest)
    missing_vlm["assets"] = missing_vlm["assets"][:1]
    manifest.write_text(json.dumps(missing_vlm), encoding="utf-8")
    with pytest.raises(ModelAcquisitionError, match="pipeline.*vlm"):
        load_model_assets(manifest)

    manifest.write_text(json.dumps(complete_manifest), encoding="utf-8")

    duplicated = json.loads(manifest.read_text(encoding="utf-8"))
    duplicated["assets"].append(dict(duplicated["assets"][0]))
    manifest.write_text(json.dumps(duplicated), encoding="utf-8")
    with pytest.raises(ModelAcquisitionError, match="unique"):
        load_model_assets(manifest)

    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_identity": "backend-release",
                "assets": [{"engine": "tesseract", "name": "x"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ModelAcquisitionError):
        load_model_assets(manifest)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "../outside"),
        ("name", "owner/model"),
        ("name", "C:/outside"),
        ("name", "safe\\..\\outside"),
        ("repository", "../outside"),
        ("repository", "C:/outside"),
        ("revision", "../outside"),
        ("revision", "CON"),
        ("file", "../sentinel.txt"),
        ("file", "/absolute.bin"),
        ("file", "nested\\escape.bin"),
        ("file", "aux.txt"),
    ],
)
def test_model_assets_manifest_rejects_unsafe_relative_fields(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    declaration = {
        "engine": "paddleocr",
        "name": "pp-ocrv5-server-rec",
        "repository": "PaddlePaddle/PP-OCRv5_server_rec",
        "revision": "v1",
        "files": [{"path": "inference.pdmodel", "size": 1, "sha256": "0" * 64}],
        "consumer": "paddleocr",
        "binding_key": "text_recognition_model_dir",
    }
    if field == "file":
        declaration["files"] = [{"path": value, "size": 1, "sha256": "0" * 64}]
    else:
        declaration[field] = value
    manifest = tmp_path / "model-assets.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_identity": "backend-release",
                "assets": [declaration],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModelAcquisitionError, match="unsafe"):
        load_model_assets(manifest)


def test_acquisition_revalidates_programmatic_assets_before_writing(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    malicious = ModelAsset(
        engine="paddleocr",
        name="safe",
        repository="owner/safe",
        revision="v1",
        files=("../../../../outside/sentinel.txt",),
        consumer="paddleocr",
        binding_key="text_recognition_model_dir",
    )

    class WritingAdapter:
        def fetch_file(self, **kwargs: object) -> None:
            destination = kwargs["destination"]
            assert isinstance(destination, Path)
            destination.write_text("changed", encoding="utf-8")

    with pytest.raises(ModelAcquisitionError, match="unsafe"):
        acquire_models(
            assets=(malicious,),
            release_identity="backend-release",
            source_id="huggingface",
            endpoint="unused",
            models_root=tmp_path / "models",
            adapter=WritingAdapter(),  # type: ignore[arg-type]
        )

    assert sentinel.read_text(encoding="utf-8") == "unchanged"


@pytest.mark.parametrize(
    "file_integrity",
    [
        (),
        (_integrity(b"model", "another.bin"),),
        (ModelFileIntegrity("inference.pdmodel", -1, "0" * 64),),
        (ModelFileIntegrity("inference.pdmodel", 5, "not-a-sha256"),),
    ],
)
def test_acquisition_requires_complete_programmatic_file_integrity(
    tmp_path: Path,
    file_integrity: tuple[ModelFileIntegrity, ...],
) -> None:
    called = False

    class WritingAdapter:
        def fetch_file(self, **kwargs: object) -> None:
            nonlocal called
            called = True
            destination = kwargs["destination"]
            assert isinstance(destination, Path)
            destination.write_bytes(b"model")

    with pytest.raises(ModelAcquisitionError, match="integrity"):
        acquire_models(
            assets=(replace(MODEL_ASSET, file_integrity=file_integrity),),
            release_identity="backend-release",
            source_id="huggingface",
            endpoint="unused",
            models_root=tmp_path / "models",
            adapter=WritingAdapter(),  # type: ignore[arg-type]
        )

    assert called is False


def test_acquisition_rejects_reparse_parent_without_touching_target(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = models / "paddleocr"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(outside)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            pytest.skip("directory symlink/junction is unavailable")

    class WritingAdapter:
        def fetch_file(self, **kwargs: object) -> None:
            destination = kwargs["destination"]
            assert isinstance(destination, Path)
            destination.write_bytes(b"model")

    with pytest.raises(ModelAcquisitionError, match="reparse"):
        acquire_models(
            assets=(_verified_asset(b"model"),),
            release_identity="backend-release",
            source_id="huggingface",
            endpoint="unused",
            models_root=models,
            adapter=WritingAdapter(),  # type: ignore[arg-type]
        )

    assert list(outside.iterdir()) == []


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
                name="pp-ocrv5-server-rec",
                repository="PaddlePaddle/PP-OCRv5_server_rec",
                revision="v1",
                files=("inference.pdmodel",),
                consumer="paddleocr",
                binding_key="text_recognition_model_dir",
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
                name="pdf-extract-kit",
                repository="opendatalab/PDF-Extract-Kit",
                revision="v1",
                files=("models.json",),
                consumer="mineru",
                binding_key="models_dir",
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
                assets=(_verified_asset(payload),),
                release_identity="backend-release",
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


def test_download_without_content_length_checks_cancel_per_chunk(
    tmp_path: Path,
) -> None:
    payload = b"z" * (192 * 1024)
    server = _FixtureServer(payload, content_length=False)
    checks = 0
    totals: list[int] = []

    class Cancelled(RuntimeError):
        pass

    def cancel_check() -> None:
        nonlocal checks
        checks += 1
        if checks >= 4:
            raise Cancelled("cancelled during chunked download")

    try:
        with pytest.raises(Cancelled, match="chunked download"):
            acquire_models(
                assets=(_verified_asset(payload),),
                release_identity="backend-release",
                source_id="huggingface",
                endpoint=server.endpoint,
                models_root=tmp_path,
                cancel_check=cancel_check,
                progress_listener=lambda progress: totals.append(progress.total),
            )
        assert checks >= 4
        assert totals and set(totals) == {0}
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
                assets=(_verified_asset(payload),),
                release_identity="backend-release",
                source_id="huggingface",
                endpoint=broken.endpoint,
                models_root=tmp_path,
            )
        acquired = acquire_models(
            assets=(_verified_asset(payload),),
            release_identity="backend-release",
            source_id="huggingface",
            endpoint=good.endpoint,
            models_root=tmp_path,
        )
        target = acquired["paddleocr/pp-ocrv5-server-rec"]
        assert (target / "inference.pdmodel").read_bytes() == payload
    finally:
        broken.stop()
        good.stop()


def test_offline_reuse_of_existing_models_skips_network_and_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "fixture"
    source = fixture / "huggingface" / MODEL_ASSET.repository / MODEL_ASSET.revision
    source.mkdir(parents=True)
    (source / "inference.pdmodel").write_bytes(b"local")
    acquired = acquire_models(
        assets=(_verified_asset(b"local"),),
        release_identity="backend-release",
        source_id="huggingface",
        endpoint="unused",
        models_root=tmp_path / "models",
        adapter=LocalDirectoryAdapter(fixture),
    )
    target = acquired["paddleocr/pp-ocrv5-server-rec"]

    class FailingAdapter:
        def fetch_file(self, **kwargs: object) -> None:
            raise AssertionError("network must not be touched for local models")

    monkeypatch.setattr(
        model_registry_module,
        "_sha256_file",
        lambda _path: (_ for _ in ()).throw(AssertionError("must not rehash")),
    )
    acquired = acquire_models(
        assets=(_verified_asset(b"local"),),
        release_identity="backend-release",
        source_id="huggingface",
        endpoint="https://definitely.invalid",
        models_root=tmp_path / "models",
        adapter=FailingAdapter(),  # type: ignore[arg-type]
    )
    assert acquired["paddleocr/pp-ocrv5-server-rec"] == target


def test_explicit_integrity_verification_repairs_same_size_corruption(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    source = fixture / "huggingface" / MODEL_ASSET.repository / MODEL_ASSET.revision
    source.mkdir(parents=True)
    (source / "inference.pdmodel").write_bytes(b"local")
    models = tmp_path / "models"
    adapter = LocalDirectoryAdapter(fixture)
    acquired = acquire_models(
        assets=(_verified_asset(b"local"),),
        release_identity="backend-release",
        source_id="huggingface",
        endpoint="unused",
        models_root=models,
        adapter=adapter,
    )
    target = acquired["paddleocr/pp-ocrv5-server-rec"]
    (target / "inference.pdmodel").write_bytes(b"wrong")

    acquire_models(
        assets=(_verified_asset(b"local"),),
        release_identity="backend-release",
        source_id="huggingface",
        endpoint="unused",
        models_root=models,
        adapter=adapter,
        verify_existing_integrity=True,
    )

    assert (target / "inference.pdmodel").read_bytes() == b"local"


def test_missing_extra_or_mismatched_marker_repairs_without_foreign_cleanup(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    source = fixture / "huggingface" / MODEL_ASSET.repository / MODEL_ASSET.revision
    source.mkdir(parents=True)
    (source / "inference.pdmodel").write_bytes(b"verified")
    models = tmp_path / "models"
    foreign = models / "downloads" / "foreign-operation" / "sentinel.txt"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("owned elsewhere", encoding="utf-8")
    adapter = LocalDirectoryAdapter(fixture)

    acquired = acquire_models(
        assets=(_verified_asset(b"verified"),),
        release_identity="backend-release",
        source_id="huggingface",
        endpoint="unused",
        models_root=models,
        adapter=adapter,
    )
    target = acquired["paddleocr/pp-ocrv5-server-rec"]
    marker = target / MODEL_READY_FILENAME
    assert json.loads(marker.read_text(encoding="utf-8"))["release_identity"] == (
        "backend-release"
    )

    (target / "inference.pdmodel").unlink()
    acquire_models(
        assets=(_verified_asset(b"verified"),),
        release_identity="backend-release",
        source_id="huggingface",
        endpoint="unused",
        models_root=models,
        adapter=adapter,
    )
    (target / "extra.bin").write_bytes(b"conflict")
    acquire_models(
        assets=(_verified_asset(b"verified"),),
        release_identity="backend-release",
        source_id="huggingface",
        endpoint="unused",
        models_root=models,
        adapter=adapter,
    )
    marker.write_text("{}", encoding="utf-8")
    acquire_models(
        assets=(_verified_asset(b"verified"),),
        release_identity="backend-release",
        source_id="huggingface",
        endpoint="unused",
        models_root=models,
        adapter=adapter,
    )

    assert (target / "inference.pdmodel").read_bytes() == b"verified"
    assert not (target / "extra.bin").exists()
    assert foreign.read_text(encoding="utf-8") == "owned elsewhere"


def test_local_directory_adapter_serves_fixtures(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    source_root = (
        fixture / "huggingface" / MODEL_ASSET.repository / MODEL_ASSET.revision
    )
    source_root.mkdir(parents=True)
    (source_root / "inference.pdmodel").write_bytes(b"fixture-bytes")

    models = tmp_path / "models"
    acquired = acquire_models(
        assets=(_verified_asset(b"fixture-bytes"),),
        release_identity="backend-release",
        source_id="huggingface",
        endpoint="ignored",
        models_root=models,
        adapter=LocalDirectoryAdapter(fixture),
    )
    target = acquired["paddleocr/pp-ocrv5-server-rec"]
    assert (target / "inference.pdmodel").read_bytes() == b"fixture-bytes"
    assert isinstance(acquired, ResolvedModelSet)
    assert acquired.binding_kwargs("paddleocr") == {
        "text_recognition_model_dir": str(target)
    }


def test_resolved_model_environment_binds_paddle_locally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = tmp_path / "fixture"
    source = fixture / "huggingface" / MODEL_ASSET.repository / MODEL_ASSET.revision
    source.mkdir(parents=True)
    (source / "inference.pdmodel").write_bytes(b"fixture-bytes")
    resolved = acquire_models(
        assets=(_verified_asset(b"fixture-bytes"),),
        release_identity="backend-release",
        source_id="huggingface",
        endpoint="unused",
        models_root=tmp_path / "state" / "models",
        adapter=LocalDirectoryAdapter(fixture),
    )
    environment = model_source_environment(
        source_id="huggingface",
        state_root=tmp_path / "state",
        models_root=tmp_path / "state" / "models",
        resolved_models=resolved,
    )
    monkeypatch.setenv(
        "VIBEOCR_RESOLVED_MODELS",
        environment["VIBEOCR_RESOLVED_MODELS"],
    )

    assert local_model_kwargs("paddleocr") == resolved.binding_kwargs("paddleocr")


def test_persisted_resolved_models_are_revalidated_before_launch(
    tmp_path: Path,
) -> None:
    payload = b"fixture-bytes"
    asset = _verified_asset(payload)
    fixture = tmp_path / "fixture"
    source = fixture / "huggingface" / asset.repository / asset.revision
    source.mkdir(parents=True)
    (source / asset.files[0]).write_bytes(payload)
    models_root = tmp_path / "state" / "models"
    resolved = acquire_models(
        assets=(asset,),
        release_identity="backend-release",
        source_id="huggingface",
        endpoint="unused",
        models_root=models_root,
        adapter=LocalDirectoryAdapter(fixture),
    )
    environment = model_source_environment(
        source_id="huggingface",
        state_root=tmp_path / "state",
        models_root=models_root,
        resolved_models=resolved,
    )
    binding_path = Path(environment["VIBEOCR_RESOLVED_MODELS"])

    restored = load_resolved_models(
        binding_path,
        assets=(asset,),
        expected_release_identity="backend-release",
        models_root=models_root,
    )
    assert restored == resolved

    document = json.loads(binding_path.read_text(encoding="utf-8"))
    document["release_identity"] = "stale-release"
    binding_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ModelAcquisitionError, match="release identity"):
        load_resolved_models(
            binding_path,
            assets=(asset,),
            expected_release_identity="backend-release",
            models_root=models_root,
        )


def test_persisted_resolved_models_reject_unbound_paths(tmp_path: Path) -> None:
    asset = _verified_asset(b"fixture-bytes")
    models_root = tmp_path / "state" / "models"
    target = models_root / asset.engine / asset.target_dirname
    target.mkdir(parents=True)
    binding_path = tmp_path / "state" / "config" / "resolved-models.json"
    binding_path.parent.mkdir(parents=True)
    binding_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_identity": "backend-release",
                "consumers": {
                    "paddleocr": {
                        "text_recognition_model_dir": str(tmp_path / "elsewhere")
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModelAcquisitionError, match="binding"):
        load_resolved_models(
            binding_path,
            assets=(asset,),
            expected_release_identity="backend-release",
            models_root=models_root,
        )


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
        "models-dir": {
            "pipeline": str(models / "mineru" / "pipeline"),
            "vlm": str(models / "mineru" / "vlm"),
        }
    }
    # Portable 移动后必须从当前 root 重建，不能保留旧绝对路径。
    config_path.write_text("{}", encoding="utf-8")
    moved_state = tmp_path / "moved" / "state"
    moved_models = moved_state / "models"
    stale_config = moved_state / "config" / "mineru.json"
    stale_config.parent.mkdir(parents=True)
    stale_config.write_text(
        json.dumps(
            {
                "models-dir": {
                    "pipeline": str(models / "mineru" / "pipeline"),
                    "vlm": str(models / "mineru" / "vlm"),
                }
            }
        ),
        encoding="utf-8",
    )
    again = model_source_environment(
        source_id=None,
        state_root=moved_state,
        models_root=moved_models,
    )
    assert "PADDLE_PDX_MODEL_SOURCE" not in again
    moved_config = Path(again["MINERU_TOOLS_CONFIG_JSON"])
    assert json.loads(moved_config.read_text(encoding="utf-8")) == {
        "models-dir": {
            "pipeline": str(moved_models / "mineru" / "pipeline"),
            "vlm": str(moved_models / "mineru" / "vlm"),
        }
    }
    assert model_assets_config_path(state) == state / "config" / "model-assets.json"
    ensure_mineru_tools_config(tmp_path / "fresh", tmp_path / "fresh-models")


def test_resolved_mineru_uses_local_source_and_pipeline_vlm_roots(
    tmp_path: Path,
) -> None:
    pipeline = tmp_path / "state" / "models" / "mineru" / "pipeline"
    vlm = tmp_path / "state" / "models" / "mineru" / "vlm"
    pipeline.mkdir(parents=True)
    vlm.mkdir(parents=True)
    resolved = ResolvedModelSet(
        "backend-release",
        (
            ResolvedModel("mineru/pipeline", "mineru", "pipeline", pipeline),
            ResolvedModel("mineru/vlm", "mineru", "vlm", vlm),
        ),
    )

    environment = model_source_environment(
        source_id="modelscope",
        state_root=tmp_path / "state",
        models_root=tmp_path / "state" / "models",
        resolved_models=resolved,
    )

    assert environment["MINERU_MODEL_SOURCE"] == "local"
    assert json.loads(
        Path(environment["MINERU_TOOLS_CONFIG_JSON"]).read_text(encoding="utf-8")
    ) == {"models-dir": {"pipeline": str(pipeline), "vlm": str(vlm)}}


def test_resolved_mineru_requires_both_pipeline_and_vlm_bindings(
    tmp_path: Path,
) -> None:
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    resolved = ResolvedModelSet(
        "backend-release",
        (ResolvedModel("mineru/pipeline", "mineru", "pipeline", pipeline),),
    )

    with pytest.raises(ModelAcquisitionError, match="pipeline.*vlm"):
        model_source_environment(
            source_id="modelscope",
            state_root=tmp_path / "state",
            models_root=tmp_path / "state" / "models",
            resolved_models=resolved,
        )
