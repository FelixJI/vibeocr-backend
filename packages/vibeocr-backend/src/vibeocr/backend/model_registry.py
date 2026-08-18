"""Model registry acquisition for the durable runtime maintenance operation.

模型源目录声明 ``model_registry`` kind(``huggingface``/``modelscope``);
本模块把声明的模型资产下载纳入 durable ensure:staging 位于
``state/models/downloads``,成功后原子切换到 ``state/models/<engine>``,
已存在的本地模型在断网时直接复用,不做懒下载。生产 Adapter 只经 HTTP
取数并把源 id 映射为 ``PADDLE_PDX_MODEL_SOURCE``/``MINERU_MODEL_SOURCE``;
仓库内部布局(endpoint/revision/file 映射)按公开 CLI 契约构造。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Protocol

DOWNLOAD_SOURCE_KIND_MODEL_REGISTRY = "model_registry"
MODEL_ENGINES = ("paddleocr", "mineru")
MODEL_CONSUMERS = ("paddleocr", "pp_structure", "paddleocr_vl", "mineru")
MODEL_STAGING_DIRNAME = "downloads"
MODEL_READY_FILENAME = ".ready.json"
_MODEL_ASSETS_FILENAME = "model-assets.json"
_RESOLVED_MODELS_FILENAME = "resolved-models.json"
_MINERU_TOOLS_FILENAME = "mineru.json"
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_BINDING_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class ModelAcquisitionError(RuntimeError):
    """模型获取失败;已存在的基础运行时与本地模型不受影响。"""


@dataclass(frozen=True, slots=True)
class ModelFileIntegrity:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ModelAsset:
    """一个引擎所需的远端模型资产(显式声明,不猜测)。"""

    engine: str
    name: str
    repository: str
    revision: str
    files: tuple[str, ...]
    consumer: str
    binding_key: str
    file_integrity: tuple[ModelFileIntegrity, ...] = ()

    @property
    def target_dirname(self) -> str:
        return f"{self.name}-{self.revision}"


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """一个已验证资产及其本地 Adapter 绑定。"""

    key: str
    consumer: str
    binding_key: str
    root: Path


@dataclass(frozen=True, slots=True)
class ResolvedModelSet:
    """同一 Backend release 已验证的本地模型集合。"""

    release_identity: str
    models: tuple[ResolvedModel, ...]

    def __getitem__(self, key: str) -> Path:
        for model in self.models:
            if model.key == key:
                return model.root
        raise KeyError(key)

    def binding_kwargs(self, consumer: str) -> dict[str, str]:
        bindings = {
            model.binding_key: str(model.root)
            for model in self.models
            if model.consumer == consumer
        }
        if not bindings:
            raise ModelAcquisitionError(
                f"resolved model consumer is missing: {consumer}"
            )
        return bindings

    def to_payload(self) -> dict[str, object]:
        consumers: dict[str, dict[str, str]] = {}
        for model in self.models:
            consumers.setdefault(model.consumer, {})[model.binding_key] = str(
                model.root
            )
        return {
            "schema_version": 1,
            "release_identity": self.release_identity,
            "consumers": consumers,
        }


@dataclass(slots=True)
class ModelProgress:
    """下载进度(bytes);由维护 reporter 投影为 measured progress。"""

    asset: str
    file_name: str
    current: int
    total: int
    _listeners: tuple[Callable[["ModelProgress"], None], ...] = field(
        default=(), repr=False
    )
    _cancel_check: Callable[[], None] | None = field(default=None, repr=False)

    def report(self) -> None:
        if self._cancel_check is not None:
            self._cancel_check()
        for listener in self._listeners:
            listener(self)


class ModelRegistryAdapter(Protocol):
    """按源 endpoint/revision 取一个模型文件到目标路径。"""

    def fetch_file(
        self,
        *,
        source_id: str,
        endpoint: str,
        asset: ModelAsset,
        file_name: str,
        destination: Path,
        progress: ModelProgress,
    ) -> None: ...


class HuggingFaceAdapter:
    """Hugging Face resolve 端点(公开 CLI 下载契约)。"""

    def fetch_file(
        self,
        *,
        source_id: str,
        endpoint: str,
        asset: ModelAsset,
        file_name: str,
        destination: Path,
        progress: ModelProgress,
    ) -> None:
        url = f"{endpoint}/{asset.repository}/resolve/{asset.revision}/{file_name}"
        _http_download(url, destination, progress)


class ModelScopeAdapter:
    """ModelScope 文件端点(公开 API 下载契约)。"""

    def fetch_file(
        self,
        *,
        source_id: str,
        endpoint: str,
        asset: ModelAsset,
        file_name: str,
        destination: Path,
        progress: ModelProgress,
    ) -> None:
        url = (
            f"{endpoint}/api/v1/models/{asset.repository}/repo"
            f"?Revision={asset.revision}&FilePath={file_name}"
        )
        _http_download(url, destination, progress)


class LocalDirectoryAdapter:
    """测试/离线 Adapter:从本地目录按相同布局读取,不经网络。"""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def fetch_file(
        self,
        *,
        source_id: str,
        endpoint: str,
        asset: ModelAsset,
        file_name: str,
        destination: Path,
        progress: ModelProgress,
    ) -> None:
        source_root = self._root / source_id
        candidate = source_root / asset.repository / asset.revision / file_name
        if not candidate.is_file():
            raise ModelAcquisitionError(f"local model fixture is missing: {candidate}")
        total = candidate.stat().st_size
        current = 0
        with candidate.open("rb") as reader, destination.open("wb") as writer:
            while chunk := reader.read(64 * 1024):
                writer.write(chunk)
                current += len(chunk)
                progress.current = current
                progress.total = total
                progress.report()


def adapter_for_source(source_id: str) -> ModelRegistryAdapter:
    if source_id == "huggingface":
        return HuggingFaceAdapter()
    if source_id == "modelscope":
        return ModelScopeAdapter()
    raise ModelAcquisitionError(f"unknown model registry source: {source_id}")


def _validate_manifest_path(value: str, *, field_name: str, nested: bool) -> None:
    """Validate one manifest-relative path using portable Windows semantics."""
    if (
        not value
        or "\\" in value
        or ":" in value
        or value.startswith("/")
        or value.endswith("/")
    ):
        raise ModelAcquisitionError(f"unsafe model {field_name}: {value!r}")
    segments = value.split("/")
    if not nested and len(segments) != 1:
        raise ModelAcquisitionError(f"unsafe model {field_name}: {value!r}")
    if PurePosixPath(value).is_absolute():
        raise ModelAcquisitionError(f"unsafe model {field_name}: {value!r}")
    for segment in segments:
        reserved_stem = segment.split(".", 1)[0].upper()
        if (
            segment in {"", ".", ".."}
            or not _SAFE_SEGMENT.fullmatch(segment)
            or segment.endswith((".", " "))
            or reserved_stem in _WINDOWS_RESERVED_NAMES
        ):
            raise ModelAcquisitionError(f"unsafe model {field_name}: {value!r}")


def _is_reparse_point(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(metadata, "st_file_attributes", 0)
    return path.is_symlink() or bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _safe_model_path(models_root: Path, *parts: str) -> Path:
    """Resolve a model-store path while rejecting every existing reparse hop."""
    root = models_root.absolute()
    candidate = root.joinpath(*parts)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ModelAcquisitionError("unsafe model path escapes models_root") from exc
    current = root
    if _is_reparse_point(current):
        raise ModelAcquisitionError("model path crosses a reparse point")
    for part in relative.parts:
        current = current / part
        if (current.exists() or current.is_symlink()) and _is_reparse_point(current):
            raise ModelAcquisitionError("model path crosses a reparse point")
    existing_parent = candidate if candidate.exists() else candidate.parent
    while not existing_parent.exists() and existing_parent != root:
        existing_parent = existing_parent.parent
    try:
        existing_parent.resolve(strict=True).relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ModelAcquisitionError(
            "unsafe model path resolves outside models_root"
        ) from exc
    return candidate


def _ready_payload(asset: ModelAsset, release_identity: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "release_identity": release_identity,
        "engine": asset.engine,
        "name": asset.name,
        "repository": asset.repository,
        "revision": asset.revision,
        "files": [
            {"path": item.path, "size": item.size, "sha256": item.sha256}
            for item in asset.file_integrity
        ]
        or list(asset.files),
    }


def _write_ready_marker(
    target: Path,
    asset: ModelAsset,
    release_identity: str,
) -> None:
    marker = target / MODEL_READY_FILENAME
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            _ready_payload(asset, release_identity),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, marker)


def _model_is_ready(
    target: Path,
    asset: ModelAsset,
    release_identity: str,
) -> bool:
    if not target.is_dir() or _is_reparse_point(target):
        return False
    actual_files: set[str] = set()
    for path in target.rglob("*"):
        if _is_reparse_point(path):
            return False
        if path.is_file():
            actual_files.add(path.relative_to(target).as_posix())
    if actual_files != {*asset.files, MODEL_READY_FILENAME}:
        return False
    try:
        marker: Any = json.loads(
            (target / MODEL_READY_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return False
    if marker != _ready_payload(asset, release_identity):
        return False
    for declared in asset.file_integrity:
        path = target.joinpath(*PurePosixPath(declared.path).parts)
        if (
            path.stat().st_size != declared.size
            or _sha256_file(path) != declared.sha256
        ):
            return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_assets(
    config_file: Path | None,
    *,
    expected_release_identity: str | None = None,
) -> tuple[ModelAsset, ...]:
    """加载显式声明的模型资产;默认为空(不为未经验证的模型猜测 pin)。"""
    if config_file is None or not config_file.is_file():
        return ()
    try:
        payload: Any = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelAcquisitionError(
            f"invalid model assets manifest: {config_file}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("release_identity"), str)
        or not payload["release_identity"].strip()
        or not isinstance(payload.get("assets"), list)
    ):
        raise ModelAcquisitionError(
            "model assets manifest identity/schema/assets are invalid"
        )
    if (
        expected_release_identity is not None
        and payload["release_identity"] != expected_release_identity
    ):
        raise ModelAcquisitionError("model assets release identity mismatch")
    assets: list[ModelAsset] = []
    asset_keys: set[tuple[str, str]] = set()
    binding_keys: set[tuple[str, str]] = set()
    for item in payload["assets"]:
        if not isinstance(item, dict):
            raise ModelAcquisitionError("model asset entry must be an object")
        engine = item.get("engine")
        name = item.get("name")
        repository = item.get("repository")
        revision = item.get("revision")
        raw_files = item.get("files")
        consumer = item.get("consumer")
        binding_key = item.get("binding_key")
        if (
            engine not in MODEL_ENGINES
            or not isinstance(name, str)
            or not name.strip()
            or not isinstance(repository, str)
            or not repository.strip()
            or not isinstance(revision, str)
            or not revision.strip()
            or not isinstance(raw_files, list)
            or not raw_files
            or consumer not in MODEL_CONSUMERS
            or not isinstance(binding_key, str)
            or not _SAFE_BINDING_KEY.fullmatch(binding_key)
            or (engine == "mineru") != (consumer == "mineru")
        ):
            raise ModelAcquisitionError(f"invalid model asset declaration: {item!r}")
        file_integrity: list[ModelFileIntegrity] = []
        for file_index, raw_file in enumerate(raw_files):
            if not isinstance(raw_file, dict):
                raise ModelAcquisitionError(
                    f"invalid model file declaration: {raw_file!r}"
                )
            file_name = raw_file.get("path")
            size = raw_file.get("size")
            digest = raw_file.get("sha256")
            if (
                not isinstance(file_name, str)
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
                or not isinstance(digest, str)
                or not _SHA256.fullmatch(digest)
            ):
                raise ModelAcquisitionError(
                    f"invalid model file declaration at index {file_index}"
                )
            _validate_manifest_path(file_name, field_name="file", nested=True)
            file_integrity.append(ModelFileIntegrity(file_name, size, digest))
        files = [item.path for item in file_integrity]
        if len(set(files)) != len(files):
            raise ModelAcquisitionError("model asset files must be unique")
        _validate_manifest_path(name, field_name="name", nested=False)
        _validate_manifest_path(repository, field_name="repository", nested=True)
        _validate_manifest_path(revision, field_name="revision", nested=False)
        asset_key = (engine, name)
        binding = (consumer, binding_key)
        if asset_key in asset_keys or binding in binding_keys:
            raise ModelAcquisitionError("model asset ids and bindings must be unique")
        asset_keys.add(asset_key)
        binding_keys.add(binding)
        assets.append(
            ModelAsset(
                engine=engine,
                name=name,
                repository=repository,
                revision=revision,
                files=tuple(files),
                consumer=consumer,
                binding_key=binding_key,
                file_integrity=tuple(file_integrity),
            )
        )
    return tuple(assets)


def model_assets_config_path(state_root: Path) -> Path:
    return state_root / "config" / _MODEL_ASSETS_FILENAME


def ensure_mineru_tools_config(
    state_root: Path,
    models_root: Path,
    *,
    mineru_models_dir: Path | None = None,
) -> Path:
    """按当前 Portable 根重建 MinerU 配置，不复用旧绝对路径。"""
    config_dir = state_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / _MINERU_TOOLS_FILENAME
    temporary = config_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"models-dir": str(mineru_models_dir or (models_root / "mineru"))},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, config_path)
    return config_path


def model_source_environment(
    *,
    source_id: str | None,
    state_root: Path,
    models_root: Path,
    resolved_models: ResolvedModelSet | None = None,
) -> dict[str, str]:
    """把已选 model_registry 源映射为推理进程环境,并收口 MinerU 配置。"""
    mineru_models_dir: Path | None = None
    if resolved_models is not None:
        mineru = [
            model for model in resolved_models.models if model.consumer == "mineru"
        ]
        if mineru:
            bindings = resolved_models.binding_kwargs("mineru")
            value = bindings.get("models_dir")
            if value is None:
                raise ModelAcquisitionError(
                    "MinerU local models_dir binding is required"
                )
            mineru_models_dir = Path(value)
        resolved_path = state_root / "config" / _RESOLVED_MODELS_FILENAME
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = resolved_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                resolved_models.to_payload(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, resolved_path)
    environment: dict[str, str] = {
        "MINERU_TOOLS_CONFIG_JSON": str(
            ensure_mineru_tools_config(
                state_root,
                models_root,
                mineru_models_dir=mineru_models_dir,
            )
        )
    }
    if resolved_models is not None:
        environment["VIBEOCR_RESOLVED_MODELS"] = str(resolved_path)
    if source_id is not None:
        environment["PADDLE_PDX_MODEL_SOURCE"] = source_id
        environment["MINERU_MODEL_SOURCE"] = source_id
    return environment


def local_model_kwargs(consumer: str) -> dict[str, str]:
    """读取 Installer 生成的已验证本地 Adapter 绑定。"""
    config = os.environ.get("VIBEOCR_RESOLVED_MODELS")
    if not config:
        return {}
    try:
        payload: Any = json.loads(Path(config).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelAcquisitionError("resolved model binding is invalid") from exc
    consumers = payload.get("consumers") if isinstance(payload, dict) else None
    binding = consumers.get(consumer) if isinstance(consumers, dict) else None
    if not isinstance(binding, dict) or not binding:
        raise ModelAcquisitionError(f"resolved model consumer is missing: {consumer}")
    result: dict[str, str] = {}
    for key, value in binding.items():
        if (
            not isinstance(key, str)
            or not _SAFE_BINDING_KEY.fullmatch(key)
            or not isinstance(value, str)
            or not Path(value).is_dir()
        ):
            raise ModelAcquisitionError("resolved model binding is invalid")
        result[key] = value
    return result


def acquire_models(
    *,
    assets: tuple[ModelAsset, ...],
    release_identity: str,
    source_id: str,
    endpoint: str,
    models_root: Path,
    adapter: ModelRegistryAdapter | None = None,
    progress_listener: Callable[[ModelProgress], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> ResolvedModelSet:
    """下载声明资产到 staging 后原子切换;本地已有则直接复用(断网可用)。

    失败时清理 staging、不影响既有模型与基础运行时;每个文件成功即落
    盘 staging,资产全部就绪后一次性 rename 到最终目录。
    """
    if not release_identity.strip():
        raise ModelAcquisitionError("model release_identity is required")
    if not assets:
        return ResolvedModelSet(release_identity, ())
    asset_keys: set[tuple[str, str]] = set()
    binding_keys: set[tuple[str, str]] = set()
    for asset in assets:
        if asset.engine not in MODEL_ENGINES:
            raise ModelAcquisitionError(f"invalid model engine: {asset.engine!r}")
        if (
            asset.consumer not in MODEL_CONSUMERS
            or not _SAFE_BINDING_KEY.fullmatch(asset.binding_key)
            or (asset.engine == "mineru") != (asset.consumer == "mineru")
        ):
            raise ModelAcquisitionError("invalid model consumer binding")
        _validate_manifest_path(asset.name, field_name="name", nested=False)
        _validate_manifest_path(asset.repository, field_name="repository", nested=True)
        _validate_manifest_path(asset.revision, field_name="revision", nested=False)
        if not asset.files:
            raise ModelAcquisitionError("model asset files must not be empty")
        for file_name in asset.files:
            _validate_manifest_path(file_name, field_name="file", nested=True)
        asset_key = (asset.engine, asset.name)
        binding = (asset.consumer, asset.binding_key)
        if asset_key in asset_keys or binding in binding_keys:
            raise ModelAcquisitionError("model asset ids and bindings must be unique")
        asset_keys.add(asset_key)
        binding_keys.add(binding)
    resolved_adapter = adapter if adapter is not None else adapter_for_source(source_id)
    models_root = models_root.absolute()
    models_root.mkdir(parents=True, exist_ok=True)
    _safe_model_path(models_root)
    staging_root = _safe_model_path(models_root, MODEL_STAGING_DIRNAME)
    staging_root.mkdir(parents=True, exist_ok=True)
    acquired: list[ResolvedModel] = []
    for asset in assets:
        if cancel_check is not None:
            cancel_check()
        target = _safe_model_path(models_root, asset.engine, asset.target_dirname)
        if _model_is_ready(target, asset, release_identity):
            # 断网本地复用只接受绑定当前 release/file-set 的完成 marker。
            acquired.append(
                ResolvedModel(
                    key=f"{asset.engine}/{asset.name}",
                    consumer=asset.consumer,
                    binding_key=asset.binding_key,
                    root=target,
                )
            )
            continue
        operation_root = _safe_model_path(
            models_root, MODEL_STAGING_DIRNAME, uuid.uuid4().hex
        )
        operation_root.mkdir()
        asset_staging = operation_root / "candidate"
        asset_staging.mkdir(parents=True)
        progress = ModelProgress(
            asset=f"{asset.engine}/{asset.name}",
            file_name="",
            current=0,
            total=0,
            _listeners=(progress_listener,) if progress_listener else (),
            _cancel_check=cancel_check,
        )
        integrity_by_path = {item.path: item for item in asset.file_integrity}
        try:
            for file_name in asset.files:
                if cancel_check is not None:
                    cancel_check()
                progress.file_name = file_name
                progress.current = 0
                progress.total = 0
                destination = asset_staging.joinpath(*PurePosixPath(file_name).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                resolved_adapter.fetch_file(
                    source_id=source_id,
                    endpoint=endpoint,
                    asset=asset,
                    file_name=file_name,
                    destination=destination,
                    progress=progress,
                )
                declared = integrity_by_path.get(file_name)
                if declared is not None and (
                    destination.stat().st_size != declared.size
                    or _sha256_file(destination) != declared.sha256
                ):
                    raise ModelAcquisitionError(
                        f"model file integrity mismatch: {asset.name}/{file_name}"
                    )
            _write_ready_marker(asset_staging, asset, release_identity)
            _safe_model_path(models_root, asset.engine)
            target.parent.mkdir(parents=True, exist_ok=True)
            _safe_model_path(models_root, asset.engine)
            previous = operation_root / "previous"
            moved_previous = False
            try:
                if target.exists():
                    target.replace(previous)
                    moved_previous = True
                asset_staging.replace(target)
            except Exception:
                if moved_previous and previous.exists() and not target.exists():
                    previous.replace(target)
                raise
        except ModelAcquisitionError:
            shutil.rmtree(operation_root, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(operation_root, ignore_errors=True)
            if cancel_check is not None:
                # Preserve the caller's cancellation type/state instead of
                # misreporting cancellation as a model acquisition failure.
                cancel_check()
            raise ModelAcquisitionError(
                f"failed to acquire model {asset.name}@{asset.revision}: {exc}"
            ) from exc
        shutil.rmtree(operation_root, ignore_errors=True)
        acquired.append(
            ResolvedModel(
                key=f"{asset.engine}/{asset.name}",
                consumer=asset.consumer,
                binding_key=asset.binding_key,
                root=target,
            )
        )
    return ResolvedModelSet(release_identity, tuple(acquired))


def _http_download(
    url: str,
    destination: Path,
    progress: ModelProgress,
    *,
    timeout: float = 60.0,
) -> None:
    try:
        with urllib.request.urlopen(  # noqa: S310
            urllib.request.Request(url), timeout=timeout
        ) as response:
            total = int(response.headers.get("Content-Length") or 0)
            progress.total = total
            current = 0
            with destination.open("wb") as writer:
                while chunk := response.read(64 * 1024):
                    writer.write(chunk)
                    current += len(chunk)
                    progress.current = current
                    progress.report()
    except urllib.error.URLError as exc:
        raise ModelAcquisitionError(f"model download failed: {url}") from exc
    except OSError as exc:
        raise ModelAcquisitionError(f"model download I/O failed: {url}") from exc
    if total and current != total:
        # 连接半途关闭表现为短读;不把截断文件当成功。
        raise ModelAcquisitionError(
            f"model download truncated: {url} ({current}/{total} bytes)"
        )
