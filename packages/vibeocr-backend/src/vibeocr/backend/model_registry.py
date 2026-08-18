"""Model registry acquisition for the durable runtime maintenance operation.

模型源目录声明 ``model_registry`` kind(``huggingface``/``modelscope``);
本模块把声明的模型资产下载纳入 durable ensure:staging 位于
``state/models/downloads``,成功后原子切换到 ``state/models/<engine>``,
已存在的本地模型在断网时直接复用,不做懒下载。生产 Adapter 只经 HTTP
取数并把源 id 映射为 ``PADDLE_PDX_MODEL_SOURCE``/``MINERU_MODEL_SOURCE``;
仓库内部布局(endpoint/revision/file 映射)按公开 CLI 契约构造。
"""

from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

DOWNLOAD_SOURCE_KIND_MODEL_REGISTRY = "model_registry"
MODEL_ENGINES = ("paddleocr", "mineru")
MODEL_STAGING_DIRNAME = "downloads"
_MODEL_ASSETS_FILENAME = "model-assets.json"
_MINERU_TOOLS_FILENAME = "mineru.json"


class ModelAcquisitionError(RuntimeError):
    """模型获取失败;已存在的基础运行时与本地模型不受影响。"""


@dataclass(frozen=True, slots=True)
class ModelAsset:
    """一个引擎所需的远端模型资产(显式声明,不猜测)。"""

    engine: str
    name: str
    revision: str
    files: tuple[str, ...]

    @property
    def target_dirname(self) -> str:
        return f"{self.name}-{self.revision}"


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

    def report(self) -> None:
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
        url = f"{endpoint}/{asset.name}/resolve/{asset.revision}/{file_name}"
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
            f"{endpoint}/api/v1/models/{asset.name}/repo"
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
        candidate = source_root / asset.name / asset.revision / file_name
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


def load_model_assets(config_file: Path | None) -> tuple[ModelAsset, ...]:
    """加载显式声明的模型资产;默认为空(不为未经验证的模型猜测 pin)。"""
    if config_file is None or not config_file.is_file():
        return ()
    try:
        payload: Any = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelAcquisitionError(
            f"invalid model assets manifest: {config_file}"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("assets"), list):
        raise ModelAcquisitionError(
            "model assets manifest must be an object with an assets array"
        )
    assets: list[ModelAsset] = []
    for item in payload["assets"]:
        if not isinstance(item, dict):
            raise ModelAcquisitionError("model asset entry must be an object")
        engine = item.get("engine")
        name = item.get("name")
        revision = item.get("revision")
        files = item.get("files")
        if (
            engine not in MODEL_ENGINES
            or not isinstance(name, str)
            or not name.strip()
            or not isinstance(revision, str)
            or not revision.strip()
            or not isinstance(files, list)
            or not files
            or not all(isinstance(entry, str) and entry for entry in files)
        ):
            raise ModelAcquisitionError(f"invalid model asset declaration: {item!r}")
        assets.append(
            ModelAsset(
                engine=engine,
                name=name,
                revision=revision,
                files=tuple(files),
            )
        )
    return tuple(assets)


def model_assets_config_path(state_root: Path) -> Path:
    return state_root / "config" / _MODEL_ASSETS_FILENAME


def ensure_mineru_tools_config(state_root: Path, models_root: Path) -> Path:
    """生成/复用 MinerU 工具配置,本地推理绑定 state 内模型目录。"""
    config_dir = state_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / _MINERU_TOOLS_FILENAME
    if not config_path.is_file():
        config_path.write_text(
            json.dumps({"models-dir": str(models_root / "mineru")}, indent=2) + "\n",
            encoding="utf-8",
        )
    return config_path


def model_source_environment(
    *,
    source_id: str | None,
    state_root: Path,
    models_root: Path,
) -> dict[str, str]:
    """把已选 model_registry 源映射为推理进程环境,并收口 MinerU 配置。"""
    environment: dict[str, str] = {
        "MINERU_TOOLS_CONFIG_JSON": str(
            ensure_mineru_tools_config(state_root, models_root)
        )
    }
    if source_id is not None:
        environment["PADDLE_PDX_MODEL_SOURCE"] = source_id
        environment["MINERU_MODEL_SOURCE"] = source_id
    return environment


def acquire_models(
    *,
    assets: tuple[ModelAsset, ...],
    source_id: str,
    endpoint: str,
    models_root: Path,
    adapter: ModelRegistryAdapter | None = None,
    progress_listener: Callable[[ModelProgress], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> dict[str, Path]:
    """下载声明资产到 staging 后原子切换;本地已有则直接复用(断网可用)。

    失败时清理 staging、不影响既有模型与基础运行时;每个文件成功即落
    盘 staging,资产全部就绪后一次性 rename 到最终目录。
    """
    if not assets:
        return {}
    resolved_adapter = adapter if adapter is not None else adapter_for_source(source_id)
    models_root.mkdir(parents=True, exist_ok=True)
    staging_root = models_root / MODEL_STAGING_DIRNAME
    acquired: dict[str, Path] = {}
    for asset in assets:
        if cancel_check is not None:
            cancel_check()
        target = models_root / asset.engine / asset.target_dirname
        if target.is_dir() and any(target.iterdir()):
            # 断网本地复用:已就绪资产不再触网。
            acquired[f"{asset.engine}/{asset.name}"] = target
            continue
        asset_staging = staging_root / asset.engine / asset.target_dirname
        if asset_staging.exists():
            shutil.rmtree(asset_staging)
        asset_staging.mkdir(parents=True)
        progress = ModelProgress(
            asset=f"{asset.engine}/{asset.name}",
            file_name="",
            current=0,
            total=0,
            _listeners=(progress_listener,) if progress_listener else (),
        )
        try:
            for file_name in asset.files:
                if cancel_check is not None:
                    cancel_check()
                progress.file_name = file_name
                progress.current = 0
                progress.total = 0
                destination = asset_staging / file_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                resolved_adapter.fetch_file(
                    source_id=source_id,
                    endpoint=endpoint,
                    asset=asset,
                    file_name=file_name,
                    destination=destination,
                    progress=progress,
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            asset_staging.replace(target)
        except ModelAcquisitionError:
            shutil.rmtree(asset_staging, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(asset_staging, ignore_errors=True)
            raise ModelAcquisitionError(
                f"failed to acquire model {asset.name}@{asset.revision}: {exc}"
            ) from exc
        acquired[f"{asset.engine}/{asset.name}"] = target
    return acquired


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
