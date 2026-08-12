"""环境管理模块：负责自动部署 Python 运行时（python-build-standalone）和管理项目依赖"""

import contextlib
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal
from urllib.request import Request, urlopen

from vibeocr.backend.runtime_layout import resolve_app_paths
from vibeocr.backend.runtime_state import (
    create_cache_entry,
    is_cache_valid,
    update_cache_field,
)
from vibeocr.backend.services.env_config import (
    PYTHON_BUILD_STANDALONE_ASSET,
    PYTHON_BUILD_STANDALONE_BASE,
    PYTHON_BUILD_STANDALONE_MIRRORS,
    PYTHON_BUILD_STANDALONE_TAG,
    get_pytorch_mirror,
)
from vibeocr.backend.utils.job_object import JobObjectGuard

# Python 运行时下载地址（python-build-standalone）
# GitHub 直链 + 国内镜像（NJU/ghproxy）已在 env_config.PYTHON_BUILD_STANDALONE_MIRRORS 定义
PYTHON_STANDALONE_URLS = [
    PYTHON_BUILD_STANDALONE_BASE,
    *PYTHON_BUILD_STANDALONE_MIRRORS,
]

# CUDA 版本映射到 PaddlePaddle 支持的版本。
# paddlepaddle-gpu 3.3.1 实际只发布三个 win cp313 wheel：cu118 / cu126 / cu129
# （直接从 paddlepaddle.org.cn/packages/stable/{tag}/ 核实）。
# 之前曾映射到 cu121/cu123，但 paddle 从未发布这些 win wheel，会导致安装 404。
#
# 本项目统一用 cu126（CUDA 12.6 构建），与 torch cu126 同源：paddle 所需的
# cublas64_12.dll 等 CUDA 12 运行时由 torch wheel 的 torch/lib 目录提供
# （见 _install_paddle_stack 与 OCRService._setup_cuda_dll_path）。
# CUDA 12.x 同大版本共享 cublas64_12.dll，故任何 12.x 驱动都能跑 cu126 runtime。
# CUDA 13.x 驱动向下兼容 CUDA 12.x 运行时，同样映射到 cu126。
# 不启用 cu129：虽然它对 RTX 50 系有专门适配，但 torch 无对应 win wheel，
# 改用 cu129 paddle 会让 torch/lib 的 CUDA 12 DLL 与之不匹配（见 TORCH_CUDA_MAP）。
# 不用 cu130：需 cublas64_13.dll，与 torch/lib 的 _12 系列 ABI 不匹配。
#
# 注意：_install_paddle_stack 接收的 cuda_version 已是 cu-tag（detect_cuda_version 输出），
# 此映射仅用于 detect_cuda_version 把原始版本（nvidia-smi 输出）转成 cu-tag。
CUDA_VERSION_MAP = {
    "11.8": "cu118",
    # CUDA 12.x 全部归并到 cu126（cu121/cu123 的 paddle wheel 不存在）。
    # CUDA 12 同大版本共享 cublas64_12.dll，cu126 runtime 向下兼容 12.0+。
    "12.0": "cu126",
    "12.1": "cu126",
    "12.2": "cu126",
    "12.3": "cu126",
    "12.4": "cu126",
    "12.5": "cu126",
    "12.6": "cu126",
    "12.7": "cu126",
    "12.8": "cu126",
    "12.9": "cu126",
    # CUDA 13.x 驱动仍兼容 CUDA 12.x 运行时，故映射到 cu126。
    "13.0": "cu126",
    "13.1": "cu126",
    "13.2": "cu126",
}

# PyTorch CUDA 版本映射（PaddlePaddle CUDA tag → PyTorch CUDA tag）。
# 开发环境 (uv) 的 torch 来源见 pyproject.toml [tool.uv.sources]，恒定 cu126；
# 便携环境 (pip) 的 torch 来源由此映射 + get_pytorch_mirror 决定。
# 只有 paddle tag 在此表中时 torch 才选对应 tag；不在表里时回退到 cu126
# （torch 无 cu129/cu130 的 win cp313 wheel，cu126 的 torch/lib 正好提供
# cu129 paddle 所需的 cublas64_12.dll，但本项目不启用 cu129，故仅作兜底）。
TORCH_CUDA_MAP = {
    "cu118": "cu118",
    "cu126": "cu126",
}

logger = logging.getLogger(__name__)

_dep_specs_cache: dict[str, str] | None = None


def _load_packaged_dependency_profiles() -> dict:
    """读取随 client wheel 分发的依赖 profile。"""
    import json

    profile_path = Path(__file__).with_name("dependency_profiles.json")
    return json.loads(profile_path.read_text(encoding="utf-8"))


def _load_dep_specs() -> dict[str, str]:
    """从 pyproject.toml 或 version.json 加载依赖版本规格

    开发环境读 pyproject.toml（权威源）；打包环境读 version.json（由 bump_version 生成）。
    两者皆缺失时抛 RuntimeError，不再悄悄回退到陈旧 fallback ——
    陈旧 fallback 会装到旧版本（paddleocr 3.6.0 / mineru 3.2.0 已落后于 pyproject）。

    Returns:
        {base_name: full_spec}，如 {"paddleocr": "paddleocr[doc-parser]>=3.7.0"}

    Raises:
        RuntimeError: pyproject.toml 与 version.json 均不存在
    """
    global _dep_specs_cache
    if _dep_specs_cache is not None:
        return _dep_specs_cache

    import re

    project_root = get_project_root()

    # 物理拆包后根 pyproject 是 meta package，不再声明后端重依赖；开发态从
    # backend 工作区读取，打包态仍优先兼容 version.json。
    pyproject = project_root / "packages" / "vibeocr-backend" / "pyproject.toml"
    if pyproject.exists():
        import tomllib

        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = data.get("project", {})
        deps = list(project.get("dependencies", []))
        for optional_deps in project.get("optional-dependencies", {}).values():
            deps.extend(optional_deps)
        specs: dict[str, str] = {}
        for dep in deps:
            dep = dep.strip()
            if dep.startswith("#"):
                continue
            m = re.match(r"^([a-zA-Z0-9_.-]+)", dep)
            if m:
                specs[m.group(1).lower()] = dep
        _dep_specs_cache = specs
        return _dep_specs_cache

    version_json = project_root / "version.json"
    if version_json.exists():
        import json
        import re as _re

        data = json.loads(version_json.read_text(encoding="utf-8"))
        raw_versions: dict = data.get("dep_versions", {})
        # dep_extras: {pkg: [extra1, extra2]}，缺失时按空（无 extras）处理。
        raw_extras: dict = data.get("dep_extras", {})
        specs: dict[str, str] = {}
        for k, v in raw_versions.items():
            # 三层向后兼容（按历史演进顺序）：
            # 1. 旧旧版：裸版本号 str（如 "3.3.1"），按 ">=3.3.1" 处理
            # 2. 曾用版：{"version": "3.3.1", "op": ">="} dict，拼成 ">=3.3.1"
            # 3. 当前版：约束串 str（如 ">=3.3.1" / "==3.3.1+cu126" / ">=1,<2"）
            #    —— 以 PEP 440 操作符开头即为约束串，否则按裸版本号处理。
            if isinstance(v, dict):
                ver = str(v.get("version", "")).strip()
                op = str(v.get("op", ">=")).strip() or ">="
                constraint = f"{op}{ver}"
            else:
                s = str(v).strip()
                # 已是约束串（以操作符开头）→ 直接用；否则视为裸版本号
                if s and _re.match(r"^(==|!=|>=|<=|~=|>|<)", s):
                    constraint = s
                elif s:
                    constraint = f">={s}"
                else:
                    constraint = ""  # 无版本约束
            # 拼回完整 PEP 508 规格：pkg[extras]constraint
            extras_list = raw_extras.get(k)
            if extras_list:
                extras_str = "[" + ",".join(extras_list) + "]"
                specs[k] = f"{k}{extras_str}{constraint}".rstrip()
            else:
                specs[k] = f"{k}{constraint}".rstrip() if constraint else k
        # 根发行包物理拆分后 version.json 可能只记录四个内部包；以后端
        # wheel 随附的 profile 为基线，再让版本文件覆盖同名约束。
        packaged = _load_packaged_dependency_profiles().get("dependencies", {})
        merged = {str(k): str(v) for k, v in packaged.items()}
        merged.update(specs)
        _dep_specs_cache = merged
        return _dep_specs_cache

    packaged = _load_packaged_dependency_profiles().get("dependencies", {})
    if packaged:
        _dep_specs_cache = {str(k): str(v) for k, v in packaged.items()}
        return _dep_specs_cache

    raise RuntimeError("无法加载随包分发的后端依赖 profile")


# 包名归一映射：pyproject/uv.lock 用 paddlepaddle-gpu，运行时检测用 paddlepaddle。
# 与 bump_version._KEY_ALIASES_LOCK 保持一致。
_LOCK_KEY_ALIASES = {"paddlepaddle-gpu": "paddlepaddle"}

# 缓存：与 _dep_specs_cache 同理，避免重复解析 uv.lock / version.json。
_locked_versions_cache: dict[str, str] | None = None


def _load_locked_versions() -> dict[str, str]:
    """加载每个依赖包的锁定版本（权威比较基准）。

    pyproject 的 ``>=3.4.0`` 只是下界，无法表达"实际锁定 3.4.2"。便携环境已装 3.4.0
    满足下界，单比下界会误判为最新、漏掉 3.4.0 → 3.4.2 这类下界内升级。锁定版解决此问题。

    数据源（与 _load_dep_specs 对称）：
    - 打包环境（无 pyproject.toml，有 version.json）：读 ``dep_locked_versions``
      字段（由 bump_version 从 uv.lock 写入）。字段缺失返回空 dict（旧版兼容）。
    - 开发环境（有 pyproject.toml）：直接解析仓库根的 ``uv.lock``。lock 不存在返回空 dict。

    返回的 key 与 _load_dep_specs 一致（paddlepaddle-gpu → paddlepaddle），
    便于 detect_dependency_updates 用同一 pkg 查找。

    Returns:
        {归一包名: 锁定版本串}，如 {"mineru": "3.4.2", "torch": "2.12.1+cu126"}。
        找不到锁定版本时返回空 dict（调用方回退到约束下界）。
    """
    global _locked_versions_cache
    if _locked_versions_cache is not None:
        return _locked_versions_cache

    project_root = get_project_root()
    pyproject = project_root / "pyproject.toml"

    # 开发环境：直接读 uv.lock（权威锁定源）
    if pyproject.exists():
        uv_lock = project_root / "uv.lock"
        versions = _parse_uv_lock(uv_lock)
        _locked_versions_cache = versions
        return versions

    # 打包环境：读 version.json 的 dep_locked_versions（bump_version 预写）
    version_json = project_root / "version.json"
    if version_json.exists():
        import json

        try:
            data = json.loads(version_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            _locked_versions_cache = {}
            return {}
        raw = data.get("dep_locked_versions", {})
        versions = (
            {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
        )
        _locked_versions_cache = versions
        return versions

    packaged = _load_packaged_dependency_profiles().get("locked_versions", {})
    if isinstance(packaged, dict):
        _locked_versions_cache = {str(k): str(v) for k, v in packaged.items()}
        return _locked_versions_cache

    # 两者皆无：无法获取锁定版，返回空（调用方回退下界）
    _locked_versions_cache = {}
    return {}


def _parse_uv_lock(lock_path: Path) -> dict[str, str]:
    """解析 uv.lock，返回 {归一包名: 锁定版本}。

    uv.lock 是 TOML，每个包为 ``[[package]]`` 表，含 ``name`` 与 ``version``。
    锁定版本含 local label（如 torch ``2.12.1+cu126``），原样返回；
    packaging.version.Version 能正确解析比较。

    Args:
        lock_path: uv.lock 路径

    Returns:
        {归一包名: 版本串}；文件不存在或解析失败时返回空 dict。
    """
    if not lock_path.exists():
        return {}
    import tomllib

    try:
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError):
        return {}
    versions: dict[str, str] = {}
    for pkg in lock.get("package", []):
        name = pkg.get("name", "")
        version = pkg.get("version", "")
        if name and version:
            key = _LOCK_KEY_ALIASES.get(name.lower()) or name.lower()
            versions[key] = version
    return versions


def detect_network_source() -> Literal["domestic", "international"]:
    """检测网络类型（委托 NetworkDetector）

    注意：返回值用于选择 pip 镜像、PaddleX/MinerU 模型源。
    """
    from vibeocr.backend.network_detector import NetworkDetector as _ND

    detector = _ND(get_project_root())
    return detector.network_type


def get_pip_source(
    network_type: Literal["domestic", "international"] = "domestic",
) -> str:
    """获取显式网络 profile 对应的 pip 镜像源 URL。"""
    from vibeocr.backend.network_detector import get_pip_mirror

    return get_pip_mirror(network_type)


def _is_active_python_environment_root(project_root: Path) -> bool:
    """Return whether ``project_root`` is the interpreter running VibeOCR."""
    if getattr(sys, "frozen", False):
        return False
    try:
        return project_root.resolve() == Path(sys.prefix).resolve()
    except OSError:
        return False


def get_environment_mode(project_root: Path) -> Literal["venv", "portable", "none"]:
    """检测当前环境模式

    Returns:
        "venv": 使用 .venv 虚拟环境
        "portable": 使用便携式 python/ 目录
        "none": 无任何环境
    """
    if _is_active_python_environment_root(project_root):
        return "venv"
    if (project_root / ".venv").exists():
        return "venv"
    if (project_root / "python").exists():
        return "portable"
    return "none"


def get_embedded_python_path(project_root: Path) -> Path:
    """获取嵌入式Python目录路径

    优先使用 .venv 虚拟环境(开发模式),如果不存在则使用 python/ 目录(便携式部署)
    """
    if _is_active_python_environment_root(project_root):
        return Path(sys.executable).resolve().parent

    venv_python = (
        project_root / ".venv" / "Scripts" / "python.exe"
        if os.name == "nt"
        else project_root / ".venv" / "bin" / "python"
    )
    portable_python = project_root / "python"

    # 优先使用虚拟环境
    if venv_python.exists():
        return venv_python.parent
    return portable_python


def get_embedded_python_executable(project_root: Path) -> Path:
    """获取嵌入式Python可执行文件路径

    优先使用 .venv 虚拟环境(开发模式),如果不存在则使用 python/ 目录(便携式部署)
    """
    if _is_active_python_environment_root(project_root):
        return Path(sys.executable).resolve()

    if os.name == "nt":  # Windows
        venv_python = project_root / ".venv" / "Scripts" / "python.exe"
        portable_python = project_root / "python" / "python.exe"
    else:
        venv_python = project_root / ".venv" / "bin" / "python"
        portable_python = project_root / "python" / "bin" / "python"

    # 优先使用虚拟环境
    if venv_python.exists():
        return venv_python
    return portable_python


def get_embedded_python(project_root: Path | None = None) -> Path:
    """获取嵌入式Python可执行文件路径（兼容别名）

    Args:
        project_root: 项目根目录，如果为 None 则自动检测

    Returns:
        Python 可执行文件路径
    """
    if project_root is None:
        project_root = get_project_root()
    return get_embedded_python_executable(project_root)


def get_embedded_venv_python(project_root: Path | None = None) -> Path:
    """获取虚拟环境 Python 可执行文件路径

    Args:
        project_root: 项目根目录，如果为 None 则自动检测

    Returns:
        虚拟环境 Python 可执行文件路径
    """
    if project_root is None:
        project_root = get_project_root()

    if _is_active_python_environment_root(project_root):
        return Path(sys.executable).resolve()

    if os.name == "nt":  # Windows
        return project_root / ".venv" / "Scripts" / "python.exe"
    return project_root / ".venv" / "bin" / "python"


def is_embedded_python_ready(project_root: Path | None = None) -> bool:
    """检查嵌入式 Python 是否准备好

    Args:
        project_root: 项目根目录，如果为 None 则自动检测

    Returns:
        是否准备好
    """
    if project_root is None:
        project_root = get_project_root()
    return get_embedded_python_executable(project_root).exists()


def get_embedded_python_info(project_root: Path | None = None) -> dict[str, str | bool]:
    """获取嵌入式 Python 信息

    Args:
        project_root: 项目根目录，如果为 None 则自动检测

    Returns:
        包含 Python 信息的字典:
        - path: Python 可执行文件路径
        - mode: 环境模式 (venv/portable/none)
        - ready: 是否准备好
    """
    if project_root is None:
        project_root = get_project_root()

    python_path = get_embedded_python_executable(project_root)
    mode = get_environment_mode(project_root)
    ready = python_path.exists()

    return {
        "path": str(python_path),
        "mode": mode,
        "ready": ready,
    }


def is_embedded_python_installed(project_root: Path) -> bool:
    """检查嵌入式Python是否已安装(支持虚拟环境和便携式两种模式)"""
    mode = get_environment_mode(project_root)
    if mode == "none":
        return False
    python_exe = get_embedded_python_executable(project_root)
    return python_exe.exists()


def download_file_with_progress(
    url: str,
    dest_path: Path,
    description: str = "下载",
    max_retries: int = 3,
) -> bool:
    """下载文件并显示进度，支持断点续传 + 失败重试

    大文件（如 torch wheel ~2.6GB）在弱网下常因连接中断报 IncompleteRead。
    本函数：
    - 每次重试时，若 dest_path 已有部分内容，通过 HTTP Range 头断点续传（追加写）；
    - 服务端不支持 Range（返回 200 而非 206）时回退为整体重下（覆盖写）；
    - 最多重试 max_retries 次，全部失败才返回 False。

    Args:
        url: 下载地址
        dest_path: 目标文件路径
        description: 日志描述
        max_retries: 最大重试次数（含首次）
    """
    chunk_size = 65536  # 64KB，比 8KB 更适配大文件
    last_pct = -1

    for attempt in range(1, max_retries + 1):
        try:
            # 断点续传：dest 已有部分内容时，用 Range 头从断点继续
            existing = dest_path.stat().st_size if dest_path.exists() else 0
            # 注意：不能用 Mozilla/5.0 等浏览器 UA。部分镜像（如南大镜像
            # mirror.nju.edu.cn）的 nginx 反爬规则会对以 Mozilla 开头的 UA
            # 返回 302 自重定向死循环，导致 urllib 报 "infinite loop"。
            # 用中性的非浏览器 UA 即可正常下载。
            headers = {"User-Agent": "VibeOCR-Downloader/1.0"}
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"

            req = Request(url, headers=headers)
            logger.info(
                "[%s] 正在下载: %s（第 %d/%d 次%s）",
                description,
                url,
                attempt,
                max_retries,
                f"，断点续传 {existing // 1024 // 1024}MB" if existing else "",
            )

            with urlopen(req, timeout=30) as response:
                # 206 = Partial Content（续传成功）；200 = 服务端忽略 Range，整体返回
                is_resume = response.status == 206
                # 续传时 content-length 是剩余大小，整体总大小 = 已下载 + 剩余
                remaining = int(response.headers.get("content-length", 0))
                total_size = (existing + remaining) if is_resume else remaining
                downloaded = existing if is_resume else 0
                last_pct = -1

                mode = "ab" if is_resume else "wb"
                # 非续传模式需先清空已有部分文件
                if not is_resume and existing > 0:
                    dest_path.unlink()

                with open(dest_path, mode) as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            progress = int(downloaded / total_size * 100)
                            # 每 10% 记一条日志（避免日志爆炸）
                            if progress >= last_pct + 10:
                                last_pct = (progress // 10) * 10
                                logger.info(
                                    "[%s] 进度: %d%% (%dMB / %dMB)",
                                    description,
                                    progress,
                                    downloaded // 1024 // 1024,
                                    total_size // 1024 // 1024,
                                )

            logger.info("[%s] 下载完成", description)
            return True

        except Exception as e:
            logger.error("[%s] 下载失败（第 %d 次）: %s", description, attempt, e)
            if attempt < max_retries:
                logger.info("[%s] 将重试（保留已下载部分用于断点续传）...", description)
            else:
                return False
    return False


DOWNLOAD_REASON_OK = "ok"
DOWNLOAD_REASON_HTTP_ERROR = "http_error"
DOWNLOAD_REASON_SHA_MISSING = "sha_missing"
DOWNLOAD_REASON_SHA_MISMATCH = "sha_mismatch"
DOWNLOAD_REASON_EXCEPTION = "exception"


def _source_label(url: str) -> str:
    """Return a stable human-readable label for an artifact source."""
    for label, marker in (
        ("gh-proxy", "gh-proxy.com"),
        ("ghproxy", "ghproxy.com"),
        ("GitHub", "github.com"),
    ):
        if marker in url:
            return label
    return url


def verify_sha256(file_path: Path, sha256_file: Path) -> bool:
    """Verify an artifact against the first digest in a sha256 sidecar."""
    import hashlib

    if not sha256_file.exists():
        return False
    expected = sha256_file.read_text(encoding="utf-8").strip().split()[0].lower()
    actual = hashlib.sha256(file_path.read_bytes()).hexdigest().lower()
    return actual == expected


def download_artifact_multi_source(
    url_candidates: list[str],
    dest_path: Path,
    description: str = "下载",
    max_retries: int = 3,
    sha_candidates: list[str] | None = None,
    sha_dest_path: Path | None = None,
    source_switch_fn: Callable[[str, str], None] | None = None,
) -> tuple[bool, str]:
    """同步多源下载编排：逐源尝试，可选 SHA256 校验，统一失败原因。

    通用多源下载（带 sha 校验）与 Python 运行时（不带 sha）共用此编排，
    统一「多源回退 + 换源提示 + 失败原因结构化」的语义（与 update_service 的
    异步 download_update 对齐，但本函数走同步 urllib 栈，供首启/重装等同步链路调用）。

    单源下载复用 download_file_with_progress（断点续传 + 重试 + 中性 UA），
    故断点续传与 UA 等关键不变量在本层不受影响。

    Args:
        url_candidates: 有序 URL 候选，逐个尝试直至成功（去重保序由调用方负责）
        dest_path: 目标文件路径
        description: 日志/进度描述
        max_retries: 单源最大重试次数（含首次），透传给 download_file_with_progress
        sha_candidates: 若提供，需与 url_candidates 同序的 sha URL；同时提供
            sha_dest_path 时，每源下载 zip 后再下 sha 并校验，校验失败换源
        sha_dest_path: sha 文件目标路径（仅当 sha_candidates 非空时有效）
        source_switch_fn: 某源失败时回调 (source_label, reason)，供 UI 实时提示
            「源 X 失败，切换源 Y…」。source_label 取自 update_service._source_label

    Returns:
        (成功?, 失败原因)：成功时原因为 DOWNLOAD_REASON_OK；
        失败原因复用 update_service.DOWNLOAD_REASON_* 常量集
        （http_error / sha_missing / sha_mismatch / exception），最后一个失败源的
        原因作为返回值（调用方可据此分桶提示用户，与 update_service 行为一致）
    """
    use_sha = bool(sha_candidates) and sha_dest_path is not None
    if use_sha and len(sha_candidates) != len(url_candidates):  # type: ignore[arg-type]
        raise ValueError("sha_candidates 长度必须与 url_candidates 一致（同源配对）")

    last_reason = DOWNLOAD_REASON_HTTP_ERROR
    sha_list = sha_candidates if use_sha else None

    for idx, url in enumerate(url_candidates):
        source_name = _source_label(url)
        logger.info(
            "[%s] 尝试下载源 %d/%d: %s",
            description,
            idx + 1,
            len(url_candidates),
            source_name,
        )
        try:
            ok = download_file_with_progress(
                url, dest_path, description=description, max_retries=max_retries
            )
            if not ok:
                last_reason = DOWNLOAD_REASON_HTTP_ERROR
                logger.warning("[%s] 下载失败，换源: %s", description, url)
                dest_path.unlink(missing_ok=True)
                if source_switch_fn:
                    source_switch_fn(source_name, last_reason)
                continue

            # 可选：下载 sha 并校验（仅带 sha 的通用多源下载走此分支）
            if use_sha:
                sha_url = sha_list[idx]  # type: ignore[index]
                sha_ok = download_file_with_progress(
                    sha_url,
                    sha_dest_path,
                    description=f"{description}校验",
                    max_retries=max_retries,  # type: ignore[arg-type]
                )
                if not sha_ok:
                    last_reason = DOWNLOAD_REASON_SHA_MISSING
                    logger.warning(
                        "[%s] 校验文件下载失败，换源: %s", description, sha_url
                    )
                    dest_path.unlink(missing_ok=True)
                    sha_dest_path.unlink(missing_ok=True)  # type: ignore[union-attr]
                    if source_switch_fn:
                        source_switch_fn(source_name, last_reason)
                    continue
                if not verify_sha256(dest_path, sha_dest_path):  # type: ignore[arg-type]
                    last_reason = DOWNLOAD_REASON_SHA_MISMATCH
                    logger.error("[%s] SHA256 校验失败，换源: %s", description, url)
                    dest_path.unlink(missing_ok=True)
                    sha_dest_path.unlink(missing_ok=True)  # type: ignore[union-attr]
                    if source_switch_fn:
                        source_switch_fn(source_name, last_reason)
                    continue

            logger.info("[%s] 下载成功（源 %s）", description, source_name)
            return True, DOWNLOAD_REASON_OK
        except Exception as e:
            last_reason = DOWNLOAD_REASON_EXCEPTION
            logger.error("[%s] 下载异常，换源: %s: %s", description, url, e)
            dest_path.unlink(missing_ok=True)
            if use_sha and sha_dest_path is not None:
                sha_dest_path.unlink(missing_ok=True)
            if source_switch_fn:
                source_switch_fn(source_name, last_reason)

    logger.error("[%s] 所有下载源均失败（最后原因: %s）", description, last_reason)
    return False, last_reason


def install_embedded_python(
    project_root: Path, network_type: Literal["domestic", "international"] = "domestic"
) -> tuple[bool, str]:
    """
    下载并安装 Python 运行时（python-build-standalone）

    相比旧 embeddable 方案的改进：
    - 完整标准库 + 自带 pip，无需 get-pip.py / 手写 ._pth / 手建 site-packages
    - 标准 site 机制，paddle/torch 的 .pth、entry_points、DLL 查找按设计行为工作
    - 上游仅发布 .tar.gz，用标准库 tarfile 解压

    Args:
        project_root: 项目根目录
        network_type: 网络类型（domestic 时优先国内镜像，international 时优先 GitHub 直链）

    Returns:
        (是否成功, 消息)
    """
    # 检查环境模式
    mode = get_environment_mode(project_root)

    # 如果已存在虚拟环境,跳过安装
    if mode == "venv":
        venv_python = get_embedded_python_executable(project_root)
        if venv_python.exists():
            return True, f"检测到虚拟环境: {venv_python}"
        return False, "虚拟环境不完整,请重新创建"

    python_dir = project_root / "python"

    if python_dir.exists():
        # 校验 python.exe 确实存在：解压中断/磁盘满等会留下不完整的 python/
        # 目录，此时 python_dir.exists() 命中短路却返回"已安装"，导致后续
        # 依赖安装因找不到解释器而失败且难以定位。半成品则清理后重装。
        python_exe_check = get_embedded_python_executable(project_root)
        if python_exe_check.exists():
            return True, f"Python 运行时已安装: {python_dir}"
        logger.warning(
            "[环境安装] python/ 存在但缺少 %s，判定为半成品，清理后重装",
            python_exe_check,
        )
        shutil.rmtree(python_dir, ignore_errors=True)

    logger.info("==================================================")
    logger.info("[环境安装] 安装 Python 运行时（python-build-standalone）")
    logger.info("==================================================")

    # 根据网络类型排序下载源：international 优先 GitHub 直链，domestic 优先国内镜像
    urls = list(PYTHON_STANDALONE_URLS)
    if network_type == "domestic":
        # 国内镜像在前
        mirrors = PYTHON_BUILD_STANDALONE_MIRRORS
        urls = [*mirrors, PYTHON_BUILD_STANDALONE_BASE]
    else:
        urls = [PYTHON_BUILD_STANDALONE_BASE, *PYTHON_BUILD_STANDALONE_MIRRORS]

    # 去重保序
    seen: set[str] = set()
    urls = [u for u in urls if not (u in seen or seen.add(u))]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tar_path = temp_path / PYTHON_BUILD_STANDALONE_ASSET.replace(".tar.gz", ".tgz")

        # python-build-standalone 上游不发 sha256，故不传 sha_candidates（跳过校验）。
        # 多源回退/换源/失败原因由 download_artifact_multi_source 统一编排。
        download_ok, _reason = download_artifact_multi_source(
            urls, tar_path, description="Python 运行时", max_retries=3
        )

        if not download_ok:
            return (
                False,
                f"无法下载 Python 运行时，请手动下载:\n"
                f"{PYTHON_BUILD_STANDALONE_BASE}\n"
                f"（国内镜像可访问 https://mirror.nju.edu.cn/github-release/"
                f"astral-sh/python-build-standalone/{PYTHON_BUILD_STANDALONE_TAG}/ ）\n"
                f"解压 {PYTHON_BUILD_STANDALONE_ASSET} 内的 python/ 到: {python_dir}",
            )

        logger.info(
            "[环境安装] 下载完成，正在解压 %s...", PYTHON_BUILD_STANDALONE_ASSET
        )
        try:
            python_dir.mkdir(parents=True, exist_ok=True)
            # tar.gz 内顶层为 install_only/python/，需 flatten 到 project_root/python/
            # （python_dir 本身即 python/，故去掉 install_only/ 和内层 python/ 两层前缀）
            with tarfile.open(tar_path, "r:gz") as tar:
                members = tar.getmembers()
                for member in members:
                    rel = member.name
                    # 去掉 install_only/ 首层前缀（若存在）
                    if rel.startswith("install_only/"):
                        rel = rel[len("install_only/") :]
                    # 去掉 python/ 首层前缀（python_dir 本身即 python/，避免 python/python/ 嵌套）
                    if rel.startswith("python/"):
                        rel = rel[len("python/") :]
                    if not rel:
                        continue
                    # 防御 path traversal
                    if rel.startswith(("/", "\\")) or ".." in rel.replace(
                        "\\", "/"
                    ).split("/"):
                        continue
                    target = python_dir / rel
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        src = tar.extractfile(member)
                        if src is not None:
                            with src, open(target, "wb") as dst:
                                dst.write(src.read())
                    elif member.issym() or member.islnk():
                        # standalone 偶尔含符号链接（如 python3 → python3.13），跳过以保 Windows 兼容
                        continue
            logger.info("[环境安装] 解压完成")
        except Exception as e:
            # 解压失败时清理半成品目录，避免误判为已安装
            shutil.rmtree(python_dir, ignore_errors=True)
            logger.error("[环境安装] 解压失败: %s", e)
            return False, f"解压失败: {e}"

    python_exe = get_embedded_python_executable(project_root)
    if not python_exe.exists():
        # 清理半成品，避免下次启动因 python_dir.exists() 短路误判为已安装
        shutil.rmtree(python_dir, ignore_errors=True)
        return False, (
            f"解压后未找到 {python_exe}，下载源可能损坏（已尝试 {len(urls)} 个源）\n"
            f"请手动下载并解压: {PYTHON_BUILD_STANDALONE_BASE}"
        )

    # build-standalone 自带 pip，做一次健康检查（失败不阻断，pip install 阶段会再报错）
    try:
        result = subprocess.run(
            [str(python_exe), "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode == 0:
            logger.info("[环境安装] pip 可用: %s", result.stdout.strip())
        else:
            logger.warning(
                "[环境安装] pip 自检失败: %s",
                result.stderr[-200:] if result.stderr else "",
            )
    except Exception as e:
        logger.warning("[环境安装] pip 自检异常: %s", e)

    return True, f"Python 运行时安装成功: {python_exe}"


def reinstall_embedded_python(
    project_root: Path,
    network_type: Literal["domestic", "international"] = "domestic",
    progress_callback: Callable[[str, str], None] | None = None,
) -> tuple[bool, str]:
    """强制删除现有 python/ 目录后重新安装 Python 运行时。

    删除范围：仅 project_root/python/ 整个目录。
    不删除：.venv、config/、resources/、logs/、模型缓存、机器检测缓存。
    删除 python/ 后 OCR 依赖随之消失，调用方应在成功后继续装依赖。

    Args:
        project_root: 项目根目录
        network_type: 网络类型
        progress_callback: 进度回调 (stage, message)

    Returns:
        (是否成功, 消息)
    """
    python_dir = project_root / "python"

    def report(stage: str, msg: str):
        logger.info("[%s] %s", stage, msg)
        if progress_callback:
            progress_callback(stage, msg)

    report(
        "环境安装",
        f"正在清理旧目录: {python_dir}（仅删除 python/，不影响配置/缓存/日志）",
    )
    shutil.rmtree(python_dir, ignore_errors=True)

    report("环境安装", "清理完成，开始重新安装 Python 运行时...")
    return install_embedded_python(project_root, network_type)


def check_current_environment_dependencies() -> dict[str, bool]:
    """检查当前运行Python环境的生产依赖是否已安装

    用于检测 PySide6, Pillow 等生产环境依赖

    Returns:
        依赖状态字典
    """
    dependencies = {
        "PySide6": False,
        "PIL": False,
    }

    for pkg in dependencies:
        try:
            __import__(pkg)
            dependencies[pkg] = True
        except ImportError:
            dependencies[pkg] = False

    return dependencies


def check_embedded_environment_dependencies(
    project_root: Path, use_cache: bool = True
) -> dict[str, bool]:
    """检查嵌入式Python环境的OCR依赖是否已安装

    用于检测 PaddlePaddle, PaddleX 等OCR功能依赖
    注意：paddlepaddle-gpu 和 paddlepaddle 是二选一关系，
    检测 paddle 模块即可（GPU或CPU版本都会导入为 paddle）

    Args:
        project_root: 项目根目录
        use_cache: 是否使用缓存（默认True）

    Returns:
        依赖状态字典，包含:
        - paddlepaddle: 是否安装了PaddlePaddle（GPU或CPU版本）
        - paddleocr: 是否安装了PaddleOCR
        - is_gpu: 是否是GPU版本（可选字段）
    """
    # 1. 尝试使用缓存
    if use_cache:
        is_valid, cached_data = is_cache_valid(project_root)
        if is_valid and cached_data:
            cached_deps = cached_data.get("dependencies", {})
            # 缓存缺 dependencies 字段或为空（如首启从未检测过、或缓存 schema
            # 变更后未重建）→ 视为缓存失效，落入下方实时检测。
            # 旧逻辑在此处 return {} 会静默跳过检测，导致设置页表格全部显示
            # "未安装"、is_embedded_environment_ready 误报缺失。
            if cached_deps:
                python_exe = get_embedded_python_executable(project_root)

                # ---- 决定要复核哪些项 ----
                # (a) 缓存显示 False 的项：每次都复核（用户可能刚装完）。
                # (b) 缓存超过 CACHE_TTL_DAYS：对 True 的项也复核一次，防止
                #     "用户删了 site-packages 但缓存仍报已装"的假阳性。
                #     本函数在 DependencyCheckTask(QRunnable) 后台线程执行，
                #     复核不阻塞 UI。
                stale_pkgs = [pkg for pkg, ok in cached_deps.items() if not ok]
                from vibeocr.backend.runtime_state import (
                    CACHE_TTL_DAYS,
                    get_cache_age_seconds,
                )

                age = get_cache_age_seconds(project_root)
                ttl_expired = age is not None and age > CACHE_TTL_DAYS * 86400
                ttl_recheck_pkgs = (
                    [pkg for pkg, ok in cached_deps.items() if ok]
                    if ttl_expired
                    else []
                )
                recheck_pkgs = stale_pkgs + ttl_recheck_pkgs

                if recheck_pkgs and python_exe.exists():
                    verified = _quick_verify_deps(python_exe)
                    changed = False
                    for pkg in recheck_pkgs:
                        # 复核结果覆盖缓存值（True→False 或 False→True 都接受）
                        new_val = bool(verified.get(pkg, False))
                        if cached_deps.get(pkg) != new_val:
                            cached_deps[pkg] = new_val
                            changed = True
                    if changed:
                        if ttl_expired:
                            logger.info(
                                "[依赖检测] 缓存超过 %d 天 TTL，已用实时结果复核刷新",
                                CACHE_TTL_DAYS,
                            )
                        else:
                            logger.info("[依赖检测] 缓存过期，已用实时结果刷新")
                        has_gpu, cuda_version = detect_gpu()
                        create_cache_entry(
                            project_root,
                            cached_deps,
                            {"has_gpu": has_gpu, "cuda_version": cuda_version},
                        )
                return cached_deps
            logger.info("[依赖检测] 缓存无 dependencies 字段，执行实时检测")

    # 2. 执行实际检测（委托给 _check_imports 原语，模块清单由 OCR_CHECK_MODULES 统一）
    python_exe = get_embedded_python_executable(project_root)

    if not python_exe.exists():
        return {}

    dependencies = _check_imports(python_exe)
    if dependencies.get("paddlepaddle"):
        print("[依赖检测] PaddlePaddle已安装")

    # 3. 更新缓存
    has_gpu, cuda_version = detect_gpu()
    hardware_info = {
        "has_gpu": has_gpu,
        "cuda_version": cuda_version,
    }
    create_cache_entry(project_root, dependencies, hardware_info)

    return dependencies


def check_embedded_environment_dependencies_fresh(
    project_root: Path,
) -> dict[str, bool]:
    """强制重新检测依赖（忽略缓存）"""
    return check_embedded_environment_dependencies(project_root, use_cache=False)


def check_dependencies(project_root: Path) -> dict[str, bool]:
    """检查嵌入式Python的依赖是否已安装

    注意：此函数检测的是嵌入式环境的依赖，包括生产依赖和OCR依赖。
    对于生产环境依赖检测，请使用 check_current_environment_dependencies()
    对于仅检测OCR依赖，请使用 check_embedded_environment_dependencies()

    注意：paddlepaddle-gpu 和 paddlepaddle 是二选一关系，
    检测 paddle 模块即可（GPU或CPU版本都会导入为 paddle）
    """
    python_exe = get_embedded_python_executable(project_root)

    if not python_exe.exists():
        return {}

    # OCR 依赖委托给 _check_imports 原语
    dependencies: dict[str, bool] = _check_imports(python_exe)

    # 生产依赖（PySide6/PIL）单独检测，timeout 较短
    for pkg in ["PySide6", "PIL"]:
        try:
            result = subprocess.run(
                [str(python_exe), "-c", f"import {pkg}"],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            dependencies[pkg] = result.returncode == 0
        except Exception:
            dependencies[pkg] = False

    return dependencies


def is_production_environment_ready() -> tuple[bool, list[str]]:
    """检查生产环境是否就绪

    Returns:
        (是否就绪, 缺失的依赖列表)
    """
    deps = check_current_environment_dependencies()
    missing = [pkg for pkg, installed in deps.items() if not installed]
    return len(missing) == 0, missing


def _module_name_matches(missing_module: str, module: str) -> bool:
    """判断 import 报错的缺失模块名是否指向被探测模块自身（A 类：本体残缺）

    `import fonttools` 报 `No module named 'fonttools'` 时，missing_module
    与 module 都是 'fonttools' → True（本体残缺）。
    `import mineru` 报 `No module named 'torch'` 时，missing_module='torch'
    ≠ module='mineru' → False（间接依赖）。

    处理边界：
    - 大小写/连字符/下划线归一（FontTools vs fonttools）。
    - 顶层包名匹配（missing_module 含 '.' 时取首段，如 'fonttools.sub' → 'fonttools'）。
    """
    a = missing_module.lower().replace("-", "_").split(".")[0]
    b = module.lower().replace("-", "_").split(".")[0]
    return a == b


def _probe_module(
    python_exe: Path, module: str, pkg: str
) -> tuple[bool, bool, str | None]:
    """双层检测一个 OCR 模块：发行版是否存在 + 是否可导入

    解决"装了发行版但 import 失败"被误判为"未安装"的问题（典型场景：
    ``mineru[core]`` 的间接依赖 torch/paddle/opencv/rapid-table 没装完时，
    ``import mineru`` 抛 ModuleNotFoundError，旧逻辑静默判 False，掩盖了
    "包已装但依赖损坏"的真实状态）。

    两层：
    1. ``importlib.metadata.version('<pkg>')`` 判**发行版是否存在**（标准库，
       无新依赖）。现代包（如 mineru）不暴露 ``__version__``，metadata 是权威源。
    2. ``import <module>`` 判**是否可导入**（间接依赖是否完整）。

    发行版不存在时跳过 import 探测（省一次 subprocess，且 import 必然失败）。
    发行版存在但 import 失败时，logger.warning 落盘"间接依赖未完成"提示，
    便于排查（stderr 截断到 200 字）。

    Args:
        python_exe: 目标（便携）Python 可执行文件
        module: import 模块名（如 "mineru"）
        pkg: pip 包名/发行版名（如 "mineru"），用于 metadata 查询

    Returns:
        (installed, usable, missing_module)：
        - installed：发行版是否存在（metadata 查询成功）
        - usable：是否可导入（import 成功）
        - missing_module：import 失败时从 stderr 抓取的缺失模块名（str | None）。
          usable=True 时恒为 None。供设置页表格状态列显示"已安装，缺 xxx"。
    """
    from vibeocr.backend.services.env_config import (
        OCR_CHECK_TIMEOUTS,
        OCR_DIST_NAME_ALIASES,
    )

    # 同一 import 模块可能来自不同发行版名（如 paddle ← paddlepaddle-gpu /
    # paddlepaddle-cpu / paddlepaddle）。任一候选发行版存在即视为已安装，
    # 否则只查归一 key 会漏掉 GPU/CPU 专用包，误报"装了却缺失"。
    dist_candidates = (pkg, *OCR_DIST_NAME_ALIASES.get(pkg, ()))
    metadata_code = (
        "import importlib.metadata as m, sys\n"
        + "".join(
            "try:\n"
            "    m.version(" + repr(c) + "); sys.exit(0)\n"
            "except m.PackageNotFoundError:\n"
            "    pass\n"
            for c in dist_candidates
        )
        + "sys.exit(1)\n"
    )
    # 第 1 层：metadata 判发行版存在
    try:
        meta_result = subprocess.run(
            [str(python_exe), "-c", metadata_code],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        installed = meta_result.returncode == 0
    except Exception:
        installed = False

    if not installed:
        # 发行版不存在 → import 必然失败，跳过第 2 层
        return False, False, None

    # 第 2 层：import 判可导入（间接依赖是否完整）
    # import_stderr 在 except 分支保留：subprocess.run 抛异常时记录异常名，
    # 正常分支记录子进程 stderr，供下方 import 失败时的 warning 落盘排查。
    import_stderr = ""
    try:
        result = subprocess.run(
            [str(python_exe), "-c", f"import {module}"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=OCR_CHECK_TIMEOUTS.get(module, 15),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        usable = result.returncode == 0
        import_stderr = result.stderr or ""
    except Exception as exc:
        usable = False
        import_stderr = f"(子进程异常: {exc})"

    # import 失败时从 stderr 抓缺失模块名，供 UI 状态列显示"已安装，缺 xxx"。
    # usable=True 时 missing_module 恒为 None（函数末尾兜底赋值）。
    missing_module = None
    if not usable:
        # 发行版存在但 import 失败：区分两种根因，给出准确诊断而非笼统的
        # "间接依赖未完成"。
        #
        # A. 包本体残缺（.dist-info 残留但模块文件缺失）：
        #    `import fonttools` 报 `No module named 'fonttools'` —— 被探测的
        #    模块自身就找不到。此时"发行版已装"的 metadata 记录来自残留的
        #    `fonttools-*.dist-info` 目录（上次装到一半中断/被手动删了部分文件）。
        #    fonttools 是纯 Python 无间接依赖，绝不是"间接依赖未完成"。
        #    更糟的是 `pip install fonttools` 会看到 .dist-info 报 "already
        #    satisfied" 跳过 → import 永远失败（用户报告"装几次还失败"的根因），
        #    补装时需 --force-reinstall 才能真正修好。
        #
        # B. 间接依赖未完成（mineru 依赖的 torch/paddle 缺失）：
        #    `import mineru` 报 `No module named 'torch'` —— 缺的是别的模块。
        import re as _re

        m = _re.search(r"No module named '([^']+)'", import_stderr)
        if m:
            missing_module = m.group(1)
        stderr_tail = import_stderr[-200:]

        if missing_module is not None and _module_name_matches(missing_module, module):
            # A 类：包本体残缺
            logger.warning(
                "[依赖检测] %s 安装残缺：发行版元数据存在（残留 .dist-info）"
                "但 import %s 失败（%s）。补装时将强制重写文件。原始错误: %s",
                pkg,
                module,
                missing_module,
                stderr_tail,
            )
        elif missing_module is not None:
            # B 类：间接依赖未完成
            logger.warning(
                "[依赖检测] %s 发行版已装但 import 失败，间接依赖未完成"
                "（缺 %s）。原始错误: %s",
                pkg,
                missing_module,
                stderr_tail,
            )
        else:
            # 其他 import 错误（非 ModuleNotFoundError）
            logger.warning(
                "[依赖检测] %s 发行版已装但 import 失败（可能间接依赖未完成）: %s",
                pkg,
                stderr_tail,
            )

    return installed, usable, missing_module


def _check_imports(python_exe: Path) -> dict[str, bool]:
    """检测嵌入式 Python 可导入哪些 OCR 模块（双层检测，单一实现，消除重复）

    遍历 env_config.OCR_CHECK_MODULES + OCR_CHECK_LEAF_MODULES，对每个模块用
    _probe_module 做双层检测（发行版是否存在 + 是否可导入），结果以包名为 key
    返回 **usable** 值。

    返回签名与旧版一致：``{包名: 是否可导入}``，保持所有调用方向后兼容
    （``is_embedded_environment_ready`` 等语义"可用即可用"不变）。
    关键改进：发行版存在但 import 失败时（间接依赖损坏），usable 仍判 False
    （不掩盖真实不可用状态），但 _probe_module 会 logger.warning 落盘指向
    "间接依赖未完成"，便于排查。

    Args:
        python_exe: 目标 Python 可执行文件

    Returns:
        {包名: 是否可导入}，如 {"paddlepaddle": True, "torch": False}
    """
    from vibeocr.backend.services.env_config import (
        OCR_CHECK_LEAF_MODULES,
        OCR_CHECK_MODULES,
    )

    deps: dict[str, bool] = {}
    for module, pkg in OCR_CHECK_MODULES.items():
        _installed, usable, _missing = _probe_module(python_exe, module, pkg)
        deps[pkg] = usable
    # paddlex[ocr] leaf 包：顶层 paddleocr import 不触发其检查（装饰器仅实例化时
    # 检查），单独探测以暴露便携安装中途失败导致的漏装。
    for module, pkg in OCR_CHECK_LEAF_MODULES.items():
        _installed, usable, _missing = _probe_module(python_exe, module, pkg)
        deps[pkg] = usable
    return deps


def _check_imports_detailed(python_exe: Path) -> dict[str, tuple[bool, bool]]:
    """检测各 OCR 模块的双层状态（发行版是否存在, 是否可导入）

    与 ``_check_imports`` 的区别：返回 ``(installed, usable)`` 二元组而非仅 usable，
    供补装逻辑区分三类状态：
    - ``(True, True)``  → 已装且可用，补装跳过。
    - ``(False, False)`` → 未安装，补装走普通 ``pip install``。
    - ``(True, False)`` → **残缺安装**（.dist-info 残留但 import 失败），补装需
      ``--force-reinstall`` 才能真正写入模块文件（否则 pip 报 already satisfied 跳过，
      永远修不好，见 _probe_module 的 A 类诊断）。

    Args:
        python_exe: 目标 Python 可执行文件

    Returns:
        {包名: (installed, usable)}，key 与 _check_imports 一致（OCR_CHECK_MODULES value）。
    """
    from vibeocr.backend.services.env_config import (
        OCR_CHECK_LEAF_MODULES,
        OCR_CHECK_MODULES,
    )

    deps: dict[str, tuple[bool, bool]] = {}
    for module, pkg in OCR_CHECK_MODULES.items():
        installed, usable, _missing = _probe_module(python_exe, module, pkg)
        deps[pkg] = (installed, usable)
    # paddlex[ocr] leaf 包同表纳入，供 install_missing_dependencies 据此判断
    # "顶层可用但 leaf 缺失" → 重装承载顶层包补齐传递树。
    for module, pkg in OCR_CHECK_LEAF_MODULES.items():
        installed, usable, _missing = _probe_module(python_exe, module, pkg)
        deps[pkg] = (installed, usable)
    return deps


def _check_imports_with_missing(
    python_exe: Path,
) -> dict[str, tuple[bool, bool, str | None]]:
    """检测各 OCR 模块的三元状态（含 import 失败时缺失的模块名）

    与 ``_check_imports_detailed`` 的区别：多返回一个 ``missing_module`` 字段，
    供设置页依赖表格状态列显示"已安装，缺 torch"这类精确诊断（而非笼统的
    "未安装"）。``missing_module`` 仅在 ``usable=False`` 且 stderr 含
    ``No module named 'xxx'`` 时非 None。

    Args:
        python_exe: 目标 Python 可执行文件

    Returns:
        {包名: (installed, usable, missing_module)}。
    """
    from vibeocr.backend.services.env_config import (
        OCR_CHECK_LEAF_MODULES,
        OCR_CHECK_MODULES,
    )

    deps: dict[str, tuple[bool, bool, str | None]] = {}
    for module, pkg in OCR_CHECK_MODULES.items():
        deps[pkg] = _probe_module(python_exe, module, pkg)
    for module, pkg in OCR_CHECK_LEAF_MODULES.items():
        deps[pkg] = _probe_module(python_exe, module, pkg)
    return deps


def check_dependencies_status_detailed(
    project_root: Path,
) -> dict[str, tuple[bool, bool, str | None]]:
    """强制重新检测依赖的三元状态（含缺失模块名），供设置页依赖树展示。

    与 check_embedded_environment_dependencies_fresh 的区别：返回三元组
    (installed, usable, missing_module) 而非仅 usable 布尔，让状态列能显示
    "已安装，缺 torch" 这类精确诊断。忽略缓存（设置页是实时状态入口）。

    Args:
        project_root: 项目根目录

    Returns:
        {包名: (installed, usable, missing_module)}。Python 运行时不存在时返回 {}。
    """
    python_exe = get_embedded_python_executable(project_root)
    if not python_exe.exists():
        return {}
    return _check_imports_with_missing(python_exe)


def _quick_verify_deps(python_exe: Path) -> dict[str, bool]:
    """轻量验证依赖是否实际已安装（用于校验缓存）

    只做简单的 import 检测，不获取 GPU 信息，速度较快。
    委托给 _check_imports 原语，模块清单由 OCR_CHECK_MODULES 统一管理。
    """
    return _check_imports(python_exe)


def is_embedded_environment_ready(project_root: Path) -> tuple[bool, list[str]]:
    """检查嵌入式OCR环境是否就绪

    Args:
        project_root: 项目根目录

    Returns:
        (是否就绪, 缺失的依赖列表)
    """
    # 首先检查 Python 运行时是否存在
    python_exe = get_embedded_python_executable(project_root)
    if not python_exe.exists():
        return False, ["Python 运行时未安装"]

    deps = check_embedded_environment_dependencies(project_root)
    # 只检查 OCR 核心依赖，排除 is_gpu 等元数据字段。
    # markdown 纳入检查：它已从 exe 包排除，由便携 Python 安装供 worker 用，
    # 缺失会导致 OCR 子进程崩溃，故与 paddleocr/mineru 同等要求。
    # PDF 后端依赖（pymupdf/fastapi/uvicorn/pydantic/fonttools）同理：
    # 已从主 exe 排除，缺失会导致 PDF 子进程启动崩溃，纳入必需检测。
    required_deps = [
        "paddlepaddle",
        "paddleocr",
        "mineru",
        "markdown",
        # PDF 后端子进程依赖
        "pymupdf",
        "fastapi",
        "uvicorn",
        "pydantic",
        "fonttools",
    ]
    missing = [pkg for pkg in required_deps if pkg not in deps or not deps[pkg]]

    # 缓存显示缺失时，做一次轻量验证排除过期缓存
    if missing:
        verified = _quick_verify_deps(python_exe)
        still_missing = [pkg for pkg in missing if not verified.get(pkg, False)]
        if still_missing != missing:
            # 缓存已过期，用验证结果更新
            print("[依赖检查] 缓存已过期，使用实时检测结果")
            missing = still_missing
            # 重新写入缓存
            for pkg, installed in verified.items():
                deps[pkg] = installed
            has_gpu, cuda_version = detect_gpu()
            hardware_info = {
                "has_gpu": has_gpu,
                "cuda_version": cuda_version,
            }
            create_cache_entry(project_root, deps, hardware_info)

    return len(missing) == 0, missing


def get_dependency_versions(python_exe: Path) -> dict[str, str]:
    """获取各 OCR 依赖的版本号（用于设置页状态表格展示）。

    优先用 ``importlib.metadata.version('<pkg>')``（标准库，Py 3.8+），
    这是获取发行版版本的权威方式。失败时（极少数旧包无 metadata）回退
    ``import <module>; print(getattr(module, '__version__', ''))``。

    为什么改 metadata：现代 mineru 包**不暴露** ``__version__`` 属性，
    旧逻辑 ``getattr(mineru, '__version__', '')`` 恒返回空串，导致设置页
    表格显示"（版本未知）"，掩盖真实版本。importlib.metadata 是 PEP 566
    的权威来源（[Python 讨论]建议弃用 __version__，统一走 metadata）。

    Args:
        python_exe: 目标 Python 可执行文件

    Returns:
        {pip包名: 版本号字符串}，未安装/无版本号为空串
    """
    from vibeocr.backend.services.env_config import (
        OCR_CHECK_MODULES,
        OCR_CHECK_TIMEOUTS,
        OCR_DIST_NAME_ALIASES,
    )

    versions: dict[str, str] = {}
    for module, pkg in OCR_CHECK_MODULES.items():
        # 第 1 层：importlib.metadata.version（权威源）
        # 同一模块可能来自不同发行版名（见 OCR_DIST_NAME_ALIASES），取首个命中
        # 候选的版本；结果仍归一到 canonical key（pkg），设置页表格 key 不变。
        dist_candidates = (pkg, *OCR_DIST_NAME_ALIASES.get(pkg, ()))
        metadata_code = (
            "from importlib.metadata import version, PackageNotFoundError\n"
            + "".join(
                "try:\n"
                "    print(version(" + repr(c) + ")); raise SystemExit\n"
                "except PackageNotFoundError:\n"
                "    pass\n"
                for c in dist_candidates
            )
        )
        try:
            result = subprocess.run(
                [str(python_exe), "-c", metadata_code],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=15,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if result.returncode == 0 and result.stdout.strip():
                versions[pkg] = result.stdout.strip()
                continue
        except Exception:
            pass

        # 第 2 层回退：getattr(module, '__version__', '')（兼容极旧包）
        try:
            result = subprocess.run(
                [
                    str(python_exe),
                    "-c",
                    f"import {module}; print(getattr({module}, '__version__', ''))",
                ],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=OCR_CHECK_TIMEOUTS.get(module, 15),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            versions[pkg] = result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            versions[pkg] = ""
    return versions


def _build_paddle_requirements(
    specs: dict[str, str],
    use_gpu: bool,
    cuda_version: str | None,
    network_type: Literal["domestic", "international"],
    report_fn: Callable[[str, str], None],
) -> list[tuple[str, str, str]]:
    """构建 paddle 项（GPU/CPU 包名 + index URL 选择）。

    只构建 paddle 这一项（最复杂、需复用的逻辑）；paddleocr/mineru/torch 项由调用方
    自行拼接（它们的 index 是 pip_source / torch_index，属调用方局部值）。

    Args:
        specs: _load_dep_specs() 返回的依赖规格
        use_gpu: 是否安装 GPU 版本
        cuda_version: CUDA 版本 cu-tag（如 "cu126"）
        network_type: 网络类型（决定 torch 镜像）
        report_fn: 日志回调 (stage, msg)

    Returns:
        [(paddle 展示名, paddle 包规格, paddle index URL)]
    """
    import re as _re

    # 打包环境 version.json 用 _KEY_ALIASES 把 paddlepaddle-gpu 归一为 paddlepaddle；
    # 开发环境 pyproject 保留 paddlepaddle-gpu。两端兼容取规格。
    raw_paddle_spec = specs.get("paddlepaddle-gpu") or specs["paddlepaddle"]
    # 提取完整 constraint 串（含 local version +cu126、多段约束、!= 等）。
    # 注意：!= 单独使用无意义（不指定要哪个版本），但组合约束里可能出现，
    # 故纳入匹配。无 constraint 时（仅包名）返回空串，pip 装最新版。
    _ver_m = _re.search(r"(==|!=|>=|<=|~=|>|<).+", raw_paddle_spec)
    paddle_version_constraint = _ver_m.group(0) if _ver_m else ""
    paddle_gpu_spec = f"paddlepaddle-gpu{paddle_version_constraint}"
    paddle_cpu_spec = f"paddlepaddle{paddle_version_constraint}"

    default_gpu_tag = "cu126"
    if use_gpu and cuda_version:
        paddle_package = paddle_gpu_spec
        paddle_index = (
            f"https://www.paddlepaddle.org.cn/packages/stable/{cuda_version}/"
        )
        paddle_name = f"PaddlePaddle GPU ({cuda_version})"
        report_fn("依赖安装", f"检测到 CUDA {cuda_version}，安装 GPU 版本")
    elif use_gpu:
        paddle_package = paddle_gpu_spec
        paddle_index = (
            f"https://www.paddlepaddle.org.cn/packages/stable/{default_gpu_tag}/"
        )
        paddle_name = f"PaddlePaddle GPU ({default_gpu_tag})"
        report_fn("依赖安装", f"安装 GPU 版本（默认 {default_gpu_tag}）")
    else:
        paddle_package = paddle_cpu_spec
        paddle_index = "https://www.paddlepaddle.org.cn/packages/stable/cpu/"
        paddle_name = "PaddlePaddle CPU"
        report_fn("依赖安装", "使用CPU版本")

    return [(paddle_name, paddle_package, paddle_index)]


def _is_gpu_requirement(name: str) -> bool:
    """判断 requirements 项是否为 GPU 专用包（回退 PyPI 会装成 CPU 版，不可回退）

    通过展示名（tuple 元素 0）子串匹配，与 _is_installed 的判别风格一致。
    - "PaddlePaddle GPU (cu126)" → True（PyPI 无 GPU wheel）
    - "PyTorch CUDA (cu126)"     → True（PyPI torch 为 CPU 版）
    - "PaddlePaddle CPU"/"PaddleOCR"/"MinerU" → False（可回退 PyPI）
    """
    return "GPU" in name or "CUDA" in name


class InstallCancelled(Exception):
    """用户取消安装（协作式取消，区别于 QThread.terminate() 的强杀）。

    由 _run_pip 在检测到 cancel_event 被置位后抛出，_install_paddle_stack
    在外层捕获并转为 (False, "用户已取消安装") 返回。
    """


def _run_pip(
    cmd: list[str],
    timeout: int = 600,
    cancel_event: threading.Event | None = None,
    on_proc: Callable[[subprocess.Popen], None] | None = None,
) -> subprocess.CompletedProcess:
    """运行 pip 命令，支持协作式取消与子进程句柄交出。

    用 subprocess.Popen 启动子进程，通过 on_proc 回调把 Popen 句柄交给调用方
    （通常是 InstallWorker），使其能在取消/关闭时真正 kill 掉 pip 子进程，
    而非留下孤儿进程（旧代码用 subprocess.run + QThread.terminate() 会导致
    pip 子进程变孤儿、Python 层 timeout 失效）。

    取消语义：cancel_event 被置位时，立即 kill 子进程并抛 InstallCancelled。
    超时语义：timeout 为 wall-clock 上限，到期抛 subprocess.TimeoutExpired。

    Args:
        cmd: 命令列表（含 python -m pip ...）
        timeout: wall-clock 超时秒数
        cancel_event: 取消事件；非 None 且被 set 时中止
        on_proc: 子进程启动后回调，参数为 Popen 句柄

    Returns:
        CompletedProcess（与 subprocess.run 返回类型兼容：.returncode/.stdout/.stderr）

    Raises:
        InstallCancelled: cancel_event 被置位
        subprocess.TimeoutExpired: 超时
    """
    if cancel_event is not None and cancel_event.is_set():
        raise InstallCancelled("用户已取消命令")
    creation = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # Native tools such as nvidia-smi may emit the active Windows code
        # page even when PYTHONUTF8=1 makes Python expect UTF-8. Never let one
        # undecodable diagnostic kill the communicate() reader thread.
        errors="replace",
        creationflags=creation,
    )
    # 交出句柄，调用方可在取消/关闭时 kill
    if on_proc is not None:
        on_proc(proc)

    deadline = time.monotonic() + timeout
    try:
        # 轮询：每 0.2s 检查一次进程状态、取消事件、超时。
        # communicate() 会阻塞到 EOF，因此用线程异步收尾 + 主线程轮询。
        stdout_buf: list[str] = []
        stderr_buf: list[str] = []
        comm_exc: BaseException | None = None

        def _communicate() -> None:
            nonlocal comm_exc
            try:
                out, err = proc.communicate()
                stdout_buf.append(out or "")
                stderr_buf.append(err or "")
            except BaseException as e:  # 需捕获含 KeyboardInterrupt 在内的所有
                # 异常并透传给主线程（comm_exc），不在后台线程静默吞掉
                comm_exc = e

        comm_thread = threading.Thread(target=_communicate, daemon=True)
        comm_thread.start()

        while True:
            if cancel_event is not None and cancel_event.is_set():
                # 协作式取消：先 kill 子进程，再等待 communicate 收尾
                with contextlib.suppress(ProcessLookupError, OSError):
                    proc.kill()
                from vibeocr.backend.core.constants import Constants

                comm_thread.join(timeout=Constants.Timeout.PROCESS_CANCEL_GRACE)
                raise InstallCancelled("用户取消安装")
            if proc.poll() is not None:
                # 进程已退出，等 communicate 收完剩余输出
                comm_thread.join(timeout=5)
                break
            if time.monotonic() >= deadline:
                with contextlib.suppress(ProcessLookupError, OSError):
                    proc.kill()
                comm_thread.join(timeout=5)
                raise subprocess.TimeoutExpired(cmd, timeout)
            time.sleep(0.2)

        if comm_exc is not None:
            raise comm_exc
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode,
            stdout=stdout_buf[0] if stdout_buf else "",
            stderr=stderr_buf[0] if stderr_buf else "",
        )
    except (InstallCancelled, subprocess.TimeoutExpired):
        raise
    except Exception:
        # 兜底：异常退出时确保子进程被回收，避免孤儿
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.kill()
        raise


def _pkg_in_force_reinstall_set(
    display_name: str, force_reinstall_pkgs: set[str] | None
) -> bool:
    """判断 requirements 项的展示名是否在强制重装集合中。

    ``force_reinstall_pkgs`` 存的是 OCR_CHECK_MODULES 的 pip 包名（归一 key，
    如 "fonttools"/"pymupdf"/"paddleocr"），由 install_missing_dependencies 从
    _check_imports_detailed 收集（installed=True 且 usable=False 的包）。
    安装循环里的 ``name`` 是展示名（如 "FontTools"/"PyMuPDF"/"PaddleOCR GPU (cu126)"），
    需要子串匹配归一后再比较（大小写/连字符/下划线无关）。

    与 install_missing_dependencies._is_installed 的展示名→包名映射风格一致。
    """
    if not force_reinstall_pkgs:
        return False
    norm = display_name.lower().replace("-", "_")
    for pkg in force_reinstall_pkgs:
        p = pkg.lower().replace("-", "_")
        # 展示名可能含后缀（"PaddleOCR GPU (cu126)"），取核心名子串匹配；
        # 纯包名（"fonttools"）则精确匹配。
        if p in norm or norm.startswith(p):
            return True
    return False


def _install_paddle_stack(
    python_exe: Path,
    specs: dict[str, str],
    pip_source: str,
    network_type: Literal["domestic", "international"],
    use_gpu: bool,
    cuda_version: str | None,
    report_fn: Callable[[str, str], None],
    success_msg: str,
    requirements_override: list[tuple[str, str, str]] | None = None,
    cancel_event: threading.Event | None = None,
    on_proc: Callable[[subprocess.Popen], None] | None = None,
    project_root: Path | None = None,
    force_reinstall_pkgs: set[str] | None = None,
    skip_pip_upgrade: bool = False,
    done_msg: str | None = None,
) -> tuple[bool, str]:
    """安装 PaddlePaddle + PaddleOCR + MinerU (+可选 torch) 依赖栈

    install_embedded_dependencies 与 switch_paddle_backend 的共享实现，
    消除 pip 升级、GPU/CPU 分支、torch index 计算、PyPI 回退等重复逻辑。

    Args:
        python_exe: 目标 Python 可执行文件
        specs: _load_dep_specs() 返回的依赖规格
        pip_source: pip 镜像源 URL
        network_type: 网络类型（决定 torch 镜像）
        use_gpu: 是否安装 GPU 版本
        cuda_version: CUDA 版本字符串
        report_fn: 日志回调 (stage, msg)
        success_msg: 全部成功时的返回消息
        requirements_override: 外部传入的 requirements 子集。指定时跳过内部完整列表
            构建，直接安装子集（用于增量安装：只装缺失的包）。None 时构建完整列表。
        cancel_event: 取消事件；非 None 且被 set 时，在下一个包安装前中止并返回取消。
            由 InstallWorker 透传，配合 closeEvent 实现协作式取消（避免强杀线程）。
        on_proc: 每个子进程启动后的回调（参数为 Popen 句柄），供调用方在取消时
            kill 子进程。
        project_root: 项目根目录，用于安装成功后写依赖缓存。None 时跳过写缓存
            （保持与 switch_paddle_backend 等非首启路径的兼容）。
        force_reinstall_pkgs: 需强制重装的 pip 包名集合（小写归一）。集合中的包在
            ``pip install`` 时追加 ``--force-reinstall --no-deps``，用于修复"残缺安装"
            （.dist-info 残留但模块文件缺失，普通 install 会报 already satisfied 跳过）。
            为空/None 时按常规 install。仅 install_missing_dependencies 增量路径传入。
        skip_pip_upgrade: 为 True 时跳过开头的 ``pip install --upgrade pip``。
            单包/批量精准重装无需升级 pip，跳过可减少噪音和潜在失败点。
        done_msg: 全部成功时打印到日志的完成语，覆盖默认的"所有OCR依赖安装完成"。
            单包/批量重装场景用此传"xxx 安装完成"避免误导性的"所有依赖全部安装完毕"。
            None 时用默认值（全量/补装场景本就装一堆，默认措辞合适）。

    Returns:
        (是否成功, 消息)
    """
    try:
        # 升级pip（单包/批量精准重装跳过：精准补漏无需升级 pip，减少噪音和失败点）
        if not skip_pip_upgrade:
            report_fn("依赖安装", "正在升级pip...")
            result = _run_pip(
                [
                    str(python_exe),
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "--retries",
                    "5",
                    "--timeout",
                    "120",
                    "pip",
                    "-i",
                    pip_source,
                ],
                timeout=120,
                cancel_event=cancel_event,
                on_proc=on_proc,
            )
            if result.returncode != 0:
                report_fn(
                    "依赖安装",
                    f"pip升级警告: {result.stderr[-100:] if result.stderr else ''}",
                )

        if requirements_override is not None:
            # 增量模式：直接用外部传入的子集，跳过完整构建
            requirements = list(requirements_override)
        else:
            # 完整模式：构建 paddle + paddleocr + mineru (+GPU torch)
            paddle_reqs = _build_paddle_requirements(
                specs=specs,
                use_gpu=use_gpu,
                cuda_version=cuda_version,
                network_type=network_type,
                report_fn=report_fn,
            )
            requirements = [
                *paddle_reqs,
                ("PaddleOCR", f'"{specs["paddleocr"]}"', pip_source),
                ("MinerU", f'"{specs["mineru"]}"', pip_source),
                # markdown 已从 PyInstaller exe 包排除，由便携 Python 安装，
                # 供 OCR/MinerU worker 子进程的 markdown_to_html 使用。
                # 用 .get() 防御：旧 specs 字典（pyproject 未声明 markdown 时）会缺 key，
                # 避免抛 KeyError 中断整个安装。
                ("Markdown", f'"{specs.get("markdown", "markdown")}"', pip_source),
                # PDF 后端子进程依赖（pdf_backend_process.py 顶层 import）。
                # 已从主 exe 排除，由便携 Python 安装；用 .get() 防御旧 specs 缺 key。
                ("PyMuPDF", f'"{specs.get("pymupdf", "pymupdf")}"', pip_source),
                ("FastAPI", f'"{specs.get("fastapi", "fastapi")}"', pip_source),
                ("Uvicorn", f'"{specs.get("uvicorn", "uvicorn")}"', pip_source),
                ("Pydantic", f'"{specs.get("pydantic", "pydantic")}"', pip_source),
                ("FontTools", f'"{specs.get("fonttools", "fonttools")}"', pip_source),
            ]

            # GPU 环境下安装 torch+CUDA 覆盖 mineru 附带的 CPU 版本
            if use_gpu:
                # 注意：cuda_version 已是 cu-tag（detect_cuda_version 输出，如 "cu126"），
                # 直接用于构造 index URL，不要再查 CUDA_VERSION_MAP。
                default_gpu_tag = "cu126"
                paddle_cuda_tag = cuda_version or default_gpu_tag
                torch_cuda_tag = TORCH_CUDA_MAP.get(paddle_cuda_tag, "cu126")
                pytorch_mirror_name = (
                    "nju" if network_type == "domestic" else "official"
                )
                torch_index = get_pytorch_mirror(pytorch_mirror_name, torch_cuda_tag)
                requirements.append(
                    (
                        f"PyTorch CUDA ({torch_cuda_tag})",
                        "torch torchvision",
                        torch_index,
                    )
                )
                report_fn("依赖安装", f"将安装 PyTorch CUDA ({torch_cuda_tag})")
                # torch wheel 自带完整的 CUDA 12.x + cuDNN 9 运行时（torch/lib 目录），
                # paddlepaddle-gpu (cu126, CUDA 12 构建) 所需的 cublas64_12.dll 等全部
                # 由 torch/lib 提供，OCRService._setup_cuda_dll_path 会注册该目录。
                # 因此无需额外安装 nvidia-*-cu12 / cu13 系列包。

        # 收集单个包的失败，循环结束后统一汇总。旧逻辑遇首个失败即 return，
        # 导致排在后面的 fonttools/fastapi 等小包被跳过，需二次补装（用户痛点）。
        # 现改为：单个包失败记录后 continue，尽量多装其余包；全成功才返回成功。
        failed: list[tuple[str, str]] = []

        for name, package_spec, index_url in requirements:
            # 协作式取消：在每个包安装前检查取消事件。
            # 已启动的子进程由 _run_pip 内部的 cancel_event 检查负责 kill，
            # 这里负责在包之间快速中止（避免启动下一个 pip）。
            if cancel_event is not None and cancel_event.is_set():
                report_fn("依赖安装", "安装已取消")
                return False, "用户已取消安装"
            report_fn("依赖安装", f"正在安装 {name}...")
            report_fn("依赖安装", f"包规格: {package_spec}")
            report_fn("依赖安装", f"使用源: {index_url}")

            # package_spec 可能含多个包（空格分隔，如 "torch torchvision"），
            # 必须拆成独立的 argv 元素传给 pip，否则 pip 把整个字符串当成一个非法 requirement。
            # 同时剥离冗余的引号（subprocess 传 list 不经过 shell，引号会变成参数的一部分）。
            raw_args = (
                package_spec.split()
                if isinstance(package_spec, str)
                else list(package_spec)
            )
            pkg_args = [a.strip('"').strip("'") for a in raw_args]

            # 残缺安装修复：force_reinstall_pkgs 中的包追加 --force-reinstall --no-deps。
            # 普通补装会因残留 .dist-info 报 already satisfied 跳过，永远修不好；
            # --force-reinstall 强制重写文件，--no-deps 避免重装整个依赖树
            # （fonttools 这类纯 Python 包无 deps，且 GPU 包有独立重试/回退逻辑不应加 --no-deps）。
            is_force_reinstall = _pkg_in_force_reinstall_set(name, force_reinstall_pkgs)
            if is_force_reinstall:
                report_fn("依赖安装", f"{name} 检测为残缺安装，强制重写文件")
            reinstall_flags = (
                ["--force-reinstall", "--no-deps"] if is_force_reinstall else []
            )

            # 首次安装走指定镜像源；带 --retries/--timeout 提升大文件（torch ~2.6GB）韧性
            result = _run_pip(
                [
                    str(python_exe),
                    "-m",
                    "pip",
                    "install",
                    "--retries",
                    "5",
                    "--timeout",
                    "120",
                    *reinstall_flags,
                    *pkg_args,
                    "-i",
                    index_url,
                ],
                timeout=600,
                cancel_event=cancel_event,
                on_proc=on_proc,
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "未知错误"
                is_version_not_found = "Could not find a version" in str(
                    error_msg
                ) or "No matching distribution" in str(error_msg)

                if _is_gpu_requirement(name):
                    # GPU 专用包（paddlepaddle-gpu / torch+cu126）：PyPI 只有 CPU 版，
                    # 回退会静默装成 CPU 导致 GPU 永不生效。
                    # 改为对同一镜像源重试最多 2 次（大文件 IncompleteRead 常见）。
                    gpu_retried = False
                    for retry in range(1, 3):
                        report_fn(
                            "依赖安装",
                            f"{name} 安装失败（第 {retry} 次），重试同源镜像...",
                        )
                        result = _run_pip(
                            [
                                str(python_exe),
                                "-m",
                                "pip",
                                "install",
                                "--retries",
                                "5",
                                "--timeout",
                                "120",
                                *reinstall_flags,
                                *pkg_args,
                                "-i",
                                index_url,
                            ],
                            timeout=600,
                            cancel_event=cancel_event,
                            on_proc=on_proc,
                        )
                        if result.returncode == 0:
                            gpu_retried = True
                            break
                    if not gpu_retried:
                        error_msg = result.stderr or result.stdout or "未知错误"
                        logger.error(
                            "%s 安装失败（GPU 包不回退 PyPI），完整输出:\n%s",
                            name,
                            error_msg,
                        )
                        # 记录失败但继续装后续包（用户可二次补装真正失败的），
                        # 避免前面 mineru 失败就跳过末尾的 fonttools 等小包。
                        failed.append(
                            (
                                name,
                                f"{name} 安装失败（已重试，GPU 包不可回退 PyPI）:"
                                f"\n{error_msg[:500]}",
                            )
                        )
                        report_fn(
                            "依赖安装",
                            f"⚠ {name} 安装失败，跳过继续装后续包（稍后可补装）",
                        )
                        continue
                elif is_version_not_found:
                    # 非 GPU 包且镜像源确无此版本：回退官方 PyPI
                    report_fn("依赖安装", f"{name} 安装失败，尝试使用官方PyPI源...")
                    result = _run_pip(
                        [
                            str(python_exe),
                            "-m",
                            "pip",
                            "install",
                            "--retries",
                            "5",
                            "--timeout",
                            "120",
                            *reinstall_flags,
                            *pkg_args,
                        ],
                        timeout=600,
                        cancel_event=cancel_event,
                        on_proc=on_proc,
                    )

                    if result.returncode != 0:
                        error_msg = result.stderr or result.stdout or "未知错误"
                        # 完整 stderr 落盘（UI 只显示截断版），便于排查
                        logger.error("%s 安装失败，完整输出:\n%s", name, error_msg)
                        failed.append((name, f"{name} 安装失败:\n{error_msg[:500]}"))
                        report_fn(
                            "依赖安装",
                            f"⚠ {name} 安装失败，跳过继续装后续包（稍后可补装）",
                        )
                        continue
                else:
                    # 非 GPU 包但非版本问题（如网络中断）：直接失败，不回退
                    logger.error("%s 安装失败，完整输出:\n%s", name, error_msg)
                    failed.append((name, f"{name} 安装失败:\n{error_msg[:500]}"))
                    report_fn(
                        "依赖安装",
                        f"⚠ {name} 安装失败，跳过继续装后续包（稍后可补装）",
                    )
                    continue

            report_fn("依赖安装", f"{name} 安装成功")

        # 循环结束：汇总失败。全成功才走成功路径（刷缓存 + success_msg）；
        # 有失败则返回失败汇总，但此时已尽量多装了其余包，用户二次补装只补真正失败的。
        if failed:
            failed_names = "、".join(n for n, _ in failed)
            detail = "\n\n".join(f"{n}:\n{m}" for n, m in failed)
            report_fn(
                "依赖安装",
                f"安装完成（部分失败）：{failed_names}。可点「补充安装缺失依赖」重试。",
            )
            return False, f"部分依赖安装失败（{failed_names}）：\n\n{detail}"

        report_fn("依赖安装", done_msg or "所有OCR依赖安装完成")
        # 安装成功后刷新依赖缓存，避免设置页表格读到旧值（与 main_window
        # 同步升级路径 _on_sync_finished 的清缓存做法对齐）。
        # 用 update_cache_field 增量写 dependencies，不会覆盖 pending_backend 等。
        if project_root is not None:
            try:
                verified = _quick_verify_deps(python_exe)
                update_cache_field(project_root, "dependencies", verified)
                logger.info("[依赖安装] 已刷新依赖缓存")
            except Exception as e:
                logger.warning("[依赖安装] 刷新依赖缓存失败: %s", e)
        return True, success_msg

    except InstallCancelled:
        return False, "用户已取消安装"
    except subprocess.TimeoutExpired:
        return False, "依赖安装超时（10分钟）"
    except Exception as e:
        return False, f"依赖安装异常: {e}"


def install_embedded_dependencies(
    project_root: Path,
    network_type: Literal["domestic", "international"] = "domestic",
    use_gpu: bool = False,
    cuda_version: str | None = None,
    progress_callback=None,
    force_backend: str | None = None,
    cancel_event: threading.Event | None = None,
    on_proc: Callable[[subprocess.Popen], None] | None = None,
) -> tuple[bool, str]:
    """
    仅安装嵌入式OCR依赖（PaddlePaddle GPU/CPU, PaddleX, MinerU）

    不安装生产依赖（PySide6, Pillow）

    Args:
        project_root: 项目根目录
        network_type: 网络类型
        use_gpu: 是否安装 GPU 版本（优先），False 则安装 CPU 版
        cuda_version: CUDA 版本 cu-tag（如 "cu126"），用于选择对应的 GPU 包
        progress_callback: 进度回调函数，接收 (stage, message) 参数
        force_backend: 强制后端 "gpu" / "cpu" / None。指定时覆盖 use_gpu/cuda_version，
            用于首启让用户选择或设置页切换。None 时走自动检测逻辑。

    Returns:
        (是否成功, 消息)
    """
    python_exe = get_embedded_python_executable(project_root)

    if not python_exe.exists():
        return False, "Python 运行时未安装"

    # force_backend 覆盖自动检测结果
    if force_backend == "gpu":
        use_gpu = True
        if not cuda_version:
            _has_gpu, cuda_version = detect_gpu()
    elif force_backend == "cpu":
        use_gpu = False
        cuda_version = None

    pip_source = get_pip_source(network_type)

    def report(stage: str, msg: str):
        logger.info("[%s] %s", stage, msg)
        if progress_callback:
            progress_callback(stage, msg)

    report("依赖安装", "开始安装OCR依赖...")
    report("依赖安装", f"pip源: {pip_source}")

    specs = _load_dep_specs()
    return _install_paddle_stack(
        python_exe=python_exe,
        specs=specs,
        pip_source=pip_source,
        network_type=network_type,
        use_gpu=use_gpu,
        cuda_version=cuda_version,
        report_fn=report,
        success_msg="OCR依赖安装成功",
        cancel_event=cancel_event,
        on_proc=on_proc,
        project_root=project_root,
    )


def install_backend_dependencies(
    python_exe: Path | None = None,
    *,
    profile: Literal["cpu", "gpu-cu126"] = "cpu",
    network_type: Literal["domestic", "international"] = "international",
    progress_callback=None,
) -> tuple[bool, str]:
    """在指定的普通 Python 环境中联网安装完整后端依赖。

    该入口与便携环境共用同一安装内核，并显式选择 Paddle/PyTorch index；
    因此 wheel 安装用户不会受 PEP 508 无法传播自定义 index 的限制。
    """
    target_python = Path(python_exe or sys.executable).resolve()
    if not target_python.exists():
        return False, f"Python 可执行文件不存在: {target_python}"

    use_gpu = profile == "gpu-cu126"
    cuda_version = "cu126" if use_gpu else None
    pip_source = get_pip_source(network_type)

    def report(stage: str, msg: str) -> None:
        logger.info("[%s] %s", stage, msg)
        if progress_callback:
            progress_callback(stage, msg)

    specs = _load_dep_specs()
    return _install_paddle_stack(
        python_exe=target_python,
        specs=specs,
        pip_source=pip_source,
        network_type=network_type,
        use_gpu=use_gpu,
        cuda_version=cuda_version,
        report_fn=report,
        success_msg=f"后端依赖 profile {profile} 安装成功",
    )


def install_missing_dependencies(
    project_root: Path,
    network_type: Literal["domestic", "international"] = "domestic",
    use_gpu: bool = False,
    cuda_version: str | None = None,
    progress_callback=None,
    force_backend: str | None = None,
    cancel_event: threading.Event | None = None,
    on_proc: Callable[[subprocess.Popen], None] | None = None,
) -> tuple[bool, str]:
    """增量安装：只装 import 失败（缺失/损坏）的依赖，已 import 成功的跳过下载。

    与 install_embedded_dependencies 的区别：安装前先 _check_imports 检测每个包，
    已可导入的包跳过 pip install（实现"非严格意义断点续传"——已装的不会重复下载）。

    Args:
        project_root: 项目根目录
        network_type: 网络类型
        use_gpu: 是否安装 GPU 版本
        cuda_version: CUDA 版本 cu-tag
        progress_callback: 进度回调 (stage, message)
        force_backend: 强制后端 "gpu"/"cpu"/None

    Returns:
        (是否成功, 消息)
    """
    python_exe = get_embedded_python_executable(project_root)
    if not python_exe.exists():
        return False, "Python 运行时未安装"

    # force_backend 覆盖自动检测结果
    if force_backend == "gpu":
        use_gpu = True
        if not cuda_version:
            _has_gpu, cuda_version = detect_gpu()
    elif force_backend == "cpu":
        use_gpu = False
        cuda_version = None

    pip_source = get_pip_source(network_type)

    def report(stage: str, msg: str):
        logger.info("[%s] %s", stage, msg)
        if progress_callback:
            progress_callback(stage, msg)

    report("依赖安装", "开始检测已安装的依赖...")
    report("依赖安装", f"pip源: {pip_source}")

    specs = _load_dep_specs()

    # 1. 构建完整 requirements 列表
    paddle_reqs = _build_paddle_requirements(
        specs=specs,
        use_gpu=use_gpu,
        cuda_version=cuda_version,
        network_type=network_type,
        report_fn=report,
    )
    requirements: list[tuple[str, str, str]] = [
        *paddle_reqs,
        ("PaddleOCR", f'"{specs["paddleocr"]}"', pip_source),
        ("MinerU", f'"{specs["mineru"]}"', pip_source),
        # markdown 已从 PyInstaller exe 包排除，由便携 Python 安装。
        # 用 .get() 防御旧 specs 字典缺 key。
        ("Markdown", f'"{specs.get("markdown", "markdown")}"', pip_source),
        # PDF 后端子进程依赖（与 _install_paddle_stack 完整模式同步）。
        ("PyMuPDF", f'"{specs.get("pymupdf", "pymupdf")}"', pip_source),
        ("FastAPI", f'"{specs.get("fastapi", "fastapi")}"', pip_source),
        ("Uvicorn", f'"{specs.get("uvicorn", "uvicorn")}"', pip_source),
        ("Pydantic", f'"{specs.get("pydantic", "pydantic")}"', pip_source),
        ("FontTools", f'"{specs.get("fonttools", "fonttools")}"', pip_source),
    ]
    if use_gpu:
        default_gpu_tag = "cu126"
        paddle_cuda_tag = cuda_version or default_gpu_tag
        torch_cuda_tag = TORCH_CUDA_MAP.get(paddle_cuda_tag, "cu126")
        pytorch_mirror_name = "nju" if network_type == "domestic" else "official"
        torch_index = get_pytorch_mirror(pytorch_mirror_name, torch_cuda_tag)
        requirements.append(
            (f"PyTorch CUDA ({torch_cuda_tag})", "torch torchvision", torch_index)
        )

    # 2. 检测每个包是否已可 import（用 detailed 版拿 (installed, usable) 二元组）
    report("依赖安装", "正在检测已安装的依赖...")
    # detailed: {pkg: (installed, usable)}；installed=True/usable=False 即残缺安装
    # （.dist-info 残留但 import 失败），补装时需 --force-reinstall 才能真正修好。
    import_detailed = _check_imports_detailed(python_exe)
    # 仅 usable 的扁平视图，供下面 _is_installed（语义不变：可用才算已装）与
    # 全部已装时的缓存写入使用。
    import_status = {pkg: usable for pkg, (_inst, usable) in import_detailed.items()}

    def _pkg_key_for_req(req_name: str) -> str | None:
        """根据 requirements 项展示名映射到 import_detailed 的归一包名 key"""
        if "PaddlePaddle" in req_name:
            return "paddlepaddle"
        if "PyTorch" in req_name:
            return "torch"
        if "PaddleOCR" in req_name:
            return "paddleocr"
        if "MinerU" in req_name:
            return "mineru"
        if req_name == "Markdown":
            return "markdown"
        # PDF 后端依赖：import_detailed 的 key 是 pip 包名(OCR_CHECK_MODULES value)
        if req_name == "PyMuPDF":
            return "pymupdf"
        if req_name == "FastAPI":
            return "fastapi"
        if req_name == "Uvicorn":
            return "uvicorn"
        if req_name == "Pydantic":
            return "pydantic"
        if req_name == "FontTools":
            return "fonttools"
        return None

    def _is_installed(req_name: str) -> bool:
        """根据 requirements 项展示名查 import 状态（usable=True 才算已装）"""
        key = _pkg_key_for_req(req_name)
        return import_status.get(key, False) if key else False

    # paddlex[ocr] leaf 包缺失检测：顶层 paddleocr 的 import 不触发 leaf 检查
    # （@pipeline_requires_extra 仅实例化时检查），故顶层 usable=True 不代表
    # leaf 齐全。若任一 leaf 缺失，承载顶层包（paddleocr）即使 usable 也要重装，
    # 让 pip 重新解析 paddleocr[doc-parser]→paddlex[ocr] 的整条传递树以补齐。
    from vibeocr.backend.services.env_config import (
        LEAF_TO_TOPLEVEL,
        OCR_CHECK_LEAF_MODULES,
    )

    leaf_missing_pkgs = {
        pkg
        for pkg in OCR_CHECK_LEAF_MODULES.values()
        if not import_status.get(pkg, False)
    }
    # 需因 leaf 缺失而重装的承载顶层包集合（leaf→toplevel 映射）
    leaf_triggered_toplevels: set[str] = {
        LEAF_TO_TOPLEVEL[pkg] for pkg in leaf_missing_pkgs if pkg in LEAF_TO_TOPLEVEL
    }
    if leaf_missing_pkgs:
        report(
            "依赖安装",
            f"⚠ 表格识别间接依赖缺失: {', '.join(sorted(leaf_missing_pkgs))}，"
            "将重装承载顶层包以补齐传递树",
        )

    # 3. 过滤掉已装的；收集残缺安装（installed=True/usable=False）需强制重装的包
    subset: list[tuple[str, str, str]] = []
    force_reinstall_pkgs: set[str] = set()
    for name, pkg_spec, index_url in requirements:
        key = _pkg_key_for_req(name)
        installed, usable = (
            import_detailed.get(key, (False, False)) if key else (False, False)
        )
        # leaf 缺失时，承载顶层包(paddleocr)即使 usable 也要重装补齐传递树
        leaf_triggered = key in leaf_triggered_toplevels
        if usable and not leaf_triggered:
            report("依赖安装", f"✓ {name} 已安装，跳过")
        else:
            subset.append((name, pkg_spec, index_url))
            # 残缺安装：metadata 在但 import 失败。普通 pip install 会报
            # "already satisfied" 跳过 → 永远修不好，必须 --force-reinstall。
            # 注意：leaf_triggered 重装不走 force-reinstall（顶层 usable 说明
            # 本体完好，只是传递 leaf 缺失，普通 install 即可触发 pip 重解析）。
            if installed and key and not usable:
                force_reinstall_pkgs.add(key)
                report(
                    "依赖安装",
                    f"⚠ {name} 安装残缺（元数据在但 import 失败），将强制重装",
                )

    # 4. 全部已装
    if not subset:
        report("依赖安装", "所有依赖已安装，无需补装")
        # 增量补装发现全部已装时也刷新缓存，纠正可能过期的 dependencies 字段
        # （如用户手动 pip install 后缓存未更新）。
        try:
            update_cache_field(project_root, "dependencies", import_status)
            logger.info("[依赖补装] 已刷新依赖缓存（全部已装）")
        except Exception as e:
            logger.warning("[依赖补装] 刷新依赖缓存失败: %s", e)
        return True, "所有OCR依赖已安装"

    missing_names = ", ".join(n for n, _, _ in subset)
    report("依赖安装", f"需补装: {missing_names}")

    # 5. 只装子集（残缺安装的包走 --force-reinstall）
    return _install_paddle_stack(
        python_exe=python_exe,
        specs=specs,
        pip_source=pip_source,
        network_type=network_type,
        use_gpu=use_gpu,
        cuda_version=cuda_version,
        report_fn=report,
        success_msg="OCR依赖补装成功",
        requirements_override=subset,
        cancel_event=cancel_event,
        on_proc=on_proc,
        project_root=project_root,
        force_reinstall_pkgs=force_reinstall_pkgs,
    )


def get_direct_dependencies(python_exe: Path, pkg: str) -> list[str]:
    """查询一个已装顶层包的**直接**依赖列表（一层，不递归）。

    供设置页依赖树展开节点用——动态推导 mineru/torch/paddleocr 等各自拉入了哪些
    一层依赖，无需在 env_config 手动维护易漂移的清单。只取一层避免逐包 subprocess
    全树展开太慢；间接依赖的实际缺失由 _probe_module 的 missing_module 单独标注。

    实现用 ``importlib.metadata.requires(pkg)``（标准库），返回 PEP 508 串列表，
    再用 packaging 解析 marker 过滤出**当前环境实际生效**的依赖（剔除仅
    ``extra == "xxx"`` 才拉入的可选依赖，否则会把 paddlex[ocr]/[doc-parser] 的
    全量 leaf 都算进来，与"直接依赖"语义不符）。

    Args:
        python_exe: 目标（便携）Python 可执行文件
        pkg: pip 包名/发行版名（如 "mineru"）

    Returns:
        直接依赖的 pip 包名列表（小写规范化），按 metadata 顺序去重保序。
        包未安装 / 无 requires / 解析失败时返回空列表。
    """
    from vibeocr.backend.services.env_config import OCR_DIST_NAME_ALIASES

    # 同一 import 可能来自不同发行版名（paddlepaddle-gpu/cpu），任一命中即查。
    dist_candidates = (pkg, *OCR_DIST_NAME_ALIASES.get(pkg, ()))
    # 用一次 subprocess 跑 importlib.metadata.requires，输出 JSON 数组避免
    # 多行 requires 的换行解析问题。
    cand_repr = ",".join(repr(c) for c in dist_candidates)
    code = (
        "import importlib.metadata as m, json, sys\n"
        "reqs = []\n"
        f"for c in [{cand_repr}]:\n"
        "    try:\n"
        "        r = m.requires(c)\n"
        "    except Exception:\n"
        "        r = None\n"
        "    if r:\n"
        "        reqs.extend(r)\n"
        "        break\n"
        "json.dump(reqs, sys.stdout)"
    )
    try:
        result = subprocess.run(
            [str(python_exe), "-c", code],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=20,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode != 0:
            return []
        import json

        raw_reqs = json.loads(result.stdout or "[]")
    except Exception:
        return []

    # 解析 marker，过滤掉仅 extra 条件才生效的可选依赖，保留当前环境默认生效项。
    try:
        from packaging.requirements import Requirement
    except ImportError:
        # packaging 不可用时退化为纯名解析（丢失 marker 过滤，但至少给出一层列表）
        from vibeocr.backend.services.env_config import _parse_pep508_name

        return _dedup_preserve_order(_parse_pep508_name(r) for r in raw_reqs)

    names: list[str] = []
    for raw in raw_reqs:
        try:
            req = Requirement(raw)
        except Exception:
            continue
        # marker 为 None → 无条件依赖，保留。
        # marker 存在且 evaluate 为 True → 当前环境生效，保留。
        # marker 含 extra == "..." → 仅可选 extras 拉入，默认环境不生效，剔除。
        if req.marker is not None:
            try:
                if not req.marker.evaluate():
                    continue
            except Exception:
                # marker 求值失败（缺环境变量等），保守保留以便用户能看到。
                pass
        if req.name:
            names.append(req.name.lower())
    return _dedup_preserve_order(names)


def _dedup_preserve_order(items: Iterable[str]) -> list[str]:
    """去重并保留首次出现顺序（items 可含空串/重复项）。"""
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


def install_single_dependency(
    project_root: Path,
    pkg: str,
    network_type: Literal["domestic", "international"] = "domestic",
    progress_callback=None,
    cancel_event: threading.Event | None = None,
    on_proc: Callable[[subprocess.Popen], None] | None = None,
) -> tuple[bool, str]:
    """单独安装一个依赖包（精准补漏）。

    用于设置页依赖表格的"重装"按钮：用户看到某个包未装（特别是 paddlex[ocr]
    的 leaf 包如 scipy/einops），点一下只装这一个，无需重装整个 paddleocr。

    - 顶层包（如 paddleocr）：从 _load_dep_specs() 取完整 spec（含 extras+版本约束，
      如 ``paddleocr[doc-parser]>=3.7.0``），重装会重新解析传递树。
    - leaf 包（如 scipy）：不在 specs 中，用纯包名安装，pip 自动解析其直接依赖
      （如 scipy 依赖 numpy，tokenizers 依赖 huggingface-hub，这些通常已装）。

    Args:
        project_root: 项目根目录
        pkg: pip 包名（归一，如 "scipy"/"paddleocr"/"scikit-learn"）
        network_type: 网络类型
        progress_callback: 进度回调 (stage, message)
        cancel_event: 协作式取消事件
        on_proc: 子进程句柄回调（取消时 kill）

    Returns:
        (是否成功, 消息)
    """
    python_exe = get_embedded_python_executable(project_root)
    if not python_exe.exists():
        return False, "Python 运行时未安装"

    pip_source = get_pip_source(network_type)

    def report(stage: str, msg: str):
        logger.info("[%s] %s", stage, msg)
        if progress_callback:
            progress_callback(stage, msg)

    # 顶层包取完整 spec（含 extras+约束）；leaf 包用纯包名（pip 选最新兼容版）
    specs = _load_dep_specs()
    spec = specs.get(pkg, pkg)
    report("依赖安装", f"开始单独安装: {spec}")
    report("依赖安装", f"pip源: {pip_source}")

    # 复用 _install_paddle_stack 的取消/超时/进度/句柄机制，单元素 requirements。
    # use_gpu/cuda_version 仅对 GPU 包（paddlepaddle-gpu/torch）有意义；单装普通
    # 包时传 False/None 即可，_is_gpu_requirement 不会命中普通包。
    return _install_paddle_stack(
        python_exe=python_exe,
        specs=specs,
        pip_source=pip_source,
        network_type=network_type,
        use_gpu=False,
        cuda_version=None,
        report_fn=report,
        success_msg=f"{pkg} 安装成功",
        requirements_override=[(pkg, spec, pip_source)],
        cancel_event=cancel_event,
        on_proc=on_proc,
        project_root=project_root,
        # 单包精准补漏：跳过 pip 升级减少噪音；完成日志用具体包名而非
        # 误导性的"所有 OCR 依赖安装完成"（用户报告"单包却提示全部安装完毕"的根因）。
        skip_pip_upgrade=True,
        done_msg=f"{pkg} 安装完成",
    )


def install_dependencies_batch(
    project_root: Path,
    packages: list[str],
    network_type: Literal["domestic", "international"] = "domestic",
    progress_callback=None,
    cancel_event: threading.Event | None = None,
    on_proc: Callable[[subprocess.Popen], None] | None = None,
) -> tuple[bool, str]:
    """批量安装多个依赖包（设置页依赖树多选"重装选中项"）。

    与逐个调 install_single_dependency 的区别：用单次 _install_paddle_stack
    调用 + requirements_override=[...] 批量装，pip 在一次会话内处理，进度带
    计数（``批量重装 (i/n)``）；失败汇总（部分失败不阻断后续包）。

    Args:
        project_root: 项目根目录
        packages: pip 包名列表（顶层包用归一名，如 paddleocr；leaf 用纯名如 scipy）
        network_type: 网络类型
        progress_callback: 进度回调 (stage, message)
        cancel_event: 协作式取消事件
        on_proc: 子进程句柄回调（取消时 kill）

    Returns:
        (是否成功, 消息)
    """
    if not packages:
        return True, "无待安装包"

    python_exe = get_embedded_python_executable(project_root)
    if not python_exe.exists():
        return False, "Python 运行时未安装"

    pip_source = get_pip_source(network_type)

    def report(stage: str, msg: str):
        logger.info("[%s] %s", stage, msg)
        if progress_callback:
            progress_callback(stage, msg)

    # 去重保序，避免用户多选同一包重复装
    unique_pkgs = _dedup_preserve_order(packages)
    n = len(unique_pkgs)
    report("批量重装", f"开始批量重装 {n} 个依赖包")

    specs = _load_dep_specs()
    # 构造 requirements 子集：顶层包取完整 spec（重解析传递树），leaf 用纯名。
    # 带计数前缀的展示名让进度日志清晰（批量重装 (i/n)）。
    requirements: list[tuple[str, str, str]] = []
    for i, pkg in enumerate(unique_pkgs, 1):
        spec = specs.get(pkg, pkg)
        requirements.append((f"{pkg} ({i}/{n})", spec, pip_source))
    report("批量重装", f"pip源: {pip_source}")

    return _install_paddle_stack(
        python_exe=python_exe,
        specs=specs,
        pip_source=pip_source,
        network_type=network_type,
        use_gpu=False,
        cuda_version=None,
        report_fn=report,
        success_msg=f"已重装 {n} 个依赖包",
        requirements_override=requirements,
        cancel_event=cancel_event,
        on_proc=on_proc,
        project_root=project_root,
        skip_pip_upgrade=True,
        done_msg=f"已重装 {n} 个依赖包",
    )


def detect_dependency_updates(project_root: Path) -> dict[str, tuple[str, str]]:
    """检测便携 Python 环境中哪些 OCR 依赖版本落后于主程序要求的版本规格。

    仅便携模式（``python/`` 目录）生效：开发态（``.venv``）由 uv 管理，依赖版本
    跟随 uv.lock，不应触发更新提示，短路返回空。

    覆盖安装场景（用户直接覆盖文件升级 app，无 pending_sync.json）下，
    version.json 里的 dep_versions 可能比已装版本新，主程序据此提示用户更新。

    对比逻辑：
    - 规格来源：``_load_dep_specs()``（version.json/pyproject.toml 的约束串）。
    - 已装来源：``get_dependency_versions(python_exe)``（importlib.metadata）。
    - 比较基准：``_load_locked_versions()`` 的锁定版（权威，能检出下界内的升级
      如 3.4.0→3.4.2）；无锁定版时回退约束下界（如 ``>=3.7.0`` → 3.7.0）。
    - 对每个 OCR_CHECK_MODULES 包，与基准比较；已装 < 基准，或未安装/空，记为"需更新"。

    Args:
        project_root: 项目根目录

    Returns:
        {pkg: (installed_version, required_spec)}，仅含需更新的包。
        installed_version 为空串表示未安装；required_spec 为原始约束串（展示用）。
        开发态/无环境时返回空 dict。
    """
    # 仅便携模式生效。开发态 .venv 由 uv 管理，避免对 .venv 已装版本与 uv.lock
    # 误报更新。此前只检查 python_exe.exists()，而 get_embedded_python_executable
    # 优先返回 .venv 的 python，开发态该路径存在 → 误触发检测。
    if get_environment_mode(project_root) != "portable":
        return {}

    python_exe = get_embedded_python_executable(project_root)
    if not python_exe.exists():
        return {}

    specs = _load_dep_specs()
    installed_versions = get_dependency_versions(python_exe)
    # 锁定版本（uv.lock / version.json 的 dep_locked_versions）作为权威比较基准；
    # 缺失时回退到约束下界（_extract_lower_bound），保证旧 version.json 兼容。
    locked_versions = _load_locked_versions()

    from vibeocr.backend.services.env_config import OCR_CHECK_MODULES

    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:
        # 便携 Python 应有 packaging（pip 依赖），但防御性兜底：仅比较字符串前缀。
        Version = None  # type: ignore[assignment]
        InvalidVersion = Exception  # type: ignore[misc,assignment]

    def _extract_lower_bound(spec_str: str) -> str | None:
        """从 PEP 508 规格串提取版本下界（如 'paddleocr[doc-parser]>=3.7.0' → '3.7.0'）。

        无法解析（无约束/复杂约束）时返回 None，调用方按"无法比较→不报更新"处理，
        避免误报。
        """
        import re as _re

        # 取约束操作符 + 版本号（首个 >= / > / == 约束段）
        m = _re.search(r"(>=|>|==|~=)\s*([0-9][0-9a-zA-Z.\-+!]*)", spec_str)
        return m.group(2) if m else None

    updates: dict[str, tuple[str, str]] = {}
    for _module, pkg in OCR_CHECK_MODULES.items():
        # specs 的 key 是归一包名（小写），paddle 特殊：specs 里是 paddlepaddle-gpu
        # 或 paddlepaddle；OCR_CHECK_MODULES value 是 paddlepaddle。取 specs.get(pkg)。
        spec_str = specs.get(pkg) or specs.get(f"{pkg}-gpu")
        if not spec_str:
            continue  # 无规格无法比较

        # 比较基准：优先锁定版（权威，能检出下界内的升级如 3.4.0→3.4.2），
        # 无锁定版时回退约束下界（兼容无 uv.lock / 旧 version.json）。
        required_ver = locked_versions.get(pkg) or _extract_lower_bound(spec_str)
        if required_ver is None:
            continue  # 约束不可解析（如纯包名无版本）

        installed_ver = installed_versions.get(pkg, "")

        # 未安装 → 视为需更新（补装/更新都会装上）
        if not installed_ver:
            updates[pkg] = (installed_ver, spec_str)
            continue

        # 版本比较：优先用 packaging（权威），失败则按字符串前缀长度兜底
        need_update = False
        if Version is not None:
            try:
                if Version(installed_ver) < Version(required_ver):
                    need_update = True
            except InvalidVersion:
                # 非标准版本（如含 local label），按字符串比较兜底
                need_update = str(installed_ver) < str(required_ver)
        else:
            need_update = str(installed_ver) < str(required_ver)

        if need_update:
            updates[pkg] = (installed_ver, spec_str)

    return updates


def uninstall_removed_deps(
    project_root: Path,
    removed_names: list[str],
    progress_callback: Callable[[str, str], None] | None = None,
    cancel_event: threading.Event | None = None,
    on_proc: Callable[[subprocess.Popen], None] | None = None,
) -> tuple[bool, str]:
    """卸载已从 dep_versions 移除的依赖（P4：依赖移除清理）。

    场景：发版者主动从 pyproject.toml 移除某依赖（如不再用 mineru），
    bump_version 计算 removed 列表写入 version.json → updater 透传到
    pending_sync.json → 主程序消费时调用本函数清理 python/Lib/site-packages
    中残留的包，避免占空间。

    安全保障：
    - removed_names 仅来自发版者的 pyproject 移除声明（白名单严格），不会误删
      用户手动装的包（用户手装的包不在 dep_versions 里，自然不会进 removed）。
    - 单个包卸载失败不阻断整体流程（包不存在时 pip 返回非零，按成功对待）。
    - 不卸载 paddle/torch 核心 CUDA 运行时（即便被移除，其它包可能仍依赖其 DLL），
      由调用方保证 removed 列表准确；本函数信任输入。

    Args:
        project_root: 项目根目录
        removed_names: 要卸载的包名列表（已归一化，如 ["mineru"]）
        progress_callback: 进度回调 (stage, message)
        cancel_event: 取消事件
        on_proc: 子进程句柄回调（与 _run_pip 一致）

    Returns:
        (是否成功, 消息)。全部卸载完成（含"包不存在"跳过）即成功。
    """
    if not removed_names:
        return True, "无依赖需移除"

    python_exe = get_embedded_python_executable(project_root)
    if not python_exe.exists():
        return False, "Python 运行时未安装"

    def report(stage: str, msg: str) -> None:
        logger.info("[%s] %s", stage, msg)
        if progress_callback:
            progress_callback(stage, msg)

    report("依赖清理", f"开始卸载已移除的依赖: {', '.join(removed_names)}")

    failed: list[str] = []
    for pkg in removed_names:
        if cancel_event is not None and cancel_event.is_set():
            report("依赖清理", "卸载已取消")
            return False, "用户已取消卸载"
        # pip uninstall -y 非交互卸载。包不存在时 pip 报 "WARNING: Skipping <pkg>"
        # 并返回非零；此处视为成功（目标"不存在"已达成）。
        report("依赖清理", f"正在卸载 {pkg}...")
        try:
            result = _run_pip(
                [str(python_exe), "-m", "pip", "uninstall", "-y", pkg],
                timeout=120,
                cancel_event=cancel_event,
                on_proc=on_proc,
            )
            # returncode != 0 通常是"包未安装"，按成功对待；仅记录 warning 供排查。
            if result.returncode != 0:
                out = (result.stdout or "") + (result.stderr or "")
                if "not installed" in out.lower() or "skip" in out.lower():
                    report("依赖清理", f"{pkg} 未安装，跳过")
                else:
                    report(
                        "依赖清理", f"{pkg} 卸载返回非零（可能部分残留）: {out[:100]}"
                    )
                    logger.warning("[依赖清理] %s 卸载异常: %s", pkg, out[:200])
        except InstallCancelled:
            return False, "用户已取消卸载"
        except subprocess.TimeoutExpired:
            failed.append(pkg)
            report("依赖清理", f"{pkg} 卸载超时")
            logger.warning("[依赖清理] %s 卸载超时", pkg)
        except Exception as e:
            failed.append(pkg)
            report("依赖清理", f"{pkg} 卸载异常: {e}")
            logger.warning("[依赖清理] %s 卸载异常: %s", pkg, e)

    # 刷新依赖缓存（卸载后 import_status 应变化）
    try:
        import_status = _check_imports(python_exe)
        update_cache_field(project_root, "dependencies", import_status)
        logger.info("[依赖清理] 已刷新依赖缓存")
    except Exception as e:
        logger.warning("[依赖清理] 刷新依赖缓存失败: %s", e)

    if failed:
        return False, f"部分依赖卸载失败: {', '.join(failed)}"
    report("依赖清理", "已移除依赖清理完成")
    return True, "已移除依赖清理完成"


def detect_cuda_version(
    cancel_event: threading.Event | None = None,
) -> str | None:
    """
    检测系统CUDA版本

    Returns:
        CUDA版本字符串（如 "cu126"），如果未检测到则返回 None
    """
    # CUDA版本映射到PaddlePaddle支持的版本
    # paddlepaddle-gpu 3.3.1 win wheel 仅提供 cu118 / cu126 / cu129；本项目统一用 cu126。
    cuda_version_map = CUDA_VERSION_MAP

    def find_best_match(major_minor: str) -> str | None:
        """查找最匹配的 PaddlePaddle CUDA 版本"""
        if major_minor in cuda_version_map:
            return cuda_version_map[major_minor]
        # 尝试找到最接近的版本（向下兼容）
        try:
            version_float = float(major_minor)
            best_match = None
            for supported_ver, paddle_tag in cuda_version_map.items():
                if float(supported_ver) <= version_float:
                    best_match = paddle_tag
            return best_match
        except ValueError:
            return None

    try:
        # 方法1: 解析 nvidia-smi 输出获取 CUDA 版本
        # 注意: nvidia-smi 的 --query-gpu 不支持 cuda_version 字段
        # 需要解析 nvidia-smi 的表头输出
        # timeout=10s：nvidia-smi 首次调用在驱动冷启动时可能 >5s（实测触发
        # TimeoutExpired → 被误判为"无 CUDA"）。提到 10s 留足冷启动余量。
        result = _run_pip(
            ["nvidia-smi"],
            timeout=10,
            cancel_event=cancel_event,
        )
        if result.returncode == 0 and result.stdout:
            # 新版 Windows 驱动显示 "CUDA UMD Version: X.Y"，
            # 旧版/其他平台仍为 "CUDA Version: X.Y"。
            import re

            match = re.search(r"CUDA(?:\s+UMD)?\s+Version:\s*(\d+\.\d+)", result.stdout)
            if match:
                cuda_version = match.group(1)
                logger.info("[硬件检测] CUDA版本 (nvidia-smi): %s", cuda_version)

                major_minor = ".".join(cuda_version.split(".")[:2])
                paddle_cuda = find_best_match(major_minor)
                if paddle_cuda:
                    logger.info("[硬件检测] 对应PaddlePaddle CUDA版本: %s", paddle_cuda)
                    return paddle_cuda

    except InstallCancelled:
        return None
    except FileNotFoundError:
        # nvidia-smi 不在 PATH（非 NVIDIA 机器 / 打包态 PATH 裁剪）→ 静默，属正常
        logger.debug("[硬件检测] nvidia-smi 不在 PATH，跳过该方法")
    except subprocess.TimeoutExpired:
        logger.warning("[硬件检测] nvidia-smi 10s 超时（驱动卡死或冷启动），跳过该方法")
    except Exception as e:
        logger.warning("[硬件检测] nvidia-smi 检测失败: %s: %s", type(e).__name__, e)

    try:
        # 方法2: 检查nvcc版本
        result = _run_pip(
            ["nvcc", "--version"],
            timeout=10,
            cancel_event=cancel_event,
        )
        if result.returncode == 0:
            # 输出包含: "release 12.6"
            import re

            match = re.search(r"release\s+(\d+\.\d+)", result.stdout)
            if match:
                cuda_version = match.group(1)
                logger.info("[硬件检测] CUDA版本 (nvcc): %s", cuda_version)

                paddle_cuda = find_best_match(cuda_version)
                if paddle_cuda:
                    logger.info("[硬件检测] 对应PaddlePaddle CUDA版本: %s", paddle_cuda)
                    return paddle_cuda

    except InstallCancelled:
        return None
    except FileNotFoundError:
        logger.debug("[硬件检测] nvcc 不在 PATH（未装 CUDA Toolkit），跳过")
    except subprocess.TimeoutExpired:
        logger.warning("[硬件检测] nvcc 10s 超时，跳过")
    except Exception as e:
        logger.warning("[硬件检测] nvcc 检测失败: %s: %s", type(e).__name__, e)

    logger.warning(
        "[硬件检测] 无法检测CUDA版本（nvidia-smi 与 nvcc 均不可用/失败，"
        "将回退 CPU；若机器确有 NVIDIA GPU，请检查驱动安装与 PATH）"
    )
    return None


def detect_gpu(
    cancel_event: threading.Event | None = None,
) -> tuple[bool, str | None]:
    """
    检测系统是否有可用的NVIDIA GPU及CUDA版本

    Returns:
        (是否有GPU, CUDA版本标识如"cu126"或None)
    """
    try:
        result = _run_pip(
            ["nvidia-smi", "-L"],
            timeout=10,
            cancel_event=cancel_event,
        )
        if result.returncode == 0 and result.stdout.strip():
            logger.info("[硬件检测] 检测到GPU: %s", result.stdout.strip())
            cuda_version = detect_cuda_version(cancel_event)
            return True, cuda_version
    except InstallCancelled:
        return False, None
    except FileNotFoundError:
        logger.debug("[硬件检测] nvidia-smi 不在 PATH（无 NVIDIA GPU 或打包态裁剪）")
    except subprocess.TimeoutExpired:
        logger.warning("[硬件检测] nvidia-smi -L 10s 超时")
    except Exception as e:
        logger.warning("[硬件检测] nvidia-smi -L 失败: %s: %s", type(e).__name__, e)

    logger.info("[硬件检测] 未检测到NVIDIA GPU，将使用CPU版本")
    return False, None


def detect_gpu_info(
    cancel_event: threading.Event | None = None,
) -> dict[str, object]:
    """一次性探测 NVIDIA GPU 的硬件信息（供 UI 展示）

    通过单次 ``nvidia-smi`` 调用解析 GPU 名称、显存、CUDA 版本，
    避免 UI 层多次 shell out。任何环节失败都回退到现有 detect_gpu() 的
    简单结果（仅 has_gpu + cuda），保证调用方总能拿到可用结构。

    Returns:
        ``{"has_gpu": bool, "name": str, "vram_mb": int, "cuda": str | None}``
        - has_gpu：是否有可用的 NVIDIA GPU
        - name：GPU 型号（无 GPU 时为空串）
        - vram_mb：总显存 MB（无 GPU 时为 0）
        - cuda：CUDA 版本 cu-tag（如 "cu126"），无 GPU 或解析失败为 None
    """
    import re

    # 结果默认值（无 GPU / 解析失败）
    info: dict[str, object] = {
        "has_gpu": False,
        "name": "",
        "vram_mb": 0,
        "cuda": None,
    }

    try:
        result = _run_pip(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            timeout=10,
            cancel_event=cancel_event,
        )
        if result.returncode != 0 or not result.stdout.strip():
            # nvidia-smi 不可用 → 回退到 detect_gpu（处理 nvcc 等次要探测路径）
            has_gpu, cuda = detect_gpu(cancel_event)
            info["has_gpu"] = has_gpu
            info["cuda"] = cuda
            return info

        # 取第一块 GPU 的信息（多卡场景仅展示首卡）
        first_line = result.stdout.strip().splitlines()[0]
        # 格式："NVIDIA GeForce RTX 4090, 24564 MiB, 560.94"（nounits 已去掉单位，
        # 但 memory.total 仍带 "MiB"，按逗号拆分后 strip）
        parts = [p.strip() for p in first_line.split(",")]
        name = parts[0] if parts else ""
        vram_mb = 0
        if len(parts) > 1:
            # "24564 MiB" → 24564；纯数字也兼容
            m = re.search(r"(\d+)", parts[1])
            if m:
                vram_mb = int(m.group(1))

        info["has_gpu"] = True
        info["name"] = name
        info["vram_mb"] = vram_mb
        info["cuda"] = detect_cuda_version(cancel_event)
        return info
    except InstallCancelled:
        return info
    except Exception as e:
        logger.warning(
            "[硬件检测] detect_gpu_info 解析失败: %s: %s", type(e).__name__, e
        )
        has_gpu, cuda = detect_gpu(cancel_event)
        info["has_gpu"] = has_gpu
        info["cuda"] = cuda
        return info


def _resolve_cached_use_gpu(project_root: Path) -> bool | None:
    """从机器缓存解析实际 GPU 后端，缓存不足时返回 None。"""
    is_valid, cached_data = is_cache_valid(project_root)
    if not (is_valid and cached_data):
        return None

    pending = cached_data.get("pending_backend")
    if pending == "gpu":
        return True
    if pending == "cpu":
        return False

    hardware_info = cached_data.get("hardware_info") or {}
    if "has_gpu" in hardware_info:
        return bool(hardware_info["has_gpu"])
    return None


def resolve_use_gpu(project_root: Path) -> bool:
    """决定运行时是否使用 GPU（缓存优先 + 探测回退）

    避免在无 GPU 的机器上硬编码 use_gpu=True，让 OCR worker 以 CPU 模式启动。

    策略：
    1. 读 machine_cache 的 hardware_info.has_gpu（依赖检测时由
       check_embedded_environment_dependencies 写入）
    2. 缓存有效且含 has_gpu → 直接返回
    3. 缓存缺失/失效/无 hardware_info → 回退 detect_gpu() 实时探测

    Args:
        project_root: 项目根目录

    Returns:
        是否使用 GPU
    """
    cached = _resolve_cached_use_gpu(project_root)
    if cached is not None:
        return cached

    # 缓存不可用，实时探测
    has_gpu, _cuda_version = detect_gpu()
    return has_gpu


# 进程级运行时 GPU 能力缓存：避免多个 UI widget 各自重复 shell out nvidia-smi。
# 仅缓存 resolve_use_gpu 的结果（尊重 pending_backend），不缓存 detect_gpu_info
# （后者是纯展示用、无副作用，且后端切换后缓存需失效，故不缓存）。
_runtime_gpu_capability_cache: bool | None = None


def get_runtime_gpu_capability(
    project_root: Path, *, detected_has_gpu: bool | None = None
) -> bool:
    """获取运行时是否具备 GPU 推理能力（进程级缓存）

    与 ``resolve_use_gpu`` 的语义一致：基于"实际运行后端"判断，而非单纯的
    物理 GPU 存在性。因此以下两种情况都返回 False：
    - 无符合 CUDA 条件的 NVIDIA GPU
    - 有 GPU 但用户在设置页选择了 CPU 后端（``pending_backend="cpu"``）

    结果做进程级缓存，避免多个 PreprocessOptionsWidget 实例各自探测。
    缓存在进程生命周期内有效——后端切换需要重启才生效（见 switch_paddle_backend
    写入 pending_backend 的注释），故无需主动失效。

    ``detected_has_gpu`` 供已在后台完成物理 GPU 探测的调用方传入。
    缓存中的 pending/current 后端仍优先；缓存不足时直接复用
    该结果，避免再次 shell out ``nvidia-smi``。

    Args:
        project_root: 项目根目录
        detected_has_gpu: 后台已探测的物理 GPU 结果，可选

    Returns:
        运行时是否使用 GPU 后端
    """
    global _runtime_gpu_capability_cache
    if _runtime_gpu_capability_cache is not None:
        return _runtime_gpu_capability_cache

    cached = _resolve_cached_use_gpu(project_root)
    if cached is not None:
        resolved = cached
    elif detected_has_gpu is not None:
        resolved = bool(detected_has_gpu)
    else:
        resolved = resolve_use_gpu(project_root)
    _runtime_gpu_capability_cache = resolved
    return _runtime_gpu_capability_cache


def switch_paddle_backend(
    project_root: Path,
    target: str,
    network_type: Literal["domestic", "international"] = "domestic",
    progress_callback=None,
    cancel_event: threading.Event | None = None,
    on_proc: Callable[[subprocess.Popen], None] | None = None,
) -> tuple[bool, str]:
    """切换 PaddlePaddle 后端（GPU ↔ CPU）

    供设置页调用：卸载当前 paddle（两包名都卸防冲突）→ 安装目标后端 →
    写入 pending_backend 到缓存（下次启动 worker 时生效）。

    paddlepaddle 和 paddlepaddle-gpu 不能共存（都装 paddle 模块），必须先卸两者。
    CUDA 运行时由 torch wheel 的 torch/lib 提供（见 _install_paddle_stack），
    切换后端时无需单独装卸 nvidia 包。

    Args:
        project_root: 项目根目录
        target: "gpu" 或 "cpu"
        network_type: 网络类型
        progress_callback: 进度回调 (stage, message)
        cancel_event: 协作式取消事件，set 后中止卸载/安装
        on_proc: 子进程启动回调，用于记录当前 Popen 供 cancel 时 kill

    Returns:
        (是否成功, 消息)
    """
    if target not in ("gpu", "cpu"):
        return False, f"无效的后端目标: {target}（应为 'gpu' 或 'cpu'）"

    python_exe = get_embedded_python_executable(project_root)
    if not python_exe.exists():
        return False, "Python 运行时未安装"

    def report(stage: str, msg: str):
        logger.info("[%s] %s", stage, msg)
        if progress_callback:
            progress_callback(stage, msg)

    report("后端切换", f"开始切换到 {target.upper()} 后端...")

    try:
        # 1. 卸载 paddle（两包名都卸，防冲突）
        report("后端切换", "卸载现有 PaddlePaddle...")
        uninstall_cmd = [
            str(python_exe),
            "-m",
            "pip",
            "uninstall",
            "-y",
            "paddlepaddle",
            "paddlepaddle-gpu",
        ]
        # 用 Popen + 手动 wait 替代 subprocess.run，支持 cancel_event 中止
        uninstall_proc = subprocess.Popen(
            uninstall_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if on_proc:
            on_proc(uninstall_proc)
        try:
            uninstall_proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            uninstall_proc.kill()
            uninstall_proc.wait(timeout=10)
        # uninstall 即使包不存在也返回 0，无需检查 returncode

        if cancel_event and cancel_event.is_set():
            return False, "后端切换已取消"

        # 2. 安装目标后端（复用 install_embedded_dependencies 的 force_backend）
        report("后端切换", f"安装 {target.upper()} 版 PaddlePaddle...")
        success, msg = install_embedded_dependencies(
            project_root,
            network_type=network_type,
            progress_callback=progress_callback,
            force_backend=target,
            cancel_event=cancel_event,
            on_proc=on_proc,
        )
        if not success:
            return False, f"{target.upper()} 安装失败: {msg}"

        if cancel_event and cancel_event.is_set():
            return False, "后端切换已取消"

        # 3. 写入 pending_backend（下次启动 worker 时 resolve_use_gpu 读取）
        if not update_cache_field(project_root, "pending_backend", target):
            report("后端切换", "警告: 缓存更新失败，切换可能不会在重启后生效")

        report("后端切换", f"已切换到 {target.upper()}，重启后生效")
        return True, f"已切换到 {target.upper()} 后端，重启应用后生效"

    except subprocess.TimeoutExpired:
        return False, "后端切换超时"
    except Exception as e:
        return False, f"后端切换异常: {e}"


def get_project_root() -> Path:
    """Return the install root while preserving the legacy interface."""

    return resolve_app_paths().install_root


def get_workspace_source_paths() -> tuple[Path, ...]:
    """Compatibility shim: production never exposes workspace source roots."""
    return ()


def get_bundled_resources_dir() -> Path:
    """Return bundled resources while preserving the legacy interface."""

    return resolve_app_paths().resources_root


def get_bundled_changelog_path() -> Path | None:
    """Return the bundled changelog while preserving the legacy interface."""

    return resolve_app_paths().changelog_path


def ensure_mineru_models(
    project_root: Path,
    timeout: int | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
) -> tuple[bool, str]:
    """下载 MinerU 所需模型（首次运行时调用）

    用 Popen 启动 mineru.cli.models_download 并逐行读取输出转发到回调，
    实现"首次使用 PDF 时下载模型 + UI 进度提示"。

    Args:
        project_root: 项目根目录
        timeout: 超时时间（秒），None 时取 Constants.Timeout.MINERU_MODEL_DOWNLOAD
        progress_callback: 进度回调 (stage, message)，None 时仅写日志

    Returns:
        (是否成功, 消息)
    """
    import threading

    if timeout is None:
        from vibeocr.backend.core.constants import Constants

        timeout = int(Constants.Timeout.MINERU_MODEL_DOWNLOAD)

    python_exe = get_embedded_python_executable(project_root)
    if not python_exe.exists():
        return False, "Python 未安装"

    def report(stage: str, msg: str):
        logger.info("[%s] %s", stage, msg)
        if progress_callback:
            progress_callback(stage, msg)

    report("模型下载", "正在下载 MinerU 模型（首次使用需数 GB，请耐心等待）...")
    proc: subprocess.Popen | None = None
    # 绑定 Windows Job Object：主进程崩溃/强杀时内核连带终止下载子进程，
    # 避免数 GB 下载进程成为孤儿（持续占用网络/磁盘数十分钟）。
    # 下载正常完成后子进程已退出，Job 句柄关闭无副作用。
    job_guard = JobObjectGuard(name="vibeocr_mineru_models_dl")
    try:
        network = detect_network_source()
        source = "modelscope" if network == "domestic" else "huggingface"
        report("模型下载", f"使用模型源: {source}")

        proc = subprocess.Popen(
            # 必须显式传 -m/--model_type=all:否则 MinerU CLI 进入 click.prompt
            # 交互选择 (pipeline/vlm/all),非 TTY 下被 Aborted! -> returncode=1,
            # 即使用户点"下载模型"也无法完成下载。stdin=DEVNULL 双保险防阻塞。
            [
                str(python_exe),
                "-m",
                "mineru.cli.models_download",
                "-s",
                source,
                "-m",
                "all",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        job_guard.assign_from_popen(proc)

        # 逐行读取转发到回调，避免数 GB 下载期间 UI 无反馈
        def _read_output():
            try:
                assert proc is not None and proc.stdout is not None
                for raw in proc.stdout:
                    text = raw.decode("utf-8", errors="replace").strip()
                    if text:
                        report("模型下载", text)
            except Exception:
                pass

        reader = threading.Thread(
            target=_read_output, daemon=True, name="MinerUModelDl"
        )
        reader.start()
        proc.wait(timeout=timeout)
        reader.join(timeout=5)

        if proc.returncode == 0:
            report("模型下载", "MinerU 模型下载完成")
            return True, "MinerU 模型下载完成"
        return False, f"模型下载失败（退出码 {proc.returncode}）"
    except subprocess.TimeoutExpired:
        if proc is not None:
            proc.kill()
        return False, "模型下载超时"
    except Exception as e:
        return False, f"模型下载异常: {e}"
    finally:
        # 关闭 Job 句柄。下载已完成则 Job 内已无活进程（no-op）；
        # 异常分支已先 proc.kill()；主进程崩溃路径由内核兜底。
        job_guard.close()
