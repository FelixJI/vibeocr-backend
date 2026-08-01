"""环境配置模块

集中管理环境相关的配置常量和工具函数。
"""

import sys
from pathlib import Path

# Python 版本（仅保留短版本；完整版本由 PYTHON_VERSION_SHORT + PATCH 拼出）
PYTHON_VERSION_SHORT = "3.13"

# ---------------------------------------------------------------------------
# python-build-standalone 运行时（替代 embeddable 发行版）
# ---------------------------------------------------------------------------
# 上游：https://github.com/astral-sh/python-build-standalone
# 升级时仅改这两个常量：BUILD_TAG（astral release tag）与 PATCH（对应 cpython 补丁号）
PYTHON_BUILD_STANDALONE_TAG = "20260325"  # astral release tag
PYTHON_BUILD_STANDALONE_PATCH = "12"  # cpython 3.13 补丁号 → 3.13.12
# Windows install_only 资产（上游仅发布 .tar.gz，无 .zip）
PYTHON_BUILD_STANDALONE_ASSET = (
    f"cpython-{PYTHON_VERSION_SHORT}.{PYTHON_BUILD_STANDALONE_PATCH}"
    f"+{PYTHON_BUILD_STANDALONE_TAG}"
    "-x86_64-pc-windows-msvc-install_only.tar.gz"
)
# GitHub 直链
PYTHON_BUILD_STANDALONE_BASE = (
    "https://github.com/astral-sh/python-build-standalone/releases/download"
    f"/{PYTHON_BUILD_STANDALONE_TAG}/{PYTHON_BUILD_STANDALONE_ASSET}"
)
# 国内镜像与加速前缀（按优先级顺序尝试）
PYTHON_BUILD_STANDALONE_MIRRORS = [
    # 南大镜像：与上游 release 同步，最稳
    f"https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/"
    f"{PYTHON_BUILD_STANDALONE_TAG}/{PYTHON_BUILD_STANDALONE_ASSET}",
    # ghproxy 公共加速前缀（拼接 GitHub 直链）
    "https://gh-proxy.com/" + PYTHON_BUILD_STANDALONE_BASE,
    "https://ghproxy.com/" + PYTHON_BUILD_STANDALONE_BASE,
]

# ---------------------------------------------------------------------------
# 发布仓库标识（SSOT）—— update_service / about_tab 共享
# ---------------------------------------------------------------------------
# 发布渠道：CNB 仅镜像代码；产物唯一源 GitHub（国内走 gh 代理加速）。
# Gitee 不再作为下载/发版源，仅保留仓库主页链接供关于页展示。
GITHUB_OWNER = "FelixJI"
GITHUB_REPO = "VibeOCR"

# repo 根：仓库主页（关于页"项目主页"链接用）
GITHUB_REPO_BASE = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}"
# Gitee 仓库主页：仅作代码仓库展示（关于页），不参与下载
GITEE_REPO_BASE = "https://gitee.com/felixjii/vibeocr"
# releases 页：发布列表（手动下载兜底链接用）
GITHUB_RELEASES_BASE = f"{GITHUB_REPO_BASE}/releases"
GITHUB_DOWNLOAD_BASE = f"{GITHUB_RELEASES_BASE}/download"  # .../download/v{ver}/{asset}

# GitHub Release API（latest）
GITHUB_API_LATEST = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
)

# GitHub 加速代理前缀（拼接在 GitHub 直链之前，按优先级）
# 与 PYTHON_BUILD_STANDALONE_MIRRORS 一致的加速策略
GITHUB_PROXY_PREFIXES = ["https://gh-proxy.com/", "https://ghproxy.com/"]


def _ordered_download_prefixes(network_type: str) -> list[str]:
    """返回各下载源的「直链前缀」候选，按 network_type 决定优先级顺序。

    每个前缀与 asset 名拼接即得完整下载 URL。约定：
    - 空串 ""：GitHub 裸连（GITHUB_DOWNLOAD_BASE 由调用方拼）
    - 代理前缀：拼在 GitHub 直链之前

    这里改用「前缀」表达，是因为同一源的 zip 与 sha256 需要分别拼 URL，
    但共享同一源序——用前缀列表配对最清晰。
    国内(domestic)：gh-proxy → ghproxy → GitHub 裸连（3 候选）
    海外(international)：GitHub 直连（1 候选）
    未知 network_type 按国际（直连优先）处理。
    """
    if network_type == "domestic":
        return [*GITHUB_PROXY_PREFIXES, GITHUB_DOWNLOAD_BASE]
    return [GITHUB_DOWNLOAD_BASE]


def _asset_url(prefix: str, version: str, asset_name: str) -> str:
    """按源前缀拼单个 asset 的完整下载 URL。

    代理前缀（gh-proxy / ghproxy）需拼在 GitHub 直链之前，GitHub 直连前缀
    自身已是完整基址。
    """
    github_url = f"{GITHUB_DOWNLOAD_BASE}/v{version}/{asset_name}"
    if prefix in GITHUB_PROXY_PREFIXES:
        return prefix + github_url
    return f"{prefix}/v{version}/{asset_name}"


def build_github_asset_urls(
    network_type: str, version: str, asset_name: str
) -> list[str]:
    """构造某个 GitHub release asset 的有序下载候选 URL 列表。

    国内(domestic)：gh-proxy → ghproxy → GitHub 裸连（3 候选）
    海外(international)：GitHub 直连（1 候选）
    未知 network_type 按国际（直连优先）处理。

    Args:
        network_type: "domestic" 或 "international"
        version: 版本号（不含 v 前缀，如 "0.3.1"）
        asset_name: 资产文件名，如 "VibeOCR-v0.3.1-win64.zip"

    Returns:
        有序 URL 候选列表，调用方逐个尝试直至下载成功
    """
    return [
        _asset_url(p, version, asset_name)
        for p in _ordered_download_prefixes(network_type)
    ]


def build_asset_url_pairs(
    network_type: str, version: str, zip_name: str, sha_name: str
) -> list[tuple[str, str]]:
    """构造 zip + 校验文件的成对下载候选（同源序，源序与 build_github_asset_urls 一致）。

    与单文件版本不同：每个候选源同时给出 zip_url 与 sha_url，二者来自同一源、
    同一 tag 目录，确保校验文件和被校验文件确实同源同版——避免此前用
    ``f"{zip_url}.sha256"`` 盲拼、可能下到无关/404 内容的问题。

    Args:
        network_type: "domestic" 或 "international"
        version: 版本号（不含 v 前缀）
        zip_name: zip 资产文件名
        sha_name: 对应 sha256 资产文件名

    Returns:
        有序 (zip_url, sha_url) 候选对列表
    """
    return [
        (
            _asset_url(p, version, zip_name),
            _asset_url(p, version, sha_name),
        )
        for p in _ordered_download_prefixes(network_type)
    ]


# PyTorch CUDA 镜像源
PYTORCH_MIRROR_SOURCES = {
    "nju": "https://mirrors.nju.edu.cn/pytorch/whl",
    "sjtu": "https://mirror.sjtu.edu.cn/pytorch-wheels",
    "official": "https://download.pytorch.org/whl",
}

# 默认 PyTorch 镜像源（国内）
DEFAULT_PYTORCH_MIRROR = "nju"

# 便携式 Python 目录名（与运行时实际使用的 project_root/python/ 一致）
PORTABLE_PYTHON_DIR = "python"

# 配置目录名
CONFIG_DIR = "config"

# ---------------------------------------------------------------------------
# OCR / PDF 依赖检测单一清单源（SSOT）
# ---------------------------------------------------------------------------
# {import 模块名: pip 包名} —— 检测环境时 import 模块名，结果/缓存用包名做 key。
# - paddle 模块：paddlepaddle-gpu / paddlepaddle-cpu / paddlepaddle 均导入为 paddle，
#   故只检 paddle；但它们的发行版名各异，额外候选见 OCR_DIST_NAME_ALIASES。
# - 版本约束不在此处，安装版本来自 pyproject.toml（env_manager._load_dep_specs）
# - PDF 后端依赖（fitz/fastapi/uvicorn/pydantic/fonttools）已从主 exe 排除，
#   由便携 Python 安装供 PDF 子进程用，故与 OCR 依赖同等纳入就绪检测。
OCR_CHECK_MODULES: dict[str, str] = {
    "paddle": "paddlepaddle",
    "paddleocr": "paddleocr",
    "mineru": "mineru",
    "torch": "torch",
    # markdown 已从 exe 包排除，由便携 Python 安装供 OCR/MinerU worker 用，
    # 故纳入便携环境就绪检测，避免装漏导致 worker 子进程崩溃。
    "markdown": "markdown",
    # PDF 后端子进程依赖（pdf_backend_process.py 顶层 import）。
    # 注意 fitz 的 import 名与发行版名不一致：PyMuPDF wheel 提供 fitz 模块。
    "fitz": "pymupdf",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    # 注意 import 名大小写：fonttools wheel 安装的顶层目录是 fontTools（大写 T，
    # 见 RECORD: fontTools/__init__.py）。Python 导入区分大小写（PEP 235），
    # 即便 Windows 文件系统不区分大小写，``import fonttools`` 仍会 ModuleNotFoundError。
    # 之前键用小写 ``fonttools`` 导致探测器永远 import 失败 → 被误判为"安装残缺"
    # → install_missing_dependencies 走 --force-reinstall 无限重装仍修不好（用户实测）。
    # 改为 fontTools 后 import 与项目内 cjk_font_resolver.py 的 ``from fontTools import ...``
    # 一致，依赖检测通过。value 仍是 pip 包名 ``fonttools``，下游 required_deps/缓存/
    # 设置页表格以 value 为 key，不受影响。
    "fontTools": "fonttools",
}

# 同一 import 模块可能来自不同发行版名的额外候选。
# paddle 模块：paddlepaddle-gpu / paddlepaddle-cpu / paddlepaddle 均导入为 paddle，
# 但它们的 PyPI/分发发行版名各异。metadata 第一层探测只查 OCR_CHECK_MODULES
# 的归一 key（"paddlepaddle"）会漏掉 GPU/CPU 专用包（其发行版名不是 paddlepaddle），
# 导致"装了 paddlepaddle-gpu 却误报缺失"。此处补全候选，探测时任一命中即视为已安装；
# 结果 dict 仍用归一 key（"paddlepaddle"），下游（required_deps/缓存/设置页）不受影响。
OCR_DIST_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "paddlepaddle": ("paddlepaddle-gpu", "paddlepaddle-cpu"),
}

# paddlex[ocr] extra 的 leaf 包——表格识别管道 (TableRecognitionPipelineV2) 经
# @pipeline_requires_extra("ocr") 强制要求，但顶层 paddleocr 的 import 不触发其检查
# （装饰器仅在管道实例化时检查），形成探测盲区：便携环境若 paddleocr[doc-parser]
# 安装事务中途失败（镜像 404/超时），这些 leaf 包会漏装，而 import paddleocr 仍成功 →
# cache 误标已装 → 直到用户跑表格识别实例化时才爆炸为无信息的 DependencyError。
# 纳入检测让漏装能在启动期/设置页暴露。
# 注意 sklearn 的 import 名与 pip 包名不一致（scikit-learn）。
OCR_CHECK_LEAF_MODULES: dict[str, str] = {
    "bs4": "beautifulsoup4",
    "einops": "einops",
    "ftfy": "ftfy",
    "latex2mathml": "latex2mathml",
    "premailer": "premailer",
    "regex": "regex",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "sentencepiece": "sentencepiece",
    "tiktoken": "tiktoken",
    "tokenizers": "tokenizers",
}

# leaf→承载顶层包映射（pip 包名）。
# 这些 leaf 全部由 paddleocr[doc-parser] → paddlex[ocr] 的传递依赖拉入，
# 故承载顶层包统一是 paddleocr。leaf 缺失时，补装应重装承载顶层包以重新解析
# 整条传递树（而非逐个 leaf 单装，后者无法覆盖 leaf 自身的传递依赖）。
LEAF_TO_TOPLEVEL: dict[str, str] = dict.fromkeys(
    OCR_CHECK_LEAF_MODULES.values(), "paddleocr"
)

# 各模块 import 检测的 timeout（秒）。
# paddle 首次导入需初始化 CUDA 上下文，显著慢于其他模块。
# paddleocr 同样会间接触发 paddle 的 CUDA 初始化——实测冷启动 ~45s
# （Supervisor preload 日志），30s 会误报「import 失败」，故与 paddle 对齐 60s。
OCR_CHECK_TIMEOUTS: dict[str, int] = {
    "paddle": 60,
    "paddleocr": 60,
    "mineru": 15,
    "torch": 15,
    "markdown": 10,
    # PDF 后端依赖：纯 Python 或轻量扩展，10s 足够。
    "fitz": 10,
    "fastapi": 10,
    "uvicorn": 10,
    "pydantic": 10,
    # 键须与 OCR_CHECK_MODULES 的 import 名一致（fontTools），否则
    # _probe_module 的 OCR_CHECK_TIMEOUTS.get(module, 15) 命不中、回退到默认 15s。
    "fontTools": 10,
    # paddlex[ocr] leaf 包：scipy 首次 import 加载 OpenBLAS 较慢给 15s，
    # tokenizers/sentencepiece/tiktoken 有原生扩展给 10s，纯 Python 包 5s。
    "scipy": 15,
    "tokenizers": 10,
    "sentencepiece": 10,
    "tiktoken": 10,
    "einops": 5,
    "bs4": 5,
    "ftfy": 5,
    "latex2mathml": 5,
    "premailer": 5,
    "regex": 5,
    "sklearn": 10,
}


# -----------------------------------------------------------------------------
# 依赖清单一致性校验
# -----------------------------------------------------------------------------
# OCR_CHECK_MODULES 是人工维护的 SSOT（携带 import 名映射、超时、别名、
# required 子集四套耦合元数据，无法从 pyproject.toml 完全自动推导）。
# 此函数做"漂移检测"：比对 OCR_CHECK_MODULES.values() 与 pyproject.toml 声明的
# OCR 相关依赖，发现不一致时返回告警列表，供启动期 logger.warning 提示开发者。
# 不自动 bump CACHE_VERSION——那需要语义判断，交给人处理更安全。

# 不自动 bump CACHE_VERSION——那需要语义判断，交给人处理更安全。


def _parse_pep508_name(dep_spec: str) -> str:
    """从 PEP 508 依赖规格提取纯包名（小写规范化）。

    例：
        "paddleocr[doc-parser]>=3.7.0" → "paddleocr"
        "paddlepaddle-gpu>=3.3.1"       → "paddlepaddle-gpu"
        "torch >= 2.6.0"                → "torch"
    """
    # PEP 508: name 在最左，后接可选 extras/markers/版本约束。
    # 取首个出现 <,>,=,!,[ ,;,~ 之前的部分作为 name。
    import re

    match = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)", dep_spec)
    return match.group(1).lower() if match else ""


def validate_dep_check_consistency(project_root: Path) -> list[str]:
    """校验 OCR_CHECK_MODULES 与 pyproject.toml 的 OCR 依赖是否同步。

    Returns:
        告警字符串列表（空列表表示一致）。告警类型：
        - pyproject 声明了 OCR 依赖但 OCR_CHECK_MODULES 未覆盖
        - OCR_CHECK_MODULES 覆盖的包在 pyproject 找不到声明
    """
    warnings: list[str] = []
    workspace_backend = project_root / "packages" / "vibeocr-backend" / "pyproject.toml"
    pyproject = (
        workspace_backend
        if workspace_backend.exists()
        else project_root / "pyproject.toml"
    )
    if not pyproject.exists():
        # 打包后无 pyproject.toml，跳过校验（正常运行时路径）
        return warnings

    try:
        import tomllib  # Python 3.11+ 标准库

        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
    except (OSError, ValueError) as e:
        warnings.append(f"无法解析 pyproject.toml：{e}（一致性校验跳过）")
        return warnings

    project = data.get("project", {})
    declared_deps = list(project.get("dependencies", []) or [])
    for optional_deps in project.get("optional-dependencies", {}).values():
        declared_deps.extend(optional_deps)
    # 提取声明的依赖名（小写规范化）
    declared_names = {_parse_pep508_name(d) for d in declared_deps if d}

    # 构造"声明名 → OCR canonical 名"映射，覆盖两类来源：
    # (a) OCR_CHECK_MODULES.values() 本身（paddleocr/torch/mineru/markdown 等）
    # (b) OCR_DIST_NAME_ALIASES 里的别名（paddlepaddle-gpu → paddlepaddle）
    name_to_canonical: dict[str, str] = {}
    for canonical in OCR_CHECK_MODULES.values():
        name_to_canonical[canonical.lower()] = canonical
    for canonical, aliases in OCR_DIST_NAME_ALIASES.items():
        name_to_canonical[canonical.lower()] = canonical
        for a in aliases:
            name_to_canonical[a.lower()] = canonical

    # 视为"已声明的 OCR canonical 名"集合：声明名能映射到 canonical 的才算
    declared_ocr: set[str] = set()
    for name in declared_names:
        canonical = name_to_canonical.get(name)
        if canonical:
            declared_ocr.add(canonical)

    check_modules_names = set(OCR_CHECK_MODULES.values())

    # 漂移 1：OCR_CHECK_MODULES 有但 pyproject 没声明
    not_declared = check_modules_names - declared_ocr
    for pkg in sorted(not_declared):
        warnings.append(
            f"OCR_CHECK_MODULES 包含 '{pkg}'，但 pyproject.toml 未声明对应依赖——"
            f"若新增 OCR 依赖请同时更新两者并 bump CACHE_VERSION"
        )

    # 漂移 2：pyproject 声明了 OCR 依赖但 OCR_CHECK_MODULES 未覆盖
    not_covered = declared_ocr - check_modules_names
    for pkg in sorted(not_covered):
        warnings.append(
            f"pyproject.toml 声明了 OCR 依赖 '{pkg}'，但 OCR_CHECK_MODULES 未覆盖——"
            f"检测时会漏检，请补全或 bump CACHE_VERSION"
        )

    return warnings


def get_pytorch_mirror(
    name: str = DEFAULT_PYTORCH_MIRROR,
    cuda_tag: str = "",
) -> str:
    """获取 PyTorch CUDA 镜像源 URL

    Args:
        name: 镜像源名称
        cuda_tag: CUDA 版本标签，如 "cu126"

    Returns:
        镜像源完整 URL，如 "https://mirrors.nju.edu.cn/pytorch/whl/cu126"
    """
    base = PYTORCH_MIRROR_SOURCES.get(
        name, PYTORCH_MIRROR_SOURCES[DEFAULT_PYTORCH_MIRROR]
    )
    if cuda_tag:
        return f"{base}/{cuda_tag}"
    return base


def is_windows() -> bool:
    """检查是否在 Windows 系统上运行"""
    return sys.platform == "win32"


def is_linux() -> bool:
    """检查是否在 Linux 系统上运行"""
    return sys.platform.startswith("linux")


def is_macos() -> bool:
    """检查是否在 macOS 系统上运行"""
    return sys.platform == "darwin"


def get_project_root() -> Path:
    """获取项目根目录

    委托 env_manager.get_project_root()，保持单一实现源（SSOT）。
    判断逻辑：打包态锚定 exe 所在目录；开发态定位四包 workspace 根。
    统一调用避免两份实现在非标准布局下返回不同结果。

    .. deprecated::
        新代码应使用 ``vibeocr.backend.runtime_layout.resolve_app_paths()`` 获取完整的
        AppPaths（含 data_root/runtime_root/model_cache_root/output_root/config_file）。
        本函数仅返回 install_root，保留供旧调用方兼容。
    """
    # 延迟导入打破循环依赖（env_manager 反向依赖本模块的常量）
    from vibeocr.backend.env_manager import get_project_root as _get_root

    return _get_root()


def get_config_dir() -> Path:
    """获取配置目录"""
    return get_project_root() / CONFIG_DIR


def get_portable_python_dir() -> Path:
    """获取便携式 Python 目录"""
    return get_project_root() / PORTABLE_PYTHON_DIR


def ensure_config_dir() -> Path:
    """确保配置目录存在"""
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


# data 目录名
DATA_DIR = "data"


def get_data_dir() -> Path:
    """获取用户数据目录"""
    return get_project_root() / DATA_DIR


def get_update_cache_dir() -> Path:
    """获取更新下载缓存目录"""
    d = get_data_dir() / "cache" / "update"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_update_settings_path() -> Path:
    """获取更新设置文件路径（skip_version 等）"""
    d = get_data_dir() / "settings"
    d.mkdir(parents=True, exist_ok=True)
    return d / "update_settings.json"


def get_update_progress_path() -> Path:
    """获取更新替换流程进度文件路径。

    替换器（updater.exe / self-update）在替换各阶段写入耗时记录（见
    update_replacer._StageTimer），新版 VibeOCR 启动后由关于页读取展示
    「上次更新各阶段耗时」，方便用户/开发者排查更新慢的瓶颈。

    放 cache/update/ 与 updater.ready 同目录：更新缓存清理时一并删除，
    不会永久残留；失败路径下 _safe_cleanup_artifacts 也覆盖此目录。
    不主动 mkdir——替换器写它时已 ensure parent（与 ready 信号一致）。
    """
    return get_data_dir() / "cache" / "update" / "progress.json"


def get_pending_sync_path() -> Path:
    """获取依赖版本待同步标记文件路径

    updater 在替换应用文件后写入此文件（含变更的 dep_versions），
    新版 VibeOCR 启动时读取并据此用 install_embedded_dependencies 升级 python/，
    升级成功后删除。与 updater_main.py 的写入路径保持一致。
    """
    d = get_data_dir() / "settings"
    d.mkdir(parents=True, exist_ok=True)
    return d / "pending_sync.json"


# 依赖同步失败重试上限：达到后主程序提示用户"建议重装嵌入式 Python"。
# 单次同步失败常为网络/镜像抖动，多次失败则更可能是 python/ 损坏，
# 此时引导用户走 reinstall_embedded_python（设置页入口）比继续重试更有效。
SYNC_MAX_ATTEMPTS = 3
