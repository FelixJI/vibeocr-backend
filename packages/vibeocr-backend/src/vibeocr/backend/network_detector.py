"""统一网络检测模块。

并行探测网络端点，结果持久化到 cache.json，供依赖下载选择使用。
"""

import contextlib
import ssl
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal
from urllib.error import URLError
from urllib.request import Request, urlopen

from vibeocr.backend.runtime_state import (
    CACHE_VERSION,
    generate_machine_id,
    load_cache,
    save_cache,
)

# 探测端点
CHINA_ENDPOINT = "https://paddleocr.bj.bcebos.com"
INTERNATIONAL_ENDPOINT = "https://huggingface.co"

# 缓存有效期
CACHE_TTL_DAYS = 7

# pip 镜像源（pip 源 SSOT：全仓唯一定义，按 network_type 派生）
_PIP_MIRRORS = {
    "domestic": "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/",
    "international": "https://pypi.org/simple",
}


def get_pip_mirror(network_type: Literal["domestic", "international"]) -> str:
    """返回显式网络 profile 对应的 pip simple index。"""
    return _PIP_MIRRORS[network_type]


class NetworkDetector:
    """统一网络检测器，单例模式。"""

    def __init__(self, project_root: Path, force_detect: bool = False) -> None:
        self._project_root = project_root
        self._network_type: Literal["domestic", "international"] = "domestic"

        if force_detect:
            self._detect()
        else:
            self._load_or_detect()

    @property
    def network_type(self) -> Literal["domestic", "international"]:
        return self._network_type

    @property
    def pip_mirror_url(self) -> str:
        return _PIP_MIRRORS[self._network_type]

    def _load_or_detect(self) -> None:
        cache = load_cache(self._project_root)
        if cache and self._is_cache_valid(cache):
            self._apply_cache(cache["network"])
        else:
            self._detect()

    def _is_cache_valid(self, cache: dict) -> bool:
        network = cache.get("network")
        if not network:
            return False
        # version + machine_id 校验与 machine_cache.is_cache_valid 对齐，
        # 避免 bump CACHE_VERSION 后 network 字段仍被当作有效（P1）。
        if cache.get("version") != CACHE_VERSION:
            return False
        if cache.get("machine_id") != generate_machine_id():
            return False
        last_detected = network.get("last_detected")
        if not last_detected:
            return False
        detected_time = datetime.fromisoformat(last_detected)
        return network.get("network_type") in {
            "domestic",
            "international",
        } and datetime.now() - detected_time < timedelta(days=CACHE_TTL_DAYS)

    def _apply_cache(self, network: dict) -> None:
        self._network_type = network["network_type"]

    def _detect(self) -> None:
        results: dict[str, float] = {}
        lock = threading.Lock()

        def test_endpoint(name: str, url: str) -> None:
            # 网络探测：DNS/连接/SSL/超时失败均视为端点不可达，记录 inf
            with contextlib.suppress(URLError, OSError, ssl.SSLError):
                start = time.time()
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                req = Request(
                    url, method="HEAD", headers={"User-Agent": "VibeOCR-Downloader/1.0"}
                )
                with urlopen(req, timeout=5, context=ctx) as resp:
                    if resp.status == 200:
                        with lock:
                            results[name] = time.time() - start
                        return
            with lock:
                results.setdefault(name, float("inf"))

        threads = [
            threading.Thread(target=test_endpoint, args=("china", CHINA_ENDPOINT)),
            threading.Thread(
                target=test_endpoint, args=("international", INTERNATIONAL_ENDPOINT)
            ),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=6)

        china_time = results.get("china", float("inf"))
        intl_time = results.get("international", float("inf"))

        if intl_time < china_time:
            self._network_type = "international"
        else:
            self._network_type = "domestic"

        self._save_to_cache()

    def _save_to_cache(self) -> None:
        cache = load_cache(self._project_root) or {}
        cache["network"] = {
            "last_detected": datetime.now().isoformat(),
            "network_type": self._network_type,
        }
        # 仅在缓存已具备 machine_id 时保留；不补写 version（旧 fallback 写 1
        # 会污染当前 CACHE_VERSION）。无 machine_id 的空缓存说明从未经过
        # create_cache_entry 正常初始化，由后续依赖检测重建，此处只填 network
        # 字段并保留 machine_id（若存在）。
        if "machine_id" not in cache:
            cache["machine_id"] = generate_machine_id()
        # 不写 version——让 machine_cache.is_cache_valid 在 version 缺失时
        # 自然失效重建，避免错误兜底值。
        save_cache(self._project_root, cache)
