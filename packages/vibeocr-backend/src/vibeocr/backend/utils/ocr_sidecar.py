# src/vibeocr/utils/ocr_sidecar.py
"""OCR 断点续传 sidecar：记录已增量落盘的页，崩溃后可跳过。

存储位置：<product_root>/data/backend/ocr_sessions/<path-slug>.json
其中 path-slug = md5(规范化绝对路径)。（复用 Backend runtime_state
目录与原子写模式。）

设计要点（修复"增量保存导致指纹漂移"的 bug）：
- **按路径命名，不按指纹命名**：OCR 每批 incremental save 会 append 到 PDF
  文件（size/mtime 都变），但文件路径不变。按路径 slug 命名保证同一会话各批
  写同一 sidecar，且重启后续传能按同一定位到它。
- **基线 + 增长校验**：sidecar 存储 `original_size`/`original_mtime_ns`
  （首次创建时捕获的 PDF 状态）。`load_sidecar` 校验当前文件"只增长未回退"
  (`size >= original AND mtime >= original`)，与 incremental save 的 append
  语义一致。若文件被替换/缩小/回退（用户换文件、回滚），返回 None 失效。
- **`refresh_baseline`**：6C 末尾全量压缩会整体重写 PDF（可能变小），此时
  把基线刷新为压缩后的状态，下一次重开才能通过增长校验。

sidecar 是"尽力而为"：写入失败只记日志，不阻断 OCR 主流程。
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from vibeocr.backend.env_manager import get_project_root
from vibeocr.backend.runtime_state import get_cache_dir

logger = logging.getLogger(__name__)

SIDECAR_VERSION = 1
_SIDECAR_SUBDIR = "ocr_sessions"


def compute_fingerprint(file_path: str) -> str:
    """文件指纹 = f"{size}:{mtime_ns}"。O(1)，不读全文件。

    仅作信息字段与诊断用——不再用于 sidecar 文件名或主校验
    （增量保存会让它在会话期漂移）。主校验改用 `original_*` 基线的增长检查。
    """
    st = Path(file_path).stat()
    return f"{st.st_size}:{int(st.st_mtime_ns)}"


def _sessions_dir() -> Path:
    return get_cache_dir(get_project_root()) / _SIDECAR_SUBDIR


def _path_slug(file_path: str) -> str:
    """规范化绝对路径的 md5 hex，作为 sidecar 文件名键。

    路径在增量保存/末尾压缩期间不变，故同会话各批 + 重启续传都定位到同一
    sidecar 文件。
    """
    abspath = str(Path(file_path).resolve())
    return hashlib.md5(abspath.encode("utf-8")).hexdigest()


def sidecar_path(file_path: str) -> Path:
    return _sessions_dir() / f"{_path_slug(file_path)}.json"


def _growth_ok(data: dict, file_path: str) -> bool:
    """增长校验：当前文件相对基线只增长未回退（incremental save 只 append）。

    文件被替换/缩小/旧版回滚 → 返回 False（sidecar 失效）。
    缺基线字段（旧格式 / 损坏）→ 返回 False。
    """
    orig_size = data.get("original_size")
    orig_mtime = data.get("original_mtime_ns")
    if orig_size is None or orig_mtime is None:
        return False
    try:
        st = Path(file_path).stat()
    except OSError:
        return False
    return st.st_size >= int(orig_size) and int(st.st_mtime_ns) >= int(orig_mtime)


def load_sidecar(file_path: str) -> dict | None:
    """读 sidecar；版本不符 / 增长校验失败 / 损坏 → 返回 None。"""
    try:
        p = sidecar_path(file_path)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("version") != SIDECAR_VERSION:
            return None
        if not _growth_ok(data, file_path):
            return None
        return data
    except Exception as e:
        logger.debug("sidecar 读取失败（忽略）: %s", e)
        return None


def save_sidecar(file_path: str, data: dict) -> bool:
    """原子写（tmp + os.replace，复用 machine_cache 模式）。"""
    p = sidecar_path(file_path)
    tmp = p.with_suffix(".json.tmp")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
        return True
    except Exception as e:
        logger.warning("sidecar 写入失败（忽略，不阻断 OCR）: %s", e)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def _new_sidecar(file_path: str) -> dict:
    """新建 sidecar：捕获当前文件状态作为增长校验基线。"""
    path = Path(file_path).resolve()
    st = path.stat()
    return {
        "version": SIDECAR_VERSION,
        "file_path": str(path),
        "fingerprint": compute_fingerprint(file_path),
        "original_size": st.st_size,
        "original_mtime_ns": int(st.st_mtime_ns),
        "completed": False,
        "pages": {},
    }


def mark_pages_saved(
    file_path: str, page_indices: list[int], angles: dict[int, int]
) -> bool:
    """增量合并：把 page_indices 标记为已落盘。angles = {page: preproc_angle}。

    通过 `load_sidecar` 做增长校验：增量保存只让文件增长，校验始终通过，
    故多批结果在同一 sidecar 累积（修复了旧版"指纹漂移丢批次"的 bug）。
    基线不在此更新——保持首会话原始状态，直到 `refresh_baseline`。
    """
    data = load_sidecar(file_path) or _new_sidecar(file_path)
    for idx in page_indices:
        data["pages"][str(idx)] = {
            "has_text_layer": True,
            "ocr_preproc_angle": int(angles.get(idx, 0)),
        }
    data["completed"] = False
    return save_sidecar(file_path, data)


def mark_completed(file_path: str) -> bool:
    """标记 sidecar 为已完成。保留已有 page 记录。

    若 sidecar 文件存在但校验失败（load_sidecar 返回 None），读取原始内容
    保留 pages 记录，而非创建空 sidecar 丢失数据。仅当文件不存在时才新建。
    这镜像 refresh_baseline 的做法（绕过校验读原文）。
    """
    data = load_sidecar(file_path)
    if data is None:
        # 校验失败或不存在。若文件存在，读原始内容保留 pages（避免丢失）。
        p = sidecar_path(file_path)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                # 版本不符（旧格式）→ 退回到新建（无 pages 可保）
                if data.get("version") != SIDECAR_VERSION:
                    data = _new_sidecar(file_path)
            except Exception as e:
                logger.debug("mark_completed 原始读取失败，回退新建: %s", e)
                data = _new_sidecar(file_path)
        else:
            data = _new_sidecar(file_path)
    data["completed"] = True
    return save_sidecar(file_path, data)


def refresh_baseline(file_path: str) -> bool:
    """把 sidecar 的增长校验基线刷新为当前文件状态。

    用于 OCR 末尾全量压缩后：压缩会整体重写 PDF（可能比原文件小），
    若不刷新基线，下次重开会因 `size < original` 通不过增长校验。直接读
    sidecar 原文（绕过 load_sidecar 的校验），更新 `original_size`/
    `original_mtime_ns`/`fingerprint`，再原子写回。
    """
    p = sidecar_path(file_path)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        st = Path(file_path).stat()
        data["original_size"] = st.st_size
        data["original_mtime_ns"] = int(st.st_mtime_ns)
        data["fingerprint"] = compute_fingerprint(file_path)
        return save_sidecar(file_path, data)
    except Exception as e:
        logger.debug("sidecar refresh_baseline 失败（忽略）: %s", e)
        return False


def restore_pending_pages(file_path: str) -> dict[int, int] | None:
    """返回 {page_index: ocr_preproc_angle} 用于续传跳过。

    None 表示：无 sidecar / 增长校验失败（文件被替换/回退）/ 已 completed。
    """
    data = load_sidecar(file_path)
    if data is None or data.get("completed"):
        return None
    return {
        int(k): v.get("ocr_preproc_angle", 0)
        for k, v in data.get("pages", {}).items()
    }
