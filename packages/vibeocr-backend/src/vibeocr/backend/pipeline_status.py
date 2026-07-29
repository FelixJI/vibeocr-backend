"""Backend 管道成功状态的读写。

管道"曾经成功运行"标记的读写。

历史上此模块自行直接读写 ``.vibeocr/cache.json``（绕过 machine_cache），
导致：
1. 校验逻辑与 ``machine_cache.is_cache_valid`` 分叉（不校验 version）；
2. fallback 路径写死 ``version:1``，会把当前 ``CACHE_VERSION`` 降级污染缓存；
3. 重复实现读写原语，与 machine_cache 的写入分叉。

现已收敛到 ``machine_cache`` 的 SSOT API：
- 读：``is_cache_valid``（version + machine_id + 存在性三重校验）
- 写：``update_cache_field``（增量写单字段，保留其余字段，自动走原子写）

由此 pipeline_success 字段自动继承 version/machine_id 校验与原子写，
不再可能写入错位的 version 或损坏文件。

**每日重置**：``pipeline_success[name]`` 存为 ``{"succeeded": bool,
"date": "YYYY-MM-DD"}``。读取时仅当 ``succeeded`` 为真且 ``date`` 是今天
才认为成功；跨天视为未成功（调用方走长超时重新验证一次）。防止模型文件被
删/损坏/换机器后仍用短超时快速失败。旧布尔格式向后兼容（见读取逻辑）。
"""

import logging
from datetime import date

from vibeocr.backend.runtime_state import is_cache_valid, update_cache_field

_logger = logging.getLogger(__name__)

PIPELINE_NAMES = {
    "OCR",
    "PP-StructureV3",
    "PaddleOCR-VL",
    "TABLE_RECOGNITION",
    "FORMULA_RECOGNITION",
    "MinerU",  # 文档解析（首次使用需下载模型，标记成功以跳过重复下载）
}

# 本地推理管道集合（走 PaddleX/registry，非远程 MinerU）。
# ``mark_pipeline_success`` 的调用方据此判断是否标记——必须覆盖此集合的
# 全部成员，否则被遗漏管道的 ``is_pipeline_ever_succeeded`` 永远为 False，
# 导致 SingleRecognitionTab.run_ocr 每次识别都同步构造 QWebEngineView
# （Chromium 冷启动数百毫秒），表现为截图遮罩在点击管道按钮后卡住。
# MinerU 由各自调用方单独硬编码标记（远程 API，独立路径）。
LOCAL_MARKABLE_PIPELINES = frozenset(
    {
        "OCR",
        "PP-StructureV3",
        "PaddleOCR-VL",
        "TABLE_RECOGNITION",
        "FORMULA_RECOGNITION",
    }
)


def _today() -> date:
    """当前日期（可被测试 patch 注入固定值）。"""
    return date.today()


def _is_success_entry_today(entry: object) -> bool:
    """单条 pipeline_success 记录是否表示「今天成功过」。

    - 新格式 dict ``{"succeeded": bool, "date": "YYYY-MM-DD"}``：
      succeeded 为真且 date 是今天才返回 True。
    - 旧格式 bool：直接返回该布尔值（宽容，避免升级即失效；下次 mark 会升级
      为新格式）。非 bool/非 dict 类型视为未成功。
    """
    if isinstance(entry, bool):
        return entry
    if isinstance(entry, dict):
        if not entry.get("succeeded"):
            return False
        entry_date = entry.get("date")
        if not isinstance(entry_date, str):
            return False
        return entry_date == _today().isoformat()
    return False


def is_pipeline_ever_succeeded(pipeline_name: str, project_root) -> bool:
    """管道是否曾在此机器上成功运行过（且是今天）。

    走 ``machine_cache.is_cache_valid`` 三重校验（version + machine_id + 存在），
    缓存无效时返回 False（保守路径：调用方会按"未成功"处理，如延长预加载超时、
    提示"首次使用需下载模型"——均不致命）。跨天的成功标记也返回 False，
    触发调用方用长超时重新验证一次。
    """
    is_valid, data = is_cache_valid(project_root)
    if not is_valid or data is None:
        return False
    entry = data.get("pipeline_success", {}).get(pipeline_name, False)
    return _is_success_entry_today(entry)


def mark_pipeline_success(pipeline_name: str, project_root) -> None:
    """标记管道今天成功运行。

    写入新格式 ``{"succeeded": True, "date": "YYYY-MM-DD"}``。缓存无效时
    **静默不标记**（不创建新缓存）——避免旧 fallback 路径写错
    version/machine_id 污染缓存。下次依赖检测重建缓存后自然会被再次标记。
    """
    is_valid, data = is_cache_valid(project_root)
    if not is_valid or data is None:
        _logger.debug(
            "缓存无效，跳过标记管道 %s 成功（待缓存重建后再标记）", pipeline_name
        )
        return
    ps = dict(data.get("pipeline_success", {}))
    ps[pipeline_name] = {"succeeded": True, "date": _today().isoformat()}
    update_cache_field(project_root, "pipeline_success", ps)
    _logger.debug("管道 %s 标记为今天已成功", pipeline_name)
