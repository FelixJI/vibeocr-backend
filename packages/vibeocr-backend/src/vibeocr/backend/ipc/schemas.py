"""PDF 后端 IPC 共享 schema(pydantic v2)。

双端契约:子进程 FastAPI 服务端 + 主进程 httpx 客户端都 import 这一套。
所有 fitz 相关类型只存在于子进程,主进程只通过这里的 schema 通信。

设计要点:
- PdfPageInfoMirror / PdfDocumentMirror 是主进程持有的"显示快照",
  字段是 PdfPageInfo 的可序列化子集(thumbnail: QPixmap 不传,渲染走独立路由)。
- ModelDiff 是变更操作的回包增量,主进程 apply 到 mirror。
- ProgressEvent 是流式进度(SSE / chunked),phase 区分阶段。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ---- 基础镜像类型 ------------------------------------------------------

class TextLayerInfoMirror(BaseModel):
    """单个文字层信息(对应 models.pdf_document.TextLayerInfo)。"""

    index: int
    text_preview: str
    char_count: int
    bbox: tuple[float, float, float, float]
    color_id: int


class TextBlockMirror(BaseModel):
    """单个 OCR 文本块(对应 models.ocr_result.TextBlock 的可序列化投影)。

    ocr_text_blocks 在 PdfPageInfo 里是 list[TextBlock],跨进程序列化时
    只保留写入 PDF / 预览所需的字段。
    """

    text: str
    score: float
    bbox: tuple[float, float, float, float] | None = None
    polygon: tuple[float, ...] | None = None  # 4 点检测多边形 [x,y,...]，归一化 [0,1000]
    page_idx: int | None = None
    is_manually_edited: bool = False
    label: str = "text"
    order: int = -1


class PdfPageInfoMirror(BaseModel):
    """单页状态镜像(对应 models.pdf_document.PdfPageInfo 的可序列化子集)。

    thumbnail: QPixmap 不跨进程,缩略图走独立渲染路由取字节。
    """

    page_index: int
    rotation: int = 0
    has_text_layer: bool = False
    text_layers: list[TextLayerInfoMirror] = Field(default_factory=list)
    is_scanned: bool = False
    rect: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    ocr_text_blocks: list[TextBlockMirror] = Field(default_factory=list)
    ocr_preproc_angle: int = 0
    deskewed: bool = False


class PdfDocumentMirror(BaseModel):
    """PDF 文档状态镜像(对应 models.pdf_document.PdfDocument 的可序列化投影)。"""

    file_path: str | None = None
    pages: list[PdfPageInfoMirror] = Field(default_factory=list)
    is_modified: bool = False
    has_structural_change: bool = False
    render_dpi: int = 300
    thumbnail_dpi: int = 96


# ---- Model 增量 --------------------------------------------------------

class ModelDiff(BaseModel):
    """变更操作的回包增量。

    主进程收到后 apply 到 mirror:
    - replaced_pages: 整页替换(变更后的完整 PageInfoMirror)
    - structural_change: 结构性变更(删页/插页/重排)→ 主进程直接用 full_model 替换
    - full_model: 结构性变更或全量刷新时携带完整 model
    - modified_flag / structural_flag: 更新 is_modified / has_structural_change
    """

    replaced_pages: list[PdfPageInfoMirror] = Field(default_factory=list)
    structural_change: bool = False
    full_model: PdfDocumentMirror | None = None
    modified_flag: bool | None = None
    structural_flag: bool | None = None
    # 缩略图失效页(主进程清缓存 + 重请)
    invalidated_thumbnails: list[int] = Field(default_factory=list)


# ---- 进度事件 ----------------------------------------------------------

class ProgressPhase(StrEnum):
    """长操作阶段标识,供主进程切换文案/确定 vs 不确定进度条。"""

    LOAD = "load"  # 打开后逐页文字层检测
    RENDER = "render"  # OCR 前置渲染 / 摆正批量渲染
    OCR = "ocr"  # OCR 识别
    WRITE = "write"  # 写文字层
    DETECT = "detect"  # 摆正方向检测
    CORRECT = "correct"  # 摆正逐页纠正
    DELETE = "delete"  # 删除文字层
    SAVE = "save"  # 保存落盘
    EXPORT = "export"  # 导出
    COMPRESS = "compress"  # OCR 末尾整文档压缩（不确定进度）


class ProgressEvent(BaseModel):
    """流式进度事件(SSE / chunked 一行一个 JSON)。"""

    phase: ProgressPhase
    current: int = 0
    total: int = 0  # total==0 表示不确定进度(主进程用滚动条)
    message: str | None = None  # 可选自定义文案
    # 某些阶段附带逐页结果(如 OCR page_done / 摆正 page_done)
    page_index: int | None = None
    page_payload: Any | None = None  # 阶段相关:OCR 块数 / 摆正是否纠正 / 删除残留等


# ---- 请求体 ------------------------------------------------------------

class OpenRequest(BaseModel):
    path: str


class SaveRequest(BaseModel):
    path: str | None = None  # None = 覆盖原文件
    pdf_settings: dict[str, Any] | None = None
    # OCR 分批写层已完成时无需删除并重写全部文字层；普通保存保持 True。
    rewrite_text_layers: bool = True


class RotateRequest(BaseModel):
    pages: list[int]
    angle: int


class DeletePagesRequest(BaseModel):
    pages: list[int]


class InsertBlankRequest(BaseModel):
    after_index: int
    width: float = 612.0
    height: float = 792.0


class InsertFromRequest(BaseModel):
    source_path: str
    after_index: int


class ReorderRequest(BaseModel):
    new_order: list[int]


class MovePageRequest(BaseModel):
    from_index: int
    to_index: int


class PageListRequest(BaseModel):
    """通用页列表请求(摆正/删除文字层/渲染缩略图/预览/检测文字层)。"""

    pages: list[int]


class AddTextLayerRequest(BaseModel):
    page: int
    ocr_result: dict[str, Any]  # 序列化的 OCRResult(含 text_blocks)
    pdf_settings: dict[str, Any] | None = None
    overwrite: bool = False


class BatchAddTextLayerPage(BaseModel):
    """批量加文字层的单页条目。"""

    page: int
    ocr_result: dict[str, Any]  # 序列化的 OCRResult(含 text_blocks)


class BatchAddTextLayerRequest(BaseModel):
    """批量写 OCR 文字层：一次接收一批页，后端聚合字符解析单一子集字体。

    避免逐页 add_text_layer 每页各解析一份子集字体（放大体积）。
    聚合逻辑复用 save_with_rewrite 已验证的"整文档一次子集"模式。

    save=True 时，写层成功后紧跟一次 incremental save 把本批落盘
    （崩溃只丢最后一批）。返回 extra.saved 标记是否成功落盘。
    """

    pages: list[BatchAddTextLayerPage]
    pdf_settings: dict[str, Any] | None = None
    overwrite: bool = False
    save: bool = False


class RewriteTextLayerRequest(BaseModel):
    page: int
    text_blocks: list[TextBlockMirror]
    preproc_angle: int = 0
    pdf_settings: dict[str, Any] | None = None


class UpdateBlockTextRequest(BaseModel):
    """双击编辑文字块(仅更新内存模型,不落盘)。"""

    page: int
    block_index: int
    new_text: str


class RenderThumbnailRequest(BaseModel):
    page: int
    size: int = 160  # 目标边长(px)


class RenderPreviewRequest(BaseModel):
    page: int
    dpi: int = 150


class OcrRequest(BaseModel):
    pages: list[int]
    options: dict[str, Any] | None = None
    pdf_settings: dict[str, Any] | None = None
    add_text_layer: bool = True  # 是否写回 PDF 文字层


class DetectTextLayersRequest(BaseModel):
    page: int


# ---- 响应体 ------------------------------------------------------------

class OpenResponse(BaseModel):
    session_id: str
    model: PdfDocumentMirror


class EmptyOk(BaseModel):
    ok: bool = True


class SaveResponse(BaseModel):
    path: str
    diff: ModelDiff


class MutateResponse(BaseModel):
    """通用变更操作响应(旋转/删除/插入/重排/加文字层/重写/摆正)。"""

    diff: ModelDiff
    extra: dict[str, Any] | None = None  # 操作特定附加(摆正 summary / 删除残留等)


class RenderBytesResponse(BaseModel):
    """渲染结果(缩略图/预览)的元信息;实际字节走 HTTP body。"""

    width: int
    height: int
    fmt: str = "png"  # png / rgb


class DetectTextLayersResponse(BaseModel):
    text_layers: list[TextLayerInfoMirror]


class HealthResponse(BaseModel):
    status: str = "ok"
    sessions: int = 0
    pid: int = 0


class ErrorResponse(BaseModel):
    detail: str
