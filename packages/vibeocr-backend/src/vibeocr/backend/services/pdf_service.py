"""PDF 操作服务（无状态工具层）

所有方法为 @staticmethod，接收 fitz.Document / PdfDocument 参数，不持有任何实例状态。
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import fitz
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

from vibeocr.backend.models.ocr_result import TextBlock
from vibeocr.backend.models.pdf_document import PdfDocument, PdfPageInfo, TextLayerInfo
from vibeocr.backend.utils.cjk_font_resolver import _CJK_RESOLVER

logger = logging.getLogger(__name__)

# insert_textbox 单行所需的最小矩形高度系数（实测：CJK≈1.58，Helvetica≈1.67，
# 含行距/上下内边距）。fontsize × LINE_LEADING ≤ rect.height 才能在首次
# insert_textbox 调用就放下文本（rc≥0 才真正写入；rc<0 时不写入任何字符，
# 只触发后续字号重试或 insert_text 兜底）。此前的 font_size_ratio=0.8 算出
# fontsize=height×0.8，恒满足不了 height ≥ fontsize×1.6，导致几乎每个块都
# 走重试缩字号（写入字号偏小、选中框与可见文字不匹配）甚至兜底 insert_text
# （窄/高块文字横向大幅溢出到无关区域 → 「严重偏离」）。
_LINE_LEADING = 1.6

# insert_text 单点写入的 ink（实际墨迹像素）几何系数（CJK 子集字体实测）：
#   ink_height / fontsize ≈ 0.955  （墨迹高度，OCR bbox 检测的就是这个）
#   (baseline_y - ink_top) / fontsize ≈ 0.83   （基线在墨迹顶部下方 0.83×fs）
#   (ink_bottom - baseline_y) / fontsize ≈ 0.125（基线下方少许下伸部分）
# 用这些系数可把 ink 区域精确对齐到 OCR bbox：fontsize = bbox_height/INK_RATIO，
# 基线 y = bbox.y0 + ASCENT_RATIO×fontsize，使 ink 顶部 = bbox 顶部、ink 高度 = bbox 高度。
# 相比 insert_textbox（行距预算 1.319×fs 把字号压到 bbox 的 ~73%），insert_text 能让
# 文字层 ink 区域与 OCR bbox 匹配，解决『区域太小』问题。
_INK_RATIO = 0.955
_ASCENT_RATIO = 0.83
_DESCENT_RATIO = 0.125


class SaveResult(NamedTuple):
    rewritten_pages: list[int]
    path: str | None
    # 全量压缩覆盖原文件时，doc 必须关闭重开（Windows 锁），新 doc 通过此字段
    # 返回，由调用方更新 session.doc 引用。incremental/另存为时为 None（doc 不变）。
    new_doc: fitz.Document | None = None


class PdfService:
    # 词级 redact 的最大循环轮数。绝大多数页 1 轮清零；仅嵌套/合并异常的
    # 文本结构需多轮。内部常量，不暴露为用户配置。
    _DELETE_LAYER_MAX_ROUNDS = 5
    _INCREMENTAL_MARKER_SUFFIX = ".vibeocr-incremental"

    # ---- open / save ------------------------------------------------

    @staticmethod
    def open_doc(file_path: str) -> tuple[fitz.Document, PdfDocument]:
        """打开 PDF 并返回 (fitz.Document, PdfDocument)。

        只做 fitz.open + 创建轻量占位页(rotation=0,不逐页读 doc[i])。
        真实 rotation 及文字层信息由 PDF 后端子进程在 /load 路由逐页填充,
        避免打开大 PDF 时遍历每页阻塞。
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        PdfService._recover_interrupted_incremental(file_path)
        doc = fitz.open(file_path)
        if doc.is_encrypted:
            doc.close()
            raise RuntimeError("不支持加密 PDF 文件")

        pdf_document = PdfDocument(file_path=file_path)
        # 创建轻量占位页面(rotation=0),避免逐页读 doc[i] 解析页对象。
        # 详细的页面信息(rotation / 文字层 / is_scanned)由 PDF 后端 /load 路由逐页填充。
        pdf_document.pages = [PdfPageInfo(page_index=i) for i in range(doc.page_count)]
        return doc, pdf_document

    @staticmethod
    def save(
        doc: fitz.Document,
        pdf_document: PdfDocument,
        path: str | None = None,
        pdf_settings: object | None = None,
    ) -> fitz.Document | None:
        """落盘 PDF，返回覆盖保存后的新 doc（全量压缩时 doc 已重开）。

        覆盖保存（path is None）按 compress_on_save 分流：True（默认）= 全量
        压缩（garbage=3 + deflate + object streams）。PyMuPDF 不能全量保存覆盖自身已打开
        源文件，且 Windows 锁定打开的文件，因此全量压缩采用
        save(tmp)→close→replace→reopen：先写同目录临时文件、关闭释放文件锁、
        原子替换原路径，再打开并返回新 doc（调用方需更新其 doc 引用）。
        False = 增量追加快路径（incremental，doc 不变，返回 None）。
        另存为（path 指定）统一 deflate + clean（新文件，doc 不变）。
        """
        from vibeocr.backend.models.pdf_ocr_options import PdfGlobalSettings

        settings = pdf_settings if pdf_settings is not None else PdfGlobalSettings()
        compress = getattr(settings, "compress_on_save", True)
        # clean_on_save 默认 False：加文字层是纯增量场景，clean=True 会解压并重写
        # 扫描件内容流，叠加 MuPDF 无法保留 ObjStm/CrossRefStream 压缩，体积可翻倍。
        clean = getattr(settings, "clean_on_save", False)

        if path is None:
            save_path = pdf_document.file_path
            if save_path is None:
                return None
            if compress:
                new_doc = PdfService._compress_in_place(doc, save_path, clean=clean)
                pdf_document.is_modified = False
                return new_doc
            # 增量快路径：incremental 可原地追加，doc 不变
            backup_path = save_path + ".bak"
            shutil.copy2(save_path, backup_path)
            try:
                doc.save(save_path, incremental=True, encryption=0)
                Path(backup_path).unlink(missing_ok=True)
            except Exception:
                shutil.copy2(backup_path, save_path)
                Path(backup_path).unlink(missing_ok=True)
                raise
            pdf_document.is_modified = False
            return None
        doc.save(path, deflate=True, clean=clean)
        pdf_document.is_modified = False
        return None

    @staticmethod
    def _compress_in_place(
        doc: fitz.Document, save_path: str, clean: bool = True
    ) -> fitz.Document:
        """全量压缩覆盖原文件（Windows 兼容），返回重开后的新 doc。

        流程：先备份原文件 → doc.save(tmp, garbage+deflate[+clean]) 落到临时文件
        → 关闭 doc 释放文件锁 → os.replace(tmp, save_path) 原子替换 → 重开。

        **关键：用 doc.save 而非 doc.tobytes**。旧版 tobytes(garbage=4) 在 doc 累积大量
        insert_text 修改后，会留下不一致的内部 xref/对象状态，随后的 doc.close()
        遍历这些失效引用触发原生内存破坏（0xC0000409 STATUS_STACK_BUFFER_OVERRUN，
        PyMuPDF 1.28.0 实测稳定复现）。doc.save 是官方推荐的"修改后落盘"路径，
        在写文件时正确 finalize 文档内部状态，使后续 close 安全。

        落到临时文件而非直接覆盖原文件：Windows 锁定被 fitz 打开的文件，不能在
        doc 仍打开时覆盖原路径；save 到新路径无锁冲突，close 后再 os.replace。
        默认采用 garbage=3 + object streams；garbage=4 会比较大型 stream 去重，
        对数百页扫描件可能从秒级恶化到数分钟。

        Args:
            clean: 是否深度清理内容流（PyMuPDF clean 参数）。True 重写规范化
            内容流；False 保留原始内容流压缩——对用 ObjStm/CrossRefStream
            高度压缩的扫描件，False 避免解压重写导致的体积膨胀。
        """
        backup_path = save_path + ".bak"
        tmp_path = save_path + ".tmp"
        started_at = time.monotonic()
        source_size = Path(save_path).stat().st_size
        page_count = doc.page_count
        stage_started = time.monotonic()
        shutil.copy2(save_path, backup_path)
        backup_elapsed = time.monotonic() - stage_started
        # 清理可能残留的临时文件（上次崩溃留下的）
        Path(tmp_path).unlink(missing_ok=True)
        try:
            stage_started = time.monotonic()
            doc.save(
                tmp_path,
                garbage=3,
                deflate=True,
                clean=clean,
                encryption=0,
                use_objstms=1,
                compression_effort=1,
            )
            save_elapsed = time.monotonic() - stage_started
            output_size = Path(tmp_path).stat().st_size
            stage_started = time.monotonic()
            doc.close()
            # close 释放了原文件锁，现在可以原子替换
            Path(tmp_path).replace(save_path)
            replace_elapsed = time.monotonic() - stage_started
            stage_started = time.monotonic()
            new_doc = fitz.open(save_path)
            reopen_elapsed = time.monotonic() - stage_started
            Path(backup_path).unlink(missing_ok=True)
            logger.info(
                "PDF 全量压缩完成: pages=%d clean=%s size=%d->%d "
                "backup=%.2fs save=%.2fs replace=%.2fs reopen=%.2fs total=%.2fs",
                page_count,
                clean,
                source_size,
                output_size,
                backup_elapsed,
                save_elapsed,
                replace_elapsed,
                reopen_elapsed,
                time.monotonic() - started_at,
            )
            return new_doc
        except Exception:
            logger.exception(
                "PDF 全量压缩失败: pages=%d clean=%s source_size=%d elapsed=%.2fs",
                page_count,
                clean,
                source_size,
                time.monotonic() - started_at,
            )
            # 失败回滚：doc 可能已关闭，原文件从备份恢复，清理临时文件
            try:
                doc.close()
            except Exception:
                pass
            try:
                Path(backup_path).replace(save_path)
            except OSError:
                shutil.copy2(backup_path, save_path)
            Path(backup_path).unlink(missing_ok=True)
            Path(tmp_path).unlink(missing_ok=True)
            raise

    @staticmethod
    def save_with_rewrite(
        doc: fitz.Document,
        pdf_document: PdfDocument,
        path: str | None = None,
        pdf_settings: object | None = None,
        *,
        rewrite_text_layers: bool = True,
    ) -> SaveResult:
        """对所有有 OCR 块的页重写文字层后落盘（保存/另存为共用）。

        rewrite 阶段用词级 redact（delete_text_layers 循环验证），并对全文档
        共享单一子集字体（整文档聚合字符一次解析，避免每页一份字体）。
        落盘覆盖原文件时：有结构改动或 compress_on_save=True → 全量压缩
        （garbage+deflate+clean，临时文件原子替换）；否则 incremental（快）。
        另存为永远 deflate+clean。

        Args:
            doc: fitz.Document 实例。
            pdf_document: PdfDocument 状态对象。
            path: None=覆盖原文件；str=另存为到该路径。
            pdf_settings: PdfGlobalSettings（rewrite 用）。
            rewrite_text_layers: False 表示文字层已由批量 OCR 正确写入，仅做
                最终落盘/压缩；普通保存必须保持默认 True 以应用编辑后的块。

        Returns:
            SaveResult(rewritten_pages, path)。
        """
        from vibeocr.backend.models.pdf_ocr_options import PdfGlobalSettings

        settings = pdf_settings if pdf_settings is not None else PdfGlobalSettings()
        compress = getattr(settings, "compress_on_save", True)

        # clean_on_save 决定全量压缩时是否深度清理内容流。默认 False：加文字层
        # 是纯增量场景，clean=True 会解压并重写扫描件内容流，叠加 MuPDF 无法保留
        # ObjStm/CrossRefStream 压缩，可使 1.6MB 扫描件膨胀到 3MB+。
        clean = getattr(settings, "clean_on_save", False)

        # 整文档一次聚合子集字体：把所有有 OCR 块的页字符汇成一个子集，
        # 全文档共享单一字体对象，避免每页一份独立子集放大体积。
        # （探测失败为 None → rewrite_text_layer 内部回退 china-s。）
        target_pages = (
            [info for info in pdf_document.pages if info.ocr_text_blocks]
            if rewrite_text_layers
            else []
        )
        shared_font_path: str | None = None
        if target_pages:
            all_chars = "".join(
                b.text for info in target_pages for b in info.ocr_text_blocks if b.text
            )
            shared_font_path = _CJK_RESOLVER.resolve(all_chars)

        rewritten: list[int] = []
        for info in target_pages:
            PdfService.rewrite_text_layer(
                doc,
                pdf_document,
                info.page_index,
                info.ocr_text_blocks,
                info.ocr_preproc_angle,
                pdf_settings=settings,
                font_path=shared_font_path,
            )
            rewritten.append(info.page_index)

        new_doc: fitz.Document | None = None
        if path is None:
            save_path = pdf_document.file_path
            if save_path is None:
                pdf_document.is_modified = False
                pdf_document.has_structural_change = False
                return SaveResult(rewritten, None)
            # 需要全量压缩（结构改动或 compress_on_save）→ _compress_in_place
            # （Windows 锁要求 close+write+reopen，返回新 doc）。
            need_full = pdf_document.has_structural_change or compress
            if need_full:
                new_doc = PdfService._compress_in_place(doc, save_path, clean=clean)
            else:
                if not PdfService.save_incremental(doc, save_path):
                    raise RuntimeError("incremental save failed and was rolled back")
        else:
            doc.save(path, deflate=True, clean=clean)

        pdf_document.is_modified = False
        pdf_document.has_structural_change = False
        return SaveResult(rewritten, path, new_doc)

    @staticmethod
    def save_incremental(doc: fitz.Document, save_path: str) -> bool:
        """增量保存（纯加文字层场景）。doc 不 close/重开，内存对象始终可用。

        PDF incremental save 只在文件尾追加。保存前把原长度写入一个小型恢复
        marker；异常时按长度截断，进程若在写入中退出则下次 open_doc 也会先
        截断恢复。避免每个 OCR 批次都复制整份、持续增长的 PDF。
        供 OCR 逐批落盘使用：每批写层后调用，崩溃只丢最后一批。

        Args:
            doc: fitz.Document 实例（已写好本批文字层）。无论成功失败都不 close。

        Returns:
            True 已落盘；False 失败已回滚文件（doc 内存文字层保留可用，调用方
            不应标记该批已落盘/不写 sidecar）。
        """
        path = Path(save_path)
        marker = Path(save_path + PdfService._INCREMENTAL_MARKER_SUFFIX)
        try:
            if not doc.can_save_incrementally():
                # 记录诊断字段，便于定位为何增量不可用：
                # - is_encrypted/needs_pass：加密 PDF（open_doc 已拦截，但防御）
                # - is_form_pdf：表单 PDF（AcroForm/XFA）MuPDF 标记结构变更后不可增量
                # - is_dirty：内存 doc 有未保存修改（正常，加文字层即如此）
                # 实测稳定复现 0xC0000409 的场景：增量恒 False → 跨批字体累积 →
                # 末尾 _compress_in_place 全量重写崩。见 add_text_layer_batch 回退。
                logger.error(
                    "save_incremental: 当前文档不支持增量保存 "
                    "(is_encrypted=%s, needs_pass=%s, is_form_pdf=%s, is_dirty=%s)",
                    doc.is_encrypted,
                    doc.needs_pass,
                    getattr(doc, "is_form_pdf", None),
                    getattr(doc, "is_dirty", None),
                )
                return False
            original_size = path.stat().st_size
            with marker.open("w", encoding="ascii") as stream:
                stream.write(str(original_size))
                stream.flush()
                os.fsync(stream.fileno())
        except Exception as e:
            logger.error("save_incremental: 创建恢复标记失败，跳过本批落盘: %s", e)
            return False
        try:
            doc.save(save_path, incremental=True, encryption=0)
            marker.unlink()
            return True
        except Exception as e:
            # incremental save 失败：文件可能只追加了一部分，截断到写前长度。
            # doc 内存对象未受影响（fitz save 失败不改内存 doc），保持可用，
            # 内存文字层保留。调用方据此返回 saved=False，不写 sidecar。
            logger.error("save_incremental: 增量保存失败，按原长度回滚: %s", e)
            try:
                PdfService._truncate_incremental(path, marker, original_size)
            except Exception:
                logger.error("save_incremental: 长度回滚失败", exc_info=True)
            return False

    @staticmethod
    def _truncate_incremental(path: Path, marker: Path, original_size: int) -> None:
        current_size = path.stat().st_size
        if current_size < original_size:
            raise RuntimeError(
                f"incremental target shrank unexpectedly: {current_size} < {original_size}"
            )
        if current_size != original_size:
            with path.open("r+b") as stream:
                stream.truncate(original_size)
                stream.flush()
                os.fsync(stream.fileno())
        marker.unlink(missing_ok=True)

    @staticmethod
    def _recover_interrupted_incremental(save_path: str) -> bool:
        """Recover an append interrupted after its marker reached disk."""
        path = Path(save_path)
        marker = Path(save_path + PdfService._INCREMENTAL_MARKER_SUFFIX)
        if not marker.exists():
            return False
        try:
            original_size = int(marker.read_text(encoding="ascii").strip())
            if original_size < 0:
                raise ValueError("negative original size")
            PdfService._truncate_incremental(path, marker, original_size)
            logger.warning("已恢复中断的 PDF 增量保存: %s", save_path)
            return True
        except Exception:
            logger.error("恢复中断的 PDF 增量保存失败: %s", save_path, exc_info=True)
            raise

    @staticmethod
    def render_page_as_array(
        doc: fitz.Document, page_index: int, dpi: int = 300
    ) -> np.ndarray:
        page = doc[page_index]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=mat)
        return (
            np.frombuffer(pixmap.samples, dtype=np.uint8)
            .reshape(pixmap.height, pixmap.width, 3)
            .copy()
        )

    # ---- text layer detection ---------------------------------------

    @staticmethod
    def detect_text_layers(doc: fitz.Document, page_index: int) -> list[TextLayerInfo]:
        page = doc[page_index]
        # 快速预检：get_text("text") ~4ms vs get_text("dict") ~173ms（扫描件）。
        # 扫描件绝大多数页无文字层，先 4ms 判空，无文字直接返回避免 173ms dict。
        # 有文字才调 dict 取 block bbox 详情。
        if not page.get_text("text").strip():
            return []
        page_dict: dict[str, Any] = page.get_text("dict")  # type: ignore[assignment]
        blocks: list[dict[str, Any]] = page_dict["blocks"]

        # 用 **line 级** bbox（而非 block 级）：PyMuPDF 的 block 会把同一文本列里
        # 纵向相邻但实际相距很远的行合并成一个超大 bbox（实测可跨 300pt+），
        # 导致预览高亮把不相邻的文字框在一起。line 级每个 bbox 是一条连续文本，
        # 与 OCR 行级粒度一致，高亮更精确、不误合并。
        layers: list[TextLayerInfo] = []
        layer_index = 0
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block.get("lines", []):
                text_parts = []
                for span in line.get("spans", []):
                    text_parts.append(span.get("text", ""))
                line_text = "".join(text_parts).strip()
                if not line_text:
                    continue
                bbox = line["bbox"]
                layers.append(
                    TextLayerInfo(
                        index=layer_index,
                        text_preview=line_text[:30],
                        char_count=len(line_text),
                        bbox=(
                            float(bbox[0]),
                            float(bbox[1]),
                            float(bbox[2]),
                            float(bbox[3]),
                        ),
                        color_id=layer_index % 8,
                    )
                )
                layer_index += 1
        return layers

    @staticmethod
    def is_page_scanned(doc: fitz.Document, page_index: int) -> bool:
        page = doc[page_index]
        images = page.get_images(full=True)
        if not images:
            return False
        page_rect = page.rect
        for img_info in images:
            xref = img_info[0]
            for rect in page.get_image_rects(xref):
                coverage = (rect.width * rect.height) / (
                    page_rect.width * page_rect.height
                )
                if coverage > 0.5:
                    return True
        return False

    # ---- geometry helpers（避免调用方直接访问 fitz 对象）-------------

    @staticmethod
    def page_rect(
        doc: fitz.Document, page_index: int
    ) -> tuple[float, float, float, float]:
        """返回页面 rect（x0, y0, x1, y1），单位 PDF point。

        供主进程预览 highlight 几何计算用——下沉子进程后主进程不再持 doc，
        此方法在子进程内调用，结果序列化回主进程。
        """
        r = doc[page_index].rect
        return (r.x0, r.y0, r.x1, r.y1)

    @staticmethod
    def page_rotation(doc: fitz.Document, page_index: int) -> int:
        """返回页面旋转角（0/90/180/270）。"""
        return int(doc[page_index].rotation)

    @staticmethod
    def page_has_text(doc: fitz.Document, page_index: int) -> bool:
        """页面是否含可见文字（get_text("text") 快速判断，~3ms）。

        供删除文字层前的预检查用——无文字直接跳过 redact 循环。
        """
        return bool(doc[page_index].get_text("text").strip())

    # ---- page infos -------------------------------------------------

    @staticmethod
    def build_page_infos(doc: fitz.Document, pdf_document: PdfDocument) -> None:
        pages: list[PdfPageInfo] = []
        for i in range(doc.page_count):
            text_layers = PdfService.detect_text_layers(doc, i)
            page = doc[i]
            pages.append(
                PdfPageInfo(
                    page_index=i,
                    rotation=page.rotation,
                    has_text_layer=len(text_layers) > 0,
                    text_layers=text_layers,
                    is_scanned=len(text_layers) == 0
                    and PdfService.is_page_scanned(doc, i),
                    rect=PdfService.page_rect(doc, i),
                )
            )
        pdf_document.pages = pages

    @staticmethod
    def update_page_info(
        doc: fitz.Document, pdf_document: PdfDocument, page_index: int
    ) -> None:
        if page_index >= len(pdf_document.pages):
            return
        text_layers = PdfService.detect_text_layers(doc, page_index)
        page = doc[page_index]
        info = pdf_document.pages[page_index]
        info.rotation = page.rotation
        info.has_text_layer = len(text_layers) > 0
        info.text_layers = text_layers
        info.is_scanned = not text_layers and PdfService.is_page_scanned(
            doc, page_index
        )
        info.rect = PdfService.page_rect(doc, page_index)
        info.thumbnail = None

    # ---- page mutations ---------------------------------------------

    @staticmethod
    def rotate_pages(
        doc: fitz.Document,
        pdf_document: PdfDocument,
        page_indices: list[int],
        angle: int,
    ) -> None:
        for idx in page_indices:
            if 0 <= idx < doc.page_count:
                page = doc[idx]
                page.set_rotation((page.rotation + angle) % 360)
                pdf_document.pages[idx].rotation = page.rotation
        pdf_document.is_modified = True
        PdfService.invalidate_thumbnails(pdf_document, page_indices)

    @staticmethod
    def delete_pages(
        doc: fitz.Document,
        pdf_document: PdfDocument,
        page_indices: list[int],
    ) -> None:
        remaining = [
            p for i, p in enumerate(pdf_document.pages) if i not in page_indices
        ]
        for idx in sorted(page_indices, reverse=True):
            if 0 <= idx < doc.page_count:
                doc.delete_page(idx)
        pdf_document.pages = remaining
        pdf_document.is_modified = True
        pdf_document.has_structural_change = True

    @staticmethod
    def insert_blank_page(
        doc: fitz.Document,
        pdf_document: PdfDocument,
        after_index: int,
        width: float = 612,
        height: float = 792,
    ) -> None:
        insert_at = after_index + 1
        doc.new_page(pno=insert_at, width=width, height=height)
        pdf_document.is_modified = True
        pdf_document.has_structural_change = True
        PdfService.build_page_infos(doc, pdf_document)

    @staticmethod
    def insert_pages_from(
        doc: fitz.Document,
        pdf_document: PdfDocument,
        source_path: str,
        after_index: int,
    ) -> None:
        src = fitz.open(source_path)
        insert_at = after_index + 1
        doc.insert_pdf(src, start_at=insert_at)
        src.close()
        pdf_document.is_modified = True
        pdf_document.has_structural_change = True
        PdfService.build_page_infos(doc, pdf_document)

    @staticmethod
    def move_page(
        doc: fitz.Document,
        pdf_document: PdfDocument,
        from_index: int,
        to_index: int,
    ) -> None:
        if from_index == to_index:
            return
        page_info = pdf_document.pages[from_index]
        doc.move_page(from_index, to_index)
        pages = list(pdf_document.pages)
        pages.pop(from_index)
        pages.insert(to_index, page_info)
        pdf_document.pages = pages
        pdf_document.is_modified = True
        pdf_document.has_structural_change = True

    @staticmethod
    def reorder_pages(
        doc: fitz.Document,
        pdf_document: PdfDocument,
        new_order: list[int],
    ) -> None:
        """按 new_order 指定的顺序重排页面。

        new_order[i] = j 表示新位置 i 应放原索引 j 的页面。
        """
        n = len(new_order)
        if n != doc.page_count or n != len(pdf_document.pages):
            return
        if new_order == list(range(n)):
            return

        doc.select(new_order)
        pdf_document.pages = [pdf_document.pages[i] for i in new_order]
        pdf_document.is_modified = True
        pdf_document.has_structural_change = True

    # ---- text layer mutations ---------------------------------------

    @staticmethod
    def add_text_layer(
        doc: fitz.Document,
        pdf_document: PdfDocument,
        page_index: int,
        ocr_result: object,
        pdf_settings: object | None = None,
        overwrite: bool = False,
    ) -> tuple[int, int]:
        """将 OCR 结果作为隐形文字层写入 PDF 页面。

        使用内置 china-s CJK CID 字体，确保中文等字符可被写入并被阅读器提取。

        写入完成后，OCR 原始块（归一化 bbox）缓存到 PdfPageInfo.ocr_text_blocks，
        作为预览/编辑/重写的唯一信源。不再用 detect_text_layers 重读（PyMuPDF 会
        把细粒度块合并成粗块，导致预览显示合并后的错误块）。

        Args:
            doc: fitz.Document 实例。
            pdf_document: PdfDocument 状态对象。
            page_index: 页码索引。
            ocr_result: OCRResult 实例。
            pdf_settings: PdfGlobalSettings 实例（None 则使用默认值）。
            overwrite: 若为 True 且该页已有文字层，先删除再写入；若为 False 且
                该页已有文字层，直接跳过返回 (0, 1)，绝不叠加。

        Returns:
            (written, skipped) 成功写入与被跳过的文本块数量。
        """
        from vibeocr.backend.models.pdf_ocr_options import PdfGlobalSettings

        settings = pdf_settings if pdf_settings is not None else PdfGlobalSettings()
        preproc_angle = getattr(ocr_result, "preproc_angle", 0)
        text_blocks = list(getattr(ocr_result, "text_blocks", []))

        # 防重复守卫：已有文字层时按 overwrite 决定跳过或先删后写
        page_info = pdf_document.pages[page_index]
        if page_info.has_text_layer:
            if not overwrite:
                logger.info("page %d 已有文字层，跳过（overwrite=False）", page_index)
                return 0, 1
            logger.info("page %d 已有文字层，overwrite=True，先删除再写入", page_index)
            PdfService.delete_text_layers(doc, pdf_document, page_index)

        written, skipped = PdfService._write_blocks_to_page(
            doc, page_index, text_blocks, preproc_angle, settings
        )

        # 缓存 OCR 原始块（预览/编辑/重写的唯一信源），替代旧的 detect_text_layers 重读
        pdf_document.is_modified = True
        info = pdf_document.pages[page_index]
        info.ocr_text_blocks = text_blocks
        info.ocr_preproc_angle = preproc_angle
        info.has_text_layer = written > 0
        info.thumbnail = None
        return written, skipped

    @staticmethod
    def add_text_layer_batch(
        doc: fitz.Document,
        pdf_document: PdfDocument,
        pages_data: list[dict],
        pdf_settings: object | None = None,
        overwrite: bool = False,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[int, tuple[int, int]]:
        """批量写 OCR 文字层，一批页共享单一聚合子集字体。

        复用 save_with_rewrite 已验证的"整文档一次子集"模式：先把本批所有页
        的文本块字符聚合，一次 _CJK_RESOLVER.resolve 得到共享子集字体路径，
        再循环调 _write_blocks_to_page(font_path=shared)。避免逐页 add_text_layer
        每页各解析一份独立子集字体（每页一份字体对象会放大 PDF 体积）。

        Args:
            doc: fitz.Document 实例。
            pdf_document: PdfDocument 状态对象。
            pages_data: [{page, ocr_result}] 列表，ocr_result 为序列化 OCRResult dict。
            pdf_settings: PdfGlobalSettings 实例（None 则使用默认值）。
            overwrite: 同 add_text_layer，控制已有文字层页的跳过/重写。
            cancel_check: 可选取消回调；在逐页写层循环每页开头调用，返回 True 时
                立即停止写后续页（已写页保留）。供后端协作式取消使用。

        Returns:
            {page_index: (written, skipped)} 每页写入/跳过块数。
            已有文字层且 overwrite=False 的页不在返回值中（调用方据此判 skipped）。
        """
        from vibeocr.backend.models.pdf_ocr_options import PdfGlobalSettings

        settings = pdf_settings if pdf_settings is not None else PdfGlobalSettings()

        # 预处理：反序列化、应用防重复守卫，收集实际要写的页
        to_write: list[
            tuple[int, list, int]
        ] = []  # (page_index, text_blocks, preproc_angle)
        skipped_pages: list[int] = []  # 因已有文字层而跳过的页
        for item in pages_data:
            page_index = item["page"]
            ocr_result_data = item["ocr_result"]
            text_blocks = [
                TextBlock(
                    text=b["text"],
                    score=b["score"],
                    bbox=tuple(b["bbox"]) if b.get("bbox") else None,
                    polygon=tuple(b["polygon"]) if b.get("polygon") else None,
                    page_idx=b.get("page_idx"),
                    is_manually_edited=b.get("is_manually_edited", False),
                    label=b.get("label", "text"),
                    order=b.get("order", -1),
                )
                for b in ocr_result_data.get("text_blocks", [])
            ]
            preproc_angle = int(ocr_result_data.get("preproc_angle", 0) or 0)

            page_info = pdf_document.pages[page_index]
            if page_info.has_text_layer:
                if not overwrite:
                    logger.info(
                        "batch: page %d 已有文字层，跳过（overwrite=False）",
                        page_index,
                    )
                    skipped_pages.append(page_index)
                    continue
                logger.info(
                    "batch: page %d 已有文字层，overwrite=True，先删除再写入",
                    page_index,
                )
                PdfService.delete_text_layers(doc, pdf_document, page_index)
            to_write.append((page_index, text_blocks, preproc_angle))

        results: dict[int, tuple[int, int]] = dict.fromkeys(skipped_pages, (0, 1))

        if not to_write:
            return results

        # 聚合本批所有页的字符 → 一次解析共享子集字体
        all_chars = "".join(
            b.text for _, blocks, _ in to_write for b in blocks if b.text
        )
        shared_font_path = _CJK_RESOLVER.resolve(all_chars) if all_chars else None

        for page_index, text_blocks, preproc_angle in to_write:
            # 协作式取消：每页开头检查，已写页保留、停止写后续页
            # （对齐 delete_text_layers 路由的 cancel_event 逐页检查语义）。
            if cancel_check is not None and cancel_check():
                logger.info("batch: 取消已请求，停止写后续页（已写页保留）")
                break
            written, skipped = PdfService._write_blocks_to_page(
                doc,
                page_index,
                text_blocks,
                preproc_angle,
                settings,
                font_path=shared_font_path,
            )
            # 缓存 OCR 原始块（与单页 add_text_layer 一致）
            pdf_document.is_modified = True
            info = pdf_document.pages[page_index]
            info.ocr_text_blocks = text_blocks
            info.ocr_preproc_angle = preproc_angle
            info.has_text_layer = written > 0
            info.thumbnail = None
            results[page_index] = (written, skipped)

        return results

    @staticmethod
    def _write_blocks_to_page(
        doc: fitz.Document,
        page_index: int,
        text_blocks: list,
        preproc_angle: int,
        settings: object,
        font_path: str | None = None,
    ) -> tuple[int, int]:
        """将文本块逐个写入指定页面（纯写入，不修改 PdfPageInfo 元信息）。

        add_text_layer（首次写入）与 rewrite_text_layer（编辑后重写）共用此方法，
        保证两条路径的字号策略、字体、兜底逻辑完全一致。

        Args:
            doc: fitz.Document 实例。
            page_index: 页码索引。
            text_blocks: TextBlock 列表（归一化 [0,1000] bbox）。
            preproc_angle: OCR 预处理旋转角度（用于坐标逆旋转）。
            settings: PdfGlobalSettings 实例。
            font_path: 调用方预先解析的共享子集字体路径。None 时按本页字符内部
                解析子集（逐页 OCR 用）；非 None 时直接复用（保存时整文档共享
                单一子集字体，避免每页一份字体对象放大体积）。

        Returns:
            (written, skipped) 成功写入与被跳过的文本块数量。
        """
        page = doc[page_index]
        page_rect = page.rect

        # 页面 /Rotate 处理：OCR 渲染图（get_pixmap 自动应用 /Rotate）与归一化
        # bbox 都在『显示空间』，但 insert_textbox 写入的是『mediabox（未旋转）
        # 空间』。当 page.rotation != 0 时，必须把『显示空间』的 rect 经
        # derotation_matrix 映射到 mediabox 空间，否则会出现『上面的字写到了
        # 右面』（90° 旋转时宽高互换 + 旋转未补偿）。
        # _denormalize_and_unrotate_bbox 用 page_rect（显示尺寸）归一化，产出
        # 仍在『显示空间』；下面 derotate_to_mediabox 把它转到 mediabox 空间。
        page_rotation = int(page.rotation or 0) % 360
        if page_rotation != 0:
            dm = page.derotation_matrix  # 显示空间 → mediabox 空间

            def _derotate_to_mediabox(rect: fitz.Rect) -> fitz.Rect:
                a, b, c, d, e, f = (
                    dm.a,
                    dm.b,
                    dm.c,
                    dm.d,
                    dm.e,
                    dm.f,
                )

                def _tr(x, y):
                    return a * x + c * y + e, b * x + d * y + f

                pts = [
                    _tr(rect.x0, rect.y0),
                    _tr(rect.x1, rect.y0),
                    _tr(rect.x0, rect.y1),
                    _tr(rect.x1, rect.y1),
                ]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                return fitz.Rect(min(xs), min(ys), max(xs), max(ys))
        else:

            def _derotate_to_mediabox(rect: fitz.Rect) -> fitz.Rect:
                return rect

        # PyMuPDF 的插入坐标与 get_text 坐标都以未旋转的可见页面（CropBox）
        # 左上角为原点，而不是以 MediaBox 原点为基准。page.rect 和 OCR 渲染图
        # 同样使用归零后的 CropBox，因此这里只需消除 /Rotate；若再加
        # page.cropbox.x0/y0，会把文字层整体重复平移一个 CropBox 原点距离。
        def _to_page_space(rect: fitz.Rect) -> fitz.Rect:
            """显示空间（旋转后的 CropBox）→ PyMuPDF 未旋转页面空间。"""
            return _derotate_to_mediabox(rect)

        # 字形方向（rotate 参数）：页面 /Rotate=90/270 时，derotate 后的 mediabox
        # 矩形宽高互换（宽框→瘦高框），若仍以默认 0°（横向）写入，insert_textbox
        # 会把字排进瘦高框 → 字竖排堆叠 → 外部阅读器渲染 /Rotate 时看起来旋转 90°
        # （程序内预览对，因预览直接在显示空间映射 bbox，不经此写入路径）。
        # 实测（见 tests/services/test_pdf_service.py::test_*_page_rotation_*）：
        # page.rotation ∈ {90,270} 时给 insert_textbox/insert_text 传 rotate=90，
        # 字形即按显示方向正确排布（渲染后长宽比与 rotation=0 基准一致）。
        # 180° 字形仍正向（宽高不互换），rotate=0 即可。
        text_rotate = 90 if page_rotation in (90, 270) else 0

        # 收集本页所有字符，解析子集字体（探测失败则 None，回退 china-s）。
        # 子集字体嵌入后 PyMuPDF 自动生成 ToUnicode CMap，使文字层在所有
        # 主流阅读器可搜索/复制（china-s 依赖阅读器自带 Adobe GB1 CMap，脆弱）。
        #
        # fontname 必须随子集字体变化：PyMuPDF 按名字缓存字体资源，同一页用
        # 相同名字插入不同 fontfile 时会复用第一个（缺字写入 \x00）。用路径
        # 的 md5 前 4 字节派生名字，保证同页两次写入（add→rewrite）不同字符集
        # 的子集不冲突。子集路径是 tempfile 随机名，fontname 随之每进程不同，
        # 但本进程内同子集（同路径）名字稳定即可。
        #
        # font_path 参数非 None 时（保存路径整文档共享子集），直接复用调用方
        # 传入的子集，跳过本页解析——避免每页一份独立子集放大体积。
        import hashlib

        if font_path is None:
            all_chars = "".join(b.text for b in text_blocks if b.text)
            font_path = _CJK_RESOLVER.resolve(all_chars)
        if font_path is not None:
            fontname = "F" + hashlib.md5(font_path.encode()).hexdigest()[:4]
        else:
            fontname = "china-s"

        # 真实字形宽度测量：用子集字体的实际 advance width 计算文本自然宽度，
        # 替代『CJK=1.0/其余=0.5×fontsize』硬编码启发式。后者把数字（实测≈0.586×fs）
        # 和拉丁字母（0.37–0.79×fs）一律按 0.5×fs 估算，导致 natural_w 偏小、
        # scale_x 偏大，数字/拉丁块的 ink 被 morph 过度横向拉伸——位置错位、bbox 偏大。
        _measure_font = fitz.Font(fontfile=font_path) if font_path else None

        def _natural_width(text: str, fs: float) -> float:
            """返回 text 在 fontsize=fs 下的自然渲染宽度（advance width 之和）。

            有子集字体时用真实字形度量；无（china-s 回退）时退回启发式估算。
            """
            if _measure_font is not None:
                return _measure_font.text_length(text, fontsize=fs)  # type: ignore[arg-type]
            units = 0.0
            for ch in text:
                code = ord(ch)
                if (
                    0x2E80 <= code <= 0x9FFF
                    or 0xF900 <= code <= 0xFAFF
                    or 0xFF00 <= code <= 0xFF60
                    or 0x3000 <= code <= 0x303F
                ):
                    units += 1.0
                else:
                    units += 0.5
            return units * fs

        written = 0
        skipped = 0
        for block in text_blocks:
            if block.text is None or not block.text.strip():
                continue
            bbox = block.bbox
            if bbox is None:
                logger.warning(
                    "page %d block skipped (bbox is None): text=%r",
                    page_index,
                    block.text[:30],
                )
                skipped += 1
                continue

            # 逆旋转(OCR 预处理) + 归一化到『显示空间』坐标，
            # 再补偿页面 /Rotate 转到 mediabox（写入）空间。
            disp_rect = PdfService._denormalize_and_unrotate_bbox(
                bbox, preproc_angle, page_rect
            )
            rect = _to_page_space(disp_rect)
            # 仅当矩形退化（宽或高 ≤ 0）才整体跳过；
            # 矮行/窄框不丢弃：字号由 min_font_size 兜底，再交给下方
            # insert_textbox 重试 + insert_text 兜底，保证文字进入文字层。
            if disp_rect.is_empty or disp_rect.width <= 0 or disp_rect.height <= 0:
                logger.warning(
                    "page %d block skipped (rect empty): rect=%s text=%r",
                    page_index,
                    disp_rect,
                    block.text[:30],
                )
                skipped += 1
                continue

            # 字号与写入策略：
            # OCR bbox 检测的是『墨迹像素高度』(ink height ≈ fontsize × _INK_RATIO)。
            # 要让文字层 ink 区域匹配 bbox 高度，需 fontsize = bbox_height / _INK_RATIO。
            #
            # 水平页（page_rotation ∈ {0,180}，最常见扫描件）：用 insert_text 单点写入，
            # fontsize = height / _INK_RATIO，基线 y = rect.y0 + _ASCENT_RATIO×fontsize，
            # 使 ink 顶部对齐 bbox 顶部、ink 高度 ≈ bbox 高度。insert_textbox 因行距开销
            # (1.319×fs) 会把字号压到 bbox 的 ~73%（『区域太小』），故不作为主路径。
            #
            # 旋转页（90/270）：mediabox 矩形宽高互换、几何复杂，沿用 insert_textbox
            # (带 text_rotate=90) 由矩形约束排版；字号仍按 ink 比例放大但受行距上限夹紧。
            # 窄/高块（width < height，竖排文字误检）：同样走 insert_textbox 自动换行。
            text = block.text
            render_mode = 0 if settings.text_layer_visible else 3
            # insert_text 主路径覆盖 page_rotation ∈ {0, 90}（扫描件最常见的两种：
            # 竖向页与横向页）。180/270 几何（上下/左右翻转）基线放置复杂，仍用
            # insert_textbox 矩形约束排版。竖排文本行（多字符）走 insert_textbox
            # 自动换行。
            #
            # 方向判据优先级（用真实阅读方向替代 bbox 长宽比启发式）：
            #   1) 多边形顶点排序（PaddleOCR DB 检测 4 点，长边方向编码阅读方向）；
            #   2) 单字符 → 永远横排（排版公理：竖排至少 2 字；单字符字形天生高瘦，
            #      多边形长边落在垂直方向会误判成竖排）；
            #   3) 无多边形（旧 OCR 结果/其它管道）→ 回退 disp_rect 长宽比，绝不回归。
            polygon = getattr(block, "polygon", None)
            poly_pts = (
                PdfService._denormalize_and_unrotate_polygon(
                    polygon, preproc_angle, page_rect
                )
                if polygon
                else None
            )
            orient = PdfService._poly_orientation(poly_pts, text)
            if orient == "unknown":
                # 无多边形兜底：保留长宽比启发式（多字符竖排误检检测）。
                is_horizontal = (
                    len(text.strip()) <= 1 or disp_rect.width >= disp_rect.height
                )
            else:
                is_horizontal = orient == "horizontal"
            use_insert_text = page_rotation in (0, 90) and is_horizontal

            if use_insert_text:
                # 在『显示空间』算基线（ink 顶部 = disp_rect.y0）：
                #   baseline_disp = (disp_rect.x0, disp_rect.y0 + ASCENT×fs)
                # 再经 derotation_matrix 转到 PyMuPDF 未旋转页面空间。
                fontsize = max(disp_rect.height / _INK_RATIO, settings.min_font_size)
                baseline_disp_x = disp_rect.x0
                baseline_disp_y = disp_rect.y0 + _ASCENT_RATIO * fontsize
                # 经 _to_page_space 同款变换（derotate）到未旋转页面空间。
                dpt = _to_page_space(
                    fitz.Rect(
                        baseline_disp_x,
                        baseline_disp_y,
                        baseline_disp_x,
                        baseline_disp_y,
                    )
                )
                baseline = fitz.Point(dpt.x0, dpt.y0)
                text_rotate = 90 if page_rotation in (90, 270) else 0

                # 宽度匹配：用子集字体真实 advance width 计算自然宽度，再算 morph
                # 水平缩放把 ink 拉伸到 bbox 宽度（隐形层 render_mode=3 下字形拉伸
                # 不可见，选中框覆盖 bbox 才是目标）。此前用『CJK=1.0/其余=0.5×fs』
                # 硬编码，数字真实宽度≈0.586×fs 被低估，scale_x 偏大导致 ink 过度拉伸、
                # 位置错位与 bbox 偏大。
                # 缩放系数夹在 [0.5, 3.0]：避免过度拉伸（稀疏 OCR 框）或过度压缩
                # （文本溢出 bbox）。scale_x=1.0 时不传 morph（与原行为一致）。
                natural_w = max(_natural_width(text, fontsize), fontsize * 0.5)
                scale_x = disp_rect.width / natural_w
                scale_x = max(0.5, min(3.0, scale_x))

                try:
                    morph = None
                    if abs(scale_x - 1.0) > 0.05:
                        # rot=0: mediabox x 缩放；rot=90: mediabox y 缩放
                        # （显示水平方向 = mediabox 竖直方向，因 derotation 旋 90°）
                        if page_rotation == 90:
                            morph = (baseline, fitz.Matrix(1.0, scale_x))
                        else:
                            morph = (baseline, fitz.Matrix(scale_x, 1.0))
                    page.insert_text(
                        baseline,
                        text,
                        fontsize=fontsize,
                        fontname=fontname,
                        fontfile=font_path,
                        color=(0, 0, 0),
                        render_mode=render_mode,
                        rotate=text_rotate,
                        morph=morph,
                    )
                    written += 1
                    continue
                except Exception as e:
                    logger.warning(
                        "page %d block insert_text 失败，回退 insert_textbox: "
                        "rect=%s fs=%.1f err=%s",
                        page_index,
                        rect,
                        fontsize,
                        e,
                    )
                    # 落到下方 insert_textbox 路径

            # insert_textbox 路径（旋转页 / 窄高块 / insert_text 失败回退）：
            # 字号受行距预算夹紧（rect.height ≥ fontsize × _LINE_LEADING），
            # 同时受宽度约束。缩字号重试确保装入。
            height_based = rect.height / _LINE_LEADING
            # 用真实字形宽度算 width_based：避免把数字（0.586×fs）按 0.5×fs
            # 估算导致的字号偏大/文字溢出 bbox。
            natural_w = _natural_width(text, 1.0)  # 单位 fontsize 宽度
            width_based = rect.width / max(natural_w, 0.5)
            fontsize = max(min(height_based, width_based), settings.min_font_size)

            inserted = False
            last_fontsize = fontsize
            for _retry_idx in range(settings.font_size_retry_count):
                rc = page.insert_textbox(
                    rect,
                    text,
                    fontsize=fontsize,
                    fontname=fontname,
                    fontfile=font_path,
                    color=(0, 0, 0),
                    render_mode=render_mode,
                    rotate=text_rotate,
                )
                if rc >= 0:
                    inserted = True
                    break
                last_fontsize = fontsize
                fontsize *= settings.font_size_shrink_factor
                if fontsize < 1:
                    break

            if inserted:
                written += 1
            else:
                # 兜底：insert_textbox 装不下时降级 insert_text 单点写入。
                # 用 min_font_size 限制溢出，基线放矩形左下，保证文字进入文字层。
                fallback_fs = max(
                    min(last_fontsize, settings.min_font_size * 1.5),
                    settings.min_font_size,
                )
                try:
                    baseline = fitz.Point(rect.x0, rect.y1 - fallback_fs * 0.2)
                    page.insert_text(
                        baseline,
                        text,
                        fontsize=fallback_fs,
                        fontname=fontname,
                        fontfile=font_path,
                        color=(0, 0, 0),
                        render_mode=render_mode,
                        rotate=text_rotate,
                    )
                    written += 1
                    logger.debug(
                        "page %d block 写入文字层（insert_text 兜底）: rect=%s "
                        "fs=%.1f text=%r",
                        page_index,
                        rect,
                        fallback_fs,
                        text[:30],
                    )
                except Exception as e:
                    logger.warning(
                        "page %d block skipped (font retry exhausted + "
                        "fallback failed): rect=%s text=%r err=%s",
                        page_index,
                        rect,
                        text[:30],
                        e,
                    )
                    skipped += 1

        return written, skipped

    @staticmethod
    def rewrite_text_layer(
        doc: fitz.Document,
        pdf_document: PdfDocument,
        page_index: int,
        text_blocks: list,
        preproc_angle: int,
        pdf_settings: object | None = None,
        font_path: str | None = None,
    ) -> tuple[int, int]:
        """删除整页文字层后，按 text_blocks 全量重写。

        供"保存"时把用户编辑后的块写回 PDF。先 redact 清空旧文字层，
        再逐块写入（复用 _write_blocks_to_page，与首次写入逻辑一致）。

        Args:
            doc: fitz.Document 实例。
            pdf_document: PdfDocument 状态对象。
            page_index: 页码索引。
            text_blocks: 编辑后的 TextBlock 列表（归一化 [0,1000] bbox）。
            preproc_angle: OCR 预处理旋转角度。
            pdf_settings: PdfGlobalSettings 实例（None 则使用默认值）。
            font_path: 调用方预先解析的共享子集字体路径（保存时整文档共享单一
                子集）。None 时按本页字符内部解析子集。

        Returns:
            (written, skipped) 成功写入与被跳过的文本块数量。
        """
        from vibeocr.backend.models.pdf_ocr_options import PdfGlobalSettings

        settings = pdf_settings if pdf_settings is not None else PdfGlobalSettings()

        # 先删除旧文字层（redact 全页文字，保留图片）
        # 注意：delete_text_layers 会 _clear_page_layer_info 并清空 ocr_text_blocks，
        # 所以必须在删除后重新设置 ocr_text_blocks。
        PdfService.delete_text_layers(doc, pdf_document, page_index)

        written, skipped = PdfService._write_blocks_to_page(
            doc, page_index, text_blocks, preproc_angle, settings, font_path=font_path
        )

        # 重设缓存（delete 清空了）
        pdf_document.is_modified = True
        info = pdf_document.pages[page_index]
        info.ocr_text_blocks = list(text_blocks)
        info.ocr_preproc_angle = preproc_angle
        info.has_text_layer = written > 0
        info.thumbnail = None
        return written, skipped

    @staticmethod
    def delete_text_layers(
        doc: fitz.Document,
        pdf_document: PdfDocument,
        page_index: int,
    ) -> tuple[int, int, bool]:
        """删除整页文字层（词级 redact + 循环验证至清零）。

        用 get_text("words") 取词级 bbox 建 redact（比 block 级精确，避免
        嵌套/合并文本块遗漏）。每轮 redact 后重新检测残留，仅当仍有文字才
        继续下一轮，最多 _DELETE_LAYER_MAX_ROUNDS 轮（防死循环）。

        Returns:
            (initial_word_count, rounds_used, has_residual)
            initial_word_count: 初始词数（删除前）；rounds_used: 实际执行轮数；
            has_residual: 多轮后是否仍有残留（True 需 UI 提示用户）。
        """
        page = doc[page_index]
        words = page.get_text("words")
        if not words:
            # 无文字 → 不做 redact，直接清状态
            PdfService._clear_page_layer_info(pdf_document, page_index)
            return 0, 0, False

        initial_word_count = len(words)
        rounds_used = 0
        for round_idx in range(PdfService._DELETE_LAYER_MAX_ROUNDS):
            current_words = page.get_text("words")
            if not current_words:
                break
            for w in current_words:
                page.add_redact_annot(fitz.Rect(w[:4]), fill=None)
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)  # type: ignore[attr-defined]
            rounds_used = round_idx + 1

        has_residual = bool(page.get_text().strip())
        if has_residual:
            logger.warning(
                "page %d 经 %d 轮 redact 仍有残留: %r",
                page_index,
                rounds_used,
                page.get_text()[:50],
            )

        pdf_document.is_modified = True
        PdfService._clear_page_layer_info(pdf_document, page_index)
        return initial_word_count, rounds_used, has_residual

    @staticmethod
    def _clear_page_layer_info(pdf_document: PdfDocument, page_index: int) -> None:
        """删除文字层后清空页状态（替代旧的 update_page_info 冗余重检）。"""
        if page_index >= len(pdf_document.pages):
            return
        info = pdf_document.pages[page_index]
        info.has_text_layer = False
        info.text_layers = []
        info.is_scanned = False
        info.ocr_text_blocks = []
        info.ocr_preproc_angle = 0

    # ---- bbox coordinate transforms --------------------------------

    @staticmethod
    def _denormalize_and_unrotate_bbox(
        bbox: tuple[float, float, float, float],
        preproc_angle: int,
        page_rect: fitz.Rect,
    ) -> fitz.Rect:
        """将 [0, 1000] 归一化 bbox 逆旋转后映射到 PDF 页面坐标。

        当 OCR 预处理旋转了图像（preproc_angle），bbox 坐标在旋转后的空间中。
        此方法执行逆变换，将坐标映射回原始页面坐标。

        Args:
            bbox: 归一化坐标 (x0, y0, x1, y1)，范围 [0, 1000]。
            preproc_angle: 预处理旋转角度 (0, 90, 180, 270)。
            page_rect: PDF 页面矩形 (points)。

        Returns:
            映射后的 fitz.Rect。
        """
        nx0, ny0, nx1, ny1 = (
            bbox[0] / 1000,
            bbox[1] / 1000,
            bbox[2] / 1000,
            bbox[3] / 1000,
        )
        pw, ph = page_rect.width, page_rect.height

        if preproc_angle == 90:
            # PaddleOCR 实测约定（scripts/verify_orient_roundtrip2/3.py）：
            # reported angle == 内容相对正向「顺时针」偏转度数；output_img 是把图
            # 「逆时针」旋转 angle 度得到的正向图，bbox 在 output(正向)空间。
            # 还原回「显示空间」(= 顺时针偏转 angle 度)：把 output bbox「顺时针 90°」。
            # 顺时针 90°: (1-y)→x, x→y
            x0 = (1 - ny1) * pw
            y0 = nx0 * ph
            x1 = (1 - ny0) * pw
            y1 = nx1 * ph
        elif preproc_angle == 180:
            # 中心对称
            x0 = (1 - nx1) * pw
            y0 = (1 - ny1) * ph
            x1 = (1 - nx0) * pw
            y1 = (1 - ny0) * ph
        elif preproc_angle == 270:
            # 顺时针 270° = 逆时针 90°: y→x, (1-x)→y
            x0 = ny0 * pw
            y0 = (1 - nx1) * ph
            x1 = ny1 * pw
            y1 = (1 - nx0) * ph
        else:
            # 0° 或未知角度：直接映射
            x0 = nx0 * pw
            y0 = ny0 * ph
            x1 = nx1 * pw
            y1 = ny1 * ph

        return fitz.Rect(x0, y0, x1, y1)

    @staticmethod
    def _denormalize_and_unrotate_polygon(
        polygon: tuple[float, ...],
        preproc_angle: int,
        page_rect: fitz.Rect,
    ) -> list[fitz.Point]:
        """将 [0,1000] 归一化的 4 点多边形逆旋转到 PDF 页面坐标（不塌缩成 AABB）。

        与 _denormalize_and_unrotate_bbox 同一套象限旋转逻辑，但作用于 4 个点，
        保留多边形的顶点排序（编码阅读方向），不取 min/max 外接矩形。
        返回显示空间的 4 个 Point，顺序与输入多边形一致（PaddleOCR 顺时针 TL,TR,BR,BL）。
        """
        pw, ph = page_rect.width, page_rect.height
        pts: list[fitz.Point] = []
        for i in range(0, len(polygon) - 1, 2):
            nx, ny = polygon[i] / 1000, polygon[i + 1] / 1000
            if preproc_angle == 90:
                x, y = (1 - ny) * pw, nx * ph
            elif preproc_angle == 180:
                x, y = (1 - nx) * pw, (1 - ny) * ph
            elif preproc_angle == 270:
                x, y = ny * pw, (1 - nx) * ph
            else:
                x, y = nx * pw, ny * ph
            pts.append(fitz.Point(x, y))
        return pts

    @staticmethod
    def _poly_orientation(polygon_pts: list[fitz.Point] | None, text: str) -> str:
        """判断文本行方向（horizontal/vertical）。

        - 单字符：永远 horizontal（排版公理——竖排至少 2 字；单字符字形天生高瘦，
          多边形长边落在垂直方向，会误判成 vertical，故单字符不看几何）。
        - 多字符 + 有多边形：按顶点排序判——PaddleOCR 顺时针 TL,TR,BR,BL，
          顶边(TL→TR)为长边则横排，左边(TL→BL)为长边则竖排（实测横排顶边角≈0°、
          竖排左边角≈90°）。这是用真实阅读方向替代 bbox 长宽比启发式的核心。
        - 多字符 + 无多边形：返回 unknown（调用方回退长宽比，绝不回归）。
        """
        if len(text.strip()) <= 1:
            return "horizontal"
        if polygon_pts is None or len(polygon_pts) < 4:
            return "unknown"
        import math

        tl, tr, _br, bl = polygon_pts[0], polygon_pts[1], polygon_pts[2], polygon_pts[3]
        top_len = math.hypot(tr.x - tl.x, tr.y - tl.y)
        left_len = math.hypot(bl.x - tl.x, bl.y - tl.y)
        # 顶边明显比左边长 → 横排（含等长容差，抗多边形轻微倾斜）。
        return "horizontal" if top_len >= left_len * 0.9 else "vertical"

    # bbox_to_pixel 已抽到 vibeocr.backend.utils.pdf_coords(纯数学,无 fitz 依赖),
    # 供主进程 PDF 预览窗口使用。保留 PdfService 入口以兼容旧调用方和测试。
    @staticmethod
    def bbox_to_pixel(
        bbox: tuple[float, float, float, float],
        page_rect: fitz.Rect | tuple[float, float, float, float],
        render_dpi: int,
        source: str = "pdf",
        rotation: int = 0,
        mediabox: tuple[float, float, float, float] | None = None,
    ) -> tuple[float, float, float, float]:
        from vibeocr.backend.utils.pdf_coords import bbox_to_pixel

        coordinates = (
            (page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1)
            if isinstance(page_rect, fitz.Rect)
            else page_rect
        )
        return bbox_to_pixel(
            bbox,
            coordinates,
            render_dpi,
            source=source,
            rotation=rotation,
            mediabox=mediabox,
        )

    # ---- helpers ----------------------------------------------------

    @staticmethod
    def invalidate_thumbnails(
        pdf_document: PdfDocument, page_indices: list[int]
    ) -> None:
        for idx in page_indices:
            if 0 <= idx < len(pdf_document.pages):
                pdf_document.pages[idx].thumbnail = None
