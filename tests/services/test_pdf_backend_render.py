"""测试 PDF 后端渲染并行化（独立 fitz.Document 并发栅格化）。

回归任务1的改造：render_preview/render_thumbnail 不再用 session.doc +
per-session fitz_lock 串行化，而是每次打开独立临时 fitz.Document 栅格化。
不同 Document 实例可安全并行（PyMuPDF 线程不安全仅限同一 Document 实例）。

验证点：
1. 多页并发渲染全部成功、后端不崩溃（无 native 段错误）。
2. 并发渲染显著快于串行（证明栅格化真正并行，而非仍被锁串行化）。
3. 页索引越界返回 400 而非 500。
4. 渲染期间 session.remove() 仍能正确等待（active_ops 同步）。
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import fitz
import pytest

from vibeocr.backend.services.pdf_backend_client import PdfBackendClient


@pytest.fixture
def backend_client():
    """启动真实 PDF 后端子进程的客户端。"""
    client = PdfBackendClient()
    yield client
    client.stop()


@pytest.fixture
def heavy_pdf(tmp_path):
    """高 DPI 栅格化耗时的多页 PDF。

    用较大的页面尺寸 + 嵌入图片，让 get_pixmap 有真实工作量（每页 ~50ms+），
    这样串行 vs 并行的耗时可测、差异稳定。
    """
    path = tmp_path / "heavy.pdf"
    doc = fitz.open()
    # 生成一张可压缩的图片（渐变），栅格化时有真实工作量
    import io

    from PIL import Image

    img = Image.new("RGB", (1200, 1600))
    px = img.load()
    assert px is not None
    for y in range(1600):
        for x in range(1200):
            px[x, y] = ((x * y) % 256, (x + y) % 256, (x - y) % 256)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    img_bytes = buf.getvalue()

    for i in range(8):
        page = doc.new_page(width=612, height=792)
        page.insert_image(page.rect, stream=img_bytes)
        page.insert_text((72, 72), f"Page {i + 1}", fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


class TestRenderParallelization:
    """验证渲染并行化：独立 Document 实例并发栅格化。"""

    def test_concurrent_render_preview_all_succeed(self, backend_client, heavy_pdf):
        """8 页并发 render_preview 应全部成功返回有效 PNG。"""
        open_resp = backend_client.open_session(str(heavy_pdf))
        sid = open_resp.session_id
        total = len(open_resp.model.pages)
        assert total == 8

        def render_one(page_idx):
            return backend_client.render_preview(sid, page_idx, dpi=150)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(render_one, range(total)))

        assert len(results) == total
        for png in results:
            assert isinstance(png, bytes)
            assert len(png) > 0
            # 验证是有效 PNG
            assert png[:8] == b"\x89PNG\r\n\x1a\n", "应返回 PNG 字节流"

    def test_concurrent_render_enters_rasterizer_in_parallel(self, monkeypatch):
        """两个预览请求必须同时进入独立 Document 栅格化路径。

        这里用栅栏直接验证并发重叠，不再用共享 CI runner 上接近 1.0x 的
        耗时比推断并发。若 render_preview 重新获取 session.fitz_lock，首个
        请求会在栅栏超时，测试将确定性失败。
        """
        from vibeocr.backend.ipc.schemas import RenderPreviewRequest
        from vibeocr.backend.services import pdf_backend_process as backend

        session = backend.BackendSession(
            session_id="parallel-render",
            file_path="unused.pdf",
            doc=MagicMock(),
            pdf_document=MagicMock(),
        )
        registry = MagicMock()
        registry.get.return_value = session
        monkeypatch.setattr(backend, "_get_registry", lambda: registry)

        entered = threading.Barrier(2, timeout=5.0)

        def rasterize(_file_path: str, _page_index: int, _dpi: float):
            entered.wait()
            return b"\x00\x00\x00", 1, 1

        monkeypatch.setattr(backend, "_render_page_pixels", rasterize)
        request = RenderPreviewRequest(page=0, dpi=150)

        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(
                pool.map(
                    lambda _: backend.render_preview(session.session_id, request),
                    range(2),
                )
            )

        assert all(response.media_type == "image/png" for response in responses)

    def test_render_preview_invalid_page_returns_400(self, backend_client, heavy_pdf):
        """页索引越界应返回 400（而非 500）。"""
        from vibeocr.backend.services.pdf_backend_client import PdfBackendError

        open_resp = backend_client.open_session(str(heavy_pdf))
        sid = open_resp.session_id
        total = len(open_resp.model.pages)

        with pytest.raises(PdfBackendError) as exc_info:
            backend_client.render_preview(sid, total + 100, dpi=72)
        # PdfBackendError 的 message 含 HTTP 状态码
        assert "400" in str(exc_info.value), f"越界应返回 400，实际：{exc_info.value}"
