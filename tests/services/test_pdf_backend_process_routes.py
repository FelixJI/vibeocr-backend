"""pdf_backend_process FastAPI 路由覆盖。

用 TestClient 直接打 app，覆盖所有路由 + _lifespan + mirror/diff 构造 +
_render_page_pixels 错误路径。不启动真实子进程。
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import fitz
import numpy as np
import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def app_client(tmp_path):
    """TestClient + 隔离的全局 registry。

    用 TestClient context manager 触发 _lifespan startup/shutdown。
    """
    from vibeocr.backend.services import pdf_backend_process as backend

    # 重置全局 registry
    backend._REGISTRY = None

    with TestClient(backend.app) as client:
        yield client, backend

    # 重置
    backend._REGISTRY = None


def _create_test_pdf(path: Path, num_pages: int = 2) -> Path:
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), f"Page {i + 1}", fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


def _create_scanned_pdf(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    img = np.ones((792, 612, 3), dtype=np.uint8) * 240
    cs = fitz.Colorspace(fitz.CS_RGB)
    pixmap = fitz.Pixmap(cs, 612, 792, img.tobytes(), 0)
    page.insert_image(fitz.Rect(0, 0, 612, 792), pixmap=pixmap)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def opened_session(app_client, tmp_path):
    """打开一个 session 并返回 (client, backend, sid, path)。"""
    client, backend = app_client
    path = _create_test_pdf(tmp_path / "test.pdf", num_pages=3)
    resp = client.post("/session/open", json={"path": str(path)})
    assert resp.status_code == 200, resp.text
    sid = resp.json()["session_id"]
    return client, backend, sid, str(path)


# ---- 基础路由 ----------------------------------------------------------


class TestHealthAndOpen:
    def test_health(self, app_client):
        client, _ = app_client
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "pid" in body
        assert body["sessions"] == 0

    def test_open_success(self, app_client, tmp_path):
        client, _ = app_client
        path = _create_test_pdf(tmp_path / "t.pdf", num_pages=2)
        resp = client.post("/session/open", json={"path": str(path)})
        assert resp.status_code == 200
        body = resp.json()
        assert "session_id" in body
        assert body["model"]["file_path"] == str(path)
        assert len(body["model"]["pages"]) == 2

    def test_open_file_not_found_returns_400(self, app_client):
        client, _ = app_client
        resp = client.post("/session/open", json={"path": "/nonexistent/x.pdf"})
        assert resp.status_code == 400

    def test_open_internal_error_returns_500(self, app_client, tmp_path, monkeypatch):
        client, backend = app_client
        path = _create_test_pdf(tmp_path / "t.pdf")

        def _boom(file_path):
            raise RuntimeError("boom")

        monkeypatch.setattr(backend.PdfService, "open_doc", _boom)
        resp = client.post("/session/open", json={"path": str(path)})
        assert resp.status_code == 500

    def test_get_registry_singleton(self):
        """_get_registry 懒初始化单例（lines 80-84）。"""
        from vibeocr.backend.services import pdf_backend_process as backend

        backend._REGISTRY = None
        r1 = backend._get_registry()
        r2 = backend._get_registry()
        assert r1 is r2
        backend._REGISTRY = None


class TestSessionLifecycle:
    def test_close_session(self, opened_session):
        client, _, sid, _ = opened_session
        resp = client.post(f"/session/{sid}/close")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_close_unknown_session_silent(self, app_client):
        """remove 对未知 session 静默处理（返回 200）。"""
        client, _ = app_client
        resp = client.post("/session/unknown/close")
        assert resp.status_code == 200

    def test_model_refresh(self, opened_session):
        client, _, sid, _ = opened_session
        resp = client.post(f"/session/{sid}/model")
        assert resp.status_code == 200
        assert "pages" in resp.json()

    def test_model_unknown_session_404(self, app_client):
        client, _ = app_client
        resp = client.post("/session/unknown/model")
        assert resp.status_code == 404


class TestLoadStream:
    def test_load_streams_pages(self, opened_session):
        client, _, sid, _ = opened_session
        resp = client.post(f"/session/{sid}/load")
        assert resp.status_code == 200
        # NDJSON：多行 JSON
        lines = [line for line in resp.text.strip().split("\n") if line]
        assert len(lines) >= 2  # 至少每页一条 + done
        last = __import__("json").loads(lines[-1])
        assert last["message"] == "done"

    def test_load_unknown_session_404(self, app_client):
        client, _ = app_client
        resp = client.post("/session/unknown/load")
        assert resp.status_code == 404


# ---- 渲染 --------------------------------------------------------------


class TestRenderRoutes:
    def test_render_thumbnail(self, opened_session):
        client, _, sid, _ = opened_session
        resp = client.post(
            f"/session/{sid}/render_thumbnail", json={"page": 0, "size": 128}
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_render_preview(self, opened_session):
        client, _, sid, _ = opened_session
        resp = client.post(
            f"/session/{sid}/render_preview", json={"page": 0, "dpi": 72}
        )
        assert resp.status_code == 200
        assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_render_thumbnail_invalid_page_400(self, opened_session):
        client, _, sid, _ = opened_session
        resp = client.post(
            f"/session/{sid}/render_thumbnail", json={"page": 999, "size": 128}
        )
        assert resp.status_code == 400

    def test_render_preview_invalid_page_400(self, opened_session):
        client, _, sid, _ = opened_session
        resp = client.post(
            f"/session/{sid}/render_preview", json={"page": 999, "dpi": 72}
        )
        assert resp.status_code == 400

    def test_render_thumbnail_internal_error_500(
        self, opened_session, monkeypatch
    ):
        client, backend, sid, _ = opened_session

        def _boom(file_path, page_index, dpi):
            raise RuntimeError("render boom")

        monkeypatch.setattr(backend, "_render_page_pixels", _boom)
        resp = client.post(
            f"/session/{sid}/render_thumbnail", json={"page": 0, "size": 128}
        )
        assert resp.status_code == 500

    def test_render_page_pixels_close_exception_swallowed(self, tmp_path, monkeypatch):
        """_render_page_pixels: doc.close 抛异常应被吞（lines 195-198）。"""
        from vibeocr.backend.services import pdf_backend_process as backend

        path = _create_test_pdf(tmp_path / "rp.pdf", num_pages=1)
        # 让 fitz.Document.close 抛异常
        original_close = fitz.Document.close

        def _fail_close(self):
            raise RuntimeError("close boom")

        monkeypatch.setattr(fitz.Document, "close", _fail_close)
        try:
            samples, w, h = backend._render_page_pixels(str(path), 0, 72.0)
            assert isinstance(samples, bytes)
            assert w > 0 and h > 0
        finally:
            monkeypatch.setattr(fitz.Document, "close", original_close)


# ---- detect_text_layers ------------------------------------------------


class TestDetectRoute:
    def test_detect_text_layers(self, opened_session):
        client, _, sid, _ = opened_session
        resp = client.post(
            f"/session/{sid}/detect_text_layers", json={"page": 0}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "text_layers" in body

    def test_detect_text_layers_internal_error_500(
        self, opened_session, monkeypatch
    ):
        client, backend, sid, _ = opened_session

        def _boom(doc, page):
            raise RuntimeError("detect boom")

        monkeypatch.setattr(backend.PdfService, "detect_text_layers", _boom)
        resp = client.post(
            f"/session/{sid}/detect_text_layers", json={"page": 0}
        )
        assert resp.status_code == 500


# ---- 变更操作 ----------------------------------------------------------


class TestMutateRoutes:
    def test_rotate(self, opened_session):
        client, _, sid, _ = opened_session
        resp = client.post(
            f"/session/{sid}/rotate", json={"pages": [0], "angle": 90}
        )
        assert resp.status_code == 200
        diff = resp.json()["diff"]
        assert "replaced_pages" in diff

    def test_rotate_internal_error_500(self, opened_session, monkeypatch):
        client, backend, sid, _ = opened_session

        def _boom(*a, **k):
            raise RuntimeError("rot boom")

        monkeypatch.setattr(backend.PdfService, "rotate_pages", _boom)
        resp = client.post(
            f"/session/{sid}/rotate", json={"pages": [0], "angle": 90}
        )
        assert resp.status_code == 500

    def test_delete_pages(self, opened_session):
        client, _, sid, _ = opened_session
        resp = client.post(
            f"/session/{sid}/delete_pages", json={"pages": [1]}
        )
        assert resp.status_code == 200
        assert resp.json()["diff"]["structural_change"] is True

    def test_insert_blank(self, opened_session):
        client, _, sid, _ = opened_session
        resp = client.post(
            f"/session/{sid}/insert_blank",
            json={"after_index": 0, "width": 612, "height": 792},
        )
        assert resp.status_code == 200

    def test_insert_from(self, opened_session, tmp_path):
        client, _, sid, _ = opened_session
        other = _create_test_pdf(tmp_path / "other.pdf", num_pages=1)
        resp = client.post(
            f"/session/{sid}/insert_from",
            json={"source_path": str(other), "after_index": 0},
        )
        assert resp.status_code == 200

    def test_move_page(self, opened_session):
        client, _, sid, _ = opened_session
        resp = client.post(
            f"/session/{sid}/move_page", json={"from_index": 0, "to_index": 2}
        )
        assert resp.status_code == 200

    def test_reorder(self, opened_session):
        client, _, sid, _ = opened_session
        resp = client.post(
            f"/session/{sid}/reorder", json={"new_order": [2, 1, 0]}
        )
        assert resp.status_code == 200

    def test_insert_blank_internal_error_500(self, opened_session, monkeypatch):
        client, backend, sid, _ = opened_session
        monkeypatch.setattr(
            backend.PdfService, "insert_blank_page", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        )
        resp = client.post(
            f"/session/{sid}/insert_blank",
            json={"after_index": 0, "width": 612, "height": 792},
        )
        assert resp.status_code == 500

    def test_insert_from_internal_error_500(self, opened_session, monkeypatch):
        client, backend, sid, _ = opened_session
        monkeypatch.setattr(
            backend.PdfService, "insert_pages_from", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        )
        resp = client.post(
            f"/session/{sid}/insert_from",
            json={"source_path": "x", "after_index": 0},
        )
        assert resp.status_code == 500

    def test_move_page_internal_error_500(self, opened_session, monkeypatch):
        client, backend, sid, _ = opened_session
        monkeypatch.setattr(
            backend.PdfService, "move_page", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        )
        resp = client.post(
            f"/session/{sid}/move_page", json={"from_index": 0, "to_index": 1}
        )
        assert resp.status_code == 500

    def test_reorder_internal_error_500(self, opened_session, monkeypatch):
        client, backend, sid, _ = opened_session
        monkeypatch.setattr(
            backend.PdfService, "reorder_pages", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        )
        resp = client.post(
            f"/session/{sid}/reorder", json={"new_order": [0, 1, 2]}
        )
        assert resp.status_code == 500

    def test_delete_pages_internal_error_500(self, opened_session, monkeypatch):
        client, backend, sid, _ = opened_session
        monkeypatch.setattr(
            backend.PdfService, "delete_pages", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        )
        resp = client.post(
            f"/session/{sid}/delete_pages", json={"pages": [0]}
        )
        assert resp.status_code == 500


# ---- 文字层操作 --------------------------------------------------------


class TestTextLayerRoutes:
    def _ocr_result_dict(self):
        return {
            "preproc_angle": 0,
            "text_blocks": [
                {
                    "text": "hello",
                    "score": 0.9,
                    "bbox": [50, 50, 200, 100],
                    "page_idx": 0,
                    "label": "text",
                    "order": 0,
                }
            ],
        }

    def test_add_text_layer(self, opened_session):
        client, _, sid, _ = opened_session
        resp = client.post(
            f"/session/{sid}/add_text_layer",
            json={
                "page": 0,
                "ocr_result": self._ocr_result_dict(),
                "overwrite": False,
                "pdf_settings": None,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["diff"]["modified_flag"] is True

    def test_add_text_layer_internal_error_500(self, opened_session, monkeypatch):
        client, backend, sid, _ = opened_session
        monkeypatch.setattr(
            backend.PdfService, "add_text_layer", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        )
        resp = client.post(
            f"/session/{sid}/add_text_layer",
            json={"page": 0, "ocr_result": self._ocr_result_dict()},
        )
        assert resp.status_code == 500

    def test_add_text_layer_batch_no_save(self, opened_session):
        client, _, sid, _ = opened_session
        resp = client.post(
            f"/session/{sid}/add_text_layer_batch",
            json={
                "pages": [{"page": 0, "ocr_result": self._ocr_result_dict()}],
                "overwrite": False,
                "save": False,
                "pdf_settings": None,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "diff" in body
        assert body.get("extra") is None

    def test_add_text_layer_batch_with_save(self, opened_session):
        client, _, sid, _path = opened_session
        resp = client.post(
            f"/session/{sid}/add_text_layer_batch",
            json={
                "pages": [{"page": 0, "ocr_result": self._ocr_result_dict()}],
                "overwrite": True,
                "save": True,
                "pdf_settings": None,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["extra"]["saved"] is True

    def test_add_text_layer_batch_internal_error_500(
        self, opened_session, monkeypatch
    ):
        client, backend, sid, _ = opened_session
        monkeypatch.setattr(
            backend.PdfService,
            "add_text_layer_batch",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")),
        )
        resp = client.post(
            f"/session/{sid}/add_text_layer_batch",
            json={
                "pages": [{"page": 0, "ocr_result": self._ocr_result_dict()}],
            },
        )
        assert resp.status_code == 500

    def test_add_text_layer_batch_save_falls_back_to_compress(
        self, app_client, tmp_path, monkeypatch
    ):
        """save_incremental 失败 → _compress_in_place 回退（lines 644-671）。"""
        from vibeocr.backend.services import pdf_backend_process as backend

        backend._REGISTRY = None
        client, _ = app_client
        path = _create_scanned_pdf(tmp_path / "batch_compress.pdf")
        resp = client.post("/session/open", json={"path": str(path)})
        sid = resp.json()["session_id"]

        # save_incremental 返回 False → 走 _compress_in_place
        monkeypatch.setattr(backend.PdfService, "save_incremental", lambda *a, **k: False)
        # 不 mock _compress_in_place，让它真正跑（单页扫描件可成功）

        resp = client.post(
            f"/session/{sid}/add_text_layer_batch",
            json={
                "pages": [{"page": 0, "ocr_result": self._ocr_result_dict()}],
                "save": True,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["extra"]["saved"] is True

    def test_add_text_layer_batch_save_compress_fail_reopens(
        self, app_client, tmp_path, monkeypatch
    ):
        """save_incremental 失败 + _compress_in_place 也失败 → 重开 doc（lines 657-671）。"""
        from vibeocr.backend.services import pdf_backend_process as backend

        backend._REGISTRY = None
        client, _ = app_client
        path = _create_scanned_pdf(tmp_path / "batch_fail.pdf")
        resp = client.post("/session/open", json={"path": str(path)})
        sid = resp.json()["session_id"]

        monkeypatch.setattr(backend.PdfService, "save_incremental", lambda *a, **k: False)

        def _compress_fail(doc, save_path, clean=False):
            # 模拟 _compress_in_place 失败：close doc + 抛异常
            try:
                doc.close()
            except Exception:
                pass
            raise RuntimeError("compress boom")

        monkeypatch.setattr(backend.PdfService, "_compress_in_place", _compress_fail)

        resp = client.post(
            f"/session/{sid}/add_text_layer_batch",
            json={
                "pages": [{"page": 0, "ocr_result": self._ocr_result_dict()}],
                "save": True,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["extra"]["saved"] is False

    def test_rewrite_text_layer(self, opened_session):
        client, _, sid, _ = opened_session
        # 先加一层
        client.post(
            f"/session/{sid}/add_text_layer",
            json={"page": 0, "ocr_result": self._ocr_result_dict()},
        )
        resp = client.post(
            f"/session/{sid}/rewrite_text_layer",
            json={
                "page": 0,
                "preproc_angle": 0,
                "text_blocks": [
                    {
                        "text": "rewritten",
                        "score": 0.9,
                        "bbox": [50, 50, 200, 100],
                        "page_idx": 0,
                        "label": "text",
                        "order": 0,
                    }
                ],
                "pdf_settings": None,
            },
        )
        assert resp.status_code == 200

    def test_rewrite_text_layer_internal_error_500(
        self, opened_session, monkeypatch
    ):
        client, backend, sid, _ = opened_session
        monkeypatch.setattr(
            backend.PdfService, "rewrite_text_layer", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        )
        resp = client.post(
            f"/session/{sid}/rewrite_text_layer",
            json={"page": 0, "preproc_angle": 0, "text_blocks": []},
        )
        assert resp.status_code == 500

    def test_update_block_text(self, opened_session):
        client, _, sid, _ = opened_session
        # 先加层
        client.post(
            f"/session/{sid}/add_text_layer",
            json={"page": 0, "ocr_result": self._ocr_result_dict()},
        )
        resp = client.post(
            f"/session/{sid}/update_block_text",
            json={"page": 0, "block_index": 0, "new_text": "edited"},
        )
        assert resp.status_code == 200

    def test_update_block_text_out_of_range_noop(self, opened_session):
        client, _, sid, _ = opened_session
        resp = client.post(
            f"/session/{sid}/update_block_text",
            json={"page": 0, "block_index": 999, "new_text": "x"},
        )
        assert resp.status_code == 200

    def test_update_block_text_internal_error_500(
        self, opened_session, monkeypatch
    ):
        client, backend, sid, _ = opened_session
        # 让 pages[req.page] 访问抛异常
        session = backend._get_registry().get(sid)
        original_pages = session.pdf_document.pages

        class _BoomList:
            def __getitem__(self, idx):
                raise RuntimeError("pages boom")

            def __len__(self):
                return 1

        session.pdf_document.pages = _BoomList()
        try:
            resp = client.post(
                f"/session/{sid}/update_block_text",
                json={"page": 0, "block_index": 0, "new_text": "x"},
            )
            assert resp.status_code == 500
        finally:
            session.pdf_document.pages = original_pages


# ---- 流式删除文字层 ----------------------------------------------------


class TestDeleteLayersStream:
    def test_delete_text_layers_stream(self, opened_session):
        client, _, sid, _ = opened_session
        # 先加层
        client.post(
            f"/session/{sid}/add_text_layer",
            json={
                "page": 0,
                "ocr_result": {
                    "preproc_angle": 0,
                    "text_blocks": [
                        {"text": "hi", "score": 0.9, "bbox": [50, 50, 200, 100]}
                    ],
                },
            },
        )
        resp = client.post(
            f"/session/{sid}/delete_text_layers", json={"pages": [0]}
        )
        assert resp.status_code == 200
        lines = [line for line in resp.text.strip().split("\n") if line]
        assert len(lines) >= 2
        last = __import__("json").loads(lines[-1])
        assert last["message"] == "done"

    def test_delete_text_layers_no_text_page(self, opened_session):
        """无文字页删除应正常返回（payload=(0,0,False)）。"""
        client, _, sid, _ = opened_session
        resp = client.post(
            f"/session/{sid}/delete_text_layers", json={"pages": [0]}
        )
        assert resp.status_code == 200

    def test_delete_text_layers_page_error_continues(self, opened_session, monkeypatch):
        """单页异常应被捕获、继续（lines 783-788）。"""
        client, backend, sid, _ = opened_session

        def _boom(doc, pdf_document, page):
            raise RuntimeError("delete boom")

        monkeypatch.setattr(backend.PdfService, "delete_text_layers", _boom)
        # page_has_text 也抛，使异常在锁内发生
        monkeypatch.setattr(
            backend.PdfService, "page_has_text", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ph boom"))
        )
        resp = client.post(
            f"/session/{sid}/delete_text_layers", json={"pages": [0]}
        )
        assert resp.status_code == 200
        lines = [line for line in resp.text.strip().split("\n") if line]
        # 第一行 payload 应为 None（异常路径）
        first = __import__("json").loads(lines[0])
        assert first["page_payload"] is None


# ---- save 路由 ---------------------------------------------------------


class TestSaveRoute:
    def test_save_in_place(self, opened_session):
        client, _, sid, _ = opened_session
        # 改动一下
        client.post(f"/session/{sid}/rotate", json={"pages": [0], "angle": 90})
        resp = client.post(
            f"/session/{sid}/save",
            json={"path": None, "pdf_settings": None, "rewrite_text_layers": False},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["path"]

    def test_save_to_new_path(self, opened_session, tmp_path):
        client, _, sid, _ = opened_session
        new_path = str(tmp_path / "saved.pdf")
        resp = client.post(
            f"/session/{sid}/save",
            json={"path": new_path, "pdf_settings": None, "rewrite_text_layers": False},
        )
        assert resp.status_code == 200
        assert resp.json()["path"] == new_path

    def test_save_internal_error_500(self, opened_session, monkeypatch):
        client, backend, sid, _ = opened_session
        monkeypatch.setattr(
            backend.PdfService, "save_with_rewrite", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        )
        resp = client.post(
            f"/session/{sid}/save",
            json={"path": None, "pdf_settings": None, "rewrite_text_layers": False},
        )
        assert resp.status_code == 500


# ---- 其它路由 ----------------------------------------------------------


class TestOtherRoutes:
    def test_deskew_returns_501(self, opened_session):
        client, _, sid, _ = opened_session
        resp = client.post(f"/session/{sid}/deskew", json={"pages": [0]})
        assert resp.status_code == 501

    def test_cancel(self, opened_session):
        client, _, sid, _ = opened_session
        resp = client.post(f"/session/{sid}/cancel")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_reset_cancel(self, opened_session):
        client, _, sid, _ = opened_session
        resp = client.post(f"/session/{sid}/reset_cancel")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True


# ---- mirror/diff 构造函数 ---------------------------------------------


class TestMirrorBuilders:
    def test_text_layer_to_mirror(self):
        from vibeocr.backend.models.pdf_document import TextLayerInfo
        from vibeocr.backend.services.pdf_backend_process import _text_layer_to_mirror

        tl = TextLayerInfo(index=0, text_preview="hi", char_count=2, bbox=(1, 2, 3, 4), color_id=0)
        m = _text_layer_to_mirror(tl)
        assert m.text_preview == "hi"
        assert m.char_count == 2

    def test_text_block_to_mirror(self):
        from vibeocr.backend.models.ocr_result import TextBlock
        from vibeocr.backend.services.pdf_backend_process import _text_block_to_mirror

        b = TextBlock(text="x", score=0.5, bbox=(1, 2, 3, 4), page_idx=0)
        m = _text_block_to_mirror(b)
        assert m.text == "x"
        assert m.score == 0.5

    def test_page_to_mirror(self):
        from vibeocr.backend.models.pdf_document import PdfPageInfo
        from vibeocr.backend.services.pdf_backend_process import _page_to_mirror

        info = PdfPageInfo(page_index=0, rotation=90)
        m = _page_to_mirror(info)
        assert m.page_index == 0
        assert m.rotation == 90

    def test_doc_to_mirror(self):
        from vibeocr.backend.models.pdf_document import PdfDocument
        from vibeocr.backend.services.pdf_backend_process import _doc_to_mirror

        d = PdfDocument(file_path="x.pdf")
        d.pages = []
        m = _doc_to_mirror(d)
        assert m.file_path == "x.pdf"

    def test_diff_full(self):
        from vibeocr.backend.models.pdf_document import PdfDocument
        from vibeocr.backend.services.pdf_backend_process import _diff_full

        d = PdfDocument(file_path="x.pdf")
        d.pages = []
        diff = _diff_full(d)
        assert diff.structural_change is True
        assert diff.full_model is not None

    def test_diff_pages_with_invalidate(self):
        from vibeocr.backend.models.pdf_document import PdfDocument
        from vibeocr.backend.services.pdf_backend_process import _diff_pages

        d = PdfDocument(file_path="x.pdf")
        from vibeocr.backend.models.pdf_document import PdfPageInfo

        d.pages = [PdfPageInfo(page_index=0), PdfPageInfo(page_index=1)]
        diff = _diff_pages(d, [0, 1], invalidate_thumbnails=[0], modified=True)
        assert len(diff.replaced_pages) == 2
        assert diff.invalidated_thumbnails == [0]
        assert diff.modified_flag is True

    def test_diff_pages_out_of_range_filtered(self):
        """_diff_pages 中越界页索引应被过滤（line 151 条件）。"""
        from vibeocr.backend.models.pdf_document import PdfDocument, PdfPageInfo
        from vibeocr.backend.services.pdf_backend_process import _diff_pages

        d = PdfDocument(file_path="x.pdf")
        d.pages = [PdfPageInfo(page_index=0)]
        diff = _diff_pages(d, [0, 999])  # 999 越界
        assert len(diff.replaced_pages) == 1


# ---- registry 边界 ----------------------------------------------------


class TestRegistryEdges:
    def test_get_unknown_raises_404(self):
        from fastapi import HTTPException

        from vibeocr.backend.services.pdf_backend_process import SessionRegistry

        reg = SessionRegistry()
        with pytest.raises(HTTPException) as exc:
            reg.get("unknown")
        assert exc.value.status_code == 404

    def test_count(self):
        from vibeocr.backend.services.pdf_backend_process import SessionRegistry

        reg = SessionRegistry()
        assert reg.count() == 0

    def test_fitz_op_rejects_closing_session(self):
        """_fitz_op 对非 OPEN 状态抛 409（lines 285-286）。"""
        from fastapi import HTTPException

        from vibeocr.backend.services.pdf_backend_process import (
            BackendSession,
            _fitz_op,
        )

        s = BackendSession(
            session_id="x",
            file_path="x",
            doc=MagicMock(),
            pdf_document=MagicMock(),
        )
        s.state = "CLOSING"
        with pytest.raises(HTTPException) as exc:
            with _fitz_op(s):
                pass
        assert exc.value.status_code == 409


# ---- _settings_from_dict ---------------------------------------------


class TestSettingsFromDict:
    def test_none_returns_none(self):
        from vibeocr.backend.services.pdf_backend_process import _settings_from_dict

        assert _settings_from_dict(None) is None

    def test_dict_returns_settings(self):
        from vibeocr.backend.models.pdf_ocr_options import PdfGlobalSettings
        from vibeocr.backend.services.pdf_backend_process import _settings_from_dict

        s = _settings_from_dict({"compress_on_save": False})
        assert isinstance(s, PdfGlobalSettings)
        assert s.compress_on_save is False


# ---- load 异常路径 ----------------------------------------------------


class TestLoadEdgeCases:
    def test_load_page_exception_continues(self, opened_session, monkeypatch):
        """load 单页异常应被捕获、继续（lines 399-400）。"""
        client, backend, sid, _ = opened_session

        call = {"n": 0}

        def _boom(session, i):
            call["n"] += 1
            if call["n"] == 1:
                raise RuntimeError("load page boom")
            # 后续页正常
            return backend._detect_one_page(session, i)

        monkeypatch.setattr(backend, "_detect_one_page", _boom)
        resp = client.post(f"/session/{sid}/load")
        assert resp.status_code == 200
        # 仍应有 done 哨兵
        lines = [line for line in resp.text.strip().split("\n") if line]
        last = __import__("json").loads(lines[-1])
        assert last["message"] == "done"

    def test_load_cancelled_breaks_early(self, opened_session):
        """load 中 cancel_event 被 set → 提前 break（line 387）。"""
        client, backend, sid, _ = opened_session
        # 先设置 cancel
        session = backend._get_registry().get(sid)
        session.cancel_event.set()
        resp = client.post(f"/session/{sid}/load")
        assert resp.status_code == 200
        lines = [line for line in resp.text.strip().split("\n") if line]
        # 第一页就 break → 只有 done 哨兵
        assert len(lines) == 1
        last = __import__("json").loads(lines[-1])
        assert last["message"] == "done"


class TestRenderPreviewError:
    def test_render_preview_internal_error_500(self, opened_session, monkeypatch):
        """render_preview 非 HTTPException → 500（lines 466-467）。"""
        client, backend, sid, _ = opened_session

        def _boom(file_path, page_index, dpi):
            raise RuntimeError("preview boom")

        monkeypatch.setattr(backend, "_render_page_pixels", _boom)
        resp = client.post(
            f"/session/{sid}/render_preview", json={"page": 0, "dpi": 72}
        )
        assert resp.status_code == 500


class TestRegistryRemoveCloseException:
    def test_remove_swallows_doc_close_exception(self):
        """remove() 中 doc.close 抛异常应被吞（lines 267-268）。"""
        from vibeocr.backend.services.pdf_backend_process import (
            BackendSession,
            SessionRegistry,
        )

        reg = SessionRegistry()
        mock_doc = MagicMock()
        mock_doc.close.side_effect = RuntimeError("close boom")
        session = BackendSession(
            session_id="x",
            file_path="x.pdf",
            doc=mock_doc,
            pdf_document=MagicMock(),
        )
        reg._sessions["x"] = session
        # 不应抛异常
        reg.remove("x")
        assert session.state == "CLOSED"


class TestBatchSaveEdgeCases:
    def test_batch_save_no_file_path_skips_save(self, app_client, tmp_path):
        """save=True 但 file_path=None → 跳过 save（branch 633->672）。"""
        from vibeocr.backend.services import pdf_backend_process as backend

        backend._REGISTRY = None
        client, _ = app_client
        path = _create_scanned_pdf(tmp_path / "no_fp.pdf")
        resp = client.post("/session/open", json={"path": str(path)})
        sid = resp.json()["session_id"]
        # 清空 file_path
        session = backend._get_registry().get(sid)
        session.pdf_document.file_path = None

        resp = client.post(
            f"/session/{sid}/add_text_layer_batch",
            json={
                "pages": [
                    {
                        "page": 0,
                        "ocr_result": {
                            "preproc_angle": 0,
                            "text_blocks": [
                                {"text": "x", "score": 0.9, "bbox": [50, 50, 200, 100]}
                            ],
                        },
                    }
                ],
                "save": True,
            },
        )
        assert resp.status_code == 200
        # extra.saved 保持默认 True（未落盘但无失败标记）
        assert resp.json()["extra"]["saved"] is True

    def test_batch_save_incremental_exception(self, app_client, tmp_path, monkeypatch):
        """save_incremental 抛异常 → saved=False + 走 compress 回退（lines 637-643）。"""
        from vibeocr.backend.services import pdf_backend_process as backend

        backend._REGISTRY = None
        client, _ = app_client
        path = _create_scanned_pdf(tmp_path / "exc.pdf")
        resp = client.post("/session/open", json={"path": str(path)})
        sid = resp.json()["session_id"]

        def _boom(doc, save_path):
            raise RuntimeError("incr boom")

        monkeypatch.setattr(backend.PdfService, "save_incremental", _boom)
        # _compress_in_place 成功（真实单页）
        resp = client.post(
            f"/session/{sid}/add_text_layer_batch",
            json={
                "pages": [
                    {
                        "page": 0,
                        "ocr_result": {
                            "preproc_angle": 0,
                            "text_blocks": [
                                {"text": "x", "score": 0.9, "bbox": [50, 50, 200, 100]}
                            ],
                        },
                    }
                ],
                "save": True,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["extra"]["saved"] is True

    def test_batch_compress_fail_reopen_fail(self, app_client, tmp_path, monkeypatch):
        """compress 失败 + 重开 doc 也失败（lines 666-667）。"""
        from vibeocr.backend.services import pdf_backend_process as backend

        backend._REGISTRY = None
        client, _ = app_client
        path = _create_scanned_pdf(tmp_path / "reopen_fail.pdf")
        resp = client.post("/session/open", json={"path": str(path)})
        sid = resp.json()["session_id"]

        monkeypatch.setattr(backend.PdfService, "save_incremental", lambda *a, **k: False)

        def _compress_fail(doc, save_path, clean=False):
            try:
                doc.close()
            except Exception:
                pass
            raise RuntimeError("compress boom")

        monkeypatch.setattr(backend.PdfService, "_compress_in_place", _compress_fail)
        # 重开也失败
        monkeypatch.setattr(fitz, "open", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("reopen boom")))
        resp = client.post(
            f"/session/{sid}/add_text_layer_batch",
            json={
                "pages": [
                    {
                        "page": 0,
                        "ocr_result": {
                            "preproc_angle": 0,
                            "text_blocks": [
                                {"text": "x", "score": 0.9, "bbox": [50, 50, 200, 100]}
                            ],
                        },
                    }
                ],
                "save": True,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["extra"]["saved"] is False


class TestDeleteLayersCancelAndPayloads:
    def test_delete_text_layers_cancelled_breaks(self, opened_session):
        """delete_text_layers 中 cancel_event 被 set → break（line 760）。"""
        client, backend, sid, _ = opened_session
        session = backend._get_registry().get(sid)
        session.cancel_event.set()
        resp = client.post(
            f"/session/{sid}/delete_text_layers", json={"pages": [0, 1]}
        )
        assert resp.status_code == 200
        lines = [line for line in resp.text.strip().split("\n") if line]
        # 立即 break → 只有 done 哨兵
        assert len(lines) == 1

    def test_delete_text_layers_residual_page_recorded(self, opened_session, monkeypatch):
        """有残留的页应被记录到 residual_pages（line 778）。"""
        client, backend, sid, _ = opened_session
        # 先加文字层
        client.post(
            f"/session/{sid}/add_text_layer",
            json={
                "page": 0,
                "ocr_result": {
                    "preproc_angle": 0,
                    "text_blocks": [
                        {"text": "hi", "score": 0.9, "bbox": [50, 50, 200, 100]}
                    ],
                },
            },
        )
        # 让 delete_text_layers 返回 residual=True
        monkeypatch.setattr(
            backend.PdfService,
            "delete_text_layers",
            lambda doc, pdf_doc, page: (5, 1, True),
        )
        resp = client.post(
            f"/session/{sid}/delete_text_layers", json={"pages": [0]}
        )
        assert resp.status_code == 200
        lines = [line for line in resp.text.strip().split("\n") if line]
        last = __import__("json").loads(lines[-1])
        assert last["page_payload"]["residual_pages"] == [0]

    def test_delete_text_layers_no_text_payload(self, app_client, tmp_path):
        """无文字页（扫描件）payload=(0,0,False)（line 769-770）。"""
        from vibeocr.backend.services import pdf_backend_process as backend

        backend._REGISTRY = None
        client, _ = app_client
        path = _create_scanned_pdf(tmp_path / "notext.pdf")
        resp = client.post("/session/open", json={"path": str(path)})
        sid = resp.json()["session_id"]
        resp = client.post(
            f"/session/{sid}/delete_text_layers", json={"pages": [0]}
        )
        assert resp.status_code == 200
        lines = [line for line in resp.text.strip().split("\n") if line]
        first = __import__("json").loads(lines[0])
        assert first["page_payload"] == [0, 0, False]


class TestSettingsFromDictNoFromDict:
    def test_settings_from_dict_via_delattr(self, monkeypatch):
        """真正删除 from_dict 让 hasattr 返回 False（lines 867-869）。"""
        from vibeocr.backend.models import pdf_ocr_options
        from vibeocr.backend.services.pdf_backend_process import _settings_from_dict

        cls = pdf_ocr_options.PdfGlobalSettings
        original_from_dict = cls.from_dict
        try:
            delattr(cls, "from_dict")
            s = _settings_from_dict({"compress_on_save": True})
            assert s.compress_on_save is True
        finally:
            cls.from_dict = original_from_dict


class TestMainEntry:
    def test_find_free_port(self):
        """_find_free_port 返回有效端口（lines 866-869）。"""
        from vibeocr.backend.services.pdf_backend_process import _find_free_port

        port = _find_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535 or port > 0

    def test_main_with_explicit_port(self, monkeypatch, capsys):
        """main() 用 --port 启动 uvicorn（lines 873-890）。"""
        import vibeocr.backend.services.pdf_backend_process as backend

        called = {}

        def _fake_run(app, host, port, log_level):
            called["host"] = host
            called["port"] = port
            called["log_level"] = log_level

        monkeypatch.setattr(backend.uvicorn, "run", _fake_run)
        monkeypatch.setattr(sys, "argv", ["prog", "--port", "9999", "--host", "0.0.0.0"])
        backend.main()
        captured = capsys.readouterr()
        assert "VIBEOCR_PDF_BACKEND_PORT=9999" in captured.out
        assert called["port"] == 9999
        assert called["host"] == "0.0.0.0"

    def test_main_with_auto_port(self, monkeypatch, capsys):
        """main() --port 0 → 自动选端口（lines 887-889）。"""
        import vibeocr.backend.services.pdf_backend_process as backend

        called = {}

        def _fake_run(app, host, port, log_level):
            called["port"] = port

        monkeypatch.setattr(backend.uvicorn, "run", _fake_run)
        monkeypatch.setattr(backend, "_find_free_port", lambda: 12345)
        monkeypatch.setattr(sys, "argv", ["prog"])
        backend.main()
        captured = capsys.readouterr()
        assert "VIBEOCR_PDF_BACKEND_PORT=12345" in captured.out
        assert called["port"] == 12345


class TestRemainingBranches:
    def test_detect_text_layers_page_out_of_range(self, opened_session, monkeypatch):
        """detect_text_layers route 页越界但无异常 → 跳过 model 更新（branch 477->480）。"""
        client, backend, sid, _ = opened_session
        # 让 PdfService.detect_text_layers 对越界页返回空（不抛）
        monkeypatch.setattr(backend.PdfService, "detect_text_layers", lambda doc, page: [])
        resp = client.post(
            f"/session/{sid}/detect_text_layers", json={"page": 999}
        )
        assert resp.status_code == 200
        # 仍返回空 layers
        assert resp.json()["text_layers"] == []

    def test_update_block_text_same_text_noop(self, opened_session):
        """update_block_text 新旧 text 相同 → 不改（branch 718->722）。"""
        client, _, sid, _ = opened_session
        # 先加层
        client.post(
            f"/session/{sid}/add_text_layer",
            json={
                "page": 0,
                "ocr_result": {
                    "preproc_angle": 0,
                    "text_blocks": [
                        {"text": "same", "score": 0.9, "bbox": [50, 50, 200, 100]}
                    ],
                },
            },
        )
        # 用相同 text 更新
        resp = client.post(
            f"/session/{sid}/update_block_text",
            json={"page": 0, "block_index": 0, "new_text": "same"},
        )
        assert resp.status_code == 200
