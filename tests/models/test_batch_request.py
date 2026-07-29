"""测试批量请求数据模型"""

import time

from vibeocr.backend.models.batch_request import (
    BatchProgress,
    BatchRequest,
    BatchRequestStatus,
    PreprocessOptions,
)


class TestBatchRequest:
    """BatchRequest 测试"""

    def test_create_request(self):
        """测试创建请求"""
        request = BatchRequest(
            file_path="/path/to/image.png",
            file_name="image.png",
            image_data=b"fake_image_data",
            options={"lang": "ch"},
        )

        assert request.status == BatchRequestStatus.PENDING
        assert request.request_id != ""
        assert request.created_at > 0
        assert request.result is None

    def test_mark_processing(self):
        """测试标记处理中"""
        request = BatchRequest()
        request.mark_processing()

        assert request.status == BatchRequestStatus.PROCESSING
        assert request.started_at is not None

    def test_mark_completed(self):
        """测试标记完成"""
        request = BatchRequest()
        request.mark_processing()
        request.mark_completed(result={"text": "Hello"})

        assert request.status == BatchRequestStatus.COMPLETED
        assert request.result == {"text": "Hello"}
        assert request.completed_at is not None

    def test_mark_failed(self):
        """测试标记失败"""
        request = BatchRequest()
        request.mark_processing()
        request.mark_failed("OCR failed")

        assert request.status == BatchRequestStatus.FAILED
        assert request.error_message == "OCR failed"

    def test_mark_cancelled(self):
        """测试标记取消"""
        request = BatchRequest()
        request.mark_cancelled()

        assert request.status == BatchRequestStatus.CANCELLED

    def test_elapsed_time(self):
        """测试耗时计算"""
        request = BatchRequest()

        # 未开始时为 None
        assert request.elapsed_time is None

        # 开始处理后
        request.mark_processing()
        time.sleep(0.1)
        elapsed = request.elapsed_time
        assert elapsed is not None and elapsed >= 0.1

        # 完成后固定
        request.mark_completed(result={})
        elapsed = request.elapsed_time
        time.sleep(0.1)
        assert request.elapsed_time == elapsed

    def test_is_finished(self):
        """测试是否结束"""
        request = BatchRequest()
        assert not request.is_finished

        request.mark_completed(result={})
        assert request.is_finished

        request2 = BatchRequest()
        request2.mark_failed("error")
        assert request2.is_finished


class TestPreprocessOptions:
    """PreprocessOptions 测试"""

    def test_default_options(self):
        """测试默认选项"""
        options = PreprocessOptions()

        assert options.use_doc_orientation_classify is True
        assert options.use_doc_unwarping is False
        assert options.use_textline_orientation is False

    def test_to_dict(self):
        """测试转换为字典"""
        options = PreprocessOptions(
            use_doc_orientation_classify=False,
            use_doc_unwarping=True,
            use_textline_orientation=True,
        )

        result = options.to_dict()

        assert result["use_doc_orientation_classify"] is False
        assert result["use_doc_unwarping"] is True
        assert result["use_textline_orientation"] is True

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": True,
        }

        options = PreprocessOptions.from_dict(data)

        assert options.use_doc_orientation_classify is False
        assert options.use_doc_unwarping is False
        assert options.use_textline_orientation is True


class TestBatchProgress:
    """BatchProgress 测试"""

    def test_progress_percent(self):
        """测试进度百分比"""
        progress = BatchProgress(total=10, completed=5)

        assert progress.progress_percent == 50.0

    def test_progress_percent_zero_total(self):
        """测试总数为零时的进度"""
        progress = BatchProgress(total=0, completed=0)

        assert progress.progress_percent == 0.0

    def test_remaining(self):
        """测试剩余数量"""
        progress = BatchProgress(total=10, completed=5, failed=2)

        assert progress.remaining == 3
