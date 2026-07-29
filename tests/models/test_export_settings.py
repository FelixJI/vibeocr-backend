"""ExportSettings 数据类测试。"""

from vibeocr.backend.models.export_settings import ExportSettings


class TestExportSettings:
    def test_defaults(self):
        s = ExportSettings()
        assert s.format == "markdown"
        assert s.location_mode == "same_as_source"
        assert s.custom_directory == ""
        assert s.last_custom_directory == ""

    def test_get_extension_known_formats(self):
        assert ExportSettings(format="markdown").get_extension() == ".md"
        assert ExportSettings(format="html").get_extension() == ".html"
        assert ExportSettings(format="docx").get_extension() == ".docx"
        assert ExportSettings(format="xlsx").get_extension() == ".xlsx"
        assert ExportSettings(format="txt").get_extension() == ".txt"

    def test_get_extension_unknown_format_falls_back_to_txt(self):
        assert ExportSettings(format="pdf").get_extension() == ".txt"
        assert ExportSettings(format="").get_extension() == ".txt"

    def test_get_label_known_formats(self):
        assert "Markdown" in ExportSettings(format="markdown").get_label()
        assert "Word" in ExportSettings(format="docx").get_label()
        assert "Excel" in ExportSettings(format="xlsx").get_label()

    def test_get_label_unknown_format(self):
        assert ExportSettings(format="pdf").get_label() == "未知格式"
