"""二维码/条形码生成服务"""

import io
import logging

from PIL import Image

logger = logging.getLogger(__name__)

QR_ERROR_CORRECTION_MAP = {
    "L": 1,
    "M": 0,
    "Q": 3,
    "H": 2,
}


class QrcodeService:
    """二维码和条形码生成服务"""

    def default_options(self) -> dict:
        return {
            "format": "qr",
            "size": 600,
            "error_correction": "M",
            "fg_color": "#000000",
            "bg_color": "#FFFFFF",
            "invert": False,
            "logo_path": None,
            "logo_ratio": 0.2,
            "label_text": "",
            "label_position": "bottom",
            "label_font_size": 12,
        }

    def generate(self, text: str, options: dict) -> Image.Image:
        fmt = options.get("format", "qr")
        if fmt == "qr":
            img = self._generate_qr(text, options)
        else:
            img = self._generate_barcode(text, options)
        return self._apply_extras(img, options)

    def _apply_extras(self, img: Image.Image, options: dict) -> Image.Image:
        """按 options 应用 invert / logo / 文字标签。

        Classic 前端一直在请求里携带这些选项；历史上端点没有应用它们，
        导致 UI 的 Logo、标签与反色开关完全无效。
        """
        if options.get("invert"):
            img = self.invert_colors(img)
        logo_path = options.get("logo_path")
        if logo_path:
            img = self.apply_logo(img, logo_path, ratio=options.get("logo_ratio", 0.2))
        label_text = options.get("label_text") or ""
        if label_text:
            img = self.apply_text_label(
                img,
                label_text,
                position=options.get("label_position", "bottom"),
                font_size=options.get("label_font_size", 12),
            )
        return img

    def _generate_qr(self, text: str, options: dict) -> Image.Image:
        import qrcode
        from PIL import ImageColor

        ec_level = QR_ERROR_CORRECTION_MAP.get(options.get("error_correction", "M"), 0)
        target_size = options.get("size", 300)

        # 先用 box_size=1 生成，确定模块数量
        qr_temp = qrcode.QRCode(
            version=None, error_correction=ec_level, box_size=1, border=4
        )
        qr_temp.add_data(text)
        qr_temp.make(fit=True)
        total_modules = qr_temp.modules_count + 8  # modules + 4 border * 2

        # box_size 取整确保每个模块像素对齐，actual_size <= target_size
        box_size = max(1, target_size // total_modules)
        actual_size = box_size * total_modules

        qr = qrcode.QRCode(
            version=None,
            error_correction=ec_level,
            box_size=box_size,
            border=4,
        )
        qr.add_data(text)
        qr.make(fit=True)

        fg_color = options.get("fg_color", "#000000")
        bg_color = options.get("bg_color", "#FFFFFF")
        img = qr.make_image(fill_color=fg_color, back_color=bg_color)
        if not isinstance(img, Image.Image):
            img = img.get_image()

        img = img.convert("RGB")

        # 居中粘贴到 target_size 画布，避免非整数缩放导致模块错位倾斜
        if actual_size < target_size:
            bg_rgb = ImageColor.getrgb(bg_color)
            canvas = Image.new("RGB", (target_size, target_size), bg_rgb)
            offset = (target_size - actual_size) // 2
            canvas.paste(img, (offset, offset))
            return canvas
        return img

    def _generate_barcode(self, text: str, options: dict) -> Image.Image:
        import barcode
        from barcode.writer import ImageWriter

        fmt = options.get("format", "code128").upper()
        fg_color = options.get("fg_color", "#000000")
        bg_color = options.get("bg_color", "#FFFFFF")

        writer = ImageWriter()
        writer.set_options(
            {
                "foreground": fg_color,
                "background": bg_color,
            }
        )

        barcode_class = barcode.get_barcode_class(fmt)
        bc = barcode_class(text, writer=writer)

        buffer = io.BytesIO()
        bc.write(buffer)
        buffer.seek(0)
        img: Image.Image = Image.open(buffer)
        img = img.convert("RGB")

        target_size = options.get("size", 300)
        w, h = img.size
        new_h = target_size
        new_w = int(w * new_h / h) if h > 0 else target_size
        return img.resize((new_w, new_h), Image.Resampling.NEAREST)

    def apply_logo(
        self, image: Image.Image, logo_path: str, ratio: float = 0.2
    ) -> Image.Image:
        logo = Image.open(logo_path).convert("RGBA")
        qr_w, qr_h = image.size
        logo_size = int(min(qr_w, qr_h) * ratio)
        logo = logo.resize((logo_size, logo_size), Image.Resampling.LANCZOS)

        image = image.convert("RGBA")
        pos_x = (qr_w - logo_size) // 2
        pos_y = (qr_h - logo_size) // 2

        image.paste(logo, (pos_x, pos_y), logo)
        return image.convert("RGB")

    @staticmethod
    def _load_font(size: int):
        import sys
        from pathlib import Path

        from PIL import ImageFont

        if sys.platform == "win32":
            candidates = [
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/segoeui.ttf",
                "C:/Windows/Fonts/arial.ttf",
            ]
        elif sys.platform == "darwin":
            candidates = [
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/Helvetica.ttc",
            ]
        else:
            candidates = [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]

        for path in candidates:
            if Path(path).is_file():
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
        return ImageFont.load_default(size=size)

    def apply_text_label(
        self,
        image: Image.Image,
        text: str,
        position: str = "bottom",
        font_size: int = 12,
    ) -> Image.Image:
        if position == "none" or not text:
            return image

        from PIL import ImageDraw

        # 字体和间距随图片尺寸等比缩放，基准 300px
        scale = min(image.size) / 300
        scaled_size = max(8, int(font_size * scale))
        padding = max(4, int(8 * scale))
        font = self._load_font(scaled_size)

        dummy = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_offset_y = -bbox[1]  # 修正 baseline 偏移

        img_w, img_h = image.size
        new_w = int(max(img_w, text_w + padding * 2))
        label_h = int(text_h + padding * 2)

        if position == "top":
            canvas = Image.new("RGB", (new_w, img_h + label_h), image.getpixel((0, 0)))
            draw = ImageDraw.Draw(canvas)
            text_x = (new_w - text_w) // 2
            draw.text(
                (text_x, padding + text_offset_y), text, fill=(0, 0, 0), font=font
            )
            canvas.paste(image, ((new_w - img_w) // 2, label_h))
        else:
            canvas = Image.new("RGB", (new_w, img_h + label_h), image.getpixel((0, 0)))
            canvas.paste(image, ((new_w - img_w) // 2, 0))
            draw = ImageDraw.Draw(canvas)
            text_x = (new_w - text_w) // 2
            draw.text(
                (text_x, img_h + padding + text_offset_y),
                text,
                fill=(0, 0, 0),
                font=font,
            )

        return canvas

    def invert_colors(self, image: Image.Image) -> Image.Image:
        from PIL import ImageOps

        return ImageOps.invert(image)

    def generate_svg(self, text: str, options: dict) -> str:
        import qrcode
        from qrcode.image.svg import SvgImage

        ec_level = QR_ERROR_CORRECTION_MAP.get(options.get("error_correction", "M"), 0)
        qr = qrcode.QRCode(
            version=None,
            error_correction=ec_level,
            box_size=10,
            border=4,
        )
        qr.add_data(text)
        qr.make(fit=True)

        fg_color = options.get("fg_color", "#000000")
        bg_color = options.get("bg_color", "#FFFFFF")

        img = qr.make_image(
            image_factory=SvgImage,
            fill_color=fg_color,
            back_color=bg_color,
        )
        buffer = io.BytesIO()
        img.save(buffer)
        return buffer.getvalue().decode("utf-8")
