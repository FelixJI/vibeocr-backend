"""UI-free VibeOCR Backend runtime."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("vibeocr-backend")
except PackageNotFoundError:
    __version__ = "0.11.2"  # x-release-please-version

__all__ = ["__version__"]
