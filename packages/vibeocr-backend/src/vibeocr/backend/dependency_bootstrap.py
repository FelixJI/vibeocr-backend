"""VibeOCR 联网后端依赖安装命令。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vibeocr.backend.env_manager import detect_gpu, install_backend_dependencies


def _resolve_profile(requested: str) -> str:
    if requested != "auto":
        return requested
    has_gpu, _cuda_tag = detect_gpu()
    return "gpu-cu126" if has_gpu else "cpu"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="为 VibeOCR 安装 CPU 或 CUDA 12.6 后端重依赖",
    )
    parser.add_argument(
        "--profile",
        choices=("auto", "cpu", "gpu-cu126"),
        default="auto",
    )
    parser.add_argument(
        "--network",
        choices=("international", "domestic"),
        default="international",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="目标 Python；默认安装到当前环境",
    )
    args = parser.parse_args(argv)
    profile = _resolve_profile(args.profile)

    def report(stage: str, message: str) -> None:
        print(f"[{stage}] {message}", flush=True)

    success, message = install_backend_dependencies(
        args.python,
        profile=profile,
        network_type=args.network,
        progress_callback=report,
    )
    print(message, file=sys.stdout if success else sys.stderr)
    return 0 if success else 1


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
