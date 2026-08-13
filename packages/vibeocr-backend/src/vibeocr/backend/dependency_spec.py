"""Runtime dependency detection specification shared across backend layers."""

from __future__ import annotations

# paddlex[ocr] extra leaf packages required by TableRecognitionPipelineV2.
# Keys are import names; values are distribution names used by pip and status output.
OCR_CHECK_LEAF_MODULES: dict[str, str] = {
    "bs4": "beautifulsoup4",
    "einops": "einops",
    "ftfy": "ftfy",
    "latex2mathml": "latex2mathml",
    "premailer": "premailer",
    "regex": "regex",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "sentencepiece": "sentencepiece",
    "tiktoken": "tiktoken",
    "tokenizers": "tokenizers",
}


__all__ = ["OCR_CHECK_LEAF_MODULES"]
