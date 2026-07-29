"""PDF process adapter (Phase 6).

The PdfProcessAdapter will own the existing model-free PyMuPDF child process
(``services/pdf_backend_process.py``) and expose session/render/mutate/save
operations to the supervisor. It is left as a stub here; Phase 6 fills it in.
"""

from __future__ import annotations
