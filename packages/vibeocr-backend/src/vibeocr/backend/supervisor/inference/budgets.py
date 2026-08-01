"""BudgetPlanner: transport-safety and compute-microbatch budgets.

Plan §4 Phase 3 separates two concerns:

* **transport safety**: file count, encoded bytes, decoded pixels / estimated
  pages — caps how much one request/transport-batch can carry.
* **compute microbatch**: derived from adapter capability, device and VRAM —
  caps how many items go into one GPU ``predict`` call.

Oversized single items run alone and must never be silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class InputItem:
    """A pre-staged input described for budgeting purposes.

    ``data`` carries the raw payload bytes (image bytes for Paddle, file
    bytes for MinerU). ``display_name`` is the original (untrusted) name for
    logging/UI; the server never uses it as a path. Both are optional so the
    same dataclass can describe already-decoded inputs in pure budget tests.
    """

    item_id: str
    encoded_bytes: int
    decoded_pixels: int
    estimated_pages: int = 1
    display_name: str = ""
    data: bytes = b""


@dataclass(frozen=True, slots=True)
class AdapterCapability:
    """What an adapter promises about real batching."""

    name: str
    real_batch: bool
    """True if one ``recognize_many`` call maps to one native multi-input call."""
    max_compute_batch: int = 8
    """Upper bound on a compute microbatch (VRAM/sampler-derived)."""
    per_item_vram_mb: int = 0
    """Estimated VRAM per item in the compute batch (0 if unknown)."""


@dataclass(frozen=True, slots=True)
class TransportBatch:
    items: tuple[InputItem, ...]
    oversized: bool = False


@dataclass(frozen=True, slots=True)
class ComputeBatch:
    items: tuple[InputItem, ...]


@dataclass
class BudgetPlanner:
    """Plan transport + compute batches for a list of inputs."""

    max_file_count: int = 16
    max_encoded_bytes: int = 64 * 1024 * 1024
    max_decoded_pixels: int = 64_000_000
    max_pages: int = 64
    device_vram_mb: int = 0

    def __post_init__(self) -> None:
        positive = {
            "max_file_count": self.max_file_count,
            "max_encoded_bytes": self.max_encoded_bytes,
            "max_decoded_pixels": self.max_decoded_pixels,
            "max_pages": self.max_pages,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError("budget limits must be positive: " + ", ".join(invalid))
        if self.device_vram_mb < 0:
            raise ValueError("device_vram_mb must be >= 0")

    # ------------------------------------------------------------------
    # Transport batches
    # ------------------------------------------------------------------

    def plan_transport(self, items: Sequence[InputItem]) -> list[TransportBatch]:
        """Group items into transport batches respecting all safety caps.

        An oversized single item forms its own batch with ``oversized=True``
        so the executor can route it to a one-element compute batch.
        """
        batches: list[TransportBatch] = []
        current: list[InputItem] = []
        cur_bytes = 0
        cur_pixels = 0
        cur_pages = 0
        for item in items:
            oversized = (
                item.encoded_bytes > self.max_encoded_bytes
                or item.decoded_pixels > self.max_decoded_pixels
                or item.estimated_pages > self.max_pages
            )
            if oversized:
                if current:
                    batches.append(TransportBatch(tuple(current)))
                    current = []
                    cur_bytes = cur_pixels = cur_pages = 0
                batches.append(TransportBatch((item,), oversized=True))
                continue
            would_count = len(current) + 1
            would_bytes = cur_bytes + item.encoded_bytes
            would_pixels = cur_pixels + item.decoded_pixels
            would_pages = cur_pages + item.estimated_pages
            if (
                would_count > self.max_file_count
                or would_bytes > self.max_encoded_bytes
                or would_pixels > self.max_decoded_pixels
                or would_pages > self.max_pages
            ):
                batches.append(TransportBatch(tuple(current)))
                current = [item]
                cur_bytes = item.encoded_bytes
                cur_pixels = item.decoded_pixels
                cur_pages = item.estimated_pages
            else:
                current.append(item)
                cur_bytes = would_bytes
                cur_pixels = would_pixels
                cur_pages = would_pages
        if current:
            batches.append(TransportBatch(tuple(current)))
        return batches

    # ------------------------------------------------------------------
    # Compute microbatches
    # ------------------------------------------------------------------

    def plan_compute(
        self, items: Sequence[InputItem], capability: AdapterCapability
    ) -> list[ComputeBatch]:
        """Split a transport batch into compute microbatches.

        If the adapter does not support real batching, each item is its own
        microbatch (we never lie about batch size).
        """
        if not capability.real_batch:
            return [ComputeBatch((item,)) for item in items]
        cap = max(1, capability.max_compute_batch)
        # Further constrain by VRAM if known.
        if capability.per_item_vram_mb and self.device_vram_mb:
            vram_cap = max(
                1, self.device_vram_mb // max(1, capability.per_item_vram_mb)
            )
            cap = min(cap, vram_cap)
        return [
            ComputeBatch(tuple(items[i : i + cap])) for i in range(0, len(items), cap)
        ]


__all__ = [
    "AdapterCapability",
    "BudgetPlanner",
    "ComputeBatch",
    "InputItem",
    "TransportBatch",
]
