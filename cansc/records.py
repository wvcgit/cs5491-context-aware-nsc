"""Archive record type shared by the adapter and the evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEVELOPMENT = "development"
FINAL = "final"


@dataclass(frozen=True)
class ArchRecord:
    """One archived Transformer architecture, as needed by the evaluation.

    Attributes:
        arch_id: identifier in the course archive.
        split: official split, ``"development"`` or ``"final"``.
        width: hidden width d (must match ArchTerms.width of the same id).
        depth: number of Transformer blocks.
        params: reported parameter count (used for the size-control pairs).
        perplexity: recorded perplexity (lower is better).
        config: the raw archive entry, handed back to the course scorer.
    """

    arch_id: str
    split: str
    width: float
    depth: int
    params: float
    perplexity: float
    config: Any = field(default=None, compare=False, hash=False, repr=False)
