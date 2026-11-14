"""Pooling primitives independent of checkpoint loading and device selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def masked_mean_pool(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Average valid token vectors; reject empty or malformed sequences."""
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("Expected hidden [batch, tokens, dim] and mask [batch, tokens].")
    mask = attention_mask.unsqueeze(-1).bool()
    counts = mask.sum(dim=1)
    if (counts == 0).any().item():
        raise ValueError("Cannot pool a sequence without valid tokens.")
    return hidden.masked_fill(~mask, 0.0).sum(dim=1) / counts
