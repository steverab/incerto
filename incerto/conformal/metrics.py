"""
incerto.conformal.metrics
-------------------------
Evaluation utilities for conformal predictors.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def empirical_coverage(y: torch.Tensor, sets: Sequence[torch.Tensor]) -> float:
    """Fraction of test examples where y_i ∈ Ŝ_i."""
    hits = [(yi in Si) for yi, Si in zip(y.cpu(), sets, strict=False)]
    return float(torch.tensor(hits).float().mean())


def average_set_size(sets: Sequence[torch.Tensor]) -> float:
    return float(torch.tensor([len(Si) for Si in sets]).float().mean())


def conditional_coverage(
    y: torch.Tensor,
    sets: Sequence[torch.Tensor],
    groups: torch.Tensor,
) -> dict[int, float]:
    """
    Coverage conditioned on *groups* (e.g., class labels).
    Returns a mapping group → coverage.
    """
    cover = {}
    for g in torch.unique(groups):
        mask = groups == g
        cover[int(g)] = empirical_coverage(
            y[mask], [s for m, s in zip(mask, sets, strict=False) if m]
        )
    return cover
