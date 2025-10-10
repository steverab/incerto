"""
incerto.shift_detection.metrics
===============================

Pure functions that measure how far two sample sets differ.
Designed so that you can mix and match with your own detectors.
"""

from __future__ import annotations
import torch


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """Szekely–Rizzo energy distance, O(n²) naive implementation."""

    def pdist(t):  # pairwise ℓ2
        return torch.cdist(t, t, p=2).mean()

    return (2 * torch.cdist(x, y, p=2).mean() - pdist(x) - pdist(y)).item()


def total_variation(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> float:
    """Total variation between *discrete* distributions p and q."""
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    return 0.5 * torch.abs(p - q).sum().item()


def population_stability_index(p_hist, q_hist, eps: float = 1e-9) -> float:
    """Classic tabular PSI used in credit scoring."""
    p, q = p_hist + eps, q_hist + eps
    return ((q - p) * torch.log(q / p)).sum().item()
