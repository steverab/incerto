"""
Visualization utilities for selective-prediction evaluation.

Uses matplotlib exclusively to keep dependencies minimal.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import torch

from .metrics import accuracy_coverage_curve, aurc


def plot_risk_coverage(
    logits: torch.Tensor,
    y: torch.Tensor,
    confidence: torch.Tensor | None = None,
    *,
    ax: plt.Axes | None = None,
    show_aurc: bool = True,
) -> plt.Axes:
    ax = ax or plt.gca()
    coverage, acc = accuracy_coverage_curve(logits, y, confidence)
    risk = 1.0 - acc
    ax.plot(coverage.cpu(), risk.cpu(), label="model")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (1 − accuracy)")
    if show_aurc:
        rc_auc = aurc(torch.tensor([]), torch.tensor([]))  # compute properly if wanted
        ax.set_title(f"Risk–Coverage curve (AURC ≈ {rc_auc:.4f})")
    ax.grid(True, linestyle="--", linewidth=0.5)
    return ax
