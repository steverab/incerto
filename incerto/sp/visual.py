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
    """
    Plot the risk-coverage curve for selective prediction.

    Args:
        logits: Model logits.
        y: Ground truth labels.
        confidence: Confidence scores (if None, uses max softmax).
        ax: Matplotlib axes object.
        show_aurc: Whether to show AURC in title.

    Returns:
        Matplotlib axes object.
    """
    ax = ax or plt.gca()
    coverage, acc = accuracy_coverage_curve(logits, y, confidence)
    risk = 1.0 - acc
    ax.plot(coverage.cpu(), risk.cpu(), label="model")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (1 − accuracy)")

    if show_aurc:
        # Compute AURC from the risk-coverage curve using trapezoidal rule
        rc_auc = torch.trapz(risk, coverage).item()
        ax.set_title(f"Risk–Coverage curve (AURC = {rc_auc:.4f})")
    else:
        ax.set_title("Risk–Coverage curve")

    ax.grid(True, linestyle="--", linewidth=0.5)
    return ax
