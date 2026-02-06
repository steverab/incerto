"""
Visualization utilities for selective-prediction evaluation.

Uses matplotlib exclusively to keep dependencies minimal.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import torch

from .metrics import accuracy_coverage_curve


def _ideal_risk_curve(
    logits: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (coverage, risk) for an oracle that rejects all errors first."""
    n = len(y)
    preds = logits.argmax(dim=-1)
    n_correct = (preds == y).sum().item()
    k = torch.arange(1, n + 1, dtype=torch.float32)
    # Oracle includes all correct samples first, then errors
    errors = torch.clamp(k - n_correct, min=0)
    return k / n, errors / k


def _ideal_accuracy_curve(
    logits: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (coverage, accuracy) for an oracle that rejects all errors first."""
    n = len(y)
    preds = logits.argmax(dim=-1)
    n_correct = (preds == y).sum().item()
    k = torch.arange(1, n + 1, dtype=torch.float32)
    correct = torch.minimum(k, torch.tensor(float(n_correct)))
    return k / n, correct / k


def plot_risk_coverage(
    logits: torch.Tensor,
    y: torch.Tensor,
    confidence: torch.Tensor | None = None,
    *,
    ax: plt.Axes | None = None,
    show_aurc: bool = True,
    show_bounds: bool = True,
) -> plt.Axes:
    """
    Plot the risk-coverage curve for selective prediction.

    Args:
        logits: Model logits.
        y: Ground truth labels.
        confidence: Confidence scores (if None, uses max softmax).
        ax: Matplotlib axes object.
        show_aurc: Whether to show AURC in title.
        show_bounds: Whether to show random and ideal (oracle) bounds.

    Returns:
        Matplotlib axes object.
    """
    ax = ax or plt.gca()
    coverage, acc = accuracy_coverage_curve(logits, y, confidence)
    risk = 1.0 - acc
    ax.plot(coverage.cpu(), risk.cpu(), label="Model")

    if show_bounds:
        # Random baseline: constant risk equal to overall error rate
        overall_risk = 1.0 - (logits.argmax(-1) == y).float().mean().item()
        ax.axhline(
            overall_risk, color="gray", linestyle=":", linewidth=1.5, label="Random"
        )
        # Ideal (oracle): reject all errors first
        ideal_cov, ideal_risk = _ideal_risk_curve(logits, y)
        ax.plot(
            ideal_cov.numpy(),
            ideal_risk.numpy(),
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Ideal",
        )

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (1 − accuracy)")

    if show_aurc:
        rc_auc = torch.trapezoid(risk, coverage).item()
        ax.set_title(f"Risk–Coverage curve (AURC = {rc_auc:.4f})")
    else:
        ax.set_title("Risk–Coverage curve")

    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5)
    return ax


def plot_accuracy_coverage(
    logits: torch.Tensor,
    y: torch.Tensor,
    confidence: torch.Tensor | None = None,
    *,
    ax: plt.Axes | None = None,
    show_bounds: bool = True,
) -> plt.Axes:
    """
    Plot the accuracy-coverage curve for selective prediction.

    Args:
        logits: Model logits.
        y: Ground truth labels.
        confidence: Confidence scores (if None, uses max softmax).
        ax: Matplotlib axes object.
        show_bounds: Whether to show random and ideal (oracle) bounds.

    Returns:
        Matplotlib axes object.
    """
    ax = ax or plt.gca()
    coverage, acc = accuracy_coverage_curve(logits, y, confidence)
    ax.plot(coverage.cpu(), acc.cpu(), label="Model")

    if show_bounds:
        # Random baseline: constant accuracy equal to overall accuracy
        overall_acc = (logits.argmax(-1) == y).float().mean().item()
        ax.axhline(
            overall_acc, color="gray", linestyle=":", linewidth=1.5, label="Random"
        )
        # Ideal (oracle): include all correct samples first
        ideal_cov, ideal_acc = _ideal_accuracy_curve(logits, y)
        ax.plot(
            ideal_cov.numpy(),
            ideal_acc.numpy(),
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Ideal",
        )

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy–Coverage curve")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5)
    return ax
