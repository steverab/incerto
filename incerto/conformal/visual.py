"""
incerto.conformal.visual
------------------------
Quick-look plots for conformal evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_coverage_vs_alpha(
    alphas: Sequence[float],
    coverages: Sequence[float],
    show: bool = True,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot empirical coverage versus the desired miscoverage rate ``alpha``.

    Args:
        alphas: Desired miscoverage rates.
        coverages: Empirical coverages observed for each ``alpha``.
        show: If True, call ``plt.show()``.
        ax: Optional matplotlib Axes to draw into. If None, a new figure is
            created.

    Returns:
        ``(fig, ax)`` tuple.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(alphas, coverages, marker="o", label="Empirical")
    ax.plot(alphas, [1 - a for a in alphas], linestyle="--", label="Target 1-α")
    ax.set_xlabel("α (miscoverage rate)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Coverage vs desired level")
    ax.grid(True)
    ax.legend()
    if show:
        plt.show()
    return fig, ax


def plot_set_size_hist(
    sets: Sequence[torch.Tensor],
    show: bool = True,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a histogram of prediction-set sizes.

    Args:
        sets: Sequence of prediction sets (one tensor of class indices per
            example).
        show: If True, call ``plt.show()``.
        ax: Optional matplotlib Axes to draw into.

    Returns:
        ``(fig, ax)`` tuple.
    """
    sizes = [len(S) for S in sets]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.hist(sizes, bins=range(1, max(sizes) + 2), align="left", rwidth=0.8)
    ax.set_xlabel("Prediction set size")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of prediction-set sizes")
    ax.grid(axis="y")
    if show:
        plt.show()
    return fig, ax
