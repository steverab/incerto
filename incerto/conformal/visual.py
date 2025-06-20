"""
incerto.conformal.visual
------------------------
Quick-look plots for conformal evaluation.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import torch
from typing import Sequence


def plot_coverage_vs_alpha(alphas: Sequence[float], coverages: Sequence[float]) -> None:
    plt.figure()
    plt.plot(alphas, coverages, marker="o")
    plt.plot(alphas, [1 - a for a in alphas], linestyle="--")
    plt.xlabel("Desired 1−α")
    plt.ylabel("Empirical coverage")
    plt.title("Coverage vs desired level")
    plt.grid(True)
    plt.show()


def plot_set_size_hist(sets: Sequence[torch.Tensor]) -> None:
    sizes = [len(S) for S in sets]
    plt.figure()
    plt.hist(sizes, bins=range(1, max(sizes) + 2), align="left", rwidth=0.8)
    plt.xlabel("Prediction set size")
    plt.ylabel("Frequency")
    plt.title("Distribution of prediction-set sizes")
    plt.grid(axis="y")
    plt.show()
