import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

from .metrics import _find_sigma_star
from .utils import get_bin_stats


def plot_reliability_diagram(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
    ax=None,
    title: str = "Reliability Diagram",
):
    """
    Plot a reliability diagram comparing confidence vs accuracy.
    """
    probs = F.softmax(logits, dim=1).cpu().detach().numpy()
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels.cpu().numpy()).astype(float)

    bin_conf, bin_acc, weight = get_bin_stats(confidences, accuracies, n_bins)

    if ax is None:
        fig, ax = plt.subplots()

    # Perfect calibration
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    # Empirical
    ax.plot(bin_conf, bin_acc, marker="o", label="Empirical")
    # Gap bars
    centers = (np.arange(n_bins) + 0.5) / n_bins
    ax.bar(
        centers,
        bin_acc - bin_conf,
        width=1.0 / n_bins,
        alpha=0.3,
        edgecolor="black",
        label="Gap",
    )

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    return ax


def plot_confidence_histogram(
    logits: torch.Tensor,
    n_bins: int = 10,
    ax=None,
    title: str = "Confidence Histogram",
):
    """
    Plot a histogram of model confidences (max softmax probability).
    """
    probs = F.softmax(logits, dim=1).cpu().detach().numpy()
    confidences = np.max(probs, axis=1)

    if ax is None:
        fig, ax = plt.subplots()

    ax.hist(confidences, bins=n_bins, range=(0, 1), edgecolor="black")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title(title)

    return ax


def plot_calibration_curve(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
    ax=None,
    title: str = "Calibration Curve",
):
    """
    Plot calibration curve: accuracy vs. confidence bin centers.
    """
    probs = F.softmax(logits, dim=1).cpu().detach().numpy()
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels.cpu().numpy()).astype(float)

    bin_conf, bin_acc, _ = get_bin_stats(confidences, accuracies, n_bins)
    centers = (np.arange(n_bins) + 0.5) / n_bins

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(centers, bin_acc, marker="o")
    ax.set_xlabel("Confidence Bin Center")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1)

    return ax


def _smooth_calibration_curve(
    confidences: np.ndarray, accuracies: np.ndarray, sigma: float, n_grid: int = 1000
):
    """
    Compute the Nadaraya-Watson smoothed calibration curve.

    Returns:
        Tuple of (grid, r_sigma, density) where r_sigma is the smoothed
        P(Y=1 | f=t) and density is the kernel density of predictions.
    """
    n = len(confidences)
    dx = 1.0 / n_grid
    bin_indices = np.clip((confidences / dx).astype(int), 0, n_grid - 1)

    # Histogram of y values (numerator of Nadaraya-Watson)
    y_hist = np.zeros(n_grid)
    np.add.at(y_hist, bin_indices, accuracies / n)

    # Histogram of counts (denominator / density)
    count_hist = np.zeros(n_grid)
    np.add.at(count_hist, bin_indices, 1.0 / n)

    sigma_pixels = sigma / dx
    smoothed_y = gaussian_filter1d(y_hist, sigma_pixels, mode="reflect")
    smoothed_density = gaussian_filter1d(count_hist, sigma_pixels, mode="reflect")

    grid = np.linspace(dx / 2, 1 - dx / 2, n_grid)
    mask = smoothed_density > 1e-10
    r = np.full_like(grid, np.nan)
    r[mask] = smoothed_y[mask] / smoothed_density[mask]

    return grid, r, smoothed_density


def plot_smooth_reliability_diagram(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ax=None,
    title: str = "Smooth Reliability Diagram",
):
    """
    Plot a smooth reliability diagram using kernel smoothing.

    Uses the SmoothECE framework: Nadaraya-Watson kernel regression with
    automatic bandwidth selection via the fixed-point condition.

    Reference:
        Blasiok & Nakkiran, "Smooth ECE: Principled Reliability Diagrams
        via Kernel Smoothing" (ICLR 2024)

    Args:
        logits: Model logits (N, C)
        labels: True labels (N,)
        ax: Optional matplotlib axes
        title: Plot title

    Returns:
        matplotlib Axes
    """
    probs = F.softmax(logits, dim=1).cpu().detach().numpy()
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels.cpu().numpy()).astype(float)

    # Find optimal bandwidth and compute smooth curve
    sigma_star = _find_sigma_star(confidences, accuracies)
    grid, r_sigma, density = _smooth_calibration_curve(confidences, accuracies, sigma_star)

    if ax is None:
        fig, ax = plt.subplots()

    # Perfect calibration
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")

    # Smooth calibration curve with linewidth proportional to density
    # Normalize density for linewidth scaling
    max_density = np.max(density)
    if max_density > 0:
        norm_density = density / max_density
    else:
        norm_density = np.ones_like(density)

    # Draw the curve as line segments with varying width
    valid = ~np.isnan(r_sigma)
    if np.any(valid):
        # Draw thin segments with width proportional to density
        for i in range(len(grid) - 1):
            if valid[i] and valid[i + 1]:
                lw = 1.0 + 3.0 * norm_density[i]
                ax.plot(
                    grid[i : i + 2],
                    r_sigma[i : i + 2],
                    color="C3",
                    linewidth=lw,
                    solid_capstyle="round",
                )
        # Invisible line for legend
        ax.plot([], [], color="C3", linewidth=2, label=f"smECE = {sigma_star:.4f}")

    # Tick marks showing raw data density
    ax.scatter(
        confidences,
        -0.02 * np.ones_like(confidences),
        marker="|",
        color="gray",
        alpha=0.3,
        s=10,
        zorder=1,
    )

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()

    return ax
