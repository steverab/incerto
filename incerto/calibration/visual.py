import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

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
