"""
Visualization utilities for LLM uncertainty.

Specialized plotting functions for language model uncertainty analysis.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_token_uncertainty(
    tokens: list[str],
    uncertainties: np.ndarray | torch.Tensor,
    ax=None,
    title: str = "Token-Level Uncertainty",
    cmap: str = "YlOrRd",
):
    """
    Plot uncertainty as a heatmap over token sequence.

    Args:
        tokens: List of tokens
        uncertainties: Uncertainty values per token
        ax: Matplotlib axes
        title: Plot title
        cmap: Colormap name
    """
    if isinstance(uncertainties, torch.Tensor):
        uncertainties = uncertainties.cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(10, len(tokens) * 0.5), 2))

    # Create heatmap
    uncertainties_2d = uncertainties.reshape(1, -1)
    im = ax.imshow(uncertainties_2d, cmap=cmap, aspect="auto")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticks([])

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Uncertainty")

    ax.set_title(title)
    return ax


def plot_confidence_vs_correctness(
    confidences: np.ndarray | torch.Tensor,
    correctness: np.ndarray | torch.Tensor,
    n_bins: int = 10,
    ax=None,
    title: str = "Confidence vs. Correctness",
):
    """
    Plot calibration diagram showing confidence vs. actual correctness.

    Args:
        confidences: Model confidence scores
        correctness: Binary correctness indicators
        n_bins: Number of bins
        ax: Matplotlib axes
        title: Plot title
    """
    if isinstance(confidences, torch.Tensor):
        confidences = confidences.cpu().numpy()
    if isinstance(correctness, torch.Tensor):
        correctness = correctness.cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Compute binned statistics
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        # Include lower boundary for first bin, upper boundary for last bin
        if i == 0:
            in_bin = (confidences >= lower) & (confidences < upper)
        elif i == n_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)

        if in_bin.sum() > 0:
            bin_centers.append((lower + upper) / 2)
            bin_accuracies.append(correctness[in_bin].mean())
            bin_counts.append(in_bin.sum())

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=2)

    # Plot actual calibration
    if bin_centers:
        ax.plot(bin_centers, bin_accuracies, "o-", label="Model", linewidth=2, markersize=8)

        # Add gap bars
        for center, accuracy, _count in zip(bin_centers, bin_accuracies, bin_counts, strict=False):
            gap = accuracy - center
            ax.bar(
                center,
                gap,
                width=1 / n_bins,
                bottom=center,
                alpha=0.3,
                color="red" if gap < 0 else "blue",
            )

    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return ax


def plot_generation_diversity(
    responses: list[str],
    max_display: int = 10,
    ax=None,
    title: str = "Generation Diversity",
):
    """
    Visualize diversity of generated responses.

    Args:
        responses: List of generated text responses
        max_display: Maximum number of responses to display
        ax: Matplotlib axes
        title: Plot title
    """
    from collections import Counter

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Count occurrences
    counts = Counter(responses)
    top_responses = counts.most_common(max_display)

    # Plot bar chart
    labels = [r[:50] + "..." if len(r) > 50 else r for r, _ in top_responses]
    values = [count for _, count in top_responses]

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Count")
    ax.set_title(f"{title}\n({len(counts)} unique out of {len(responses)} total)")
    ax.invert_yaxis()

    return ax


def plot_semantic_clusters(
    responses: list[str],
    clusters: list[int],
    ax=None,
    title: str = "Semantic Clusters",
):
    """
    Visualize semantic clustering of responses.

    Args:
        responses: List of generated responses
        clusters: Cluster assignments for each response
        ax: Matplotlib axes
        title: Plot title
    """
    from collections import Counter

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Count cluster sizes
    cluster_counts = Counter(clusters)
    n_clusters = len(cluster_counts)

    # Plot cluster sizes
    cluster_ids = sorted(cluster_counts.keys())
    sizes = [cluster_counts[cid] for cid in cluster_ids]

    ax.bar(cluster_ids, sizes, color="steelblue")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of Responses")
    ax.set_title(f"{title}\n({n_clusters} clusters from {len(responses)} responses)")
    ax.grid(True, alpha=0.3, axis="y")

    return ax


def plot_risk_coverage_llm(
    confidences: np.ndarray | torch.Tensor,
    correctness: np.ndarray | torch.Tensor,
    ax=None,
    title: str = "Risk-Coverage Curve",
):
    """
    Plot risk-coverage curve for selective prediction.

    Args:
        confidences: Confidence scores
        correctness: Binary correctness
        ax: Matplotlib axes
        title: Plot title
    """
    if isinstance(confidences, torch.Tensor):
        confidences = confidences.cpu().numpy()
    if isinstance(correctness, torch.Tensor):
        correctness = correctness.cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Sort by confidence (descending)
    sorted_indices = np.argsort(confidences)[::-1]
    sorted_correct = correctness[sorted_indices]

    # Compute cumulative risk and coverage
    cumsum = np.cumsum(sorted_correct)
    coverage_points = np.arange(1, len(sorted_correct) + 1)
    accuracy_curve = cumsum / coverage_points
    risk_curve = 1 - accuracy_curve
    coverage_normalized = coverage_points / len(sorted_correct)

    # Compute AURC
    aurc = np.trapezoid(risk_curve, coverage_normalized)

    # Plot
    ax.plot(coverage_normalized, risk_curve, linewidth=2, label=f"AURC={aurc:.4f}")
    ax.set_xlabel("Coverage", fontsize=12)
    ax.set_ylabel("Risk (Error Rate)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax


def plot_uncertainty_distribution(
    uncertainties: np.ndarray | torch.Tensor,
    correctness: np.ndarray | torch.Tensor | None = None,
    bins: int = 50,
    ax=None,
    title: str = "Uncertainty Distribution",
):
    """
    Plot distribution of uncertainty scores.

    Args:
        uncertainties: Uncertainty values
        correctness: Optional binary correctness to separate distributions
        bins: Number of histogram bins
        ax: Matplotlib axes
        title: Plot title
    """
    if isinstance(uncertainties, torch.Tensor):
        uncertainties = uncertainties.cpu().numpy()
    if isinstance(correctness, torch.Tensor):
        correctness = correctness.cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if correctness is not None:
        # Separate by correctness
        correct_unc = uncertainties[correctness == 1]
        incorrect_unc = uncertainties[correctness == 0]

        ax.hist(correct_unc, bins=bins, alpha=0.6, label="Correct", color="green")
        ax.hist(incorrect_unc, bins=bins, alpha=0.6, label="Incorrect", color="red")
        ax.legend()
    else:
        ax.hist(uncertainties, bins=bins, alpha=0.7, color="steelblue")

    ax.set_xlabel("Uncertainty", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    return ax


def plot_length_vs_confidence(
    lengths: list[int],
    confidences: list[float],
    ax=None,
    title: str = "Sequence Length vs. Confidence",
):
    """
    Plot relationship between sequence length and confidence.

    Args:
        lengths: Sequence lengths
        confidences: Confidence scores
        ax: Matplotlib axes
        title: Plot title
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(lengths, confidences, alpha=0.5, s=20)

    # Add trend line
    z = np.polyfit(lengths, confidences, 1)
    p = np.poly1d(z)
    ax.plot(
        lengths,
        p(lengths),
        "r--",
        alpha=0.8,
        linewidth=2,
        label=f"Trend: y={z[0]:.4f}x+{z[1]:.4f}",
    )

    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Confidence", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax
