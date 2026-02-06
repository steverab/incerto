"""
incerto.shift_detection.visual
==============================

Tiny helpers for fast diagnostics; rely on matplotlib but never seaborn.
"""

from typing import Iterable
import matplotlib.pyplot as plt
import torch


def plot_feature_histograms(
    ref: torch.Tensor,
    test: torch.Tensor,
    feature_ids: Iterable[int] | None = None,
    bins: int = 30,
    show: bool = True,
) -> plt.Figure:
    """
    Overlay 1-D histograms for a handful of features.

    Args:
        ref: Reference distribution samples of shape (n, d)
        test: Test distribution samples of shape (m, d)
        feature_ids: Which feature indices to plot (default: first 5)
        bins: Number of histogram bins
        show: Whether to call plt.show() (default: True)

    Returns:
        The matplotlib Figure object for further customization
    """
    feature_ids = (
        list(feature_ids) if feature_ids is not None else range(min(5, ref.shape[1]))
    )
    n = len(feature_ids)
    fig, axes = plt.subplots(nrows=n, figsize=(6, 2 * n))
    axes = axes if n > 1 else [axes]
    for ax, idx in zip(axes, feature_ids):
        ax.hist(
            ref[:, idx].cpu(), bins=bins, alpha=0.5, label="reference", density=True
        )
        ax.hist(test[:, idx].cpu(), bins=bins, alpha=0.5, label="test", density=True)
        ax.set_title(f"Feature {idx}")
    axes[0].legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_embedding_space(
    ref_emb: torch.Tensor,
    test_emb: torch.Tensor,
    method: str = "tsne",
    show: bool = True,
) -> plt.Figure:
    """
    Visualize reference and test embeddings in 2D space.

    Args:
        ref_emb: Reference embeddings of shape (n, d)
        test_emb: Test embeddings of shape (m, d)
        method: Dimensionality reduction method ('tsne' or 'pca')
        show: Whether to call plt.show() (default: True)

    Returns:
        The matplotlib Figure object for further customization

    Raises:
        ValueError: If method is not 'tsne' or 'pca'
    """
    if method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=2, perplexity=30)
    elif method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Unknown method '{method}'. Supported: 'tsne', 'pca'")

    z = reducer.fit_transform(torch.cat([ref_emb, test_emb]).cpu().numpy())
    n_ref = len(ref_emb)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(z[:n_ref, 0], z[:n_ref, 1], s=5, alpha=0.5, label="reference")
    ax.scatter(z[n_ref:, 0], z[n_ref:, 1], s=5, alpha=0.5, label="test")
    ax.set_title(f"Embedding space ({method.upper()})")
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_confidence_distributions(
    ref_confidences: torch.Tensor,
    test_confidences: torch.Tensor,
    bins: int = 50,
    show: bool = True,
) -> plt.Figure:
    """
    Compare model confidence distributions between reference and test data.

    Useful for detecting calibration degradation under distribution shift.

    Args:
        ref_confidences: Confidence scores for reference data, shape (n,)
        test_confidences: Confidence scores for test data, shape (m,)
        bins: Number of histogram bins
        show: Whether to call plt.show() (default: True)

    Returns:
        The matplotlib Figure object for further customization
    """
    ref_np = (
        ref_confidences.cpu().numpy()
        if isinstance(ref_confidences, torch.Tensor)
        else ref_confidences
    )
    test_np = (
        test_confidences.cpu().numpy()
        if isinstance(test_confidences, torch.Tensor)
        else test_confidences
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        ref_np,
        bins=bins,
        alpha=0.5,
        label="reference",
        density=True,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.hist(
        test_np,
        bins=bins,
        alpha=0.5,
        label="test",
        density=True,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axvline(
        ref_np.mean(),
        color="blue",
        linestyle="--",
        linewidth=1.5,
        label=f"ref mean: {ref_np.mean():.3f}",
    )
    ax.axvline(
        test_np.mean(),
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"test mean: {test_np.mean():.3f}",
    )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title("Confidence Distribution: Reference vs Test")
    ax.legend()
    ax.set_xlim(0, 1)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_shift_severity(
    severity_values: Iterable[float],
    shift_scores: Iterable[float],
    severity_label: str = "Shift Severity",
    score_label: str = "Shift Score",
    warning_threshold: float | None = None,
    critical_threshold: float | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot shift scores against a severity measure (e.g., rotation degrees, time).

    Args:
        severity_values: X-axis values (e.g., [0, 10, 20, 30, 45])
        shift_scores: Corresponding shift scores
        severity_label: Label for x-axis
        score_label: Label for y-axis
        warning_threshold: Optional threshold line for warnings
        critical_threshold: Optional threshold line for critical alerts
        show: Whether to call plt.show() (default: True)

    Returns:
        The matplotlib Figure object for further customization
    """
    severity_values = list(severity_values)
    shift_scores = list(shift_scores)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        severity_values,
        shift_scores,
        marker="o",
        linewidth=2,
        markersize=8,
        color="steelblue",
    )

    if warning_threshold is not None:
        ax.axhline(
            y=warning_threshold,
            color="orange",
            linestyle="--",
            linewidth=2,
            label="Warning",
        )
    if critical_threshold is not None:
        ax.axhline(
            y=critical_threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Critical",
        )

    ax.set_xlabel(severity_label)
    ax.set_ylabel(score_label)
    ax.set_title(f"{score_label} vs {severity_label}")
    if warning_threshold is not None or critical_threshold is not None:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_ks_statistics(
    ks_stats: torch.Tensor | Iterable[float],
    feature_names: Iterable[str] | None = None,
    top_k: int | None = 10,
    show: bool = True,
) -> plt.Figure:
    """
    Bar chart of per-feature KS statistics showing which features shifted most.

    Args:
        ks_stats: KS statistic for each feature, shape (d,)
        feature_names: Optional names for features (default: "Feature 0", etc.)
        top_k: Show only top K features by KS statistic (default: 10, None for all)
        show: Whether to call plt.show() (default: True)

    Returns:
        The matplotlib Figure object for further customization
    """
    if isinstance(ks_stats, torch.Tensor):
        ks_stats = ks_stats.cpu().numpy()
    else:
        import numpy as np

        ks_stats = np.array(list(ks_stats))

    n_features = len(ks_stats)
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]
    else:
        feature_names = list(feature_names)

    # Sort by KS statistic (descending)
    sorted_indices = ks_stats.argsort()[::-1]
    if top_k is not None:
        sorted_indices = sorted_indices[:top_k]

    sorted_stats = ks_stats[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_indices) * 0.3)))
    y_pos = range(len(sorted_indices))
    colors = plt.cm.RdYlGn_r(sorted_stats / max(sorted_stats.max(), 1e-6))
    ax.barh(y_pos, sorted_stats, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.set_xlabel("KS Statistic")
    ax.set_title("Per-Feature KS Statistics (Most Shifted Features)")
    ax.axvline(x=0.1, color="orange", linestyle="--", alpha=0.7, label="Moderate (0.1)")
    ax.axvline(x=0.2, color="red", linestyle="--", alpha=0.7, label="Significant (0.2)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    if show:
        plt.show()
    return fig
