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
) -> None:
    """Overlay 1-D histograms for a handful of features."""
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
    plt.show()


def plot_embedding_space(
    ref_emb: torch.Tensor, test_emb: torch.Tensor, method: str = "tsne"
) -> None:
    from sklearn.manifold import TSNE
    import numpy as np

    reducer = TSNE(n_components=2, perplexity=30) if method == "tsne" else None
    z = reducer.fit_transform(torch.cat([ref_emb, test_emb]).cpu())
    n_ref = len(ref_emb)
    plt.scatter(z[:n_ref, 0], z[:n_ref, 1], s=5, alpha=0.5, label="reference")
    plt.scatter(z[n_ref:, 0], z[n_ref:, 1], s=5, alpha=0.5, label="test")
    plt.title("Embedding space")
    plt.legend()
    plt.show()
