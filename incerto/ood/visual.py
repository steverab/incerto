import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_roc(id_scores, ood_scores, label=None, ax=None):
    """
    Plot ROC curve for OOD detection.

    Args:
        id_scores: Scores for in-distribution samples.
        ood_scores: Scores for out-of-distribution samples.
        label: Label for the plot.
        ax: Matplotlib axes object.
    """
    from sklearn.metrics import RocCurveDisplay

    scores = torch.cat([id_scores, ood_scores]).cpu().numpy()
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    RocCurveDisplay.from_predictions(labels, scores, ax=ax, name=label)
    if ax is None:
        ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    return ax


def score_hist(id_scores, ood_scores, ax=None, bins=50):
    """
    Plot histogram of OOD scores for ID and OOD samples.

    Args:
        id_scores: Scores for in-distribution samples.
        ood_scores: Scores for out-of-distribution samples.
        ax: Matplotlib axes object.
        bins: Number of histogram bins.

    Returns:
        Matplotlib axes object.
    """
    ax = ax or plt.gca()
    ax.hist(id_scores.cpu(), bins=bins, alpha=0.6, label="ID")
    ax.hist(ood_scores.cpu(), bins=bins, alpha=0.6, label="OOD")
    ax.set_xlabel("OOD score")
    ax.set_ylabel("# samples")
    ax.legend()
    return ax
