import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_roc(id_scores, ood_scores, label=None, ax=None):
    from sklearn.metrics import RocCurveDisplay

    scores = torch.cat([id_scores, ood_scores]).cpu().numpy()
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    RocCurveDisplay.from_predictions(labels, scores, ax=ax, name=label)
    plt.gca().set_aspect("equal", adjustable="box")


def score_hist(id_scores, ood_scores, ax=None, bins=50):
    ax = ax or plt.gca()
    ax.hist(id_scores.cpu(), bins=bins, alpha=0.6, label="ID")
    ax.hist(ood_scores.cpu(), bins=bins, alpha=0.6, label="OOD")
    ax.set_xlabel("OOD score")
    ax.set_ylabel("# samples")
    ax.legend()
