import torch
import numpy as np
from sklearn import metrics as skm


def auroc(id_scores, ood_scores):
    """
    Compute AUROC for OOD detection.

    Args:
        id_scores: Scores for in-distribution samples (lower = more ID-like).
        ood_scores: Scores for out-of-distribution samples (higher = more OOD-like).

    Returns:
        AUROC score.
    """
    scores = torch.cat([id_scores, ood_scores]).cpu().numpy()
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    return float(skm.roc_auc_score(labels, scores))


def fpr_at_tpr(id_scores, ood_scores, tpr=0.95):
    """
    Compute False Positive Rate at a target True Positive Rate.

    Args:
        id_scores: Scores for in-distribution samples.
        ood_scores: Scores for out-of-distribution samples.
        tpr: Target true positive rate (default: 0.95).

    Returns:
        FPR at the target TPR.
    """
    scores = torch.cat([id_scores, ood_scores]).cpu().numpy()
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    fpr, tpr_arr, _ = skm.roc_curve(labels, scores)
    return float(np.interp(tpr, tpr_arr, fpr))


def detection_accuracy(id_scores, ood_scores, id_accept_rate=0.95):
    """
    Compute detection accuracy at a given ID acceptance-rate threshold.

    The threshold is set at the ``id_accept_rate``-th percentile of the
    ID scores, so that approximately ``id_accept_rate`` fraction of
    in-distribution samples are correctly classified as ID.

    Args:
        id_scores: Scores for in-distribution samples (lower = more ID-like).
        ood_scores: Scores for out-of-distribution samples (higher = more OOD-like).
        id_accept_rate: Fraction of ID samples to accept (default: 0.95).

    Returns:
        Detection accuracy (fraction of correctly classified samples).
    """
    thresh = torch.quantile(id_scores, id_accept_rate)
    correct = (id_scores <= thresh).sum() + (ood_scores > thresh).sum()
    return float(correct.item() / (len(id_scores) + len(ood_scores)))
