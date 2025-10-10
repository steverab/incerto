import torch, numpy as np
from sklearn import metrics as skm


def auroc(id_scores, ood_scores):
    scores = torch.cat([id_scores, ood_scores]).cpu().numpy()
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    return skm.roc_auc_score(labels, scores)


def fpr_at_tpr(id_scores, ood_scores, tpr=0.95):
    scores = torch.cat([id_scores, ood_scores]).cpu().numpy()
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    fpr, tpr_arr, _ = skm.roc_curve(labels, scores)
    return np.interp(tpr, tpr_arr, fpr)


def detection_accuracy(id_scores, ood_scores):
    thresh = torch.quantile(id_scores, 0.95)
    correct = (id_scores <= thresh).sum() + (ood_scores > thresh).sum()
    return correct.item() / (len(id_scores) + len(ood_scores))
