import torch
import torch.nn.functional as F
import numpy as np

from .utils import get_bin_stats


def nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Negative Log-Likelihood (cross-entropy) averaged over samples.
    """
    return F.cross_entropy(logits, labels, reduction="mean").item()


def brier_score(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Brier score: mean squared error between one-hot labels and predicted probabilities.
    """
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    n_samples, n_classes = probs.shape
    one_hot = np.eye(n_classes)[labels_np]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def ece_score(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).
    """
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels.detach().cpu().numpy()).astype(float)

    bin_conf, bin_acc, weight = get_bin_stats(confidences, accuracies, n_bins)
    return float(np.sum(weight * np.abs(bin_acc - bin_conf)))


def mce_score(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """
    Maximum Calibration Error (MCE).
    """
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels.detach().cpu().numpy()).astype(float)

    bin_conf, bin_acc, _ = get_bin_stats(confidences, accuracies, n_bins)
    return float(np.max(np.abs(bin_acc - bin_conf)))


def classwise_ece(
    logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 10
) -> float:
    """
    Class-wise ECE: average ECE computed separately for each class.
    """
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    n_samples, n_classes = probs.shape
    eces = []

    for k in range(n_classes):
        idx = labels_np == k
        if not np.any(idx):
            continue
        conf_k = probs[idx, k]
        acc_k = (labels_np[idx] == k).astype(float)
        bin_conf, bin_acc, weight = get_bin_stats(conf_k, acc_k, n_bins)
        eces.append(np.sum(weight * np.abs(bin_acc - bin_conf)))

    return float(np.mean(eces)) if eces else 0.0
