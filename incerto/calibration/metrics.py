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


def adaptive_ece_score(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
    norm: str = "l1",
) -> float:
    """
    Adaptive Expected Calibration Error (Nixon et al., 2019).

    Uses equal-mass binning instead of equal-width binning, making it
    more robust to varying confidence distributions.

    Reference:
        Nixon et al., "Measuring Calibration in Deep Learning" (CVPR Workshops 2019)

    Args:
        logits: Model logits (N, C)
        labels: True labels (N,)
        n_bins: Number of bins
        norm: Norm to use ('l1' or 'l2')

    Returns:
        Adaptive ECE score
    """
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels.detach().cpu().numpy()).astype(float)

    # Sort by confidence
    sorted_indices = np.argsort(confidences)
    confidences_sorted = confidences[sorted_indices]
    accuracies_sorted = accuracies[sorted_indices]

    # Create adaptive bins (equal mass)
    n = len(confidences)
    bin_size = n // n_bins

    ece = 0.0
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else n

        if start_idx >= end_idx:
            continue

        bin_conf = confidences_sorted[start_idx:end_idx].mean()
        bin_acc = accuracies_sorted[start_idx:end_idx].mean()
        weight = (end_idx - start_idx) / n

        if norm == "l1":
            ece += weight * abs(bin_acc - bin_conf)
        elif norm == "l2":
            ece += weight * (bin_acc - bin_conf) ** 2
        else:
            raise ValueError(f"Unknown norm: {norm}")

    if norm == "l2":
        ece = np.sqrt(ece)

    return float(ece)
