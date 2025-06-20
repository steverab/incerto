import torch
import torch.nn.functional as F
import numpy as np


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


def _get_bin_stats(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int):
    """
    Helper to compute per-bin average confidence, accuracy, and counts.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidences, bins, right=True) - 1
    bin_conf = np.zeros(n_bins)
    bin_acc = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    total = len(confidences)

    for b in range(n_bins):
        idx = bin_ids == b
        if np.any(idx):
            bin_conf[b] = np.mean(confidences[idx])
            bin_acc[b] = np.mean(accuracies[idx])
            counts[b] = np.sum(idx)
    return bin_conf, bin_acc, counts / total


def ece_score(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).
    """
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels.detach().cpu().numpy()).astype(float)

    _, _, weight = _get_bin_stats(confidences, accuracies, n_bins)
    bin_conf, bin_acc, _ = _get_bin_stats(confidences, accuracies, n_bins)
    return float(np.sum(weight * np.abs(bin_acc - bin_conf)))


def mce_score(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """
    Maximum Calibration Error (MCE).
    """
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels.detach().cpu().numpy()).astype(float)

    bin_conf, bin_acc, _ = _get_bin_stats(confidences, accuracies, n_bins)
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
        bin_conf, bin_acc, weight = _get_bin_stats(conf_k, acc_k, n_bins)
        eces.append(np.sum(weight * np.abs(bin_acc - bin_conf)))

    return float(np.mean(eces)) if eces else 0.0
