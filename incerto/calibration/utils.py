"""
Utility functions for calibration methods and metrics.
"""

import numpy as np
import torch
import torch.nn.functional as F


def get_bin_stats(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int):
    """
    Helper to compute per-bin average confidence, accuracy, and counts.

    Args:
        confidences: Array of confidence scores.
        accuracies: Array of binary accuracies (0 or 1).
        n_bins: Number of bins.

    Returns:
        Tuple of (bin_confidences, bin_accuracies, bin_weights).
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidences, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    bin_conf = np.zeros(n_bins)
    bin_acc = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    total = len(confidences)
    if total == 0:
        return bin_conf, bin_acc, counts

    for b in range(n_bins):
        idx = bin_ids == b
        if np.any(idx):
            bin_conf[b] = np.mean(confidences[idx])
            bin_acc[b] = np.mean(accuracies[idx])
            counts[b] = np.sum(idx)
    return bin_conf, bin_acc, counts / total


def extract_confidences_and_predictions(logits: torch.Tensor):
    """
    Extract confidence scores and predictions from logits.

    Args:
        logits: Tensor of shape (n_samples, n_classes).

    Returns:
        Tuple of (confidences, predictions) as numpy arrays.
    """
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    return confidences, predictions


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert logits to probabilities using softmax.

    Args:
        logits: Tensor of shape (n_samples, n_classes).

    Returns:
        Probabilities tensor of same shape.
    """
    return F.softmax(logits, dim=1)
