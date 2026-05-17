import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

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


def classwise_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """
    Class-wise ECE: average ECE computed separately for each class.
    """
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    n_samples, n_classes = probs.shape
    eces = []

    for k in range(n_classes):
        conf_k = probs[:, k]
        acc_k = (labels_np == k).astype(float)
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


def _smece_at_sigma(
    confidences: np.ndarray, accuracies: np.ndarray, sigma: float, n_grid: int = 2000
) -> float:
    """
    Compute smECE at a fixed bandwidth sigma.

    Uses histogram binning + Gaussian convolution for efficiency.

    Reference:
        Blasiok & Nakkiran, "Smooth ECE: Principled Reliability Diagrams
        via Kernel Smoothing" (ICLR 2024)
    """
    n = len(confidences)
    residuals = accuracies - confidences
    dx = 1.0 / n_grid

    # Bin residuals into fine histogram
    bin_indices = np.clip((confidences / dx).astype(int), 0, n_grid - 1)
    h = np.zeros(n_grid)
    np.add.at(h, bin_indices, residuals / n)

    # Convolve with reflected Gaussian kernel
    sigma_pixels = sigma / dx
    if sigma_pixels < 0.5:
        return float(np.sum(np.abs(h)))
    smoothed = gaussian_filter1d(h, sigma_pixels, mode="reflect")

    return float(np.sum(np.abs(smoothed)))


def _find_sigma_star(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_grid: int = 2000,
    tol: float = 1e-6,
) -> float:
    """
    Find sigma* via bisection where smECE_{sigma*}(D) = sigma*.

    Since smECE_sigma is non-increasing in sigma, the fixed-point equation
    has a unique solution found by bisection.
    """
    lo, hi = tol, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        val = _smece_at_sigma(confidences, accuracies, mid, n_grid)
        if val > mid:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def smooth_ece(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Smooth Expected Calibration Error (smECE).

    A binning-free calibration measure based on kernel smoothing. The bandwidth
    is selected automatically via a fixed-point condition, yielding a consistent
    calibration measure.

    Reference:
        Blasiok & Nakkiran, "Smooth ECE: Principled Reliability Diagrams
        via Kernel Smoothing" (ICLR 2024)

    Args:
        logits: Model logits (N, C)
        labels: True labels (N,)

    Returns:
        smECE score (float in [0, 1])
    """
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels.detach().cpu().numpy()).astype(float)

    return _find_sigma_star(confidences, accuracies)
