"""
Utility functions for conformal prediction methods.
"""

from __future__ import annotations
import math
import torch


def compute_quantile(
    scores: torch.Tensor, alpha: float, adjusted: bool = True
) -> float:
    """
    Compute the conformal quantile at level (1 - alpha).

    Uses exact order-statistic indexing (ceil-quantile) rather than
    interpolation, which is required for the finite-sample coverage
    guarantee to hold.

    Args:
        scores: Conformity scores from calibration set.
        alpha: Desired miscoverage rate.
        adjusted: If True, uses the finite-sample correction ⌈(1-α)(n+1)⌉.

    Returns:
        Quantile threshold.
    """
    n = len(scores)
    sorted_scores = torch.sort(scores)[0]
    if adjusted:
        k = math.ceil((1 - alpha) * (n + 1))
    else:
        k = math.ceil((1 - alpha) * n)
    k = max(1, min(k, n))
    return sorted_scores[k - 1].item()


def prediction_set_from_scores(
    scores: torch.Tensor, threshold: float, descending: bool = True
) -> list[torch.Tensor]:
    """
    Convert conformity scores to prediction sets.

    Args:
        scores: Score tensor of shape (n_samples, n_classes).
        threshold: Conformity threshold.
        descending: If True, higher scores are better.

    Returns:
        List of prediction sets (class indices) for each sample.
    """
    if descending:
        mask = scores >= threshold
    else:
        mask = scores <= threshold

    return [indices.nonzero().squeeze(-1) for indices in mask]


def split_data(
    dataset: torch.utils.data.Dataset, cal_ratio: float = 0.5, seed: int | None = None
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """
    Split dataset into calibration and test sets.

    Args:
        dataset: PyTorch dataset to split.
        cal_ratio: Fraction of data to use for calibration.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (calibration_dataset, test_dataset).
    """
    if seed is not None:
        torch.manual_seed(seed)

    n = len(dataset)
    indices = torch.randperm(n)
    n_cal = int(n * cal_ratio)

    cal_indices = indices[:n_cal]
    test_indices = indices[n_cal:]

    cal_set = torch.utils.data.Subset(dataset, cal_indices.tolist())
    test_set = torch.utils.data.Subset(dataset, test_indices.tolist())

    return cal_set, test_set
