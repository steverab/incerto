"""
Utility functions for Bayesian deep learning.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Tuple


def predictive_entropy(predictions: torch.Tensor) -> torch.Tensor:
    """
    Compute predictive entropy (total uncertainty).

    H[y|x] = -∑ p(y|x) log p(y|x)

    where p(y|x) is the predictive distribution averaged over the posterior.

    Args:
        predictions: Tensor of shape (num_samples, batch_size, num_classes)
                    containing probability distributions

    Returns:
        Predictive entropy of shape (batch_size,)
    """
    # Average predictions over samples
    mean_probs = predictions.mean(dim=0)

    # Compute entropy
    entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)
    return entropy


def mutual_information(predictions: torch.Tensor) -> torch.Tensor:
    """
    Compute mutual information (epistemic uncertainty).

    I[y;θ|x] = H[y|x] - E_θ[H[y|x,θ]]

    This measures the information gained about the prediction by
    observing the model parameters.

    Args:
        predictions: Tensor of shape (num_samples, batch_size, num_classes)

    Returns:
        Mutual information of shape (batch_size,)
    """
    # Expected entropy: E_θ[H[y|x,θ]]
    expected_entropy = (
        -(predictions * torch.log(predictions + 1e-10)).sum(dim=-1).mean(dim=0)
    )

    # Entropy of mean: H[E_θ[y|x,θ]]
    mean_probs = predictions.mean(dim=0)
    entropy_of_mean = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

    # Mutual information
    mi = entropy_of_mean - expected_entropy
    return mi


def expected_calibration_error(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error for Bayesian predictions.

    Args:
        predictions: Mean predictions (batch_size, num_classes)
        labels: True labels (batch_size,)
        n_bins: Number of bins for calibration

    Returns:
        ECE score
    """
    confidences, pred_labels = predictions.max(dim=-1)
    accuracies = (pred_labels == labels).float()

    # Bin predictions
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)

    return ece.item()


def decompose_uncertainty(
    predictions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decompose predictive uncertainty into epistemic and aleatoric components.

    Total uncertainty = Epistemic + Aleatoric
    H[y|x] = I[y;θ|x] + E_θ[H[y|x,θ]]

    Args:
        predictions: Tensor of shape (num_samples, batch_size, num_classes)

    Returns:
        Tuple of (total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty)
        Each has shape (batch_size,)
    """
    # Total uncertainty (predictive entropy)
    total = predictive_entropy(predictions)

    # Epistemic uncertainty (mutual information)
    epistemic = mutual_information(predictions)

    # Aleatoric uncertainty (expected entropy)
    aleatoric = -(predictions * torch.log(predictions + 1e-10)).sum(dim=-1).mean(dim=0)

    return total, epistemic, aleatoric


def compute_disagreement(predictions: torch.Tensor) -> torch.Tensor:
    """
    Compute disagreement among ensemble members.

    Disagreement is measured as the variance of predictions.

    Args:
        predictions: Tensor of shape (num_models, batch_size, num_classes)

    Returns:
        Disagreement score of shape (batch_size,)
    """
    variance = predictions.var(dim=0)
    disagreement = variance.mean(dim=-1)
    return disagreement


def sample_from_posterior(
    mean: torch.Tensor,
    variance: torch.Tensor,
    num_samples: int = 1,
) -> torch.Tensor:
    """
    Sample from a Gaussian posterior.

    Args:
        mean: Mean of the posterior
        variance: Variance of the posterior
        num_samples: Number of samples to draw

    Returns:
        Samples of shape (num_samples, *mean.shape)
    """
    std = torch.sqrt(variance)
    samples = []
    for _ in range(num_samples):
        sample = mean + torch.randn_like(mean) * std
        samples.append(sample)
    return torch.stack(samples)


def ensemble_predictions_to_distribution(
    predictions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert ensemble predictions to mean and variance.

    Args:
        predictions: Tensor of shape (num_models, batch_size, *)

    Returns:
        Tuple of (mean, variance)
    """
    mean = predictions.mean(dim=0)
    variance = predictions.var(dim=0)
    return mean, variance


__all__ = [
    "predictive_entropy",
    "mutual_information",
    "expected_calibration_error",
    "decompose_uncertainty",
    "compute_disagreement",
    "sample_from_posterior",
    "ensemble_predictions_to_distribution",
]
