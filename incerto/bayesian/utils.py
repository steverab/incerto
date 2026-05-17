"""
Utility functions for Bayesian deep learning.
"""

from __future__ import annotations

import torch


def predictive_entropy(predictions: torch.Tensor) -> torch.Tensor:
    """
    Compute predictive entropy for batched Bayesian predictions (total uncertainty).

    This function is specialized for Bayesian deep learning where you have multiple
    posterior samples. For simple entropy of a single probability distribution,
    use `incerto.core.entropy` instead.

    Computes: H[y|x] = -∑ p(y|x) log p(y|x)
    where p(y|x) is the predictive distribution averaged over the posterior.

    Args:
        predictions: Tensor of shape (num_samples, batch_size, num_classes)
                    containing probability distributions from multiple posterior samples.
                    Each sample represents p(y|x,θ) for a different parameter θ.

    Returns:
        Predictive entropy of shape (batch_size,) representing the total
        uncertainty for each input.

    Example:
        >>> # Ensemble of 10 models, batch of 32, 5 classes
        >>> predictions = torch.softmax(torch.randn(10, 32, 5), dim=-1)
        >>> total_uncertainty = predictive_entropy(predictions)
        >>> total_uncertainty.shape
        torch.Size([32])

    See Also:
        - mutual_information: For epistemic uncertainty
        - decompose_uncertainty: For full decomposition
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
    expected_entropy = -(predictions * torch.log(predictions + 1e-10)).sum(dim=-1).mean(dim=0)

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

    This is a convenience wrapper around incerto.calibration.ece_score
    that handles the mean prediction from an ensemble.

    Args:
        predictions: Mean predictions (batch_size, num_classes).
                    For ensembles, average the predictions first.
        labels: True labels (batch_size,)
        n_bins: Number of bins for calibration

    Returns:
        ECE score

    Example:
        >>> # For ensemble predictions
        >>> ensemble_preds = torch.softmax(torch.randn(10, 32, 5), dim=-1)
        >>> mean_preds = ensemble_preds.mean(dim=0)
        >>> ece = expected_calibration_error(mean_preds, labels, n_bins=10)

    See Also:
        incerto.calibration.ece_score: The canonical ECE implementation
    """
    # Convert probabilities to logits for the canonical ece_score function
    # Use log to convert, with small epsilon to avoid log(0)
    logits = torch.log(predictions + 1e-10)

    # Import here to avoid circular dependency
    from incerto.calibration.metrics import ece_score

    return ece_score(logits, labels, n_bins=n_bins)


def decompose_uncertainty(
    predictions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        Samples of shape ``(num_samples, *mean.shape)``
    """
    std = torch.sqrt(variance)
    samples = []
    for _ in range(num_samples):
        sample = mean + torch.randn_like(mean) * std
        samples.append(sample)
    return torch.stack(samples)


def ensemble_predictions_to_distribution(
    predictions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert ensemble predictions to mean and variance.

    Args:
        predictions: Tensor of shape ``(num_models, batch_size, ...)``

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
