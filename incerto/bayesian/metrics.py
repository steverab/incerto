"""
Metrics for evaluating Bayesian deep learning models.
"""

from __future__ import annotations
import math
import torch
from typing import Tuple

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def ensemble_diversity(
    predictions: torch.Tensor,
    metric: str = "variance",
) -> float:
    """
    Measure diversity among ensemble members.

    Args:
        predictions: Tensor of shape (num_models, batch_size, num_classes)
        metric: Diversity metric ('variance', 'disagreement', 'kl')

    Returns:
        Diversity score (higher = more diverse)
    """
    if metric == "variance":
        # Average variance across predictions
        variance = predictions.var(dim=0)
        return variance.mean().item()

    elif metric == "disagreement":
        # Fraction of samples where models disagree
        pred_labels = predictions.argmax(dim=-1)  # (num_models, batch_size)
        # Check if all models agree
        mode = torch.mode(pred_labels, dim=0).values
        agreement = (pred_labels == mode.unsqueeze(0)).all(dim=0)
        disagreement_rate = 1.0 - agreement.float().mean()
        return disagreement_rate.item()

    elif metric == "kl":
        # Average KL divergence between pairs
        num_models = predictions.size(0)
        kl_sum = 0.0
        count = 0

        for i in range(num_models):
            for j in range(i + 1, num_models):
                kl = (
                    (
                        predictions[i]
                        * (
                            torch.log(predictions[i] + 1e-10)
                            - torch.log(predictions[j] + 1e-10)
                        )
                    )
                    .sum(dim=-1)
                    .mean()
                )
                kl_sum += kl.item()
                count += 1

        return kl_sum / count if count > 0 else 0.0

    else:
        raise ValueError(f"Unknown metric: {metric}")


def uncertainty_quality(
    uncertainties: torch.Tensor,
    errors: torch.Tensor,
) -> Tuple[float, float]:
    """
    Evaluate quality of uncertainty estimates.

    Measures correlation between uncertainty and prediction error.
    Good uncertainty should be high when the model is wrong.

    Args:
        uncertainties: Predicted uncertainties (batch_size,)
        errors: Binary error indicators (batch_size,) where 1=error, 0=correct

    Returns:
        Tuple of (spearman_correlation, auroc)
    """
    # Spearman correlation between uncertainty and error
    uncertainties_np = uncertainties.detach().cpu().numpy()
    errors_np = errors.detach().cpu().numpy()

    # Rank correlation
    correlation, _ = spearmanr(uncertainties_np, errors_np)

    # Handle NaN (occurs when one array is constant)
    if math.isnan(correlation):
        correlation = 0.0

    # AUROC for using uncertainty to detect errors
    try:
        auroc = roc_auc_score(errors_np, uncertainties_np)
        # sklearn returns NaN with a warning when only one class is present
        if math.isnan(auroc):
            auroc = 0.5
    except ValueError:
        # All errors are the same (all correct or all wrong)
        auroc = 0.5

    return float(correlation), float(auroc)


def disagreement(
    predictions: torch.Tensor,
    method: str = "variance",
) -> torch.Tensor:
    """
    Compute disagreement score for each sample.

    Args:
        predictions: Tensor of shape (num_models, batch_size, num_classes)
        method: Disagreement method ('variance', 'entropy')

    Returns:
        Disagreement scores of shape (batch_size,)
    """
    if method == "variance":
        # Variance of predictions
        variance = predictions.var(dim=0)
        return variance.mean(dim=-1)

    elif method == "entropy":
        # Entropy of the mean prediction
        mean_probs = predictions.mean(dim=0)
        entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)
        return entropy

    else:
        raise ValueError(f"Unknown method: {method}")


def negative_log_likelihood(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute negative log-likelihood of predictions.

    Args:
        predictions: Mean predictions (batch_size, num_classes)
        labels: True labels (batch_size,)

    Returns:
        NLL score
    """
    log_probs = torch.log(predictions + 1e-10)
    nll = -log_probs[torch.arange(len(labels)), labels].mean()
    return nll.item()


def brier_score(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute Brier score for probabilistic predictions.

    Args:
        predictions: Predictions (batch_size, num_classes)
        labels: True labels (batch_size,)

    Returns:
        Brier score
    """
    predictions.size(-1)
    one_hot = torch.zeros_like(predictions)
    one_hot[torch.arange(len(labels)), labels] = 1.0

    brier = ((predictions - one_hot) ** 2).sum(dim=-1).mean()
    return brier.item()


def predictive_log_likelihood(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute predictive log-likelihood (averaged over ensemble).

    Args:
        predictions: Ensemble predictions (num_models, batch_size, num_classes)
        labels: True labels (batch_size,)

    Returns:
        Predictive log-likelihood
    """
    # Average predictions
    mean_probs = predictions.mean(dim=0)

    # Log-likelihood
    log_probs = torch.log(mean_probs + 1e-10)
    ll = log_probs[torch.arange(len(labels)), labels].mean()
    return ll.item()


def sharpness(predictions: torch.Tensor) -> float:
    """
    Compute sharpness of probabilistic predictions.

    Sharpness measures how concentrated the predictive distribution is.
    Lower entropy = sharper predictions.

    Args:
        predictions: Predictions (batch_size, num_classes)

    Returns:
        Average entropy (lower = sharper)
    """
    entropy = -(predictions * torch.log(predictions + 1e-10)).sum(dim=-1)
    return entropy.mean().item()


__all__ = [
    "ensemble_diversity",
    "uncertainty_quality",
    "disagreement",
    "negative_log_likelihood",
    "brier_score",
    "predictive_log_likelihood",
    "sharpness",
]
