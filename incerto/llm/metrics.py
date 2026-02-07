"""
Evaluation metrics for LLM uncertainty quantification.

Metrics specific to evaluating uncertainty estimates in language models.
"""

from __future__ import annotations
import torch
import numpy as np
from typing import List


def selective_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    confidences: torch.Tensor,
    threshold: float,
) -> dict:
    """
    Compute accuracy on high-confidence predictions.

    Args:
        predictions: Predicted classes/tokens (batch,)
        targets: True classes/tokens (batch,)
        confidences: Confidence scores (batch,)
        threshold: Confidence threshold for selection

    Returns:
        Dictionary with accuracy, coverage, and count
    """
    selected = confidences >= threshold
    n_selected = selected.sum().item()

    if n_selected == 0:
        return {
            "accuracy": 0.0,
            "coverage": 0.0,
            "n_selected": 0,
        }

    correct = (predictions[selected] == targets[selected]).sum().item()
    accuracy = correct / n_selected
    coverage = n_selected / len(predictions)

    return {
        "accuracy": accuracy,
        "coverage": coverage,
        "n_selected": n_selected,
    }


def calibration_error(
    confidences: torch.Tensor,
    correctness: torch.Tensor,
    n_bins: int = 10,
) -> dict:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        confidences: Confidence scores (batch,)
        correctness: Binary correctness (batch,)
        n_bins: Number of bins

    Returns:
        Dictionary with ECE and MCE
    """
    confidences = confidences.cpu().numpy()
    correctness = correctness.cpu().numpy()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0

    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Find samples in this bin
        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = correctness[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()

            # Calibration error for this bin
            cal_error = abs(accuracy_in_bin - avg_confidence_in_bin)

            ece += cal_error * prop_in_bin
            mce = max(mce, cal_error)

    return {
        "ece": ece,
        "mce": mce,
    }


def brier_score(confidences: torch.Tensor, correctness: torch.Tensor) -> float:
    """
    Compute Brier score for binary correctness prediction.

    Args:
        confidences: Predicted probabilities of being correct (batch,)
        correctness: Binary indicators (batch,)

    Returns:
        Brier score (lower is better)
    """
    confidences = confidences.cpu().numpy()
    correctness = correctness.cpu().numpy()

    return float(np.mean((confidences - correctness) ** 2))


def aur_c(
    confidences: torch.Tensor,
    correctness: torch.Tensor,
) -> float:
    """
    Area Under Risk-Coverage curve.

    Measures selective prediction quality - how well uncertainty
    correlates with correctness.

    Args:
        confidences: Confidence scores (batch,)
        correctness: Binary correctness (batch,)

    Returns:
        AURC value (lower is better)
    """
    # Sort by confidence (descending)
    sorted_indices = torch.argsort(confidences, descending=True)
    sorted_correct = correctness[sorted_indices].cpu().numpy()

    # Compute cumulative accuracy (1 - risk)
    cumsum = np.cumsum(sorted_correct)
    coverage_points = np.arange(1, len(sorted_correct) + 1)
    accuracy_curve = cumsum / coverage_points
    risk_curve = 1 - accuracy_curve

    # Compute area using trapezoidal rule
    coverage_normalized = coverage_points / len(sorted_correct)
    aurc = np.trapezoid(risk_curve, coverage_normalized)

    return float(aurc)


def uncertainty_auc(
    uncertainties: torch.Tensor,
    correctness: torch.Tensor,
) -> float:
    """
    AUC for using uncertainty to filter incorrect predictions.

    Higher uncertainty should correlate with incorrectness.

    Args:
        uncertainties: Uncertainty scores (batch,)
        correctness: Binary correctness (batch,)

    Returns:
        AUC value (higher is better)
    """
    from sklearn.metrics import roc_auc_score

    uncertainties = uncertainties.cpu().numpy()
    correctness = correctness.cpu().numpy()

    # Incorrectness as positive class
    incorrectness = 1 - correctness

    try:
        auc = roc_auc_score(incorrectness, uncertainties)
        # Handle NaN (sklearn returns NaN when only one class present)
        if np.isnan(auc):
            auc = 0.5
    except ValueError:
        # Handle case where all predictions are correct/incorrect
        auc = 0.5

    return float(auc)


def token_level_accuracy(
    pred_tokens: torch.Tensor,
    true_tokens: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    """
    Compute token-level accuracy.

    Args:
        pred_tokens: Predicted token IDs (batch, seq_len)
        true_tokens: True token IDs (batch, seq_len)
        mask: Optional mask for valid positions (batch, seq_len)

    Returns:
        Token accuracy
    """
    correct = pred_tokens == true_tokens

    if mask is not None:
        correct = correct & mask.bool()
        total = mask.sum().item()
    else:
        total = pred_tokens.numel()

    if total == 0:
        return 0.0

    return float(correct.sum().item() / total)


def sequence_level_accuracy(
    pred_sequences: List[str],
    true_sequences: List[str],
    normalize: bool = True,
) -> float:
    """
    Compute sequence-level exact match accuracy.

    Args:
        pred_sequences: List of predicted sequences
        true_sequences: List of true sequences
        normalize: Whether to normalize (lowercase, strip) before comparing

    Returns:
        Exact match accuracy
    """
    if normalize:
        pred_sequences = [s.lower().strip() for s in pred_sequences]
        true_sequences = [s.lower().strip() for s in true_sequences]

    correct = sum(p == t for p, t in zip(pred_sequences, true_sequences))
    return correct / len(pred_sequences)


def f1_score_tokens(
    pred_tokens: torch.Tensor,
    true_tokens: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict:
    """
    Compute precision, recall, and F1 at token level.

    This treats token prediction as a retrieval problem where:
    - True Positive: correct token at a valid (masked-in) position
    - False Positive: wrong token at a valid position
    - False Negative: true token at a masked-out position (not predicted)

    When mask covers all positions (default), FN=0 and recall=1.0,
    making F1 equal to 2*precision/(1+precision). In this case,
    consider using token_level_accuracy() instead.

    Args:
        pred_tokens: Predicted token IDs (batch, seq_len)
        true_tokens: True token IDs (batch, seq_len)
        mask: Optional mask for valid positions. Positions where mask=0
              contribute to false negatives (tokens that should have
              been predicted but weren't).

    Returns:
        Dictionary with precision, recall, F1, and token counts
    """
    if mask is None:
        mask = torch.ones_like(pred_tokens, dtype=torch.bool)
    else:
        mask = mask.bool()

    # True positives: correct predictions at valid positions
    tp = ((pred_tokens == true_tokens) & mask).sum().item()

    # False positives: wrong predictions at valid positions
    fp = ((pred_tokens != true_tokens) & mask).sum().item()

    # False negatives: true tokens at masked-out positions (not evaluated)
    fn = (~mask).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }
