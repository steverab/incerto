"""
Core metrics for selective prediction.

Notation:
    Y     – ground-truth labels
    Ŷ     – model predictions
    R     – reject mask; 1 = reject/defer, 0 = predict
"""

from __future__ import annotations
import torch
from typing import Tuple


# ----------------------------------------------------------------------
#                      BASIC MEASURES ON A BATCH
# ----------------------------------------------------------------------
def coverage(reject: torch.Tensor) -> torch.Tensor:
    """Proportion of *accepted* samples."""
    return 1.0 - reject.float().mean()


def risk(pred: torch.Tensor, y: torch.Tensor, reject: torch.Tensor) -> torch.Tensor:
    """Error rate on *accepted* samples (a.k.a. selective risk)."""
    correct = (pred == y).float()
    accepted = 1.0 - reject.float()
    # avoid division by zero with eps
    return 1.0 - (correct * accepted).sum() / (accepted.sum() + 1e-9)


def aurc(
    sorted_conf: torch.Tensor,
    sorted_errors: torch.Tensor,
) -> torch.Tensor:
    """
    Area under the Risk-Coverage curve.
    Inputs must be sorted *descending* by confidence.
    """
    n = sorted_conf.numel()
    cum_errors = torch.cumsum(sorted_errors, dim=0)
    risk_curve = cum_errors / torch.arange(1, n + 1, device=sorted_conf.device)
    coverage_curve = torch.arange(1, n + 1, device=sorted_conf.device) / n
    # simple trapezoidal rule
    return torch.trapz(risk_curve, coverage_curve)


def accuracy_coverage_curve(
    logits: torch.Tensor,
    y: torch.Tensor,
    confidence: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return `(coverage, accuracy)` curve for each possible threshold.
    """
    if confidence is None:
        confidence = torch.softmax(logits, dim=-1).max(dim=-1).values
    sorted_conf, idx = confidence.sort(descending=True)
    sorted_pred = logits.argmax(dim=-1)[idx]
    sorted_y = y[idx]

    cumul_correct = torch.cumsum((sorted_pred == sorted_y).float(), dim=0)
    coverage = torch.arange(1, len(y) + 1, device=y.device) / len(y)
    accuracy = cumul_correct / torch.arange(1, len(y) + 1, device=y.device)
    return coverage, accuracy
