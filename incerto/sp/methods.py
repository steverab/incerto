"""
Selective-prediction algorithms and helper layers.

All methods expose a `forward(x, return_confidence=False)` signature
and a `.reject(confidence, threshold)` utility that returns a boolean
mask indicating which samples are *rejected* (i.e. deferred).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseSelectivePredictor(nn.Module, ABC):
    """Abstract base class for any selective predictor."""

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        *,
        return_confidence: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        logits = self._forward_logits(x)
        if return_confidence:
            confidence = self.confidence_from_logits(logits)
            return logits, confidence
        return logits

    @abstractmethod
    def _forward_logits(self, x: torch.Tensor) -> torch.Tensor: ...

    # ------------------------------------------------------------------
    #                             UTILITIES
    # ------------------------------------------------------------------
    @staticmethod
    def confidence_from_logits(logits: torch.Tensor) -> torch.Tensor:
        """Default: max softmax probability (MSP)."""
        return F.softmax(logits, dim=-1).max(dim=-1).values

    @staticmethod
    def reject(confidence: torch.Tensor, threshold: float) -> torch.Tensor:
        """Return `True` for samples that should be rejected (deferred)."""
        return confidence < threshold


# ----------------------------------------------------------------------
#                         1. Softmax-Threshold (MSP)
# ----------------------------------------------------------------------
class SoftmaxThreshold(BaseSelectivePredictor):
    """Classical confidence-thresholding à la Chow (1957)."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def _forward_logits(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.backbone(x)


# ----------------------------------------------------------------------
#                         2. Deep Gambler
# ----------------------------------------------------------------------
class DeepGambler(BaseSelectivePredictor):
    """
    Add an extra *abstain* logit and train with the gambler's loss:

        L = −log( (1 − r) * p_y + r / C )

    where `r` is the reserve (confidence to abstain) and `C` is
    the number of classes.
    """

    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        # small linear head that outputs C + 1 logits (extra abstain)
        last_dim = list(backbone.parameters())[-1].shape[0]
        self.head = nn.Linear(last_dim, num_classes + 1)

    def _forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)

    def confidence_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        *class_logits, abstain_logit = logits.split_with_sizes(
            [logits.size(-1) - 1, 1], dim=-1
        )
        class_logits = torch.cat(class_logits, dim=-1)
        # confidence is 1 − probability of abstain
        probs = F.softmax(torch.cat([class_logits, abstain_logit], dim=-1), dim=-1)
        return 1.0 - probs[..., -1]


# ----------------------------------------------------------------------
#                         3. SelectiveNet
# ----------------------------------------------------------------------
class SelectiveNet(BaseSelectivePredictor):
    """
    Implementation of SelectiveNet (Geifman & El-Yaniv, 2019).
    The model outputs:
        * h(x): class logits
        * g(x): selection probability
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        hidden: int = 128,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.backbone = backbone
        last_dim = list(backbone.parameters())[-1].shape[0]

        self.h = nn.Linear(last_dim, num_classes)
        self.g = nn.Sequential(
            nn.Linear(last_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.alpha = alpha  # coverage target in loss

    def _forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.h(feats)

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        *,
        return_confidence: bool = False,
    ):
        feats = self.backbone(x)
        logits = self.h(feats)
        sel_prob = self.g(feats).squeeze(-1)  # confidence ∈ [0,1]
        if return_confidence:
            return logits, sel_prob
        return logits

    def confidence_from_logits(self, logits):  # unused (override forward)
        raise NotImplementedError


# ----------------------------------------------------------------------
#                         FACTORY UTIL
# ----------------------------------------------------------------------
def make(selector: str, *args, **kwargs) -> BaseSelectivePredictor:
    """Quick factory: `make('msp', backbone)` or `make('selectivenet', ...)`."""
    selector = selector.lower()
    if selector in {"msp", "softmax", "threshold"}:
        return SoftmaxThreshold(*args, **kwargs)
    if selector in {"selectivenet", "sn"}:
        return SelectiveNet(*args, **kwargs)
    if selector in {"gambler", "deepgambler"}:
        return DeepGambler(*args, **kwargs)
    raise ValueError(f"Unknown selector {selector!r}")
