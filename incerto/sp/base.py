"""
Base classes for selective prediction methods.
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
