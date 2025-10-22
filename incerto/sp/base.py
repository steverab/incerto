"""
Base classes for selective prediction methods.

Selective prediction (also called prediction with rejection) allows models
to abstain from making predictions when uncertain, trading coverage for accuracy.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseSelectivePredictor(nn.Module, ABC):
    """
    Abstract base class for selective prediction methods.

    Selective predictors enable models to abstain from predictions when uncertain,
    optimizing the risk-coverage tradeoff. This is crucial for safety-critical
    applications where wrong predictions are costly.

    All selective predictors:
    1. Generate logits via _forward_logits()
    2. Compute confidence scores (default: max softmax probability)
    3. Reject low-confidence predictions

    Subclasses must implement _forward_logits() which defines how predictions
    are made (may include rejection mechanism in architecture).

    Example:
        >>> class MySelectiveModel(BaseSelectivePredictor):
        ...     def __init__(self, backbone):
        ...         super().__init__()
        ...         self.backbone = backbone
        ...
        ...     def _forward_logits(self, x):
        ...         return self.backbone(x)
        ...
        >>> model = MySelectiveModel(resnet)
        >>> logits, conf = model(x, return_confidence=True)
        >>> should_reject = model.reject(conf, threshold=0.9)
        >>> # Make predictions only on high-confidence samples
        >>> predictions = logits[~should_reject].argmax(dim=-1)

    See Also:
        - SoftmaxThreshold: Simple confidence thresholding
        - SelfAdaptiveTraining: SAT with learned rejection
        - DeepGambler: Gambler's loss for selective classification
        - SelectiveNet: Auxiliary selection head
    """

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        *,
        return_confidence: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Forward pass with optional confidence scores.

        Args:
            x: Input tensor of shape (batch_size, *input_dims)
            return_confidence: If True, return (logits, confidence) tuple.
                             If False, return only logits.

        Returns:
            If return_confidence=False: logits tensor
            If return_confidence=True: (logits, confidence) tuple

        Example:
            >>> logits = model(x)  # Just predictions
            >>> logits, conf = model(x, return_confidence=True)  # With confidence
        """
        logits = self._forward_logits(x)
        if return_confidence:
            confidence = self.confidence_from_logits(logits)
            return logits, confidence
        return logits

    @abstractmethod
    def _forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for input samples.

        This is the core method that subclasses must implement.

        Args:
            x: Input tensor of shape (batch_size, *input_dims)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        ...

    # ------------------------------------------------------------------
    #                             UTILITIES
    # ------------------------------------------------------------------
    @staticmethod
    def confidence_from_logits(logits: torch.Tensor) -> torch.Tensor:
        """
        Extract confidence scores from logits.

        Default implementation uses maximum softmax probability (MSP).
        Subclasses can override for different confidence measures.

        Args:
            logits: Tensor of shape (batch_size, num_classes)

        Returns:
            Confidence scores of shape (batch_size,) in range [0, 1]
        """
        return F.softmax(logits, dim=-1).max(dim=-1).values

    @staticmethod
    def reject(confidence: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Determine which samples should be rejected based on confidence.

        Args:
            confidence: Confidence scores of shape (batch_size,)
            threshold: Confidence threshold. Samples below this are rejected.

        Returns:
            Boolean tensor of shape (batch_size,) where True indicates
            the sample should be rejected (abstained).

        Example:
            >>> conf = torch.tensor([0.95, 0.60, 0.85, 0.40])
            >>> should_reject = BaseSelectivePredictor.reject(conf, threshold=0.7)
            >>> should_reject
            tensor([False, True, False, True])
        """
        return confidence < threshold
