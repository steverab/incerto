"""
Selective-prediction algorithms and helper layers.

All methods expose a `forward(x, return_confidence=False)` signature
and a `.reject(confidence, threshold)` utility that returns a boolean
mask indicating which samples are *rejected* (i.e. deferred).
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSelectivePredictor


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
#                         4. Self-Adaptive Training (SAT)
# ----------------------------------------------------------------------
class SelfAdaptiveTraining(BaseSelectivePredictor):
    """
    Self-Adaptive Training for better calibration and selective prediction.

    Trains with adaptive soft labels that blend ground truth and model predictions:
        y_adaptive = (1 - alpha) * y_hard + alpha * softmax(logits)

    This improves calibration naturally during training, making the model better
    at knowing when to reject/abstain on uncertain samples.

    Reference:
        Huang et al., "Self-Adaptive Training: beyond Empirical Risk Minimization"
        NeurIPS 2020.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        alpha_start: float = 0.0,
        alpha_end: float = 0.9,
        warmup_epochs: int = 5,
    ):
        """
        Args:
            backbone: The base model to train
            num_classes: Number of classes
            alpha_start: Initial alpha value (0 = pure hard labels)
            alpha_end: Final alpha value (higher = more self-supervision)
            warmup_epochs: Number of epochs before starting SAT
        """
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def _forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_alpha(self, epoch: int, total_epochs: int) -> float:
        """
        Compute current alpha value based on training progress.

        Args:
            epoch: Current epoch number
            total_epochs: Total number of training epochs

        Returns:
            Alpha value for blending hard and soft labels
        """
        if epoch < self.warmup_epochs:
            return self.alpha_start

        # Linear schedule from alpha_start to alpha_end
        progress = (epoch - self.warmup_epochs) / max(
            total_epochs - self.warmup_epochs, 1
        )
        alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * progress
        return min(alpha, self.alpha_end)

    def sat_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        Compute Self-Adaptive Training loss.

        Args:
            logits: Model predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            alpha: Mixing coefficient for soft labels

        Returns:
            SAT loss value
        """
        # Standard cross-entropy with hard labels
        if alpha == 0.0:
            return F.cross_entropy(logits, targets)

        # Create one-hot encoding of hard labels
        hard_labels = F.one_hot(targets, num_classes=self.num_classes).float()

        # Get soft labels from current model predictions (detached to avoid gradient flow)
        with torch.no_grad():
            soft_labels = F.softmax(logits.detach(), dim=1)

        # Blend hard and soft labels
        adaptive_labels = (1 - alpha) * hard_labels + alpha * soft_labels

        # Compute cross-entropy with adaptive labels
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(adaptive_labels * log_probs).sum(dim=1).mean()

        return loss


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
    if selector in {"sat", "selfadaptive", "self-adaptive"}:
        return SelfAdaptiveTraining(*args, **kwargs)
    raise ValueError(f"Unknown selector {selector!r}")
