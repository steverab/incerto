"""
Training-time OOD detection methods.

Methods that improve OOD detection during training, rather than
post-hoc after training is complete.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply mixup data augmentation.

    Mixup creates virtual training examples by convex combination:
        x_tilde = λ*x_i + (1-λ)*x_j
        y_tilde = λ*y_i + (1-λ)*y_j

    where λ ~ Beta(α, α). This encourages linear behavior between
    examples and improves OOD detection by smoothing decision boundaries.

    Reference:
        Zhang et al. "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)

    Args:
        x: Input batch ``(N, ...)``
        y: Target labels (N,)
        alpha: Beta distribution parameter (default: 1.0)

    Returns:
        Tuple of (mixed_x, y_a, y_b, lambda) where:
            - mixed_x: Mixed inputs
            - y_a, y_b: Original labels for mixing
            - lambda: Mixing coefficient

    Example:
        >>> mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)
        >>> outputs = model(mixed_x)
        >>> loss = lam * criterion(outputs, y_a) + (1-lam) * criterion(outputs, y_b)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Compute mixup loss.

    Linearly interpolates the loss for mixed labels:
        Loss = λ * L(pred, y_a) + (1-λ) * L(pred, y_b)

    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: Mixing coefficient

    Returns:
        Mixup loss value

    Example:
        >>> mixed_x, y_a, y_b, lam = mixup_data(x, y)
        >>> outputs = model(mixed_x)
        >>> loss = mixup_criterion(nn.CrossEntropyLoss(), outputs, y_a, y_b, lam)
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class OutlierExposureLoss(nn.Module):
    """
    Outlier Exposure (OE) for improved OOD detection.

    Trains model to produce uniform predictions on auxiliary
    outlier dataset, improving OOD detection at test time.

    Loss = L_CE(f(x_in), y) + λ * L_OE(f(x_out))

    where L_OE encourages uniform predictions:
        L_OE = -log(1/K) * sum_i p_i = KL(uniform || p)

    Reference:
        Hendrycks et al. "Deep Anomaly Detection with Outlier Exposure"
        (ICLR 2019)

    Args:
        lambda_oe: Weight for OE loss (default: 0.5)

    Example:
        >>> criterion = OutlierExposureLoss(lambda_oe=0.5)
        >>> loss = criterion(logits_in, targets_in, logits_out)
    """

    def __init__(self, lambda_oe: float = 0.5):
        super().__init__()
        self.lambda_oe = lambda_oe

    def forward(
        self,
        logits_in: torch.Tensor,
        targets_in: torch.Tensor,
        logits_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute OE loss.

        Args:
            logits_in: In-distribution logits (N_in, C)
            targets_in: In-distribution targets (N_in,)
            logits_out: Outlier logits (N_out, C), optional

        Returns:
            Combined loss value
        """
        # Standard cross-entropy on in-distribution data
        loss_ce = F.cross_entropy(logits_in, targets_in)

        if logits_out is None:
            return loss_ce

        # Outlier exposure: encourage uniform predictions
        num_classes = logits_out.size(-1)
        log_probs_out = F.log_softmax(logits_out, dim=-1)
        uniform = torch.ones_like(log_probs_out) / num_classes

        # KL divergence from uniform: KL(uniform || p_out)
        loss_oe = F.kl_div(
            log_probs_out,
            uniform,
            reduction="batchmean",
        )

        total_loss = loss_ce + self.lambda_oe * loss_oe

        return total_loss


class EnergyRegularizedLoss(nn.Module):
    """
    Energy-based regularization for OOD detection.

    Regularizes energy scores to be lower for in-distribution
    and higher for out-of-distribution samples.

    Energy(x) = -log sum_i exp(f_i(x))

    Reference:
        Liu et al. "Energy-based Out-of-distribution Detection" (NeurIPS 2020)

    Args:
        lambda_energy: Energy regularization weight (default: 0.1)
        margin: Energy margin between ID and OOD (default: 10.0)

    Example:
        >>> criterion = EnergyRegularizedLoss()
        >>> loss = criterion(logits_in, targets_in, logits_out)
    """

    def __init__(self, lambda_energy: float = 0.1, margin: float = 10.0):
        super().__init__()
        self.lambda_energy = lambda_energy
        self.margin = margin

    def forward(
        self,
        logits_in: torch.Tensor,
        targets_in: torch.Tensor,
        logits_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute energy-regularized loss.

        Args:
            logits_in: In-distribution logits (N_in, C)
            targets_in: In-distribution targets (N_in,)
            logits_out: Out-of-distribution logits (N_out, C), optional

        Returns:
            Combined loss value
        """
        # Standard cross-entropy
        loss_ce = F.cross_entropy(logits_in, targets_in)

        if logits_out is None:
            return loss_ce

        # Energy scores
        energy_in = -torch.logsumexp(logits_in, dim=-1)
        energy_out = -torch.logsumexp(logits_out, dim=-1)

        # Hinge loss: encourage energy_in < energy_out - margin
        loss_energy = F.relu(energy_in - energy_out + self.margin).mean()

        total_loss = loss_ce + self.lambda_energy * loss_energy

        return total_loss


class CutMix:
    """
    CutMix augmentation for improved robustness and OOD detection.

    Instead of linearly interpolating images (mixup), cuts and pastes
    patches between images:
        x_tilde = M ⊙ x_i + (1-M) ⊙ x_j
        y_tilde = λ*y_i + (1-λ)*y_j

    where M is a binary mask and λ is the area ratio.

    Reference:
        Yun et al. "CutMix: Regularization Strategy to Train Strong
        Classifiers with Localizable Features" (ICCV 2019)

    Args:
        alpha: Beta distribution parameter (default: 1.0)

    Example:
        >>> cutmix = CutMix(alpha=1.0)
        >>> mixed_x, y_a, y_b, lam = cutmix(x, y)
        >>> outputs = model(mixed_x)
        >>> loss = lam * criterion(outputs, y_a) + (1-lam) * criterion(outputs, y_b)
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix augmentation.

        Args:
            x: Input batch (N, C, H, W)
            y: Target labels (N,)

        Returns:
            Tuple of (mixed_x, y_a, y_b, lambda)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        # Generate random bounding box
        _, _, H, W = x.size()
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # Apply CutMix
        mixed_x = x.clone()
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda based on actual cut area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam


__all__ = [
    "mixup_data",
    "mixup_criterion",
    "OutlierExposureLoss",
    "EnergyRegularizedLoss",
    "CutMix",
]
