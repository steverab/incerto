"""
Training-time calibration methods.

Methods that improve calibration during training, rather than
post-hoc after training is complete.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing for improved calibration.

    Softens hard labels to prevent overconfidence:
        y_smooth = (1 - α) * y_hard + α / K

    where α is the smoothing parameter and K is the number of classes.

    Reference:
        Szegedy et al. "Rethinking the Inception Architecture" (CVPR 2016)
        Müller et al. "When Does Label Smoothing Help?" (NeurIPS 2019)

    Args:
        smoothing: Smoothing parameter (default: 0.1)
        reduction: Reduction method ('mean', 'sum', or 'none')

    Example:
        >>> criterion = LabelSmoothingLoss(smoothing=0.1)
        >>> logits = torch.randn(32, 10)
        >>> targets = torch.randint(0, 10, (32,))
        >>> loss = criterion(logits, targets)
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.

        Args:
            logits: Model outputs (N, C)
            targets: Ground truth labels (N,)

        Returns:
            Loss value
        """
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smooth labels
        targets_one_hot = F.one_hot(targets, num_classes).float()
        smooth_targets = (
            1 - self.smoothing
        ) * targets_one_hot + self.smoothing / num_classes

        # Compute loss
        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling hard examples.

    Down-weights easy examples to focus on hard ones:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    The (1 - p_t)^γ modulating factor reduces the loss contribution
    from easy examples and extends the range where examples receive
    low loss.

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)

    Args:
        alpha: Weighting factor (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('mean', 'sum', or 'none')

    Example:
        >>> criterion = FocalLoss(gamma=2.0)
        >>> logits = torch.randn(32, 10)
        >>> targets = torch.randint(0, 10, (32,))
        >>> loss = criterion(logits, targets)
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model outputs (N, C)
            targets: Ground truth labels (N,)

        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ConfidencePenalty(nn.Module):
    """
    Confidence Penalty to prevent overconfidence.

    Regularizes model predictions to have higher entropy:
        Loss = CE + β * (-H(p))

    where H(p) is the entropy of predictions. The negative entropy
    term penalizes confident predictions.

    Reference:
        Pereyra et al. "Regularizing Neural Networks by Penalizing
        Confident Output Distributions" (ICLR 2017)

    Args:
        beta: Penalty weight (default: 0.1)

    Example:
        >>> criterion = ConfidencePenalty(beta=0.1)
        >>> logits = torch.randn(32, 10)
        >>> targets = torch.randint(0, 10, (32,))
        >>> loss = criterion(logits, targets)
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence penalty loss.

        Args:
            logits: Model outputs (N, C)
            targets: Ground truth labels (N,)

        Returns:
            Loss value
        """
        # Standard cross-entropy
        ce_loss = F.cross_entropy(logits, targets)

        # Confidence penalty (negative entropy)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        confidence_penalty = -entropy

        total_loss = ce_loss + self.beta * confidence_penalty

        return total_loss


def evidential_loss(
    evidence: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    epoch: int,
    num_epochs: int,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evidential Deep Learning loss.

    Learns Dirichlet distributions over class probabilities,
    enabling second-order uncertainty estimation.

    Loss = MSE(p, y) + λ * KL[Dir(α_tilde) || Dir(1)]

    where α = evidence + 1 (Dirichlet parameters) and
    λ is annealed from 0 to 1 over training.

    Reference:
        Sensoy et al. "Evidential Deep Learning to Quantify
        Classification Uncertainty" (NeurIPS 2018)

    Args:
        evidence: Non-negative evidence values (N, C)
        targets: Ground truth labels (N,)
        num_classes: Number of classes
        epoch: Current epoch
        num_epochs: Total number of epochs
        kl_weight: Maximum KL weight (default: 1.0)

    Returns:
        tuple of (total_loss, mse_loss, kl_loss)

    Example:
        >>> evidence = F.softplus(model(x))  # Ensure non-negative
        >>> loss, mse, kl = evidential_loss(evidence, targets, 10, epoch, total_epochs)
    """
    # Dirichlet parameters
    alpha = evidence + 1
    S = alpha.sum(dim=1, keepdim=True)

    # One-hot encode targets
    targets_one_hot = F.one_hot(targets, num_classes).float()

    # MSE loss between predicted probabilities and true labels
    prob = alpha / S
    mse_loss = ((targets_one_hot - prob) ** 2).sum(dim=1).mean()

    # KL divergence regularization
    # Encourages high uncertainty on wrong predictions
    alpha_tilde = targets_one_hot + (1 - targets_one_hot) * alpha
    S_tilde = alpha_tilde.sum(dim=1, keepdim=True)

    # KL[Dir(alpha_tilde) || Dir(1)]
    first_term = torch.lgamma(S_tilde) - torch.lgamma(alpha_tilde).sum(
        dim=1, keepdim=True
    )
    second_term = (
        (alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))
    ).sum(dim=1, keepdim=True)
    kl_loss = (first_term + second_term).mean()

    # Anneal KL coefficient from 0 to kl_weight
    kl_coeff = min(kl_weight, epoch / (num_epochs * 0.5))

    total_loss = mse_loss + kl_coeff * kl_loss

    return total_loss, mse_loss, kl_loss


class TemperatureAwareTraining(nn.Module):
    """
    Temperature-aware training with learnable temperature.

    Instead of post-hoc temperature scaling, learns the temperature
    parameter during training for better calibration.

    Args:
        backbone: Base neural network
        init_temp: Initial temperature (default: 1.0)
        learn_temp: Whether to learn temperature (default: True)

    Example:
        >>> backbone = ResNet18(num_classes=10)
        >>> model = TemperatureAwareTraining(backbone)
        >>> logits = model(x)
        >>> loss = F.cross_entropy(logits, targets)
    """

    def __init__(
        self,
        backbone: nn.Module,
        init_temp: float = 1.0,
        learn_temp: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.temperature = nn.Parameter(
            torch.ones(1) * init_temp,
            requires_grad=learn_temp,
        )

    def forward(self, x: torch.Tensor, return_unscaled: bool = False) -> torch.Tensor:
        """
        Forward pass with temperature scaling.

        Args:
            x: Input tensor
            return_unscaled: If True, return unscaled logits

        Returns:
            Temperature-scaled logits
        """
        logits = self.backbone(x)

        if return_unscaled:
            return logits

        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        return scaled_logits


def get_uncertainty_from_evidence(evidence: torch.Tensor, num_classes: int) -> dict:
    """
    Compute uncertainty measures from evidential outputs.

    Args:
        evidence: Non-negative evidence values (N, C)
        num_classes: Number of classes

    Returns:
        Dictionary with:
            - alpha: Dirichlet parameters (N, C)
            - belief: Predicted probabilities (N, C)
            - uncertainty: Total uncertainty / vacuity (N, 1)
            - epistemic: Epistemic uncertainty (N, 1)

    Example:
        >>> evidence = F.softplus(model(x))
        >>> uncertainty = get_uncertainty_from_evidence(evidence, num_classes=10)
        >>> total_unc = uncertainty['uncertainty']
    """
    alpha = evidence + 1
    S = alpha.sum(dim=1, keepdim=True)

    # Predicted probabilities
    belief = alpha / S

    # Total uncertainty (vacuity)
    uncertainty = num_classes / S

    # Epistemic uncertainty (model uncertainty)
    epistemic = (alpha * (S - alpha)) / (S * S * (S + 1))
    epistemic = epistemic.sum(dim=1, keepdim=True)

    return {
        "alpha": alpha,
        "belief": belief,
        "uncertainty": uncertainty,
        "epistemic": epistemic,
    }


__all__ = [
    "LabelSmoothingLoss",
    "FocalLoss",
    "ConfidencePenalty",
    "evidential_loss",
    "get_uncertainty_from_evidence",
    "TemperatureAwareTraining",
]
