"""
Selective-prediction algorithms and helper layers.

All methods expose a `forward(x, return_confidence=False)` signature
and a `.reject(confidence, threshold)` utility that returns a boolean
mask indicating which samples are *rejected* (i.e. deferred).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSelectivePredictor


def _infer_output_dim(backbone: nn.Module) -> int:
    """Infer output dimension via a probe forward pass."""
    was_training = backbone.training
    backbone.eval()
    # Find an input dimension from the first parameter
    first_param = next(backbone.parameters())
    in_features = first_param.shape[-1]
    with torch.no_grad():
        probe = torch.zeros(1, in_features, device=first_param.device)
        out = backbone(probe)
    if was_training:
        backbone.train()
    return out.shape[-1]


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

        L = −log( p_y + r / o )

    where ``o`` is the reward for a correct prediction, ``p_y`` is the
    probability assigned to the true class, and ``r`` = P(abstain).

    Reference:
        Ziyin et al., "Deep Gamblers: Learning to Abstain with Portfolio
        Theory", NeurIPS 2019.
    """

    def __init__(self, backbone: nn.Module, num_classes: int, num_features: int | None = None):
        super().__init__()
        self.backbone = backbone
        # small linear head that outputs C + 1 logits (extra abstain)
        last_dim = num_features if num_features is not None else _infer_output_dim(backbone)
        self.head = nn.Linear(last_dim, num_classes + 1)
        # Initialise the abstain logit bias to a negative value so the model
        # starts by predicting classes rather than collapsing to always-abstain.
        with torch.no_grad():
            self.head.bias[-1] = -3.0

    def _forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)

    def confidence_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # confidence is 1 − P(abstain), where abstain is the last logit
        probs = F.softmax(logits, dim=-1)
        return 1.0 - probs[..., -1]

    @staticmethod
    def gambler_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        reward: float = 2.2,
    ) -> torch.Tensor:
        """Gambler's loss from Ziyin et al. (NeurIPS 2019).

        Stateless utility — callable either as ``DeepGambler.gambler_loss(...)``
        or ``gambler.gambler_loss(...)``. (Note: by contrast,
        :meth:`SelectiveNet.selective_loss` is an instance method because it
        consumes ``self.alpha`` and ``self.lam``.)

        The loss encourages the model to either predict correctly or abstain:

            L = −log( p_y + r / o )

        where ``o`` = reward, ``p_y`` = P(true class), ``r`` = P(abstain).
        Higher ``reward`` penalises abstention more, pushing towards prediction.

        Note:
            For very small ``p_target + reserve / reward`` (e.g. strongly
            confident wrong predictions) the loss can be large (~20+). This
            is expected: the loss is unbounded and the ``1e-9`` term is only
            for numerical stability around log(0).

        Args:
            logits: Model output of shape ``(batch, num_classes + 1)``.
                Last column is the abstain logit.
            targets: Ground-truth class labels of shape ``(batch,)``.
            reward: Reward for correct prediction (called *o* in the paper).
                Must be > 1. Higher values discourage abstention.

        Returns:
            Scalar loss.
        """
        probs = F.softmax(logits, dim=-1)
        num_classes = logits.size(-1) - 1
        class_probs = probs[:, :num_classes]
        reserve = probs[:, -1]

        p_target = class_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        loss = -torch.log(p_target + reserve / reward + 1e-9)
        return loss.mean()


# ----------------------------------------------------------------------
#                         3. SelectiveNet
# ----------------------------------------------------------------------
class SelectiveNet(BaseSelectivePredictor):
    """Implementation of SelectiveNet (Geifman & El-Yaniv, 2019).

    The model outputs:

    - ``h(x)``: class logits
    - ``g(x)``: selection probability
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        hidden: int = 128,
        alpha: float = 0.5,
        lam: float = 32.0,
        num_features: int | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        last_dim = num_features if num_features is not None else _infer_output_dim(backbone)

        self.h = nn.Linear(last_dim, num_classes)
        self.g = nn.Sequential(
            nn.Linear(last_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.alpha = alpha  # coverage target in loss
        self.lam = lam  # penalty coefficient for coverage constraint

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

    def confidence_from_logits(self, logits):
        """Not applicable — SelectiveNet uses a dedicated selection head g(x).

        Use ``forward(x, return_confidence=True)`` instead.
        """
        raise NotImplementedError(
            "SelectiveNet uses a dedicated selection head (g). "
            "Call forward(x, return_confidence=True) to get confidence."
        )

    def selective_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        selection: torch.Tensor,
        coverage_target: float | None = None,
    ) -> torch.Tensor:
        """
        SelectiveNet loss from Geifman & El-Yaniv (ICML 2019).

        Combines a selection-weighted prediction loss with a quadratic
        coverage penalty:

            L = L_selective + λ * max(0, c − Φ)²

        where ``L_selective`` is the cross-entropy weighted by selection
        probabilities, ``c`` is the coverage target (``self.alpha``),
        ``Φ`` is the empirical coverage, and ``λ`` is the penalty
        coefficient (``self.lam``).

        Args:
            logits: Class logits of shape (batch, num_classes).
            targets: Ground-truth labels of shape (batch,).
            selection: Selection probabilities g(x) of shape (batch,)
                       in [0, 1], as returned by
                       ``forward(x, return_confidence=True)``.
            coverage_target: Desired coverage. Defaults to ``self.alpha``
                             set at construction time.

        Returns:
            Scalar loss.
        """
        c = coverage_target if coverage_target is not None else self.alpha

        # Selection-weighted cross-entropy
        per_sample_loss = F.cross_entropy(logits, targets, reduction="none")
        empirical_coverage = selection.mean()
        selective_loss = (per_sample_loss * selection).mean() / (empirical_coverage + 1e-9)

        # Quadratic coverage penalty
        coverage_penalty = self.lam * torch.clamp(c - empirical_coverage, min=0.0) ** 2

        return selective_loss + coverage_penalty


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
        progress = (epoch - self.warmup_epochs) / max(total_epochs - self.warmup_epochs, 1)
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
