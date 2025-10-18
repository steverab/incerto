"""
Base classes for calibration methods.
"""

import torch
from torch.distributions import Categorical


class BaseCalibrator:
    """
    Abstract base class for calibration methods.
    """

    def fit(self, logits: torch.Tensor, labels: torch.Tensor):  # noqa: ARG001
        """
        Fit the calibrator on a validation set.

        Args:
            logits: Tensor of shape (n_samples, n_classes).
            labels: Tensor of shape (n_samples,) with integer class labels.
        """
        raise NotImplementedError

    def predict(self, logits: torch.Tensor) -> Categorical:
        """
        Apply calibration to logits and return a Categorical distribution.

        Args:
            logits: Tensor of shape (n_samples, n_classes).

        Returns:
            A torch.distributions.Categorical over calibrated probabilities.
        """
        raise NotImplementedError
