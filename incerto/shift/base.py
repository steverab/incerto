"""
Base classes for distribution shift detection methods.
"""

from __future__ import annotations
import torch
from torch.utils.data import DataLoader


class BaseShiftDetector:
    """
    Shared machinery; child classes implement _compute().
    """

    def fit(self, reference_loader: DataLoader) -> "BaseShiftDetector":
        """
        Fit the detector on reference (source) data.

        Args:
            reference_loader: DataLoader for reference distribution.

        Returns:
            Self for method chaining.
        """
        self._reference = torch.cat([x[0].detach() for x in reference_loader])
        return self

    @torch.no_grad()
    def score(self, test_loader: DataLoader) -> float:
        """
        Compute shift score between reference and test distributions.

        Args:
            test_loader: DataLoader for test distribution.

        Returns:
            Scalar shift score (higher = more shift).
        """
        test_batch = torch.cat([x[0].detach() for x in test_loader])
        return self._compute(test_batch)

    def _compute(self, test: torch.Tensor) -> float:
        """
        Compute shift metric. Subclasses must implement this.

        Args:
            test: Test distribution samples.

        Returns:
            Shift score.
        """
        raise NotImplementedError
