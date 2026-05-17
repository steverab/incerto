"""
Base classes for distribution shift detection methods.

All shift detectors should inherit from BaseShiftDetector and implement
the _compute() method for their specific test statistic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader


class BaseShiftDetector(ABC):
    """
    Abstract base class for distribution shift detection methods.

    Shift detectors compare two distributions (reference and test) to determine
    if they differ significantly. This is crucial for detecting dataset drift,
    covariate shift, or concept drift in deployed models.

    All detectors follow a fit-score pattern:
    1. fit(): Store reference (source) distribution
    2. score(): Compute shift metric against test (target) distribution

    Subclasses must implement _compute() which defines the specific test statistic.

    Example:
        >>> class MyShiftDetector(BaseShiftDetector):
        ...     def _compute(self, test: torch.Tensor) -> float:
        ...         # Compute your shift metric
        ...         return some_distance(self._reference, test)
        ...
        >>> detector = MyShiftDetector()
        >>> detector.fit(reference_loader)
        >>> shift_score = detector.score(test_loader)
        >>> if shift_score > threshold:
        ...     print("Significant shift detected!")

    Attributes:
        _reference: Cached reference distribution samples after calling fit()

    See Also:
        - MMDShiftDetector: Maximum Mean Discrepancy-based detection
        - EnergyDistanceDetector: Energy distance-based detection
        - KSTest: Kolmogorov-Smirnov test for 1D features
    """

    def fit(self, reference_loader: DataLoader) -> BaseShiftDetector:
        """
        Fit the detector on reference (source) distribution.

        This method caches the reference data for comparison with test data.
        The reference distribution is typically your training data or a known
        good distribution.

        Args:
            reference_loader: DataLoader for reference distribution.
                             Will concatenate all batches into a single tensor.

        Returns:
            self: For method chaining

        Example:
            >>> detector = MyShiftDetector()
            >>> detector.fit(train_loader).score(test_loader)
        """
        reference = torch.cat([x[0].detach() for x in reference_loader])
        # Ensure at least 2D (n, d)
        if reference.ndim == 1:
            reference = reference.unsqueeze(1)
        elif reference.ndim > 2:
            reference = reference.flatten(1)
        self._reference = reference
        return self

    @torch.no_grad()
    def score(self, test_loader: DataLoader) -> float:
        """
        Compute shift score between reference and test distributions.

        Higher scores indicate more significant distributional shift.

        Args:
            test_loader: DataLoader for test (target) distribution.

        Returns:
            Scalar shift score (higher = more shift detected).
            The scale and interpretation depend on the specific test statistic.

        Raises:
            AttributeError: If fit() has not been called first.

        Example:
            >>> shift_score = detector.score(deployment_loader)
            >>> print(f"Shift detected: {shift_score:.4f}")
        """
        if not hasattr(self, "_reference"):
            from ..exceptions import NotFittedError

            raise NotFittedError("Detector has not been fitted. Call fit() before score().")

        test_batch = torch.cat([x[0].detach() for x in test_loader])
        # Ensure at least 2D (n, d)
        if test_batch.ndim == 1:
            test_batch = test_batch.unsqueeze(1)
        elif test_batch.ndim > 2:
            test_batch = test_batch.flatten(1)
        return self._compute(test_batch)

    @abstractmethod
    def _compute(self, test: torch.Tensor) -> float:
        """
        Compute shift metric between reference and test samples.

        This is the core method that subclasses must implement to define
        their specific shift detection statistic.

        Args:
            test: Test distribution samples as a tensor.
                 The reference samples are available as self._reference.

        Returns:
            Shift score as a float (higher = more shift).

        Note:
            You can access the reference distribution via self._reference.
        """
        ...

    def state_dict(self) -> dict:
        """
        Return a dictionary containing the detector's state.

        Returns:
            Dictionary containing reference data and detector parameters.
        """
        state = {}
        if hasattr(self, "_reference"):
            state["_reference"] = self._reference
        return state

    def load_state_dict(self, state: dict) -> None:
        """
        Load detector state from a dictionary.

        Args:
            state: Dictionary containing detector state.

        Raises:
            SerializationError: If state is invalid.
        """
        from ..exceptions import serialization_errors

        with serialization_errors("load state"):
            if "_reference" in state:
                self._reference = state["_reference"]

    def save(self, path: str) -> None:
        """
        Save detector state to a file.

        Args:
            path: File path where the state will be saved.

        Raises:
            SerializationError: If saving fails.

        Example:
            >>> detector.fit(reference_loader)
            >>> detector.save('detector_state.pt')
        """
        from ..exceptions import serialization_errors

        with serialization_errors("save to {path}"):
            torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str) -> BaseShiftDetector:
        """
        Load a detector from a file.

        Args:
            path: File path to load the state from.

        Returns:
            A new detector instance with loaded state.

        Raises:
            SerializationError: If loading fails.

        Example:
            >>> detector = MyShiftDetector.load('detector_state.pt')
            >>> shift_score = detector.score(test_loader)
        """
        from ..exceptions import serialization_errors

        with serialization_errors("load from {path}"):
            state = torch.load(path, weights_only=True)
            instance = cls()
            instance.load_state_dict(state)
            return instance
