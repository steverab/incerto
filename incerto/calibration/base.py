"""
Base classes for calibration methods.

All calibration methods should inherit from BaseCalibrator and implement
the fit() and predict() methods.
"""

from abc import ABC, abstractmethod
import torch
from torch.distributions import Categorical


class BaseCalibrator(ABC):
    """
    Abstract base class for all calibration methods.

    Calibrators are post-hoc methods that adjust a trained model's predicted
    probabilities to better match empirical frequencies. All calibrators follow
    a fit-predict pattern:

    1. fit(): Learn calibration parameters on a validation set
    2. predict(): Apply calibration to new logits

    Subclasses must implement both methods.

    Example:
        >>> class MyCalibrator(BaseCalibrator):
        ...     def fit(self, logits, labels):
        ...         # Learn calibration parameters
        ...         self.temperature = find_optimal_temperature(logits, labels)
        ...         return self
        ...
        ...     def predict(self, logits):
        ...         # Apply calibration
        ...         calibrated_logits = logits / self.temperature
        ...         probs = F.softmax(calibrated_logits, dim=-1)
        ...         return Categorical(probs=probs)
        ...
        >>> calibrator = MyCalibrator()
        >>> calibrator.fit(val_logits, val_labels)
        >>> calibrated_dist = calibrator.predict(test_logits)

    See Also:
        - TemperatureScaling: Simple and effective temperature-based calibration
        - VectorScaling: Class-wise temperature scaling
        - IsotonicRegression: Non-parametric calibration
    """

    @abstractmethod
    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> "BaseCalibrator":
        """
        Fit the calibrator on a validation set.

        This method should learn any calibration parameters needed to map
        uncalibrated logits to calibrated probabilities.

        Args:
            logits: Uncalibrated model outputs of shape (n_samples, n_classes).
                   These should be raw logits before softmax.
            labels: True class labels of shape (n_samples,) with integer values
                   in range [0, n_classes-1].

        Returns:
            self: For method chaining

        Note:
            The validation set used here should be separate from both training
            and test sets to avoid overfitting the calibration parameters.
        """
        ...

    @abstractmethod
    def predict(self, logits: torch.Tensor) -> Categorical:
        """
        Apply calibration to logits and return a Categorical distribution.

        This method applies the learned calibration to transform uncalibrated
        logits into calibrated probabilities.

        Args:
            logits: Uncalibrated model outputs of shape (n_samples, n_classes).

        Returns:
            A torch.distributions.Categorical distribution over calibrated
            probabilities. Access probabilities via .probs or sample via .sample().

        Example:
            >>> calibrated_dist = calibrator.predict(test_logits)
            >>> calibrated_probs = calibrated_dist.probs  # shape: (n_samples, n_classes)
            >>> predictions = calibrated_dist.sample()     # shape: (n_samples,)
            >>> log_probs = calibrated_dist.log_prob(labels)  # shape: (n_samples,)
        """
        ...

    @abstractmethod
    def state_dict(self) -> dict:
        """
        Return a dictionary containing the calibrator's state.

        This should include all learned parameters needed to reproduce
        the calibrator's behavior after fitting.

        Returns:
            Dictionary mapping parameter names to their values.

        Example:
            >>> calibrator.fit(val_logits, val_labels)
            >>> state = calibrator.state_dict()
            >>> # state might be {'temperature': 1.5, 'n_classes': 10}
        """
        ...

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        """
        Load calibrator state from a dictionary.

        This restores the calibrator to a previously saved state,
        allowing it to be used without refitting.

        Args:
            state: Dictionary containing calibrator state, typically
                  from a previous call to state_dict().

        Raises:
            SerializationError: If state is invalid or incompatible.

        Example:
            >>> state = calibrator.state_dict()
            >>> new_calibrator = MyCalibrator()
            >>> new_calibrator.load_state_dict(state)
            >>> # new_calibrator now behaves identically to calibrator
        """
        ...

    def save(self, path: str) -> None:
        """
        Save calibrator state to a file.

        Args:
            path: File path where the state will be saved.

        Raises:
            SerializationError: If saving fails.

        Example:
            >>> calibrator.fit(val_logits, val_labels)
            >>> calibrator.save('calibrator.pt')
        """
        from ..exceptions import SerializationError

        try:
            torch.save(self.state_dict(), path)
        except Exception as e:
            raise SerializationError(f"Failed to save to {path}: {e}") from e

    @classmethod
    def load(cls, path: str) -> "BaseCalibrator":
        """
        Load a calibrator from a file.

        Args:
            path: File path to load the state from.

        Returns:
            A new calibrator instance with loaded state.

        Raises:
            SerializationError: If loading fails.

        Example:
            >>> calibrator = MyCalibrator.load('calibrator.pt')
            >>> calibrated_dist = calibrator.predict(test_logits)
        """
        from ..exceptions import SerializationError

        try:
            state = torch.load(path)
            instance = cls()
            instance.load_state_dict(state)
            return instance
        except Exception as e:
            raise SerializationError(f"Failed to load from {path}: {e}") from e
