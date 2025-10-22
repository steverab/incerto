"""
Base classes for out-of-distribution (OOD) detection methods.

All OOD detectors should inherit from OODDetector and implement
the score() method for their specific detection strategy.
"""

from abc import ABC, abstractmethod
import torch


class OODDetector(ABC):
    """
    Abstract base class for out-of-distribution (OOD) detection methods.

    OOD detectors identify when input data comes from a different distribution
    than the training data. This is critical for safety in deployed models,
    as predictions on OOD data are often unreliable.

    All detectors implement a scoring function where:
    - **Higher scores → more OOD-like**
    - **Lower scores → more in-distribution**

    The model is automatically set to eval mode and gradients are disabled
    for efficiency.

    Subclasses must implement score() which defines the OOD scoring method.

    Example:
        >>> class MyOODDetector(OODDetector):
        ...     def score(self, x: torch.Tensor) -> torch.Tensor:
        ...         # Your OOD scoring logic
        ...         logits = self.model(x)
        ...         return -torch.max(logits, dim=1).values  # Higher = more OOD
        ...
        >>> detector = MyOODDetector(model)
        >>> ood_scores = detector.score(test_data)
        >>> is_ood = detector.predict(test_data, threshold=0.5)

    Attributes:
        model: The neural network in eval mode with gradients disabled.

    See Also:
        - Energy: Energy-based OOD detection (Liu et al., 2020)
        - ODIN: ODIN method with temperature and perturbations
        - Mahalanobis: Distance-based detection in feature space
        - MSP: Maximum softmax probability baseline
    """

    def __init__(self, model):
        """
        Initialize the OOD detector with a trained model.

        The model is automatically:
        1. Set to eval mode
        2. Has gradients disabled (requires_grad=False)

        Args:
            model: A trained PyTorch model (nn.Module)
        """
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @abstractmethod
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute OOD scores for input samples.

        Higher scores indicate the input is more likely to be out-of-distribution.

        Args:
            x: Input tensor of shape (batch_size, *input_dims)

        Returns:
            OOD scores of shape (batch_size,) where higher values indicate
            more OOD-like samples.

        Note:
            The scale of scores depends on the detection method. Use the
            predict() method with a threshold for binary OOD decisions.
        """
        ...

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Predict whether inputs are OOD using a threshold.

        Args:
            x: Input tensor of shape (batch_size, *input_dims)
            threshold: Score threshold for OOD classification.
                      Scores > threshold are classified as OOD.

        Returns:
            Boolean tensor of shape (batch_size,) where True indicates OOD.

        Example:
            >>> is_ood = detector.predict(test_data, threshold=0.5)
            >>> ood_count = is_ood.sum().item()
            >>> print(f"Detected {ood_count} OOD samples")
        """
        return self.score(x) > threshold  # Bool mask

    def state_dict(self) -> dict:
        """
        Return a dictionary containing the detector's state.

        Note:
            The model is NOT saved as part of the state dict.
            When loading, you must provide the model separately.

        Returns:
            Dictionary containing detector-specific parameters and fitted state.
        """
        return {}

    def load_state_dict(self, state: dict) -> None:
        """
        Load detector state from a dictionary.

        Note:
            This does not load the model. The model must be set separately
            via the __init__ method.

        Args:
            state: Dictionary containing detector state.

        Raises:
            SerializationError: If state is invalid.
        """
        pass

    def save(self, path: str) -> None:
        """
        Save detector state to a file (excluding the model).

        Args:
            path: File path where the state will be saved.

        Raises:
            SerializationError: If saving fails.

        Example:
            >>> detector.fit(train_loader)
            >>> detector.save('detector_state.pt')
        """
        from ..exceptions import SerializationError

        try:
            torch.save(self.state_dict(), path)
        except Exception as e:
            raise SerializationError(f"Failed to save to {path}: {e}") from e
