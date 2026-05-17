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

        Raises:
            TypeError: If model is not an nn.Module.
        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @abstractmethod
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute OOD scores for input samples.

        Higher scores indicate the input is more likely to be out-of-distribution.

        Args:
            x: Input tensor of shape ``(batch_size, ...)``

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
            x: Input tensor of shape ``(batch_size, ...)``
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

    def load_state_dict(self, state: dict) -> None:  # noqa: B027
        """
        Load detector state from a dictionary.

        Default no-op; subclasses override to restore detector-specific
        state. Not marked ``@abstractmethod`` because stateless detectors
        (e.g. MSP, MaxLogit) need no load behaviour.

        Note:
            This does not load the model. The model must be set separately
            via the __init__ method.

        Args:
            state: Dictionary containing detector state.

        Raises:
            SerializationError: If state is invalid.
        """

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
        from ..exceptions import serialization_errors

        with serialization_errors("save to {path}"):
            torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, model: torch.nn.Module, **kwargs) -> "OODDetector":
        """
        Load detector state from a file.

        Args:
            path: File path to load from.
            model: A trained PyTorch model to attach to the detector.
            **kwargs: Additional arguments for the detector constructor.

        Returns:
            An OODDetector instance with loaded state.

        Raises:
            SerializationError: If loading fails.
        """
        from ..exceptions import serialization_errors

        with serialization_errors("load from {path}"):
            state = torch.load(path, weights_only=True)

        detector = cls(model, **kwargs)
        detector.load_state_dict(state)
        return detector
