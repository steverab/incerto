"""
Base classes for OOD detection methods.
"""

from abc import ABC, abstractmethod
import torch


class OODDetector(ABC):
    """
    Base class: any detector only needs to implement `score`.
    Higher scores  ⇒  more OOD-like.
    """

    def __init__(self, model):
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @abstractmethod
    def score(self, x: torch.Tensor) -> torch.Tensor: ...

    @torch.no_grad()
    def predict(self, x, threshold):
        return self.score(x) > threshold  # Bool mask
