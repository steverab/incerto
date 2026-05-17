"""
Acquisition functions for active learning.

Acquisition functions score unlabeled samples based on their informativeness,
allowing selection of the most valuable samples for labeling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class BaseAcquisition(ABC):
    """Base class for acquisition functions."""

    @abstractmethod
    def score(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute acquisition scores for unlabeled samples.

        Args:
            model: Trained model
            x: Unlabeled samples ``(N, ...)``
            **kwargs: Additional arguments

        Returns:
            Acquisition scores (N,), higher = more informative
        """
        pass


class RandomAcquisition(BaseAcquisition):
    """
    Random sampling (baseline).

    Samples are selected uniformly at random.
    """

    def score(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Return random scores."""
        return torch.rand(len(x), device=x.device)


class EntropyAcquisition(BaseAcquisition):
    """
    Entropy-based acquisition.

    Select samples with highest predictive entropy (most uncertain).

    Reference:
        Shannon, "A Mathematical Theory of Communication" (1948)
    """

    @torch.no_grad()
    def score(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute entropy scores."""
        was_training = model.training
        try:
            model.eval()
            logits = model(x)
            probs = F.softmax(logits, dim=-1)

            # Entropy: -∑ p log p
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            return entropy
        finally:
            model.train(was_training)


class LeastConfidenceAcquisition(BaseAcquisition):
    """
    Least confidence acquisition.

    Select samples where the model is least confident about its prediction.

    Reference:
        Lewis & Gale, "A Sequential Algorithm for Training Text Classifiers" (1994)
    """

    @torch.no_grad()
    def score(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute least confidence scores."""
        was_training = model.training
        try:
            model.eval()
            logits = model(x)
            probs = F.softmax(logits, dim=-1)

            # Least confidence: 1 - max(p)
            max_probs, _ = probs.max(dim=-1)
            return 1.0 - max_probs
        finally:
            model.train(was_training)


class MarginAcquisition(BaseAcquisition):
    """
    Margin sampling acquisition.

    Select samples with smallest margin between top-2 predictions.

    Reference:
        Scheffer et al., "Active Hidden Markov Models for Information Extraction" (2001)
    """

    @torch.no_grad()
    def score(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute margin scores."""
        was_training = model.training
        try:
            model.eval()
            logits = model(x)
            probs = F.softmax(logits, dim=-1)

            # Guard: need at least 2 classes for margin
            if probs.size(-1) < 2:
                # Single class: no margin, return zeros (no uncertainty)
                return torch.zeros(len(x), device=x.device)

            # Sort probabilities
            sorted_probs, _ = torch.sort(probs, descending=True, dim=-1)

            # Margin: difference between top-2
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]

            # Return negative margin (higher = smaller margin = more uncertain)
            return -margin
        finally:
            model.train(was_training)


class BALDAcquisition(BaseAcquisition):
    """
    Bayesian Active Learning by Disagreement (BALD).

    Selects samples that maximize the mutual information between
    predictions and model parameters.

    I[y;θ|x] = H[y|x] - E_θ[H[y|x,θ]]

    Reference:
        Houlsby et al., "Bayesian Active Learning for Classification" (ICML 2011)
    """

    def __init__(self, num_samples: int = 20):
        """
        Initialize BALD acquisition.

        Args:
            num_samples: Number of MC samples for Bayesian inference
        """
        self.num_samples = num_samples

    @torch.no_grad()
    def score(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute BALD scores."""
        was_training = model.training
        try:
            # Enable dropout for MC sampling
            model.train()

            # Collect predictions
            predictions = []
            for _ in range(self.num_samples):
                logits = model(x)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)

            predictions = torch.stack(predictions)  # (num_samples, batch_size, num_classes)

            # Expected entropy: E_θ[H[y|x,θ]]
            expected_entropy = (
                -(predictions * torch.log(predictions + 1e-10)).sum(dim=-1).mean(dim=0)
            )

            # Entropy of mean: H[E_θ[y|x,θ]]
            mean_probs = predictions.mean(dim=0)
            entropy_of_mean = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

            # Mutual information (BALD score)
            bald = entropy_of_mean - expected_entropy

            return bald
        finally:
            model.train(was_training)


class VarianceRatioAcquisition(BaseAcquisition):
    """
    Variance ratio acquisition.

    Measures disagreement as: 1 - (mode_count / num_samples)

    Reference:
        Freeman, "Learning to be uncertain" (1970)
    """

    def __init__(self, num_samples: int = 20):
        """
        Initialize variance ratio acquisition.

        Args:
            num_samples: Number of samples for estimation
        """
        self.num_samples = num_samples

    @torch.no_grad()
    def score(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute variance ratio scores."""
        was_training = model.training
        try:
            model.train()

            # Collect predictions
            predictions = []
            for _ in range(self.num_samples):
                logits = model(x)
                pred_labels = logits.argmax(dim=-1)
                predictions.append(pred_labels)

            predictions = torch.stack(predictions)  # (num_samples, batch_size)

            # Compute mode frequency (vectorized)
            mode_values = torch.mode(predictions, dim=0).values  # (batch_size,)
            mode_freq = (predictions == mode_values.unsqueeze(0)).float().sum(
                dim=0
            ) / self.num_samples

            return 1.0 - mode_freq
        finally:
            model.train(was_training)


class MeanSTDAcquisition(BaseAcquisition):
    """
    Mean standard deviation acquisition.

    Uses the average standard deviation across output probabilities
    as a measure of uncertainty.
    """

    def __init__(self, num_samples: int = 20):
        """
        Initialize mean STD acquisition.

        Args:
            num_samples: Number of samples for estimation
        """
        self.num_samples = num_samples

    @torch.no_grad()
    def score(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute mean STD scores."""
        was_training = model.training
        try:
            model.train()

            # Collect predictions
            predictions = []
            for _ in range(self.num_samples):
                logits = model(x)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)

            predictions = torch.stack(predictions)  # (num_samples, batch_size, num_classes)

            # Compute standard deviation
            std = predictions.std(dim=0)

            # Average over classes
            mean_std = std.mean(dim=-1)

            return mean_std
        finally:
            model.train(was_training)


class BatchBALDAcquisition(BaseAcquisition):
    """
    Approximate BatchBALD via individual BALD scores.

    Full BatchBALD (Kirsch et al., NeurIPS 2019) greedily selects batches
    that jointly maximise information gain by computing joint entropies.
    This implementation returns per-sample BALD scores as a tractable
    approximation; for true batch-aware selection, pair with a
    diversity-aware strategy (e.g., ``DiversitySampling``).

    Reference:
        Kirsch et al., "BatchBALD: Efficient and Diverse Batch Acquisition
        for Deep Bayesian Active Learning" (NeurIPS 2019)
    """

    def __init__(self, num_samples: int = 20):
        """
        Initialize BatchBALD acquisition.

        Args:
            num_samples: Number of MC samples
        """
        self.num_samples = num_samples

    @torch.no_grad()
    def score(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute per-sample BALD scores as a BatchBALD approximation.

        Note: This is a simplified version that returns individual BALD
        scores. Full BatchBALD requires joint entropy computation which
        is computationally expensive. For batch-aware diversity, pair
        with a diversity-aware strategy like ``DiversitySampling``.

        Args:
            model: Model
            x: Unlabeled samples

        Returns:
            Per-sample BALD scores
        """
        was_training = model.training
        try:
            model.train()

            # Collect predictions
            predictions = []
            for _ in range(self.num_samples):
                logits = model(x)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)

            predictions = torch.stack(predictions)

            # Compute individual BALD scores
            expected_entropy = (
                -(predictions * torch.log(predictions + 1e-10)).sum(dim=-1).mean(dim=0)
            )
            mean_probs = predictions.mean(dim=0)
            entropy_of_mean = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)
            bald_scores = entropy_of_mean - expected_entropy

            return bald_scores
        finally:
            model.train(was_training)


__all__ = [
    "BaseAcquisition",
    "RandomAcquisition",
    "EntropyAcquisition",
    "LeastConfidenceAcquisition",
    "MarginAcquisition",
    "BALDAcquisition",
    "VarianceRatioAcquisition",
    "MeanSTDAcquisition",
    "BatchBALDAcquisition",
]
