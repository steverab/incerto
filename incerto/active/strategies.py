"""
Query strategies for active learning.

Strategies for selecting batches of samples to label, combining
acquisition functions with diversity and batch considerations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .acquisition import BaseAcquisition


class UncertaintySampling:
    """
    Uncertainty-based sampling strategy.

    Selects samples with highest acquisition scores.

    Args:
        acquisition_fn: Acquisition function to use
        batch_size: Number of samples to select per query

    Example:
        >>> acquisition = BALDAcquisition(num_samples=20)
        >>> strategy = UncertaintySampling(acquisition, batch_size=100)
        >>> indices = strategy.query(model, unlabeled_data)
    """

    def __init__(
        self,
        acquisition_fn: BaseAcquisition,
        batch_size: int = 100,
    ):
        self.acquisition_fn = acquisition_fn
        self.batch_size = batch_size

    def query(
        self,
        model: torch.nn.Module,
        x_unlabeled: torch.Tensor,
    ) -> torch.Tensor:
        """
        Query samples based on uncertainty.

        Args:
            model: Trained model
            x_unlabeled: Unlabeled data ``(N, ...)``

        Returns:
            Indices of selected samples (batch_size,)
        """
        # Compute acquisition scores
        scores = self.acquisition_fn.score(model, x_unlabeled)

        # Select top-k samples
        _, indices = torch.topk(scores, k=min(self.batch_size, len(scores)))

        return indices


class DiversitySampling:
    """
    Diversity-based sampling with uncertainty.

    Balances uncertainty with diversity to avoid selecting redundant samples.

    Reference:
        Brinker, "Incorporating Diversity in Active Learning" (ICML 2003)

    Args:
        acquisition_fn: Acquisition function
        batch_size: Number of samples to select
        diversity_weight: Weight for diversity term (0-1)
    """

    def __init__(
        self,
        acquisition_fn: BaseAcquisition,
        batch_size: int = 100,
        diversity_weight: float = 0.5,
    ):
        self.acquisition_fn = acquisition_fn
        self.batch_size = batch_size
        self.diversity_weight = diversity_weight

    def query(
        self,
        model: torch.nn.Module,
        x_unlabeled: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Query samples balancing uncertainty and diversity.

        Args:
            model: Trained model
            x_unlabeled: Unlabeled data
            features: Optional precomputed features for diversity computation

        Returns:
            Indices of selected samples
        """
        # Compute uncertainty scores
        uncertainty_scores = self.acquisition_fn.score(model, x_unlabeled)

        # Extract features if not provided
        if features is None:
            was_training = model.training
            try:
                with torch.no_grad():
                    model.eval()
                    # Use model embeddings as features
                    features = model(x_unlabeled)
                    if features.dim() > 2:
                        features = features.flatten(1)
            finally:
                model.train(was_training)

        # Normalize scores
        uncertainty_scores = (uncertainty_scores - uncertainty_scores.min()) / (
            uncertainty_scores.max() - uncertainty_scores.min() + 1e-10
        )

        # Greedy selection with diversity
        selected = []
        available = torch.arange(len(x_unlabeled))

        for _ in range(min(self.batch_size, len(x_unlabeled))):
            if len(selected) == 0:
                # First sample: select most uncertain
                idx = uncertainty_scores.argmax()
                selected.append(idx.item())
            else:
                # Compute diversity scores
                selected_features = features[selected]
                diversity_scores = (
                    torch.cdist(features[available], selected_features).min(dim=1).values
                )

                # Normalize diversity
                diversity_scores = (diversity_scores - diversity_scores.min()) / (
                    diversity_scores.max() - diversity_scores.min() + 1e-10
                )

                # Combined score
                combined = (1 - self.diversity_weight) * uncertainty_scores[
                    available
                ] + self.diversity_weight * diversity_scores

                # Select best
                best_idx = combined.argmax()
                selected.append(available[best_idx].item())

            # Update available indices (use set for O(1) lookup)
            selected_set = set(selected)
            available = torch.tensor(
                [i for i in range(len(x_unlabeled)) if i not in selected_set],
                device=x_unlabeled.device,
            )

        return torch.tensor(selected, device=x_unlabeled.device)


class CoreSetSelection:
    """
    Core-Set selection for active learning.

    Selects samples that best represent the overall data distribution
    using k-center greedy algorithm.

    Reference:
        Sener & Savarese, "Active Learning for Convolutional Neural Networks:
        A Core-Set Approach" (ICLR 2018)

    Args:
        batch_size: Number of samples to select
    """

    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size

    def query(
        self,
        model: torch.nn.Module,
        x_unlabeled: torch.Tensor,
        x_labeled: torch.Tensor | None = None,
        features_unlabeled: torch.Tensor | None = None,
        features_labeled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Select core-set using greedy k-center.

        Args:
            model: Model for feature extraction
            x_unlabeled: Unlabeled data
            x_labeled: Labeled data (optional)
            features_unlabeled: Precomputed features for unlabeled data
            features_labeled: Precomputed features for labeled data

        Returns:
            Indices of selected samples
        """
        # Extract features
        was_training = model.training
        try:
            if features_unlabeled is None:
                with torch.no_grad():
                    model.eval()
                    features_unlabeled = model(x_unlabeled)
                    if features_unlabeled.dim() > 2:
                        features_unlabeled = features_unlabeled.flatten(1)

            if x_labeled is not None and features_labeled is None:
                with torch.no_grad():
                    model.eval()
                    features_labeled = model(x_labeled)
                    if features_labeled.dim() > 2:
                        features_labeled = features_labeled.flatten(1)
        finally:
            model.train(was_training)

        # Greedy k-center
        if features_labeled is not None:
            # Start with labeled data as centers
            centers = features_labeled
        else:
            # Start with empty set
            centers = torch.empty(0, features_unlabeled.size(1), device=features_unlabeled.device)

        selected = []

        for _ in range(min(self.batch_size, len(x_unlabeled))):
            if len(centers) == 0:
                # Select random first point
                idx = torch.randint(len(features_unlabeled), (1,)).item()
            else:
                # Compute distances to nearest center
                dists = torch.cdist(features_unlabeled, centers).min(dim=1).values

                # Select point farthest from centers
                idx = dists.argmax().item()

            selected.append(idx)

            # Add to centers
            centers = torch.cat([centers, features_unlabeled[idx : idx + 1]], dim=0)

        return torch.tensor(selected, device=x_unlabeled.device)


class BadgeSampling:
    """
    BADGE (Batch Active learning by Diverse Gradient Embeddings).

    Combines gradient-based embeddings with k-MEANS++ for diverse batch selection.

    Reference:
        Ash et al., "Deep Batch Active Learning by Diverse, Uncertain Gradient
        Lower Bounds" (ICLR 2020)

    Args:
        batch_size: Number of samples to select
    """

    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size

    @staticmethod
    def _find_last_linear(model: torch.nn.Module) -> torch.nn.Linear | None:
        """Find the last nn.Linear layer in the model."""
        last_linear = None
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                last_linear = module
        return last_linear

    def _compute_gradient_embeddings(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gradient embeddings per the BADGE paper.

        The gradient embedding for sample x is the gradient of the
        cross-entropy loss (with hallucinated label y_hat = argmax)
        w.r.t. the last linear layer's weight, which equals
        ``(p - e_{y_hat}) outer h`` where p is the softmax output,
        e_{y_hat} is one-hot of the predicted class, and h is the
        penultimate-layer features.
        """
        was_training = model.training
        model.eval()

        last_linear = self._find_last_linear(model)
        if last_linear is None:
            model.train(was_training)
            raise ValueError("BadgeSampling requires a model with at least one nn.Linear layer")

        # Hook to capture the input to the last linear layer
        captured_features = []

        def hook_fn(module, input, output):
            captured_features.append(input[0].detach())

        handle = last_linear.register_forward_hook(hook_fn)

        try:
            embeddings = []
            with torch.no_grad():
                for i in range(len(x)):
                    captured_features.clear()
                    output = model(x[i : i + 1])
                    probs = F.softmax(output, dim=-1)  # (1, C)

                    h = captured_features[0]  # (1, D) penultimate features
                    y_hat = probs.argmax(dim=-1)  # (1,)

                    # One-hot of predicted class
                    e_y = torch.zeros_like(probs)
                    e_y[0, y_hat] = 1.0

                    # Gradient embedding: (p - e_y) outer h, flattened
                    diff = (probs - e_y).squeeze(0)  # (C,)
                    h_flat = h.squeeze(0)  # (D,)
                    embedding = torch.outer(diff, h_flat).flatten()  # (C*D,)
                    embeddings.append(embedding)

            return torch.stack(embeddings)
        finally:
            handle.remove()
            model.train(was_training)

    def query(
        self,
        model: torch.nn.Module,
        x_unlabeled: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select batch using BADGE.

        Args:
            model: Model
            x_unlabeled: Unlabeled data

        Returns:
            Indices of selected samples
        """
        # Compute gradient embeddings
        embeddings = self._compute_gradient_embeddings(model, x_unlabeled)

        # k-MEANS++ initialization for diversity
        selected = []
        available = torch.arange(len(embeddings))

        # First point: random
        idx = torch.randint(len(embeddings), (1,)).item()
        selected.append(idx)

        # Remaining points: proportional to distance from nearest selected
        for _ in range(min(self.batch_size - 1, len(embeddings) - 1)):
            # Compute distances to nearest selected point
            dists = torch.cdist(embeddings[available], embeddings[selected]).min(dim=1).values

            # Sample proportionally to squared distance
            probs = (dists**2) / (dists**2).sum()
            idx = available[torch.multinomial(probs, 1).item()].item()
            selected.append(idx)

            # Update available (use set for O(1) lookup)
            selected_set = set(selected)
            available = torch.tensor(
                [i for i in range(len(embeddings)) if i not in selected_set],
                device=x_unlabeled.device,
            )

        return torch.tensor(selected, device=x_unlabeled.device)


class QueryByCommittee:
    """
    Query by Committee (QBC).

    Uses disagreement among an ensemble of models to select informative samples.

    Reference:
        Seung et al., "Query by Committee" (COLT 1992)

    Args:
        models: List of committee members
        batch_size: Number of samples to select
        disagreement: Disagreement measure ('vote_entropy' or 'kl')
    """

    def __init__(
        self,
        models: list[torch.nn.Module],
        batch_size: int = 100,
        disagreement: str = "vote_entropy",
    ):
        self.models = models
        self.batch_size = batch_size
        self.disagreement = disagreement

    @torch.no_grad()
    def query(
        self,
        model: torch.nn.Module | None = None,
        x_unlabeled: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Query using committee disagreement.

        Args:
            model: Unused (committee members are provided at init).
                   Accepted for interface compatibility with other strategies.
            x_unlabeled: Unlabeled data

        Returns:
            Indices of selected samples
        """
        if x_unlabeled is None:
            raise ValueError("x_unlabeled is required")
        # Collect predictions from all committee members
        training_states = [m.training for m in self.models]
        predictions = []
        try:
            for member in self.models:
                member.eval()
                logits = member(x_unlabeled)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)
        finally:
            for member, was_training in zip(self.models, training_states, strict=False):
                member.train(was_training)

        predictions = torch.stack(predictions)  # (num_models, batch_size, num_classes)

        if self.disagreement == "vote_entropy":
            # Vote entropy: entropy of vote distribution
            votes = predictions.argmax(dim=-1)  # (num_models, batch_size)

            # Count votes for each class
            num_classes = predictions.size(-1)
            vote_counts = torch.zeros(len(x_unlabeled), num_classes, device=x_unlabeled.device)

            for i in range(len(self.models)):
                for j in range(len(x_unlabeled)):
                    vote_counts[j, votes[i, j]] += 1

            # Compute entropy of vote distribution
            vote_probs = vote_counts / len(self.models)
            scores = -(vote_probs * torch.log(vote_probs + 1e-10)).sum(dim=-1)

        elif self.disagreement == "kl":
            # Average KL divergence from mean
            mean_probs = predictions.mean(dim=0)
            scores = torch.zeros(len(x_unlabeled), device=x_unlabeled.device)

            for i in range(len(self.models)):
                kl = (
                    predictions[i]
                    * (torch.log(predictions[i] + 1e-10) - torch.log(mean_probs + 1e-10))
                ).sum(dim=-1)
                scores += kl

            scores /= len(self.models)

        else:
            raise ValueError(f"Unknown disagreement measure: {self.disagreement}")

        # Select top-k
        _, indices = torch.topk(scores, k=min(self.batch_size, len(scores)))

        return indices


__all__ = [
    "UncertaintySampling",
    "DiversitySampling",
    "CoreSetSelection",
    "BadgeSampling",
    "QueryByCommittee",
]
