"""
Utility functions for active learning.
"""

from __future__ import annotations
import torch
import numpy as np
from typing import Tuple, Optional


def split_labeled_unlabeled(
    data: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    labeled_indices: Optional[torch.Tensor] = None,
    unlabeled_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Split data into labeled and unlabeled sets.

    Args:
        data: Full dataset (N, *)
        labels: Labels for all data (N,), can have -1 for unlabeled
        labeled_indices: Indices of labeled samples
        unlabeled_indices: Indices of unlabeled samples

    Returns:
        Tuple of (x_labeled, x_unlabeled, y_labeled, labeled_indices)
    """
    if labeled_indices is None and unlabeled_indices is None:
        if labels is None:
            raise ValueError("Must provide either labels or indices")

        # Use labels to determine split
        labeled_mask = labels != -1
        labeled_indices = torch.where(labeled_mask)[0]
        unlabeled_indices = torch.where(~labeled_mask)[0]

    elif labeled_indices is None:
        # Compute labeled from unlabeled
        all_indices = torch.arange(len(data))
        labeled_mask = ~torch.isin(all_indices, unlabeled_indices)
        labeled_indices = all_indices[labeled_mask]

    elif unlabeled_indices is None:
        # Compute unlabeled from labeled
        all_indices = torch.arange(len(data))
        unlabeled_mask = ~torch.isin(all_indices, labeled_indices)
        unlabeled_indices = all_indices[unlabeled_mask]

    x_labeled = data[labeled_indices]
    x_unlabeled = data[unlabeled_indices]

    y_labeled = labels[labeled_indices] if labels is not None else None

    return x_labeled, x_unlabeled, y_labeled, labeled_indices


def compute_diversity_penalty(
    selected: torch.Tensor,
    features: torch.Tensor,
    method: str = "min_distance",
) -> torch.Tensor:
    """
    Compute diversity penalty for selected samples.

    Args:
        selected: Indices of selected samples
        features: Feature representations of all samples
        method: Diversity measure ('min_distance', 'mean_distance', 'determinant')

    Returns:
        Diversity penalty (lower = more diverse)
    """
    if len(selected) == 0:
        return torch.tensor(0.0, device=features.device)

    selected_features = features[selected]

    if method == "min_distance":
        # Minimum pairwise distance (lower = less diverse)
        if len(selected) < 2:
            return torch.tensor(float("inf"), device=features.device)

        dists = torch.cdist(selected_features, selected_features)
        # Mask diagonal
        mask = torch.eye(len(selected), device=dists.device).bool()
        dists[mask] = float("inf")
        return dists.min()

    elif method == "mean_distance":
        # Mean pairwise distance
        if len(selected) < 2:
            return torch.tensor(float("inf"), device=features.device)

        dists = torch.cdist(selected_features, selected_features)
        mask = torch.eye(len(selected), device=dists.device).bool()
        dists[mask] = 0
        return dists.sum() / (len(selected) * (len(selected) - 1))

    elif method == "determinant":
        # Log determinant of covariance (higher = more diverse)
        if len(selected) < features.size(1):
            # Need at least d samples for d-dimensional features
            return torch.tensor(0.0, device=features.device)

        cov = torch.cov(selected_features.T)
        try:
            det = torch.linalg.det(cov)
            return -torch.log(det + 1e-10)  # Negative log for penalty
        except:
            return torch.tensor(0.0, device=features.device)

    else:
        raise ValueError(f"Unknown method: {method}")


def greedy_k_center(
    features: torch.Tensor,
    k: int,
    initial_centers: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Greedy k-center algorithm for diverse sample selection.

    Iteratively selects points that are farthest from all previously
    selected points.

    Args:
        features: Feature representations (N, D)
        k: Number of centers to select
        initial_centers: Optional initial centers

    Returns:
        Indices of selected centers
    """
    n = len(features)
    selected = []

    # Initialize with existing centers if provided
    if initial_centers is not None:
        centers = initial_centers
    else:
        centers = torch.empty(0, features.size(1), device=features.device)

    # Initialize distances
    if len(centers) > 0:
        min_dists = torch.cdist(features, centers).min(dim=1).values
    else:
        min_dists = torch.full((n,), float("inf"), device=features.device)

    for _ in range(k):
        # Select point with maximum distance to nearest center
        idx = min_dists.argmax().item()
        selected.append(idx)

        # Update centers
        centers = torch.cat([centers, features[idx : idx + 1]], dim=0)

        # Update minimum distances
        new_dists = torch.norm(features - features[idx : idx + 1], dim=1)
        min_dists = torch.minimum(min_dists, new_dists)

    return torch.tensor(selected, device=features.device)


def subsample_for_efficiency(
    data: torch.Tensor,
    max_samples: int = 10000,
    random_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Subsample data for computational efficiency.

    Args:
        data: Full dataset
        max_samples: Maximum number of samples to keep
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (subsampled_data, indices)
    """
    if len(data) <= max_samples:
        return data, torch.arange(len(data), device=data.device)

    if random_seed is not None:
        torch.manual_seed(random_seed)

    indices = torch.randperm(len(data), device=data.device)[:max_samples]
    indices = indices.sort().values  # Sort for consistency

    return data[indices], indices


def active_learning_loop(
    model: torch.nn.Module,
    x_pool: torch.Tensor,
    y_pool: torch.Tensor,
    strategy,
    num_rounds: int = 10,
    batch_size: int = 100,
    initial_labeled: int = 100,
    train_fn: Optional[callable] = None,
    eval_fn: Optional[callable] = None,
    random_seed: Optional[int] = None,
) -> dict:
    """
    Run a full active learning loop.

    Args:
        model: Model to train
        x_pool: Full data pool
        y_pool: Full labels
        strategy: Query strategy
        num_rounds: Number of active learning rounds
        batch_size: Samples to label per round
        initial_labeled: Number of initial labeled samples
        train_fn: Function to train model (model, x_train, y_train)
        eval_fn: Function to evaluate model (model, x_test, y_test)
        random_seed: Random seed

    Returns:
        Dictionary with results
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # Initialize with random labeled samples
    n_total = len(x_pool)
    all_indices = torch.arange(n_total)
    labeled_indices = all_indices[torch.randperm(n_total)[:initial_labeled]]
    unlabeled_indices = torch.tensor(
        [i for i in range(n_total) if i not in labeled_indices]
    )

    results = {
        "labeled_sizes": [],
        "accuracies": [],
        "selected_indices": [],
    }

    for round_idx in range(num_rounds):
        print(f"Round {round_idx + 1}/{num_rounds}")

        # Get current split
        x_train = x_pool[labeled_indices]
        y_train = y_pool[labeled_indices]
        x_unlabeled = x_pool[unlabeled_indices]

        # Train model
        if train_fn is not None:
            train_fn(model, x_train, y_train)

        # Evaluate
        if eval_fn is not None:
            accuracy = eval_fn(model)
            results["accuracies"].append(accuracy)
            print(f"  Accuracy: {accuracy:.4f}")

        results["labeled_sizes"].append(len(labeled_indices))

        # Query next batch
        if len(unlabeled_indices) == 0:
            print("No more unlabeled samples")
            break

        query_indices = strategy.query(model, x_unlabeled)

        # Map query indices back to pool indices
        selected = unlabeled_indices[query_indices]
        results["selected_indices"].append(selected)

        # Update labeled/unlabeled sets
        labeled_indices = torch.cat([labeled_indices, selected])
        unlabeled_mask = torch.ones(n_total, dtype=torch.bool)
        unlabeled_mask[labeled_indices] = False
        unlabeled_indices = torch.where(unlabeled_mask)[0]

    return results


__all__ = [
    "split_labeled_unlabeled",
    "compute_diversity_penalty",
    "greedy_k_center",
    "subsample_for_efficiency",
    "active_learning_loop",
]
