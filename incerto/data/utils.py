"""
Dataset utilities for uncertainty quantification.

Helper functions for dataset manipulation and analysis.
"""

from __future__ import annotations
import torch
from torch.utils.data import Dataset, Subset
from typing import List, Optional, Callable
import numpy as np


def _get_targets(dataset: Dataset) -> np.ndarray:
    """Extract target labels from a dataset, handling nested Subset/TransformDataset wrapping."""
    indices = None
    current = dataset
    while isinstance(current, (Subset, TransformDataset)):
        if isinstance(current, Subset):
            subset_indices = np.array(current.indices)
            if indices is None:
                indices = subset_indices
            else:
                indices = subset_indices[indices]
            current = current.dataset
        elif isinstance(current, TransformDataset):
            current = current.dataset

    if hasattr(current, "targets"):
        all_targets = np.array(current.targets)
    elif hasattr(current, "labels"):
        all_targets = np.array(current.labels)
    else:
        raise ValueError("Dataset must have 'targets' or 'labels' attribute")

    if indices is not None:
        return all_targets[indices]
    return all_targets


def split_dataset(
    dataset: Dataset,
    splits: List[float],
    seed: int = 42,
) -> List[Subset]:
    """
    Split dataset into multiple subsets.

    Args:
        dataset: Dataset to split
        splits: List of split fractions (must sum to 1.0)
        seed: Random seed for reproducibility

    Returns:
        List of Subset objects

    Example:
        >>> train, val, test = split_dataset(dataset, [0.7, 0.15, 0.15])
    """
    if not np.isclose(sum(splits), 1.0):
        raise ValueError(f"Splits must sum to 1.0, got {sum(splits)}")

    n_total = len(dataset)
    split_sizes = [int(n_total * split) for split in splits]

    # Adjust last split to account for rounding
    split_sizes[-1] = n_total - sum(split_sizes[:-1])

    # Create reproducible permutation
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator).tolist()

    # Create subsets
    subsets = []
    start_idx = 0
    for size in split_sizes:
        end_idx = start_idx + size
        subset_indices = indices[start_idx:end_idx]
        subsets.append(Subset(dataset, subset_indices))
        start_idx = end_idx

    return subsets


def filter_dataset_by_class(
    dataset: Dataset,
    classes: List[int],
    invert: bool = False,
) -> Subset:
    """
    Filter dataset to only include (or exclude) specific classes.

    Args:
        dataset: Dataset with 'targets' or 'labels' attribute
        classes: List of class indices to keep
        invert: If True, exclude specified classes instead

    Returns:
        Subset containing filtered samples
    """
    targets = _get_targets(dataset)

    # Find matching indices
    if invert:
        mask = ~np.isin(targets, classes)
    else:
        mask = np.isin(targets, classes)

    indices = np.where(mask)[0].tolist()

    return Subset(dataset, indices)


def get_class_balanced_subset(
    dataset: Dataset,
    samples_per_class: int,
    seed: int = 42,
) -> Subset:
    """
    Create a class-balanced subset with equal samples per class.

    Args:
        dataset: Dataset with 'targets' or 'labels' attribute
        samples_per_class: Number of samples per class
        seed: Random seed

    Returns:
        Balanced subset
    """
    targets = _get_targets(dataset)

    rng = np.random.default_rng(seed)

    # Sample from each class
    selected_indices = []
    for class_idx in np.unique(targets):
        class_indices = np.where(targets == class_idx)[0]

        if len(class_indices) < samples_per_class:
            raise ValueError(
                f"Class {class_idx} has only {len(class_indices)} samples, "
                f"but {samples_per_class} requested"
            )

        sampled = rng.choice(
            class_indices,
            size=samples_per_class,
            replace=False,
        )
        selected_indices.extend(sampled.tolist())

    return Subset(dataset, selected_indices)


def compute_dataset_statistics(
    dataset: Dataset,
) -> dict:
    """
    Compute statistics about a dataset.

    Args:
        dataset: Dataset to analyze

    Returns:
        Dictionary with statistics
    """
    stats = {
        "size": len(dataset),
    }

    # Class distribution
    try:
        targets = _get_targets(dataset)
        unique, counts = np.unique(targets, return_counts=True)
        stats["num_classes"] = len(unique)
        stats["class_distribution"] = dict(zip(unique.tolist(), counts.tolist()))
        stats["min_class_size"] = int(counts.min())
        stats["max_class_size"] = int(counts.max())
        stats["mean_class_size"] = float(counts.mean())
        stats["class_balance_ratio"] = float(counts.min() / counts.max())
    except ValueError:
        pass  # Dataset has no targets/labels attribute

    return stats


def create_imbalanced_dataset(
    dataset: Dataset,
    imbalance_ratio: float = 0.1,
    minority_classes: Optional[List[int]] = None,
    seed: int = 42,
) -> Subset:
    """
    Create an imbalanced version of a dataset.

    Args:
        dataset: Original dataset
        imbalance_ratio: Ratio of minority to majority class size
        minority_classes: Classes to make minority (default: half of classes)
        seed: Random seed

    Returns:
        Imbalanced subset
    """
    targets = _get_targets(dataset)

    rng = np.random.default_rng(seed)

    unique_classes = np.unique(targets)

    # Determine minority classes
    if minority_classes is None:
        # Make half the classes minority
        n_minority = len(unique_classes) // 2
        minority_classes = rng.choice(
            unique_classes,
            size=n_minority,
            replace=False,
        ).tolist()

    # Find majority class size
    majority_classes = [c for c in unique_classes if c not in minority_classes]
    majority_indices = np.where(np.isin(targets, majority_classes))[0]
    majority_size = len(majority_indices) // len(majority_classes)

    # Calculate minority size
    minority_size = int(majority_size * imbalance_ratio)

    # Sample from each class
    selected_indices = []

    for class_idx in unique_classes:
        class_indices = np.where(targets == class_idx)[0]

        if class_idx in minority_classes:
            size = min(minority_size, len(class_indices))
        else:
            size = min(majority_size, len(class_indices))

        sampled = rng.choice(class_indices, size=size, replace=False)
        selected_indices.extend(sampled.tolist())

    return Subset(dataset, selected_indices)


class TransformDataset(Dataset):
    """
    Wrapper to apply transformations to a dataset.

    Useful for applying different transforms to an existing dataset.
    """

    def __init__(self, dataset: Dataset, transform: Callable):
        """
        Initialize transform dataset.

        Args:
            dataset: Base dataset
            transform: Transform to apply to data
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        data = self.transform(data)
        return data, label


class LabelNoiseDataset(Dataset):
    """
    Add label noise to a dataset.

    Useful for studying robustness to noisy labels.
    """

    def __init__(
        self,
        dataset: Dataset,
        noise_rate: float = 0.1,
        num_classes: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Initialize label noise dataset.

        Args:
            dataset: Base dataset
            noise_rate: Fraction of labels to corrupt (0-1)
            num_classes: Number of classes (auto-detected if None)
            seed: Random seed
        """
        self.dataset = dataset
        self.noise_rate = noise_rate
        self.seed = seed

        self.original_labels = _get_targets(dataset)

        # Determine number of classes
        if num_classes is None:
            num_classes = len(np.unique(self.original_labels))
        self.num_classes = num_classes

        # Generate noisy labels
        self.noisy_labels = self._generate_noisy_labels()

    def _generate_noisy_labels(self) -> np.ndarray:
        """Generate noisy labels."""
        rng = np.random.default_rng(self.seed)

        labels = self.original_labels.copy()
        n_samples = len(labels)
        n_noisy = int(n_samples * self.noise_rate)

        # Select samples to corrupt
        noisy_indices = rng.choice(n_samples, size=n_noisy, replace=False)

        # Corrupt labels (change to random different class)
        for idx in noisy_indices:
            original_label = labels[idx]
            # Choose different label
            possible_labels = [
                i for i in range(self.num_classes) if i != original_label
            ]
            labels[idx] = rng.choice(possible_labels)

        return labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, _ = self.dataset[idx]  # Ignore original label
        noisy_label = int(self.noisy_labels[idx])
        return data, noisy_label


def merge_datasets(*datasets: Dataset) -> Dataset:
    """
    Merge multiple datasets into one.

    Args:
        *datasets: Datasets to merge

    Returns:
        Combined dataset
    """
    from torch.utils.data import ConcatDataset

    return ConcatDataset(list(datasets))


def subsample_dataset(
    dataset: Dataset,
    fraction: float = 0.1,
    seed: int = 42,
) -> Subset:
    """
    Randomly subsample a fraction of the dataset.

    Args:
        dataset: Dataset to subsample
        fraction: Fraction to keep (0-1)
        seed: Random seed

    Returns:
        Subsampled subset
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"Fraction must be in (0, 1], got {fraction}")

    n_total = len(dataset)
    n_keep = int(n_total * fraction)

    # Random selection
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator)[:n_keep].tolist()

    return Subset(dataset, indices)
