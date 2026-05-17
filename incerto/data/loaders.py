"""
Data loading utilities for uncertainty quantification.

Provides standardized data loaders and sampling strategies.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .utils import _get_targets


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle_train: bool = True,
) -> tuple[DataLoader, ...]:
    """
    Create standard data loaders for train/val/test.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        test_dataset: Test dataset (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        shuffle_train: Whether to shuffle training data

    Returns:
        Tuple of DataLoaders (train, val, test)
        If val or test is None, returns None for that loader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader, test_loader


def create_balanced_dataloader(
    dataset: Dataset,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create data loader with balanced class sampling.

    Ensures equal representation of all classes in each batch.

    Args:
        dataset: Dataset with 'targets' or 'labels' attribute
        batch_size: Batch size
        num_workers: Number of workers
        pin_memory: Whether to pin memory

    Returns:
        DataLoader with balanced sampling
    """
    targets = _get_targets(dataset)

    # Compute class weights (use np.unique to avoid division by zero with non-contiguous classes)
    unique_classes, class_counts = np.unique(targets, return_counts=True)
    class_weight_map = dict(zip(unique_classes, 1.0 / class_counts, strict=False))
    sample_weights = np.array([class_weight_map[t] for t in targets])

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Create loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader


def create_ood_dataloader(
    id_dataset: Dataset,
    ood_dataset: Dataset,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    mixed: bool = False,
    mix_ratio: float = 0.5,
) -> tuple[DataLoader, DataLoader] | DataLoader:
    """
    Create data loaders for OOD detection evaluation.

    Args:
        id_dataset: In-distribution dataset
        ood_dataset: Out-of-distribution dataset
        batch_size: Batch size
        num_workers: Number of workers
        pin_memory: Whether to pin memory
        mixed: If True, create single mixed loader; if False, separate loaders
        mix_ratio: Ratio of ID samples in mixed loader (only if mixed=True)

    Returns:
        If mixed=False: Tuple of (id_loader, ood_loader)
        If mixed=True: Single mixed loader
    """
    if not mixed:
        # Separate loaders
        id_loader = DataLoader(
            id_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        ood_loader = DataLoader(
            ood_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return id_loader, ood_loader

    else:
        # Mixed loader
        from torch.utils.data import ConcatDataset

        # Combine datasets
        combined = ConcatDataset([id_dataset, ood_dataset])

        # Create weights for mixing
        n_id = len(id_dataset)
        n_ood = len(ood_dataset)

        # Weight to achieve desired ratio
        id_weight = mix_ratio / n_id
        ood_weight = (1 - mix_ratio) / n_ood

        weights = [id_weight] * n_id + [ood_weight] * n_ood

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )

        mixed_loader = DataLoader(
            combined,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return mixed_loader


class InfiniteDataLoader:
    """
    Wrapper for infinite data loading.

    Useful for training with unequal dataset sizes or continuous sampling.
    """

    def __init__(self, dataloader: DataLoader):
        """
        Initialize infinite data loader.

        Args:
            dataloader: Base DataLoader to wrap
        """
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            # Restart iterator
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch

    def __len__(self):
        return len(self.dataloader)


def create_calibration_loaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    calib_split: float = 0.5,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create loaders for calibration experiments.

    Splits training data into train/calibration sets, plus test set.

    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        calib_split: Fraction of training data to use for calibration
        batch_size: Batch size
        num_workers: Number of workers
        pin_memory: Whether to pin memory
        seed: Random seed

    Returns:
        Tuple of (train_loader, calib_loader, test_loader)
    """
    from torch.utils.data import Subset

    # Split training data
    n_train = len(train_dataset)
    n_calib = int(n_train * calib_split)
    n_train_actual = n_train - n_calib

    # Create reproducible split
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_train, generator=generator).tolist()

    train_indices = indices[:n_train_actual]
    calib_indices = indices[n_train_actual:]

    train_subset = Subset(train_dataset, train_indices)
    calib_subset = Subset(train_dataset, calib_indices)

    # Create loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    calib_loader = DataLoader(
        calib_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, calib_loader, test_loader


def get_dataloader_stats(dataloader: DataLoader) -> dict:
    """
    Compute statistics about a DataLoader.

    Args:
        dataloader: DataLoader to analyze

    Returns:
        Dictionary with statistics
    """
    stats = {
        "num_batches": len(dataloader),
        "batch_size": dataloader.batch_size,
        "num_workers": dataloader.num_workers,
        "pin_memory": dataloader.pin_memory,
    }

    # Try to get dataset size
    if hasattr(dataloader.dataset, "__len__"):
        stats["dataset_size"] = len(dataloader.dataset)

    # Try to get class distribution
    try:
        targets = _get_targets(dataloader.dataset)
        unique, counts = np.unique(targets, return_counts=True)
        stats["num_classes"] = len(unique)
        stats["class_distribution"] = dict(zip(unique.tolist(), counts.tolist(), strict=False))
    except ValueError:
        pass  # Dataset has no targets/labels attribute

    return stats
