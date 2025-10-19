"""
Data utilities for uncertainty quantification.

This module provides:
- Standard vision datasets with consistent splits
- OOD detection benchmarks
- Data loading utilities
- Dataset manipulation tools
"""

# Vision datasets
from .vision import (
    VisionDataset,
    MNIST,
    FashionMNIST,
    CIFAR10,
    CIFAR100,
    SVHN,
)

# OOD benchmarks
from .ood_benchmarks import (
    OODBenchmark,
    MNIST_vs_FashionMNIST,
    CIFAR10_vs_CIFAR100,
    CIFAR10_vs_SVHN,
    MNIST_vs_NotMNIST,
    SubclassOOD,
    CorruptedDataOOD,
    get_ood_benchmark,
)

# Data loaders
from .loaders import (
    create_dataloaders,
    create_balanced_dataloader,
    create_ood_dataloader,
    create_calibration_loaders,
    InfiniteDataLoader,
    get_dataloader_stats,
)

# Dataset utilities
from .utils import (
    split_dataset,
    filter_dataset_by_class,
    get_class_balanced_subset,
    compute_dataset_statistics,
    create_imbalanced_dataset,
    TransformDataset,
    LabelNoiseDataset,
    merge_datasets,
    subsample_dataset,
)

__all__ = [
    # Vision datasets
    "VisionDataset",
    "MNIST",
    "FashionMNIST",
    "CIFAR10",
    "CIFAR100",
    "SVHN",
    # OOD benchmarks
    "OODBenchmark",
    "MNIST_vs_FashionMNIST",
    "CIFAR10_vs_CIFAR100",
    "CIFAR10_vs_SVHN",
    "MNIST_vs_NotMNIST",
    "SubclassOOD",
    "CorruptedDataOOD",
    "get_ood_benchmark",
    # Data loaders
    "create_dataloaders",
    "create_balanced_dataloader",
    "create_ood_dataloader",
    "create_calibration_loaders",
    "InfiniteDataLoader",
    "get_dataloader_stats",
    # Dataset utilities
    "split_dataset",
    "filter_dataset_by_class",
    "get_class_balanced_subset",
    "compute_dataset_statistics",
    "create_imbalanced_dataset",
    "TransformDataset",
    "LabelNoiseDataset",
    "merge_datasets",
    "subsample_dataset",
]
