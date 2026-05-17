"""
Data utilities for uncertainty quantification.

This module provides:
- Standard vision datasets with consistent splits
- OOD detection benchmarks
- Data loading utilities
- Dataset manipulation tools
"""

# Vision datasets
# Data loaders
from .loaders import (
    InfiniteDataLoader,
    create_balanced_dataloader,
    create_calibration_loaders,
    create_dataloaders,
    create_ood_dataloader,
    get_dataloader_stats,
)

# OOD benchmarks
from .ood_benchmarks import (
    CIFAR10_vs_CIFAR100,
    CIFAR10_vs_SVHN,
    CorruptedDataOOD,
    MNIST_vs_FashionMNIST,
    MNIST_vs_NotMNIST,
    OODBenchmark,
    SubclassOOD,
    get_ood_benchmark,
)

# Dataset utilities
from .utils import (
    LabelNoiseDataset,
    TransformDataset,
    compute_dataset_statistics,
    create_imbalanced_dataset,
    filter_dataset_by_class,
    get_class_balanced_subset,
    merge_datasets,
    split_dataset,
    subsample_dataset,
)
from .vision import (
    CIFAR10,
    CIFAR100,
    MNIST,
    SVHN,
    FashionMNIST,
    VisionDataset,
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
