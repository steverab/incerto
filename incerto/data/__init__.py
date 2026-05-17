"""
Data utilities for uncertainty quantification.

This module provides:
- Standard vision datasets with consistent splits (requires ``incerto[vision]``)
- OOD detection benchmarks (requires ``incerto[vision]``)
- Data loading utilities
- Dataset manipulation tools
"""

# Always available — pure torch / numpy, no extras required.
from .loaders import (
    InfiniteDataLoader,
    create_balanced_dataloader,
    create_calibration_loaders,
    create_dataloaders,
    create_ood_dataloader,
    get_dataloader_stats,
)
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

# Torchvision-dependent — only exposed when ``incerto[vision]`` is installed.
# Attempting to use these without the extra raises a clear ImportError at the
# call site (see ``vision.py`` / ``ood_benchmarks.py``).
# The ``import X as X`` form is the standard re-export idiom that tools
# (ruff F401, pyright) recognise so they don't flag these as unused.
try:
    from .ood_benchmarks import CIFAR10_vs_CIFAR100 as CIFAR10_vs_CIFAR100
    from .ood_benchmarks import CIFAR10_vs_SVHN as CIFAR10_vs_SVHN
    from .ood_benchmarks import CorruptedDataOOD as CorruptedDataOOD
    from .ood_benchmarks import MNIST_vs_FashionMNIST as MNIST_vs_FashionMNIST
    from .ood_benchmarks import MNIST_vs_NotMNIST as MNIST_vs_NotMNIST
    from .ood_benchmarks import OODBenchmark as OODBenchmark
    from .ood_benchmarks import SubclassOOD as SubclassOOD
    from .ood_benchmarks import get_ood_benchmark as get_ood_benchmark
    from .vision import CIFAR10 as CIFAR10
    from .vision import CIFAR100 as CIFAR100
    from .vision import MNIST as MNIST
    from .vision import SVHN as SVHN
    from .vision import FashionMNIST as FashionMNIST
    from .vision import VisionDataset as VisionDataset

    _VISION_NAMES = [
        "VisionDataset",
        "MNIST",
        "FashionMNIST",
        "CIFAR10",
        "CIFAR100",
        "SVHN",
        "OODBenchmark",
        "MNIST_vs_FashionMNIST",
        "CIFAR10_vs_CIFAR100",
        "CIFAR10_vs_SVHN",
        "MNIST_vs_NotMNIST",
        "SubclassOOD",
        "CorruptedDataOOD",
        "get_ood_benchmark",
    ]
except ImportError:
    _VISION_NAMES = []


__all__ = [
    # Data loaders (always available)
    "create_dataloaders",
    "create_balanced_dataloader",
    "create_ood_dataloader",
    "create_calibration_loaders",
    "InfiniteDataLoader",
    "get_dataloader_stats",
    # Dataset utilities (always available)
    "split_dataset",
    "filter_dataset_by_class",
    "get_class_balanced_subset",
    "compute_dataset_statistics",
    "create_imbalanced_dataset",
    "TransformDataset",
    "LabelNoiseDataset",
    "merge_datasets",
    "subsample_dataset",
    # Vision (require incerto[vision]; populated only when installed)
    *_VISION_NAMES,
]
