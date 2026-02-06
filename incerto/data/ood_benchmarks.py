"""
Standard OOD detection benchmarks.

Provides utilities for creating common OOD detection benchmark scenarios
used in uncertainty quantification research.
"""

from __future__ import annotations
import torch
from torch.utils.data import Dataset, Subset

try:
    from torchvision import datasets, transforms
except ImportError:
    raise ImportError(
        "torchvision is required for incerto.data.ood_benchmarks. "
        "Install it with: pip install incerto[vision]"
    )
from typing import Tuple, List
from pathlib import Path
import numpy as np


class OODBenchmark:
    """
    Base class for OOD detection benchmarks.

    Provides in-distribution (ID) and out-of-distribution (OOD) datasets
    for evaluation.
    """

    def __init__(self, root: str | Path = "./data"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get_datasets(self) -> Tuple[Dataset, Dataset]:
        """
        Get ID and OOD datasets.

        Returns:
            Tuple of (id_dataset, ood_dataset)
        """
        raise NotImplementedError


class MNIST_vs_FashionMNIST(OODBenchmark):
    """
    MNIST (ID) vs Fashion-MNIST (OOD) benchmark.

    A common near-OOD benchmark where both datasets have similar
    structure but different semantics.
    """

    def __init__(
        self,
        root: str | Path = "./data",
        normalize: bool = True,
    ):
        super().__init__(root)
        self.normalize = normalize

    def get_transforms(self) -> transforms.Compose:
        """Get data transformations."""
        transform_list = [transforms.ToTensor()]

        if self.normalize:
            # Use MNIST normalization for both
            transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))

        return transforms.Compose(transform_list)

    def get_datasets(self) -> Tuple[Dataset, Dataset]:
        """Get MNIST (ID) and Fashion-MNIST (OOD) test sets."""
        transform = self.get_transforms()

        id_dataset = datasets.MNIST(
            root=self.root,
            train=False,
            download=True,
            transform=transform,
        )

        ood_dataset = datasets.FashionMNIST(
            root=self.root,
            train=False,
            download=True,
            transform=transform,
        )

        return id_dataset, ood_dataset


class CIFAR10_vs_CIFAR100(OODBenchmark):
    """
    CIFAR-10 (ID) vs CIFAR-100 (OOD) benchmark.

    Near-OOD benchmark with similar image statistics but different classes.
    """

    def __init__(
        self,
        root: str | Path = "./data",
        normalize: bool = True,
    ):
        super().__init__(root)
        self.normalize = normalize

    def get_transforms(self) -> transforms.Compose:
        """Get data transformations."""
        transform_list = [transforms.ToTensor()]

        if self.normalize:
            # Use CIFAR-10 normalization
            transform_list.append(
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616),
                )
            )

        return transforms.Compose(transform_list)

    def get_datasets(self) -> Tuple[Dataset, Dataset]:
        """Get CIFAR-10 (ID) and CIFAR-100 (OOD) test sets."""
        transform = self.get_transforms()

        id_dataset = datasets.CIFAR10(
            root=self.root,
            train=False,
            download=True,
            transform=transform,
        )

        ood_dataset = datasets.CIFAR100(
            root=self.root,
            train=False,
            download=True,
            transform=transform,
        )

        return id_dataset, ood_dataset


class CIFAR10_vs_SVHN(OODBenchmark):
    """
    CIFAR-10 (ID) vs SVHN (OOD) benchmark.

    Far-OOD benchmark with different image statistics.
    """

    def __init__(
        self,
        root: str | Path = "./data",
        normalize: bool = True,
    ):
        super().__init__(root)
        self.normalize = normalize

    def get_transforms(self) -> transforms.Compose:
        """Get data transformations."""
        transform_list = [transforms.ToTensor()]

        if self.normalize:
            # Use CIFAR-10 normalization
            transform_list.append(
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616),
                )
            )

        return transforms.Compose(transform_list)

    def get_datasets(self) -> Tuple[Dataset, Dataset]:
        """Get CIFAR-10 (ID) and SVHN (OOD) test sets."""
        transform = self.get_transforms()

        id_dataset = datasets.CIFAR10(
            root=self.root,
            train=False,
            download=True,
            transform=transform,
        )

        ood_dataset = datasets.SVHN(
            root=self.root,
            split="test",
            download=True,
            transform=transform,
        )

        return id_dataset, ood_dataset


class MNIST_vs_NotMNIST(OODBenchmark):
    """
    MNIST (ID) vs NotMNIST (OOD) benchmark.

    NotMNIST contains letters instead of digits.
    Note: NotMNIST requires manual download.
    """

    def __init__(
        self,
        root: str | Path = "./data",
        normalize: bool = True,
    ):
        super().__init__(root)
        self.normalize = normalize

    def get_transforms(self) -> transforms.Compose:
        """Get data transformations."""
        transform_list = [transforms.ToTensor()]

        if self.normalize:
            transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))

        return transforms.Compose(transform_list)

    def get_datasets(self) -> Tuple[Dataset, Dataset]:
        """Get MNIST (ID) and NotMNIST (OOD) test sets."""
        transform = self.get_transforms()

        id_dataset = datasets.MNIST(
            root=self.root,
            train=False,
            download=True,
            transform=transform,
        )

        # NotMNIST is not in torchvision, would need custom implementation
        # For now, we'll use a placeholder
        # Users should download NotMNIST separately
        notmnist_path = self.root / "notMNIST"
        if notmnist_path.exists():
            ood_dataset = datasets.ImageFolder(
                root=notmnist_path,
                transform=transform,
            )
        else:
            raise FileNotFoundError(
                f"NotMNIST not found at {notmnist_path}. "
                "Please download NotMNIST dataset manually."
            )

        return id_dataset, ood_dataset


class SubclassOOD(OODBenchmark):
    """
    Create OOD benchmark by holding out specific classes.

    Example: Train on classes 0-7 of CIFAR-10, test with classes 8-9 as OOD.
    """

    def __init__(
        self,
        dataset_class,
        root: str | Path = "./data",
        id_classes: List[int] = None,
        ood_classes: List[int] = None,
        normalize: bool = True,
        **dataset_kwargs,
    ):
        """
        Initialize subclass OOD benchmark.

        Args:
            dataset_class: Dataset class (e.g., datasets.CIFAR10)
            root: Data root directory
            id_classes: List of class indices to use as ID
            ood_classes: List of class indices to use as OOD
            normalize: Whether to normalize
            **dataset_kwargs: Additional arguments for dataset
        """
        super().__init__(root)
        self.dataset_class = dataset_class
        self.id_classes = id_classes
        self.ood_classes = ood_classes
        self.normalize = normalize
        self.dataset_kwargs = dataset_kwargs

    def get_datasets(self) -> Tuple[Dataset, Dataset]:
        """Get ID and OOD subsets."""
        # Load full dataset
        full_dataset = self.dataset_class(
            root=self.root,
            train=False,
            download=True,
            **self.dataset_kwargs,
        )

        # Get targets
        if hasattr(full_dataset, "targets"):
            targets = np.array(full_dataset.targets)
        elif hasattr(full_dataset, "labels"):
            targets = np.array(full_dataset.labels)
        else:
            raise ValueError("Dataset must have 'targets' or 'labels' attribute")

        # Find ID and OOD indices
        id_indices = []
        ood_indices = []

        for idx, target in enumerate(targets):
            if self.id_classes is not None and target in self.id_classes:
                id_indices.append(idx)
            elif self.ood_classes is not None and target in self.ood_classes:
                ood_indices.append(idx)

        # Create subsets
        id_dataset = Subset(full_dataset, id_indices)
        ood_dataset = Subset(full_dataset, ood_indices)

        return id_dataset, ood_dataset


class CorruptedDataOOD(OODBenchmark):
    """
    Create OOD benchmark using corrupted versions of ID data.

    Applies various corruptions (noise, blur, etc.) to create OOD samples.
    """

    def __init__(
        self,
        dataset: Dataset,
        corruption_type: str = "gaussian_noise",
        severity: float = 0.5,
    ):
        """
        Initialize corrupted data OOD benchmark.

        Args:
            dataset: Base dataset
            corruption_type: Type of corruption to apply
            severity: Corruption severity (0-1)
        """
        # CorruptedDataOOD does not need a root directory; skip OODBenchmark.__init__
        self.root = None
        self.dataset = dataset
        self.corruption_type = corruption_type
        self.severity = severity

    def apply_corruption(self, image: torch.Tensor) -> torch.Tensor:
        """Apply corruption to image."""
        if self.corruption_type == "gaussian_noise":
            noise = torch.randn_like(image) * self.severity
            return torch.clamp(image + noise, 0, 1)

        elif self.corruption_type == "salt_pepper":
            mask = torch.rand_like(image)
            corrupted = image.clone()
            corrupted[mask < self.severity / 2] = 0
            corrupted[mask > 1 - self.severity / 2] = 1
            return corrupted

        elif self.corruption_type == "blur":
            # Simple box blur via average pooling
            kernel_size = int(self.severity * 10) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            padding = kernel_size // 2
            # (C, H, W) -> (1, C, H, W) for avg_pool2d
            blurred = torch.nn.functional.avg_pool2d(
                image.unsqueeze(0), kernel_size, stride=1, padding=padding
            ).squeeze(0)
            return blurred

        else:
            raise ValueError(f"Unknown corruption type: {self.corruption_type}")

    def get_datasets(self) -> Tuple[Dataset, Dataset]:
        """Get ID (clean) and OOD (corrupted) datasets."""
        # ID is the original dataset
        id_dataset = self.dataset

        # OOD is corrupted version
        class CorruptedDataset(Dataset):
            def __init__(self, base_dataset, corruption_fn):
                self.base_dataset = base_dataset
                self.corruption_fn = corruption_fn

            def __len__(self):
                return len(self.base_dataset)

            def __getitem__(self, idx):
                image, label = self.base_dataset[idx]
                corrupted_image = self.corruption_fn(image)
                return corrupted_image, label

        ood_dataset = CorruptedDataset(self.dataset, self.apply_corruption)

        return id_dataset, ood_dataset


def get_ood_benchmark(
    name: str,
    root: str | Path = "./data",
    **kwargs,
) -> OODBenchmark:
    """
    Factory function to get OOD benchmark by name.

    Args:
        name: Benchmark name
        root: Data root directory
        **kwargs: Additional arguments

    Returns:
        OODBenchmark instance

    Available benchmarks:
        - "mnist_vs_fmnist": MNIST vs Fashion-MNIST
        - "cifar10_vs_cifar100": CIFAR-10 vs CIFAR-100
        - "cifar10_vs_svhn": CIFAR-10 vs SVHN
        - "mnist_vs_notmnist": MNIST vs NotMNIST
    """
    benchmarks = {
        "mnist_vs_fmnist": MNIST_vs_FashionMNIST,
        "cifar10_vs_cifar100": CIFAR10_vs_CIFAR100,
        "cifar10_vs_svhn": CIFAR10_vs_SVHN,
        "mnist_vs_notmnist": MNIST_vs_NotMNIST,
    }

    if name not in benchmarks:
        raise ValueError(
            f"Unknown benchmark: {name}. " f"Available: {list(benchmarks.keys())}"
        )

    return benchmarks[name](root=root, **kwargs)
