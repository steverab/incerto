"""
Vision datasets for uncertainty quantification benchmarks.

Provides standardized access to common vision datasets used in
uncertainty quantification research.
"""

from __future__ import annotations
import torch
from torch.utils.data import Dataset, Subset

try:
    from torchvision import datasets, transforms
except ImportError:
    raise ImportError(
        "torchvision is required for incerto.data.vision. "
        "Install it with: pip install incerto[vision]"
    )
from typing import Tuple
from pathlib import Path
from .utils import TransformDataset


class VisionDataset:
    """
    Base class for vision datasets with standardized splits.

    Provides train/val/test splits and optional transformations.
    """

    def __init__(
        self,
        root: str | Path = "./data",
        val_split: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize vision dataset.

        Args:
            root: Root directory for dataset storage
            val_split: Fraction of training data to use for validation
            seed: Random seed for reproducible splits
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.val_split = val_split
        self.seed = seed

    def _split_train_val(self, train_dataset: Dataset) -> Tuple[Subset, Subset]:
        """Split training dataset into train/val."""
        n_train = len(train_dataset)
        n_val = int(n_train * self.val_split)
        n_train_actual = n_train - n_val

        # Create reproducible split
        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(n_train, generator=generator).tolist()

        train_indices = indices[:n_train_actual]
        val_indices = indices[n_train_actual:]

        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)

        return train_subset, val_subset


class MNIST(VisionDataset):
    """
    MNIST dataset with standardized splits.

    28x28 grayscale images of handwritten digits (0-9).
    60,000 training + 10,000 test images.
    """

    def __init__(
        self,
        root: str | Path = "./data",
        val_split: float = 0.1,
        seed: int = 42,
        normalize: bool = True,
    ):
        super().__init__(root, val_split, seed)
        self.normalize = normalize

    def get_transforms(self, train: bool = False) -> transforms.Compose:
        """Get data transformations."""
        transform_list = [transforms.ToTensor()]

        if self.normalize:
            # MNIST mean and std
            transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))

        return transforms.Compose(transform_list)

    def get_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Get train, validation, and test datasets.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        train_transform = self.get_transforms(train=True)
        test_transform = self.get_transforms(train=False)

        # Load without transforms so train/val get different pipelines
        full_train = datasets.MNIST(
            root=self.root,
            train=True,
            download=True,
        )

        # Load test set
        test_dataset = datasets.MNIST(
            root=self.root,
            train=False,
            download=True,
            transform=test_transform,
        )

        # Split first, then apply appropriate transforms
        train_subset, val_subset = self._split_train_val(full_train)
        train_dataset = TransformDataset(train_subset, train_transform)
        val_dataset = TransformDataset(val_subset, test_transform)

        return train_dataset, val_dataset, test_dataset


class FashionMNIST(VisionDataset):
    """
    Fashion-MNIST dataset with standardized splits.

    28x28 grayscale images of fashion items (10 classes).
    60,000 training + 10,000 test images.
    """

    def __init__(
        self,
        root: str | Path = "./data",
        val_split: float = 0.1,
        seed: int = 42,
        normalize: bool = True,
    ):
        super().__init__(root, val_split, seed)
        self.normalize = normalize

    def get_transforms(self, train: bool = False) -> transforms.Compose:
        """Get data transformations."""
        transform_list = [transforms.ToTensor()]

        if self.normalize:
            # Fashion-MNIST mean and std
            transform_list.append(transforms.Normalize((0.2860,), (0.3530,)))

        return transforms.Compose(transform_list)

    def get_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Get train, validation, and test datasets."""
        train_transform = self.get_transforms(train=True)
        test_transform = self.get_transforms(train=False)

        # Load without transforms so train/val get different pipelines
        full_train = datasets.FashionMNIST(
            root=self.root,
            train=True,
            download=True,
        )

        test_dataset = datasets.FashionMNIST(
            root=self.root,
            train=False,
            download=True,
            transform=test_transform,
        )

        # Split first, then apply appropriate transforms
        train_subset, val_subset = self._split_train_val(full_train)
        train_dataset = TransformDataset(train_subset, train_transform)
        val_dataset = TransformDataset(val_subset, test_transform)

        return train_dataset, val_dataset, test_dataset


class CIFAR10(VisionDataset):
    """
    CIFAR-10 dataset with standardized splits.

    32x32 color images in 10 classes.
    50,000 training + 10,000 test images.
    """

    def __init__(
        self,
        root: str | Path = "./data",
        val_split: float = 0.1,
        seed: int = 42,
        normalize: bool = True,
        augmentation: bool = True,
    ):
        super().__init__(root, val_split, seed)
        self.normalize = normalize
        self.augmentation = augmentation

    def get_transforms(self, train: bool = False) -> transforms.Compose:
        """Get data transformations."""
        transform_list = []

        if train and self.augmentation:
            transform_list.extend(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            )

        transform_list.append(transforms.ToTensor())

        if self.normalize:
            # CIFAR-10 mean and std
            transform_list.append(
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616),
                )
            )

        return transforms.Compose(transform_list)

    def get_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Get train, validation, and test datasets."""
        train_transform = self.get_transforms(train=True)
        test_transform = self.get_transforms(train=False)

        # Load without transforms so train/val get different pipelines
        full_train = datasets.CIFAR10(
            root=self.root,
            train=True,
            download=True,
        )

        test_dataset = datasets.CIFAR10(
            root=self.root,
            train=False,
            download=True,
            transform=test_transform,
        )

        # Split first, then apply appropriate transforms
        train_subset, val_subset = self._split_train_val(full_train)
        train_dataset = TransformDataset(train_subset, train_transform)
        val_dataset = TransformDataset(val_subset, test_transform)

        return train_dataset, val_dataset, test_dataset


class CIFAR100(VisionDataset):
    """
    CIFAR-100 dataset with standardized splits.

    32x32 color images in 100 classes.
    50,000 training + 10,000 test images.
    """

    def __init__(
        self,
        root: str | Path = "./data",
        val_split: float = 0.1,
        seed: int = 42,
        normalize: bool = True,
        augmentation: bool = True,
    ):
        super().__init__(root, val_split, seed)
        self.normalize = normalize
        self.augmentation = augmentation

    def get_transforms(self, train: bool = False) -> transforms.Compose:
        """Get data transformations."""
        transform_list = []

        if train and self.augmentation:
            transform_list.extend(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            )

        transform_list.append(transforms.ToTensor())

        if self.normalize:
            # CIFAR-100 mean and std
            transform_list.append(
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408),
                    (0.2675, 0.2565, 0.2761),
                )
            )

        return transforms.Compose(transform_list)

    def get_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Get train, validation, and test datasets."""
        train_transform = self.get_transforms(train=True)
        test_transform = self.get_transforms(train=False)

        # Load without transforms so train/val get different pipelines
        full_train = datasets.CIFAR100(
            root=self.root,
            train=True,
            download=True,
        )

        test_dataset = datasets.CIFAR100(
            root=self.root,
            train=False,
            download=True,
            transform=test_transform,
        )

        # Split first, then apply appropriate transforms
        train_subset, val_subset = self._split_train_val(full_train)
        train_dataset = TransformDataset(train_subset, train_transform)
        val_dataset = TransformDataset(val_subset, test_transform)

        return train_dataset, val_dataset, test_dataset


class SVHN(VisionDataset):
    """
    SVHN (Street View House Numbers) dataset.

    32x32 color images of house numbers.
    73,257 training + 26,032 test images.
    """

    def __init__(
        self,
        root: str | Path = "./data",
        val_split: float = 0.1,
        seed: int = 42,
        normalize: bool = True,
    ):
        super().__init__(root, val_split, seed)
        self.normalize = normalize

    def get_transforms(self, train: bool = False) -> transforms.Compose:
        """Get data transformations."""
        transform_list = [transforms.ToTensor()]

        if self.normalize:
            # SVHN mean and std
            transform_list.append(
                transforms.Normalize(
                    (0.4377, 0.4438, 0.4728),
                    (0.1980, 0.2010, 0.1970),
                )
            )

        return transforms.Compose(transform_list)

    def get_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Get train, validation, and test datasets."""
        train_transform = self.get_transforms(train=True)
        test_transform = self.get_transforms(train=False)

        # Load without transforms so train/val get different pipelines
        full_train = datasets.SVHN(
            root=self.root,
            split="train",
            download=True,
        )

        test_dataset = datasets.SVHN(
            root=self.root,
            split="test",
            download=True,
            transform=test_transform,
        )

        # Split first, then apply appropriate transforms
        train_subset, val_subset = self._split_train_val(full_train)
        train_dataset = TransformDataset(train_subset, train_transform)
        val_dataset = TransformDataset(val_subset, test_transform)

        return train_dataset, val_dataset, test_dataset
