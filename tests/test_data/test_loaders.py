"""Tests for incerto.data.loaders."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from incerto.data.loaders import (
    InfiniteDataLoader,
    create_balanced_dataloader,
    create_calibration_loaders,
    create_dataloaders,
    create_ood_dataloader,
    get_dataloader_stats,
)


class LabeledTensorDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y
        self.targets = y.tolist()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], int(self.y[idx])


@pytest.fixture
def imbalanced_dataset():
    """10 of class 0, 60 of class 1."""
    torch.manual_seed(0)
    X = torch.randn(70, 5)
    y = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(60, dtype=torch.long)])
    return LabeledTensorDataset(X, y)


@pytest.fixture
def balanced_dataset():
    torch.manual_seed(0)
    X = torch.randn(60, 5)
    y = torch.cat([torch.full((20,), c, dtype=torch.long) for c in range(3)])
    return LabeledTensorDataset(X, y)


class TestCreateDataloaders:
    def test_train_only(self, balanced_dataset):
        train, val, test = create_dataloaders(
            balanced_dataset, batch_size=8, num_workers=0, pin_memory=False
        )
        assert train is not None
        assert val is None
        assert test is None

    def test_all_three(self, balanced_dataset):
        train, val, test = create_dataloaders(
            balanced_dataset,
            val_dataset=balanced_dataset,
            test_dataset=balanced_dataset,
            batch_size=8,
            num_workers=0,
            pin_memory=False,
        )
        assert train is not None
        assert val is not None
        assert test is not None

    def test_train_shuffles_val_does_not(self, balanced_dataset):
        train, val, _ = create_dataloaders(
            balanced_dataset,
            val_dataset=balanced_dataset,
            batch_size=8,
            num_workers=0,
            pin_memory=False,
        )
        # val/test loaders use SequentialSampler; train uses RandomSampler
        from torch.utils.data import RandomSampler, SequentialSampler

        assert isinstance(train.sampler, RandomSampler)
        assert isinstance(val.sampler, SequentialSampler)


class TestBalancedDataloader:
    def test_balances_imbalanced(self, imbalanced_dataset):
        loader = create_balanced_dataloader(
            imbalanced_dataset, batch_size=32, num_workers=0, pin_memory=False
        )
        # Draw a few batches and check class balance is approximately 50/50
        counts = {0: 0, 1: 0}
        for i, (_, y) in enumerate(loader):
            for label in y.tolist():
                counts[label] += 1
            if i >= 5:
                break
        ratio = counts[0] / max(counts[1], 1)
        # Should be balanced (~1.0), original was 1:6
        assert 0.5 < ratio < 2.0


class TestCalibrationLoaders:
    def test_split_sizes(self, balanced_dataset):
        train, calib, test = create_calibration_loaders(
            balanced_dataset,
            balanced_dataset,
            calib_split=0.4,
            batch_size=8,
            num_workers=0,
            pin_memory=False,
        )
        # 60 samples total; calib = 24, train = 36
        assert len(train.dataset) == 36
        assert len(calib.dataset) == 24

    def test_disjoint_indices(self, balanced_dataset):
        train, calib, _ = create_calibration_loaders(
            balanced_dataset,
            balanced_dataset,
            calib_split=0.5,
            batch_size=8,
            num_workers=0,
            pin_memory=False,
        )
        train_idx = set(train.dataset.indices)
        calib_idx = set(calib.dataset.indices)
        assert train_idx.isdisjoint(calib_idx)


class TestInfiniteDataLoader:
    def test_wraps_around(self, balanced_dataset):
        from torch.utils.data import DataLoader

        loader = DataLoader(balanced_dataset, batch_size=16, shuffle=False)
        n_batches = len(loader)
        inf = InfiniteDataLoader(loader)
        # Iterate beyond loader length
        for _ in range(n_batches * 3 + 1):
            batch = next(inf)
            assert batch is not None

    def test_len(self, balanced_dataset):
        from torch.utils.data import DataLoader

        loader = DataLoader(balanced_dataset, batch_size=16)
        inf = InfiniteDataLoader(loader)
        assert len(inf) == len(loader)


class TestGetDataloaderStats:
    def test_basic_keys(self, balanced_dataset):
        from torch.utils.data import DataLoader

        loader = DataLoader(balanced_dataset, batch_size=16, num_workers=0, pin_memory=False)
        stats = get_dataloader_stats(loader)
        assert stats["num_batches"] == len(loader)
        assert stats["batch_size"] == 16


class TestOODDataloader:
    def test_creates_loader(self, balanced_dataset):
        loader = create_ood_dataloader(
            balanced_dataset,
            balanced_dataset,
            batch_size=8,
            num_workers=0,
            pin_memory=False,
        )
        # Each iteration should produce a tuple of (id_batch, ood_batch)
        batch = next(iter(loader))
        assert batch is not None
