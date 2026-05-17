"""Tests for incerto.data.utils dataset manipulation helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from incerto.data.utils import (
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


class LabeledTensorDataset(Dataset):
    """TensorDataset with a `.targets` attribute for _get_targets compatibility."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y
        self.targets = y.tolist()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], int(self.y[idx])


@pytest.fixture
def balanced_dataset():
    torch.manual_seed(0)
    n_per_class = 30
    X = torch.randn(n_per_class * 4, 5)
    y = torch.cat([torch.full((n_per_class,), c, dtype=torch.long) for c in range(4)])
    return LabeledTensorDataset(X, y)


@pytest.fixture
def imbalanced_dataset():
    """20 of class 0, 5 of class 1, 80 of class 2."""
    torch.manual_seed(1)
    X = torch.randn(105, 5)
    y = torch.cat(
        [
            torch.zeros(20, dtype=torch.long),
            torch.ones(5, dtype=torch.long),
            torch.full((80,), 2, dtype=torch.long),
        ]
    )
    return LabeledTensorDataset(X, y)


class TestSplitDataset:
    def test_basic_split(self, balanced_dataset):
        train, val, test = split_dataset(balanced_dataset, [0.7, 0.15, 0.15])
        assert len(train) + len(val) + len(test) == len(balanced_dataset)
        assert len(train) > len(val)

    def test_invalid_splits_raise(self, balanced_dataset):
        with pytest.raises(ValueError, match="sum to 1"):
            split_dataset(balanced_dataset, [0.5, 0.2])

    def test_deterministic_with_seed(self, balanced_dataset):
        a = split_dataset(balanced_dataset, [0.5, 0.5], seed=7)
        b = split_dataset(balanced_dataset, [0.5, 0.5], seed=7)
        assert a[0].indices == b[0].indices

    def test_two_way_split_no_overlap(self, balanced_dataset):
        a, b = split_dataset(balanced_dataset, [0.6, 0.4])
        assert set(a.indices).isdisjoint(set(b.indices))


class TestFilterByClass:
    def test_keep_subset(self, balanced_dataset):
        filtered = filter_dataset_by_class(balanced_dataset, classes=[0, 2])
        targets = np.array([balanced_dataset.targets[i] for i in filtered.indices])
        assert set(targets.tolist()) == {0, 2}

    def test_invert(self, balanced_dataset):
        filtered = filter_dataset_by_class(balanced_dataset, classes=[0], invert=True)
        targets = np.array([balanced_dataset.targets[i] for i in filtered.indices])
        assert 0 not in set(targets.tolist())


class TestClassBalancedSubset:
    def test_balanced_count(self, balanced_dataset):
        subset = get_class_balanced_subset(balanced_dataset, samples_per_class=10)
        targets = np.array([balanced_dataset.targets[i] for i in subset.indices])
        _, counts = np.unique(targets, return_counts=True)
        assert (counts == 10).all()

    def test_raises_when_too_many_requested(self, imbalanced_dataset):
        # Class 1 only has 5 samples
        with pytest.raises(ValueError, match="only"):
            get_class_balanced_subset(imbalanced_dataset, samples_per_class=10)


class TestComputeStatistics:
    def test_balanced_stats(self, balanced_dataset):
        stats = compute_dataset_statistics(balanced_dataset)
        assert stats["size"] == len(balanced_dataset)
        assert stats["num_classes"] == 4
        assert stats["class_balance_ratio"] == 1.0
        assert stats["min_class_size"] == stats["max_class_size"]

    def test_imbalanced_stats(self, imbalanced_dataset):
        stats = compute_dataset_statistics(imbalanced_dataset)
        assert stats["num_classes"] == 3
        assert stats["min_class_size"] == 5
        assert stats["max_class_size"] == 80
        assert stats["class_balance_ratio"] == pytest.approx(5 / 80)


class TestCreateImbalanced:
    def test_imbalance_ratio_respected(self, balanced_dataset):
        imb = create_imbalanced_dataset(
            balanced_dataset,
            imbalance_ratio=0.1,
            minority_classes=[0, 1],
            seed=0,
        )
        targets = np.array([balanced_dataset.targets[i] for i in imb.indices])
        _, counts = np.unique(targets, return_counts=True)
        # The minority classes should be ~10x smaller than majority
        assert counts.min() * 5 < counts.max()


class TestTransformDataset:
    def test_transform_applied(self, balanced_dataset):
        ds = TransformDataset(balanced_dataset, transform=lambda x: x * 2.0)
        original_x, _ = balanced_dataset[0]
        transformed_x, _ = ds[0]
        assert torch.allclose(transformed_x, original_x * 2.0)

    def test_length_unchanged(self, balanced_dataset):
        ds = TransformDataset(balanced_dataset, transform=lambda x: x)
        assert len(ds) == len(balanced_dataset)


class TestLabelNoiseDataset:
    def test_noise_rate_zero_preserves_labels(self, balanced_dataset):
        ds = LabelNoiseDataset(balanced_dataset, noise_rate=0.0, seed=0)
        for i in range(len(ds)):
            _, y = ds[i]
            assert y == balanced_dataset.targets[i]

    def test_noise_rate_changes_some_labels(self, balanced_dataset):
        ds = LabelNoiseDataset(balanced_dataset, noise_rate=0.5, seed=0)
        changed = 0
        for i in range(len(ds)):
            _, y = ds[i]
            if y != balanced_dataset.targets[i]:
                changed += 1
        n = len(ds)
        # Approximately 50% should be flipped
        assert n * 0.3 < changed < n * 0.7

    def test_labels_remain_valid_class_indices(self, balanced_dataset):
        ds = LabelNoiseDataset(balanced_dataset, noise_rate=0.5, num_classes=4)
        for i in range(len(ds)):
            _, y = ds[i]
            assert 0 <= y < 4


class TestMergeDatasets:
    def test_lengths_add(self, balanced_dataset, imbalanced_dataset):
        merged = merge_datasets(balanced_dataset, imbalanced_dataset)
        assert len(merged) == len(balanced_dataset) + len(imbalanced_dataset)


class TestSubsample:
    def test_fraction_size(self, balanced_dataset):
        sub = subsample_dataset(balanced_dataset, fraction=0.25, seed=0)
        assert len(sub) == int(0.25 * len(balanced_dataset))

    def test_invalid_fraction_raises(self, balanced_dataset):
        with pytest.raises(ValueError):
            subsample_dataset(balanced_dataset, fraction=0.0)
        with pytest.raises(ValueError):
            subsample_dataset(balanced_dataset, fraction=1.5)

    def test_deterministic_seed(self, balanced_dataset):
        a = subsample_dataset(balanced_dataset, fraction=0.5, seed=11)
        b = subsample_dataset(balanced_dataset, fraction=0.5, seed=11)
        assert a.indices == b.indices
