"""
Tests for active learning utility functions.
"""

import pytest
import torch
import torch.nn as nn

from incerto.active.utils import (
    active_learning_loop,
    compute_diversity_penalty,
    greedy_k_center,
    split_labeled_unlabeled,
    subsample_for_efficiency,
)

# ---------------------------------------------------------------------------
# split_labeled_unlabeled
# ---------------------------------------------------------------------------


class TestSplitLabeledUnlabeled:
    def test_split_by_labels(self):
        data = torch.randn(10, 5)
        labels = torch.tensor([0, 1, -1, -1, 2, -1, 0, 1, -1, -1])

        x_lab, x_unlab, y_lab, lab_idx = split_labeled_unlabeled(data, labels=labels)

        assert len(x_lab) == 5  # indices 0,1,4,6,7
        assert len(x_unlab) == 5  # indices 2,3,5,8,9
        assert torch.all(y_lab >= 0)
        assert len(lab_idx) == 5

    def test_split_by_labeled_indices(self):
        data = torch.randn(10, 5)
        labeled_idx = torch.tensor([0, 2, 4])

        x_lab, x_unlab, y_lab, lab_idx = split_labeled_unlabeled(data, labeled_indices=labeled_idx)

        assert len(x_lab) == 3
        assert len(x_unlab) == 7
        assert y_lab is None
        assert torch.equal(lab_idx, labeled_idx)

    def test_split_by_unlabeled_indices(self):
        data = torch.randn(10, 5)
        unlabeled_idx = torch.tensor([1, 3, 5, 7, 9])

        x_lab, x_unlab, y_lab, lab_idx = split_labeled_unlabeled(
            data, unlabeled_indices=unlabeled_idx
        )

        assert len(x_lab) == 5
        assert len(x_unlab) == 5

    def test_no_args_raises(self):
        data = torch.randn(10, 5)
        with pytest.raises(ValueError, match="Must provide either labels or indices"):
            split_labeled_unlabeled(data)

    def test_labels_with_indices(self):
        """When labels and labeled_indices are both provided, indices win."""
        data = torch.randn(10, 5)
        labels = torch.arange(10).float()
        labeled_idx = torch.tensor([0, 1])

        x_lab, x_unlab, y_lab, lab_idx = split_labeled_unlabeled(
            data, labels=labels, labeled_indices=labeled_idx
        )

        assert len(x_lab) == 2
        assert y_lab is not None
        assert len(y_lab) == 2


# ---------------------------------------------------------------------------
# compute_diversity_penalty
# ---------------------------------------------------------------------------


class TestComputeDiversityPenalty:
    def test_empty_selection(self):
        features = torch.randn(10, 5)
        result = compute_diversity_penalty(torch.tensor([], dtype=torch.long), features)
        assert result.item() == 0.0

    def test_single_sample_min_distance(self):
        features = torch.randn(10, 5)
        result = compute_diversity_penalty(torch.tensor([0]), features, method="min_distance")
        assert result.item() == float("inf")

    def test_min_distance(self):
        features = torch.tensor([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
        selected = torch.tensor([0, 1, 2])
        result = compute_diversity_penalty(selected, features, method="min_distance")
        # min distance is between [0,0] and [1,0] = 1.0
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-5)

    def test_mean_distance(self):
        features = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        selected = torch.tensor([0, 1])
        result = compute_diversity_penalty(selected, features, method="mean_distance")
        # mean of d(0,1) + d(1,0) / 2 = (1+1)/2 = 1.0
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-5)

    def test_determinant(self):
        torch.manual_seed(42)
        features = torch.randn(10, 2)
        selected = torch.tensor([0, 1, 2, 3, 4])
        result = compute_diversity_penalty(selected, features, method="determinant")
        assert torch.isfinite(result)

    def test_unknown_method_raises(self):
        features = torch.randn(5, 3)
        with pytest.raises(ValueError, match="Unknown method"):
            compute_diversity_penalty(torch.tensor([0]), features, method="invalid")


# ---------------------------------------------------------------------------
# greedy_k_center
# ---------------------------------------------------------------------------


class TestGreedyKCenter:
    def test_basic_selection(self):
        torch.manual_seed(42)
        features = torch.randn(20, 5)
        selected = greedy_k_center(features, k=5)

        assert len(selected) == 5
        assert len(selected.unique()) == 5
        assert torch.all(selected >= 0) and torch.all(selected < 20)

    def test_with_initial_centers(self):
        torch.manual_seed(42)
        features = torch.randn(20, 5)
        centers = features[:2]
        selected = greedy_k_center(features, k=3, initial_centers=centers)

        assert len(selected) == 3

    def test_k_equals_n(self):
        features = torch.randn(5, 3)
        selected = greedy_k_center(features, k=5)
        assert len(selected) == 5

    def test_k_larger_than_n_clamps(self):
        """When k > n, should clamp to n and return unique indices."""
        features = torch.randn(5, 3)
        selected = greedy_k_center(features, k=10)

        assert len(selected) == 5  # Clamped to n
        assert len(selected.unique()) == 5  # All unique


# ---------------------------------------------------------------------------
# subsample_for_efficiency
# ---------------------------------------------------------------------------


class TestSubsampleForEfficiency:
    def test_data_smaller_than_max(self):
        data = torch.randn(10, 5)
        result, indices = subsample_for_efficiency(data, max_samples=20)

        assert torch.equal(result, data)
        assert len(indices) == 10

    def test_data_larger_than_max(self):
        data = torch.randn(100, 5)
        result, indices = subsample_for_efficiency(data, max_samples=10)

        assert len(result) == 10
        assert len(indices) == 10
        # Indices should be sorted
        assert torch.all(indices[1:] >= indices[:-1])

    def test_seed_reproducibility(self):
        data = torch.randn(100, 5)
        _, idx1 = subsample_for_efficiency(data, max_samples=10, random_seed=42)
        _, idx2 = subsample_for_efficiency(data, max_samples=10, random_seed=42)
        assert torch.equal(idx1, idx2)

    def test_different_seeds_differ(self):
        data = torch.randn(100, 5)
        _, idx1 = subsample_for_efficiency(data, max_samples=10, random_seed=0)
        _, idx2 = subsample_for_efficiency(data, max_samples=10, random_seed=1)
        assert not torch.equal(idx1, idx2)


# ---------------------------------------------------------------------------
# active_learning_loop
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 3)

    def forward(self, x):
        return self.fc(x)


class TestActiveLearningLoop:
    def test_basic_loop(self):
        from incerto.active.acquisition import EntropyAcquisition
        from incerto.active.strategies import UncertaintySampling

        torch.manual_seed(42)
        model = _TinyModel()
        x_pool = torch.randn(50, 5)
        y_pool = torch.randint(0, 3, (50,))

        strategy = UncertaintySampling(EntropyAcquisition(), batch_size=5)

        def train_fn(m, x, y):
            pass  # no-op for test

        def eval_fn(m):
            return 0.5

        results = active_learning_loop(
            model,
            x_pool,
            y_pool,
            strategy,
            num_rounds=3,
            initial_labeled=10,
            train_fn=train_fn,
            eval_fn=eval_fn,
            random_seed=42,
        )

        assert len(results["labeled_sizes"]) == 3
        assert results["labeled_sizes"][0] == 10
        assert results["labeled_sizes"][1] == 15
        assert results["labeled_sizes"][2] == 20
        assert len(results["accuracies"]) == 3
        assert all(a == 0.5 for a in results["accuracies"])
        assert len(results["selected_indices"]) == 3

    def test_loop_exhausts_pool(self):
        from incerto.active.acquisition import EntropyAcquisition
        from incerto.active.strategies import UncertaintySampling

        torch.manual_seed(42)
        model = _TinyModel()
        x_pool = torch.randn(15, 5)
        y_pool = torch.randint(0, 3, (15,))

        strategy = UncertaintySampling(EntropyAcquisition(), batch_size=5)

        results = active_learning_loop(
            model,
            x_pool,
            y_pool,
            strategy,
            num_rounds=10,
            initial_labeled=5,
            random_seed=0,
        )

        # Should stop after 2 rounds: 5 initial + 5 + 5 = 15 = all
        assert results["labeled_sizes"][-1] == 15
