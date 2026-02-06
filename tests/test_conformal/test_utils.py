"""Tests for incerto.conformal.utils."""

import torch
from torch.utils.data import TensorDataset

from incerto.conformal.utils import (
    compute_quantile,
    prediction_set_from_scores,
    split_data,
)


class TestComputeQuantile:
    """Test compute_quantile utility."""

    def test_basic_quantile(self):
        """Median of [0, 0.25, 0.5, 0.75, 1.0] with alpha=0.5 unadjusted."""
        scores = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        q = compute_quantile(scores, alpha=0.5, adjusted=False)
        assert abs(q - 0.5) < 1e-5

    def test_adjusted_higher_than_unadjusted(self):
        """Finite-sample correction should give a higher quantile."""
        scores = torch.rand(50)
        q_adj = compute_quantile(scores, alpha=0.1, adjusted=True)
        q_raw = compute_quantile(scores, alpha=0.1, adjusted=False)
        assert q_adj >= q_raw

    def test_small_n_clamps_to_one(self):
        """With very small n, adjusted level would exceed 1.0; should clamp."""
        # n=2, alpha=0.05 → (1-0.05)*(1+1/2) = 0.95*1.5 = 1.425 → clamp to 1.0
        scores = torch.tensor([0.1, 0.9])
        q = compute_quantile(scores, alpha=0.05, adjusted=True)
        # Should return the max score (level clamped to 1.0)
        assert abs(q - 0.9) < 1e-5


class TestPredictionSetFromScores:
    """Test prediction_set_from_scores utility."""

    def test_descending_threshold(self):
        """Higher scores are better: include scores >= threshold."""
        scores = torch.tensor(
            [
                [0.9, 0.05, 0.05],
                [0.4, 0.4, 0.2],
            ]
        )
        sets = prediction_set_from_scores(scores, threshold=0.3, descending=True)
        assert len(sets) == 2
        assert sets[0].tolist() == [0]  # only class 0 >= 0.3
        assert sorted(sets[1].tolist()) == [0, 1]  # classes 0,1 >= 0.3

    def test_ascending_threshold(self):
        """Lower scores are better: include scores <= threshold."""
        scores = torch.tensor(
            [
                [0.1, 0.5, 0.9],
                [0.3, 0.3, 0.3],
            ]
        )
        sets = prediction_set_from_scores(scores, threshold=0.3, descending=False)
        assert len(sets) == 2
        assert sets[0].tolist() == [0]  # only class 0 <= 0.3
        assert sets[1].tolist() == [0, 1, 2]  # all classes <= 0.3


class TestSplitData:
    """Test split_data utility."""

    def test_split_sizes(self):
        """Calibration and test set sizes should match cal_ratio."""
        X = torch.randn(100, 4)
        y = torch.randint(0, 2, (100,))
        dataset = TensorDataset(X, y)

        cal, test = split_data(dataset, cal_ratio=0.3, seed=42)
        assert len(cal) == 30
        assert len(test) == 70

    def test_seed_reproducibility(self):
        """Same seed → identical splits."""
        X = torch.randn(50, 2)
        y = torch.randint(0, 2, (50,))
        dataset = TensorDataset(X, y)

        cal1, test1 = split_data(dataset, cal_ratio=0.5, seed=123)
        cal2, test2 = split_data(dataset, cal_ratio=0.5, seed=123)

        assert cal1.indices == cal2.indices
        assert test1.indices == test2.indices

    def test_no_overlap(self):
        """Calibration and test indices should be disjoint."""
        X = torch.randn(40, 2)
        y = torch.randint(0, 2, (40,))
        dataset = TensorDataset(X, y)

        cal, test = split_data(dataset, cal_ratio=0.5, seed=0)
        cal_set = set(cal.indices)
        test_set = set(test.indices)
        assert len(cal_set & test_set) == 0
        assert len(cal_set | test_set) == 40
