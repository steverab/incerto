"""Tests for incerto.conformal.metrics."""

import torch

from incerto.conformal.metrics import (
    average_set_size,
    conditional_coverage,
    empirical_coverage,
)


class TestEmpiricalCoverage:
    """Test empirical_coverage metric."""

    def test_perfect_coverage(self):
        """All true labels in their prediction sets → coverage = 1.0."""
        y = torch.tensor([0, 1, 2])
        sets = [torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([2])]
        assert empirical_coverage(y, sets) == 1.0

    def test_zero_coverage(self):
        """No true labels in any set → coverage = 0.0."""
        y = torch.tensor([0, 1, 2])
        sets = [torch.tensor([1, 2]), torch.tensor([0, 2]), torch.tensor([0, 1])]
        assert empirical_coverage(y, sets) == 0.0

    def test_partial_coverage(self):
        """Half of true labels covered → coverage ≈ 0.5."""
        y = torch.tensor([0, 1, 2, 3])
        sets = [
            torch.tensor([0]),  # covered
            torch.tensor([0]),  # NOT covered (true=1)
            torch.tensor([2, 3]),  # covered
            torch.tensor([0]),  # NOT covered (true=3)
        ]
        assert empirical_coverage(y, sets) == 0.5

    def test_empty_prediction_sets(self):
        """Empty prediction sets → not covered."""
        y = torch.tensor([0, 1])
        sets = [torch.tensor([]), torch.tensor([1])]
        assert empirical_coverage(y, sets) == 0.5


class TestAverageSetSize:
    """Test average_set_size metric."""

    def test_known_sizes(self):
        """Average of sets with sizes [1, 2, 3] → 2.0."""
        sets = [torch.tensor([0]), torch.tensor([0, 1]), torch.tensor([0, 1, 2])]
        assert average_set_size(sets) == 2.0

    def test_all_singletons(self):
        """All singletons → 1.0."""
        sets = [torch.tensor([i]) for i in range(5)]
        assert average_set_size(sets) == 1.0

    def test_empty_sets(self):
        """Empty sets have size 0."""
        sets = [torch.tensor([]), torch.tensor([0, 1])]
        assert average_set_size(sets) == 1.0  # (0 + 2) / 2


class TestConditionalCoverage:
    """Test conditional_coverage metric."""

    def test_two_groups(self):
        """Compute coverage separately for two groups."""
        y = torch.tensor([0, 1, 0, 1])
        sets = [
            torch.tensor([0]),  # group 0, covered
            torch.tensor([0]),  # group 1, NOT covered
            torch.tensor([1]),  # group 0, NOT covered
            torch.tensor([1]),  # group 1, covered
        ]
        groups = torch.tensor([0, 1, 0, 1])
        result = conditional_coverage(y, sets, groups)
        assert result[0] == 0.5
        assert result[1] == 0.5

    def test_single_group(self):
        """Single group equals global coverage."""
        y = torch.tensor([0, 1, 2])
        sets = [torch.tensor([0]), torch.tensor([1]), torch.tensor([0])]
        groups = torch.tensor([0, 0, 0])
        result = conditional_coverage(y, sets, groups)
        # 2 out of 3 covered
        assert abs(result[0] - 2.0 / 3.0) < 1e-6
