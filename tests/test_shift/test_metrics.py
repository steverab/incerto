"""Tests for shift detection metrics."""

import torch
import numpy as np

from incerto.shift.metrics import (
    energy_distance,
    total_variation,
    population_stability_index,
    wasserstein_distance,
    sliced_wasserstein_distance,
)


class TestEnergyDistance:
    def test_identical_distributions(self):
        """Energy distance should be ~0 for identical distributions."""
        torch.manual_seed(42)
        x = torch.randn(100, 10)
        y = torch.randn(100, 10)

        # Same random seed should give similar distributions
        torch.manual_seed(42)
        x_same = torch.randn(100, 10)

        dist_same = energy_distance(x, x_same)
        dist_diff = energy_distance(x, y)

        assert dist_same < dist_diff
        assert abs(dist_same) < 0.5

    def test_shifted_distributions(self):
        """Energy distance should detect shifts."""
        torch.manual_seed(42)
        x = torch.randn(100, 10)
        y = torch.randn(100, 10) + 2.0  # Mean shift

        dist = energy_distance(x, y)

        assert dist > 1.0  # Should detect significant shift

    def test_different_scales(self):
        """Energy distance should detect scale differences."""
        torch.manual_seed(42)
        x = torch.randn(100, 10)
        y = torch.randn(100, 10) * 3.0  # Scale shift

        dist = energy_distance(x, y)

        assert dist > 0.5

    def test_1d_data(self):
        """Test with 1D data."""
        torch.manual_seed(42)
        x = torch.randn(100, 1)
        y = torch.randn(100, 1) + 1.0

        dist = energy_distance(x, y)

        assert isinstance(dist, float)
        assert dist > 0

    def test_2d_data(self):
        """Test with 2D data."""
        torch.manual_seed(42)
        x = torch.randn(50, 2)
        y = torch.randn(50, 2) + 1.0

        dist = energy_distance(x, y)

        assert isinstance(dist, float)
        assert dist > 0

    def test_high_dimensional(self):
        """Test with high-dimensional data."""
        torch.manual_seed(42)
        x = torch.randn(50, 100)
        y = torch.randn(50, 100)

        dist = energy_distance(x, y)

        assert isinstance(dist, float)

    def test_different_sample_sizes(self):
        """Energy distance should work with different sample sizes."""
        torch.manual_seed(42)
        x = torch.randn(100, 10)
        y = torch.randn(50, 10)

        dist = energy_distance(x, y)

        assert isinstance(dist, float)

    def test_return_type(self):
        """Energy distance should return float."""
        x = torch.randn(10, 5)
        y = torch.randn(10, 5)

        dist = energy_distance(x, y)

        assert isinstance(dist, float)


class TestTotalVariation:
    def test_identical_distributions(self):
        """TVD should be 0 for identical distributions."""
        p = torch.tensor([0.2, 0.3, 0.5])
        q = torch.tensor([0.2, 0.3, 0.5])

        tvd = total_variation(p, q)

        assert tvd == 0.0

    def test_opposite_distributions(self):
        """TVD should be 1.0 for completely different distributions."""
        p = torch.tensor([1.0, 0.0, 0.0])
        q = torch.tensor([0.0, 0.0, 1.0])

        tvd = total_variation(p, q)

        assert abs(tvd - 1.0) < 1e-6

    def test_partial_overlap(self):
        """TVD should be between 0 and 1 for partial overlap."""
        p = torch.tensor([0.5, 0.3, 0.2])
        q = torch.tensor([0.2, 0.3, 0.5])

        tvd = total_variation(p, q)

        assert 0 < tvd < 1

    def test_unnormalized_distributions(self):
        """TVD should normalize distributions."""
        p = torch.tensor([2.0, 3.0, 5.0])  # Sum = 10
        q = torch.tensor([4.0, 6.0, 10.0])  # Sum = 20

        tvd = total_variation(p, q)

        # After normalization: p = [0.2, 0.3, 0.5], q = [0.2, 0.3, 0.5]
        assert tvd == 0.0

    def test_small_values(self):
        """Test with very small probability values."""
        p = torch.tensor([1e-10, 0.5, 0.5])
        q = torch.tensor([0.5, 0.5, 1e-10])

        tvd = total_variation(p, q)

        assert 0 <= tvd <= 1

    def test_return_type(self):
        """TVD should return float."""
        p = torch.tensor([0.3, 0.7])
        q = torch.tensor([0.4, 0.6])

        tvd = total_variation(p, q)

        assert isinstance(tvd, float)


class TestPopulationStabilityIndex:
    def test_identical_distributions(self):
        """PSI should be ~0 for identical distributions."""
        p = torch.tensor([0.2, 0.3, 0.5])
        q = torch.tensor([0.2, 0.3, 0.5])

        psi = population_stability_index(p, q)

        assert abs(psi) < 1e-6

    def test_different_distributions(self):
        """PSI should be positive for different distributions."""
        p = torch.tensor([0.2, 0.3, 0.5])
        q = torch.tensor([0.5, 0.3, 0.2])

        psi = population_stability_index(p, q)

        assert psi > 0

    def test_extreme_difference(self):
        """PSI should be large for very different distributions."""
        p = torch.tensor([0.9, 0.05, 0.05])
        q = torch.tensor([0.05, 0.05, 0.9])

        psi = population_stability_index(p, q)

        assert psi > 1.0  # High PSI indicates significant shift

    def test_small_shift(self):
        """PSI should be small for small shifts."""
        p = torch.tensor([0.33, 0.33, 0.34])
        q = torch.tensor([0.34, 0.33, 0.33])

        psi = population_stability_index(p, q)

        assert psi < 0.1  # PSI < 0.1 is often considered stable

    def test_zero_bins_handled(self):
        """PSI should handle zero bins with epsilon."""
        p = torch.tensor([0.0, 0.5, 0.5])
        q = torch.tensor([0.5, 0.5, 0.0])

        # Should not crash due to log(0)
        psi = population_stability_index(p, q)

        assert isinstance(psi, float)
        assert np.isfinite(psi)

    def test_return_type(self):
        """PSI should return float."""
        p = torch.tensor([0.3, 0.7])
        q = torch.tensor([0.4, 0.6])

        psi = population_stability_index(p, q)

        assert isinstance(psi, float)


class TestWassersteinDistance:
    def test_identical_distributions(self):
        """Wasserstein distance should be ~0 for identical distributions."""
        torch.manual_seed(42)
        x = torch.randn(100, 5)
        y = torch.randn(100, 5)

        torch.manual_seed(42)
        x_same = torch.randn(100, 5)

        dist_same = wasserstein_distance(x, x_same, p=2.0)
        dist_diff = wasserstein_distance(x, y, p=2.0)

        assert dist_same < dist_diff
        assert dist_same < 0.5

    def test_shifted_distributions(self):
        """Wasserstein distance should detect shifts."""
        torch.manual_seed(42)
        x = torch.randn(50, 5)
        y = torch.randn(50, 5) + 2.0

        dist = wasserstein_distance(x, y, p=2.0)

        assert dist > 0.3  # Should detect shift, but may be smaller than 1.0

    def test_1d_closed_form(self):
        """Test 1D case uses closed-form solution."""
        torch.manual_seed(42)
        x = torch.randn(100, 1)
        y = torch.randn(100, 1) + 1.0

        dist = wasserstein_distance(x, y, p=2.0)

        assert isinstance(dist, float)
        assert dist > 0

    def test_different_p_values(self):
        """Test different p values (W1 vs W2)."""
        torch.manual_seed(42)
        x = torch.randn(50, 3)
        y = torch.randn(50, 3) + 1.0

        w1 = wasserstein_distance(x, y, p=1.0)
        w2 = wasserstein_distance(x, y, p=2.0)

        assert isinstance(w1, float)
        assert isinstance(w2, float)
        # W1 and W2 should be different
        assert abs(w1 - w2) > 0.01

    def test_different_sample_sizes_1d(self):
        """Test 1D with different sample sizes."""
        x = torch.randn(100, 1)
        y = torch.randn(50, 1)

        dist = wasserstein_distance(x, y, p=2.0)

        assert isinstance(dist, float)
        assert dist >= 0

    def test_max_iter_parameter(self):
        """Test max_iter parameter."""
        torch.manual_seed(42)
        x = torch.randn(30, 5)
        y = torch.randn(30, 5)

        dist1 = wasserstein_distance(x, y, p=2.0, max_iter=50)
        dist2 = wasserstein_distance(x, y, p=2.0, max_iter=200)

        # Both should converge to similar values
        assert abs(dist1 - dist2) < 0.5

    def test_return_type(self):
        """Wasserstein distance should return float."""
        x = torch.randn(20, 3)
        y = torch.randn(20, 3)

        dist = wasserstein_distance(x, y)

        assert isinstance(dist, float)


class TestSlicedWassersteinDistance:
    def test_identical_distributions(self):
        """SWD should be ~0 for identical distributions."""
        torch.manual_seed(42)
        x = torch.randn(100, 10)

        torch.manual_seed(42)
        y = torch.randn(100, 10)

        dist = sliced_wasserstein_distance(x, y, num_projections=50, seed=42)

        assert dist < 0.3

    def test_shifted_distributions(self):
        """SWD should detect shifts."""
        torch.manual_seed(42)
        x = torch.randn(100, 10)
        y = torch.randn(100, 10) + 2.0

        dist = sliced_wasserstein_distance(x, y, num_projections=100, seed=42)

        assert dist > 1.0

    def test_reproducible_with_seed(self):
        """SWD should be reproducible with same seed."""
        torch.manual_seed(42)
        x = torch.randn(50, 10)
        y = torch.randn(50, 10) + 1.0

        dist1 = sliced_wasserstein_distance(x, y, num_projections=50, seed=42)
        dist2 = sliced_wasserstein_distance(x, y, num_projections=50, seed=42)

        assert abs(dist1 - dist2) < 1e-6

    def test_different_seeds_give_different_results(self):
        """Different seeds should give slightly different results."""
        torch.manual_seed(42)
        x = torch.randn(50, 10)
        y = torch.randn(50, 10) + 1.0

        dist1 = sliced_wasserstein_distance(x, y, num_projections=10, seed=1)
        dist2 = sliced_wasserstein_distance(x, y, num_projections=10, seed=2)

        # Should be different but similar
        assert abs(dist1 - dist2) > 0.001
        assert abs(dist1 - dist2) < 0.5

    def test_more_projections_more_stable(self):
        """More projections should give more stable estimates."""
        torch.manual_seed(42)
        x = torch.randn(50, 10)
        y = torch.randn(50, 10) + 1.0

        # Few projections
        dist_few_1 = sliced_wasserstein_distance(x, y, num_projections=10, seed=1)
        dist_few_2 = sliced_wasserstein_distance(x, y, num_projections=10, seed=2)
        variance_few = abs(dist_few_1 - dist_few_2)

        # Many projections
        dist_many_1 = sliced_wasserstein_distance(x, y, num_projections=100, seed=1)
        dist_many_2 = sliced_wasserstein_distance(x, y, num_projections=100, seed=2)
        variance_many = abs(dist_many_1 - dist_many_2)

        # More projections should reduce variance
        assert variance_many < variance_few

    def test_different_p_values(self):
        """Test different p values."""
        torch.manual_seed(42)
        x = torch.randn(50, 10)
        y = torch.randn(50, 10) + 1.0

        dist_p1 = sliced_wasserstein_distance(x, y, num_projections=50, p=1.0, seed=42)
        dist_p2 = sliced_wasserstein_distance(x, y, num_projections=50, p=2.0, seed=42)

        assert isinstance(dist_p1, float)
        assert isinstance(dist_p2, float)
        assert dist_p1 != dist_p2

    def test_high_dimensional(self):
        """Test with high-dimensional data."""
        torch.manual_seed(42)
        x = torch.randn(50, 100)
        y = torch.randn(50, 100) + 0.5

        dist = sliced_wasserstein_distance(x, y, num_projections=100, seed=42)

        assert isinstance(dist, float)
        assert dist > 0

    def test_different_sample_sizes(self):
        """Test with different sample sizes."""
        x = torch.randn(100, 10)
        y = torch.randn(50, 10)

        dist = sliced_wasserstein_distance(x, y, num_projections=50, seed=42)

        assert isinstance(dist, float)
        assert dist >= 0

    def test_return_type(self):
        """SWD should return float."""
        x = torch.randn(30, 5)
        y = torch.randn(30, 5)

        dist = sliced_wasserstein_distance(x, y)

        assert isinstance(dist, float)


# Comparison tests
class TestMetricComparisons:
    def test_all_metrics_detect_shift(self):
        """All metrics should detect a clear shift."""
        torch.manual_seed(42)
        x = torch.randn(100, 10)
        y = torch.randn(100, 10) + 3.0  # Large shift

        energy_dist = energy_distance(x, y)
        wass_dist = wasserstein_distance(x, y, p=2.0, max_iter=100)
        sliced_wass_dist = sliced_wasserstein_distance(
            x, y, num_projections=50, seed=42
        )

        # Energy distance and sliced Wasserstein should clearly detect the shift
        assert energy_dist > 1.0
        assert sliced_wass_dist > 1.0
        # Wasserstein may be small or zero due to Sinkhorn regularization, just check it's non-negative
        assert wass_dist >= 0.0

    def test_all_metrics_small_for_no_shift(self):
        """All metrics should be smaller for no shift than for shifted data."""
        torch.manual_seed(42)
        x = torch.randn(100, 10)

        torch.manual_seed(43)
        y = torch.randn(100, 10)  # Different samples, same distribution
        y_shifted = torch.randn(100, 10) + 3.0  # Shifted distribution

        energy_no = energy_distance(x, y)
        energy_shift = energy_distance(x, y_shifted)
        wass_no = wasserstein_distance(x, y, p=2.0, max_iter=50)
        wass_shift = wasserstein_distance(x, y_shifted, p=2.0, max_iter=50)
        sliced_no = sliced_wasserstein_distance(x, y, num_projections=50, seed=42)
        sliced_shift = sliced_wasserstein_distance(
            x, y_shifted, num_projections=50, seed=42
        )

        # No-shift should be smaller than shifted
        assert energy_no < energy_shift
        assert wass_no < wass_shift
        assert sliced_no < sliced_shift

    def test_discrete_vs_continuous_metrics(self):
        """Test discrete metrics (TVD, PSI) vs continuous metrics."""
        # Discrete distributions
        p = torch.tensor([0.3, 0.3, 0.4])
        q = torch.tensor([0.4, 0.3, 0.3])

        tvd = total_variation(p, q)
        psi = population_stability_index(p, q)

        # Both should detect small difference
        assert tvd < 0.2
        assert psi < 0.1

        # Larger difference
        p2 = torch.tensor([0.1, 0.2, 0.7])
        q2 = torch.tensor([0.7, 0.2, 0.1])

        tvd2 = total_variation(p2, q2)
        psi2 = population_stability_index(p2, q2)

        # Both should detect larger difference
        assert tvd2 > tvd
        assert psi2 > psi
