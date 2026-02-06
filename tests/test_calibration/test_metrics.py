"""
Tests for calibration metrics.

All metrics expect logits (not probabilities) and return Python floats (not tensors).
"""

import pytest
import torch
import numpy as np

from incerto.calibration.metrics import (
    nll,
    brier_score,
    ece_score,
    mce_score,
    classwise_ece,
    adaptive_ece_score,
    smooth_ece,
    _smece_at_sigma,
    _find_sigma_star,
)


class TestNLL:
    """Test negative log-likelihood metric."""

    def test_perfect_predictions(self, num_classes):
        """Test NLL for perfect predictions."""
        n = 100
        logits = torch.zeros(n, num_classes)
        labels = torch.randint(0, num_classes, (n,))

        # Set correct class to very high logit
        logits[torch.arange(n), labels] = 100.0

        nll_value = nll(logits, labels)

        assert isinstance(nll_value, float)
        assert nll_value < 0.01  # Very low NLL for perfect predictions

    def test_uniform_predictions(self, num_classes):
        """Test NLL for uniform predictions."""
        n = 100
        logits = torch.zeros(n, num_classes)  # Uniform after softmax
        labels = torch.randint(0, num_classes, (n,))

        nll_value = nll(logits, labels)

        # Uniform predictions should have NLL ≈ log(K)
        expected = np.log(num_classes)
        assert isinstance(nll_value, float)
        assert abs(nll_value - expected) < 0.1

    def test_random_predictions(self, multiclass_logits, multiclass_labels):
        """Test NLL for random predictions."""
        nll_value = nll(multiclass_logits, multiclass_labels)

        assert isinstance(nll_value, float)
        assert nll_value >= 0  # NLL should be non-negative
        assert np.isfinite(nll_value)


class TestBrierScore:
    """Test Brier score metric."""

    def test_perfect_predictions(self, num_classes):
        """Test Brier score for perfect predictions."""
        n = 100
        logits = torch.zeros(n, num_classes)
        labels = torch.randint(0, num_classes, (n,))

        # Set correct class to very high logit
        logits[torch.arange(n), labels] = 100.0

        bs = brier_score(logits, labels)

        assert isinstance(bs, float)
        assert bs < 0.01  # Very low Brier score for perfect predictions

    def test_worst_predictions(self, num_classes):
        """Test Brier score for worst predictions."""
        n = 100
        logits = torch.zeros(n, num_classes)
        labels = torch.randint(0, num_classes, (n,))

        # Set wrong class to very high logit
        wrong_labels = (labels + 1) % num_classes
        logits[torch.arange(n), wrong_labels] = 100.0

        bs = brier_score(logits, labels)

        assert isinstance(bs, float)
        assert bs > 1.5  # High Brier score for wrong predictions

    def test_range(self, multiclass_logits, multiclass_labels):
        """Test Brier score is in valid range [0, 2]."""
        bs = brier_score(multiclass_logits, multiclass_labels)

        assert isinstance(bs, float)
        assert 0 <= bs <= 2

    def test_random_predictions(self, multiclass_logits, multiclass_labels):
        """Test Brier score for random predictions."""
        bs = brier_score(multiclass_logits, multiclass_labels)

        assert isinstance(bs, float)
        assert np.isfinite(bs)


class TestECE:
    """Test Expected Calibration Error."""

    def test_perfect_calibration(self):
        """Test ECE for well-calibrated predictions."""
        n = 1000
        num_classes = 2

        # Create calibrated predictions: confidence matches accuracy
        logits = torch.zeros(n, num_classes)
        labels = torch.randint(0, num_classes, (n,))

        # Adjust logits so that predictions are somewhat calibrated
        # (This is approximate - perfect calibration is hard to construct)
        logits[torch.arange(n), labels] = torch.randn(n) * 2

        ece = ece_score(logits, labels, n_bins=10)

        assert isinstance(ece, float)
        assert 0 <= ece <= 1

    def test_overconfident_predictions(self):
        """Test ECE for overconfident predictions."""
        n = 1000
        num_classes = 2

        # Always predict with high confidence, but only 50% accurate
        logits = torch.zeros(n, num_classes)
        logits[:, 0] = 10.0  # Always predict class 0 with high confidence
        labels = torch.randint(0, num_classes, (n,))  # Random labels

        ece = ece_score(logits, labels, n_bins=10)

        assert isinstance(ece, float)
        assert ece > 0.1  # Should have high ECE due to overconfidence

    def test_different_bins(self, multiclass_logits, multiclass_labels):
        """Test ECE with different number of bins."""
        for n_bins in [5, 10, 15, 20]:
            ece = ece_score(multiclass_logits, multiclass_labels, n_bins=n_bins)
            assert isinstance(ece, float)
            assert 0 <= ece <= 1
            assert np.isfinite(ece)

    def test_empty_bins(self, num_classes):
        """Test ECE handles empty bins gracefully."""
        # Small dataset with high confidence (all in one bin)
        n = 10
        logits = torch.zeros(n, num_classes)
        labels = torch.randint(0, num_classes, (n,))
        logits[torch.arange(n), labels] = 10.0  # All in high confidence bin

        ece = ece_score(logits, labels, n_bins=100)  # Many bins

        assert isinstance(ece, float)
        assert np.isfinite(ece)
        assert 0 <= ece <= 1


class TestMCE:
    """Test Maximum Calibration Error."""

    def test_perfect_calibration(self):
        """Test MCE for well-calibrated predictions."""
        n = 1000
        num_classes = 2
        logits = torch.zeros(n, num_classes)
        labels = torch.randint(0, num_classes, (n,))
        logits[torch.arange(n), labels] = torch.randn(n) * 2

        mce = mce_score(logits, labels, n_bins=10)

        assert isinstance(mce, float)
        assert 0 <= mce <= 1

    def test_mce_greater_than_ece(self, multiclass_logits, multiclass_labels):
        """Test MCE >= ECE (maximum >= average)."""
        ece = ece_score(multiclass_logits, multiclass_labels, n_bins=10)
        mce = mce_score(multiclass_logits, multiclass_labels, n_bins=10)

        assert isinstance(ece, float)
        assert isinstance(mce, float)
        assert mce >= ece - 1e-5  # MCE should be >= ECE

    def test_range(self, multiclass_logits, multiclass_labels):
        """Test MCE is in valid range [0, 1]."""
        mce = mce_score(multiclass_logits, multiclass_labels, n_bins=10)

        assert isinstance(mce, float)
        assert 0 <= mce <= 1


class TestClasswiseECE:
    """Test class-wise ECE."""

    def test_returns_float(self, multiclass_logits, multiclass_labels):
        """Test classwise ECE returns a single float (not array)."""
        cwece = classwise_ece(multiclass_logits, multiclass_labels, n_bins=10)

        # classwise_ece returns single float (mean over classes), not array!
        assert isinstance(cwece, float)
        assert np.isfinite(cwece)

    def test_range(self, multiclass_logits, multiclass_labels):
        """Test classwise ECE is in [0, 1]."""
        cwece = classwise_ece(multiclass_logits, multiclass_labels, n_bins=10)

        assert isinstance(cwece, float)
        assert 0 <= cwece <= 1

    def test_different_bins(self, multiclass_logits, multiclass_labels):
        """Test classwise ECE with different bins."""
        for n_bins in [5, 10, 15]:
            cwece = classwise_ece(multiclass_logits, multiclass_labels, n_bins=n_bins)
            assert isinstance(cwece, float)
            assert 0 <= cwece <= 1


# Integration tests
class TestMetricsIntegration:
    """Integration tests for calibration metrics."""

    def test_all_metrics_work(self, multiclass_logits, multiclass_labels):
        """Test all metrics can be computed."""
        nll_val = nll(multiclass_logits, multiclass_labels)
        bs_val = brier_score(multiclass_logits, multiclass_labels)
        ece_val = ece_score(multiclass_logits, multiclass_labels)
        mce_val = mce_score(multiclass_logits, multiclass_labels)
        cwece_val = classwise_ece(multiclass_logits, multiclass_labels)

        # All should be Python floats
        assert isinstance(nll_val, float)
        assert isinstance(bs_val, float)
        assert isinstance(ece_val, float)
        assert isinstance(mce_val, float)
        assert isinstance(cwece_val, float)

        # All should be finite
        assert np.isfinite(nll_val)
        assert np.isfinite(bs_val)
        assert np.isfinite(ece_val)
        assert np.isfinite(mce_val)
        assert np.isfinite(cwece_val)


# Edge case tests
class TestEdgeCases:
    """Test edge cases for calibration metrics."""

    def test_single_sample(self, num_classes):
        """Test metrics with single sample."""
        logits = torch.randn(1, num_classes)
        labels = torch.tensor([0])

        # Should handle single sample
        nll_val = nll(logits, labels)
        bs_val = brier_score(logits, labels)

        assert isinstance(nll_val, float)
        assert isinstance(bs_val, float)
        assert np.isfinite(nll_val)
        assert np.isfinite(bs_val)

    def test_binary_classification(self):
        """Test metrics on binary classification."""
        n = 100
        logits = torch.randn(n, 2)
        labels = torch.randint(0, 2, (n,))

        # All metrics should work
        nll_val = nll(logits, labels)
        bs_val = brier_score(logits, labels)
        ece_val = ece_score(logits, labels, n_bins=10)
        mce_val = mce_score(logits, labels, n_bins=10)

        assert np.isfinite(nll_val)
        assert np.isfinite(bs_val)
        assert np.isfinite(ece_val)
        assert np.isfinite(mce_val)

    def test_extreme_logits(self, num_classes):
        """Test metrics with extreme logit values."""
        n = 100
        logits = torch.randn(n, num_classes) * 100  # Very large magnitude
        labels = torch.randint(0, num_classes, (n,))

        # Should handle without overflow/underflow
        nll_val = nll(logits, labels)
        bs_val = brier_score(logits, labels)

        assert np.isfinite(nll_val)
        assert np.isfinite(bs_val)

    def test_deterministic_predictions(self, num_classes):
        """Test metrics with deterministic (very confident) predictions."""
        n = 100
        logits = torch.zeros(n, num_classes)
        labels = torch.randint(0, num_classes, (n,))
        preds = torch.randint(0, num_classes, (n,))

        # Set predicted class to very high logit
        logits[torch.arange(n), preds] = 100.0

        nll_val = nll(logits, labels)
        bs_val = brier_score(logits, labels)
        ece_val = ece_score(logits, labels, n_bins=10)

        assert np.isfinite(nll_val)
        assert np.isfinite(bs_val)
        assert np.isfinite(ece_val)


class TestAdaptiveECE:
    """Test Adaptive Expected Calibration Error."""

    def test_basic_functionality(self, multiclass_logits, multiclass_labels):
        """Test Adaptive ECE basic functionality."""
        aece = adaptive_ece_score(multiclass_logits, multiclass_labels, n_bins=10)

        assert isinstance(aece, float)
        assert 0 <= aece <= 1
        assert np.isfinite(aece)

    def test_overconfident_predictions(self):
        """Test Adaptive ECE for overconfident predictions."""
        n = 1000
        num_classes = 2

        # Always predict with high confidence, but only 50% accurate
        logits = torch.zeros(n, num_classes)
        logits[:, 0] = 10.0  # Always predict class 0 with high confidence
        labels = torch.randint(0, num_classes, (n,))  # Random labels

        aece = adaptive_ece_score(logits, labels, n_bins=10)

        assert isinstance(aece, float)
        assert aece > 0.1  # Should have high Adaptive ECE due to overconfidence

    def test_different_bins(self, multiclass_logits, multiclass_labels):
        """Test Adaptive ECE with different number of bins."""
        for n_bins in [5, 10, 15, 20]:
            aece = adaptive_ece_score(
                multiclass_logits, multiclass_labels, n_bins=n_bins
            )
            assert isinstance(aece, float)
            assert 0 <= aece <= 1
            assert np.isfinite(aece)

    def test_different_norms(self, multiclass_logits, multiclass_labels):
        """Test Adaptive ECE with different norms."""
        aece_l1 = adaptive_ece_score(multiclass_logits, multiclass_labels, norm="l1")
        aece_l2 = adaptive_ece_score(multiclass_logits, multiclass_labels, norm="l2")

        assert isinstance(aece_l1, float)
        assert isinstance(aece_l2, float)
        assert 0 <= aece_l1 <= 1
        assert 0 <= aece_l2 <= 1

    def test_equal_mass_binning(self):
        """Test that Adaptive ECE uses equal-mass binning."""
        n = 1000
        num_classes = 2

        # Create skewed confidence distribution
        logits = torch.zeros(n, num_classes)
        # Most predictions have low confidence
        logits[:900, 0] = 0.1
        # Few predictions have high confidence
        logits[900:, 0] = 10.0

        labels = torch.randint(0, num_classes, (n,))

        # Adaptive ECE should handle this better than standard ECE
        aece = adaptive_ece_score(logits, labels, n_bins=10)

        assert isinstance(aece, float)
        assert 0 <= aece <= 1
        assert np.isfinite(aece)

    def test_invalid_norm_raises_error(self, multiclass_logits, multiclass_labels):
        """Test that invalid norm raises ValueError."""
        with pytest.raises(ValueError, match="Unknown norm"):
            adaptive_ece_score(multiclass_logits, multiclass_labels, norm="l3")

    def test_comparison_with_standard_ece(self, multiclass_logits, multiclass_labels):
        """Test Adaptive ECE vs standard ECE."""
        ece = ece_score(multiclass_logits, multiclass_labels, n_bins=10)
        aece = adaptive_ece_score(
            multiclass_logits, multiclass_labels, n_bins=10, norm="l1"
        )

        # Both should be in valid range
        assert 0 <= ece <= 1
        assert 0 <= aece <= 1

        # Values may differ due to different binning strategies
        assert isinstance(ece, float)
        assert isinstance(aece, float)


class TestSmoothECE:
    """Test Smooth Expected Calibration Error (Blasiok & Nakkiran, ICLR 2024)."""

    def test_basic_functionality(self, multiclass_logits, multiclass_labels):
        """Test smooth_ece returns a valid float."""
        smece = smooth_ece(multiclass_logits, multiclass_labels)

        assert isinstance(smece, float)
        assert 0 <= smece <= 1
        assert np.isfinite(smece)

    def test_low_miscalibration(self):
        """Test smooth_ece is moderate for roughly calibrated predictions."""
        n = 1000
        num_classes = 10

        # Random logits — not perfectly calibrated, but not pathologically bad
        logits = torch.randn(n, num_classes)
        labels = torch.randint(0, num_classes, (n,))

        smece = smooth_ece(logits, labels)

        assert isinstance(smece, float)
        assert 0 <= smece <= 1

    def test_overconfident_predictions(self):
        """Test smooth_ece is high for overconfident predictions."""
        n = 1000
        num_classes = 2

        logits = torch.zeros(n, num_classes)
        logits[:, 0] = 10.0  # Always predict class 0 with high confidence
        labels = torch.randint(0, num_classes, (n,))  # Random labels

        smece = smooth_ece(logits, labels)

        assert isinstance(smece, float)
        assert smece > 0.1  # Should have high smECE

    def test_comparison_with_ece(self, multiclass_logits, multiclass_labels):
        """Test smooth_ece alongside standard ECE — both valid."""
        ece = ece_score(multiclass_logits, multiclass_labels, n_bins=10)
        smece = smooth_ece(multiclass_logits, multiclass_labels)

        assert 0 <= ece <= 1
        assert 0 <= smece <= 1
        assert isinstance(ece, float)
        assert isinstance(smece, float)

    def test_fixed_point_property(self, multiclass_logits, multiclass_labels):
        """Test that sigma* satisfies the fixed-point condition smECE_{sigma*} = sigma*."""
        probs = torch.nn.functional.softmax(multiclass_logits, dim=1).numpy()
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == multiclass_labels.numpy()).astype(float)

        sigma_star = _find_sigma_star(confidences, accuracies)
        smece_at_sigma_star = _smece_at_sigma(confidences, accuracies, sigma_star)

        # Fixed point: smECE_{sigma*} should approximately equal sigma*
        assert abs(smece_at_sigma_star - sigma_star) < 1e-4

    def test_binary_classification(self):
        """Test smooth_ece on binary classification."""
        n = 200
        logits = torch.randn(n, 2)
        labels = torch.randint(0, 2, (n,))

        smece = smooth_ece(logits, labels)

        assert isinstance(smece, float)
        assert 0 <= smece <= 1
        assert np.isfinite(smece)
