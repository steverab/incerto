"""
Tests for Bayesian utility functions.
"""

import torch

from incerto.bayesian.utils import (
    compute_disagreement,
    decompose_uncertainty,
    ensemble_predictions_to_distribution,
    expected_calibration_error,
    mutual_information,
    predictive_entropy,
    sample_from_posterior,
)


class TestPredictiveEntropy:
    """Tests for predictive_entropy function."""

    def test_basic_functionality(self):
        """Test basic predictive entropy computation."""
        # 10 samples, 20 batch, 5 classes
        predictions = torch.softmax(torch.randn(10, 20, 5), dim=-1)

        entropy = predictive_entropy(predictions)

        assert entropy.shape == (20,)
        assert torch.all(entropy >= 0)

    def test_confident_predictions_low_entropy(self):
        """Test that confident predictions have low entropy."""
        # All samples predict same confident distribution
        predictions = torch.zeros(10, 20, 5)
        predictions[:, :, 0] = 0.95
        predictions[:, :, 1:] = 0.0125

        entropy = predictive_entropy(predictions)

        assert torch.all(entropy < 0.5)

    def test_uniform_predictions_high_entropy(self):
        """Test that uniform predictions have high entropy."""
        # All samples predict uniform
        predictions = torch.ones(10, 20, 5) / 5

        entropy = predictive_entropy(predictions)

        # Max entropy for 5 classes is log(5) ≈ 1.609
        assert torch.all(entropy > 1.5)

    def test_single_sample(self):
        """Test with single sample."""
        predictions = torch.softmax(torch.randn(1, 10, 3), dim=-1)

        entropy = predictive_entropy(predictions)

        assert entropy.shape == (10,)


class TestMutualInformation:
    """Tests for mutual_information function."""

    def test_basic_functionality(self):
        """Test basic mutual information computation."""
        predictions = torch.softmax(torch.randn(10, 20, 5), dim=-1)

        mi = mutual_information(predictions)

        assert mi.shape == (20,)
        # MI should be non-negative (allow small numerical errors)
        assert torch.all(mi >= -1e-5)

    def test_identical_samples_zero_mi(self):
        """Test that identical samples have near-zero MI."""
        # All samples predict same distribution
        single_pred = torch.softmax(torch.randn(1, 20, 5), dim=-1)
        predictions = single_pred.expand(10, 20, 5)

        mi = mutual_information(predictions)

        # Should be close to zero when all samples agree
        assert torch.all(mi < 1e-5)

    def test_disagreeing_samples_positive_mi(self):
        """Test that disagreeing samples have positive MI."""
        # Create predictions where samples disagree
        predictions = torch.zeros(5, 10, 3)
        for i in range(5):
            predictions[i, :, i % 3] = 0.9
            predictions[i, :, (i + 1) % 3] = 0.05
            predictions[i, :, (i + 2) % 3] = 0.05

        mi = mutual_information(predictions)

        assert torch.all(mi > 0.1)


class TestExpectedCalibrationError:
    """Tests for expected_calibration_error function."""

    def test_basic_functionality(self):
        """Test basic ECE computation."""
        # Create predictions as probabilities
        predictions = torch.softmax(torch.randn(100, 3), dim=-1)
        labels = torch.randint(0, 3, (100,))

        ece = expected_calibration_error(predictions, labels, n_bins=10)

        assert isinstance(ece, float)
        assert 0 <= ece <= 1

    def test_calibrated_predictions(self):
        """Test that well-calibrated predictions have low ECE."""
        # Create calibrated predictions (confidence ≈ accuracy)
        torch.manual_seed(42)
        n = 1000
        predictions = torch.softmax(torch.randn(n, 3) * 2, dim=-1)
        confidences, predicted = predictions.max(dim=-1)

        # Generate labels that match predictions based on confidence
        # Higher confidence -> more likely to be correct
        labels = predicted.clone()
        # Flip some labels based on confidence
        flip_mask = torch.rand(n) > confidences
        labels[flip_mask] = (labels[flip_mask] + 1) % 3

        ece = expected_calibration_error(predictions, labels, n_bins=10)

        # Well-calibrated predictions should have low ECE
        assert ece < 0.3


class TestDecomposeUncertainty:
    """Tests for decompose_uncertainty function."""

    def test_basic_functionality(self):
        """Test basic uncertainty decomposition."""
        predictions = torch.softmax(torch.randn(10, 20, 5), dim=-1)

        total, epistemic, aleatoric = decompose_uncertainty(predictions)

        assert total.shape == (20,)
        assert epistemic.shape == (20,)
        assert aleatoric.shape == (20,)

    def test_decomposition_adds_up(self):
        """Test that epistemic + aleatoric ≈ total."""
        predictions = torch.softmax(torch.randn(10, 20, 5), dim=-1)

        total, epistemic, aleatoric = decompose_uncertainty(predictions)

        # Total = Epistemic + Aleatoric
        reconstructed = epistemic + aleatoric
        assert torch.allclose(total, reconstructed, atol=1e-5)

    def test_non_negative_components(self):
        """Test that all uncertainty components are non-negative."""
        predictions = torch.softmax(torch.randn(10, 20, 5), dim=-1)

        total, epistemic, aleatoric = decompose_uncertainty(predictions)

        assert torch.all(total >= 0)
        assert torch.all(epistemic >= -1e-5)  # Allow small numerical errors
        assert torch.all(aleatoric >= 0)


class TestComputeDisagreement:
    """Tests for compute_disagreement function."""

    def test_basic_functionality(self):
        """Test basic disagreement computation."""
        predictions = torch.softmax(torch.randn(5, 10, 3), dim=-1)

        disag = compute_disagreement(predictions)

        assert disag.shape == (10,)
        assert torch.all(disag >= 0)

    def test_identical_predictions_zero_disagreement(self):
        """Test that identical predictions have zero disagreement."""
        single_pred = torch.softmax(torch.randn(1, 10, 3), dim=-1)
        predictions = single_pred.expand(5, 10, 3)

        disag = compute_disagreement(predictions)

        assert torch.all(disag < 1e-6)

    def test_diverse_predictions_high_disagreement(self):
        """Test that diverse predictions have higher disagreement."""
        predictions = torch.zeros(5, 10, 3)
        for i in range(5):
            predictions[i, :, i % 3] = 0.9
            predictions[i, :, (i + 1) % 3] = 0.05
            predictions[i, :, (i + 2) % 3] = 0.05

        disag = compute_disagreement(predictions)

        assert torch.all(disag > 0.05)


class TestSampleFromPosterior:
    """Tests for sample_from_posterior function."""

    def test_basic_functionality(self):
        """Test basic sampling from posterior."""
        mean = torch.randn(10, 5)
        variance = torch.rand(10, 5) + 0.1  # Positive variance

        samples = sample_from_posterior(mean, variance, num_samples=20)

        assert samples.shape == (20, 10, 5)

    def test_single_sample(self):
        """Test sampling single sample."""
        mean = torch.randn(10, 5)
        variance = torch.rand(10, 5) + 0.1

        samples = sample_from_posterior(mean, variance, num_samples=1)

        assert samples.shape == (1, 10, 5)

    def test_sample_mean_close_to_mean(self):
        """Test that sample mean is close to true mean."""
        mean = torch.randn(10, 5)
        variance = torch.ones(10, 5) * 0.01  # Small variance

        samples = sample_from_posterior(mean, variance, num_samples=1000)
        sample_mean = samples.mean(dim=0)

        assert torch.allclose(sample_mean, mean, atol=0.1)

    def test_sample_variance_close_to_variance(self):
        """Test that sample variance is close to true variance."""
        mean = torch.zeros(10, 5)
        variance = torch.ones(10, 5) * 0.5

        samples = sample_from_posterior(mean, variance, num_samples=10000)
        sample_var = samples.var(dim=0)

        assert torch.allclose(sample_var, variance, atol=0.1)


class TestEnsemblePredictionsToDistribution:
    """Tests for ensemble_predictions_to_distribution function."""

    def test_basic_functionality(self):
        """Test basic conversion to distribution."""
        predictions = torch.randn(5, 10, 3)

        mean, variance = ensemble_predictions_to_distribution(predictions)

        assert mean.shape == (10, 3)
        assert variance.shape == (10, 3)

    def test_mean_correct(self):
        """Test that mean is computed correctly."""
        predictions = torch.randn(5, 10, 3)

        mean, _ = ensemble_predictions_to_distribution(predictions)

        expected_mean = predictions.mean(dim=0)
        assert torch.allclose(mean, expected_mean)

    def test_variance_correct(self):
        """Test that variance is computed correctly."""
        predictions = torch.randn(5, 10, 3)

        _, variance = ensemble_predictions_to_distribution(predictions)

        expected_variance = predictions.var(dim=0)
        assert torch.allclose(variance, expected_variance)

    def test_variance_non_negative(self):
        """Test that variance is non-negative."""
        predictions = torch.randn(5, 10, 3)

        _, variance = ensemble_predictions_to_distribution(predictions)

        assert torch.all(variance >= 0)
