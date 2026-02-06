"""
Tests for Bayesian metrics functions.
"""

import pytest
import torch

from incerto.bayesian.metrics import (
    ensemble_diversity,
    uncertainty_quality,
    disagreement,
    negative_log_likelihood,
    brier_score,
    predictive_log_likelihood,
    sharpness,
)


class TestEnsembleDiversity:
    """Tests for ensemble_diversity function."""

    @pytest.fixture
    def ensemble_predictions(self):
        """Create sample ensemble predictions."""
        # 5 models, 10 samples, 3 classes
        # Softmax to ensure valid probabilities
        logits = torch.randn(5, 10, 3)
        return torch.softmax(logits, dim=-1)

    def test_variance_metric(self, ensemble_predictions):
        """Test variance-based diversity metric."""
        diversity = ensemble_diversity(ensemble_predictions, metric="variance")
        assert isinstance(diversity, float)
        assert diversity >= 0

    def test_disagreement_metric(self, ensemble_predictions):
        """Test disagreement-based diversity metric."""
        diversity = ensemble_diversity(ensemble_predictions, metric="disagreement")
        assert isinstance(diversity, float)
        assert 0 <= diversity <= 1

    def test_kl_metric(self, ensemble_predictions):
        """Test KL divergence-based diversity metric."""
        diversity = ensemble_diversity(ensemble_predictions, metric="kl")
        assert isinstance(diversity, float)
        assert diversity >= 0

    def test_unknown_metric_raises(self, ensemble_predictions):
        """Test that unknown metric raises ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            ensemble_diversity(ensemble_predictions, metric="invalid")

    def test_identical_predictions_low_diversity(self):
        """Test that identical predictions have low diversity."""
        # All models predict same distribution
        pred = torch.tensor([[0.7, 0.2, 0.1]])
        predictions = pred.unsqueeze(0).expand(5, 10, 3)

        diversity = ensemble_diversity(predictions, metric="variance")
        assert diversity < 1e-6

    def test_diverse_predictions_high_diversity(self):
        """Test that diverse predictions have higher diversity."""
        # Models predict very different distributions
        preds = []
        for i in range(5):
            p = torch.zeros(10, 3)
            p[:, i % 3] = 0.8
            p[:, (i + 1) % 3] = 0.15
            p[:, (i + 2) % 3] = 0.05
            preds.append(p)
        predictions = torch.stack(preds)

        diversity = ensemble_diversity(predictions, metric="variance")
        assert diversity > 0.01


class TestUncertaintyQuality:
    """Tests for uncertainty_quality function."""

    def test_basic_functionality(self):
        """Test basic functionality returns two floats."""
        uncertainties = torch.rand(100)
        errors = torch.randint(0, 2, (100,)).float()

        correlation, auroc = uncertainty_quality(uncertainties, errors)

        assert isinstance(correlation, float)
        assert isinstance(auroc, float)
        assert -1 <= correlation <= 1
        assert 0 <= auroc <= 1

    def test_perfect_uncertainty(self):
        """Test that perfect uncertainty correlation gives high values."""
        # High uncertainty where errors occur
        errors = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]).float()
        uncertainties = torch.tensor([0.1, 0.1, 0.2, 0.2, 0.1, 0.8, 0.9, 0.7, 0.8, 0.9])

        correlation, auroc = uncertainty_quality(uncertainties, errors)

        assert correlation > 0.5
        assert auroc > 0.8

    def test_constant_errors_returns_05_auroc(self):
        """Test that constant errors return 0.5 AUROC."""
        uncertainties = torch.rand(50)
        # All correct (no errors)
        errors = torch.zeros(50)

        correlation, auroc = uncertainty_quality(uncertainties, errors)

        assert auroc == 0.5

    def test_constant_uncertainty_returns_zero_correlation(self):
        """Test that constant uncertainty returns 0 correlation."""
        uncertainties = torch.ones(50) * 0.5  # All same uncertainty
        errors = torch.randint(0, 2, (50,)).float()

        correlation, auroc = uncertainty_quality(uncertainties, errors)

        # Correlation should be 0 when one variable is constant
        assert correlation == 0.0


class TestDisagreement:
    """Tests for disagreement function."""

    @pytest.fixture
    def ensemble_predictions(self):
        """Create sample ensemble predictions."""
        logits = torch.randn(5, 10, 3)
        return torch.softmax(logits, dim=-1)

    def test_variance_method(self, ensemble_predictions):
        """Test variance-based disagreement."""
        scores = disagreement(ensemble_predictions, method="variance")

        assert scores.shape == (10,)
        assert torch.all(scores >= 0)

    def test_entropy_method(self, ensemble_predictions):
        """Test entropy-based disagreement."""
        scores = disagreement(ensemble_predictions, method="entropy")

        assert scores.shape == (10,)
        assert torch.all(scores >= 0)

    def test_unknown_method_raises(self, ensemble_predictions):
        """Test that unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            disagreement(ensemble_predictions, method="invalid")


class TestNegativeLogLikelihood:
    """Tests for negative_log_likelihood function."""

    def test_basic_functionality(self):
        """Test basic NLL computation."""
        predictions = torch.softmax(torch.randn(10, 3), dim=-1)
        labels = torch.randint(0, 3, (10,))

        nll = negative_log_likelihood(predictions, labels)

        assert isinstance(nll, float)
        assert nll >= 0

    def test_perfect_predictions_low_nll(self):
        """Test that perfect predictions have low NLL."""
        # Create predictions that match labels
        predictions = torch.zeros(10, 3)
        labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        for i, label in enumerate(labels):
            predictions[i, label] = 0.99
            predictions[i, (label + 1) % 3] = 0.005
            predictions[i, (label + 2) % 3] = 0.005

        nll = negative_log_likelihood(predictions, labels)

        assert nll < 0.1

    def test_wrong_predictions_high_nll(self):
        """Test that wrong predictions have high NLL."""
        # Create predictions that don't match labels
        predictions = torch.zeros(10, 3)
        labels = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        for i in range(10):
            predictions[i, 1] = 0.99  # Predict class 1 but label is 0
            predictions[i, 0] = 0.005
            predictions[i, 2] = 0.005

        nll = negative_log_likelihood(predictions, labels)

        assert nll > 3.0


class TestBrierScore:
    """Tests for brier_score function."""

    def test_basic_functionality(self):
        """Test basic Brier score computation."""
        predictions = torch.softmax(torch.randn(10, 3), dim=-1)
        labels = torch.randint(0, 3, (10,))

        bs = brier_score(predictions, labels)

        assert isinstance(bs, float)
        assert 0 <= bs <= 2  # Max Brier score for 3 classes

    def test_perfect_predictions_zero_brier(self):
        """Test that perfect predictions have zero Brier score."""
        predictions = torch.zeros(5, 3)
        labels = torch.tensor([0, 1, 2, 0, 1])
        for i, label in enumerate(labels):
            predictions[i, label] = 1.0

        bs = brier_score(predictions, labels)

        assert bs < 1e-6

    def test_uniform_predictions(self):
        """Test Brier score for uniform predictions."""
        # Uniform predictions
        predictions = torch.ones(10, 3) / 3
        labels = torch.randint(0, 3, (10,))

        bs = brier_score(predictions, labels)

        # Expected: (1/3 - 1)^2 + 2*(1/3 - 0)^2 = 4/9 + 2/9 = 6/9 ≈ 0.667
        assert 0.5 < bs < 0.8


class TestPredictiveLogLikelihood:
    """Tests for predictive_log_likelihood function."""

    def test_basic_functionality(self):
        """Test basic predictive log-likelihood computation."""
        predictions = torch.softmax(torch.randn(5, 10, 3), dim=-1)
        labels = torch.randint(0, 3, (10,))

        ll = predictive_log_likelihood(predictions, labels)

        assert isinstance(ll, float)
        assert ll <= 0  # Log-likelihood is negative

    def test_perfect_predictions(self):
        """Test that perfect predictions have near-zero log-likelihood."""
        predictions = torch.zeros(5, 10, 3)
        labels = torch.randint(0, 3, (10,))
        for i, label in enumerate(labels):
            predictions[:, i, label] = 0.99
            predictions[:, i, (label + 1) % 3] = 0.005
            predictions[:, i, (label + 2) % 3] = 0.005

        ll = predictive_log_likelihood(predictions, labels)

        assert ll > -0.1


class TestSharpness:
    """Tests for sharpness function."""

    def test_basic_functionality(self):
        """Test basic sharpness computation."""
        predictions = torch.softmax(torch.randn(10, 3), dim=-1)

        sharp = sharpness(predictions)

        assert isinstance(sharp, float)
        assert sharp >= 0

    def test_confident_predictions_low_sharpness(self):
        """Test that confident predictions have low sharpness (entropy)."""
        # Very confident predictions
        predictions = torch.zeros(10, 3)
        predictions[:, 0] = 0.98
        predictions[:, 1] = 0.01
        predictions[:, 2] = 0.01

        sharp = sharpness(predictions)

        # Low entropy for confident predictions
        assert sharp < 0.2

    def test_uniform_predictions_high_sharpness(self):
        """Test that uniform predictions have high sharpness (entropy)."""
        # Uniform predictions
        predictions = torch.ones(10, 3) / 3

        sharp = sharpness(predictions)

        # Max entropy for 3 classes is log(3) ≈ 1.099
        assert sharp > 1.0
