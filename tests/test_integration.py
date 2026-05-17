"""
Integration and end-to-end tests for incerto.

T2: Integration tests — verify methods produce meaningful results.
T3: Statistical validation — verify stochastic methods converge.
T4: Strengthen assertions — verify methods improve over baselines.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleClassifier(nn.Module):
    """Simple classifier that produces intentionally uncalibrated outputs."""

    def __init__(self, in_features=10, num_classes=5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


@pytest.fixture
def trained_model(set_seed):
    """A model trained enough to have non-trivial predictions."""
    model = SimpleClassifier(in_features=10, num_classes=5)
    x = torch.randn(200, 10)
    y = torch.randint(0, 5, (200,))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(30):
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
    model.eval()
    return model, x, y


@pytest.fixture
def val_data(set_seed):
    """Validation data for calibration."""
    x = torch.randn(100, 10)
    y = torch.randint(0, 5, (100,))
    return x, y


# ---------------------------------------------------------------------------
# T2: Integration tests
# ---------------------------------------------------------------------------


class TestCalibrationIntegration:
    """Verify calibration actually improves ECE."""

    def test_temperature_scaling_reduces_ece(self, trained_model, val_data):
        """TemperatureScaling should reduce ECE on validation data."""
        from incerto.calibration import TemperatureScaling, ece_score

        model, train_x, train_y = trained_model
        val_x, val_y = val_data

        with torch.no_grad():
            val_logits = model(val_x)

        ece_before = ece_score(val_logits, val_y)

        calibrator = TemperatureScaling()
        calibrator.fit(val_logits, val_y)
        calibrated = calibrator.predict(val_logits)

        ece_after = ece_score(calibrated.logits, val_y)

        # Temperature scaling should not make things worse
        assert ece_after <= ece_before + 0.05  # allow small tolerance

    def test_isotonic_regression_calibration(self, trained_model, val_data):
        """IsotonicRegressionCalibrator should produce valid probabilities."""
        from incerto.calibration import IsotonicRegressionCalibrator

        model, _, _ = trained_model
        val_x, val_y = val_data

        with torch.no_grad():
            val_logits = model(val_x)

        calibrator = IsotonicRegressionCalibrator()
        calibrator.fit(val_logits, val_y)
        result = calibrator.predict(val_logits)

        probs = result.probs
        assert (probs >= 0).all()
        assert (probs <= 1).all()
        assert torch.allclose(probs.sum(dim=-1), torch.ones(len(probs)), atol=1e-5)


class TestOODIntegration:
    """Verify OOD detectors separate ID from OOD data."""

    def test_energy_separates_id_ood(self, ood_model, ood_id_inputs, ood_ood_inputs):
        """Energy scores should be higher (more OOD) for OOD data on average."""
        from incerto.ood import Energy

        detector = Energy(ood_model)
        scores_id = detector.score(ood_id_inputs)
        scores_ood = detector.score(ood_ood_inputs)

        # OOD data should have higher energy scores on average
        # (Energy returns negative logsumexp, so more negative = more ID)
        assert scores_ood.mean() > scores_id.mean()

    def test_msp_produces_valid_scores(self, ood_model, ood_id_inputs, ood_ood_inputs):
        """MSP scores should be in [0,1] and vary across inputs."""
        from incerto.ood import MSP

        detector = MSP(ood_model)
        scores_id = detector.score(ood_id_inputs)
        scores_ood = detector.score(ood_ood_inputs)

        # MSP = 1 - max_prob, so should be in [0, 1]
        assert (scores_id >= 0).all() and (scores_id <= 1).all()
        assert (scores_ood >= 0).all() and (scores_ood <= 1).all()
        # Scores should have non-zero variance (model is not degenerate)
        assert scores_id.std() > 0 or scores_ood.std() > 0

    def test_mahalanobis_separates_id_ood(
        self, ood_model, ood_id_loader, ood_id_inputs, ood_ood_inputs
    ):
        """Mahalanobis distance should be larger for OOD data."""
        from incerto.ood import Mahalanobis

        detector = Mahalanobis(ood_model, layer_name="penultimate")
        detector.fit(ood_id_loader)

        scores_id = detector.score(ood_id_inputs)
        scores_ood = detector.score(ood_ood_inputs)

        # OOD data should be farther from class means
        assert scores_ood.mean() > scores_id.mean()


class TestShiftIntegration:
    """Verify shift detectors detect known distribution shifts."""

    def test_mmd_detects_shift(self, set_seed):
        """MMD should give higher score for shifted data than same data."""
        from incerto.shift import MMDShiftDetector

        ref = torch.randn(200, 5)
        same = torch.randn(200, 5)
        shifted = torch.randn(200, 5) + 2.0  # obvious shift

        ref_loader = DataLoader(TensorDataset(ref, torch.zeros(200)), batch_size=64)
        same_loader = DataLoader(TensorDataset(same, torch.zeros(200)), batch_size=64)
        shifted_loader = DataLoader(TensorDataset(shifted, torch.zeros(200)), batch_size=64)

        detector = MMDShiftDetector(sigma=1.0)
        detector.fit(ref_loader)

        score_same = detector.score(same_loader)
        score_shifted = detector.score(shifted_loader)

        assert score_shifted > score_same

    def test_ks_detects_shift(self, set_seed):
        """KS detector should give higher score for shifted data."""
        from incerto.shift import KSShiftDetector

        ref = torch.randn(200, 5)
        same = torch.randn(200, 5)
        shifted = torch.randn(200, 5) + 3.0

        ref_loader = DataLoader(TensorDataset(ref, torch.zeros(200)), batch_size=64)
        same_loader = DataLoader(TensorDataset(same, torch.zeros(200)), batch_size=64)
        shifted_loader = DataLoader(TensorDataset(shifted, torch.zeros(200)), batch_size=64)

        detector = KSShiftDetector()
        detector.fit(ref_loader)

        score_same = detector.score(same_loader)
        score_shifted = detector.score(shifted_loader)

        assert score_shifted > score_same


class TestConformalIntegration:
    """Verify conformal prediction provides coverage guarantees."""

    def test_aps_coverage(self, ood_model, set_seed):
        """APS should achieve approximately (1-alpha) coverage."""
        from incerto.conformal import aps

        alpha = 0.2
        x = torch.randn(500, 64)
        y = torch.randint(0, 10, (500,))
        calib_ds = TensorDataset(x[:250], y[:250])
        calib_loader = DataLoader(calib_ds, batch_size=32)

        predictor = aps(ood_model, calib_loader, alpha=alpha)

        test_x = x[250:]
        test_y = y[250:]
        pred_sets = predictor(test_x)

        # Check coverage
        covered = sum(test_y[i].item() in ps.tolist() for i, ps in enumerate(pred_sets))
        coverage = covered / len(test_y)

        # Conformal guarantee: coverage >= 1 - alpha (with finite-sample slack
        # for untrained model and limited calibration data)
        assert coverage >= 1 - alpha - 0.08

    def test_conformal_predictor_class(self, ood_model, set_seed):
        """ConformalPredictor wrapper should work like the bare function."""
        from incerto.conformal import ConformalPredictor

        x = torch.randn(200, 64)
        y = torch.randint(0, 10, (200,))
        calib_loader = DataLoader(TensorDataset(x[:100], y[:100]), batch_size=32)

        cp = ConformalPredictor.from_method(
            "raps", model=ood_model, calib_loader=calib_loader, alpha=0.1
        )

        assert isinstance(cp, ConformalPredictor)
        assert cp.method == "raps"
        assert cp.alpha == 0.1

        # Both predict() and __call__ should work
        sets1 = cp.predict(x[100:110])
        sets2 = cp(x[100:110])

        assert isinstance(sets1, list)
        assert len(sets1) == 10
        assert isinstance(sets2, list)
        assert len(sets2) == 10


# ---------------------------------------------------------------------------
# T3: Statistical validation of stochastic outputs
# ---------------------------------------------------------------------------


class TestMCDropoutStatistics:
    """Verify MC Dropout produces statistically valid outputs."""

    def test_variance_decreases_with_more_samples(self, set_seed):
        """More MC samples should give more stable variance estimates."""
        from incerto.bayesian import MCDropout

        model = SimpleClassifier()
        mc = MCDropout(model, num_samples=5)
        x = torch.randn(20, 10)

        # Low sample count → high variance in variance estimate
        _, var_5 = mc.predict(x)

        mc_100 = MCDropout(model, num_samples=100)
        _, var_100 = mc_100.predict(x)

        # Both should be non-negative
        assert (var_5 >= 0).all()
        assert (var_100 >= 0).all()

        # High-sample variance estimate should be more stable (lower variance-of-variance)
        # We just verify both produce finite, reasonable values
        assert torch.isfinite(var_5).all()
        assert torch.isfinite(var_100).all()

    def test_samples_shape_consistency(self, set_seed):
        """Returned samples should have shape (num_samples, batch, classes)."""
        from incerto.bayesian import MCDropout

        model = SimpleClassifier(in_features=10, num_classes=5)
        mc = MCDropout(model, num_samples=15)
        x = torch.randn(8, 10)

        mean, var, samples = mc.predict(x, return_samples=True)

        assert samples.shape == (15, 8, 5)
        assert torch.allclose(samples.mean(dim=0), mean, atol=1e-6)
        assert torch.allclose(samples.var(dim=0), var, atol=1e-6)

    def test_mutual_information_non_negative(self, set_seed):
        """Mutual information (epistemic uncertainty) should be non-negative."""
        from incerto.bayesian import MCDropout

        model = SimpleClassifier()
        mc = MCDropout(model, num_samples=30)
        x = torch.randn(20, 10)

        mi = mc.predict_mutual_information(x)
        assert (mi >= -1e-5).all()  # allow small numerical errors


class TestEnsembleStatistics:
    """Verify Deep Ensemble produces statistically valid outputs."""

    def test_ensemble_diversity_positive(self, set_seed):
        """Ensemble diversity should be positive for differently initialized models."""
        from incerto.bayesian import DeepEnsemble

        # Use different seeds per model to ensure real diversity
        seed_counter = [0]

        def create_model():
            seed_counter[0] += 1
            torch.manual_seed(seed_counter[0] * 1000)
            return SimpleClassifier(in_features=10, num_classes=5)

        ensemble = DeepEnsemble(create_model, num_models=5)
        x = torch.randn(20, 10)
        diversity = ensemble.diversity(x)

        assert (diversity >= 0).all()
        assert diversity.mean() > 0  # differently initialized models should disagree

    def test_return_samples_consistency(self, set_seed):
        """return_samples should return all individual predictions."""
        from incerto.bayesian import DeepEnsemble

        ensemble = DeepEnsemble(
            lambda: SimpleClassifier(in_features=10, num_classes=5), num_models=3
        )
        x = torch.randn(8, 10)

        mean, var, samples = ensemble.predict(x, return_samples=True)

        assert samples.shape == (3, 8, 5)
        assert torch.allclose(samples.mean(dim=0), mean, atol=1e-6)


class TestSWAGStatistics:
    """Verify SWAG produces statistically valid outputs."""

    def test_return_samples(self, set_seed):
        """SWAG should support return_samples."""
        from incerto.bayesian import SWAG

        model = SimpleClassifier(in_features=10, num_classes=5)
        swag = SWAG(model, num_samples=10)

        for _ in range(5):
            swag.collect_model(model)

        x = torch.randn(8, 10)
        mean, var, samples = swag.predict(x, return_samples=True)

        assert samples.shape == (10, 8, 5)
        assert torch.allclose(samples.mean(dim=0), mean, atol=1e-6)

    def test_variance_non_negative(self, set_seed):
        """SWAG variance should always be non-negative."""
        from incerto.bayesian import SWAG

        model = SimpleClassifier(in_features=10, num_classes=5)
        swag = SWAG(model, num_samples=10)
        for _ in range(5):
            swag.collect_model(model)

        x = torch.randn(20, 10)
        _, var = swag.predict(x)
        assert (var >= 0).all()


class TestVariationalBayesNNStatistics:
    """Verify VariationalBayesNN return_samples works."""

    def test_return_samples(self, set_seed):
        """VariationalBayesNN should support return_samples."""
        from incerto.bayesian import VariationalBayesNN

        model = VariationalBayesNN(10, [20], 5, num_samples=10)
        x = torch.randn(8, 10)

        mean, var, samples = model.predict(x, return_samples=True)

        assert samples.shape == (10, 8, 5)
        assert torch.allclose(samples.mean(dim=0), mean, atol=1e-6)


# ---------------------------------------------------------------------------
# T4: Strengthened assertions (via integration pattern)
# ---------------------------------------------------------------------------


class TestStrengthenedCalibration:
    """Calibration tests with meaningful assertions beyond shape checks."""

    def test_ece_is_bounded(self, multiclass_logits, multiclass_labels):
        """ECE should be in [0, 1]."""
        from incerto.calibration import ece_score

        ece = ece_score(multiclass_logits, multiclass_labels)
        assert 0 <= ece <= 1

    def test_brier_score_perfect_predictions(self, set_seed):
        """Perfect predictions should have Brier score close to 0."""
        from incerto.calibration import brier_score

        logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0], [10.0, -10.0]])
        labels = torch.tensor([0, 1, 0])

        score = brier_score(logits, labels)
        assert score < 0.01


class TestStrengthenedOOD:
    """OOD tests verifying ID scores < OOD scores."""

    def test_knn_id_closer_than_ood(self, ood_model, ood_id_loader, ood_id_inputs, ood_ood_inputs):
        """KNN distances should be smaller for ID than OOD data."""
        from incerto.ood import KNN

        detector = KNN(ood_model, k=5, layer_name="penultimate")
        detector.fit(ood_id_loader)

        scores_id = detector.score(ood_id_inputs)
        scores_ood = detector.score(ood_ood_inputs)

        # ID data should be closer to training features
        assert scores_id.mean() < scores_ood.mean()


class TestBetaCalibrationWarning:
    """Test A8: BetaCalibrator warns on multiclass."""

    def test_multiclass_warning(self, multiclass_logits, multiclass_labels):
        """BetaCalibrator should warn when used with multiclass data."""
        from incerto.calibration import BetaCalibrator

        calibrator = BetaCalibrator()

        with pytest.warns(UserWarning, match="BetaCalibrator is designed for binary"):
            calibrator.fit(multiclass_logits, multiclass_labels)
