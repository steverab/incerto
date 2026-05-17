"""Tests for shift detection methods."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from incerto.shift import (
    BBSDDetector,
    ClassifierShiftDetector,
    EnergyShiftDetector,
    ImportanceWeightingShift,
    KSShiftDetector,
    LabelShiftDetector,
    MMDShiftDetector,
)


# Fixtures
@pytest.fixture
def reference_data():
    """Create reference distribution (standard normal)."""
    torch.manual_seed(42)
    X = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    return TensorDataset(X, y)


@pytest.fixture
def no_shift_data():
    """Create test distribution with no shift (same as reference)."""
    torch.manual_seed(43)
    X = torch.randn(80, 10)
    y = torch.randint(0, 3, (80,))
    return TensorDataset(X, y)


@pytest.fixture
def shifted_data():
    """Create test distribution with shift (mean=1, std=2)."""
    torch.manual_seed(44)
    X = torch.randn(80, 10) * 2.0 + 1.0
    y = torch.randint(0, 3, (80,))
    return TensorDataset(X, y)


@pytest.fixture
def simple_model():
    """Simple classifier for label shift detection."""
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 3))
    return model


# Test MMD Shift Detector
class TestMMDShiftDetector:
    def test_init(self):
        detector = MMDShiftDetector(sigma=1.0)
        assert detector.sigma == 1.0
        assert not hasattr(detector, "_reference")

    def test_fit(self, reference_data):
        detector = MMDShiftDetector(sigma=1.0)
        ref_loader = DataLoader(reference_data, batch_size=32)
        detector.fit(ref_loader)

        assert hasattr(detector, "_reference")
        assert detector._reference.shape[0] > 0
        assert detector._reference.shape[1] == 10

    def test_score_no_shift(self, reference_data, no_shift_data):
        detector = MMDShiftDetector(sigma=1.0)
        ref_loader = DataLoader(reference_data, batch_size=32)
        test_loader = DataLoader(no_shift_data, batch_size=32)

        detector.fit(ref_loader)
        score = detector.score(test_loader)

        assert isinstance(score, float)
        assert score >= 0  # MMD is always non-negative
        assert score < 0.1  # Should be small for no shift

    def test_score_with_shift(self, reference_data, shifted_data):
        detector = MMDShiftDetector(sigma=1.0)
        ref_loader = DataLoader(reference_data, batch_size=32)
        test_loader = DataLoader(shifted_data, batch_size=32)

        detector.fit(ref_loader)
        score = detector.score(test_loader)

        assert isinstance(score, float)
        assert score > 0.01  # Should be larger for shifted data

    def test_score_detects_shift(self, reference_data, no_shift_data, shifted_data):
        detector = MMDShiftDetector(sigma=1.0)
        ref_loader = DataLoader(reference_data, batch_size=32)
        no_shift_loader = DataLoader(no_shift_data, batch_size=32)
        shift_loader = DataLoader(shifted_data, batch_size=32)

        detector.fit(ref_loader)
        score_no_shift = detector.score(no_shift_loader)
        score_shift = detector.score(shift_loader)

        # Shifted data should have higher score
        assert score_shift > score_no_shift

    def test_different_sigma(self, reference_data, shifted_data):
        ref_loader = DataLoader(reference_data, batch_size=32)
        test_loader = DataLoader(shifted_data, batch_size=32)

        detector1 = MMDShiftDetector(sigma=0.5)
        detector1.fit(ref_loader)
        score1 = detector1.score(test_loader)

        detector2 = MMDShiftDetector(sigma=2.0)
        detector2.fit(ref_loader)
        score2 = detector2.score(test_loader)

        # Different sigmas should give different scores
        assert score1 != score2

    def test_repr(self, reference_data):
        detector = MMDShiftDetector(sigma=1.5)
        repr_str = repr(detector)
        assert "MMDShiftDetector" in repr_str
        assert "sigma=1.5" in repr_str
        assert "not fitted" in repr_str

        # After fitting
        ref_loader = DataLoader(reference_data, batch_size=32)
        detector.fit(ref_loader)
        repr_str = repr(detector)
        assert "n_ref_samples" in repr_str
        assert "not fitted" not in repr_str


# Test Energy Shift Detector
class TestEnergyShiftDetector:
    def test_init(self):
        detector = EnergyShiftDetector()
        assert not hasattr(detector, "_reference")

    def test_fit(self, reference_data):
        detector = EnergyShiftDetector()
        ref_loader = DataLoader(reference_data, batch_size=32)
        detector.fit(ref_loader)

        assert hasattr(detector, "_reference")

    def test_score_no_shift(self, reference_data, no_shift_data):
        detector = EnergyShiftDetector()
        ref_loader = DataLoader(reference_data, batch_size=32)
        test_loader = DataLoader(no_shift_data, batch_size=32)

        detector.fit(ref_loader)
        score = detector.score(test_loader)

        assert isinstance(score, float)
        assert abs(score) < 0.5  # Should be close to 0 for no shift

    def test_score_with_shift(self, reference_data, shifted_data):
        detector = EnergyShiftDetector()
        ref_loader = DataLoader(reference_data, batch_size=32)
        test_loader = DataLoader(shifted_data, batch_size=32)

        detector.fit(ref_loader)
        score = detector.score(test_loader)

        assert isinstance(score, float)
        assert abs(score) > 0.5  # Should be larger for shifted data

    def test_repr(self, reference_data):
        detector = EnergyShiftDetector()
        repr_str = repr(detector)
        assert "EnergyShiftDetector" in repr_str
        assert "not fitted" in repr_str

        ref_loader = DataLoader(reference_data, batch_size=32)
        detector.fit(ref_loader)
        repr_str = repr(detector)
        assert "n_ref_samples" in repr_str


# Test KS Shift Detector
class TestKSShiftDetector:
    def test_init(self):
        detector = KSShiftDetector()
        assert not hasattr(detector, "_reference")

    def test_fit(self, reference_data):
        detector = KSShiftDetector()
        ref_loader = DataLoader(reference_data, batch_size=32)
        detector.fit(ref_loader)

        assert hasattr(detector, "_reference")

    def test_score_no_shift(self, reference_data, no_shift_data):
        detector = KSShiftDetector()
        ref_loader = DataLoader(reference_data, batch_size=32)
        test_loader = DataLoader(no_shift_data, batch_size=32)

        detector.fit(ref_loader)
        score = detector.score(test_loader)

        assert isinstance(score, float)
        assert 0 <= score <= 1  # KS statistic is in [0, 1]
        assert score < 0.3  # Should be small for no shift

    def test_score_with_shift(self, reference_data, shifted_data):
        detector = KSShiftDetector()
        ref_loader = DataLoader(reference_data, batch_size=32)
        test_loader = DataLoader(shifted_data, batch_size=32)

        detector.fit(ref_loader)
        score = detector.score(test_loader)

        assert isinstance(score, float)
        assert score > 0.3  # Should be larger for shifted data

    def test_1d_data(self):
        """Test KS with 1D data."""
        torch.manual_seed(42)
        ref_data = TensorDataset(torch.randn(100, 1), torch.zeros(100))
        shift_data = TensorDataset(torch.randn(80, 1) + 1.0, torch.zeros(80))

        detector = KSShiftDetector()
        ref_loader = DataLoader(ref_data, batch_size=32)
        test_loader = DataLoader(shift_data, batch_size=32)

        detector.fit(ref_loader)
        score = detector.score(test_loader)

        assert score > 0.3  # Strong shift in 1D

    def test_repr(self, reference_data):
        detector = KSShiftDetector()
        repr_str = repr(detector)
        assert "KSShiftDetector" in repr_str
        assert "not fitted" in repr_str

        ref_loader = DataLoader(reference_data, batch_size=32)
        detector.fit(ref_loader)
        repr_str = repr(detector)
        assert "n_ref_samples" in repr_str
        assert "n_features" in repr_str


# Test Classifier Shift Detector
class TestClassifierShiftDetector:
    def test_init(self):
        detector = ClassifierShiftDetector()
        assert detector.clf is not None

    def test_fit_and_score_no_shift(self, reference_data, no_shift_data):
        detector = ClassifierShiftDetector()
        ref_loader = DataLoader(reference_data, batch_size=32)
        test_loader = DataLoader(no_shift_data, batch_size=32)

        detector.fit(ref_loader)
        score = detector.score(test_loader)

        assert isinstance(score, float)
        assert 0 <= score <= 1  # Score in [0, 1]
        assert score < 0.3  # Should be small for no shift

    def test_fit_and_score_with_shift(self, reference_data, shifted_data):
        detector = ClassifierShiftDetector()
        ref_loader = DataLoader(reference_data, batch_size=32)
        test_loader = DataLoader(shifted_data, batch_size=32)

        detector.fit(ref_loader)
        score = detector.score(test_loader)

        assert isinstance(score, float)
        assert score > 0.3  # Should be larger for shifted data

    def test_custom_classifier(self, reference_data, shifted_data):
        from sklearn.ensemble import RandomForestClassifier

        def clf_factory():
            return RandomForestClassifier(n_estimators=10, random_state=42)

        detector = ClassifierShiftDetector(clf_factory=clf_factory)

        ref_loader = DataLoader(reference_data, batch_size=32)
        test_loader = DataLoader(shifted_data, batch_size=32)

        detector.fit(ref_loader)
        score = detector.score(test_loader)

        assert isinstance(score, float)
        assert score > 0

    def test_repr(self, reference_data):
        detector = ClassifierShiftDetector()
        repr_str = repr(detector)
        assert "ClassifierShiftDetector" in repr_str

        ref_loader = DataLoader(reference_data, batch_size=32)
        detector.fit(ref_loader)
        repr_str = repr(detector)
        assert "n_ref_samples" in repr_str


# Test BBSD Detector (alias)
def test_bbsd_detector_alias():
    """Test that BBSDDetector is an alias for ClassifierShiftDetector."""
    assert BBSDDetector is ClassifierShiftDetector


# Test Label Shift Detector
class TestLabelShiftDetector:
    def test_init(self):
        detector = LabelShiftDetector(num_classes=3)
        assert detector.num_classes == 3
        assert not detector.calibrated
        assert detector.source_label_dist is None
        assert detector.confusion_matrix is None

    def test_fit(self, reference_data, simple_model):
        detector = LabelShiftDetector(num_classes=3)
        ref_loader = DataLoader(reference_data, batch_size=32)
        val_loader = DataLoader(reference_data, batch_size=32)

        detector.fit(simple_model, ref_loader, val_loader)

        assert detector.source_label_dist is not None
        assert detector.confusion_matrix is not None
        assert detector.source_label_dist.shape == (3,)
        assert detector.confusion_matrix.shape == (3, 3)

        # Label distribution should sum to 1
        assert torch.allclose(detector.source_label_dist.sum(), torch.tensor(1.0), atol=1e-6)

    def test_estimate_target_distribution(self, reference_data, simple_model):
        detector = LabelShiftDetector(num_classes=3)
        ref_loader = DataLoader(reference_data, batch_size=32)
        val_loader = DataLoader(reference_data, batch_size=32)
        target_loader = DataLoader(reference_data, batch_size=32)

        detector.fit(simple_model, ref_loader, val_loader)
        target_dist = detector.estimate_target_distribution(simple_model, target_loader)

        assert target_dist.shape == (3,)
        assert torch.allclose(target_dist.sum(), torch.tensor(1.0), atol=1e-6)
        assert (target_dist >= 0).all()

    def test_compute_shift_magnitude_tvd(self, reference_data, simple_model):
        detector = LabelShiftDetector(num_classes=3)
        ref_loader = DataLoader(reference_data, batch_size=32)
        val_loader = DataLoader(reference_data, batch_size=32)
        target_loader = DataLoader(reference_data, batch_size=32)

        detector.fit(simple_model, ref_loader, val_loader)
        shift_mag = detector.compute_shift_magnitude(simple_model, target_loader, metric="tvd")

        assert isinstance(shift_mag, float)
        assert 0 <= shift_mag <= 1  # TVD is in [0, 1]

    def test_compute_shift_magnitude_kl(self, reference_data, simple_model):
        detector = LabelShiftDetector(num_classes=3)
        ref_loader = DataLoader(reference_data, batch_size=32)
        val_loader = DataLoader(reference_data, batch_size=32)
        target_loader = DataLoader(reference_data, batch_size=32)

        detector.fit(simple_model, ref_loader, val_loader)
        shift_mag = detector.compute_shift_magnitude(simple_model, target_loader, metric="kl")

        assert isinstance(shift_mag, float)
        # KL should be non-negative, but numerical issues can cause small negative values
        assert shift_mag > -0.1

    def test_compute_shift_magnitude_l2(self, reference_data, simple_model):
        detector = LabelShiftDetector(num_classes=3)
        ref_loader = DataLoader(reference_data, batch_size=32)
        val_loader = DataLoader(reference_data, batch_size=32)
        target_loader = DataLoader(reference_data, batch_size=32)

        detector.fit(simple_model, ref_loader, val_loader)
        shift_mag = detector.compute_shift_magnitude(simple_model, target_loader, metric="l2")

        assert isinstance(shift_mag, float)
        assert shift_mag >= 0

    def test_compute_shift_magnitude_invalid_metric(self, reference_data, simple_model):
        detector = LabelShiftDetector(num_classes=3)
        ref_loader = DataLoader(reference_data, batch_size=32)
        val_loader = DataLoader(reference_data, batch_size=32)
        target_loader = DataLoader(reference_data, batch_size=32)

        detector.fit(simple_model, ref_loader, val_loader)

        with pytest.raises(ValueError, match="Unknown metric"):
            detector.compute_shift_magnitude(simple_model, target_loader, metric="invalid")

    def test_not_fitted_error(self, reference_data, simple_model):
        detector = LabelShiftDetector(num_classes=3)
        target_loader = DataLoader(reference_data, batch_size=32)

        with pytest.raises(RuntimeError, match="Must call fit"):
            detector.estimate_target_distribution(simple_model, target_loader)

    def test_repr(self, reference_data, simple_model):
        detector = LabelShiftDetector(num_classes=3, calibrated=True)
        repr_str = repr(detector)
        assert "LabelShiftDetector" in repr_str
        assert "num_classes=3" in repr_str
        assert "calibrated=True" in repr_str
        assert "fitted=False" in repr_str

        ref_loader = DataLoader(reference_data, batch_size=32)
        val_loader = DataLoader(reference_data, batch_size=32)
        detector.fit(simple_model, ref_loader, val_loader)
        repr_str = repr(detector)
        assert "fitted=True" in repr_str


# Test Importance Weighting Shift
class TestImportanceWeightingShift:
    def test_init(self):
        iw = ImportanceWeightingShift(method="logistic", alpha=0.01)
        assert iw.method == "logistic"
        assert iw.alpha == 0.01
        assert iw.weights_model is None

    def test_fit_logistic(self, reference_data, shifted_data):
        iw = ImportanceWeightingShift(method="logistic")

        source_features = torch.randn(100, 10)
        target_features = torch.randn(80, 10) + 1.0

        iw.fit(source_features, target_features)

        assert iw.weights_model is not None

    def test_fit_kernel(self, reference_data, shifted_data):
        iw = ImportanceWeightingShift(method="kernel", alpha=0.1)

        # Use smaller data for kernel method (it's slower)
        source_features = torch.randn(50, 10)
        target_features = torch.randn(40, 10) + 1.0

        iw.fit(source_features, target_features)

        assert iw.weights_model is not None

    def test_compute_weights_logistic(self):
        iw = ImportanceWeightingShift(method="logistic")

        source_features = torch.randn(100, 10)
        target_features = torch.randn(80, 10) + 1.0

        iw.fit(source_features, target_features)
        weights = iw.compute_weights(source_features)

        assert weights.shape == (100,)
        assert (weights > 0).all()
        # Weights should average to ~1
        assert torch.abs(weights.mean() - 1.0) < 0.5

    def test_compute_weights_kernel(self):
        iw = ImportanceWeightingShift(method="kernel")

        source_features = torch.randn(50, 10)
        target_features = torch.randn(40, 10) + 1.0

        iw.fit(source_features, target_features)
        weights = iw.compute_weights(source_features)

        assert weights.shape == (50,)
        assert (weights >= 0).all()

    def test_compute_weights_not_fitted(self):
        iw = ImportanceWeightingShift(method="logistic")
        source_features = torch.randn(100, 10)

        with pytest.raises(RuntimeError, match="Must call fit"):
            iw.compute_weights(source_features)

    def test_weighted_loss(self):
        iw = ImportanceWeightingShift(method="logistic")

        loss = torch.tensor([1.0, 2.0, 3.0, 4.0])
        weights = torch.tensor([0.5, 1.0, 1.5, 2.0])

        weighted = iw.weighted_loss(loss, weights)

        assert isinstance(weighted, torch.Tensor)
        assert weighted.shape == ()  # scalar
        # Should be weighted average
        expected = (loss * weights).mean()
        assert torch.allclose(weighted, expected)

    def test_invalid_method(self):
        iw = ImportanceWeightingShift(method="invalid")

        source_features = torch.randn(100, 10)
        target_features = torch.randn(80, 10)

        with pytest.raises(ValueError, match="Unknown method"):
            iw.fit(source_features, target_features)

    def test_repr(self):
        iw = ImportanceWeightingShift(method="logistic", alpha=0.05)
        repr_str = repr(iw)
        assert "ImportanceWeightingShift" in repr_str
        assert "method='logistic'" in repr_str
        assert "alpha=0.05" in repr_str
        assert "fitted=False" in repr_str

        source_features = torch.randn(100, 10)
        target_features = torch.randn(80, 10)
        iw.fit(source_features, target_features)
        repr_str = repr(iw)
        assert "fitted=True" in repr_str


# Edge cases and error handling
class TestShiftDetectorEdgeCases:
    def test_empty_data(self):
        """Test with empty data."""
        detector = MMDShiftDetector()
        empty_data = TensorDataset(torch.empty(0, 10), torch.empty(0, dtype=torch.long))
        loader = DataLoader(empty_data, batch_size=32)

        # Should handle empty data gracefully (or may raise error, which is also acceptable)
        try:
            detector.fit(loader)
            assert detector._reference.shape[0] == 0
        except (ValueError, RuntimeError):
            # It's acceptable to raise an error for empty data
            pass

    def test_single_sample(self):
        """Test with single sample."""
        detector = EnergyShiftDetector()
        single_data = TensorDataset(torch.randn(1, 10), torch.tensor([0]))
        loader = DataLoader(single_data, batch_size=1)

        detector.fit(loader)
        score = detector.score(loader)
        assert isinstance(score, float)

    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        detector = MMDShiftDetector(sigma=1.0)

        torch.manual_seed(42)
        ref_data = TensorDataset(torch.randn(50, 100), torch.zeros(50))
        test_data = TensorDataset(torch.randn(40, 100) + 0.5, torch.zeros(40))

        ref_loader = DataLoader(ref_data, batch_size=16)
        test_loader = DataLoader(test_data, batch_size=16)

        detector.fit(ref_loader)
        score = detector.score(test_loader)

        assert isinstance(score, float)
        assert score > 0

    def test_different_batch_sizes(self, reference_data, no_shift_data):
        """Test that different batch sizes don't affect scores significantly."""
        detector = MMDShiftDetector(sigma=1.0)

        ref_loader_small = DataLoader(reference_data, batch_size=16)
        ref_loader_large = DataLoader(reference_data, batch_size=64)
        test_loader = DataLoader(no_shift_data, batch_size=32)

        detector.fit(ref_loader_small)
        score1 = detector.score(test_loader)

        detector.fit(ref_loader_large)
        score2 = detector.score(test_loader)

        # Scores should be similar (not exact due to floating point)
        assert abs(score1 - score2) < 0.05


class TestShiftDetectorsActuallyDetectShifts:
    """Shift score on shifted data should be > shift score on same-distribution data."""

    @pytest.fixture
    def ref_loader(self, reference_data):
        return DataLoader(reference_data, batch_size=32)

    @pytest.fixture
    def no_shift_loader(self, no_shift_data):
        return DataLoader(no_shift_data, batch_size=32)

    @pytest.fixture
    def shifted_loader(self, shifted_data):
        return DataLoader(shifted_data, batch_size=32)

    def test_mmd_detects_shift(self, ref_loader, no_shift_loader, shifted_loader):
        detector = MMDShiftDetector(sigma=1.0)
        detector.fit(ref_loader)
        s_no_shift = detector.score(no_shift_loader)
        s_shifted = detector.score(shifted_loader)
        assert s_shifted > s_no_shift, (
            f"MMD failed to detect shift: no_shift={s_no_shift:.4f}, " f"shifted={s_shifted:.4f}"
        )

    def test_energy_detects_shift(self, ref_loader, no_shift_loader, shifted_loader):
        detector = EnergyShiftDetector()
        detector.fit(ref_loader)
        s_no_shift = detector.score(no_shift_loader)
        s_shifted = detector.score(shifted_loader)
        assert s_shifted > s_no_shift, (
            f"EnergyShiftDetector failed to detect shift: "
            f"no_shift={s_no_shift:.4f}, shifted={s_shifted:.4f}"
        )

    def test_ks_detects_shift(self, ref_loader, no_shift_loader, shifted_loader):
        detector = KSShiftDetector()
        detector.fit(ref_loader)
        s_no_shift = detector.score(no_shift_loader)
        s_shifted = detector.score(shifted_loader)
        assert s_shifted > s_no_shift, (
            f"KS failed to detect shift: no_shift={s_no_shift:.4f}, " f"shifted={s_shifted:.4f}"
        )

    def test_classifier_detects_shift(self, ref_loader, no_shift_loader, shifted_loader):
        detector = ClassifierShiftDetector()
        detector.fit(ref_loader)
        s_no_shift = detector.score(no_shift_loader)
        s_shifted = detector.score(shifted_loader)
        assert s_shifted > s_no_shift, (
            f"ClassifierShiftDetector failed to detect shift: "
            f"no_shift={s_no_shift:.4f}, shifted={s_shifted:.4f}"
        )

    def test_classifier_identical_data_score_zero(self, reference_data):
        """When test == reference exactly, score should be ~0 (no shift)."""
        detector = ClassifierShiftDetector()
        ref_loader = DataLoader(reference_data, batch_size=32)
        detector.fit(ref_loader)
        # Same data
        score = detector.score(ref_loader)
        # Should be very close to 0 (short-circuited)
        assert score < 0.2, f"Expected near-zero shift score, got {score:.3f}"
