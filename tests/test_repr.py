"""
Tests for __repr__ methods across all classes.
"""

import pytest
import torch
import torch.nn as nn

from incerto.calibration import (
    TemperatureScaling,
    VectorScaling,
    MatrixScaling,
    DirichletCalibrator,
    BetaCalibrator,
    IsotonicRegressionCalibrator,
    HistogramBinningCalibrator,
    PlattScalingCalibrator,
    IdentityCalibrator,
)
from incerto.ood import Energy, ODIN, Mahalanobis, KNN, MSP, MaxLogit
from incerto.shift import (
    MMDShiftDetector,
    EnergyShiftDetector,
    KSShiftDetector,
    ClassifierShiftDetector,
    LabelShiftDetector,
    ImportanceWeightingShift,
)


@pytest.fixture
def sample_logits():
    """Generate sample logits."""
    torch.manual_seed(42)
    return torch.randn(100, 10)


@pytest.fixture
def sample_labels():
    """Generate sample labels."""
    torch.manual_seed(42)
    return torch.randint(0, 10, (100,))


@pytest.fixture
def simple_model():
    """Create a simple model."""
    return nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))


# ============================================================================
# Calibrator __repr__ Tests
# ============================================================================


class TestCalibratorRepr:
    """Tests for calibrator __repr__ methods."""

    def test_identity_calibrator_repr(self):
        """Test IdentityCalibrator __repr__."""
        calibrator = IdentityCalibrator()
        repr_str = repr(calibrator)
        assert "IdentityCalibrator" in repr_str
        assert isinstance(repr_str, str)

    def test_temperature_scaling_repr(self, sample_logits, sample_labels):
        """Test TemperatureScaling __repr__."""
        calibrator = TemperatureScaling(init_temp=1.5)
        calibrator.fit(sample_logits, sample_labels, max_iters=10)

        repr_str = repr(calibrator)
        assert "TemperatureScaling" in repr_str
        assert "temperature" in repr_str
        # Should show the fitted temperature value
        assert isinstance(repr_str, str)

    def test_vector_scaling_repr(self, sample_logits, sample_labels):
        """Test VectorScaling __repr__."""
        calibrator = VectorScaling(n_classes=10)
        calibrator.fit(sample_logits, sample_labels, max_iters=10)

        repr_str = repr(calibrator)
        assert "VectorScaling" in repr_str
        assert "n_classes" in repr_str
        assert "10" in repr_str

    def test_matrix_scaling_repr(self, sample_logits, sample_labels):
        """Test MatrixScaling __repr__."""
        calibrator = MatrixScaling(n_classes=10)
        calibrator.fit(sample_logits, sample_labels, max_iters=10)

        repr_str = repr(calibrator)
        assert "MatrixScaling" in repr_str
        assert "n_classes" in repr_str
        assert "10" in repr_str

    def test_dirichlet_calibrator_repr(self):
        """Test DirichletCalibrator __repr__."""
        calibrator = DirichletCalibrator(n_classes=5, mu=0.01)

        repr_str = repr(calibrator)
        assert "DirichletCalibrator" in repr_str
        assert "n_classes" in repr_str
        assert "5" in repr_str
        assert "mu" in repr_str

    def test_isotonic_calibrator_repr(self, sample_logits, sample_labels):
        """Test IsotonicRegressionCalibrator __repr__."""
        calibrator = IsotonicRegressionCalibrator()
        calibrator.fit(sample_logits, sample_labels)

        repr_str = repr(calibrator)
        assert "IsotonicRegressionCalibrator" in repr_str
        assert "n_classes" in repr_str

    def test_histogram_binning_repr(self, sample_logits, sample_labels):
        """Test HistogramBinningCalibrator __repr__."""
        calibrator = HistogramBinningCalibrator(n_bins=15)
        calibrator.fit(sample_logits, sample_labels)

        repr_str = repr(calibrator)
        assert "HistogramBinningCalibrator" in repr_str
        assert "n_bins" in repr_str
        assert "15" in repr_str

    def test_platt_scaling_repr(self, sample_logits, sample_labels):
        """Test PlattScalingCalibrator __repr__."""
        calibrator = PlattScalingCalibrator()
        calibrator.fit(sample_logits, sample_labels)

        repr_str = repr(calibrator)
        assert "PlattScalingCalibrator" in repr_str
        assert "n_classes" in repr_str

    def test_beta_calibrator_repr(self):
        """Test BetaCalibrator __repr__."""
        calibrator = BetaCalibrator()

        repr_str = repr(calibrator)
        assert "BetaCalibrator" in repr_str


# ============================================================================
# OOD Detector __repr__ Tests
# ============================================================================


class TestOODDetectorRepr:
    """Tests for OOD detector __repr__ methods."""

    def test_msp_repr(self, simple_model):
        """Test MSP __repr__."""
        detector = MSP(simple_model)
        repr_str = repr(detector)
        assert "MSP" in repr_str

    def test_energy_repr(self, simple_model):
        """Test Energy __repr__."""
        detector = Energy(simple_model, temperature=2.0)
        repr_str = repr(detector)
        assert "Energy" in repr_str
        assert "temperature" in repr_str
        assert "2.0" in repr_str

    def test_odin_repr(self, simple_model):
        """Test ODIN __repr__."""
        detector = ODIN(simple_model, temperature=1000.0, epsilon=0.002)
        repr_str = repr(detector)
        assert "ODIN" in repr_str
        assert "temperature" in repr_str
        assert "epsilon" in repr_str

    def test_maxlogit_repr(self, simple_model):
        """Test MaxLogit __repr__."""
        detector = MaxLogit(simple_model)
        repr_str = repr(detector)
        assert "MaxLogit" in repr_str

    def test_mahalanobis_repr_not_fitted(self):
        """Test Mahalanobis __repr__ before fitting."""

        # Create model with named "penultimate" layer
        class ModelWithPenultimate(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten()
                self.penultimate = nn.Linear(28 * 28, 128)
                self.output = nn.Linear(128, 10)

            def forward(self, x):
                x = self.flatten(x)
                x = self.penultimate(x)
                return self.output(x)

        model = ModelWithPenultimate()
        detector = Mahalanobis(model)
        repr_str = repr(detector)
        assert "Mahalanobis" in repr_str
        assert "not fitted" in repr_str

    def test_knn_repr_not_fitted(self):
        """Test KNN __repr__ before fitting."""

        # Create model with named "penultimate" layer
        class ModelWithPenultimate(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten()
                self.penultimate = nn.Linear(28 * 28, 128)
                self.output = nn.Linear(128, 10)

            def forward(self, x):
                x = self.flatten(x)
                x = self.penultimate(x)
                return self.output(x)

        model = ModelWithPenultimate()
        detector = KNN(model, k=10)
        repr_str = repr(detector)
        assert "KNN" in repr_str
        assert "k=10" in repr_str
        assert "not fitted" in repr_str


# ============================================================================
# Shift Detector __repr__ Tests
# ============================================================================


class TestShiftDetectorRepr:
    """Tests for shift detector __repr__ methods."""

    @pytest.fixture
    def shift_data_loader(self):
        """Create a simple data loader."""
        from torch.utils.data import DataLoader, TensorDataset

        data = torch.randn(100, 10)
        labels = torch.zeros(100)
        dataset = TensorDataset(data, labels)
        return DataLoader(dataset, batch_size=32)

    def test_mmd_shift_repr_not_fitted(self):
        """Test MMDShiftDetector __repr__ before fitting."""
        detector = MMDShiftDetector(sigma=1.5)
        repr_str = repr(detector)
        assert "MMDShiftDetector" in repr_str
        assert "sigma=1.5" in repr_str
        assert "not fitted" in repr_str

    def test_mmd_shift_repr_fitted(self, shift_data_loader):
        """Test MMDShiftDetector __repr__ after fitting."""
        detector = MMDShiftDetector(sigma=2.0)
        detector.fit(shift_data_loader)

        repr_str = repr(detector)
        assert "MMDShiftDetector" in repr_str
        assert "sigma=2.0" in repr_str
        assert "n_ref_samples" in repr_str

    def test_energy_shift_repr(self):
        """Test EnergyShiftDetector __repr__."""
        detector = EnergyShiftDetector()
        repr_str = repr(detector)
        assert "EnergyShiftDetector" in repr_str

    def test_ks_shift_repr(self):
        """Test KSShiftDetector __repr__."""
        detector = KSShiftDetector()
        repr_str = repr(detector)
        assert "KSShiftDetector" in repr_str

    def test_classifier_shift_repr(self):
        """Test ClassifierShiftDetector __repr__."""
        detector = ClassifierShiftDetector()
        repr_str = repr(detector)
        assert "ClassifierShiftDetector" in repr_str

    def test_label_shift_repr(self):
        """Test LabelShiftDetector __repr__."""
        detector = LabelShiftDetector(num_classes=10, calibrated=True)
        repr_str = repr(detector)
        assert "LabelShiftDetector" in repr_str
        assert "num_classes=10" in repr_str
        assert "calibrated=True" in repr_str
        assert "fitted=False" in repr_str

    def test_importance_weighting_repr(self):
        """Test ImportanceWeightingShift __repr__."""
        shift = ImportanceWeightingShift(method="logistic", alpha=0.05)
        repr_str = repr(shift)
        assert "ImportanceWeightingShift" in repr_str
        assert "method='logistic'" in repr_str
        assert "alpha=0.05" in repr_str
        assert "fitted=False" in repr_str


# ============================================================================
# General __repr__ Tests
# ============================================================================


class TestReprGeneral:
    """General tests for __repr__ methods."""

    def test_repr_returns_string(self, simple_model):
        """Test that all __repr__ methods return strings."""
        objects = [
            IdentityCalibrator(),
            TemperatureScaling(),
            MSP(simple_model),
            Energy(simple_model),
            MMDShiftDetector(),
            LabelShiftDetector(10),
        ]

        for obj in objects:
            repr_str = repr(obj)
            assert isinstance(repr_str, str)
            assert len(repr_str) > 0

    def test_repr_contains_class_name(self, simple_model):
        """Test that __repr__ contains the class name."""
        objects = [
            (IdentityCalibrator(), "IdentityCalibrator"),
            (TemperatureScaling(), "TemperatureScaling"),
            (MSP(simple_model), "MSP"),
            (Energy(simple_model), "Energy"),
            (MMDShiftDetector(), "MMDShiftDetector"),
        ]

        for obj, class_name in objects:
            repr_str = repr(obj)
            assert class_name in repr_str

    def test_repr_eval_not_required(self):
        """Test that __repr__ doesn't need to be eval-able.

        Note: We don't require eval(repr(obj)) == obj, just informative strings.
        """
        calibrator = TemperatureScaling(init_temp=1.5)
        repr_str = repr(calibrator)

        # Just check it's informative
        assert "TemperatureScaling" in repr_str
        assert "temperature" in repr_str
