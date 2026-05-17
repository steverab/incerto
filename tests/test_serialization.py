"""
Tests for serialization (state_dict, load_state_dict, save, load) across all classes.
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from incerto.calibration import (
    BetaCalibrator,
    DirichletCalibrator,
    HistogramBinningCalibrator,
    IdentityCalibrator,
    IsotonicRegressionCalibrator,
    MatrixScaling,
    PlattScalingCalibrator,
    TemperatureScaling,
    VectorScaling,
)
from incerto.exceptions import SerializationError
from incerto.ood import MSP, ODIN, Energy, MaxLogit
from incerto.shift import (
    ClassifierShiftDetector,
    EnergyShiftDetector,
    ImportanceWeightingShift,
    KSShiftDetector,
    LabelShiftDetector,
    MMDShiftDetector,
)


@pytest.fixture
def sample_logits():
    """Generate sample logits for calibration testing."""
    torch.manual_seed(42)
    return torch.randn(100, 10)


@pytest.fixture
def sample_labels():
    """Generate sample labels for calibration testing."""
    torch.manual_seed(42)
    return torch.randint(0, 10, (100,))


@pytest.fixture
def simple_model():
    """Create a simple model for OOD detection testing."""
    return nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    torch.manual_seed(42)
    return torch.randn(50, 1, 28, 28)


# ============================================================================
# Calibrator Serialization Tests
# ============================================================================


class TestCalibratorSerialization:
    """Tests for calibrator serialization."""

    def test_identity_calibrator_serialization(self, sample_logits, sample_labels):
        """Test IdentityCalibrator save/load."""
        calibrator = IdentityCalibrator()
        calibrator.fit(sample_logits, sample_labels)

        # Test state_dict
        state = calibrator.state_dict()
        assert isinstance(state, dict)

        # Test load_state_dict
        new_calibrator = IdentityCalibrator()
        new_calibrator.load_state_dict(state)

    def test_temperature_scaling_serialization(self, sample_logits, sample_labels):
        """Test TemperatureScaling save/load."""
        calibrator = TemperatureScaling()
        calibrator.fit(sample_logits, sample_labels, max_iters=10)

        # Test state_dict (inherited from nn.Module)
        state = calibrator.state_dict()
        assert "temperature" in state

        # Test load_state_dict
        new_calibrator = TemperatureScaling()
        new_calibrator.load_state_dict(state)
        assert torch.allclose(calibrator.temperature, new_calibrator.temperature)

        # Test save/load via file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "calibrator.pt")
            calibrator.save(path)
            assert os.path.exists(path)

            loaded = TemperatureScaling()
            loaded_state = torch.load(path, weights_only=True)
            loaded.load_state_dict(loaded_state)
            assert torch.allclose(calibrator.temperature, loaded.temperature)

    def test_vector_scaling_serialization(self, sample_logits, sample_labels):
        """Test VectorScaling save/load."""
        calibrator = VectorScaling(n_classes=10)
        calibrator.fit(sample_logits, sample_labels, max_iters=10)

        state = calibrator.state_dict()
        assert "temperature" in state

        new_calibrator = VectorScaling(n_classes=10)
        new_calibrator.load_state_dict(state)
        assert torch.allclose(calibrator.temperature, new_calibrator.temperature)

    def test_vector_scaling_file_roundtrip(self, sample_logits, sample_labels):
        """Test VectorScaling save/load via file with custom load()."""
        calibrator = VectorScaling(n_classes=10)
        calibrator.fit(sample_logits, sample_labels, max_iters=10)
        original_preds = calibrator.predict(sample_logits).probs

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "vec.pt")
            calibrator.save(path)
            loaded = VectorScaling.load(path)
            assert torch.allclose(original_preds, loaded.predict(sample_logits).probs)

    def test_matrix_scaling_serialization(self, sample_logits, sample_labels):
        """Test MatrixScaling save/load."""
        calibrator = MatrixScaling(n_classes=10)
        calibrator.fit(sample_logits, sample_labels, max_iters=10)

        state = calibrator.state_dict()
        assert "weight" in state
        assert "bias" in state

        new_calibrator = MatrixScaling(n_classes=10)
        new_calibrator.load_state_dict(state)
        assert torch.allclose(calibrator.weight, new_calibrator.weight)
        assert torch.allclose(calibrator.bias, new_calibrator.bias)

    def test_matrix_scaling_file_roundtrip(self, sample_logits, sample_labels):
        """Test MatrixScaling save/load via file with custom load()."""
        calibrator = MatrixScaling(n_classes=10)
        calibrator.fit(sample_logits, sample_labels, max_iters=10)
        original_preds = calibrator.predict(sample_logits).probs

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mat.pt")
            calibrator.save(path)
            loaded = MatrixScaling.load(path)
            assert torch.allclose(original_preds, loaded.predict(sample_logits).probs)

    def test_dirichlet_calibrator_serialization(self, sample_logits, sample_labels):
        """Test DirichletCalibrator save/load."""
        calibrator = DirichletCalibrator(n_classes=10, mu=0.01)
        calibrator.fit(sample_logits, sample_labels, max_iters=10)

        state = calibrator.state_dict()
        assert "_mu" in state
        assert state["_mu"] == 0.01

        new_calibrator = DirichletCalibrator(n_classes=10)
        new_calibrator.load_state_dict(state)
        assert torch.allclose(calibrator.weight, new_calibrator.weight)
        assert new_calibrator.mu == 0.01

    def test_dirichlet_file_roundtrip(self, sample_logits, sample_labels):
        """Test DirichletCalibrator save/load via file preserves mu."""
        calibrator = DirichletCalibrator(n_classes=10, mu=0.05)
        calibrator.fit(sample_logits, sample_labels, max_iters=10)
        original_preds = calibrator.predict(sample_logits).probs

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "dirichlet.pt")
            calibrator.save(path)
            loaded = DirichletCalibrator.load(path)
            assert loaded.mu == 0.05
            assert torch.allclose(original_preds, loaded.predict(sample_logits).probs)

    def test_isotonic_calibrator_serialization(self, sample_logits, sample_labels):
        """Test IsotonicRegressionCalibrator save/load."""
        calibrator = IsotonicRegressionCalibrator()
        calibrator.fit(sample_logits, sample_labels)

        state = calibrator.state_dict()
        assert "n_classes" in state
        assert "calibrators" in state

        new_calibrator = IsotonicRegressionCalibrator()
        new_calibrator.load_state_dict(state)
        assert new_calibrator.n_classes == calibrator.n_classes

    def test_isotonic_file_roundtrip(self, sample_logits, sample_labels):
        """Test IsotonicRegressionCalibrator file round-trip with weights_only."""
        calibrator = IsotonicRegressionCalibrator()
        calibrator.fit(sample_logits, sample_labels)
        original_preds = calibrator.predict(sample_logits).probs

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "isotonic.pt")
            calibrator.save(path)
            loaded = IsotonicRegressionCalibrator.load(path)
            assert torch.allclose(original_preds, loaded.predict(sample_logits).probs)

    def test_histogram_binning_serialization(self, sample_logits, sample_labels):
        """Test HistogramBinningCalibrator save/load."""
        calibrator = HistogramBinningCalibrator(n_bins=10)
        calibrator.fit(sample_logits, sample_labels)

        state = calibrator.state_dict()
        assert "n_bins" in state
        assert "bin_edges" in state

        new_calibrator = HistogramBinningCalibrator(n_bins=10)
        new_calibrator.load_state_dict(state)
        assert new_calibrator.n_bins == calibrator.n_bins

    def test_histogram_file_roundtrip(self, sample_logits, sample_labels):
        """Test HistogramBinningCalibrator file round-trip with weights_only."""
        calibrator = HistogramBinningCalibrator(n_bins=10)
        calibrator.fit(sample_logits, sample_labels)
        original_preds = calibrator.predict(sample_logits).probs

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "histogram.pt")
            calibrator.save(path)
            loaded = HistogramBinningCalibrator.load(path)
            assert torch.allclose(original_preds, loaded.predict(sample_logits).probs)

    def test_platt_scaling_serialization(self, sample_logits, sample_labels):
        """Test PlattScalingCalibrator save/load."""
        calibrator = PlattScalingCalibrator()
        calibrator.fit(sample_logits, sample_labels)

        state = calibrator.state_dict()
        assert "n_classes" in state
        assert "models" in state

        new_calibrator = PlattScalingCalibrator()
        new_calibrator.load_state_dict(state)
        assert new_calibrator.n_classes == calibrator.n_classes

    def test_platt_file_roundtrip(self, sample_logits, sample_labels):
        """Test PlattScalingCalibrator file round-trip with weights_only."""
        calibrator = PlattScalingCalibrator()
        calibrator.fit(sample_logits, sample_labels)
        original_preds = calibrator.predict(sample_logits).probs

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "platt.pt")
            calibrator.save(path)
            loaded = PlattScalingCalibrator.load(path)
            assert torch.allclose(original_preds, loaded.predict(sample_logits).probs)

    def test_beta_calibrator_serialization(self):
        """Test BetaCalibrator save/load."""
        # Binary classification
        logits = torch.randn(100, 2)
        labels = torch.randint(0, 2, (100,))

        calibrator = BetaCalibrator()
        calibrator.fit(logits, labels)

        state = calibrator.state_dict()
        assert "a" in state
        assert "b" in state
        assert "c" in state

        new_calibrator = BetaCalibrator()
        new_calibrator.load_state_dict(state)
        assert new_calibrator.a == calibrator.a
        assert new_calibrator.b == calibrator.b
        assert new_calibrator.c == calibrator.c

    def test_beta_file_roundtrip(self):
        """Test BetaCalibrator file round-trip with weights_only."""
        logits = torch.randn(100, 2)
        labels = torch.randint(0, 2, (100,))

        calibrator = BetaCalibrator()
        calibrator.fit(logits, labels)
        original_preds = calibrator.predict(logits).probs

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "beta.pt")
            calibrator.save(path)
            loaded = BetaCalibrator.load(path)
            assert torch.allclose(original_preds, loaded.predict(logits).probs)


# ============================================================================
# OOD Detector Serialization Tests
# ============================================================================


class TestOODDetectorSerialization:
    """Tests for OOD detector serialization."""

    def test_energy_serialization(self, simple_model):
        """Test Energy detector save/load."""
        detector = Energy(simple_model, temperature=2.0)

        state = detector.state_dict()
        assert "temperature" in state
        assert state["temperature"] == 2.0

        new_detector = Energy(simple_model)
        new_detector.load_state_dict(state)
        assert new_detector.temperature == 2.0

    def test_odin_serialization(self, simple_model):
        """Test ODIN detector save/load."""
        detector = ODIN(simple_model, temperature=1000.0, epsilon=0.002)

        state = detector.state_dict()
        assert "temperature" in state
        assert "epsilon" in state

        new_detector = ODIN(simple_model)
        new_detector.load_state_dict(state)
        assert new_detector.temperature == 1000.0
        assert new_detector.epsilon == 0.002

    def test_msp_serialization(self, simple_model):
        """Test MSP detector (no state to save)."""
        detector = MSP(simple_model)
        state = detector.state_dict()
        assert isinstance(state, dict)

    def test_maxlogit_serialization(self, simple_model):
        """Test MaxLogit detector (no state to save)."""
        detector = MaxLogit(simple_model)
        state = detector.state_dict()
        assert isinstance(state, dict)

    def test_save_load_with_file(self, simple_model):
        """Test OOD detector save/load via file."""
        detector = Energy(simple_model, temperature=2.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "detector.pt")
            detector.save(path)
            assert os.path.exists(path)


# ============================================================================
# Shift Detector Serialization Tests
# ============================================================================


class TestShiftDetectorSerialization:
    """Tests for shift detector serialization."""

    @pytest.fixture
    def shift_data_loader(self):
        """Create a simple data loader for shift detection."""
        from torch.utils.data import DataLoader, TensorDataset

        data = torch.randn(100, 10)
        labels = torch.zeros(100)
        dataset = TensorDataset(data, labels)
        return DataLoader(dataset, batch_size=32)

    def test_mmd_shift_serialization(self, shift_data_loader):
        """Test MMDShiftDetector save/load."""
        detector = MMDShiftDetector(sigma=2.0)
        detector.fit(shift_data_loader)

        state = detector.state_dict()
        assert "sigma" in state
        assert "_reference" in state

        new_detector = MMDShiftDetector()
        new_detector.load_state_dict(state)
        assert new_detector.sigma == 2.0
        assert torch.allclose(detector._reference, new_detector._reference)

    def test_energy_shift_serialization(self, shift_data_loader):
        """Test EnergyShiftDetector save/load."""
        detector = EnergyShiftDetector()
        detector.fit(shift_data_loader)

        state = detector.state_dict()
        assert "_reference" in state

        new_detector = EnergyShiftDetector()
        new_detector.load_state_dict(state)
        assert torch.allclose(detector._reference, new_detector._reference)

    def test_ks_shift_serialization(self, shift_data_loader):
        """Test KSShiftDetector save/load."""
        detector = KSShiftDetector()
        detector.fit(shift_data_loader)

        state = detector.state_dict()
        new_detector = KSShiftDetector()
        new_detector.load_state_dict(state)

    def test_classifier_shift_serialization(self, shift_data_loader):
        """Test ClassifierShiftDetector save/load."""
        detector = ClassifierShiftDetector()
        detector.fit(shift_data_loader)

        state = detector.state_dict()
        assert "_reference" in state

        new_detector = ClassifierShiftDetector()
        new_detector.load_state_dict(state)

    def test_label_shift_serialization(self):
        """Test LabelShiftDetector save/load."""
        detector = LabelShiftDetector(num_classes=10)

        # Set some dummy state
        detector.source_label_dist = torch.ones(10) / 10
        detector.confusion_matrix = torch.eye(10)

        state = detector.state_dict()
        assert "num_classes" in state
        assert "source_label_dist" in state

        new_detector = LabelShiftDetector(num_classes=10)
        new_detector.load_state_dict(state)
        assert new_detector.num_classes == 10
        assert torch.allclose(detector.source_label_dist, new_detector.source_label_dist)

    def test_importance_weighting_serialization(self):
        """Test ImportanceWeightingShift save/load."""
        shift = ImportanceWeightingShift(method="logistic", alpha=0.05)

        # Would normally fit here, but we'll just test serialization structure
        state = shift.state_dict()
        assert "method" in state
        assert "alpha" in state

        new_shift = ImportanceWeightingShift()
        new_shift.load_state_dict(state)
        assert new_shift.method == "logistic"
        assert new_shift.alpha == 0.05

    def test_shift_save_load_file(self, shift_data_loader):
        """Test shift detector save/load via file."""
        detector = MMDShiftDetector(sigma=1.5)
        detector.fit(shift_data_loader)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "shift_detector.pt")
            detector.save(path)
            assert os.path.exists(path)

            loaded = MMDShiftDetector.load(path)
            assert loaded.sigma == 1.5
            assert torch.allclose(detector._reference, loaded._reference)


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestSerializationErrors:
    """Tests for serialization error handling."""

    def test_invalid_state_dict(self, sample_logits, sample_labels):
        """Test loading invalid state dict raises SerializationError."""
        calibrator = IsotonicRegressionCalibrator()

        with pytest.raises(SerializationError):
            calibrator.load_state_dict({"invalid": "state"})

    def test_save_to_invalid_path(self, sample_logits, sample_labels):
        """Test saving to invalid path raises SerializationError."""
        calibrator = TemperatureScaling()
        calibrator.fit(sample_logits, sample_labels, max_iters=10)

        with pytest.raises(SerializationError):
            calibrator.save("/nonexistent/directory/file.pt")

    def test_load_from_nonexistent_file(self):
        """Test loading from nonexistent file raises SerializationError."""
        with pytest.raises(SerializationError):
            MMDShiftDetector.load("/nonexistent/file.pt")
