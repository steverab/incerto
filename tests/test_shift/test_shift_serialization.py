"""Tests for shift detector serialization."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from incerto.exceptions import SerializationError
from incerto.shift import (
    ClassifierShiftDetector,
    EnergyShiftDetector,
    ImportanceWeightingShift,
    KSShiftDetector,
    LabelShiftDetector,
    MMDShiftDetector,
)


@pytest.fixture
def temp_file():
    """Create temporary file for saving."""
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def reference_data():
    """Create reference distribution."""
    torch.manual_seed(42)
    X = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    return TensorDataset(X, y)


@pytest.fixture
def test_data():
    """Create test distribution."""
    torch.manual_seed(43)
    X = torch.randn(80, 10) + 0.5
    y = torch.randint(0, 3, (80,))
    return TensorDataset(X, y)


@pytest.fixture
def simple_model():
    """Simple classifier."""
    return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 3))


class TestMMDShiftDetectorSerialization:
    def test_state_dict(self, reference_data):
        """Test state_dict saves all necessary information."""
        detector = MMDShiftDetector(sigma=2.5)
        ref_loader = DataLoader(reference_data, batch_size=32)
        detector.fit(ref_loader)

        state = detector.state_dict()

        assert "sigma" in state
        assert "_reference" in state
        assert state["sigma"] == 2.5
        assert isinstance(state["_reference"], torch.Tensor)

    def test_load_state_dict(self, reference_data):
        """Test load_state_dict restores detector."""
        detector1 = MMDShiftDetector(sigma=2.5)
        ref_loader = DataLoader(reference_data, batch_size=32)
        detector1.fit(ref_loader)

        state = detector1.state_dict()

        detector2 = MMDShiftDetector()
        detector2.load_state_dict(state)

        assert detector2.sigma == 2.5
        assert torch.allclose(detector2._reference, detector1._reference)

    def test_save_load(self, reference_data, test_data, temp_file):
        """Test save and load methods."""
        detector1 = MMDShiftDetector(sigma=1.5)
        ref_loader = DataLoader(reference_data, batch_size=32)
        test_loader = DataLoader(test_data, batch_size=32)

        detector1.fit(ref_loader)
        score1 = detector1.score(test_loader)

        # Save
        detector1.save(temp_file)
        assert os.path.exists(temp_file)

        # Load
        detector2 = MMDShiftDetector.load(temp_file)
        score2 = detector2.score(test_loader)

        # Scores should be identical
        assert abs(score1 - score2) < 1e-6

    def test_save_load_preserves_parameters(self, reference_data, temp_file):
        """Test that save/load preserves all parameters."""
        detector1 = MMDShiftDetector(sigma=3.0)
        ref_loader = DataLoader(reference_data, batch_size=32)
        detector1.fit(ref_loader)

        detector1.save(temp_file)
        detector2 = MMDShiftDetector.load(temp_file)

        assert detector2.sigma == 3.0
        assert hasattr(detector2, "_reference")
        assert torch.allclose(detector2._reference, detector1._reference)


class TestEnergyShiftDetectorSerialization:
    def test_save_load(self, reference_data, test_data, temp_file):
        """Test save and load for Energy detector."""
        detector1 = EnergyShiftDetector()
        ref_loader = DataLoader(reference_data, batch_size=32)
        test_loader = DataLoader(test_data, batch_size=32)

        detector1.fit(ref_loader)
        score1 = detector1.score(test_loader)

        detector1.save(temp_file)
        detector2 = EnergyShiftDetector.load(temp_file)
        score2 = detector2.score(test_loader)

        assert abs(score1 - score2) < 1e-6


class TestKSShiftDetectorSerialization:
    def test_save_load(self, reference_data, test_data, temp_file):
        """Test save and load for KS detector."""
        detector1 = KSShiftDetector()
        ref_loader = DataLoader(reference_data, batch_size=32)
        test_loader = DataLoader(test_data, batch_size=32)

        detector1.fit(ref_loader)
        score1 = detector1.score(test_loader)

        detector1.save(temp_file)
        detector2 = KSShiftDetector.load(temp_file)
        score2 = detector2.score(test_loader)

        assert abs(score1 - score2) < 1e-6


class TestClassifierShiftDetectorSerialization:
    def test_state_dict(self, reference_data):
        """Test state_dict for Classifier detector."""
        detector = ClassifierShiftDetector()
        ref_loader = DataLoader(reference_data, batch_size=32)
        detector.fit(ref_loader)

        state = detector.state_dict()

        assert "device" in state
        assert "_reference" in state

    def test_load_state_dict(self, reference_data):
        """Test load_state_dict for Classifier detector."""
        detector1 = ClassifierShiftDetector()
        ref_loader = DataLoader(reference_data, batch_size=32)
        detector1.fit(ref_loader)

        state = detector1.state_dict()

        detector2 = ClassifierShiftDetector()
        detector2.load_state_dict(state)

        assert hasattr(detector2, "clf")
        assert hasattr(detector2, "_reference")

    def test_save_load(self, reference_data, test_data, temp_file):
        """Test save and load for Classifier detector."""
        detector1 = ClassifierShiftDetector()
        ref_loader = DataLoader(reference_data, batch_size=32)
        test_loader = DataLoader(test_data, batch_size=32)

        detector1.fit(ref_loader)
        score1 = detector1.score(test_loader)

        detector1.save(temp_file)
        detector2 = ClassifierShiftDetector.load(temp_file)
        score2 = detector2.score(test_loader)

        # Scores should be very similar (may have slight numerical differences)
        assert abs(score1 - score2) < 0.1

    def test_save_load_with_custom_classifier(self, reference_data, temp_file):
        """Test serialization with custom classifier."""
        from sklearn.ensemble import RandomForestClassifier

        def clf_factory():
            return RandomForestClassifier(n_estimators=10, random_state=42)

        detector1 = ClassifierShiftDetector(clf_factory=clf_factory)
        ref_loader = DataLoader(reference_data, batch_size=32)
        detector1.fit(ref_loader)

        detector1.save(temp_file)
        detector2 = ClassifierShiftDetector.load(temp_file)

        assert hasattr(detector2, "clf")


class TestLabelShiftDetectorSerialization:
    def test_state_dict(self, reference_data, simple_model):
        """Test state_dict for Label Shift detector."""
        detector = LabelShiftDetector(num_classes=3)
        ref_loader = DataLoader(reference_data, batch_size=32)
        val_loader = DataLoader(reference_data, batch_size=32)
        detector.fit(simple_model, ref_loader, val_loader)

        state = detector.state_dict()

        assert "num_classes" in state
        assert "calibrated" in state
        assert "source_label_dist" in state
        assert "confusion_matrix" in state
        assert state["num_classes"] == 3
        assert state["source_label_dist"] is not None
        assert state["confusion_matrix"] is not None

    def test_load_state_dict(self, reference_data, simple_model):
        """Test load_state_dict for Label Shift detector."""
        detector1 = LabelShiftDetector(num_classes=3, calibrated=True)
        ref_loader = DataLoader(reference_data, batch_size=32)
        val_loader = DataLoader(reference_data, batch_size=32)
        detector1.fit(simple_model, ref_loader, val_loader)

        state = detector1.state_dict()

        detector2 = LabelShiftDetector(num_classes=3)
        detector2.load_state_dict(state)

        assert detector2.num_classes == 3
        assert detector2.calibrated
        assert detector2.source_label_dist is not None
        assert detector2.confusion_matrix is not None
        assert torch.allclose(detector2.source_label_dist, detector1.source_label_dist)
        assert torch.allclose(detector2.confusion_matrix, detector1.confusion_matrix)

    def test_save_load(self, reference_data, simple_model, temp_file):
        """Test save and load for Label Shift detector."""
        detector1 = LabelShiftDetector(num_classes=3)
        ref_loader = DataLoader(reference_data, batch_size=32)
        val_loader = DataLoader(reference_data, batch_size=32)
        target_loader = DataLoader(reference_data, batch_size=32)

        detector1.fit(simple_model, ref_loader, val_loader)
        dist1 = detector1.estimate_target_distribution(simple_model, target_loader)

        # Save
        detector1.save(temp_file)
        assert os.path.exists(temp_file)

        # Load
        detector2 = LabelShiftDetector.load(temp_file, num_classes=3)
        dist2 = detector2.estimate_target_distribution(simple_model, target_loader)

        # Distributions should be identical
        assert torch.allclose(dist1, dist2, atol=1e-6)

    def test_load_with_wrong_num_classes(self, reference_data, simple_model, temp_file):
        """Test that loading with wrong num_classes still works."""
        detector1 = LabelShiftDetector(num_classes=3)
        ref_loader = DataLoader(reference_data, batch_size=32)
        val_loader = DataLoader(reference_data, batch_size=32)
        detector1.fit(simple_model, ref_loader, val_loader)

        detector1.save(temp_file)

        # Load with different num_classes (should override with saved value)
        detector2 = LabelShiftDetector.load(temp_file, num_classes=5)

        # Should have the saved num_classes, not the argument
        assert detector2.num_classes == 3


class TestImportanceWeightingShiftSerialization:
    def test_state_dict_logistic(self):
        """Test state_dict for Importance Weighting (logistic)."""
        iw = ImportanceWeightingShift(method="logistic", alpha=0.05)
        source_features = torch.randn(100, 10)
        target_features = torch.randn(80, 10) + 0.5
        iw.fit(source_features, target_features)

        state = iw.state_dict()

        assert "method" in state
        assert "alpha" in state
        assert "weights_model" in state
        assert state["method"] == "logistic"
        assert state["alpha"] == 0.05
        assert isinstance(state["weights_model"], dict)

    def test_load_state_dict_logistic(self):
        """Test load_state_dict for Importance Weighting (logistic)."""
        iw1 = ImportanceWeightingShift(method="logistic", alpha=0.05)
        source_features = torch.randn(100, 10)
        target_features = torch.randn(80, 10) + 0.5
        iw1.fit(source_features, target_features)

        state = iw1.state_dict()

        iw2 = ImportanceWeightingShift()
        iw2.load_state_dict(state)

        assert iw2.method == "logistic"
        assert iw2.alpha == 0.05
        assert iw2.weights_model is not None

    def test_save_load_logistic(self, temp_file):
        """Test save and load for Importance Weighting (logistic)."""
        iw1 = ImportanceWeightingShift(method="logistic", alpha=0.1)
        source_features = torch.randn(100, 10)
        target_features = torch.randn(80, 10) + 0.5
        iw1.fit(source_features, target_features)

        weights1 = iw1.compute_weights(source_features)

        # Save
        iw1.save(temp_file)
        assert os.path.exists(temp_file)

        # Load
        iw2 = ImportanceWeightingShift.load(temp_file)
        weights2 = iw2.compute_weights(source_features)

        # Weights should be identical
        assert torch.allclose(weights1, weights2, atol=1e-6)

    def test_save_load_kernel(self, temp_file):
        """Test save and load for Importance Weighting (kernel)."""
        iw1 = ImportanceWeightingShift(method="kernel", alpha=0.1)
        source_features = torch.randn(50, 10)
        target_features = torch.randn(40, 10) + 0.5
        iw1.fit(source_features, target_features)

        weights1 = iw1.compute_weights(source_features)

        iw1.save(temp_file)
        iw2 = ImportanceWeightingShift.load(temp_file)
        weights2 = iw2.compute_weights(source_features)

        # Weights should be identical
        assert torch.allclose(weights1, weights2, atol=1e-6)

    def test_save_load_preserves_method(self, temp_file):
        """Test that save/load preserves method and alpha."""
        iw1 = ImportanceWeightingShift(method="logistic", alpha=0.02)
        source_features = torch.randn(100, 10)
        target_features = torch.randn(80, 10)
        iw1.fit(source_features, target_features)

        iw1.save(temp_file)
        iw2 = ImportanceWeightingShift.load(temp_file)

        assert iw2.method == "logistic"
        assert iw2.alpha == 0.02


# Error handling tests
class TestSerializationErrorHandling:
    def test_save_to_invalid_path(self, reference_data):
        """Test saving to invalid path raises SerializationError."""
        detector = MMDShiftDetector()
        ref_loader = DataLoader(reference_data, batch_size=32)
        detector.fit(ref_loader)

        with pytest.raises(SerializationError):
            detector.save("/invalid/path/that/does/not/exist/model.pt")

    def test_load_from_nonexistent_file(self):
        """Test loading from nonexistent file raises SerializationError."""
        with pytest.raises(SerializationError):
            MMDShiftDetector.load("/nonexistent/file.pt")

    def test_load_corrupted_state(self, temp_file):
        """Test loading corrupted state raises SerializationError."""
        # Save garbage data
        torch.save({"invalid": "state"}, temp_file)

        with pytest.raises(SerializationError):
            MMDShiftDetector.load(temp_file)

    def test_label_shift_load_state_dict_error(self):
        """Test LabelShiftDetector load_state_dict with invalid state."""
        detector = LabelShiftDetector(num_classes=3)

        with pytest.raises(SerializationError):
            detector.load_state_dict({"invalid": "state"})

    def test_importance_weighting_load_state_dict_error(self):
        """Test ImportanceWeightingShift load_state_dict with invalid state."""
        iw = ImportanceWeightingShift()

        with pytest.raises(SerializationError):
            iw.load_state_dict({"invalid": "state"})


# Integration tests
class TestSerializationIntegration:
    def test_multiple_detectors(self, reference_data, test_data):
        """Test saving and loading multiple detectors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_loader = DataLoader(reference_data, batch_size=32)
            test_loader = DataLoader(test_data, batch_size=32)

            # Create and save multiple detectors
            detectors = {
                "mmd": MMDShiftDetector(sigma=1.5),
                "energy": EnergyShiftDetector(),
                "ks": KSShiftDetector(),
            }

            scores_before = {}
            for name, detector in detectors.items():
                detector.fit(ref_loader)
                scores_before[name] = detector.score(test_loader)
                detector.save(os.path.join(tmpdir, f"{name}.pt"))

            # Load and verify
            loaded_detectors = {
                "mmd": MMDShiftDetector.load(os.path.join(tmpdir, "mmd.pt")),
                "energy": EnergyShiftDetector.load(os.path.join(tmpdir, "energy.pt")),
                "ks": KSShiftDetector.load(os.path.join(tmpdir, "ks.pt")),
            }

            scores_after = {}
            for name, detector in loaded_detectors.items():
                scores_after[name] = detector.score(test_loader)

            # All scores should match
            for name in scores_before:
                assert abs(scores_before[name] - scores_after[name]) < 1e-5

    def test_save_load_cycle_multiple_times(self, reference_data, test_data):
        """Test multiple save/load cycles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_loader = DataLoader(reference_data, batch_size=32)
            test_loader = DataLoader(test_data, batch_size=32)

            detector = MMDShiftDetector(sigma=2.0)
            detector.fit(ref_loader)
            original_score = detector.score(test_loader)

            # Save and load multiple times
            path = os.path.join(tmpdir, "detector.pt")
            for _ in range(3):
                detector.save(path)
                detector = MMDShiftDetector.load(path)
                score = detector.score(test_loader)
                assert abs(score - original_score) < 1e-6
