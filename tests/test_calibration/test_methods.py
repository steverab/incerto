"""
Tests for post-hoc calibration methods.

All calibrators:
- .fit(logits, labels) - fit on validation data
- .predict(logits) -> torch.distributions.Categorical
- To get probabilities: calibrator.predict(logits).probs
"""

import pytest
import torch

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
from incerto.exceptions import NotFittedError


class TestIdentityCalibrator:
    """Test IdentityCalibrator (no-op calibrator)."""

    def test_initialization(self):
        """Test calibrator can be initialized."""
        calibrator = IdentityCalibrator()
        assert calibrator is not None

    def test_fit(self, multiclass_logits, multiclass_labels):
        """Test fit method."""
        calibrator = IdentityCalibrator()
        result = calibrator.fit(multiclass_logits, multiclass_labels)
        assert result is calibrator  # Returns self

    def test_predict_returns_categorical(self, multiclass_logits, multiclass_labels):
        """Test predict returns Categorical distribution."""
        calibrator = IdentityCalibrator()
        calibrator.fit(multiclass_logits, multiclass_labels)

        dist = calibrator.predict(multiclass_logits)

        assert isinstance(dist, torch.distributions.Categorical)
        assert hasattr(dist, "probs")
        assert hasattr(dist, "logits")

    def test_predict_unchanged_probs(self, multiclass_logits, multiclass_labels):
        """Test Identity returns unchanged probabilities."""
        calibrator = IdentityCalibrator()
        calibrator.fit(multiclass_logits, multiclass_labels)

        probs_before = torch.softmax(multiclass_logits, dim=1)
        probs_after = calibrator.predict(multiclass_logits).probs

        assert torch.allclose(probs_before, probs_after, atol=1e-5)

    def test_predict_shape(self, multiclass_logits, multiclass_labels):
        """Test predict preserves shape."""
        calibrator = IdentityCalibrator()
        calibrator.fit(multiclass_logits, multiclass_labels)

        probs = calibrator.predict(multiclass_logits).probs

        assert probs.shape == multiclass_logits.shape


class TestTemperatureScaling:
    """Test TemperatureScaling calibrator."""

    def test_initialization(self):
        """Test calibrator can be initialized."""
        calibrator = TemperatureScaling()
        assert calibrator.temperature.item() == 1.0

        calibrator2 = TemperatureScaling(init_temp=2.0)
        assert calibrator2.temperature.item() == 2.0

    def test_fit_learns_temperature(self, calibration_split):
        """Test fit learns a temperature."""
        calibrator = TemperatureScaling()
        calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])
        # Temperature should be positive and may have changed
        assert calibrator.temperature.item() > 0

    def test_predict_shape(self, calibration_split):
        """Test predict preserves shape."""
        calibrator = TemperatureScaling()
        calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])

        logits = calibration_split["val_logits"]
        probs = calibrator.predict(logits).probs

        assert probs.shape == logits.shape

    def test_valid_probabilities(self, calibration_split, check_probability):
        """Test predict returns valid probabilities."""
        calibrator = TemperatureScaling()
        calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])

        probs = calibrator.predict(calibration_split["val_logits"]).probs
        check_probability(probs, dim=1)

    def test_predictions_unchanged(self, calibration_split):
        """Test temperature scaling doesn't change predictions (rank order)."""
        calibrator = TemperatureScaling()
        calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])

        logits = calibration_split["val_logits"]
        preds_before = torch.argmax(logits, dim=1)
        preds_after = calibrator.predict(logits).probs.argmax(dim=1)

        assert torch.equal(preds_before, preds_after)


class TestVectorScaling:
    """Test VectorScaling calibrator."""

    def test_initialization(self, num_classes):
        """Test calibrator requires n_classes."""
        calibrator = VectorScaling(n_classes=num_classes)
        assert calibrator.temperature.shape == (num_classes,)

    def test_fit_learns_per_class_temps(self, calibration_split, num_classes):
        """Test fit learns per-class temperatures."""
        calibrator = VectorScaling(n_classes=num_classes)
        calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])
        # Should have one temperature per class
        assert calibrator.temperature.shape == (num_classes,)
        assert (calibrator.temperature > 0).all()

    def test_predict_shape(self, calibration_split, num_classes):
        """Test predict preserves shape."""
        calibrator = VectorScaling(n_classes=num_classes)
        calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])

        probs = calibrator.predict(calibration_split["val_logits"]).probs

        assert probs.shape == calibration_split["val_logits"].shape

    def test_valid_probabilities(self, calibration_split, num_classes, check_probability):
        """Test predict returns valid probabilities."""
        calibrator = VectorScaling(n_classes=num_classes)
        calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])

        probs = calibrator.predict(calibration_split["val_logits"]).probs
        check_probability(probs, dim=1)

    # NOTE: VectorScaling CAN change predictions because it applies
    # different temperature per class, unlike uniform TemperatureScaling


class TestMatrixScaling:
    """Test MatrixScaling calibrator."""

    def test_initialization(self, num_classes):
        """Test calibrator requires n_classes."""
        calibrator = MatrixScaling(n_classes=num_classes)
        assert calibrator.weight.shape == (num_classes, num_classes)
        assert calibrator.bias.shape == (num_classes,)

    def test_fit_learns_transformation(self, calibration_split, num_classes):
        """Test fit learns affine transformation."""
        calibrator = MatrixScaling(n_classes=num_classes)
        calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])
        # Should have learned weight matrix and bias
        assert calibrator.weight.shape == (num_classes, num_classes)
        assert calibrator.bias.shape == (num_classes,)

    def test_predict_shape(self, calibration_split, num_classes):
        """Test predict preserves shape."""
        calibrator = MatrixScaling(n_classes=num_classes)
        calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])

        probs = calibrator.predict(calibration_split["val_logits"]).probs

        assert probs.shape == calibration_split["val_logits"].shape

    def test_valid_probabilities(self, calibration_split, num_classes, check_probability):
        """Test predict returns valid probabilities."""
        calibrator = MatrixScaling(n_classes=num_classes)
        calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])

        probs = calibrator.predict(calibration_split["val_logits"]).probs
        check_probability(probs, dim=1)


class TestIsotonicRegressionCalibrator:
    """Test IsotonicRegressionCalibrator."""

    def test_initialization(self):
        """Test calibrator can be initialized."""
        calibrator = IsotonicRegressionCalibrator()
        assert calibrator.out_of_bounds == "clip"

        calibrator2 = IsotonicRegressionCalibrator(out_of_bounds="nan")
        assert calibrator2.out_of_bounds == "nan"

    def test_fit(self, multiclass_logits, multiclass_labels):
        """Test fit method."""
        calibrator = IsotonicRegressionCalibrator()
        calibrator.fit(multiclass_logits, multiclass_labels)

        # Should create one calibrator per class
        assert len(calibrator.calibrators) == multiclass_logits.shape[1]
        assert calibrator.n_classes == multiclass_logits.shape[1]

    def test_predict_shape(self, calibration_split):
        """Test predict preserves shape."""
        calibrator = IsotonicRegressionCalibrator()
        calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])

        probs = calibrator.predict(calibration_split["val_logits"]).probs

        assert probs.shape == calibration_split["val_logits"].shape

    def test_valid_probabilities(self, calibration_split, check_probability):
        """Test predict returns valid probabilities."""
        calibrator = IsotonicRegressionCalibrator()
        calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])

        probs = calibrator.predict(calibration_split["val_logits"]).probs
        check_probability(probs, dim=1)


class TestHistogramBinningCalibrator:
    """Test HistogramBinningCalibrator."""

    def test_initialization(self):
        """Test calibrator can be initialized."""
        calibrator = HistogramBinningCalibrator(n_bins=10)
        assert calibrator.n_bins == 10

    def test_fit(self, calibration_split):
        """Test fit method."""
        calibrator = HistogramBinningCalibrator(n_bins=10)
        calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])
        # Should have bin information
        assert len(calibrator.bin_edges) > 0
        assert len(calibrator.bin_true_rates) > 0

    def test_predict_shape(self, calibration_split):
        """Test predict preserves shape."""
        calibrator = HistogramBinningCalibrator(n_bins=10)
        calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])

        probs = calibrator.predict(calibration_split["val_logits"]).probs

        assert probs.shape == calibration_split["val_logits"].shape

    def test_valid_probabilities(self, calibration_split, check_probability):
        """Test predict returns valid probabilities."""
        calibrator = HistogramBinningCalibrator(n_bins=10)
        calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])

        probs = calibrator.predict(calibration_split["val_logits"]).probs
        check_probability(probs, dim=1)

    def test_different_bins(self, calibration_split):
        """Test different number of bins."""
        for n_bins in [5, 10, 15, 20]:
            calibrator = HistogramBinningCalibrator(n_bins=n_bins)
            calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])
            probs = calibrator.predict(calibration_split["val_logits"]).probs
            assert probs.shape == calibration_split["val_logits"].shape


class TestPlattScalingCalibrator:
    """Test PlattScalingCalibrator (logistic regression)."""

    def test_initialization(self):
        """Test calibrator can be initialized."""
        calibrator = PlattScalingCalibrator()
        assert calibrator is not None

    def test_fit(self, multiclass_logits, multiclass_labels):
        """Test fit method."""
        calibrator = PlattScalingCalibrator()
        calibrator.fit(multiclass_logits, multiclass_labels)

        # Should create one calibrator per class
        assert len(calibrator.models) == multiclass_logits.shape[1]
        assert calibrator.n_classes == multiclass_logits.shape[1]

    def test_predict_shape(self, calibration_split):
        """Test predict preserves shape."""
        calibrator = PlattScalingCalibrator()
        calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])

        probs = calibrator.predict(calibration_split["val_logits"]).probs

        assert probs.shape == calibration_split["val_logits"].shape

    def test_valid_probabilities(self, calibration_split, check_probability):
        """Test predict returns valid probabilities."""
        calibrator = PlattScalingCalibrator()
        calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])

        probs = calibrator.predict(calibration_split["val_logits"]).probs
        check_probability(probs, dim=1)


# Cross-calibrator comparison tests
class TestCalibratorComparison:
    """Tests comparing different calibrators."""

    def test_all_calibrators_work(self, calibration_split, num_classes):
        """Test all calibrators can fit and predict."""
        calibrators = [
            ("Identity", IdentityCalibrator()),
            ("Temperature", TemperatureScaling()),
            ("Vector", VectorScaling(n_classes=num_classes)),
            ("Matrix", MatrixScaling(n_classes=num_classes)),
            ("Isotonic", IsotonicRegressionCalibrator()),
            ("Histogram", HistogramBinningCalibrator(n_bins=10)),
            ("Platt", PlattScalingCalibrator()),
        ]

        for name, calibrator in calibrators:
            # Fit
            calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])
            # Predict
            dist = calibrator.predict(calibration_split["val_logits"])
            assert isinstance(
                dist, torch.distributions.Categorical
            ), f"{name} didn't return Categorical"
            assert dist.probs.shape == calibration_split["val_logits"].shape

    def test_uniform_scaling_preserves_predictions(self, calibration_split, num_classes):
        """Test uniform scaling calibrators preserve predictions."""
        # Only TemperatureScaling and Identity preserve predictions
        # VectorScaling and MatrixScaling CAN change predictions because
        # they apply different scales per class
        calibrators = [
            IdentityCalibrator(),
            TemperatureScaling(),
        ]

        preds_original = torch.argmax(calibration_split["val_logits"], dim=1)

        for calibrator in calibrators:
            calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])
            preds_calibrated = calibrator.predict(calibration_split["val_logits"]).probs.argmax(
                dim=1
            )
            assert torch.equal(
                preds_original, preds_calibrated
            ), f"{calibrator.__class__.__name__} changed predictions"

    def test_all_return_valid_probabilities(
        self, calibration_split, num_classes, check_probability
    ):
        """Test all calibrators return valid probabilities."""
        calibrators = [
            IdentityCalibrator(),
            TemperatureScaling(),
            VectorScaling(n_classes=num_classes),
            MatrixScaling(n_classes=num_classes),
            IsotonicRegressionCalibrator(),
            HistogramBinningCalibrator(n_bins=10),
            PlattScalingCalibrator(),
        ]

        for calibrator in calibrators:
            calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])
            probs = calibrator.predict(calibration_split["val_logits"]).probs
            check_probability(probs, dim=1)


# Edge case tests
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_sample(self, num_classes):
        """Test handling of single sample."""
        calibrator = TemperatureScaling()
        single_logits = torch.randn(1, num_classes)
        single_labels = torch.tensor([0])

        # Should handle single sample
        calibrator.fit(single_logits, single_labels)
        probs = calibrator.predict(single_logits).probs
        assert probs.shape == (1, num_classes)

    def test_perfect_predictions(self, num_classes):
        """Test calibration with perfect predictions."""
        # Create perfect predictions
        n_samples = 100
        logits = torch.zeros(n_samples, num_classes)
        labels = torch.randint(0, num_classes, (n_samples,))

        # Set correct class to high value
        logits[torch.arange(n_samples), labels] = 10.0

        calibrator = TemperatureScaling()
        calibrator.fit(logits, labels)
        probs = calibrator.predict(logits).probs

        assert probs.shape == (n_samples, num_classes)
        # Predictions should still be correct
        assert torch.equal(torch.argmax(probs, dim=1), labels)

    def test_uniform_predictions(self, num_classes):
        """Test calibration with uniform predictions."""
        # All predictions are uniform
        n_samples = 100
        logits = torch.zeros(n_samples, num_classes)
        labels = torch.randint(0, num_classes, (n_samples,))

        calibrator = TemperatureScaling()
        calibrator.fit(logits, labels)
        probs = calibrator.predict(logits).probs

        assert probs.shape == (n_samples, num_classes)
        # Should still produce valid probabilities
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_binary_classification(self):
        """Test calibrators on binary classification."""
        n_samples = 100
        num_classes = 2
        logits = torch.randn(n_samples, num_classes)
        labels = torch.randint(0, num_classes, (n_samples,))

        calibrator = TemperatureScaling()
        calibrator.fit(logits, labels)
        probs = calibrator.predict(logits).probs

        assert probs.shape == (n_samples, num_classes)
        assert torch.allclose(probs.sum(dim=1), torch.ones(n_samples), atol=1e-5)


class TestDirichletCalibrator:
    """Test DirichletCalibrator."""

    def test_initialization(self, num_classes):
        """Test calibrator can be initialized."""
        calibrator = DirichletCalibrator(n_classes=num_classes)
        assert calibrator.weight.shape == (num_classes, num_classes)
        assert calibrator.bias.shape == (num_classes,)

    def test_initialization_with_mu(self, num_classes):
        """Test initialization with explicit mu."""
        calibrator = DirichletCalibrator(n_classes=num_classes, mu=0.01)
        assert calibrator.mu == 0.01

    def test_fit_learns_transformation(self, calibration_split, num_classes):
        """Test fit learns Dirichlet transformation."""
        calibrator = DirichletCalibrator(n_classes=num_classes)
        calibrator.fit(
            calibration_split["train_logits"],
            calibration_split["train_labels"],
            max_iters=50,  # Reduced for faster testing
        )
        # Should have learned weight matrix and bias
        assert calibrator.weight.shape == (num_classes, num_classes)
        assert calibrator.bias.shape == (num_classes,)

    def test_predict_shape(self, calibration_split, num_classes):
        """Test predict preserves shape."""
        calibrator = DirichletCalibrator(n_classes=num_classes)
        calibrator.fit(
            calibration_split["train_logits"],
            calibration_split["train_labels"],
            max_iters=50,
        )

        probs = calibrator.predict(calibration_split["val_logits"]).probs

        assert probs.shape == calibration_split["val_logits"].shape

    def test_valid_probabilities(self, calibration_split, num_classes, check_probability):
        """Test predict returns valid probabilities."""
        calibrator = DirichletCalibrator(n_classes=num_classes)
        calibrator.fit(
            calibration_split["train_logits"],
            calibration_split["train_labels"],
            max_iters=50,
        )

        probs = calibrator.predict(calibration_split["val_logits"]).probs
        check_probability(probs, dim=1)

    def test_regularization_mu(self, calibration_split, num_classes):
        """Test different regularization strengths."""
        for mu in [None, 0.01, 0.1]:
            calibrator = DirichletCalibrator(n_classes=num_classes, mu=mu)
            calibrator.fit(
                calibration_split["train_logits"],
                calibration_split["train_labels"],
                max_iters=30,
            )
            probs = calibrator.predict(calibration_split["val_logits"]).probs
            assert probs.shape == calibration_split["val_logits"].shape


class TestBetaCalibrator:
    """Test BetaCalibrator (binary classification)."""

    def test_initialization(self):
        """Test calibrator can be initialized."""
        calibrator = BetaCalibrator()
        assert calibrator is not None

    def test_fit_binary(self):
        """Test fit on binary classification."""
        calibrator = BetaCalibrator()

        # Create binary data
        n_samples = 200
        logits = torch.randn(n_samples, 2)
        labels = torch.randint(0, 2, (n_samples,))

        calibrator.fit(logits, labels)

        # Should have fitted Beta parameters
        assert calibrator.a is not None
        assert calibrator.b is not None
        assert calibrator.c is not None
        assert calibrator.is_binary is True

    def test_predict_shape_binary(self):
        """Test predict preserves shape for binary classification."""
        calibrator = BetaCalibrator()

        # Create binary data
        n_samples = 200
        logits_train = torch.randn(n_samples, 2)
        labels_train = torch.randint(0, 2, (n_samples,))

        calibrator.fit(logits_train, labels_train)

        # Test prediction
        logits_test = torch.randn(50, 2)
        probs = calibrator.predict(logits_test).probs

        assert probs.shape == (50, 2)

    def test_valid_probabilities_binary(self, check_probability):
        """Test predict returns valid probabilities."""
        calibrator = BetaCalibrator()

        # Create binary data
        n_samples = 200
        logits_train = torch.randn(n_samples, 2)
        labels_train = torch.randint(0, 2, (n_samples,))

        calibrator.fit(logits_train, labels_train)

        # Test prediction
        logits_test = torch.randn(50, 2)
        probs = calibrator.predict(logits_test).probs

        check_probability(probs, dim=1)

    def test_multiclass_fallback_to_isotonic(self, calibration_split):
        """Test Beta calibrator falls back to Isotonic for multiclass."""
        calibrator = BetaCalibrator()

        # Fit on multiclass data (should use isotonic regression internally)
        calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])

        probs = calibrator.predict(calibration_split["val_logits"]).probs

        # Should still work and return valid probabilities
        assert probs.shape == calibration_split["val_logits"].shape
        assert (probs >= 0).all() and (probs <= 1).all()
        assert torch.allclose(probs.sum(dim=1), torch.ones(probs.shape[0]), atol=1e-5)


class TestNotFittedError:
    """Test that predict() before fit() raises NotFittedError."""

    def test_isotonic_not_fitted(self, multiclass_logits):
        calibrator = IsotonicRegressionCalibrator()
        with pytest.raises(NotFittedError):
            calibrator.predict(multiclass_logits)

    def test_histogram_not_fitted(self, multiclass_logits):
        calibrator = HistogramBinningCalibrator()
        with pytest.raises(NotFittedError):
            calibrator.predict(multiclass_logits)

    def test_platt_not_fitted(self, multiclass_logits):
        calibrator = PlattScalingCalibrator()
        with pytest.raises(NotFittedError):
            calibrator.predict(multiclass_logits)

    def test_beta_not_fitted(self):
        calibrator = BetaCalibrator()
        logits = torch.randn(10, 2)
        with pytest.raises(NotFittedError):
            calibrator.predict(logits)
