"""
T6: Parametrized tests for duplicated patterns across modules.

These tests consolidate common API patterns that are repeated across
multiple classes (calibrators, OOD detectors, etc.).
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from incerto.calibration import (
    DirichletCalibrator,
    HistogramBinningCalibrator,
    IdentityCalibrator,
    IsotonicRegressionCalibrator,
    MatrixScaling,
    PlattScalingCalibrator,
    TemperatureScaling,
    VectorScaling,
)
from incerto.ood import MSP, ODIN, Energy, MaxLogit

# ---------------------------------------------------------------------------
# Calibrator parametrized tests
# ---------------------------------------------------------------------------


def _make_calibrators(num_classes):
    """Create all calibrator instances for parametrized tests."""
    return [
        pytest.param(IdentityCalibrator(), id="Identity"),
        pytest.param(TemperatureScaling(), id="Temperature"),
        pytest.param(VectorScaling(n_classes=num_classes), id="Vector"),
        pytest.param(MatrixScaling(n_classes=num_classes), id="Matrix"),
        pytest.param(IsotonicRegressionCalibrator(), id="Isotonic"),
        pytest.param(HistogramBinningCalibrator(n_bins=10), id="Histogram"),
        pytest.param(PlattScalingCalibrator(), id="Platt"),
        pytest.param(DirichletCalibrator(n_classes=num_classes, mu=0.01), id="Dirichlet"),
    ]


@pytest.fixture(params=_make_calibrators(10))
def calibrator(request):
    return request.param


class TestAllCalibratorsParametrized:
    """Parametrized tests that all calibrators must pass."""

    def test_fit_returns_self(self, calibrator, calibration_split):
        """All calibrators should return self from fit()."""
        result = calibrator.fit(
            calibration_split["train_logits"], calibration_split["train_labels"]
        )
        assert result is calibrator

    def test_predict_shape(self, calibrator, calibration_split):
        """All calibrators should preserve shape."""
        calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])
        probs = calibrator.predict(calibration_split["val_logits"]).probs
        assert probs.shape == calibration_split["val_logits"].shape

    def test_predict_returns_categorical(self, calibrator, calibration_split):
        """All calibrators should return Categorical distribution."""
        calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])
        dist = calibrator.predict(calibration_split["val_logits"])
        assert isinstance(dist, torch.distributions.Categorical)

    def test_valid_probabilities(self, calibrator, calibration_split, check_probability):
        """All calibrators should return valid probability distributions."""
        calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])
        probs = calibrator.predict(calibration_split["val_logits"]).probs
        check_probability(probs, dim=1)

    def test_finite_output(self, calibrator, calibration_split):
        """All calibrators should produce finite values."""
        calibrator.fit(calibration_split["train_logits"], calibration_split["train_labels"])
        probs = calibrator.predict(calibration_split["val_logits"]).probs
        assert torch.isfinite(probs).all()


# ---------------------------------------------------------------------------
# Simple OOD detector parametrized tests (no fitting required)
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        pytest.param("MSP", id="MSP"),
        pytest.param("Energy", id="Energy"),
        pytest.param("MaxLogit", id="MaxLogit"),
        pytest.param("ODIN", id="ODIN"),
    ]
)
def simple_ood_detector(request, ood_model):
    """Create simple OOD detectors (no fitting required)."""
    name = request.param
    if name == "MSP":
        return MSP(ood_model)
    elif name == "Energy":
        return Energy(ood_model)
    elif name == "MaxLogit":
        return MaxLogit(ood_model)
    elif name == "ODIN":
        return ODIN(ood_model, temperature=1000.0, epsilon=0.001)


class TestAllSimpleOODDetectorsParametrized:
    """Parametrized tests for simple OOD detectors."""

    def test_score_shape(self, simple_ood_detector, ood_id_inputs):
        """All detectors should return 1D scores matching batch size."""
        scores = simple_ood_detector.score(ood_id_inputs)
        assert scores.shape == (len(ood_id_inputs),)

    def test_score_finite(self, simple_ood_detector, ood_id_inputs):
        """All detectors should produce finite scores."""
        scores = simple_ood_detector.score(ood_id_inputs)
        assert torch.isfinite(scores).all()

    def test_predict_returns_bool(self, simple_ood_detector, ood_id_inputs):
        """All detectors should return boolean predictions."""
        preds = simple_ood_detector.predict(ood_id_inputs, threshold=0.5)
        assert preds.shape == (len(ood_id_inputs),)
        assert preds.dtype == torch.bool

    def test_batch_size_one(self, simple_ood_detector):
        """All detectors should handle batch size 1."""
        x = torch.randn(1, 64)
        scores = simple_ood_detector.score(x)
        assert scores.shape == (1,)

    def test_different_batch_sizes(self, simple_ood_detector):
        """All detectors should handle different batch sizes."""
        for bs in [1, 5, 20]:
            x = torch.randn(bs, 64)
            scores = simple_ood_detector.score(x)
            assert scores.shape == (bs,)
            assert torch.isfinite(scores).all()


# ---------------------------------------------------------------------------
# Bayesian method parametrized tests
# ---------------------------------------------------------------------------


class SimpleBayesModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


@pytest.mark.parametrize(
    "make_predictor",
    [
        pytest.param(
            lambda: __import__("incerto.bayesian", fromlist=["MCDropout"]).MCDropout(
                SimpleBayesModel(), num_samples=10
            ),
            id="MCDropout",
        ),
        pytest.param(
            lambda: __import__(
                "incerto.bayesian", fromlist=["VariationalBayesNN"]
            ).VariationalBayesNN(10, [20], 5),
            id="VariationalBayesNN",
        ),
    ],
)
class TestBayesianPredictors:
    """Parametrized tests for Bayesian predictors that don't need fitting."""

    def test_predict_returns_mean_variance(self, make_predictor):
        """All Bayesian methods should return (mean, variance) 2-tuple."""
        predictor = make_predictor()
        x = torch.randn(8, 10)
        result = predictor.predict(x)
        assert len(result) == 2
        mean, var = result
        assert mean.shape == (8, 5)
        assert var.shape == (8, 5)
        assert (var >= 0).all()

    def test_return_samples(self, make_predictor):
        """All Bayesian methods should support return_samples."""
        predictor = make_predictor()
        x = torch.randn(8, 10)
        result = predictor.predict(x, return_samples=True)
        assert len(result) == 3
        mean, var, samples = result
        assert samples.dim() == 3
        assert samples.shape[1:] == (8, 5)
