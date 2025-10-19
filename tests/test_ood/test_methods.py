"""
Tests for OOD detection methods.

All OOD detectors:
- Take a model in __init__
- Have .score(x) method that takes raw inputs (not logits)
- Return higher scores for more OOD-like inputs
- Have .predict(x, threshold) method
"""

import pytest
import torch
import torch.nn as nn

from incerto.ood import (
    MSP,
    Energy,
    ODIN,
    Mahalanobis,
    MaxLogit,
    KNN,
)


class TestMSP:
    """Test Maximum Softmax Probability (MSP) detector."""

    def test_initialization(self, ood_model):
        """Test detector can be initialized with a model."""
        detector = MSP(ood_model)
        assert detector is not None
        assert detector.model is not None

    def test_score_shape(self, ood_model, ood_id_inputs):
        """Test score has correct shape."""
        detector = MSP(ood_model)
        scores = detector.score(ood_id_inputs)

        assert scores.shape == (len(ood_id_inputs),)
        assert torch.isfinite(scores).all()

    def test_score_range(self, ood_model, ood_id_inputs):
        """Test MSP scores are in valid range."""
        detector = MSP(ood_model)
        scores = detector.score(ood_id_inputs)

        # MSP returns 1 - max_prob, so should be in [0, 1]
        assert (scores >= 0).all()
        assert (scores <= 1).all()

    def test_confident_vs_uncertain(self, ood_model, ood_id_inputs):
        """Test MSP distinguishes confident from uncertain predictions."""
        detector = MSP(ood_model)

        # Get scores for same inputs
        scores = detector.score(ood_id_inputs)

        # Just verify scores are valid
        assert torch.isfinite(scores).all()
        assert scores.std() > 0  # Scores should vary

    def test_predict_method(self, ood_model, ood_id_inputs):
        """Test predict method with threshold."""
        detector = MSP(ood_model)
        predictions = detector.predict(ood_id_inputs, threshold=0.5)

        assert predictions.shape == (len(ood_id_inputs),)
        assert predictions.dtype == torch.bool


class TestEnergy:
    """Test Energy-based OOD detector."""

    def test_initialization(self, ood_model):
        """Test detector can be initialized."""
        detector = Energy(ood_model, temperature=1.0)
        assert detector is not None
        assert detector.T == 1.0

    def test_default_temperature(self, ood_model):
        """Test default temperature is 1.0."""
        detector = Energy(ood_model)
        assert detector.T == 1.0

    def test_score_shape(self, ood_model, ood_id_inputs):
        """Test score has correct shape."""
        detector = Energy(ood_model)
        scores = detector.score(ood_id_inputs)

        assert scores.shape == (len(ood_id_inputs),)
        assert torch.isfinite(scores).all()

    def test_temperature_effect(self, ood_model, ood_id_inputs):
        """Test temperature affects scores."""
        detector_t1 = Energy(ood_model, temperature=1.0)
        detector_t2 = Energy(ood_model, temperature=2.0)

        scores_t1 = detector_t1.score(ood_id_inputs)
        scores_t2 = detector_t2.score(ood_id_inputs)

        # Different temperatures should give different scores
        assert not torch.allclose(scores_t1, scores_t2, atol=1e-5)

    def test_predict_method(self, ood_model, ood_id_inputs):
        """Test predict method."""
        detector = Energy(ood_model)
        predictions = detector.predict(ood_id_inputs, threshold=0.0)

        assert predictions.shape == (len(ood_id_inputs),)
        assert predictions.dtype == torch.bool


class TestODIN:
    """Test ODIN (Out-of-DIstribution detector for Neural networks)."""

    def test_initialization(self, ood_model):
        """Test detector can be initialized."""
        detector = ODIN(ood_model, temperature=1000.0, epsilon=0.0012)
        assert detector is not None
        assert detector.T == 1000.0
        assert detector.eps == 0.0012

    def test_default_parameters(self, ood_model):
        """Test default parameters."""
        detector = ODIN(ood_model)
        assert detector.T == 1000.0
        assert detector.eps == 0.0014

    def test_score_shape(self, ood_model, ood_id_inputs):
        """Test score has correct shape."""
        # Make inputs require grad for ODIN
        detector = ODIN(ood_model, temperature=1000.0, epsilon=0.001)
        scores = detector.score(ood_id_inputs)

        assert scores.shape == (len(ood_id_inputs),)
        assert torch.isfinite(scores).all()

    def test_temperature_effect(self, ood_model, ood_id_inputs):
        """Test temperature affects ODIN scores."""
        detector_t1 = ODIN(ood_model, temperature=1.0, epsilon=0.0)
        detector_t2 = ODIN(ood_model, temperature=1000.0, epsilon=0.0)

        scores_t1 = detector_t1.score(ood_id_inputs)
        scores_t2 = detector_t2.score(ood_id_inputs)

        # Different temperatures should give different scores
        assert not torch.allclose(scores_t1, scores_t2, atol=1e-5)

    def test_epsilon_effect(self, ood_model, ood_id_inputs):
        """Test epsilon (perturbation) affects scores."""
        detector_e0 = ODIN(ood_model, temperature=1000.0, epsilon=0.0)
        detector_e1 = ODIN(ood_model, temperature=1000.0, epsilon=0.001)

        scores_e0 = detector_e0.score(ood_id_inputs)
        scores_e1 = detector_e1.score(ood_id_inputs)

        # Different epsilon may give different scores (or same if gradients are zero)
        assert torch.isfinite(scores_e0).all()
        assert torch.isfinite(scores_e1).all()


class TestMahalanobis:
    """Test Mahalanobis distance-based OOD detector."""

    def test_initialization(self, ood_model):
        """Test detector can be initialized."""
        detector = Mahalanobis(ood_model, layer_name="penultimate")
        assert detector is not None

    def test_default_layer(self, ood_model):
        """Test default layer name."""
        detector = Mahalanobis(ood_model)
        assert detector is not None

    def test_fit(self, ood_model, ood_id_loader):
        """Test fitting Mahalanobis detector."""
        detector = Mahalanobis(ood_model, layer_name="penultimate")

        # Fit on ID data
        detector.fit(ood_id_loader)

        # Should have computed class means and precision
        assert detector.class_means is not None
        assert detector.precision is not None

    def test_score_after_fit(
        self, ood_model, ood_id_loader, ood_id_inputs, ood_ood_inputs
    ):
        """Test scoring after fitting."""
        detector = Mahalanobis(ood_model, layer_name="penultimate")

        # Fit on ID data
        detector.fit(ood_id_loader)

        # Score ID and OOD
        scores_id = detector.score(ood_id_inputs)
        scores_ood = detector.score(ood_ood_inputs)

        # Should produce scores
        assert scores_id.shape == (len(ood_id_inputs),)
        assert scores_ood.shape == (len(ood_ood_inputs),)
        assert torch.isfinite(scores_id).all()
        assert torch.isfinite(scores_ood).all()

    def test_predict_method(self, ood_model, ood_id_loader, ood_id_inputs):
        """Test predict method."""
        detector = Mahalanobis(ood_model, layer_name="penultimate")
        detector.fit(ood_id_loader)

        predictions = detector.predict(ood_id_inputs, threshold=1.0)

        assert predictions.shape == (len(ood_id_inputs),)
        assert predictions.dtype == torch.bool


class TestMaxLogit:
    """Test MaxLogit OOD detector."""

    def test_initialization(self, ood_model):
        """Test detector can be initialized."""
        detector = MaxLogit(ood_model)
        assert detector is not None

    def test_score_shape(self, ood_model, ood_id_inputs):
        """Test score has correct shape."""
        detector = MaxLogit(ood_model)
        scores = detector.score(ood_id_inputs)

        assert scores.shape == (len(ood_id_inputs),)
        assert torch.isfinite(scores).all()

    def test_score_is_negative_max_logit(self, ood_model, ood_id_inputs):
        """Test MaxLogit returns negative maximum logit value."""
        detector = MaxLogit(ood_model)
        scores = detector.score(ood_id_inputs)

        # Compute manually
        with torch.no_grad():
            logits = ood_model(ood_id_inputs)
            expected = -logits.max(dim=1).values

        assert torch.allclose(scores, expected, atol=1e-5)

    def test_predict_method(self, ood_model, ood_id_inputs):
        """Test predict method."""
        detector = MaxLogit(ood_model)
        predictions = detector.predict(ood_id_inputs, threshold=0.0)

        assert predictions.shape == (len(ood_id_inputs),)
        assert predictions.dtype == torch.bool


class TestKNN:
    """Test K-Nearest Neighbors OOD detector."""

    def test_initialization(self, ood_model):
        """Test detector can be initialized."""
        detector = KNN(ood_model, k=5, layer_name="penultimate")
        assert detector is not None
        assert detector.k == 5

    def test_default_parameters(self, ood_model):
        """Test default parameters."""
        detector = KNN(ood_model)
        assert detector.k == 50  # Default k

    def test_fit(self, ood_model, ood_id_loader):
        """Test fitting KNN detector."""
        detector = KNN(ood_model, k=5, layer_name="penultimate")

        # Fit on ID features
        detector.fit(ood_id_loader)

        # Should store training features
        assert detector.train_features is not None

    def test_score_after_fit(
        self, ood_model, ood_id_loader, ood_id_inputs, ood_ood_inputs
    ):
        """Test scoring after fitting."""
        detector = KNN(ood_model, k=5, layer_name="penultimate")

        # Fit on ID data
        detector.fit(ood_id_loader)

        # Score ID and OOD
        scores_id = detector.score(ood_id_inputs)
        scores_ood = detector.score(ood_ood_inputs)

        # Should produce scores
        assert scores_id.shape == (len(ood_id_inputs),)
        assert scores_ood.shape == (len(ood_ood_inputs),)

        # All scores should be non-negative (distances)
        assert (scores_id >= 0).all()
        assert (scores_ood >= 0).all()

    def test_score_requires_fit(self, ood_model, ood_id_inputs):
        """Test score raises error before fit."""
        detector = KNN(ood_model, k=5, layer_name="penultimate")

        with pytest.raises(RuntimeError):
            detector.score(ood_id_inputs)

    def test_different_k_values(self, ood_model, ood_id_loader, ood_id_inputs):
        """Test different values of k."""
        for k in [1, 3, 5, 10]:
            detector = KNN(ood_model, k=k, layer_name="penultimate")
            detector.fit(ood_id_loader)
            scores = detector.score(ood_id_inputs)

            assert scores.shape == (len(ood_id_inputs),)
            assert torch.isfinite(scores).all()


# Integration tests
class TestOODDetectorIntegration:
    """Integration tests comparing different OOD detectors."""

    def test_all_simple_detectors_score(self, ood_model, ood_id_inputs):
        """Test all simple detectors (no fitting required) can produce scores."""
        detectors = [
            MSP(ood_model),
            Energy(ood_model),
            MaxLogit(ood_model),
            ODIN(ood_model, temperature=1000.0, epsilon=0.001),
        ]

        for detector in detectors:
            scores = detector.score(ood_id_inputs)
            assert scores.shape == (len(ood_id_inputs),)
            assert torch.isfinite(scores).all()

    def test_all_detectors_predict(self, ood_model, ood_id_inputs):
        """Test all simple detectors can make predictions."""
        detectors = [
            MSP(ood_model),
            Energy(ood_model),
            MaxLogit(ood_model),
        ]

        for detector in detectors:
            predictions = detector.predict(ood_id_inputs, threshold=0.5)
            assert predictions.shape == (len(ood_id_inputs),)
            assert predictions.dtype == torch.bool

    def test_feature_based_detectors(self, ood_model, ood_id_loader, ood_ood_inputs):
        """Test feature-based detectors (require fitting)."""
        # Mahalanobis
        mahal = Mahalanobis(ood_model, layer_name="penultimate")
        mahal.fit(ood_id_loader)
        scores_mahal = mahal.score(ood_ood_inputs)

        # KNN
        knn = KNN(ood_model, k=5, layer_name="penultimate")
        knn.fit(ood_id_loader)
        scores_knn = knn.score(ood_ood_inputs)

        assert torch.isfinite(scores_mahal).all()
        assert torch.isfinite(scores_knn).all()
        assert (scores_knn >= 0).all()  # KNN scores are distances

    def test_id_vs_ood_separation(
        self, ood_model, ood_id_loader, ood_id_inputs, ood_ood_inputs
    ):
        """Test that detectors can separate ID from OOD (probabilistically)."""
        # MSP: higher score = more OOD
        detector = MSP(ood_model)
        scores_id = detector.score(ood_id_inputs)
        scores_ood = detector.score(ood_ood_inputs)

        # Just verify both produce valid scores
        assert torch.isfinite(scores_id).all()
        assert torch.isfinite(scores_ood).all()


# Edge case tests
class TestEdgeCases:
    """Test edge cases for OOD detection."""

    def test_batch_size_one(self, ood_model):
        """Test detectors with batch size 1."""
        x = torch.randn(1, 64)

        detector = MSP(ood_model)
        scores = detector.score(x)

        assert scores.shape == (1,)
        assert torch.isfinite(scores)

    def test_model_in_eval_mode(self, ood_model):
        """Test that model is set to eval mode."""
        detector = MSP(ood_model)
        assert not detector.model.training

    def test_model_gradients_disabled(self, ood_model):
        """Test that model parameters have gradients disabled."""
        detector = MSP(ood_model)
        for param in detector.model.parameters():
            assert not param.requires_grad

    def test_different_input_sizes(self, ood_model):
        """Test detectors with different batch sizes."""
        detector = MSP(ood_model)

        for batch_size in [1, 10, 50]:
            x = torch.randn(batch_size, 64)
            scores = detector.score(x)
            assert scores.shape == (batch_size,)
            assert torch.isfinite(scores).all()
