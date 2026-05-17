"""
Tests for LLM uncertainty metrics.
"""

import pytest
import torch

from incerto.llm import (
    aur_c,
    brier_score,
    calibration_error,
    f1_score_tokens,
    selective_accuracy,
    sequence_level_accuracy,
    token_level_accuracy,
    uncertainty_auc,
)


class TestSelectiveAccuracy:
    """Test selective accuracy computation."""

    def test_all_selected(self):
        """Test when all predictions are selected."""
        predictions = torch.tensor([0, 1, 2, 0, 1])
        targets = torch.tensor([0, 1, 2, 0, 1])
        confidences = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])

        result = selective_accuracy(predictions, targets, confidences, threshold=0.4)

        assert result["accuracy"] == 1.0
        assert result["coverage"] == 1.0
        assert result["n_selected"] == 5

    def test_none_selected(self):
        """Test when no predictions meet threshold."""
        predictions = torch.tensor([0, 1, 2])
        targets = torch.tensor([0, 1, 2])
        confidences = torch.tensor([0.3, 0.2, 0.1])

        result = selective_accuracy(predictions, targets, confidences, threshold=0.5)

        assert result["accuracy"] == 0.0
        assert result["coverage"] == 0.0
        assert result["n_selected"] == 0

    def test_partial_selection(self):
        """Test partial selection with mixed accuracy."""
        predictions = torch.tensor([0, 1, 2, 3])
        targets = torch.tensor([0, 1, 0, 0])  # 2 correct, 2 wrong
        confidences = torch.tensor([0.9, 0.8, 0.3, 0.2])

        result = selective_accuracy(predictions, targets, confidences, threshold=0.5)

        # Only first 2 selected (conf >= 0.5), both correct
        assert result["accuracy"] == 1.0
        assert result["coverage"] == 0.5
        assert result["n_selected"] == 2


class TestCalibrationError:
    """Test calibration error computation."""

    def test_perfect_calibration(self):
        """Test perfectly calibrated predictions."""
        # 10 bins, each with accuracy matching confidence
        confidences = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        # At confidence 0.1, 10% correct (rounded to 0 or 1)
        correctness = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])

        result = calibration_error(confidences, correctness, n_bins=5)

        # ECE should be low for roughly calibrated predictions
        assert "ece" in result
        assert "mce" in result
        assert result["ece"] >= 0
        assert result["mce"] >= 0

    def test_overconfident(self):
        """Test overconfident predictions have positive ECE."""
        # High confidence, low accuracy
        confidences = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.9])
        correctness = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])  # 20% accuracy

        result = calibration_error(confidences, correctness, n_bins=5)

        # ECE should be high (0.9 - 0.2 = 0.7)
        assert result["ece"] > 0.5

    def test_underconfident(self):
        """Test underconfident predictions."""
        # Low confidence, high accuracy
        confidences = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1])
        correctness = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])  # 80% accuracy

        result = calibration_error(confidences, correctness, n_bins=5)

        # ECE should be significant
        assert result["ece"] > 0.5


class TestBrierScore:
    """Test Brier score computation."""

    def test_perfect_predictions(self):
        """Test Brier score is 0 for perfect predictions."""
        confidences = torch.tensor([1.0, 0.0, 1.0, 0.0])
        correctness = torch.tensor([1.0, 0.0, 1.0, 0.0])

        score = brier_score(confidences, correctness)
        assert score == pytest.approx(0.0)

    def test_worst_predictions(self):
        """Test Brier score is 1 for worst predictions."""
        confidences = torch.tensor([0.0, 1.0, 0.0, 1.0])
        correctness = torch.tensor([1.0, 0.0, 1.0, 0.0])

        score = brier_score(confidences, correctness)
        assert score == pytest.approx(1.0)

    def test_uniform_predictions(self):
        """Test Brier score for uniform predictions."""
        confidences = torch.tensor([0.5, 0.5, 0.5, 0.5])
        correctness = torch.tensor([1.0, 0.0, 1.0, 0.0])

        score = brier_score(confidences, correctness)
        assert score == pytest.approx(0.25)


class TestAURC:
    """Test Area Under Risk-Coverage curve."""

    def test_perfect_ranking(self):
        """Test AURC is low for perfect uncertainty ranking."""
        # Higher confidence = always correct
        confidences = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        correctness = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

        aurc = aur_c(confidences, correctness)

        # AURC should be relatively low for good ranking
        assert 0 <= aurc <= 1

    def test_random_ranking(self):
        """Test AURC for random uncertainty ranking."""
        # Confidence doesn't correlate with correctness
        confidences = torch.tensor([0.9, 0.2, 0.8, 0.1, 0.7, 0.3])
        correctness = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

        aurc = aur_c(confidences, correctness)
        assert 0 <= aurc <= 1


class TestUncertaintyAUC:
    """Test AUC for uncertainty-based filtering."""

    def test_perfect_uncertainty(self):
        """Test AUC is 1 when uncertainty perfectly predicts errors."""
        # High uncertainty = incorrect
        uncertainties = torch.tensor([0.1, 0.2, 0.9, 0.8])
        correctness = torch.tensor([1.0, 1.0, 0.0, 0.0])

        auc = uncertainty_auc(uncertainties, correctness)
        assert auc == pytest.approx(1.0)

    def test_inverse_uncertainty(self):
        """Test AUC is 0 when uncertainty is inverted."""
        # High uncertainty = correct (wrong direction)
        uncertainties = torch.tensor([0.9, 0.8, 0.1, 0.2])
        correctness = torch.tensor([1.0, 1.0, 0.0, 0.0])

        auc = uncertainty_auc(uncertainties, correctness)
        assert auc == pytest.approx(0.0)

    def test_all_correct(self):
        """Test AUC is 0.5 when all predictions are correct."""
        uncertainties = torch.tensor([0.5, 0.6, 0.4, 0.7])
        correctness = torch.tensor([1.0, 1.0, 1.0, 1.0])

        auc = uncertainty_auc(uncertainties, correctness)
        # With no errors, AUC defaults to 0.5
        assert auc == 0.5


class TestTokenLevelAccuracy:
    """Test token-level accuracy computation."""

    def test_perfect_accuracy(self):
        """Test accuracy is 1.0 for perfect predictions."""
        pred = torch.tensor([[1, 2, 3], [4, 5, 6]])
        true = torch.tensor([[1, 2, 3], [4, 5, 6]])

        acc = token_level_accuracy(pred, true)
        assert acc == 1.0

    def test_zero_accuracy(self):
        """Test accuracy is 0.0 for all wrong predictions."""
        pred = torch.tensor([[1, 2, 3], [4, 5, 6]])
        true = torch.tensor([[7, 8, 9], [10, 11, 12]])

        acc = token_level_accuracy(pred, true)
        assert acc == 0.0

    def test_partial_accuracy(self):
        """Test partial accuracy."""
        pred = torch.tensor([[1, 2, 3, 4]])
        true = torch.tensor([[1, 2, 0, 0]])

        acc = token_level_accuracy(pred, true)
        assert acc == 0.5  # 2/4

    def test_with_mask(self):
        """Test accuracy with mask."""
        pred = torch.tensor([[1, 2, 3, 4]])
        true = torch.tensor([[1, 0, 0, 4]])
        mask = torch.tensor([[1, 0, 0, 1]])  # Only check first and last

        acc = token_level_accuracy(pred, true, mask=mask)
        assert acc == 1.0  # Both masked positions correct

    def test_empty_mask(self):
        """Test with all-zero mask."""
        pred = torch.tensor([[1, 2, 3]])
        true = torch.tensor([[4, 5, 6]])
        mask = torch.tensor([[0, 0, 0]])

        acc = token_level_accuracy(pred, true, mask=mask)
        assert acc == 0.0


class TestSequenceLevelAccuracy:
    """Test sequence-level accuracy computation."""

    def test_exact_match(self):
        """Test exact match accuracy."""
        pred = ["Paris", "London", "Berlin"]
        true = ["Paris", "London", "Berlin"]

        acc = sequence_level_accuracy(pred, true)
        assert acc == 1.0

    def test_no_match(self):
        """Test no matches."""
        pred = ["Paris", "London", "Berlin"]
        true = ["Rome", "Madrid", "Vienna"]

        acc = sequence_level_accuracy(pred, true)
        assert acc == 0.0

    def test_partial_match(self):
        """Test partial matches."""
        pred = ["Paris", "London", "Berlin"]
        true = ["Paris", "Madrid", "Berlin"]

        acc = sequence_level_accuracy(pred, true)
        assert acc == pytest.approx(2 / 3)

    def test_normalization(self):
        """Test normalization (lowercase, strip)."""
        pred = ["PARIS", "London ", "  Berlin"]
        true = ["paris", "london", "berlin"]

        acc = sequence_level_accuracy(pred, true, normalize=True)
        assert acc == 1.0

    def test_no_normalization(self):
        """Test without normalization."""
        pred = ["PARIS", "London"]
        true = ["paris", "London"]

        acc = sequence_level_accuracy(pred, true, normalize=False)
        assert acc == 0.5  # Only "London" matches exactly


class TestF1ScoreTokens:
    """Test token-level F1 score computation."""

    def test_perfect_f1(self):
        """Test F1 is 1.0 for perfect predictions with full mask."""
        pred = torch.tensor([[1, 2, 3]])
        true = torch.tensor([[1, 2, 3]])

        result = f1_score_tokens(pred, true)

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_with_errors(self):
        """Test F1 with some errors."""
        pred = torch.tensor([[1, 2, 0, 0]])  # 2 correct, 2 wrong
        true = torch.tensor([[1, 2, 3, 4]])

        result = f1_score_tokens(pred, true)

        # TP=2, FP=2, FN=0 (full mask)
        assert result["precision"] == 0.5
        assert result["recall"] == 1.0  # FN=0 with full mask
        assert result["f1"] == pytest.approx(2 / 3)

    def test_with_mask(self):
        """Test F1 with mask creating false negatives."""
        pred = torch.tensor([[1, 2, 3, 4]])
        true = torch.tensor([[1, 2, 3, 4]])
        mask = torch.tensor([[1, 1, 0, 0]])  # Only first 2 evaluated

        result = f1_score_tokens(pred, true, mask=mask)

        # TP=2 (correct in mask), FP=0, FN=2 (masked out)
        assert result["tp"] == 2
        assert result["fp"] == 0
        assert result["fn"] == 2
        assert result["precision"] == 1.0
        assert result["recall"] == 0.5
        assert result["f1"] == pytest.approx(2 / 3)

    def test_all_wrong(self):
        """Test F1 when all predictions are wrong."""
        pred = torch.tensor([[0, 0, 0]])
        true = torch.tensor([[1, 2, 3]])

        result = f1_score_tokens(pred, true)

        assert result["precision"] == 0.0
        assert result["f1"] == 0.0
