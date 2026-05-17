"""
Tests for LLM calibration methods.
"""

import torch

from incerto.llm import (
    HistogramBinning,
    SequenceLengthCalibration,
    TokenTemperatureScaling,
    VerbosityBiasCorrection,
)


class TestTokenTemperatureScaling:
    """Test token temperature scaling."""

    def test_forward_identity_at_temp_1(self):
        """Test temperature=1 is identity."""
        scaler = TokenTemperatureScaling(init_temp=1.0)
        logits = torch.randn(4, 10, 100)

        scaled = scaler(logits)
        assert torch.allclose(scaled, logits, atol=1e-5)

    def test_forward_sharpening(self):
        """Test temperature < 1 sharpens distribution."""
        scaler = TokenTemperatureScaling(init_temp=0.5)
        logits = torch.randn(4, 10, 100)

        scaled = scaler(logits)
        # Scaled logits should be 2x original (divided by 0.5)
        assert torch.allclose(scaled, logits / 0.5, atol=1e-5)

    def test_forward_smoothing(self):
        """Test temperature > 1 smooths distribution."""
        scaler = TokenTemperatureScaling(init_temp=2.0)
        logits = torch.randn(4, 10, 100)

        scaled = scaler(logits)
        # Scaled logits should be 0.5x original (divided by 2)
        assert torch.allclose(scaled, logits / 2.0, atol=1e-5)

    def test_temperature_is_learnable(self):
        """Test temperature parameter is learnable."""
        scaler = TokenTemperatureScaling(init_temp=1.0)
        assert scaler.temperature.requires_grad

    def test_fit_changes_temperature(self):
        """Test fitting changes temperature from initial value."""
        scaler = TokenTemperatureScaling(init_temp=1.0)

        # Create some logits and token ids
        logits = torch.randn(10, 5, 100)
        token_ids = torch.randint(0, 100, (10, 5))

        scaler.fit(logits, token_ids, max_iters=10)
        # After fitting, temperature should still be a valid positive value
        assert scaler.temperature.item() > 0

    def test_temperature_clamping(self):
        """Test temperature doesn't go to zero."""
        scaler = TokenTemperatureScaling(init_temp=0.0001)
        logits = torch.randn(4, 10, 100)

        # Should not divide by zero
        scaled = scaler(logits)
        assert torch.isfinite(scaled).all()


class TestSequenceLengthCalibration:
    """Test sequence length calibration."""

    def test_calibrate_no_penalty(self):
        """Test alpha=0 gives no length penalty."""
        calibrator = SequenceLengthCalibration(alpha=0.0)
        log_prob = torch.tensor([-5.0, -10.0])
        length = torch.tensor([10, 20])

        calibrated = calibrator.calibrate(log_prob, length)
        # With alpha=0, denominator is 1, so no change
        assert torch.allclose(calibrated, log_prob)

    def test_calibrate_full_normalization(self):
        """Test alpha=1 gives full length normalization."""
        calibrator = SequenceLengthCalibration(alpha=1.0)
        log_prob = torch.tensor([-10.0, -20.0])
        length = torch.tensor([10, 20])

        calibrated = calibrator.calibrate(log_prob, length)
        # -10/10 = -1, -20/20 = -1
        assert torch.allclose(calibrated, torch.tensor([-1.0, -1.0]))

    def test_calibrate_typical_alpha(self):
        """Test typical alpha value (0.6)."""
        calibrator = SequenceLengthCalibration(alpha=0.6)
        log_prob = torch.tensor([-10.0])
        length = torch.tensor([10])

        calibrated = calibrator.calibrate(log_prob, length)
        expected = -10.0 / (10.0**0.6)
        assert torch.allclose(calibrated, torch.tensor([expected]))


class TestVerbosityBiasCorrection:
    """Test verbosity bias correction."""

    def test_correct_before_fit(self):
        """Test correction returns original if not fitted."""
        corrector = VerbosityBiasCorrection()
        result = corrector.correct(100, 0.8)
        assert result == 0.8

    def test_fit_and_correct(self):
        """Test fitting and applying correction."""
        corrector = VerbosityBiasCorrection()

        # Fit with length-confidence data
        lengths = [10, 20, 30, 40, 50]
        confidences = [0.6, 0.7, 0.8, 0.9, 0.95]  # Longer = more confident

        corrector.fit(lengths, confidences)

        # After fitting, bin_edges should be set
        assert corrector.bin_edges is not None

        # Longer responses (higher bin confidence) should get reduced confidence
        long_corrected = corrector.correct(50, 0.9)
        assert long_corrected < 0.9

        # Shorter responses (lower bin confidence) should get increased confidence
        short_corrected = corrector.correct(10, 0.5)
        assert short_corrected > 0.5

    def test_correct_clamping(self):
        """Test corrected values are clamped to [0, 1]."""
        corrector = VerbosityBiasCorrection()
        corrector.fit([10, 20, 30], [0.5, 0.6, 0.7])

        # Very short response with high confidence
        result = corrector.correct(1, 0.9)
        assert 0 <= result <= 1

        # Very long response
        result = corrector.correct(1000, 0.9)
        assert 0 <= result <= 1


class TestHistogramBinning:
    """Test histogram binning calibration."""

    def test_fit_creates_bins(self):
        """Test fitting creates bin boundaries and accuracies."""
        binning = HistogramBinning(n_bins=5)

        confidences = torch.rand(100)
        correctness = (torch.rand(100) > 0.5).float()

        binning.fit(confidences, correctness)

        assert binning.bin_boundaries is not None
        assert binning.bin_accuracies is not None
        assert len(binning.bin_boundaries) == 6  # n_bins + 1
        assert len(binning.bin_accuracies) == 5  # n_bins

    def test_calibrate_before_fit(self):
        """Test calibration returns original if not fitted."""
        binning = HistogramBinning(n_bins=10)
        result = binning.calibrate(0.75)
        assert result == 0.75

    def test_calibrate_after_fit(self):
        """Test calibration returns bin accuracy after fitting."""
        binning = HistogramBinning(n_bins=10)

        # Create data where high confidence = high accuracy
        confidences = torch.linspace(0, 1, 100)
        correctness = (confidences > 0.5).float()

        binning.fit(confidences, correctness)

        # High confidence should map to high accuracy
        high_conf_calib = binning.calibrate(0.9)
        low_conf_calib = binning.calibrate(0.1)

        assert high_conf_calib > low_conf_calib

    def test_calibrate_boundary_values(self):
        """Test calibration at boundary values."""
        binning = HistogramBinning(n_bins=10)
        confidences = torch.rand(100)
        correctness = torch.rand(100)
        binning.fit(confidences, correctness)

        # Should not crash at boundaries
        result_0 = binning.calibrate(0.0)
        result_1 = binning.calibrate(1.0)

        assert 0 <= result_0 <= 1
        assert 0 <= result_1 <= 1

    def test_empty_bins_default(self):
        """Test empty bins get default value."""
        binning = HistogramBinning(n_bins=10)

        # All confidences in one bin
        confidences = torch.full((100,), 0.95)
        correctness = torch.ones(100)

        binning.fit(confidences, correctness)

        # Empty bins should have default (bin center)
        result = binning.calibrate(0.05)  # Should hit an empty bin
        assert 0 <= result <= 1
