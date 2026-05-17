"""
Calibration methods for LLMs.

Adapt calibration techniques for language model outputs, including
temperature scaling for token distributions and sequence-level calibration.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenTemperatureScaling(nn.Module):
    """
    Temperature scaling for token-level probabilities.

    Applies a learnable temperature parameter to logits before softmax,
    making the distribution sharper (T < 1) or smoother (T > 1).
    """

    def __init__(self, init_temp: float = 1.0):
        """
        Args:
            init_temp: Initial temperature value
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temp))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling.

        Args:
            logits: Token logits of shape (..., vocab_size)

        Returns:
            Temperature-scaled logits
        """
        return logits / self.temperature.clamp(min=1e-6)

    def fit(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        lr: float = 0.01,
        max_iters: int = 50,
    ):
        """
        Fit temperature on validation data.

        Args:
            logits: Validation logits of shape (batch, seq_len, vocab_size)
            token_ids: True token IDs of shape (batch, seq_len)
            lr: Learning rate
            max_iters: Maximum optimization iterations
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iters)

        def eval_fn():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                scaled_logits.view(-1, scaled_logits.size(-1)), token_ids.view(-1)
            )
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        return self


class SequenceLengthCalibration:
    """
    Calibrate for length bias in sequence probabilities.

    Longer sequences tend to have lower probabilities. This adjusts
    for that bias using length normalization.
    """

    def __init__(self, alpha: float = 0.6):
        """
        Args:
            alpha: Length penalty factor (common range: 0.5-1.0)
        """
        self.alpha = alpha

    def calibrate(
        self,
        seq_log_prob: torch.Tensor,
        seq_length: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply length normalization.

        Args:
            seq_log_prob: Log probability of sequence (batch,)
            seq_length: Length of sequence (batch,)

        Returns:
            Length-normalized scores
        """
        # Divide by length^alpha
        normalized = seq_log_prob / (seq_length.float() ** self.alpha)
        return normalized


class VerbosityBiasCorrection:
    """
    Correct for the model's tendency to be more confident on verbose outputs.

    Some models produce higher probabilities when generating longer,
    more detailed responses, even if they're not more accurate.
    """

    def __init__(self):
        self.mean_confidence = None
        self.bin_edges = None
        self.bin_mean_confidences = None

    def fit(self, lengths: list[int], confidences: list[float]):
        """
        Fit correction based on length-confidence relationship.

        Args:
            lengths: List of response lengths
            confidences: List of confidence scores
        """
        import numpy as np

        length_array = np.array(lengths, dtype=float)
        conf_array = np.array(confidences, dtype=float)

        self.mean_confidence = conf_array.mean()

        # Create quartile bins by length
        self.bin_edges = np.percentile(length_array, [0, 25, 50, 75, 100])
        self.bin_mean_confidences = []

        for i in range(len(self.bin_edges) - 1):
            if i < len(self.bin_edges) - 2:
                mask = (length_array >= self.bin_edges[i]) & (length_array < self.bin_edges[i + 1])
            else:
                mask = (length_array >= self.bin_edges[i]) & (length_array <= self.bin_edges[i + 1])
            if mask.sum() > 0:
                self.bin_mean_confidences.append(conf_array[mask].mean())
            else:
                self.bin_mean_confidences.append(self.mean_confidence)

    def correct(self, length: int, confidence: float) -> float:
        """
        Apply verbosity bias correction.

        Args:
            length: Response length
            confidence: Original confidence score

        Returns:
            Corrected confidence
        """
        if self.bin_edges is None:
            return confidence

        import numpy as np

        # Find which length bin this falls into
        bin_idx = np.digitize(length, self.bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, len(self.bin_mean_confidences) - 1)

        # Correct by the ratio of overall mean confidence to bin mean confidence
        bin_mean = self.bin_mean_confidences[bin_idx]
        if bin_mean > 0:
            corrected = confidence * (self.mean_confidence / bin_mean)
        else:
            corrected = confidence

        return max(0.0, min(1.0, corrected))


class HistogramBinning:
    """
    Histogram binning calibration for LLM confidence scores.

    Groups predictions by confidence and adjusts to empirical accuracy.
    """

    def __init__(self, n_bins: int = 10):
        """
        Args:
            n_bins: Number of bins for calibration
        """
        self.n_bins = n_bins
        self.bin_boundaries = None
        self.bin_accuracies = None

    def fit(self, confidences: torch.Tensor, correctness: torch.Tensor):
        """
        Fit binning calibration.

        Args:
            confidences: Model confidence scores (batch,)
            correctness: Binary correctness indicators (batch,)
        """
        import numpy as np

        confidences = confidences.cpu().numpy()
        correctness = correctness.cpu().numpy()

        # Create bins
        self.bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        self.bin_accuracies = np.zeros(self.n_bins)

        # Compute empirical accuracy in each bin
        for i in range(self.n_bins):
            lower = self.bin_boundaries[i]
            upper = self.bin_boundaries[i + 1]

            in_bin = (confidences >= lower) & (confidences < upper)
            if i == self.n_bins - 1:  # Include upper boundary in last bin
                in_bin = (confidences >= lower) & (confidences <= upper)

            if in_bin.sum() > 0:
                self.bin_accuracies[i] = correctness[in_bin].mean()
            else:
                self.bin_accuracies[i] = (lower + upper) / 2  # Default to bin center

    def calibrate(self, confidence: float) -> float:
        """
        Apply calibration to a confidence score.

        Args:
            confidence: Original confidence (0-1)

        Returns:
            Calibrated confidence
        """
        if self.bin_boundaries is None:
            return confidence

        # Find bin using numpy digitize for consistency with fit()
        import numpy as np

        # np.digitize returns bin index where bin_boundaries[i-1] <= x < bin_boundaries[i]
        # We subtract 1 to get 0-indexed bins, and clip to valid range
        bin_idx = np.digitize(confidence, self.bin_boundaries) - 1
        bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)

        return float(self.bin_accuracies[bin_idx])
