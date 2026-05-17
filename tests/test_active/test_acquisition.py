"""
Tests for acquisition functions.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from incerto.active.acquisition import (
    BALDAcquisition,
    BatchBALDAcquisition,
    EntropyAcquisition,
    LeastConfidenceAcquisition,
    MarginAcquisition,
    MeanSTDAcquisition,
    RandomAcquisition,
    VarianceRatioAcquisition,
)


class SimpleModel(nn.Module):
    """Simple model for testing (no dropout)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)

    def forward(self, x):
        return self.fc(x)


class DropoutModel(nn.Module):
    """Model with dropout for MC-sampling tests."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(20, 3)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


@pytest.fixture
def simple_model():
    return SimpleModel()


@pytest.fixture
def dropout_model():
    return DropoutModel()


@pytest.fixture
def test_data():
    return torch.randn(20, 10)


# ---------------------------------------------------------------------------
# RandomAcquisition
# ---------------------------------------------------------------------------


class TestRandomAcquisition:
    def test_score_shape_and_range(self, simple_model, test_data):
        acq = RandomAcquisition()
        scores = acq.score(simple_model, test_data)

        assert scores.shape == (20,)
        assert torch.all(scores >= 0) and torch.all(scores <= 1)

    def test_score_varies(self, simple_model, test_data):
        acq = RandomAcquisition()
        scores = acq.score(simple_model, test_data)
        assert scores.std() > 0


# ---------------------------------------------------------------------------
# EntropyAcquisition
# ---------------------------------------------------------------------------


class TestEntropyAcquisition:
    def test_score_shape_and_nonneg(self, simple_model, test_data):
        acq = EntropyAcquisition()
        scores = acq.score(simple_model, test_data)

        assert scores.shape == (20,)
        assert torch.all(scores >= 0)

    def test_uniform_gives_max_entropy(self):
        """Uniform distribution should give maximum entropy."""

        class UniformModel(nn.Module):
            def forward(self, x):
                return torch.zeros(x.size(0), 4)  # equal logits → uniform

        model = UniformModel()
        data = torch.randn(5, 10)
        scores = EntropyAcquisition().score(model, data)

        expected = -4 * (0.25 * torch.log(torch.tensor(0.25)))
        assert torch.allclose(scores, expected.expand(5), atol=1e-5)

    def test_confident_gives_low_entropy(self):
        """Highly confident distribution → near-zero entropy."""

        class ConfidentModel(nn.Module):
            def forward(self, x):
                logits = torch.zeros(x.size(0), 3)
                logits[:, 0] = 100.0  # very confident
                return logits

        model = ConfidentModel()
        data = torch.randn(5, 10)
        scores = EntropyAcquisition().score(model, data)

        assert torch.all(scores < 0.01)


# ---------------------------------------------------------------------------
# LeastConfidenceAcquisition
# ---------------------------------------------------------------------------


class TestLeastConfidenceAcquisition:
    def test_score_shape_and_range(self, simple_model, test_data):
        acq = LeastConfidenceAcquisition()
        scores = acq.score(simple_model, test_data)

        assert scores.shape == (20,)
        assert torch.all(scores >= 0) and torch.all(scores <= 1)

    def test_confident_model_gives_low_scores(self):
        class ConfidentModel(nn.Module):
            def forward(self, x):
                logits = torch.zeros(x.size(0), 3)
                logits[:, 0] = 100.0
                return logits

        model = ConfidentModel()
        data = torch.randn(5, 10)
        scores = LeastConfidenceAcquisition().score(model, data)
        assert torch.all(scores < 0.01)

    def test_uniform_gives_high_scores(self):
        class UniformModel(nn.Module):
            def forward(self, x):
                return torch.zeros(x.size(0), 3)

        model = UniformModel()
        data = torch.randn(5, 10)
        scores = LeastConfidenceAcquisition().score(model, data)
        # 1 - 1/3 ≈ 0.667
        assert torch.allclose(scores, torch.tensor(1 - 1 / 3).expand(5), atol=1e-5)


# ---------------------------------------------------------------------------
# MarginAcquisition
# ---------------------------------------------------------------------------


class TestMarginAcquisition:
    def test_score_shape(self, simple_model, test_data):
        acq = MarginAcquisition()
        scores = acq.score(simple_model, test_data)
        assert scores.shape == (20,)

    def test_score_range(self, simple_model, test_data):
        scores = MarginAcquisition().score(simple_model, test_data)
        # Negative margin: in [-1, 0]
        assert torch.all(scores <= 0) and torch.all(scores >= -1)

    def test_single_class_returns_zeros(self):
        """Single-class output should return zeros (no margin possible)."""

        class SingleClassModel(nn.Module):
            def forward(self, x):
                return torch.ones(x.size(0), 1)

        model = SingleClassModel()
        data = torch.randn(5, 10)
        scores = MarginAcquisition().score(model, data)
        assert scores.shape == (5,)
        assert torch.allclose(scores, torch.zeros(5))

    def test_equal_top2_gives_zero(self):
        """When top-2 probs are equal, margin is 0 → score is 0."""

        class EqualTop2Model(nn.Module):
            def forward(self, x):
                logits = torch.zeros(x.size(0), 3)
                logits[:, 0] = 1.0
                logits[:, 1] = 1.0
                logits[:, 2] = -100.0
                return logits

        model = EqualTop2Model()
        data = torch.randn(5, 10)
        scores = MarginAcquisition().score(model, data)
        assert torch.allclose(scores, torch.zeros(5), atol=1e-4)


# ---------------------------------------------------------------------------
# BALDAcquisition
# ---------------------------------------------------------------------------


class TestBALDAcquisition:
    def test_init(self):
        acq = BALDAcquisition(num_samples=10)
        assert acq.num_samples == 10

    def test_score_shape(self, dropout_model, test_data):
        acq = BALDAcquisition(num_samples=5)
        scores = acq.score(dropout_model, test_data)

        assert scores.shape == (20,)
        assert torch.all(scores >= -1e-5)

    def test_restores_model_state(self, dropout_model, test_data):
        """Model should be left in its original training state."""
        dropout_model.eval()
        BALDAcquisition(num_samples=3).score(dropout_model, test_data)
        assert not dropout_model.training

        dropout_model.train()
        BALDAcquisition(num_samples=3).score(dropout_model, test_data)
        assert dropout_model.training


# ---------------------------------------------------------------------------
# VarianceRatioAcquisition
# ---------------------------------------------------------------------------


class TestVarianceRatioAcquisition:
    def test_score_shape_and_range(self, dropout_model, test_data):
        acq = VarianceRatioAcquisition(num_samples=5)
        scores = acq.score(dropout_model, test_data)

        assert scores.shape == (20,)
        assert torch.all(scores >= 0) and torch.all(scores <= 1)

    def test_deterministic_model_gives_zero(self, simple_model, test_data):
        """A model with no dropout → all MC samples identical → ratio 0."""
        acq = VarianceRatioAcquisition(num_samples=5)
        scores = acq.score(simple_model, test_data)
        assert torch.allclose(scores, torch.zeros(20), atol=1e-7)

    def test_restores_model_state(self, dropout_model, test_data):
        dropout_model.eval()
        VarianceRatioAcquisition(num_samples=3).score(dropout_model, test_data)
        assert not dropout_model.training


# ---------------------------------------------------------------------------
# MeanSTDAcquisition
# ---------------------------------------------------------------------------


class TestMeanSTDAcquisition:
    def test_score_shape_and_nonneg(self, dropout_model, test_data):
        acq = MeanSTDAcquisition(num_samples=5)
        scores = acq.score(dropout_model, test_data)

        assert scores.shape == (20,)
        assert torch.all(scores >= 0)

    def test_deterministic_model_gives_zero(self, simple_model, test_data):
        """A model with no dropout → zero std."""
        acq = MeanSTDAcquisition(num_samples=5)
        scores = acq.score(simple_model, test_data)
        assert torch.allclose(scores, torch.zeros(20), atol=1e-7)

    def test_restores_model_state(self, dropout_model, test_data):
        dropout_model.eval()
        MeanSTDAcquisition(num_samples=3).score(dropout_model, test_data)
        assert not dropout_model.training


# ---------------------------------------------------------------------------
# BatchBALDAcquisition
# ---------------------------------------------------------------------------


class TestBatchBALDAcquisition:
    def test_score_shape(self, dropout_model, test_data):
        acq = BatchBALDAcquisition(num_samples=5)
        scores = acq.score(dropout_model, test_data)

        assert scores.shape == (20,)
        assert torch.all(scores >= -1e-5)

    def test_restores_model_state(self, dropout_model, test_data):
        dropout_model.eval()
        BatchBALDAcquisition(num_samples=3).score(dropout_model, test_data)
        assert not dropout_model.training
