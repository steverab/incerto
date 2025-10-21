"""
Tests for acquisition functions.
"""

import pytest
import torch
import torch.nn as nn

from incerto.active.acquisition import (
    RandomAcquisition,
    EntropyAcquisition,
    LeastConfidenceAcquisition,
    MarginAcquisition,
    BALDAcquisition,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def simple_model():
    return SimpleModel()


@pytest.fixture
def test_data():
    return torch.randn(20, 10)


class TestRandomAcquisition:
    """Tests for RandomAcquisition."""

    def test_score(self, simple_model, test_data):
        """Test random scoring."""
        acq = RandomAcquisition()
        scores = acq.score(simple_model, test_data)

        assert scores.shape == (20,)
        assert torch.all(scores >= 0) and torch.all(scores <= 1)


class TestEntropyAcquisition:
    """Tests for EntropyAcquisition."""

    def test_score(self, simple_model, test_data):
        """Test entropy scoring."""
        acq = EntropyAcquisition()
        scores = acq.score(simple_model, test_data)

        assert scores.shape == (20,)
        assert torch.all(scores >= 0)


class TestLeastConfidenceAcquisition:
    """Tests for LeastConfidenceAcquisition."""

    def test_score(self, simple_model, test_data):
        """Test least confidence scoring."""
        acq = LeastConfidenceAcquisition()
        scores = acq.score(simple_model, test_data)

        assert scores.shape == (20,)
        assert torch.all(scores >= 0) and torch.all(scores <= 1)


class TestMarginAcquisition:
    """Tests for MarginAcquisition."""

    def test_score(self, simple_model, test_data):
        """Test margin scoring."""
        acq = MarginAcquisition()
        scores = acq.score(simple_model, test_data)

        assert scores.shape == (20,)


class TestBALDAcquisition:
    """Tests for BALDAcquisition."""

    def test_init(self):
        """Test BALD initialization."""
        acq = BALDAcquisition(num_samples=10)
        assert acq.num_samples == 10

    def test_score(self, simple_model, test_data):
        """Test BALD scoring."""
        # Add dropout to model
        simple_model.dropout = nn.Dropout(0.1)

        acq = BALDAcquisition(num_samples=5)
        scores = acq.score(simple_model, test_data)

        assert scores.shape == (20,)
        assert torch.all(scores >= -1e-5)  # Allow small numerical errors
