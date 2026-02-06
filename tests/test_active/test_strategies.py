"""
Tests for active learning query strategies.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from incerto.active.acquisition import EntropyAcquisition
from incerto.active.strategies import (
    UncertaintySampling,
    DiversitySampling,
    CoreSetSelection,
    BadgeSampling,
    QueryByCommittee,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, in_features=10, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 20)
        self.fc2 = nn.Linear(20, num_classes)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


@pytest.fixture
def model():
    torch.manual_seed(42)
    return SimpleModel()


@pytest.fixture
def data():
    torch.manual_seed(42)
    return torch.randn(30, 10)


# ---------------------------------------------------------------------------
# UncertaintySampling
# ---------------------------------------------------------------------------


class TestUncertaintySampling:
    def test_returns_correct_count(self, model, data):
        strategy = UncertaintySampling(EntropyAcquisition(), batch_size=5)
        indices = strategy.query(model, data)
        assert len(indices) == 5

    def test_indices_valid(self, model, data):
        strategy = UncertaintySampling(EntropyAcquisition(), batch_size=5)
        indices = strategy.query(model, data)
        assert torch.all(indices >= 0) and torch.all(indices < len(data))

    def test_indices_unique(self, model, data):
        strategy = UncertaintySampling(EntropyAcquisition(), batch_size=5)
        indices = strategy.query(model, data)
        assert len(indices.unique()) == len(indices)

    def test_batch_size_larger_than_pool(self, model, data):
        strategy = UncertaintySampling(EntropyAcquisition(), batch_size=100)
        indices = strategy.query(model, data)
        assert len(indices) == len(data)


# ---------------------------------------------------------------------------
# DiversitySampling
# ---------------------------------------------------------------------------


class TestDiversitySampling:
    def test_returns_correct_count(self, model, data):
        strategy = DiversitySampling(EntropyAcquisition(), batch_size=5)
        indices = strategy.query(model, data)
        assert len(indices) == 5

    def test_indices_valid(self, model, data):
        strategy = DiversitySampling(EntropyAcquisition(), batch_size=5)
        indices = strategy.query(model, data)
        assert torch.all(indices >= 0) and torch.all(indices < len(data))

    def test_indices_unique(self, model, data):
        strategy = DiversitySampling(EntropyAcquisition(), batch_size=5)
        indices = strategy.query(model, data)
        assert len(indices.unique()) == len(indices)

    def test_diversity_weight_zero_matches_uncertainty(self, model, data):
        """With diversity_weight=0, should match pure uncertainty sampling."""
        torch.manual_seed(0)
        strat_div = DiversitySampling(
            EntropyAcquisition(), batch_size=5, diversity_weight=0.0
        )
        indices_div = strat_div.query(model, data)

        strat_unc = UncertaintySampling(EntropyAcquisition(), batch_size=5)
        indices_unc = strat_unc.query(model, data)

        # First selected sample should be the same (most uncertain)
        assert indices_div[0] == indices_unc[0]


# ---------------------------------------------------------------------------
# CoreSetSelection
# ---------------------------------------------------------------------------


class TestCoreSetSelection:
    def test_returns_correct_count(self, model, data):
        strategy = CoreSetSelection(batch_size=5)
        indices = strategy.query(model, data)
        assert len(indices) == 5

    def test_indices_valid(self, model, data):
        strategy = CoreSetSelection(batch_size=5)
        indices = strategy.query(model, data)
        assert torch.all(indices >= 0) and torch.all(indices < len(data))

    def test_with_labeled_data(self, model, data):
        labeled = torch.randn(10, 10)
        strategy = CoreSetSelection(batch_size=5)
        indices = strategy.query(model, data, x_labeled=labeled)
        assert len(indices) == 5

    def test_with_precomputed_features(self, model, data):
        with torch.no_grad():
            features = model(data)
        strategy = CoreSetSelection(batch_size=5)
        indices = strategy.query(model, data, features_unlabeled=features)
        assert len(indices) == 5


# ---------------------------------------------------------------------------
# BadgeSampling
# ---------------------------------------------------------------------------


class TestBadgeSampling:
    def test_returns_correct_count(self, model, data):
        strategy = BadgeSampling(batch_size=5)
        # Use small data for speed
        small_data = data[:10]
        indices = strategy.query(model, small_data)
        assert len(indices) == 5

    def test_indices_valid(self, model, data):
        strategy = BadgeSampling(batch_size=3)
        small_data = data[:8]
        indices = strategy.query(model, small_data)
        assert torch.all(indices >= 0) and torch.all(indices < len(small_data))

    def test_find_last_linear(self, model):
        last = BadgeSampling._find_last_linear(model)
        assert last is model.fc2

    def test_no_linear_raises(self):
        class NoLinearModel(nn.Module):
            def forward(self, x):
                return x

        strategy = BadgeSampling(batch_size=2)
        with pytest.raises(ValueError, match="nn.Linear"):
            strategy.query(NoLinearModel(), torch.randn(5, 10))


# ---------------------------------------------------------------------------
# QueryByCommittee
# ---------------------------------------------------------------------------


class TestQueryByCommittee:
    def _make_committee(self, n=3):
        torch.manual_seed(0)
        return [SimpleModel() for _ in range(n)]

    def test_vote_entropy(self, data):
        models = self._make_committee()
        strategy = QueryByCommittee(models, batch_size=5, disagreement="vote_entropy")
        indices = strategy.query(model=None, x_unlabeled=data)
        assert len(indices) == 5
        assert torch.all(indices >= 0) and torch.all(indices < len(data))

    def test_kl(self, data):
        models = self._make_committee()
        strategy = QueryByCommittee(models, batch_size=5, disagreement="kl")
        indices = strategy.query(model=None, x_unlabeled=data)
        assert len(indices) == 5

    def test_invalid_disagreement_raises(self, data):
        models = self._make_committee()
        strategy = QueryByCommittee(models, batch_size=5, disagreement="invalid")
        with pytest.raises(ValueError, match="Unknown disagreement"):
            strategy.query(model=None, x_unlabeled=data)

    def test_accepts_model_param(self, model, data):
        """QBC should accept a model param for interface compatibility."""
        models = self._make_committee()
        strategy = QueryByCommittee(models, batch_size=5)
        # Should not raise even though model is passed
        indices = strategy.query(model=model, x_unlabeled=data)
        assert len(indices) == 5

    def test_x_unlabeled_required(self):
        models = self._make_committee()
        strategy = QueryByCommittee(models, batch_size=5)
        with pytest.raises(ValueError, match="x_unlabeled"):
            strategy.query()
