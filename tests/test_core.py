"""Tests for incerto.core (entropy and pairwise distance utilities)."""

import numpy as np
import pytest
import torch

from incerto.core import entropy, predictive_entropy, pairwise_squared_euclidean


class TestEntropy:
    def test_uniform_distribution(self):
        """Uniform distribution should have maximum entropy = log(K)."""
        k = 10
        probs = np.ones(k) / k
        result = entropy(probs)
        assert np.isclose(result, np.log(k), atol=1e-6)

    def test_deterministic_distribution(self):
        """One-hot distribution should have entropy close to 0."""
        probs = np.array([1.0, 0.0, 0.0])
        result = entropy(probs)
        assert np.isclose(result, 0.0, atol=1e-6)

    def test_binary_distribution(self):
        """Binary entropy H(0.5, 0.5) = log(2)."""
        probs = np.array([0.5, 0.5])
        result = entropy(probs)
        assert np.isclose(result, np.log(2), atol=1e-6)

    def test_zero_probs(self):
        """All-zero probabilities should return 0."""
        probs = np.zeros(5)
        assert entropy(probs) == 0.0

    def test_rejects_non_numpy(self):
        """Should raise TypeError for non-numpy input."""
        with pytest.raises(TypeError):
            entropy([0.5, 0.5])

    def test_rejects_negative_probs(self):
        """Should raise ValueError for negative probabilities."""
        with pytest.raises(ValueError):
            entropy(np.array([-0.1, 1.1]))

    def test_rejects_probs_above_one(self):
        """Should raise ValueError for probabilities > 1."""
        with pytest.raises(ValueError):
            entropy(np.array([0.5, 1.5]))

    def test_predictive_entropy_alias(self):
        """predictive_entropy should be an alias for entropy."""
        assert predictive_entropy is entropy


class TestPairwiseSquaredEuclidean:
    def test_self_distance_diagonal(self):
        """Self-distance diagonal should be zero."""
        x = torch.randn(10, 5)
        d = pairwise_squared_euclidean(x, x)
        diag = d.diag()
        assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-5)

    def test_shape(self):
        """Output shape should be (n, m)."""
        x = torch.randn(10, 5)
        y = torch.randn(20, 5)
        d = pairwise_squared_euclidean(x, y)
        assert d.shape == (10, 20)

    def test_non_negative(self):
        """Squared distances should be non-negative."""
        x = torch.randn(10, 3)
        y = torch.randn(15, 3)
        d = pairwise_squared_euclidean(x, y)
        assert (d >= 0).all()

    def test_symmetry(self):
        """D(x, y) should equal D(y, x).T."""
        x = torch.randn(10, 5)
        y = torch.randn(15, 5)
        dxy = pairwise_squared_euclidean(x, y)
        dyx = pairwise_squared_euclidean(y, x)
        assert torch.allclose(dxy, dyx.T, atol=1e-5)

    def test_known_values(self):
        """Verify against manual computation."""
        x = torch.tensor([[1.0, 0.0]])
        y = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        d = pairwise_squared_euclidean(x, y)
        assert torch.allclose(d, torch.tensor([[1.0, 1.0]]), atol=1e-5)

    def test_identical_points(self):
        """Identical points should have zero distance."""
        x = torch.tensor([[3.0, 4.0], [3.0, 4.0]])
        d = pairwise_squared_euclidean(x, x)
        assert torch.allclose(d, torch.zeros(2, 2), atol=1e-5)
