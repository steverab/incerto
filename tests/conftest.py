"""
Shared pytest fixtures for incerto tests.

This module provides common test data and utilities used across all test modules.
"""

import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def device():
    """Device for testing (CPU by default)."""
    return torch.device("cpu")


@pytest.fixture
def seed():
    """Random seed for reproducibility."""
    return 42


@pytest.fixture
def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Binary classification fixtures
@pytest.fixture
def binary_logits(set_seed):
    """Binary classification logits (100 samples, 2 classes)."""
    return torch.randn(100, 2)


@pytest.fixture
def binary_labels(set_seed):
    """Binary classification labels (100 samples)."""
    return torch.randint(0, 2, (100,))


@pytest.fixture
def binary_probs(binary_logits):
    """Binary classification probabilities."""
    return torch.softmax(binary_logits, dim=1)


# Multi-class classification fixtures
@pytest.fixture
def num_classes():
    """Number of classes for multi-class classification."""
    return 10


@pytest.fixture
def num_samples():
    """Number of samples for testing."""
    return 100


@pytest.fixture
def multiclass_logits(set_seed, num_samples, num_classes):
    """Multi-class logits (100 samples, 10 classes)."""
    return torch.randn(num_samples, num_classes)


@pytest.fixture
def multiclass_labels(set_seed, num_samples, num_classes):
    """Multi-class labels (100 samples)."""
    return torch.randint(0, num_classes, (num_samples,))


@pytest.fixture
def multiclass_probs(multiclass_logits):
    """Multi-class probabilities."""
    return torch.softmax(multiclass_logits, dim=1)


# Batch fixtures
@pytest.fixture
def batch_logits(set_seed):
    """Batched logits (batch=32, seq_len=10, vocab=100)."""
    return torch.randn(32, 10, 100)


@pytest.fixture
def batch_labels(set_seed):
    """Batched labels (batch=32, seq_len=10)."""
    return torch.randint(0, 100, (32, 10))


# Regression fixtures
@pytest.fixture
def regression_predictions(set_seed, num_samples):
    """Regression predictions."""
    return torch.randn(num_samples)


@pytest.fixture
def regression_targets(set_seed, num_samples):
    """Regression targets."""
    return torch.randn(num_samples)


@pytest.fixture
def regression_residuals(regression_predictions, regression_targets):
    """Regression residuals."""
    return torch.abs(regression_predictions - regression_targets)


# Dataset fixtures
@pytest.fixture
def simple_dataset(set_seed):
    """Simple 2D classification dataset (200 samples)."""
    # Create two Gaussian blobs
    X1 = torch.randn(100, 2) + torch.tensor([2.0, 2.0])
    X2 = torch.randn(100, 2) + torch.tensor([-2.0, -2.0])
    X = torch.cat([X1, X2], dim=0)
    y = torch.cat([torch.zeros(100), torch.ones(100)]).long()

    # Shuffle
    perm = torch.randperm(200)
    X = X[perm]
    y = y[perm]

    return TensorDataset(X, y)


@pytest.fixture
def simple_dataloader(simple_dataset):
    """DataLoader for simple dataset."""
    return DataLoader(simple_dataset, batch_size=32, shuffle=False)


# Feature fixtures for OOD detection
@pytest.fixture
def id_features(set_seed):
    """In-distribution features (100 samples, 64 dims)."""
    return torch.randn(100, 64)


@pytest.fixture
def ood_features(set_seed):
    """Out-of-distribution features (100 samples, 64 dims)."""
    # Shifted distribution
    return torch.randn(100, 64) + 2.0


# OOD detection data fixtures
@pytest.fixture
def ood_id_data(set_seed, num_classes):
    """In-distribution data for OOD detection (100 samples, 64 dims)."""
    X = torch.randn(100, 64)
    y = torch.randint(0, num_classes, (100,))
    return TensorDataset(X, y)


@pytest.fixture
def ood_ood_data(set_seed):
    """Out-of-distribution data for OOD detection (50 samples, 64 dims)."""
    # Different distribution (shifted and scaled)
    X = torch.randn(50, 64) * 2.0 + 3.0
    y = torch.zeros(50).long()  # Dummy labels
    return TensorDataset(X, y)


@pytest.fixture
def ood_id_loader(ood_id_data):
    """DataLoader for in-distribution OOD data."""
    return DataLoader(ood_id_data, batch_size=32, shuffle=False)


@pytest.fixture
def ood_id_inputs(ood_id_data):
    """Raw inputs from ID distribution (for scoring)."""
    return ood_id_data.tensors[0][:20]  # First 20 samples


@pytest.fixture
def ood_ood_inputs(ood_ood_data):
    """Raw inputs from OOD distribution (for scoring)."""
    return ood_ood_data.tensors[0][:20]  # First 20 samples


# Model fixtures
@pytest.fixture
def simple_model(num_classes):
    """Simple linear model for testing (2D input)."""
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 32), torch.nn.ReLU(), torch.nn.Linear(32, num_classes)
    )
    # Initialize to reasonable values
    torch.nn.init.xavier_uniform_(model[0].weight)
    torch.nn.init.xavier_uniform_(model[2].weight)
    model[0].bias.data.zero_()
    model[2].bias.data.zero_()
    return model


@pytest.fixture
def ood_model(num_classes):
    """Model for OOD detection testing with penultimate layer."""
    import torch.nn as nn

    class TestModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.penultimate = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            self.classifier = nn.Linear(32, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = self.penultimate(x)
            return self.classifier(x)

    model = TestModel(num_classes)
    model.eval()
    return model


# Calibration-specific fixtures
@pytest.fixture
def calibration_split(multiclass_logits, multiclass_labels):
    """Split data into train/val for calibration."""
    n = len(multiclass_logits)
    n_train = n // 2

    train_logits = multiclass_logits[:n_train]
    train_labels = multiclass_labels[:n_train]
    val_logits = multiclass_logits[n_train:]
    val_labels = multiclass_labels[n_train:]

    return {
        "train_logits": train_logits,
        "train_labels": train_labels,
        "val_logits": val_logits,
        "val_labels": val_labels,
    }


# Ensemble fixtures
@pytest.fixture
def ensemble_predictions(set_seed, num_samples, num_classes):
    """Ensemble predictions (5 models, 100 samples, 10 classes)."""
    n_models = 5
    predictions = []
    for _ in range(n_models):
        logits = torch.randn(num_samples, num_classes)
        probs = torch.softmax(logits, dim=1)
        predictions.append(probs)
    return torch.stack(predictions)


# LLM fixtures
@pytest.fixture
def token_logits(set_seed):
    """Token-level logits for LLM testing (batch=4, seq_len=20, vocab=50257)."""
    return torch.randn(4, 20, 50257)


@pytest.fixture
def token_ids(set_seed):
    """Token IDs for LLM testing (batch=4, seq_len=20)."""
    return torch.randint(0, 50257, (4, 20))


@pytest.fixture
def generation_samples(set_seed):
    """Multiple generated responses for sampling-based methods."""
    return [
        "The capital of France is Paris.",
        "Paris is the capital of France.",
        "The capital of France is Paris.",
        "France's capital is Paris.",
        "The capital of France is Lyon.",  # Wrong answer
    ]


# Conformal prediction fixtures
@pytest.fixture
def conformal_scores(set_seed, num_samples):
    """Non-conformity scores for conformal prediction."""
    return torch.rand(num_samples)  # Scores between 0 and 1


@pytest.fixture
def alpha():
    """Significance level for conformal prediction."""
    return 0.1  # 90% coverage


# Helper fixtures
@pytest.fixture
def check_tensor():
    """Helper function to check tensor properties."""

    def _check(tensor, shape=None, dtype=None, device=None):
        assert isinstance(tensor, torch.Tensor)
        if shape is not None:
            assert tensor.shape == shape, f"Expected shape {shape}, got {tensor.shape}"
        if dtype is not None:
            assert tensor.dtype == dtype, f"Expected dtype {dtype}, got {tensor.dtype}"
        if device is not None:
            assert tensor.device == device, f"Expected device {device}, got {tensor.device}"

    return _check


@pytest.fixture
def check_finite():
    """Helper function to check tensor is finite."""

    def _check(tensor):
        assert torch.isfinite(tensor).all(), "Tensor contains inf or nan values"

    return _check


@pytest.fixture
def check_probability():
    """Helper function to check tensor is valid probability distribution."""

    def _check(probs, dim=-1):
        assert (probs >= 0).all(), "Probabilities must be non-negative"
        assert (probs <= 1).all(), "Probabilities must be <= 1"
        sums = probs.sum(dim=dim)
        assert torch.allclose(
            sums, torch.ones_like(sums), atol=1e-5
        ), f"Probabilities must sum to 1, got {sums}"

    return _check
