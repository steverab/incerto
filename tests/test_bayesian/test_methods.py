"""
Tests for Bayesian deep learning methods.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from incerto.bayesian import (
    SWAG,
    DeepEnsemble,
    LaplaceApproximation,
    MCDropout,
    VariationalBayesNN,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, in_features=10, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 20)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(20, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


@pytest.fixture
def simple_data():
    """Generate simple test data."""
    x = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    return x, y


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel()


class TestMCDropout:
    """Tests for MC Dropout."""

    def test_init(self, simple_model):
        """Test MCDropout initialization."""
        mc_model = MCDropout(simple_model, num_samples=20)
        assert mc_model.num_samples == 20
        assert mc_model.model is simple_model

    def test_predict(self, simple_model, simple_data):
        """Test MC Dropout prediction."""
        x, y = simple_data
        mc_model = MCDropout(simple_model, num_samples=10)

        mean, variance = mc_model.predict(x[:10])

        assert mean.shape == (10, 3)
        assert variance.shape == (10, 3)
        assert torch.all(variance >= 0)

    def test_predict_with_samples(self, simple_model, simple_data):
        """Test returning all MC samples."""
        x, y = simple_data
        mc_model = MCDropout(simple_model, num_samples=10)

        mean, variance, samples = mc_model.predict(x[:10], return_samples=True)

        assert samples.shape == (10, 10, 3)  # (num_samples, batch_size, num_classes)
        assert torch.allclose(samples.mean(dim=0), mean, atol=1e-6)

    def test_predict_entropy(self, simple_model, simple_data):
        """Test predictive entropy computation."""
        x, y = simple_data
        mc_model = MCDropout(simple_model, num_samples=10)

        entropy = mc_model.predict_entropy(x[:10])

        assert entropy.shape == (10,)
        assert torch.all(entropy >= 0)

    def test_predict_mutual_information(self, simple_model, simple_data):
        """Test mutual information computation."""
        x, y = simple_data
        mc_model = MCDropout(simple_model, num_samples=10)

        mi = mc_model.predict_mutual_information(x[:10])

        assert mi.shape == (10,)
        assert torch.all(mi >= -1e-5)  # Should be non-negative (allow small numerical errors)


class TestDeepEnsemble:
    """Tests for Deep Ensembles."""

    def test_init(self):
        """Test Deep Ensemble initialization."""

        def create_model():
            return SimpleModel()

        ensemble = DeepEnsemble(create_model, num_models=5)

        assert len(ensemble.models) == 5
        assert all(isinstance(m, SimpleModel) for m in ensemble.models)

    def test_forward_single_model(self, simple_data):
        """Test forward pass through a single model."""
        x, y = simple_data

        def create_model():
            return SimpleModel()

        ensemble = DeepEnsemble(create_model, num_models=3)

        output = ensemble.forward(x[:10], model_idx=0)

        assert output.shape == (10, 3)

    def test_forward_all_models(self, simple_data):
        """Test forward pass averaging all models."""
        x, y = simple_data

        def create_model():
            return SimpleModel()

        ensemble = DeepEnsemble(create_model, num_models=3)

        output = ensemble.forward(x[:10])

        assert output.shape == (10, 3)

    def test_predict(self, simple_data):
        """Test ensemble prediction."""
        x, y = simple_data

        def create_model():
            return SimpleModel()

        ensemble = DeepEnsemble(create_model, num_models=5)

        mean, variance = ensemble.predict(x[:10])

        assert mean.shape == (10, 3)
        assert variance.shape == (10, 3)
        assert torch.all(variance >= 0)

    def test_predict_with_all(self, simple_data):
        """Test returning all ensemble predictions."""
        x, y = simple_data

        def create_model():
            return SimpleModel()

        ensemble = DeepEnsemble(create_model, num_models=5)

        mean, variance, all_preds = ensemble.predict(x[:10], return_samples=True)

        assert all_preds.shape == (5, 10, 3)
        assert torch.allclose(all_preds.mean(dim=0), mean, atol=1e-6)

    def test_diversity(self, simple_data):
        """Test ensemble diversity computation."""
        x, y = simple_data

        def create_model():
            return SimpleModel()

        ensemble = DeepEnsemble(create_model, num_models=5)

        diversity = ensemble.diversity(x[:10])

        assert diversity.shape == (10,)
        assert torch.all(diversity >= 0)


class TestSWAG:
    """Tests for SWAG."""

    def test_init(self, simple_model):
        """Test SWAG initialization."""
        swag = SWAG(simple_model, num_samples=20)

        assert swag.num_samples == 20
        assert swag.n_models == 0

    def test_collect_model(self, simple_model):
        """Test collecting model statistics."""
        swag = SWAG(simple_model, num_samples=10)

        # Collect a few models
        for _ in range(5):
            swag.collect_model(simple_model)

        assert swag.n_models == 5

    def test_predict(self, simple_model, simple_data):
        """Test SWAG prediction."""
        x, y = simple_data
        swag = SWAG(simple_model, num_samples=10)

        # Collect some models first
        for _ in range(5):
            swag.collect_model(simple_model)

        mean, variance = swag.predict(x[:10])

        assert mean.shape == (10, 3)
        assert variance.shape == (10, 3)
        assert torch.all(variance >= 0)

    def test_predict_without_collect(self, simple_model, simple_data):
        """Test that prediction fails without collecting models."""
        from incerto.exceptions import NotFittedError

        x, y = simple_data
        swag = SWAG(simple_model)

        with pytest.raises(NotFittedError):
            swag.predict(x[:10])


class TestLaplaceApproximation:
    """Tests for Laplace Approximation."""

    def test_init(self, simple_model):
        """Test Laplace initialization."""
        laplace = LaplaceApproximation(simple_model, likelihood="classification")

        assert laplace.likelihood == "classification"
        assert laplace.posterior_precision is None

    def test_predict_without_fit(self, simple_model, simple_data):
        """Test that prediction fails without fitting."""
        from incerto.exceptions import NotFittedError

        x, y = simple_data
        laplace = LaplaceApproximation(simple_model)

        with pytest.raises(NotFittedError):
            laplace.predict(x[:10])

    def test_fit_and_predict(self, simple_model, simple_data):
        """Happy-path: fit on training data, then predict."""
        from torch.utils.data import DataLoader, TensorDataset

        x, y = simple_data
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=32)

        laplace = LaplaceApproximation(simple_model, likelihood="classification", num_samples=5)
        laplace.fit(loader, device="cpu")

        # Posterior should be populated after fit
        assert laplace.mean is not None
        assert laplace.posterior_precision is not None

        # Predict
        x_test = x[:10]
        mean, variance = laplace.predict(x_test)

        assert mean.shape[0] == 10
        assert variance.shape[0] == 10
        assert torch.isfinite(mean).all()
        assert torch.all(variance >= 0)

    def test_fit_and_predict_with_samples(self, simple_model, simple_data):
        """Test return_samples=True returns three tensors."""
        from torch.utils.data import DataLoader, TensorDataset

        x, y = simple_data
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=32)

        laplace = LaplaceApproximation(simple_model, likelihood="classification", num_samples=5)
        laplace.fit(loader, device="cpu")

        x_test = x[:10]
        mean, variance, samples = laplace.predict(x_test, return_samples=True)

        assert mean.shape[0] == 10
        assert variance.shape[0] == 10
        assert samples.shape[0] == 5  # num_samples
        assert samples.shape[1] == 10  # batch size

    def test_fit_restores_original_params(self, simple_model, simple_data):
        """Test that predict restores model weights after sampling."""
        from torch.utils.data import DataLoader, TensorDataset

        x, y = simple_data
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=32)

        laplace = LaplaceApproximation(simple_model, likelihood="classification", num_samples=3)
        laplace.fit(loader, device="cpu")

        # Get original weights
        original_weights = {n: p.data.clone() for n, p in simple_model.named_parameters()}

        # Predict (internally samples different weights)
        laplace.predict(x[:5])

        # Verify weights restored
        for name, param in simple_model.named_parameters():
            assert torch.allclose(param.data, original_weights[name])


class TestVariationalBayesNN:
    """Tests for Variational Bayes NN."""

    def test_init(self):
        """Test Variational NN initialization."""
        model = VariationalBayesNN(10, [20, 20], 3)

        assert len(model.layers) == 3  # 2 hidden + 1 output

    def test_forward(self):
        """Test forward pass."""
        model = VariationalBayesNN(10, [20], 3)
        x = torch.randn(5, 10)

        output = model(x)

        assert output.shape == (5, 3)

    def test_kl_divergence(self):
        """Test KL divergence computation."""
        model = VariationalBayesNN(10, [20], 3)

        kl = model.kl_divergence()

        assert isinstance(kl, torch.Tensor)
        assert kl.item() >= 0

    def test_variational_loss(self):
        """Test variational loss computation."""
        model = VariationalBayesNN(10, [20], 3)
        x = torch.randn(5, 10)
        y = torch.randint(0, 3, (5,))

        loss = model.variational_loss(x, y, num_samples=5)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_predict(self):
        """Test prediction with uncertainty."""
        model = VariationalBayesNN(10, [20], 3, num_samples=10)
        x = torch.randn(5, 10)

        mean, variance = model.predict(x)

        assert mean.shape == (5, 3)
        assert variance.shape == (5, 3)
        assert torch.all(variance >= 0)


class TestSWAGSerialization:
    """Tests for SWAG serialization."""

    def test_state_dict_contains_swag_state(self, simple_model):
        """Test that state_dict includes SWAG-specific state."""
        swag = SWAG(simple_model, num_samples=10)

        # Collect some models
        for _ in range(3):
            swag.collect_model(simple_model)

        state = swag.state_dict()

        assert "_swag_mean" in state
        assert "_swag_sq_mean" in state
        assert "_swag_n_models" in state
        assert state["_swag_n_models"] == 3

    def test_load_state_dict_restores_swag_state(self, simple_model, simple_data):
        """Test that load_state_dict restores SWAG state."""
        x, y = simple_data
        swag = SWAG(simple_model, num_samples=5)

        # Collect some models and save state
        for _ in range(3):
            swag.collect_model(simple_model)

        state = swag.state_dict()

        # Create new SWAG and load state
        new_model = SimpleModel()
        new_swag = SWAG(new_model, num_samples=5)
        new_swag.load_state_dict(state)

        assert new_swag.n_models == 3
        assert len(new_swag.mean) > 0

        # Should be able to predict now
        mean, variance = new_swag.predict(x[:5])
        assert mean.shape[0] == 5

    def test_serialization_roundtrip(self, simple_model, simple_data, tmp_path):
        """Test full save/load cycle."""
        x, y = simple_data
        swag = SWAG(simple_model, num_samples=5)

        for _ in range(3):
            swag.collect_model(simple_model)

        # Get prediction before saving
        mean_before, _ = swag.predict(x[:5])

        # Save to file
        path = tmp_path / "swag.pt"
        torch.save(swag.state_dict(), path)

        # Load into new model
        new_model = SimpleModel()
        new_swag = SWAG(new_model, num_samples=5)
        new_swag.load_state_dict(torch.load(path, weights_only=True))

        assert new_swag.n_models == 3


class TestLaplaceApproximationSerialization:
    """Tests for LaplaceApproximation serialization."""

    def test_state_dict_contains_laplace_state(self, simple_model, simple_data):
        """Test that state_dict includes Laplace-specific state."""
        from torch.utils.data import DataLoader, TensorDataset

        x, y = simple_data
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=32)

        laplace = LaplaceApproximation(simple_model, num_samples=5)
        laplace.fit(loader, device="cpu")

        state = laplace.state_dict()

        assert "_laplace_mean" in state
        assert "_laplace_posterior_precision" in state
        assert "_laplace_likelihood" in state
        assert state["_laplace_likelihood"] == "classification"

    def test_load_state_dict_restores_laplace_state(self, simple_model, simple_data):
        """Test that load_state_dict restores Laplace state."""
        from torch.utils.data import DataLoader, TensorDataset

        x, y = simple_data
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=32)

        laplace = LaplaceApproximation(simple_model, num_samples=5)
        laplace.fit(loader, device="cpu")

        state = laplace.state_dict()

        # Create new Laplace and load state
        new_model = SimpleModel()
        new_laplace = LaplaceApproximation(new_model, num_samples=5)
        new_laplace.load_state_dict(state)

        assert new_laplace.mean is not None
        assert new_laplace.posterior_precision is not None

        # Should be able to predict now
        mean, variance = new_laplace.predict(x[:5])
        assert mean.shape[0] == 5

    def test_serialization_roundtrip(self, simple_model, simple_data, tmp_path):
        """Test full save/load cycle."""
        from torch.utils.data import DataLoader, TensorDataset

        x, y = simple_data
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=32)

        laplace = LaplaceApproximation(simple_model, num_samples=5)
        laplace.fit(loader, device="cpu")

        # Save to file
        path = tmp_path / "laplace.pt"
        torch.save(laplace.state_dict(), path)

        # Load into new model
        new_model = SimpleModel()
        new_laplace = LaplaceApproximation(new_model, num_samples=5)
        new_laplace.load_state_dict(torch.load(path, weights_only=True))

        assert new_laplace.mean is not None
        assert new_laplace.posterior_precision is not None
