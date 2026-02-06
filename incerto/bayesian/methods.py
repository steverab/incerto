"""
Bayesian Deep Learning Methods.

Implementations of popular Bayesian approaches for uncertainty quantification
in neural networks.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Callable, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class BaseBayesianMethod(nn.Module, ABC):
    """
    Abstract base class for Bayesian deep learning methods.

    All Bayesian methods produce uncertainty estimates by maintaining or
    sampling from a distribution over model parameters (or predictions).
    They follow a common interface:

    - ``predict(x, return_samples=False)`` returns ``(mean, variance)``
      or ``(mean, variance, samples)`` when ``return_samples=True``.

    The returned variance captures **epistemic** (model) uncertainty —
    disagreement between posterior samples.  Aleatoric (data) uncertainty
    is not directly modelled by this interface.
    """

    @abstractmethod
    def predict(
        self,
        x: torch.Tensor,
        return_samples: bool = False,
        **kwargs,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Prediction with uncertainty estimation.

        Args:
            x: Input tensor.
            return_samples: If True, also return individual samples/predictions.

        Returns:
            ``(mean, variance)`` or ``(mean, variance, samples)`` when
            *return_samples* is True.  Variance is computed as disagreement
            across posterior samples and captures epistemic (model) uncertainty.
        """
        ...


class MCDropout(BaseBayesianMethod):
    """
    Monte Carlo Dropout for uncertainty estimation.

    Applies dropout at test time and aggregates predictions from multiple
    forward passes to estimate predictive uncertainty.

    Reference:
        Gal & Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016)

    Args:
        model: Base neural network
        num_samples: Number of MC samples for inference (default: 20)
        dropout_rate: Dropout probability (default: 0.1)

    Example:
        >>> backbone = ResNet18(num_classes=10)
        >>> mc_model = MCDropout(backbone, num_samples=20)
        >>> mean, variance = mc_model.predict(x)
    """

    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 20,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate

        # Enable dropout in all dropout layers
        self._enable_dropout()

    def _enable_dropout(self):
        """Enable dropout at test time and optionally set dropout rate."""
        found = False
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
                # Override dropout rate if specified
                if self.dropout_rate is not None:
                    module.p = self.dropout_rate
                found = True
        if not found:
            import warnings

            warnings.warn(
                "MCDropout: no nn.Dropout layers found in the model. "
                "All forward samples will be identical, yielding zero variance.",
                stacklevel=2,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single forward pass (for training)."""
        return self.model(x)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        return_samples: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Monte Carlo prediction with uncertainty estimation.

        Args:
            x: Input tensor ``(N, ...)``
            return_samples: If True, return all MC samples

        Returns:
            Tuple of (mean_prediction, predictive_variance)
            If return_samples=True: (mean, variance, samples)
        """
        self._enable_dropout()

        samples = []
        for _ in range(self.num_samples):
            output = self.model(x)
            # Convert logits to probabilities for classification
            if output.dim() == 2 and output.size(-1) > 1:
                output = F.softmax(output, dim=-1)
            samples.append(output)

        samples = torch.stack(samples)  # (num_samples, batch_size, *)

        # Compute mean and variance
        mean = samples.mean(dim=0)
        variance = samples.var(dim=0)

        if return_samples:
            return mean, variance, samples
        return mean, variance

    def predict_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute predictive entropy (total uncertainty).

        Args:
            x: Input tensor

        Returns:
            Predictive entropy for each sample
        """
        mean_probs, _ = self.predict(x)
        entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)
        return entropy

    def predict_mutual_information(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute mutual information (epistemic uncertainty).

        MI = H[y|x] - E[H[y|x,θ]]

        Args:
            x: Input tensor

        Returns:
            Mutual information for each sample
        """
        _, _, samples = self.predict(x, return_samples=True)

        # Expected entropy: E[H[y|x,θ]]
        expected_entropy = (
            -(samples * torch.log(samples + 1e-10)).sum(dim=-1).mean(dim=0)
        )

        # Entropy of mean: H[E[y|x,θ]]
        mean_probs = samples.mean(dim=0)
        entropy_of_mean = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

        # Mutual information
        mutual_info = entropy_of_mean - expected_entropy
        return mutual_info


class DeepEnsemble(BaseBayesianMethod):
    """
    Deep Ensembles for uncertainty quantification.

    Trains multiple neural networks independently and aggregates their
    predictions. This is one of the most effective methods for uncertainty
    estimation in deep learning.

    Reference:
        Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty
        Estimation using Deep Ensembles" (NeurIPS 2017)

    Args:
        model_fn: Function that creates a new model instance
        num_models: Number of ensemble members (default: 5)

    Example:
        >>> def create_model():
        ...     return ResNet18(num_classes=10)
        >>> ensemble = DeepEnsemble(create_model, num_models=5)
        >>> # Train each model separately
        >>> for i in range(5):
        ...     train_model(ensemble.models[i], train_loader)
        >>> mean, variance = ensemble.predict(x)
    """

    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        num_models: int = 5,
    ):
        super().__init__()
        self.num_models = num_models
        self.models = nn.ModuleList([model_fn() for _ in range(num_models)])

    def forward(self, x: torch.Tensor, model_idx: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass through a specific model or all models.

        Args:
            x: Input tensor
            model_idx: If specified, use only this model. Otherwise average all.

        Returns:
            Model output(s)
        """
        if model_idx is not None:
            return self.models[model_idx](x)

        # Average predictions from all models
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs).mean(dim=0)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        return_samples: bool = False,
        return_all: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Ensemble prediction with uncertainty.

        Args:
            x: Input tensor ``(N, ...)``
            return_samples: If True, return all individual predictions
            return_all: Deprecated alias for return_samples

        Returns:
            Tuple of (mean_prediction, epistemic_variance).
            Variance is computed as disagreement across ensemble members
            and captures epistemic (model) uncertainty.
            If return_samples=True: (mean, variance, all_predictions)
        """
        _return = return_samples or return_all

        predictions = []
        for model in self.models:
            model.eval()
            output = model(x)
            # Convert to probabilities for classification
            if output.dim() == 2 and output.size(-1) > 1:
                output = F.softmax(output, dim=-1)
            predictions.append(output)

        predictions = torch.stack(predictions)  # (num_models, batch_size, *)

        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)

        if _return:
            return mean, variance, predictions
        return mean, variance

    def train_member(
        self,
        model_idx: int,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int = 10,
        device: str | None = None,
    ):
        """
        Train a specific ensemble member.

        Args:
            model_idx: Index of model to train
            train_loader: Training data loader
            optimizer: Optimizer instance
            criterion: Loss function
            num_epochs: Number of training epochs
            device: Device to train on (default: auto-detect CUDA/CPU)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self.models[model_idx].to(device)
        model.train()

        for epoch in range(num_epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

    def diversity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ensemble diversity (disagreement).

        Args:
            x: Input tensor

        Returns:
            Per-sample diversity score
        """
        _, variance, _ = self.predict(x, return_samples=True)
        return variance.mean(dim=-1)


class SWAG(BaseBayesianMethod):
    """
    Stochastic Weight Averaging - Gaussian (SWAG).

    Approximates the posterior over weights using a Gaussian distribution
    fitted to the trajectory of SGD. Efficient and scalable Bayesian
    inference method.

    Reference:
        Maddox et al., "A Simple Baseline for Bayesian Uncertainty Estimation
        in Deep Learning" (NeurIPS 2019)

    Args:
        model: Base neural network
        num_samples: Number of samples for prediction (default: 20)
        var_clamp: Variance clamping value (default: 1e-6)

    Note:
        This is a diagonal SWAG implementation. The full SWAG algorithm
        also includes a low-rank covariance component, which is not
        implemented here for simplicity and memory efficiency.

    Example:
        >>> model = ResNet18(num_classes=10)
        >>> swag = SWAG(model)
        >>> # Train normally, then collect SWAG statistics
        >>> for epoch in range(start_swag, num_epochs):
        ...     train_epoch(model, ...)
        ...     swag.collect_model(model)
        >>> mean, variance = swag.predict(x)
    """

    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 20,
        var_clamp: float = 1e-6,
    ):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        self.var_clamp = var_clamp

        # Statistics for diagonal SWAG
        self.mean = {}
        self.sq_mean = {}
        self.n_models = 0

        # Initialize statistics
        for name, param in model.named_parameters():
            self.mean[name] = torch.zeros_like(param.data)
            self.sq_mean[name] = torch.zeros_like(param.data)

    def collect_model(self, model: nn.Module):
        """
        Collect model statistics for SWAG.

        Call this method periodically during training (e.g., every epoch
        after a warmup period) to collect weight statistics.

        Args:
            model: Current model to collect statistics from
        """
        self.n_models += 1

        for name, param in model.named_parameters():
            # Update first moment
            self.mean[name] = (self.n_models - 1) / self.n_models * self.mean[
                name
            ] + 1.0 / self.n_models * param.data

            # Update second moment
            self.sq_mean[name] = (self.n_models - 1) / self.n_models * self.sq_mean[
                name
            ] + 1.0 / self.n_models * param.data**2

    def sample_parameters(self) -> dict:
        """
        Sample parameters from the SWAG posterior.

        Returns:
            Dictionary of sampled parameters
        """
        sampled_params = {}

        for name in self.mean.keys():
            # Compute variance
            var = torch.clamp(
                self.sq_mean[name] - self.mean[name] ** 2, min=self.var_clamp
            )

            # Sample from Gaussian
            std = torch.sqrt(var)
            sampled_params[name] = self.mean[name] + torch.randn_like(std) * std

        return sampled_params

    def state_dict(self) -> dict:
        """
        Return state dictionary for serialization.

        Returns:
            State dictionary containing model state and SWAG statistics.
        """
        state = super().state_dict()
        state["_swag_mean"] = self.mean
        state["_swag_sq_mean"] = self.sq_mean
        state["_swag_n_models"] = self.n_models
        return state

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """
        Load state dictionary.

        Args:
            state_dict: State dictionary to load.
            strict: Whether to strictly enforce key matching.
        """
        # Copy to avoid mutating caller's dict
        state_dict = dict(state_dict)
        # Extract SWAG-specific state
        self.mean = state_dict.pop("_swag_mean", {})
        self.sq_mean = state_dict.pop("_swag_sq_mean", {})
        self.n_models = state_dict.pop("_swag_n_models", 0)
        # Load remaining state (model parameters)
        super().load_state_dict(state_dict, strict=strict)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        return_samples: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        SWAG prediction with uncertainty.

        Args:
            x: Input tensor ``(N, ...)``
            return_samples: If True, return all sampled predictions

        Returns:
            Tuple of (mean_prediction, predictive_variance)
            If return_samples=True: (mean, variance, samples)
        """
        if self.n_models == 0:
            from ..exceptions import NotFittedError

            raise NotFittedError("No models collected. Call collect_model() first.")

        predictions = []

        # Save original parameters
        original_params = {
            name: param.data.clone() for name, param in self.model.named_parameters()
        }

        try:
            # Sample and predict
            for _ in range(self.num_samples):
                sampled_params = self.sample_parameters()

                # Load sampled parameters
                for name, param in self.model.named_parameters():
                    param.data = sampled_params[name]

                # Forward pass
                output = self.model(x)
                if output.dim() == 2 and output.size(-1) > 1:
                    output = F.softmax(output, dim=-1)
                predictions.append(output)
        finally:
            # Restore original parameters even if forward pass fails
            for name, param in self.model.named_parameters():
                param.data = original_params[name]

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)

        if return_samples:
            return mean, variance, predictions
        return mean, variance


class LaplaceApproximation(BaseBayesianMethod):
    """
    Laplace Approximation for Bayesian Neural Networks.

    Approximates the posterior over weights using a Gaussian centered at
    the MAP estimate (trained weights) with covariance derived from the
    Hessian of the loss function.

    Reference:
        MacKay, "A Practical Bayesian Framework for Backpropagation Networks" (1992)
        Daxberger et al., "Laplace Redux" (NeurIPS 2021)

    Args:
        model: Trained neural network (at MAP estimate)
        likelihood: Likelihood type ('classification' or 'regression')
        prior_precision: Prior precision (inverse variance) (default: 1.0)

    Example:
        >>> model = ResNet18(num_classes=10)
        >>> # Train model to convergence
        >>> train_model(model, train_loader)
        >>> laplace = LaplaceApproximation(model, likelihood='classification')
        >>> laplace.fit(train_loader)
        >>> mean, variance = laplace.predict(x)
    """

    def __init__(
        self,
        model: nn.Module,
        likelihood: str = "classification",
        prior_precision: float = 1.0,
        num_samples: int = 20,
    ):
        super().__init__()
        self.model = model
        self.likelihood = likelihood
        self.prior_precision = prior_precision
        self.num_samples = num_samples

        # Will store the posterior covariance
        self.posterior_precision = None
        self.mean = None

    def _compute_hessian_diag(
        self,
        data_loader: DataLoader,
        device: str | None = None,
    ) -> dict:
        """
        Compute diagonal of the empirical Fisher (GGN approximation).

        Args:
            data_loader: Data loader for computing Hessian
            device: Device to use (default: auto-detect CUDA/CPU)

        Returns:
            Dictionary of Hessian diagonal per parameter
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        hessian_diag = {}

        # Initialize
        for name, param in self.model.named_parameters():
            hessian_diag[name] = torch.zeros_like(param.data)

        self.model.to(device)
        self.model.eval()

        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Zero gradients before forward pass
            self.model.zero_grad()

            # Forward pass
            outputs = self.model(batch_x)

            if self.likelihood == "classification":
                # Compute gradient of cross-entropy loss
                loss = F.cross_entropy(outputs, batch_y)
            else:  # regression
                # For regression with Gaussian likelihood
                loss = F.mse_loss(outputs.squeeze(), batch_y.float())

            loss.backward()

            # Accumulate squared gradients (Fisher diagonal approximation)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    hessian_diag[name] += param.grad.data**2

        # Average over dataset
        num_batches = len(data_loader)
        for name in hessian_diag:
            hessian_diag[name] /= num_batches

        return hessian_diag

    def fit(self, data_loader: DataLoader, device: str | None = None):
        """
        Fit Laplace approximation by computing Hessian.

        Args:
            data_loader: Data loader for computing Hessian
            device: Device to use (default: auto-detect CUDA/CPU)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # Store MAP estimate (current weights)
        self.mean = {
            name: param.data.clone() for name, param in self.model.named_parameters()
        }

        # Compute Hessian diagonal
        hessian_diag = self._compute_hessian_diag(data_loader, device)

        # Posterior precision = prior precision + Hessian
        self.posterior_precision = {
            name: self.prior_precision + hessian_diag[name] for name in hessian_diag
        }

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        return_samples: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Laplace prediction with uncertainty.

        Args:
            x: Input tensor ``(N, ...)``
            return_samples: If True, return all sampled predictions

        Returns:
            Tuple of (mean_prediction, predictive_variance)
            If return_samples=True: (mean, variance, samples)
        """
        if self.mean is None:
            from ..exceptions import NotFittedError

            raise NotFittedError("Model not fitted. Call fit() first.")

        predictions = []

        # Save original parameters
        original_params = {
            name: param.data.clone() for name, param in self.model.named_parameters()
        }

        try:
            # Sample from posterior
            for _ in range(self.num_samples):
                for name, param in self.model.named_parameters():
                    # Sample from Gaussian posterior
                    std = 1.0 / torch.sqrt(self.posterior_precision[name])
                    param.data = self.mean[name] + torch.randn_like(std) * std

                output = self.model(x)
                if output.dim() == 2 and output.size(-1) > 1:
                    output = F.softmax(output, dim=-1)
                predictions.append(output)
        finally:
            # Restore original parameters even if forward pass fails
            for name, param in self.model.named_parameters():
                param.data = original_params[name]

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)

        if return_samples:
            return mean, variance, predictions
        return mean, variance

    def state_dict(self) -> dict:
        """
        Return state dictionary for serialization.

        Returns:
            State dictionary containing model state and Laplace statistics.
        """
        state = super().state_dict()
        state["_laplace_mean"] = self.mean
        state["_laplace_posterior_precision"] = self.posterior_precision
        state["_laplace_likelihood"] = self.likelihood
        state["_laplace_prior_precision"] = self.prior_precision
        state["_laplace_num_samples"] = self.num_samples
        return state

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """
        Load state dictionary.

        Args:
            state_dict: State dictionary to load.
            strict: Whether to strictly enforce key matching.
        """
        # Copy to avoid mutating caller's dict
        state_dict = dict(state_dict)
        # Extract Laplace-specific state
        self.mean = state_dict.pop("_laplace_mean", None)
        self.posterior_precision = state_dict.pop("_laplace_posterior_precision", None)
        self.likelihood = state_dict.pop("_laplace_likelihood", "classification")
        self.prior_precision = state_dict.pop("_laplace_prior_precision", 1.0)
        self.num_samples = state_dict.pop("_laplace_num_samples", 20)
        # Load remaining state (model parameters)
        super().load_state_dict(state_dict, strict=strict)


class VariationalBayesNN(BaseBayesianMethod):
    """
    Variational Bayesian Neural Network (Bayes by Backprop).

    Learns a distribution over weights using variational inference.
    Each weight has a learned mean and variance.

    Reference:
        Blundell et al., "Weight Uncertainty in Neural Networks" (ICML 2015)

    Args:
        in_features: Input dimension
        hidden_sizes: List of hidden layer sizes
        out_features: Output dimension
        prior_std: Prior standard deviation (default: 1.0)
        num_samples: Number of MC samples for prediction (default: 20)

    Example:
        >>> model = VariationalBayesNN(784, [512, 256], 10)
        >>> # Train with variational loss
        >>> for x, y in train_loader:
        ...     loss = model.variational_loss(x, y, num_samples=10)
        ...     loss.backward()
        >>> mean, variance = model.predict(x)
    """

    def __init__(
        self,
        in_features: int,
        hidden_sizes: List[int],
        out_features: int,
        prior_std: float = 1.0,
        num_samples: int = 20,
    ):
        super().__init__()
        self.prior_std = prior_std
        self.num_samples = num_samples

        # Build network with Gaussian weights
        layers = []
        prev_size = in_features

        for hidden_size in hidden_sizes:
            layers.append(GaussianLinear(prev_size, hidden_size, prior_std=prior_std))
            prev_size = hidden_size

        layers.append(GaussianLinear(prev_size, out_features, prior_std=prior_std))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sampled weights."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between posterior and prior."""
        kl = 0
        for layer in self.layers:
            kl += layer.kl_divergence()
        return kl

    def variational_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        num_samples: int = 10,
        kl_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute variational loss (ELBO).

        Loss = E[NLL] + KL[q(w) || p(w)]

        Args:
            x: Input tensor
            y: Target labels
            num_samples: Number of samples for MC estimate
            kl_weight: Weight for KL term

        Returns:
            Variational loss
        """
        # Likelihood term (averaged over samples)
        nll = 0
        for _ in range(num_samples):
            outputs = self.forward(x)
            nll += F.cross_entropy(outputs, y)
        nll /= num_samples

        # KL term
        kl = self.kl_divergence()

        # ELBO
        return nll + kl_weight * kl / len(x)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        return_samples: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Variational prediction with uncertainty.

        Args:
            x: Input tensor
            return_samples: If True, return all sampled predictions

        Returns:
            Tuple of (mean_prediction, predictive_variance)
            If return_samples=True: (mean, variance, samples)
        """
        predictions = []
        for _ in range(self.num_samples):
            output = self.forward(x)
            if output.dim() == 2 and output.size(-1) > 1:
                output = F.softmax(output, dim=-1)
            predictions.append(output)

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)

        if return_samples:
            return mean, variance, predictions
        return mean, variance


class GaussianLinear(nn.Module):
    """
    Linear layer with Gaussian weights for variational inference.

    Each weight has a learnable mean (mu) and log-variance (rho).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        # Initialize rho such that softplus(rho) ≈ prior_std
        # This ensures initial KL ≈ 0 (posterior matches prior)
        # inverse_softplus(x) = log(exp(x) - 1)
        init_rho = torch.log(torch.tensor(prior_std).exp() - 1).item()

        # Variational parameters - initialize mu from prior N(0, prior_std)
        # This ensures initial KL ≈ 0 since posterior = prior at initialization
        self.weight_mu = nn.Parameter(
            torch.randn(out_features, in_features) * prior_std
        )
        self.weight_rho = nn.Parameter(
            torch.full((out_features, in_features), init_rho)
        )
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), init_rho))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sampled weights."""
        # Sample weights (softplus is numerically stable for large rho)
        weight_std = F.softplus(self.weight_rho)
        weight = self.weight_mu + weight_std * torch.randn_like(weight_std)

        # Sample bias
        bias_std = F.softplus(self.bias_rho)
        bias = self.bias_mu + bias_std * torch.randn_like(bias_std)

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """KL divergence between posterior and prior."""
        # Weight KL
        weight_std = F.softplus(self.weight_rho)
        weight_kl = (
            torch.log(self.prior_std / weight_std)
            + (weight_std**2 + self.weight_mu**2) / (2 * self.prior_std**2)
            - 0.5
        ).sum()

        # Bias KL
        bias_std = F.softplus(self.bias_rho)
        bias_kl = (
            torch.log(self.prior_std / bias_std)
            + (bias_std**2 + self.bias_mu**2) / (2 * self.prior_std**2)
            - 0.5
        ).sum()

        return weight_kl + bias_kl


__all__ = [
    "BaseBayesianMethod",
    "MCDropout",
    "DeepEnsemble",
    "SWAG",
    "LaplaceApproximation",
    "VariationalBayesNN",
    "GaussianLinear",
]
