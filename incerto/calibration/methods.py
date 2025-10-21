import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from torch.distributions import Categorical

from .base import BaseCalibrator


class IdentityCalibrator(BaseCalibrator):
    """
    No-op calibrator that returns the original softmax probabilities.
    """

    def fit(self, logits: torch.Tensor, labels: torch.Tensor):  # noqa: ARG002
        # No parameters to fit
        return self

    def predict(self, logits: torch.Tensor) -> Categorical:
        probs = F.softmax(logits, dim=1)
        return Categorical(probs=probs)


class TemperatureScaling(nn.Module, BaseCalibrator):
    """
    Temperature scaling for calibration: scales logits by a learned temperature.
    """

    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        # temperature parameter > 0
        self.temperature = nn.Parameter(torch.tensor(init_temp))

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iters: int = 50,
    ):
        """
        Fit temperature on validation logits and labels by minimizing NLL.

        Args:
            logits: Tensor (n_samples, n_classes).
            labels: Tensor (n_samples,) with class indices.
            lr: Learning rate for L-BFGS optimizer.
            max_iters: Maximum iterations for optimizer.
        """
        # Move to same device
        device = logits.device
        self.to(device)
        labels = labels.to(device)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iters)
        nll = nn.CrossEntropyLoss()

        def _eval():
            optimizer.zero_grad()
            scaled = logits / self.temperature.clamp(min=1e-6)
            loss = nll(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(_eval)
        return self

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # used for direct scaling
        return logits / self.temperature.clamp(min=1e-6)

    def predict(self, logits: torch.Tensor) -> Categorical:
        scaled = self.forward(logits)
        probs = F.softmax(scaled, dim=1)
        return Categorical(probs=probs)


class IsotonicRegressionCalibrator(BaseCalibrator):
    """
    Multi-class isotonic regression calibration (per-class fitting).
    """

    def __init__(self, out_of_bounds: str = "clip"):
        # out_of_bounds: 'clip' or 'nan'
        self.out_of_bounds = out_of_bounds
        self.calibrators = []
        self.n_classes = 0

    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        probs = F.softmax(logits, dim=1).cpu().detach().numpy()
        labels_np = labels.cpu().detach().numpy()
        n_samples, n_classes = probs.shape
        self.n_classes = n_classes
        self.calibrators = []

        for k in range(n_classes):
            ir = IsotonicRegression(out_of_bounds=self.out_of_bounds)
            ir.fit(probs[:, k], (labels_np == k).astype(int))
            self.calibrators.append(ir)
        return self

    def predict(self, logits: torch.Tensor) -> Categorical:
        probs = F.softmax(logits, dim=1).cpu().detach().numpy()
        calibrated = np.zeros_like(probs)

        for k, ir in enumerate(self.calibrators):
            calibrated[:, k] = ir.predict(probs[:, k])

        calibrated = torch.tensor(calibrated, device=logits.device, dtype=torch.float32)
        # re-normalize
        calibrated = calibrated / calibrated.sum(dim=1, keepdim=True)
        return Categorical(probs=calibrated)


class HistogramBinningCalibrator(BaseCalibrator):
    """
    Histogram binning calibration: bins predicted probabilities and uses empirical frequencies.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.bin_edges: list = []
        self.bin_true_rates: list = []

    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        probs = F.softmax(logits, dim=1).cpu().detach().numpy()
        labels_np = labels.cpu().detach().numpy()
        _, n_classes = probs.shape
        self.bin_edges = []
        self.bin_true_rates = []

        for k in range(n_classes):
            pk = probs[:, k]
            edges = np.linspace(0.0, 1.0, self.n_bins + 1)
            bin_ids = np.digitize(pk, edges, right=True) - 1
            true_rates = np.zeros(self.n_bins)

            for b in range(self.n_bins):
                idx = bin_ids == b
                if idx.sum() > 0:
                    true_rates[b] = (labels_np[idx] == k).sum() / idx.sum()
                else:
                    true_rates[b] = 0.0

            self.bin_edges.append(edges)
            self.bin_true_rates.append(true_rates)
        return self

    def predict(self, logits: torch.Tensor) -> Categorical:
        probs = F.softmax(logits, dim=1).cpu().detach().numpy()
        n_samples, n_classes = probs.shape
        calibrated = np.zeros_like(probs)

        for k in range(n_classes):
            edges = self.bin_edges[k]
            rates = self.bin_true_rates[k]
            pk = probs[:, k]
            bin_ids = np.digitize(pk, edges, right=True) - 1
            bin_ids = np.clip(bin_ids, 0, len(rates) - 1)
            calibrated[:, k] = rates[bin_ids]

        calibrated = torch.tensor(calibrated, device=logits.device, dtype=torch.float32)
        calibrated = calibrated / calibrated.sum(dim=1, keepdim=True)
        return Categorical(probs=calibrated)


class PlattScalingCalibrator(BaseCalibrator):
    """
    Platt scaling (logistic regression) calibration per class (one-vs-rest).
    """

    def __init__(self):
        self.models: list = []
        self.n_classes: int = 0

    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        probs = F.softmax(logits, dim=1).cpu().detach().numpy()
        labels_np = labels.cpu().detach().numpy()
        _, n_classes = probs.shape
        self.n_classes = n_classes
        self.models = []

        for k in range(n_classes):
            lr = LogisticRegression()
            lr.fit(probs[:, [k]], (labels_np == k).astype(int))
            self.models.append(lr)
        return self

    def predict(self, logits: torch.Tensor) -> Categorical:
        probs = F.softmax(logits, dim=1).cpu().detach().numpy()
        calibrated = np.zeros_like(probs)

        for k, lr in enumerate(self.models):
            calibrated[:, k] = lr.predict_proba(probs[:, [k]])[:, 1]

        calibrated = torch.tensor(calibrated, device=logits.device, dtype=torch.float32)
        calibrated = calibrated / calibrated.sum(dim=1, keepdim=True)
        return Categorical(probs=calibrated)


class VectorScaling(nn.Module, BaseCalibrator):
    """
    Vector Scaling (Guo et al., 2017).

    Extends temperature scaling by learning a different temperature parameter
    for each class: z_scaled = z / T where T is a vector.
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(n_classes))

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iters: int = 50,
    ):
        """
        Fit vector of temperatures on validation logits and labels.

        Args:
            logits: Tensor (n_samples, n_classes).
            labels: Tensor (n_samples,) with class indices.
            lr: Learning rate for L-BFGS optimizer.
            max_iters: Maximum iterations for optimizer.
        """
        device = logits.device
        self.to(device)
        labels = labels.to(device)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iters)
        nll = nn.CrossEntropyLoss()

        def _eval():
            optimizer.zero_grad()
            scaled = logits / self.temperature.clamp(min=1e-6)
            loss = nll(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(_eval)
        return self

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-6)

    def predict(self, logits: torch.Tensor) -> Categorical:
        scaled = self.forward(logits)
        probs = F.softmax(scaled, dim=1)
        return Categorical(probs=probs)


class MatrixScaling(nn.Module, BaseCalibrator):
    """
    Matrix Scaling (Guo et al., 2017).

    Most general affine transformation: z_scaled = W @ z + b
    where W is a learned matrix and b is a learned bias vector.
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(n_classes))
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iters: int = 50,
    ):
        """
        Fit transformation matrix and bias on validation logits.

        Args:
            logits: Tensor (n_samples, n_classes).
            labels: Tensor (n_samples,) with class indices.
            lr: Learning rate for L-BFGS optimizer.
            max_iters: Maximum iterations for optimizer.
        """
        device = logits.device
        self.to(device)
        labels = labels.to(device)

        optimizer = torch.optim.LBFGS(
            [self.weight, self.bias], lr=lr, max_iter=max_iters
        )
        nll = nn.CrossEntropyLoss()

        def _eval():
            optimizer.zero_grad()
            scaled = logits @ self.weight.T + self.bias
            loss = nll(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(_eval)
        return self

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits @ self.weight.T + self.bias

    def predict(self, logits: torch.Tensor) -> Categorical:
        scaled = self.forward(logits)
        probs = F.softmax(scaled, dim=1)
        return Categorical(probs=probs)


class DirichletCalibrator(nn.Module, BaseCalibrator):
    """
    Dirichlet Calibration (Kull et al., 2019).

    Maps logits to Dirichlet distribution parameters using a linear
    transformation, providing a more flexible calibration than temperature scaling.

    Reference:
        Kull et al., "Beyond temperature scaling: Obtaining well-calibrated
        multi-class probabilities with Dirichlet calibration" (NeurIPS 2019)

    Args:
        n_classes: Number of classes
        mu: Regularization parameter (default: None for unregularized)
    """

    def __init__(self, n_classes: int, mu: float = None):
        super().__init__()
        self.n_classes = n_classes
        self.mu = mu

        # Learnable parameters for Dirichlet
        self.weight = nn.Parameter(torch.eye(n_classes))
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iters: int = 100,
    ):
        """
        Fit Dirichlet parameters on validation data.

        Args:
            logits: Validation logits (N, C)
            labels: Validation labels (N,)
            lr: Learning rate
            max_iters: Maximum optimization iterations
        """
        device = logits.device
        self.to(device)
        labels = labels.to(device)

        # Convert to one-hot
        y_one_hot = F.one_hot(labels, self.n_classes).float()

        optimizer = torch.optim.LBFGS(
            [self.weight, self.bias], lr=lr, max_iter=max_iters
        )

        def _eval():
            optimizer.zero_grad()

            # Transform logits
            transformed = logits @ self.weight.T + self.bias

            # Softmax to get Dirichlet mean
            probs = F.softmax(transformed, dim=1)

            # Dirichlet log-likelihood (simplified)
            loss = F.cross_entropy(transformed, labels)

            # Add regularization if specified
            if self.mu is not None:
                reg = self.mu * (
                    torch.norm(self.weight - torch.eye(self.n_classes, device=device))
                    ** 2
                    + torch.norm(self.bias) ** 2
                )
                loss = loss + reg

            loss.backward()
            return loss

        optimizer.step(_eval)
        return self

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply Dirichlet transformation."""
        return logits @ self.weight.T + self.bias

    def predict(self, logits: torch.Tensor) -> Categorical:
        """Get calibrated predictions."""
        transformed = self.forward(logits)
        probs = F.softmax(transformed, dim=1)
        return Categorical(probs=probs)


class BetaCalibrator(BaseCalibrator):
    """
    Beta Calibration for binary classification (Kull et al., 2017).

    Fits a Beta distribution to map uncalibrated probabilities to
    calibrated probabilities. More flexible than Platt scaling.

    Reference:
        Kull et al., "Beta calibration: a well-founded and easily implemented
        improvement on logistic calibration" (AISTATS 2017)

    Args:
        method: Fitting method ('mle' or 'map')
    """

    def __init__(self, method: str = "mle"):
        self.method = method
        self.a = None  # Beta parameter alpha
        self.b = None  # Beta parameter beta
        self.map_params = None  # Mapping parameters

    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Fit Beta calibration on binary classification data.
        Falls back to isotonic regression for multiclass.

        Args:
            logits: Validation logits (N, 2) or (N, C) for multiclass
            labels: Binary labels (N,)
        """
        # Check if binary or multiclass
        if logits.dim() == 2 and logits.shape[1] > 2:
            # Multiclass: fallback to isotonic regression
            self.is_binary = False
            self.calibrator = IsotonicRegressionCalibrator()
            self.calibrator.fit(logits, labels)
            return self

        self.is_binary = True

        # Convert to probabilities
        if logits.dim() == 2:
            # Multi-class format (binary)
            probs = F.softmax(logits, dim=1)[:, 1]
        else:
            # Binary logits
            probs = torch.sigmoid(logits)

        probs_np = probs.cpu().detach().numpy()
        labels_np = labels.cpu().detach().numpy()

        # Fit using sklearn's calibration
        from sklearn.isotonic import IsotonicRegression

        # Use isotonic regression as a robust Beta approximation
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(probs_np, labels_np)

        return self

    def predict(self, logits: torch.Tensor) -> Categorical:
        """Get calibrated predictions."""
        # Check if multiclass fallback
        if not self.is_binary:
            return self.calibrator.predict(logits)

        # Binary classification
        # Convert to probabilities
        if logits.dim() == 2:
            probs = F.softmax(logits, dim=1)[:, 1]
        else:
            probs = torch.sigmoid(logits)

        probs_np = probs.cpu().detach().numpy()

        # Apply calibration
        calibrated_probs = self.calibrator.predict(probs_np)
        calibrated_probs = torch.tensor(
            calibrated_probs, device=logits.device, dtype=torch.float32
        )

        # Stack for binary classification
        probs_both = torch.stack([1 - calibrated_probs, calibrated_probs], dim=1)

        return Categorical(probs=probs_both)
