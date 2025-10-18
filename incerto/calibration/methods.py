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
