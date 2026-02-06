import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from torch.distributions import Categorical

from .base import BaseCalibrator, _validate_fit_inputs


class IdentityCalibrator(BaseCalibrator):
    """
    No-op calibrator that returns the original softmax probabilities.
    """

    def fit(self, logits: torch.Tensor, labels: torch.Tensor):  # noqa: ARG002
        _validate_fit_inputs(logits, labels)
        # No parameters to fit
        return self

    def predict(self, logits: torch.Tensor) -> Categorical:
        probs = F.softmax(logits, dim=1)
        return Categorical(probs=probs)

    def state_dict(self) -> dict:
        """Return empty state dict (no parameters to save)."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Load state (no-op for identity calibrator)."""
        pass

    def __repr__(self) -> str:
        return "IdentityCalibrator()"


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
        _validate_fit_inputs(logits, labels)
        # Move to same device
        device = logits.device
        self.to(device)
        labels = labels.to(device)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iters)
        nll = nn.CrossEntropyLoss()

        def _eval():
            optimizer.zero_grad()
            scaled = logits / self.temperature.clamp(min=1e-4)
            loss = nll(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(_eval)
        return self

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # used for direct scaling
        return logits / self.temperature.clamp(min=1e-4)

    def predict(self, logits: torch.Tensor) -> Categorical:
        scaled = self.forward(logits)
        probs = F.softmax(scaled, dim=1)
        return Categorical(probs=probs)

    def __repr__(self) -> str:
        return f"TemperatureScaling(temperature={self.temperature.item():.4f})"


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
        _validate_fit_inputs(logits, labels)
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
        from ..exceptions import NotFittedError

        if not self.calibrators:
            raise NotFittedError(
                "IsotonicRegressionCalibrator has not been fitted. Call fit() first."
            )
        probs = F.softmax(logits, dim=1).cpu().detach().numpy()
        calibrated = np.zeros_like(probs)

        for k, ir in enumerate(self.calibrators):
            calibrated[:, k] = ir.predict(probs[:, k])

        calibrated = torch.tensor(calibrated, device=logits.device, dtype=torch.float32)
        # re-normalize
        calibrated = calibrated / calibrated.sum(dim=1, keepdim=True).clamp(min=1e-10)
        return Categorical(probs=calibrated)

    def state_dict(self) -> dict:
        """Save isotonic regression calibrators."""
        from .._sklearn_io import serialize_isotonic

        return {
            "n_classes": self.n_classes,
            "out_of_bounds": self.out_of_bounds,
            "calibrators": [serialize_isotonic(ir) for ir in self.calibrators],
        }

    def load_state_dict(self, state: dict) -> None:
        """Load isotonic regression calibrators."""
        from .._sklearn_io import deserialize_isotonic
        from ..exceptions import SerializationError

        try:
            self.n_classes = state["n_classes"]
            self.out_of_bounds = state["out_of_bounds"]
            self.calibrators = [deserialize_isotonic(d) for d in state["calibrators"]]
        except Exception as e:
            raise SerializationError(f"Failed to load state: {e}") from e

    def __repr__(self) -> str:
        return f"IsotonicRegressionCalibrator(n_classes={self.n_classes}, out_of_bounds='{self.out_of_bounds}')"


class HistogramBinningCalibrator(BaseCalibrator):
    """
    Histogram binning calibration: bins predicted probabilities and uses empirical frequencies.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.bin_edges: list = []
        self.bin_true_rates: list = []

    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        _validate_fit_inputs(logits, labels)
        probs = F.softmax(logits, dim=1).cpu().detach().numpy()
        labels_np = labels.cpu().detach().numpy()
        _, n_classes = probs.shape
        self.bin_edges = []
        self.bin_true_rates = []

        for k in range(n_classes):
            pk = probs[:, k]
            edges = np.linspace(0.0, 1.0, self.n_bins + 1)
            bin_ids = np.digitize(pk, edges, right=True) - 1
            bin_ids = np.clip(bin_ids, 0, self.n_bins - 1)
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
        from ..exceptions import NotFittedError

        if not self.bin_edges:
            raise NotFittedError(
                "HistogramBinningCalibrator has not been fitted. Call fit() first."
            )
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
        calibrated = calibrated / calibrated.sum(dim=1, keepdim=True).clamp(min=1e-10)
        return Categorical(probs=calibrated)

    def state_dict(self) -> dict:
        """Save histogram binning state."""
        return {
            "n_bins": self.n_bins,
            "bin_edges": [e.tolist() for e in self.bin_edges],
            "bin_true_rates": [r.tolist() for r in self.bin_true_rates],
        }

    def load_state_dict(self, state: dict) -> None:
        """Load histogram binning state."""
        from ..exceptions import SerializationError

        try:
            self.n_bins = state["n_bins"]
            self.bin_edges = [np.array(e) for e in state["bin_edges"]]
            self.bin_true_rates = [np.array(r) for r in state["bin_true_rates"]]
        except Exception as e:
            raise SerializationError(f"Failed to load state: {e}") from e

    def __repr__(self) -> str:
        n_classes = len(self.bin_edges) if self.bin_edges else 0
        return (
            f"HistogramBinningCalibrator(n_bins={self.n_bins}, n_classes={n_classes})"
        )


class PlattScalingCalibrator(BaseCalibrator):
    """
    Platt scaling (logistic regression) calibration per class (one-vs-rest).
    """

    def __init__(self):
        self.models: list = []
        self.n_classes: int = 0

    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        _validate_fit_inputs(logits, labels)
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
        from ..exceptions import NotFittedError

        if not self.models:
            raise NotFittedError(
                "PlattScalingCalibrator has not been fitted. Call fit() first."
            )
        probs = F.softmax(logits, dim=1).cpu().detach().numpy()
        calibrated = np.zeros_like(probs)

        for k, lr in enumerate(self.models):
            calibrated[:, k] = lr.predict_proba(probs[:, [k]])[:, 1]

        calibrated = torch.tensor(calibrated, device=logits.device, dtype=torch.float32)
        calibrated = calibrated / calibrated.sum(dim=1, keepdim=True).clamp(min=1e-10)
        return Categorical(probs=calibrated)

    def state_dict(self) -> dict:
        """Save Platt scaling models."""
        from .._sklearn_io import serialize_logistic

        return {
            "n_classes": self.n_classes,
            "models": [serialize_logistic(lr) for lr in self.models],
        }

    def load_state_dict(self, state: dict) -> None:
        """Load Platt scaling models."""
        from .._sklearn_io import deserialize_logistic
        from ..exceptions import SerializationError

        try:
            self.n_classes = state["n_classes"]
            self.models = [deserialize_logistic(d) for d in state["models"]]
        except Exception as e:
            raise SerializationError(f"Failed to load state: {e}") from e

    def __repr__(self) -> str:
        return f"PlattScalingCalibrator(n_classes={self.n_classes})"


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
        _validate_fit_inputs(logits, labels)
        device = logits.device
        self.to(device)
        labels = labels.to(device)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iters)
        nll = nn.CrossEntropyLoss()

        def _eval():
            optimizer.zero_grad()
            scaled = logits / self.temperature.clamp(min=1e-4)
            loss = nll(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(_eval)
        return self

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-4)

    def predict(self, logits: torch.Tensor) -> Categorical:
        scaled = self.forward(logits)
        probs = F.softmax(scaled, dim=1)
        return Categorical(probs=probs)

    @classmethod
    def load(cls, path: str) -> "VectorScaling":
        """Load VectorScaling from a file."""
        from ..exceptions import SerializationError

        try:
            state = torch.load(path, weights_only=True)
            n_classes = state["temperature"].shape[0]
            instance = cls(n_classes=n_classes)
            instance.load_state_dict(state)
            return instance
        except Exception as e:
            raise SerializationError(f"Failed to load from {path}: {e}") from e

    def __repr__(self) -> str:
        temps = self.temperature.detach().cpu().numpy()
        return f"VectorScaling(n_classes={len(temps)}, temperature_range=[{temps.min():.4f}, {temps.max():.4f}])"


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
        _validate_fit_inputs(logits, labels)
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

    @classmethod
    def load(cls, path: str) -> "MatrixScaling":
        """Load MatrixScaling from a file."""
        from ..exceptions import SerializationError

        try:
            state = torch.load(path, weights_only=True)
            n_classes = state["weight"].shape[0]
            instance = cls(n_classes=n_classes)
            instance.load_state_dict(state)
            return instance
        except Exception as e:
            raise SerializationError(f"Failed to load from {path}: {e}") from e

    def __repr__(self) -> str:
        n_classes = self.weight.shape[0]
        return f"MatrixScaling(n_classes={n_classes})"


class DirichletCalibrator(nn.Module, BaseCalibrator):
    """
    Dirichlet Calibration (Kull et al., 2019).

    Affine transformation of logits with optional L2 regularization toward
    the identity matrix, inspired by the Dirichlet calibration framework.
    More flexible than temperature scaling (generalizes MatrixScaling).

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
        _validate_fit_inputs(logits, labels)
        device = logits.device
        self.to(device)
        labels = labels.to(device)

        optimizer = torch.optim.LBFGS(
            [self.weight, self.bias], lr=lr, max_iter=max_iters
        )

        def _eval():
            optimizer.zero_grad()

            # Transform logits
            transformed = logits @ self.weight.T + self.bias

            # Cross-entropy on affine-transformed logits
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

    def state_dict(self) -> dict:
        """Return state dict including mu and nn.Module parameters."""
        state = super().state_dict()
        state["_mu"] = self.mu
        return state

    def load_state_dict(self, state: dict, **kwargs) -> None:
        """Load state dict, restoring mu alongside nn.Module parameters."""
        self.mu = state.get("_mu")
        module_state = {k: v for k, v in state.items() if k != "_mu"}
        super().load_state_dict(module_state, **kwargs)

    @classmethod
    def load(cls, path: str) -> "DirichletCalibrator":
        """Load DirichletCalibrator from a file."""
        from ..exceptions import SerializationError

        try:
            state = torch.load(path, weights_only=True)
            n_classes = state["weight"].shape[0]
            mu = state.get("_mu")
            instance = cls(n_classes=n_classes, mu=mu)
            instance.load_state_dict(state)
            return instance
        except Exception as e:
            raise SerializationError(f"Failed to load from {path}: {e}") from e

    def __repr__(self) -> str:
        mu_str = f"{self.mu:.4f}" if self.mu is not None else "None"
        return f"DirichletCalibrator(n_classes={self.n_classes}, mu={mu_str})"


class BetaCalibrator(BaseCalibrator):
    """
    Beta Calibration for binary classification (Kull et al., 2017).

    Fits a three-parameter model: logit(q) = a*log(p) + b*log(1-p) + c,
    where p is the uncalibrated probability and q is the calibrated probability.
    This is equivalent to assuming the scores for each class follow Beta
    distributions with different parameters.

    Reference:
        Kull et al., "Beta calibration: a well-founded and easily implemented
        improvement on logistic calibration" (AISTATS 2017)
    """

    def __init__(self):
        self.a = None
        self.b = None
        self.c = None
        self.is_binary = None
        self._multiclass_calibrator = None

    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Fit Beta calibration on binary classification data.
        Falls back to isotonic regression for multiclass.

        Args:
            logits: Validation logits (N, 2) or (N, C) for multiclass
            labels: Binary labels (N,)
        """
        _validate_fit_inputs(logits, labels)
        # Check if binary or multiclass
        if logits.dim() == 2 and logits.shape[1] > 2:
            import warnings

            warnings.warn(
                "BetaCalibrator is designed for binary classification. "
                f"Got {logits.shape[1]} classes — falling back to "
                "IsotonicRegressionCalibrator. Use IsotonicRegressionCalibrator "
                "directly for multiclass calibration.",
                stacklevel=2,
            )
            self.is_binary = False
            self._multiclass_calibrator = IsotonicRegressionCalibrator()
            self._multiclass_calibrator.fit(logits, labels)
            return self

        self.is_binary = True

        # Convert to probabilities
        if logits.dim() == 2:
            probs = F.softmax(logits, dim=1)[:, 1]
        else:
            probs = torch.sigmoid(logits)

        probs_np = probs.cpu().detach().numpy().astype(np.float64)
        labels_np = labels.cpu().detach().numpy()

        # Clip to avoid log(0)
        eps = 1e-12
        probs_np = np.clip(probs_np, eps, 1.0 - eps)

        # Build Beta calibration features: [log(p), log(1-p)]
        # Model: logit(q) = a*log(p) + b*log(1-p) + c
        features = np.column_stack([np.log(probs_np), np.log(1.0 - probs_np)])

        lr = LogisticRegression(solver="lbfgs", max_iter=1000, C=1e10)
        lr.fit(features, labels_np)

        self.a = float(lr.coef_[0, 0])
        self.b = float(lr.coef_[0, 1])
        self.c = float(lr.intercept_[0])

        return self

    def _calibrate_probs(self, probs_np: np.ndarray) -> np.ndarray:
        """Apply the fitted Beta calibration map to probabilities."""
        eps = 1e-12
        probs_np = np.clip(probs_np, eps, 1.0 - eps)
        logit_q = self.a * np.log(probs_np) + self.b * np.log(1.0 - probs_np) + self.c
        return 1.0 / (1.0 + np.exp(-logit_q))

    def predict(self, logits: torch.Tensor) -> Categorical:
        """Get calibrated predictions."""
        from ..exceptions import NotFittedError

        if self.is_binary is None:
            raise NotFittedError(
                "BetaCalibrator has not been fitted. Call fit() first."
            )

        # Check if multiclass fallback
        if not self.is_binary:
            return self._multiclass_calibrator.predict(logits)

        # Binary classification
        if logits.dim() == 2:
            probs = F.softmax(logits, dim=1)[:, 1]
        else:
            probs = torch.sigmoid(logits)

        probs_np = probs.cpu().detach().numpy().astype(np.float64)
        calibrated_np = self._calibrate_probs(probs_np)

        calibrated_probs = torch.tensor(
            calibrated_np, device=logits.device, dtype=torch.float32
        )
        probs_both = torch.stack([1 - calibrated_probs, calibrated_probs], dim=1)

        return Categorical(probs=probs_both)

    def state_dict(self) -> dict:
        """Save Beta calibrator state."""
        is_binary = self.is_binary

        if not is_binary and self._multiclass_calibrator is not None:
            mc_state = self._multiclass_calibrator.state_dict()
        else:
            mc_state = None

        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "is_binary": is_binary,
            "multiclass_calibrator": mc_state,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load Beta calibrator state."""
        from ..exceptions import SerializationError

        try:
            self.a = state["a"]
            self.b = state["b"]
            self.c = state["c"]
            self.is_binary = state["is_binary"]

            mc_state = state["multiclass_calibrator"]
            if mc_state is not None:
                self._multiclass_calibrator = IsotonicRegressionCalibrator()
                self._multiclass_calibrator.load_state_dict(mc_state)
            else:
                self._multiclass_calibrator = None
        except Exception as e:
            raise SerializationError(f"Failed to load state: {e}") from e

    def __repr__(self) -> str:
        if self.a is not None:
            return f"BetaCalibrator(a={self.a:.4f}, b={self.b:.4f}, c={self.c:.4f})"
        return "BetaCalibrator()"
