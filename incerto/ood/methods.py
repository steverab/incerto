# incerto/ood/methods.py
import torch
import torch.nn.functional as F

from .base import OODDetector


class MSP(OODDetector):
    """Maximum-Softmax-Probability (Hendrycks & Gimpel, 2017)."""

    def score(self, x):
        logits = self.model(x)
        return 1 - F.softmax(logits, dim=-1).max(dim=-1).values

    def __repr__(self) -> str:
        return "MSP()"


class Energy(OODDetector):
    """Energy-based score (Liu et al., NeurIPS 2020)."""

    def __init__(self, model, temperature=1.0):
        super().__init__(model)
        self.T = temperature

    def score(self, x):
        e = -torch.logsumexp(self.model(x) / self.T, dim=-1)
        return e

    def state_dict(self) -> dict:
        """Save temperature parameter."""
        return {"temperature": self.T}

    def load_state_dict(self, state: dict) -> None:
        """Load temperature parameter."""
        self.T = state["temperature"]

    def __repr__(self) -> str:
        return f"Energy(temperature={self.T})"


class ODIN(OODDetector):
    """ODIN (Liang et al., ICLR 2018)."""

    def __init__(self, model, temperature=1000.0, epsilon=0.0014):
        super().__init__(model)
        self.T, self.eps = temperature, epsilon

    def score(self, x):
        x = x.clone().requires_grad_(True)
        logits = self.model(x) / self.T
        smax = F.softmax(logits, dim=-1)
        loss = -smax.max(dim=-1).values.mean()
        loss.backward()
        x_adv = x + self.eps * x.grad.sign()
        logits_adv = self.model(x_adv) / self.T
        return -F.softmax(logits_adv, dim=-1).max(dim=-1).values

    def state_dict(self) -> dict:
        """Save ODIN hyperparameters."""
        return {"temperature": self.T, "epsilon": self.eps}

    def load_state_dict(self, state: dict) -> None:
        """Load ODIN hyperparameters."""
        self.T = state["temperature"]
        self.eps = state["epsilon"]

    def __repr__(self) -> str:
        return f"ODIN(temperature={self.T}, epsilon={self.eps})"


class Mahalanobis(OODDetector):
    """Feature-space Mahalanobis (Lee et al., NeurIPS 2018)."""

    def __init__(self, model, layer_name="penultimate"):
        super().__init__(model)
        self.layer = self._hook(layer_name)
        self.class_means, self.precision = None, None  # filled by `fit`

    def fit(self, loader):
        acts, labels = [], []
        for x, y in loader:
            self.model(x.to(next(self.model.parameters()).device))
            acts.append(self.layer().flatten(1).cpu())
            labels.append(y.cpu())
        acts = torch.cat(acts)
        labels = torch.cat(labels)
        self.class_means = torch.stack(
            [acts[labels == c].mean(0) for c in torch.unique(labels)]
        )
        cov = torch.cov(acts.T)
        self.precision = torch.linalg.inv(cov + 1e-6 * torch.eye(cov.size(0)))

    def score(self, x):
        self.model(x)
        f = self.layer().flatten(1)
        d2 = (
            (f[:, None] - self.class_means)  # N×C×D
            @ self.precision
            * (f[:, None] - self.class_means)
        ).sum(
            -1
        )  # N×C
        return d2.min(dim=-1).values

    def _hook(self, name):
        for n, m in self.model.named_modules():
            if n.endswith(name):
                handle = m.register_forward_hook(
                    lambda _, __, out: setattr(self, "_tmp", out)
                )
                return lambda: self._tmp
        raise ValueError(f"Layer {name} not found")

    def state_dict(self) -> dict:
        """Save fitted Mahalanobis parameters."""
        return {
            "class_means": self.class_means,
            "precision": self.precision,
            "layer_name": getattr(self, "_layer_name", "penultimate"),
        }

    def load_state_dict(self, state: dict) -> None:
        """Load fitted Mahalanobis parameters."""
        from ..exceptions import SerializationError

        try:
            self.class_means = state["class_means"]
            self.precision = state["precision"]
            self._layer_name = state.get("layer_name", "penultimate")
        except Exception as e:
            raise SerializationError(f"Failed to load state: {e}") from e

    def __repr__(self) -> str:
        fitted = self.class_means is not None
        n_classes = len(self.class_means) if fitted else "not fitted"
        return f"Mahalanobis(layer='penultimate', n_classes={n_classes})"


class MaxLogit(OODDetector):
    """
    MaxLogit OOD detection (Hendrycks et al., 2019).

    Uses the maximum logit value as the OOD score. Simpler than MSP
    and often more effective as it doesn't require softmax normalization.
    """

    def score(self, x):
        logits = self.model(x)
        # Return negative max logit (higher logit = more ID-like)
        return -logits.max(dim=-1).values

    def __repr__(self) -> str:
        return "MaxLogit()"


class KNN(OODDetector):
    """
    KNN-based OOD detection (Sun et al., NeurIPS 2022).

    Computes OOD score as distance to k-th nearest neighbor in feature space.
    Requires fitting on training data.
    """

    def __init__(self, model, k=50, layer_name="penultimate"):
        super().__init__(model)
        self.k = k
        self.layer = self._hook(layer_name)
        self.train_features = None

    def fit(self, loader):
        """Store training features for KNN computation."""
        features = []
        for x, _ in loader:
            self.model(x.to(next(self.model.parameters()).device))
            features.append(self.layer().flatten(1).cpu())
        self.train_features = torch.cat(features)

    def score(self, x):
        """Compute OOD score as distance to k-th nearest neighbor."""
        if self.train_features is None:
            raise RuntimeError("Must call .fit() before .score()")

        self.model(x)
        test_features = self.layer().flatten(1).cpu()

        # Compute pairwise distances
        dists = torch.cdist(test_features, self.train_features)

        # Get k-th nearest neighbor distance
        kth_dist, _ = torch.kthvalue(dists, self.k, dim=-1)
        return kth_dist.to(x.device)

    def _hook(self, name):
        for n, m in self.model.named_modules():
            if n.endswith(name):
                handle = m.register_forward_hook(
                    lambda _, __, out: setattr(self, "_tmp", out)
                )
                return lambda: self._tmp
        raise ValueError(f"Layer {name} not found")

    def state_dict(self) -> dict:
        """Save KNN training features."""
        return {
            "k": self.k,
            "layer_name": getattr(self, "_layer_name", "penultimate"),
            "train_features": self.train_features,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load KNN training features."""
        from ..exceptions import SerializationError

        try:
            self.k = state["k"]
            self._layer_name = state.get("layer_name", "penultimate")
            self.train_features = state["train_features"]
        except Exception as e:
            raise SerializationError(f"Failed to load state: {e}") from e

    def __repr__(self) -> str:
        fitted = self.train_features is not None
        n_samples = len(self.train_features) if fitted else "not fitted"
        return f"KNN(k={self.k}, n_train_samples={n_samples})"
