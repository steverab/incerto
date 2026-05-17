# incerto/ood/methods.py
import torch
import torch.nn.functional as F

from .base import OODDetector


class _FeatureHookMixin:
    """Mixin providing a forward-hook mechanism for feature extraction."""

    _hook_handle: object

    def _hook(self, name):
        for n, m in self.model.named_modules():
            if n.endswith(name):
                if self._hook_handle is not None:
                    self._hook_handle.remove()
                self._hook_handle = m.register_forward_hook(
                    lambda _, __, out: setattr(self, "_tmp", out)
                )
                return lambda: self._tmp
        raise ValueError(f"Layer {name} not found")

    def __del__(self):
        if getattr(self, "_hook_handle", None) is not None:
            self._hook_handle.remove()


class MSP(OODDetector):
    """Maximum-Softmax-Probability (MSP) baseline OOD detector.

    Returns ``1 - max softmax probability`` as the OOD score, so higher means
    more likely OOD. Despite its simplicity, MSP is a strong baseline.

    Reference:
        Hendrycks & Gimpel, "A Baseline for Detecting Misclassified and
        Out-of-Distribution Examples in Neural Networks", ICLR 2017.

    Args:
        model: A trained classifier returning logits of shape
            ``(batch, n_classes)``.
    """

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute OOD scores for a batch of inputs.

        Args:
            x: Input batch passed to ``model``.

        Returns:
            Tensor of shape ``(batch,)`` with values in ``[0, 1)``. Higher
            values indicate higher OOD likelihood.
        """
        logits = self.model(x)
        return 1 - F.softmax(logits, dim=-1).max(dim=-1).values

    def __repr__(self) -> str:
        return "MSP()"


class Energy(OODDetector):
    """Energy-based OOD detector.

    Computes the free-energy score ``-T * logsumexp(logits / T)`` for each
    input. Lower energy is associated with in-distribution data, so the
    returned score (negated logsumexp) is *higher* for OOD inputs.

    Reference:
        Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020.

    Args:
        model: A trained classifier returning logits of shape
            ``(batch, n_classes)``.
        temperature: Temperature ``T`` applied to logits before logsumexp.
            Default: 1.0.
    """

    def __init__(self, model, temperature: float = 1.0):
        super().__init__(model)
        self.temperature = temperature

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy-based OOD scores.

        Args:
            x: Input batch passed to ``model``.

        Returns:
            Tensor of shape ``(batch,)`` of energy scores; higher means more OOD.
        """
        e = -torch.logsumexp(self.model(x) / self.temperature, dim=-1)
        return e

    def state_dict(self) -> dict:
        """Save temperature parameter."""
        return {"temperature": self.temperature}

    def load_state_dict(self, state: dict) -> None:
        """Load temperature parameter."""
        self.temperature = state["temperature"]

    def __repr__(self) -> str:
        return f"Energy(temperature={self.temperature})"


class ODIN(OODDetector):
    """
    ODIN: Out-of-Distribution detector for Neural Networks (Liang et al., ICLR 2018).

    Uses temperature scaling and input perturbation to separate ID/OOD scores.

    Note:
        ``score()`` computes input gradients for perturbation and must NOT
        be called inside a ``torch.no_grad()`` context. Use ``score()``
        directly instead of ``predict()`` — the inherited ``predict()``
        wraps the call in ``@torch.no_grad()`` which silently disables the
        perturbation mechanism and degrades detection performance.
    """

    def __init__(self, model, temperature=1000.0, epsilon=0.0014):
        super().__init__(model)
        self.temperature = temperature
        self.epsilon = epsilon

    def score(self, x):
        if not torch.is_grad_enabled():
            raise RuntimeError(
                "ODIN.score() requires gradients for input perturbation but "
                "is running inside a torch.no_grad() context. Call score() "
                "directly instead of predict()."
            )
        x = x.clone().requires_grad_(True)
        logits = self.model(x) / self.temperature
        smax = F.softmax(logits, dim=-1)
        loss = -smax.max(dim=-1).values.mean()
        loss.backward()
        x_adv = x - self.epsilon * x.grad.sign()
        with torch.no_grad():
            logits_adv = self.model(x_adv) / self.temperature
        return -F.softmax(logits_adv, dim=-1).max(dim=-1).values

    def predict(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Predict whether inputs are OOD using a threshold.

        Overrides the base class to avoid ``@torch.no_grad()``, which
        would break the input-perturbation mechanism.
        """
        return self.score(x) > threshold

    def state_dict(self) -> dict:
        """Save ODIN hyperparameters."""
        return {"temperature": self.temperature, "epsilon": self.epsilon}

    def load_state_dict(self, state: dict) -> None:
        """Load ODIN hyperparameters."""
        self.temperature = state["temperature"]
        self.epsilon = state["epsilon"]

    def __repr__(self) -> str:
        return f"ODIN(temperature={self.temperature}, epsilon={self.epsilon})"


class Mahalanobis(_FeatureHookMixin, OODDetector):
    """Feature-space Mahalanobis (Lee et al., NeurIPS 2018)."""

    def __init__(self, model, layer_name="penultimate"):
        super().__init__(model)
        self._layer_name = layer_name
        self._hook_handle = None
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
        self.class_means = torch.stack([acts[labels == c].mean(0) for c in torch.unique(labels)])
        cov = torch.cov(acts.T)
        ridge = max(1e-6, 1e-6 * cov.diag().mean().item())
        self.precision = torch.linalg.inv(cov + ridge * torch.eye(cov.size(0)))

    @torch.no_grad()
    def score(self, x):
        if self.class_means is None:
            from ..exceptions import NotFittedError

            raise NotFittedError("Must call .fit() before .score()")
        self.model(x)
        f = self.layer().flatten(1)
        # Move fitted stats to feature device for GPU compatibility
        cm = self.class_means.to(f.device)
        prec = self.precision.to(f.device)
        diff = f[:, None] - cm  # N×C×D
        d2 = (diff @ prec * diff).sum(-1)  # N×C
        return d2.min(dim=-1).values

    def state_dict(self) -> dict:
        """Save fitted Mahalanobis parameters."""
        return {
            "class_means": self.class_means,
            "precision": self.precision,
            "layer_name": getattr(self, "_layer_name", "penultimate"),
        }

    def load_state_dict(self, state: dict) -> None:
        """Load fitted Mahalanobis parameters."""
        from ..exceptions import serialization_errors

        with serialization_errors("load state"):
            self.class_means = state["class_means"]
            self.precision = state["precision"]
            new_layer = state.get("layer_name", "penultimate")
            if new_layer != self._layer_name:
                self._layer_name = new_layer
                self.layer = self._hook(new_layer)

    def __repr__(self) -> str:
        fitted = self.class_means is not None
        n_classes = len(self.class_means) if fitted else "not fitted"
        return f"Mahalanobis(layer='{self._layer_name}', n_classes={n_classes})"


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


class KNN(_FeatureHookMixin, OODDetector):
    """
    KNN-based OOD detection (Sun et al., NeurIPS 2022).

    Computes OOD score as distance to k-th nearest neighbor in feature space.
    Requires fitting on training data.
    """

    def __init__(self, model, k=50, layer_name="penultimate"):
        super().__init__(model)
        self.k = k
        self._layer_name = layer_name
        self._hook_handle = None
        self.layer = self._hook(layer_name)
        self.train_features = None

    def fit(self, loader):
        """Store training features for KNN computation."""
        features = []
        for x, _ in loader:
            self.model(x.to(next(self.model.parameters()).device))
            features.append(self.layer().flatten(1).cpu())
        self.train_features = torch.cat(features)

    @torch.no_grad()
    def score(self, x):
        """Compute OOD score as distance to k-th nearest neighbor."""
        if self.train_features is None:
            from ..exceptions import NotFittedError

            raise NotFittedError("Must call .fit() before .score()")

        self.model(x)
        test_features = self.layer().flatten(1)

        # Compute pairwise distances (both on same device)
        train_f = self.train_features.to(test_features.device)
        dists = torch.cdist(test_features, train_f)

        # Get k-th nearest neighbor distance
        # Note: kthvalue not supported on MPS, fall back to CPU
        if dists.device.type == "mps":
            kth_dist, _ = torch.kthvalue(dists.cpu(), self.k, dim=-1)
            return kth_dist.to(test_features.device)
        kth_dist, _ = torch.kthvalue(dists, self.k, dim=-1)
        return kth_dist

    def state_dict(self) -> dict:
        """Save KNN training features."""
        return {
            "k": self.k,
            "layer_name": getattr(self, "_layer_name", "penultimate"),
            "train_features": self.train_features,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load KNN training features."""
        from ..exceptions import serialization_errors

        with serialization_errors("load state"):
            self.k = state["k"]
            self.train_features = state["train_features"]
            new_layer = state.get("layer_name", "penultimate")
            if new_layer != self._layer_name:
                self._layer_name = new_layer
                self.layer = self._hook(new_layer)

    def __repr__(self) -> str:
        fitted = self.train_features is not None
        n_samples = len(self.train_features) if fitted else "not fitted"
        return f"KNN(k={self.k}, n_train_samples={n_samples})"
