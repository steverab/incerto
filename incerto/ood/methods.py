# incerto/ood/methods.py
from abc import ABC, abstractmethod
import torch, torch.nn.functional as F


class OODDetector(ABC):
    """
    Base class: any detector only needs to implement `score`.
    Higher scores  ⇒  more OOD-like.
    """

    def __init__(self, model):
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @abstractmethod
    def score(self, x: torch.Tensor) -> torch.Tensor: ...

    @torch.no_grad()
    def predict(self, x, threshold):
        return self.score(x) > threshold  # Bool mask


class MSP(OODDetector):
    """Maximum-Softmax-Probability (Hendrycks & Gimpel, 2017)."""

    def score(self, x):
        logits = self.model(x)
        return 1 - F.softmax(logits, dim=-1).max(dim=-1).values


class Energy(OODDetector):
    """Energy-based score (Liu et al., NeurIPS 2020)."""

    def __init__(self, model, temperature=1.0):
        super().__init__(model)
        self.T = temperature

    def score(self, x):
        e = -torch.logsumexp(self.model(x) / self.T, dim=-1)
        return e


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
            acts.append(self.layer.flatten(1).cpu())
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
        f = self.layer.flatten(1)
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
