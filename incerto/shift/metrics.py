"""
incerto.shift_detection.metrics
===============================

Pure functions that measure how far two sample sets differ.
Designed so that you can mix and match with your own detectors.
"""

from __future__ import annotations

import torch


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """Szekely–Rizzo energy distance, O(n²) naive implementation."""

    def pdist(t):  # pairwise ℓ2, excluding zero diagonal
        d = torch.cdist(t, t, p=2)
        n = t.shape[0]
        return d.sum() / (n * (n - 1)) if n > 1 else d.mean()

    return (2 * torch.cdist(x, y, p=2).mean() - pdist(x) - pdist(y)).item()


def total_variation(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> float:
    """Total variation between *discrete* distributions p and q."""
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    return 0.5 * torch.abs(p - q).sum().item()


def population_stability_index(p_hist, q_hist, eps: float = 1e-9) -> float:
    """Classic tabular PSI used in credit scoring."""
    p = p_hist / (p_hist.sum() + eps)
    q = q_hist / (q_hist.sum() + eps)
    p, q = p + eps, q + eps
    return ((q - p) * torch.log(q / p)).sum().item()


def wasserstein_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    p: float = 2.0,
    max_iter: int = 100,
    epsilon: float | None = None,
) -> float:
    """
    Wasserstein distance (Earth Mover's Distance) between two empirical distributions.

    Computes the p-Wasserstein distance between two samples using Sinkhorn algorithm
    for optimal transport. For p=1, this is the classic Earth Mover's Distance.

    Args:
        x: Source samples of shape (n, d)
        y: Target samples of shape (m, d)
        p: Order of the Wasserstein distance (default: 2.0 for W2 distance)
        max_iter: Maximum iterations for Sinkhorn algorithm
        epsilon: Entropy regularization parameter for Sinkhorn. If None (default),
            automatically scaled based on the cost matrix median to avoid numerical
            issues. Smaller values give more accurate results but may cause underflow.

    Returns:
        Wasserstein distance between the two distributions

    Reference:
        Cuturi, "Sinkhorn Distances: Lightspeed Computation of Optimal Transport"
        (NeurIPS 2013)

    Example:
        >>> source_features = model(source_data)
        >>> target_features = model(target_data)
        >>> distance = wasserstein_distance(source_features, target_features)
    """
    # For 1D case, use closed-form solution
    if x.shape[1] == 1:
        x_sorted = torch.sort(x.squeeze())[0]
        y_sorted = torch.sort(y.squeeze())[0]

        # Interpolate to same size
        n, m = len(x_sorted), len(y_sorted)
        if n != m:
            max_size = max(n, m)
            x_interp = torch.nn.functional.interpolate(
                x_sorted.view(1, 1, -1),
                size=max_size,
                mode="linear",
                align_corners=True,
            ).squeeze()
            y_interp = torch.nn.functional.interpolate(
                y_sorted.view(1, 1, -1),
                size=max_size,
                mode="linear",
                align_corners=True,
            ).squeeze()
        else:
            x_interp, y_interp = x_sorted, y_sorted

        return torch.mean(torch.abs(x_interp - y_interp) ** p).item() ** (1.0 / p)

    # For multi-dimensional case, use Sinkhorn algorithm
    n, m = x.shape[0], y.shape[0]

    # Compute cost matrix (pairwise L2 distances raised to power p)
    C = torch.cdist(x, y, p=2) ** p

    # Auto-scale epsilon if not provided to avoid numerical issues
    # Use median of cost matrix as a robust scale estimate
    if epsilon is None:
        epsilon = float(C.median()) * 0.05
        epsilon = max(epsilon, 1e-3)  # Ensure minimum regularization

    # Uniform distribution over samples
    a = torch.ones(n, device=x.device) / n
    b = torch.ones(m, device=y.device) / m

    # Sinkhorn iterations with entropy regularization
    K = torch.exp(-C / epsilon)

    u = torch.ones(n, device=x.device) / n
    v = torch.ones(m, device=y.device) / m

    for _ in range(max_iter):
        u = a / (K @ v + 1e-10)
        v = b / (K.T @ u + 1e-10)

    # Compute optimal transport cost
    transport_plan = u.unsqueeze(1) * K * v.unsqueeze(0)
    cost = (transport_plan * C).sum().item()

    return cost ** (1.0 / p)


def sliced_wasserstein_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    num_projections: int = 100,
    p: float = 2.0,
    seed: int | None = None,
) -> float:
    """
    Sliced Wasserstein distance between two empirical distributions.

    Projects the distributions onto random 1D lines and computes the average
    Wasserstein distance across projections. This is much faster than the full
    Wasserstein distance and scales to high dimensions.

    Args:
        x: Source samples of shape (n, d)
        y: Target samples of shape (m, d)
        num_projections: Number of random projections (default: 100)
        p: Order of the Wasserstein distance (default: 2.0)
        seed: Random seed for reproducibility (does not affect global state)

    Returns:
        Sliced Wasserstein distance averaged over random projections

    Reference:
        Rabin et al., "Wasserstein Barycenter and Its Application to Texture Mixing"
        (SSVM 2011)
        Kolouri et al., "Sliced-Wasserstein Autoencoder" (ICLR 2019)

    Example:
        >>> source_features = model(source_data)
        >>> target_features = model(target_data)
        >>> distance = sliced_wasserstein_distance(source_features, target_features)
    """
    # Use a local Generator to avoid modifying global random state
    generator = torch.Generator(device=x.device)
    if seed is not None:
        generator.manual_seed(seed)

    d = x.shape[1]
    distances = []

    for _ in range(num_projections):
        # Random projection direction
        theta = torch.randn(d, device=x.device, generator=generator)
        theta = theta / torch.norm(theta)

        # Project samples onto theta
        x_proj = x @ theta
        y_proj = y @ theta

        # Sort projected samples
        x_sorted = torch.sort(x_proj)[0]
        y_sorted = torch.sort(y_proj)[0]

        # Compute 1D Wasserstein distance
        # If different sizes, interpolate to match
        n, m = len(x_sorted), len(y_sorted)
        if n != m:
            max_size = max(n, m)
            x_interp = torch.nn.functional.interpolate(
                x_sorted.view(1, 1, -1),
                size=max_size,
                mode="linear",
                align_corners=True,
            ).squeeze()
            y_interp = torch.nn.functional.interpolate(
                y_sorted.view(1, 1, -1),
                size=max_size,
                mode="linear",
                align_corners=True,
            ).squeeze()
        else:
            x_interp, y_interp = x_sorted, y_sorted

        # p-th power of 1D Wasserstein distance
        dist_p = torch.mean(torch.abs(x_interp - y_interp) ** p)
        distances.append(dist_p)

    # SW_p = (E[W_p^p])^{1/p}
    return torch.stack(distances).mean().item() ** (1.0 / p)
