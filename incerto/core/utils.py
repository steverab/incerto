"""
Core utility functions used across multiple modules.
"""

import torch


def pairwise_squared_euclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distances between two sets of vectors.

    Args:
        x: Tensor of shape (n, d)
        y: Tensor of shape (m, d)

    Returns:
        Tensor of shape (n, m) containing pairwise squared distances.
    """
    # Expand dims for broadcasting
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x^T y
    x_norm = (x**2).sum(dim=1, keepdim=True)  # (n, 1)
    y_norm = (y**2).sum(dim=1, keepdim=True)  # (m, 1)
    dist = x_norm + y_norm.T - 2.0 * torch.mm(x, y.T)  # (n, m)
    return dist.clamp(min=0.0)  # Numerical stability
