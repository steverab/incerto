"""
Entropy and information-theoretic measures.

This module provides numpy-based entropy calculations for single probability distributions.
For batched torch tensor operations, see incerto.bayesian.utils.
"""

import numpy as np


def entropy(probs: np.ndarray) -> float:
    """
    Calculate the Shannon entropy of a probability distribution.

    This is a general-purpose entropy function for numpy arrays.
    For PyTorch tensors with batched predictions, use
    `incerto.bayesian.utils.predictive_entropy` instead.

    Args:
        probs: A 1D numpy array of probabilities that sum to 1.

    Returns:
        The Shannon entropy H(p) = -∑ p(x) log p(x)

    Raises:
        TypeError: If probs is not a numpy array
        ValueError: If probabilities are not in [0, 1]

    Example:
        >>> probs = np.array([0.7, 0.2, 0.1])
        >>> entropy(probs)
        0.8018...
    """
    if not isinstance(probs, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("Probabilities must be in the range [0, 1].")

    if np.isclose(np.sum(probs), 0):
        return 0.0

    return float(-np.sum(probs * np.log(probs + 1e-10)))


# Alias for backward compatibility
predictive_entropy = entropy
