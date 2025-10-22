"""
Core utilities and functions for incerto.

This module provides fundamental utilities used across the library.
"""

from .utils import pairwise_squared_euclidean
from .entropy import entropy, predictive_entropy

__all__ = [
    "pairwise_squared_euclidean",
    "entropy",
    "predictive_entropy",  # Alias for backward compatibility
]
