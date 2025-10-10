import numpy as np


def predictive_entropy(probs):
    """
    Calculate the predictive entropy of a probability distribution.

    Parameters:
    probs (np.ndarray): A 1D array of probabilities.

    Returns:
    float: The predictive entropy.
    """
    if not isinstance(probs, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("Probabilities must be in the range [0, 1].")

    if np.isclose(np.sum(probs), 0):
        return 0.0

    return -np.sum(
        probs * np.log(probs + 1e-10)
    )  # Adding a small constant to avoid log(0)
