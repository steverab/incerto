"""
incerto: A PyTorch library for uncertainty quantification in deep learning.

Submodules:
    - calibration: Calibration methods for neural networks
    - conformal: Conformal prediction for classification and regression
    - llm: LLM uncertainty quantification methods
    - ood: Out-of-distribution detection methods
    - shift: Distribution shift detection
    - sp: Selective prediction with rejection/abstention
    - bayesian: Bayesian deep learning methods
    - active: Active learning for efficient data labeling
"""

__version__ = "0.1.0"

# Core utilities
# Submodules - import them so users can do `from incerto import calibration`
from . import active, bayesian, calibration, conformal, data, llm, ood, shift, sp, utils
from .core import entropy

# Exceptions
from .exceptions import (
    CalibrationError,
    ConfigurationError,
    DataError,
    IncertoError,
    NotFittedError,
    SerializationError,
)

__all__ = [
    "__version__",
    "entropy",
    # Exceptions
    "IncertoError",
    "NotFittedError",
    "CalibrationError",
    "SerializationError",
    "ConfigurationError",
    "DataError",
    # Modules
    "calibration",
    "conformal",
    "llm",
    "ood",
    "shift",
    "sp",
    "bayesian",
    "active",
    "data",
    "utils",
]
