"""
incerto: A PyTorch library for uncertainty quantification in deep learning.

Submodules:
    - calibration: Calibration methods for neural networks
    - conformal: Conformal prediction for classification and regression
    - llm: LLM uncertainty quantification methods
    - ood: Out-of-distribution detection methods
    - shift: Distribution shift detection
    - sp: Selective prediction with rejection/abstention
"""

__version__ = "0.1.0"

# Core utilities
from .core import predictive_entropy

# Submodules - import them so users can do `from incerto import calibration`
from . import calibration
from . import conformal
from . import llm
from . import ood
from . import shift
from . import sp

__all__ = [
    "__version__",
    "predictive_entropy",
    "calibration",
    "conformal",
    "llm",
    "ood",
    "shift",
    "sp",
]
