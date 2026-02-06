Overview
========

**incerto** is a comprehensive Python library for uncertainty quantification in machine learning.
The library is built on PyTorch and provides a unified interface for various uncertainty quantification methods.

Why Uncertainty Quantification?
--------------------------------

Modern machine learning models are increasingly deployed in safety-critical applications where
understanding when a model is uncertain is crucial. Uncertainty quantification enables:

- **Trustworthy AI**: Know when your model is confident vs. uncertain
- **Safe Deployment**: Avoid catastrophic failures by abstaining on uncertain predictions
- **Active Learning**: Select the most informative samples for labeling
- **Model Improvement**: Identify data distribution shifts and calibration issues
- **Regulatory Compliance**: Meet requirements for explainable and reliable AI

Library Structure
-----------------

incerto is organized into eight core modules:

**calibration**
    Ensures that model confidence scores match empirical accuracy. Includes post-hoc methods
    like temperature scaling and training-time methods like label smoothing.

**ood**
    Detects when inputs are out-of-distribution (OOD) - significantly different from training data.
    Includes score-based and distance-based methods.

**conformal**
    Provides distribution-free prediction sets with finite-sample coverage guarantees.
    Works for both classification and regression.

**sp (selective prediction)**
    Enables models to abstain from predictions when uncertain, optimizing risk-coverage tradeoffs.

**llm**
    Specialized uncertainty quantification for large language models, including token-level,
    sequence-level, and sampling-based methods.

**bayesian**
    Approximate Bayesian inference methods like MC Dropout, Deep Ensembles, and SWAG
    for epistemic uncertainty estimation.

**active**
    Active learning acquisition functions and query strategies for efficient data labeling.

**shift**
    Statistical tests for detecting distribution shifts between training and deployment data.

Design Principles
-----------------

**Unified API**
    Consistent interface across all methods - fit, predict, score patterns

**PyTorch Native**
    Seamless integration with PyTorch models and training loops

**Research-Backed**
    Implementations based on peer-reviewed publications with proper citations

**Well-Tested**
    900+ tests ensuring correctness and reliability

**Modular**
    Use individual components or combine them for comprehensive uncertainty quantification

Quick Comparison
----------------

+-------------------+------------------+--------------------+------------------+
| Method            | Training Cost    | Inference Cost     | Coverage Type    |
+===================+==================+====================+==================+
| Calibration       | None (post-hoc)  | Negligible         | Approximate      |
+-------------------+------------------+--------------------+------------------+
| OOD Detection     | Optional         | Low                | Heuristic        |
+-------------------+------------------+--------------------+------------------+
| Conformal         | None             | Low                | Guaranteed       |
+-------------------+------------------+--------------------+------------------+
| Bayesian Methods  | High             | Medium-High        | Principled       |
+-------------------+------------------+--------------------+------------------+
| Selective Pred    | Optional         | Low                | Risk-controlled  |
+-------------------+------------------+--------------------+------------------+

Next Steps
----------

- :doc:`installation` - Install incerto
- :doc:`quickstart` - Get started with examples
- :doc:`method_selection` - Which method should I use?
- :doc:`calibration` - Learn about calibration methods
