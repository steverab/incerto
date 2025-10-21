incerto: Uncertainty Quantification for Machine Learning
=========================================================

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.10+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://github.com/steverab/incerto/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/steverab/incerto/actions/workflows/tests.yml
   :alt: Tests

**incerto** is a comprehensive Python library for **uncertainty quantification in machine learning**.
It provides state-of-the-art methods for calibration, out-of-distribution detection, conformal prediction,
selective prediction, and uncertainty estimation in deep learning and LLMs.

*Latin "incerto" = "uncertain, doubtful, unsure" - embracing uncertainty in ML predictions.*

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install incerto

From source:

.. code-block:: bash

   git clone https://github.com/steverab/incerto.git
   cd incerto
   pip install -e .

Quick Example
~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from incerto.calibration import TemperatureScaling, ece_score

   # Post-hoc calibration
   calibrator = TemperatureScaling()
   calibrator.fit(val_logits, val_labels)

   # Get calibrated predictions
   calibrated_probs = calibrator.predict(test_logits).probs

   # Evaluate calibration
   ece = ece_score(test_logits, test_labels)
   print(f"Expected Calibration Error: {ece:.4f}")

Key Features
------------

📊 **Calibration**
   Post-hoc methods (Temperature scaling, Platt scaling, Isotonic regression, Dirichlet calibration, Beta calibration)
   and training-time methods (Label smoothing, Focal loss, Evidential deep learning)

🎲 **Out-of-Distribution Detection**
   Score-based methods (MSP, MaxLogit, Energy, ODIN), distance-based methods (Mahalanobis, KNN),
   and training methods (Mixup, CutMix, Outlier Exposure)

🎯 **Conformal Prediction**
   Distribution-free uncertainty with coverage guarantees for classification (Inductive CP, APS, RAPS, Mondrian CP)
   and regression (Jackknife+, CV+, CQR)

🔍 **Selective Prediction**
   Confidence thresholding, Self-Adaptive Training (SAT), Deep Gambler, SelectiveNet,
   and risk-coverage tradeoffs

🤖 **LLM Uncertainty**
   Token-level (Entropy, confidence, perplexity), sequence-level (Sequence probability, average log-prob),
   and sampling-based uncertainty (Self-consistency, semantic entropy, predictive entropy)

🎲 **Bayesian Deep Learning**
   MC Dropout, Deep Ensembles, SWAG, Laplace Approximation, Variational Inference,
   and uncertainty decomposition

🎯 **Active Learning**
   Acquisition functions (Entropy, BALD, margin, variance ratio) and query strategies
   (Uncertainty sampling, diversity sampling, Core-Set, BADGE, BatchBALD)

📈 **Distribution Shift Detection**
   Statistical tests (MMD, Energy distance, Kolmogorov-Smirnov, Wasserstein distance),
   classifier-based detection, label shift detection, and importance weighting

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/overview
   guide/installation
   guide/quickstart
   guide/calibration
   guide/ood
   guide/conformal
   guide/selective
   guide/llm
   guide/bayesian
   guide/active
   guide/shift

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/calibration
   api/ood
   api/conformal
   api/sp
   api/llm
   api/bayesian
   api/active
   api/shift
   api/data
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/calibration
   examples/ood
   examples/conformal
   examples/llm

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/contributing
   development/changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
