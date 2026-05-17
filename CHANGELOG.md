# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-05-17

### Fixed

- **`import incerto` fails without `[vision]` extra** (regression in 0.1.0).
  `incerto.data.__init__` eagerly imported torchvision-dependent submodules
  (`vision`, `ood_benchmarks`) at package import time, breaking the base
  install for every user who didn't install `incerto[vision]`. Vision-related
  symbols are now imported conditionally; calling them without the extra
  still raises a clear ImportError, but the rest of the package imports
  cleanly. ([incerto/data/__init__.py](incerto/data/__init__.py))

### Notes

- **0.1.0 has been deleted from PyPI** due to the broken base install above.
  Per PyPI policy, the 0.1.0 version number remains permanently reserved and
  cannot be reused. 0.1.1 is the first installable release; use
  `pip install incerto` (which now resolves to 0.1.1 or later).

## [0.1.0] - 2026-05-15

Initial release of incerto, a comprehensive Python library for uncertainty quantification in machine learning.

### Tested Versions

This release was tested against the following dependency versions; later
versions may work but are unverified:

- Python 3.10 – 3.13
- PyTorch ≥ 2.0
- NumPy ≥ 1.24
- scikit-learn ≥ 1.3
- scipy ≥ 1.11
- matplotlib ≥ 3.8
- (optional, `[vision]`) torchvision ≥ 0.16
- (optional, `[llm]`) transformers ≥ 4.36, accelerate ≥ 0.25,
  sentence-transformers ≥ 2.2

### Added

#### Calibration (`incerto.calibration`)
- **Post-hoc methods**: Temperature Scaling, Vector Scaling, Matrix Scaling, Platt Scaling, Isotonic Regression, Histogram Binning, Dirichlet Calibration, Beta Calibration
- **Training-time methods**: Label Smoothing, Focal Loss, Confidence Penalty, Evidential Deep Learning, Temperature-Aware Training
- **Metrics**: ECE, MCE, Classwise ECE, Adaptive ECE, Brier Score, NLL
- **Visualization**: Reliability diagrams, calibration curves, confidence histograms

#### Out-of-Distribution Detection (`incerto.ood`)
- **Score-based methods**: Maximum Softmax Probability (MSP), MaxLogit, Energy Score, ODIN
- **Distance-based methods**: Mahalanobis Distance, K-Nearest Neighbors (KNN)
- **Training methods**: Mixup, CutMix, Outlier Exposure, Energy Regularization
- **Metrics**: AUROC, FPR@TPR

#### Conformal Prediction (`incerto.conformal`)
- **Classification**: Inductive Conformal Prediction (ICP), Adaptive Prediction Sets (APS), Regularized APS (RAPS), Mondrian Conformal Prediction
- **Regression**: Jackknife+, CV+, Conformalized Quantile Regression
- **Metrics**: Empirical coverage, conditional coverage, average set size
- **Visualization**: Coverage vs alpha plots, set size histograms

#### Selective Prediction (`incerto.sp`)
- Softmax Threshold (confidence thresholding)
- Self-Adaptive Training (SAT)
- Deep Gambler
- SelectiveNet
- **Metrics**: AURC, coverage-accuracy curves

#### Bayesian Deep Learning (`incerto.bayesian`)
- MC Dropout
- Deep Ensembles
- SWAG (Stochastic Weight Averaging - Gaussian)
- Laplace Approximation
- Variational Inference (Bayes by Backprop)
- **Utilities**: Uncertainty decomposition (epistemic/aleatoric), predictive entropy, mutual information

#### Distribution Shift Detection (`incerto.shift`)
- **Statistical tests**: MMD (Maximum Mean Discrepancy), Energy Distance, Kolmogorov-Smirnov Test, Wasserstein Distance
- **Classifier-based**: Black-Box Shift Detection (BBSD)
- Label Shift Detection
- Importance Weighting for covariate shift adaptation

#### LLM Uncertainty (`incerto.llm`)
- **Token-level**: Token Entropy, Token Confidence, Perplexity, Surprisal Score, Top-K Confidence
- **Sequence-level**: Sequence Probability, Average Log-Probability, Sequence Entropy
- **Sampling-based**: Self-Consistency, Semantic Entropy, Predictive Entropy, Mutual Information
- **Generation methods**: Beam Search Uncertainty, Nucleus Sampling Uncertainty, I Don't Know Detection, Contrastive Decoding
- **Calibration**: Histogram Binning, Platt Scaling for LLMs

#### Active Learning (`incerto.active`)
- **Acquisition functions**: Entropy Sampling, BALD, Least Confidence, Margin Sampling, Variance Ratio, Mean STD, BatchBALD
- **Query strategies**: Uncertainty Sampling, Diversity Sampling, Core-Set Selection, BADGE, Query by Committee

#### Data & Utilities (`incerto.data`, `incerto.utils`)
- Built-in vision datasets: MNIST, CIFAR-10, CIFAR-100, SVHN
- OOD benchmark datasets with standard splits
- Common architectures: ConvNet, MLP, ResNet18
- Training utilities and visualization tools

#### Documentation & Examples
- 8 Jupyter notebook tutorials covering all modules
- Sphinx documentation with API reference
- User guides for calibration and OOD detection

[0.1.0]: https://github.com/steverab/incerto/releases/tag/v0.1.0
