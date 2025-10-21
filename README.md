# incerto

[![Tests](https://github.com/stephanrabanser/incerto/actions/workflows/tests.yml/badge.svg)](https://github.com/stephanrabanser/incerto/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**incerto** is a comprehensive Python library for **uncertainty quantification in machine learning**. It provides state-of-the-art methods for calibration, out-of-distribution detection, conformal prediction, selective prediction, and uncertainty estimation in deep learning and LLMs.

Latin *incerto* = "uncertain, doubtful, unsure" - embracing uncertainty in ML predictions.

## 🎯 Key Features

**incerto** provides a unified interface for:

### 📊 **Calibration**
- **Post-hoc calibration**: Temperature scaling, Platt scaling, isotonic regression, histogram binning
- **Training-time methods**: Label smoothing, focal loss, confidence penalty, evidential deep learning
- **Metrics**: ECE, MCE, Brier score, NLL, reliability diagrams

### 🎲 **Out-of-Distribution (OOD) Detection**
- **Score-based methods**: MSP, MaxLogit, Energy, ODIN
- **Distance-based methods**: Mahalanobis distance, KNN
- **Training methods**: Mixup, CutMix, Outlier Exposure, Energy regularization

### 🎯 **Conformal Prediction**
- **Classification**: Inductive CP, APS, RAPS, Mondrian CP
- **Regression**: Jackknife+, CV+
- Distribution-free uncertainty quantification with coverage guarantees

### 🔍 **Selective Prediction**
- Confidence thresholding (Softmax Threshold)
- Self-Adaptive Training (SAT)
- Deep Gambler, SelectiveNet
- Risk-coverage tradeoffs

### 🤖 **LLM Uncertainty**
- **Token-level**: Entropy, confidence, perplexity, surprisal
- **Sequence-level**: Sequence probability, average log-prob
- **Sampling-based**: Self-consistency, semantic entropy, predictive entropy
- **Generation methods**: Beam search uncertainty, nucleus sampling, contrastive decoding

### 🎲 **Bayesian Deep Learning** ⭐ NEW
- **MC Dropout**: Uncertainty via dropout at test time
- **Deep Ensembles**: Train multiple models for robust predictions
- **SWAG**: Stochastic Weight Averaging - Gaussian
- **Laplace Approximation**: Gaussian posterior around MAP estimate
- **Variational Inference**: Bayes by Backprop
- **Uncertainty decomposition**: Separate epistemic & aleatoric uncertainty

### 🎯 **Active Learning** ⭐ NEW
- **Acquisition functions**: Entropy, BALD, margin, variance ratio
- **Query strategies**: Uncertainty sampling, diversity sampling, Core-Set, BADGE
- **Batch selection**: BatchBALD for efficient batch queries
- **Committee methods**: Query by Committee (QBC)

### 📈 **Distribution Shift Detection**
- **Statistical tests**: MMD, Energy distance, Kolmogorov-Smirnov
- **Classifier-based**: Black-Box Shift Detection (BBSD)
- **Label shift**: Detect and correct label distribution changes ⭐ NEW
- **Importance weighting**: Covariate shift adaptation ⭐ NEW

### 📦 **Data & Utilities**
- Built-in datasets (MNIST, CIFAR-10/100, SVHN)
- OOD benchmark datasets
- Visualization utilities
- Common architectures (ConvNet, ResNet)

## 🚀 Installation

### From PyPI (coming soon)
```bash
pip install incerto
```

### From source
```bash
git clone https://github.com/stephanrabanser/incerto.git
cd incerto
pip install -e .
```

### With development dependencies
```bash
git clone https://github.com/stephanrabanser/incerto.git
cd incerto
pip install -e ".[dev]"
```

## 📖 Quick Start

### Calibration

```python
import torch
from incerto.calibration import TemperatureScaling, ece_score

# Train your model
model = YourModel()
# ... training code ...

# Post-hoc calibration
calibrator = TemperatureScaling()
calibrator.fit(val_logits, val_labels)

# Get calibrated predictions
test_logits = model(test_data)
calibrated_probs = calibrator.predict(test_logits).probs

# Evaluate calibration
ece = ece_score(test_logits, test_labels)
print(f"Expected Calibration Error: {ece:.4f}")
```

### OOD Detection

```python
from incerto.ood import Energy

# Initialize detector with trained model
detector = Energy(model, temperature=1.0)

# Score test samples (higher = more OOD)
id_scores = detector.score(id_data)
ood_scores = detector.score(ood_data)

# Make predictions with threshold
predictions = detector.predict(test_data, threshold=0.5)
```

### Conformal Prediction

```python
from incerto.conformal import aps

# Create conformal predictor
predictor = aps(model, calibration_loader, alpha=0.1)  # 90% coverage

# Get prediction sets
prediction_sets = predictor(test_data)
# Each set contains classes that cover the true label with 90% probability
```

### LLM Uncertainty

```python
from incerto.llm import TokenEntropy, SequenceEntropy, SelfConsistency

# Token-level uncertainty
logits = llm(prompt)  # (batch, seq_len, vocab_size)
token_entropy = TokenEntropy.compute(logits)

# Sequence-level uncertainty
seq_entropy = SequenceEntropy.compute(logits, aggregation='mean')

# Sampling-based uncertainty
responses = [llm.generate(prompt) for _ in range(10)]
consistency = SelfConsistency.compute(responses)
print(f"Agreement rate: {consistency['agreement_rate']:.2f}")
```

## 📚 Examples

The `examples/` directory contains comprehensive tutorials:

### 01_synthetic - Basic Concepts
- `calibration_basics.py` - Calibration methods on synthetic data
- `ood_detection_basics.py` - OOD detection fundamentals
- `conformal_prediction.py` - Conformal prediction sets
- `selective_prediction.py` - Selective classification

### 02_tabular - Tabular Data
- `posthoc_calibration.py` - Post-hoc calibration methods
- `training_methods.py` - Training-time calibration

### 03_vision - Computer Vision
- `mnist/` - MNIST examples (post-hoc, training methods, ensembles)
- `cifar/` - CIFAR-10 examples (advanced training, OOD detection)

### 04_nlp - Natural Language Processing
- `llm_uncertainty.py` - LLM uncertainty quantification

## 🧪 Testing

**incerto** has comprehensive test coverage (190 tests, 100% passing):

```bash
# Run all tests
pytest

# Run specific module tests
pytest tests/test_calibration/
pytest tests/test_ood/
pytest tests/test_conformal/

# Run with coverage
pytest --cov=incerto --cov-report=term-missing
```

## 📊 Supported Methods

<details>
<summary><b>Calibration Methods</b></summary>

**Post-hoc:**
- Temperature Scaling
- Vector Scaling
- Matrix Scaling
- Platt Scaling
- Isotonic Regression
- Histogram Binning

**Training-time:**
- Label Smoothing
- Focal Loss
- Confidence Penalty
- Evidential Deep Learning
- Temperature-Aware Training

**Metrics:**
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Classwise ECE
- Brier Score
- Negative Log-Likelihood (NLL)
</details>

<details>
<summary><b>OOD Detection Methods</b></summary>

**Score-based:**
- Maximum Softmax Probability (MSP)
- MaxLogit
- Energy Score
- ODIN

**Distance-based:**
- Mahalanobis Distance
- K-Nearest Neighbors (KNN)

**Training-time:**
- Mixup
- CutMix
- Outlier Exposure
- Energy Regularization
</details>

<details>
<summary><b>Conformal Prediction Methods</b></summary>

**Classification:**
- Inductive Conformal Prediction (ICP)
- Adaptive Prediction Sets (APS)
- Regularized APS (RAPS)
- Mondrian Conformal Prediction

**Regression:**
- Jackknife+
- CV+
</details>

<details>
<summary><b>LLM Uncertainty Methods</b></summary>

**Token-level:**
- Token Entropy
- Token Confidence
- Perplexity
- Surprisal Score
- Top-K Confidence

**Sequence-level:**
- Sequence Probability
- Average Log-Probability
- Sequence Entropy

**Sampling-based:**
- Self-Consistency
- Semantic Entropy
- Predictive Entropy
- Mutual Information

**Generation:**
- Beam Search Uncertainty
- Nucleus Sampling Uncertainty
- I Don't Know Detection
- Contrastive Decoding
</details>

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use **incerto** in your research, please cite:

```bibtex
@software{incerto2025,
  author = {Rabanser, Stephan},
  title = {incerto: Uncertainty Quantification for Machine Learning},
  year = {2025},
  url = {https://github.com/stephanrabanser/incerto},
  version = {0.1.0}
}
```

## 🙏 Acknowledgments

This library implements methods from many research papers. Key references:

- **Calibration**: Guo et al. (2017), Platt (1999), Zadrozny & Elkan (2002)
- **OOD Detection**: Hendrycks & Gimpel (2017), Liu et al. (2020), Lee et al. (2018)
- **Conformal Prediction**: Vovk et al. (2005), Romano et al. (2020), Angelopoulos et al. (2021)
- **Selective Prediction**: Geifman & El-Yaniv (2019), Huang et al. (2020)
- **LLM Uncertainty**: Kuhn et al. (2023), Lin et al. (2023)

## 🔗 Links

- **Documentation**: (coming soon)
- **Paper**: (coming soon)
- **Issues**: [GitHub Issues](https://github.com/stephanrabanser/incerto/issues)
- **Discussions**: [GitHub Discussions](https://github.com/stephanrabanser/incerto/discussions)

---

**Status**: Active development | **Version**: 0.1.0 | **Python**: 3.10+
