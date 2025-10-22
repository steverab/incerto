Out-of-Distribution Detection Guide
===================================

Out-of-Distribution (OOD) detection identifies when test inputs are from a different distribution than training data. This is critical for deploying safe and reliable ML systems.

Why OOD Detection Matters
--------------------------

Neural networks make confident predictions even on data they've never seen before:

- **Safety**: Autonomous vehicles must detect unusual scenarios
- **Reliability**: Medical AI should abstain on rare cases
- **Trust**: Users need to know when predictions are unreliable

Example:
   A digit classifier trained on MNIST confidently predicts "3" when shown a cat image.

Problem Formulation
-------------------

Given:
   - **In-distribution (ID)**: Training data distribution :math:`P_{in}`
   - **Out-of-distribution (OOD)**: Test data from :math:`P_{out} \neq P_{in}`

Goal:
   Detect when test sample :math:`x` comes from :math:`P_{out}` rather than :math:`P_{in}`

OOD Detection Methods
---------------------

Maximum Softmax Probability (MSP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Best for**: Baseline method, quick implementation

Uses the maximum softmax probability as confidence:

.. code-block:: python

   from incerto.ood import MSP

   detector = MSP(model)

   # Higher score = more OOD-like
   id_scores = detector.score(in_distribution_data)
   ood_scores = detector.score(out_of_distribution_data)

   # Typically: id_scores.mean() < ood_scores.mean()

**Intuition**: OOD inputs have lower max probability

**Advantages**:
   - No training or calibration needed
   - Very fast
   - Simple to interpret

**Disadvantages**:
   - Often unreliable (networks overconfident)
   - No better than random in many cases

**Reference**: Hendrycks & Gimpel (ICLR 2017)

Energy Score
^^^^^^^^^^^^

**Best for**: Most scenarios, good default choice

Uses the energy function (log-sum-exp of logits):

.. math::

   E(x) = -T \cdot \log \sum_i \exp(f_i(x) / T)

.. code-block:: python

   from incerto.ood import Energy

   # Temperature controls sensitivity
   detector = Energy(model, temperature=1.0)

   scores = detector.score(test_data)
   # Lower energy = more ID-like
   # Higher energy = more OOD-like

   # Save detector configuration
   detector.save('energy_detector.pt')

**Intuition**: OOD samples have higher energy (less confident)

**Advantages**:
   - Significantly better than MSP
   - Single hyperparameter (temperature)
   - Theoretically motivated

**Disadvantages**:
   - Needs temperature tuning for best results

**Reference**: Liu et al., "Energy-based OOD Detection" (NeurIPS 2020)

MaxLogit
^^^^^^^^

**Best for**: When you want simplicity without softmax

Uses the maximum logit value directly:

.. code-block:: python

   from incerto.ood import MaxLogit

   detector = MaxLogit(model)
   scores = detector.score(test_data)

   # Lower maxlogit = more OOD-like

**Intuition**: ID samples have higher maximum logits

**Advantages**:
   - Simpler than MSP (no softmax)
   - Often more effective than MSP
   - Fast

**Disadvantages**:
   - Still uses uncalibrated model outputs

**Reference**: Hendrycks et al. (2019)

ODIN
^^^^

**Best for**: When you can afford preprocessing overhead

Uses input preprocessing and temperature scaling:

.. code-block:: python

   from incerto.ood import ODIN

   detector = ODIN(
       model,
       temperature=1000.0,  # Higher = more separation
       epsilon=0.0014      # Input perturbation magnitude
   )

   scores = detector.score(test_data)

**How it works**:
   1. Apply temperature scaling to logits
   2. Add small adversarial perturbation to input
   3. Use maximum softmax probability

**Advantages**:
   - Better separation than MSP
   - Interpretable hyperparameters

**Disadvantages**:
   - Requires backpropagation through model
   - Slower than simple methods
   - Needs hyperparameter tuning

**Reference**: Liang et al., "Enhancing Reliability" (ICLR 2018)

Mahalanobis Distance
^^^^^^^^^^^^^^^^^^^^

**Best for**: When you have labeled ID data for calibration

Uses Mahalanobis distance in feature space:

.. code-block:: python

   from incerto.ood import Mahalanobis

   # Model must have accessible intermediate layer
   detector = Mahalanobis(model, layer_name='penultimate')

   # Fit on ID training data
   detector.fit(train_loader)

   # Detect OOD
   scores = detector.score(test_data)
   # Lower score = more ID-like

   print(repr(detector))
   # Mahalanobis(layer='penultimate', n_classes=10)

**How it works**:
   1. Compute class-conditional Gaussian distributions in feature space
   2. Measure distance to nearest class center

**Advantages**:
   - Uses learned feature representations
   - Theoretically well-founded
   - Often state-of-the-art performance

**Disadvantages**:
   - Requires fitting on ID data
   - Needs model with extractable features
   - Higher memory cost (stores class statistics)

**Reference**: Lee et al., "A Simple Unified Framework" (NeurIPS 2018)

KNN (k-Nearest Neighbors)
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Best for**: Non-parametric detection, when distributional assumptions don't hold

Uses distance to k-th nearest neighbor in feature space:

.. code-block:: python

   from incerto.ood import KNN

   detector = KNN(model, k=50, layer_name='penultimate')

   # Store training features
   detector.fit(train_loader)

   # Compute OOD scores
   scores = detector.score(test_data)
   # Larger distance = more OOD-like

   # Save fitted detector
   detector.save('knn_detector.pt')

**Intuition**: OOD samples are far from training examples

**Advantages**:
   - Non-parametric (no distributional assumptions)
   - Often competitive with more complex methods
   - Intuitive

**Disadvantages**:
   - Stores all training features (memory intensive)
   - Slow for large datasets (can be mitigated with approximate NN)
   - Sensitive to k choice

**Reference**: Sun et al., "Out-of-Distribution Detection with Deep Nearest Neighbors" (ICML 2022)

See Also
--------

- :doc:`../api/ood` - Complete API reference
- :doc:`calibration` - Calibration improves OOD detection
- :doc:`selective` - Selective prediction with abstention
- :doc:`shift` - Distribution shift detection
