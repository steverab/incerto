Bayesian Deep Learning Guide
============================

Bayesian deep learning quantifies **epistemic uncertainty** - uncertainty due to limited training data. This is crucial for knowing when a model is uncertain due to lack of knowledge vs. inherent randomness.

Types of Uncertainty
--------------------

**Aleatoric (data) uncertainty**:
   Irreducible noise in the data
   Example: Image is blurry, label is inherently ambiguous

**Epistemic (model) uncertainty**:
   Reducible - decreases with more training data
   Example: Model hasn't seen examples of this class

Bayesian approach: Maintain distribution over model weights instead of point estimate.

Methods
-------

MC Dropout
^^^^^^^^^^

**Best for**: Easy retrofitting, practical epistemic uncertainty

Approximate Bayesian inference by using dropout at test time:

.. code-block:: python

   from incerto.bayesian import MCDropout

   # Use dropout during inference
   mc_dropout = MCDropout(model, n_samples=10)

   # Get predictions with uncertainty
   result = mc_dropout.predict(test_data)

   mean_pred = result['mean']           # Average prediction
   epistemic = result['epistemic']      # Model uncertainty
   aleatoric = result['aleatoric']      # Data uncertainty
   total_unc = result['total']          # Total uncertainty

   # Entropy and mutual information
   entropy = result['predictive_entropy']
   mutual_info = result['mutual_information']

   print(f"Epistemic uncertainty: {epistemic.mean():.4f}")
   print(f"Model is uncertain where it hasn't seen data")

**How it works**:
   1. Enable dropout during inference
   2. Run multiple forward passes (Monte Carlo sampling)
   3. Aggregate predictions to estimate uncertainty

**Advantages**:
   - Works with any model with dropout
   - No retraining needed
   - Fast and practical

**Disadvantages**:
   - Approximate (not true Bayesian posterior)
   - Requires dropout in model
   - Quality depends on n_samples

**Reference**: Gal & Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016)

Deep Ensembles
^^^^^^^^^^^^^^

**Best for**: Best empirical performance, production systems

Train multiple models with different initializations:

.. code-block:: python

   from incerto.bayesian import DeepEnsemble

   # Train multiple models
   models = [train_model(seed=i) for i in range(5)]

   ensemble = DeepEnsemble(models)

   # Get predictions
   result = ensemble.predict(test_data, return_all=True)

   mean_pred = result['mean']
   epistemic = result['epistemic']

   # Measure diversity
   diversity = ensemble.diversity(test_data)
   print(f"Ensemble diversity: {diversity:.4f}")

**Advantages**:
   - State-of-the-art uncertainty estimates
   - Simple to implement
   - Reliable

**Disadvantages**:
   - 5-10x training cost
   - 5-10x inference cost
   - Significant memory overhead

**Reference**: Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty" (NeurIPS 2017)

SWAG (Stochastic Weight Averaging Gaussian)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Best for**: Efficient approximation, low overhead

Approximates posterior using weight statistics during training:

.. code-block:: python

   from incerto.bayesian import SWAG

   swag = SWAG(model, num_classes=10)

   # Collect model snapshots during training
   for epoch in range(epochs):
       for batch in train_loader:
           # Train normally
           loss = train_step(model, batch)

       # Collect model statistics
       swag.collect_model(model)

   # Sample from approximate posterior
   result = swag.predict(test_data, n_samples=10)

   epistemic = result['epistemic']

**Advantages**:
   - Low overhead (one training run)
   - Good uncertainty estimates
   - Efficient inference

**Disadvantages**:
   - Requires specific training procedure
   - Less accurate than deep ensembles
   - Needs careful tuning

**Reference**: Maddox et al., "A Simple Baseline for Bayesian Uncertainty" (NeurIPS 2019)

Laplace Approximation
^^^^^^^^^^^^^^^^^^^^^

**Best for**: Post-hoc uncertainty without retraining

Gaussian approximation around MAP estimate:

.. code-block:: python

   from incerto.bayesian import LaplaceApproximation

   # Train model normally
   model = train_model(train_loader)

   # Fit Laplace approximation
   laplace = LaplaceApproximation(model, num_classes=10)
   laplace.fit(train_loader)

   # Get predictions with uncertainty
   result = laplace.predict(test_data, n_samples=10)

**Advantages**:
   - Works with pre-trained models
   - Theoretically motivated
   - Efficient

**Disadvantages**:
   - Requires Hessian computation (expensive)
   - Gaussian assumption may be poor
   - Less accurate than MC Dropout or ensembles

Variational Inference
^^^^^^^^^^^^^^^^^^^^^

**Best for**: True Bayesian approach, research

Learn distribution over weights via variational inference:

.. code-block:: python

   from incerto.bayesian import VariationalBayesNN

   # Create Bayesian neural network
   model = VariationalBayesNN(
       input_dim=784,
       hidden_dim=256,
       output_dim=10
   )

   # Training with VI loss
   for inputs, labels in train_loader:
       # Forward pass samples from weight distribution
       outputs = model(inputs)

       # VI loss = likelihood + KL divergence
       kl_div = model.kl_divergence()
       nll = F.cross_entropy(outputs, labels)
       loss = nll + kl_div / len(train_loader)

       loss.backward()
       optimizer.step()

   # Inference
   result = model.predict(test_data, n_samples=10)

**Advantages**:
   - Principled Bayesian approach
   - Learns weight distributions explicitly

**Disadvantages**:
   - Requires model redesign
   - Computationally expensive
   - Difficult to tune

Complete Workflow
-----------------

.. code-block:: python

   import torch
   from incerto.bayesian import MCDropout

   # 1. Train model with dropout
   model = create_model_with_dropout(p=0.1)
   train_model(model, train_loader)

   # 2. Create MC Dropout predictor
   mc_dropout = MCDropout(model, n_samples=20)

   # 3. Get predictions with uncertainty
   all_epistemic = []
   all_correct = []

   for inputs, labels in test_loader:
       result = mc_dropout.predict(inputs)

       predictions = result['mean'].argmax(dim=-1)
       epistemic = result['epistemic']

       correct = (predictions == labels).float()

       all_epistemic.append(epistemic)
       all_correct.append(correct)

   epistemic = torch.cat(all_epistemic)
   correct = torch.cat(all_correct)

   # 4. Analyze uncertainty vs. correctness
   # High epistemic → likely incorrect
   import matplotlib.pyplot as plt

   plt.scatter(epistemic[correct==1], label='Correct')
   plt.scatter(epistemic[correct==0], label='Incorrect')
   plt.xlabel('Epistemic Uncertainty')
   plt.legend()

   # 5. Use for selective prediction
   threshold = epistemic.quantile(0.8)  # Abstain on top 20% uncertain
   predictions[epistemic > threshold] = -1  # Abstain

Evaluation
----------

**Negative Log-Likelihood (NLL)**:
   Measures both accuracy and uncertainty calibration

.. code-block:: python

   from incerto.bayesian.metrics import negative_log_likelihood

   nll = negative_log_likelihood(predictions, labels)
   # Lower is better

**Brier Score**:
   Proper scoring rule for probabilistic predictions

.. code-block:: python

   from incerto.calibration import brier_score

   bs = brier_score(predictions, labels)

**Expected Calibration Error (ECE)**:
   Check if uncertainties are calibrated

.. code-block:: python

   from incerto.calibration import ece_score

   ece = ece_score(predictions, labels)

Best Practices
--------------

1. **Start with MC Dropout**
      Easiest to implement, works with existing models

2. **Use enough samples**
      MC Dropout: 10-20 samples minimum
      SWAG: 20-30 samples

3. **Combine with calibration**
      Bayesian uncertainties can still be miscalibrated

4. **Monitor epistemic uncertainty**
      High on out-of-distribution data

5. **Use for active learning**
      Query samples with high epistemic uncertainty

6. **Validate uncertainty quality**
      Plot uncertainty vs. correctness

Comparison
----------

.. list-table::
   :header-rows: 1

   * - Method
     - Training Cost
     - Inference Cost
     - Quality
   * - MC Dropout
     - 1x
     - 10-20x
     - Good
   * - Deep Ensembles
     - 5-10x
     - 5-10x
     - Excellent
   * - SWAG
     - ~1.2x
     - 10-30x
     - Good
   * - Laplace
     - 1x + Hessian
     - 10x
     - Moderate
   * - Variational
     - 1-2x
     - 10x
     - Good (theory)

References
----------

1. Gal & Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016)
2. Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty" (NeurIPS 2017)
3. Maddox et al., "A Simple Baseline for Bayesian Uncertainty Estimation" (NeurIPS 2019)
4. Wilson & Izmailov, "Bayesian Deep Learning and a Probabilistic Perspective" (NeurIPS 2020)

See Also
--------

- :doc:`../api/bayesian` - Complete API reference
- :doc:`active` - Use epistemic uncertainty for active learning
- :doc:`selective` - Use uncertainty for selective prediction
