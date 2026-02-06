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
   mc_dropout = MCDropout(model, num_samples=10)

   # Get predictions with uncertainty
   mean_pred, variance = mc_dropout.predict(test_data)

   # Variance captures epistemic (model) uncertainty
   print(f"Epistemic uncertainty: {variance.mean():.4f}")

   # Entropy and mutual information
   entropy = mc_dropout.predict_entropy(test_data)
   mutual_info = mc_dropout.predict_mutual_information(test_data)

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

   # Create ensemble with a model factory function
   def create_model():
       return MyModel()

   ensemble = DeepEnsemble(create_model, num_models=5)

   # Train each model separately
   for i, model in enumerate(ensemble.models):
       train_model(model, seed=i)

   # Get predictions with uncertainty
   mean_pred, variance = ensemble.predict(test_data)

   # Or get all individual predictions
   mean_pred, variance, all_preds = ensemble.predict(test_data, return_samples=True)

   # Measure diversity
   diversity = ensemble.diversity(test_data)
   print(f"Ensemble diversity: {diversity.mean():.4f}")

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

   swag = SWAG(model, num_samples=20)

   # Collect model snapshots during training (after warmup)
   for epoch in range(epochs):
       for batch in train_loader:
           # Train normally
           loss = train_step(model, batch)

       # Collect model statistics (typically after learning rate schedule)
       if epoch >= warmup_epochs:
           swag.collect_model(model)

   # Sample from approximate posterior
   mean_pred, variance = swag.predict(test_data)

   # Variance captures epistemic uncertainty
   print(f"Epistemic uncertainty: {variance.mean():.4f}")

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
   laplace = LaplaceApproximation(
       model,
       likelihood='classification',
       num_samples=20
   )
   laplace.fit(train_loader, device='cuda')

   # Get predictions with uncertainty
   mean_pred, variance = laplace.predict(test_data)

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
       in_features=784,
       hidden_sizes=[256, 128],
       out_features=10,
       num_samples=20
   )

   # Training with variational loss (combines NLL + KL)
   optimizer = torch.optim.Adam(model.parameters())
   for inputs, labels in train_loader:
       optimizer.zero_grad()
       loss = model.variational_loss(inputs, labels, num_samples=5)
       loss.backward()
       optimizer.step()

   # Inference with uncertainty
   mean_pred, variance = model.predict(test_data)

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
   mc_dropout = MCDropout(model, num_samples=20)

   # 3. Get predictions with uncertainty
   all_variance = []
   all_correct = []

   for inputs, labels in test_loader:
       mean_pred, variance = mc_dropout.predict(inputs)

       predictions = mean_pred.argmax(dim=-1)
       # Average variance across classes as uncertainty measure
       uncertainty = variance.mean(dim=-1)

       correct = (predictions == labels).float()

       all_variance.append(uncertainty)
       all_correct.append(correct)

   uncertainty = torch.cat(all_variance)
   correct = torch.cat(all_correct)

   # 4. Analyze uncertainty vs. correctness
   # High uncertainty → likely incorrect
   import matplotlib.pyplot as plt

   plt.hist(uncertainty[correct==1].numpy(), alpha=0.5, label='Correct')
   plt.hist(uncertainty[correct==0].numpy(), alpha=0.5, label='Incorrect')
   plt.xlabel('Uncertainty (Variance)')
   plt.legend()

   # 5. Use for selective prediction
   threshold = uncertainty.quantile(0.8)  # Abstain on top 20% uncertain
   # Samples with uncertainty > threshold should be reviewed by human

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
