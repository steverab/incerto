Active Learning Guide
====================

Active learning reduces labeling costs by **strategically selecting which samples to label**. Instead of random sampling, query the most informative examples.

Why Active Learning
-------------------

Labeling is expensive:
   - Medical image annotation requires expert radiologists
   - NLP tasks need careful human review
   - Robotics needs real-world interaction

Active learning can achieve same performance with **10-100x less labeled data**.

Core Idea
---------

1. Train model on small labeled set
2. **Query strategy**: Select most informative unlabeled samples
3. Get labels for selected samples (human annotation)
4. Add to training set, retrain
5. Repeat until budget exhausted or performance adequate

Acquisition Functions
---------------------

Acquisition functions score how informative each unlabeled sample is.

Entropy Acquisition
^^^^^^^^^^^^^^^^^^^

**Best for**: Starting point, simple and effective

Query samples where model is most uncertain:

.. code-block:: python

   from incerto.active import EntropyAcquisition, UncertaintySampling

   # Create acquisition function
   acquisition = EntropyAcquisition()

   # Use with uncertainty sampling strategy
   strategy = UncertaintySampling(
       acquisition_fn=acquisition,
       batch_size=100
   )

   # Query most uncertain samples
   query_indices = strategy.query(model, unlabeled_data)

   # Label these samples
   samples_to_label = unlabeled_data[query_indices]

Least Confidence
^^^^^^^^^^^^^^^^

Query samples where model is least confident in its prediction:

.. code-block:: python

   from incerto.active import LeastConfidenceAcquisition, UncertaintySampling

   acquisition = LeastConfidenceAcquisition()
   strategy = UncertaintySampling(acquisition, batch_size=100)

   query_indices = strategy.query(model, unlabeled_data)

Margin Sampling
^^^^^^^^^^^^^^^

Query samples with smallest margin between top-2 predictions:

.. code-block:: python

   from incerto.active import MarginAcquisition, UncertaintySampling

   acquisition = MarginAcquisition()
   strategy = UncertaintySampling(acquisition, batch_size=100)

   query_indices = strategy.query(model, unlabeled_data)

BALD (Bayesian Active Learning by Disagreement)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Best for**: When using Bayesian methods (MC Dropout, ensembles)

Query samples with highest mutual information:

.. code-block:: python

   from incerto.active import BALDAcquisition, UncertaintySampling

   # BALD uses multiple forward passes for MC Dropout
   acquisition = BALDAcquisition(num_samples=10)
   strategy = UncertaintySampling(acquisition, batch_size=100)

   # Model should have dropout enabled
   query_indices = strategy.query(model, unlabeled_data)

**Intuition**: Query where model weights disagree most (high epistemic uncertainty)

**Reference**: Houlsby et al., "Bayesian Active Learning for Classification" (AIStats 2011)

Query Strategies
----------------

Uncertainty Sampling
^^^^^^^^^^^^^^^^^^^^

Simple top-k selection based on acquisition scores:

.. code-block:: python

   from incerto.active import EntropyAcquisition, UncertaintySampling

   acquisition = EntropyAcquisition()
   strategy = UncertaintySampling(acquisition, batch_size=100)

   # Returns indices of top 100 most uncertain samples
   indices = strategy.query(model, unlabeled_data)

Diversity Sampling
^^^^^^^^^^^^^^^^^^

Balance uncertainty with diversity to avoid redundant samples:

.. code-block:: python

   from incerto.active import EntropyAcquisition, DiversitySampling

   acquisition = EntropyAcquisition()
   strategy = DiversitySampling(
       acquisition_fn=acquisition,
       batch_size=100,
       diversity_weight=0.5  # Balance uncertainty and diversity
   )

   indices = strategy.query(model, unlabeled_data)

CoreSet Selection
^^^^^^^^^^^^^^^^^

Select samples that best cover the feature space:

.. code-block:: python

   from incerto.active import CoreSetSelection

   strategy = CoreSetSelection(batch_size=100)

   # Requires features (can extract from model)
   indices = strategy.query(
       features_unlabeled,
       features_labeled
   )

BADGE Sampling
^^^^^^^^^^^^^^

Diverse gradients for batch selection:

.. code-block:: python

   from incerto.active import BadgeSampling

   strategy = BadgeSampling(batch_size=100)
   indices = strategy.query(model, unlabeled_data)

**Reference**: Ash et al., "Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds" (ICLR 2020)

Complete Active Learning Loop
------------------------------

.. code-block:: python

   import torch
   from incerto.active import EntropyAcquisition, UncertaintySampling

   # Manual loop example
   labeled_indices = initial_labeled_indices
   unlabeled_indices = torch.arange(len(dataset))
   unlabeled_indices = unlabeled_indices[~torch.isin(unlabeled_indices, labeled_indices)]

   acquisition = EntropyAcquisition()
   strategy = UncertaintySampling(acquisition, batch_size=100)

   for round in range(10):
       # Train model on labeled data
       model = train_model(dataset, labeled_indices)

       # Get unlabeled samples
       unlabeled_data = dataset[unlabeled_indices][0]  # features only

       # Query most informative samples
       query_local_indices = strategy.query(model, unlabeled_data)

       # Map back to global indices
       query_global_indices = unlabeled_indices[query_local_indices]

       # Update labeled/unlabeled sets
       labeled_indices = torch.cat([labeled_indices, query_global_indices])
       unlabeled_mask = ~torch.isin(unlabeled_indices, query_global_indices)
       unlabeled_indices = unlabeled_indices[unlabeled_mask]

       # Evaluate
       accuracy = evaluate(model, test_loader)
       print(f"Round {round+1}: {len(labeled_indices)} labeled, accuracy={accuracy:.2%}")

You can also use the utility function:

.. code-block:: python

   from incerto.active import active_learning_loop

   # Run active learning with custom training function
   results = active_learning_loop(
       model=model,
       train_loader=train_loader,
       unlabeled_loader=unlabeled_loader,
       acquisition=EntropyAcquisition(),
       train_fn=train_fn,
       n_rounds=10,
       samples_per_round=100,
   )

Practical Tips
--------------

**Batch mode**:
   Query multiple samples at once for efficiency

.. code-block:: python

   # Configure batch size in strategy
   strategy = UncertaintySampling(acquisition, batch_size=100)

**Diversity**:
   Use DiversitySampling to avoid querying similar samples

.. code-block:: python

   from incerto.active import DiversitySampling, EntropyAcquisition

   strategy = DiversitySampling(
       acquisition_fn=EntropyAcquisition(),
       batch_size=100,
       diversity_weight=0.5  # 0.5 = equal weight to uncertainty and diversity
   )

**Cold start**:
   Begin with random sampling or stratified sampling

.. code-block:: python

   # Initial random sample
   initial_size = 100
   initial_indices = torch.randperm(len(dataset))[:initial_size]

**Stopping criteria**:
   Stop when performance plateaus or budget exhausted

.. code-block:: python

   if accuracy > target_accuracy:
       print("Target accuracy reached!")
       break

   if len(labeled_indices) >= max_budget:
       print("Budget exhausted!")
       break

Evaluation
----------

**Learning curve**:
   Plot accuracy vs. number of labeled samples

.. code-block:: python

   import matplotlib.pyplot as plt

   plt.plot(n_labeled_samples, accuracies, label='Active')
   plt.plot(n_labeled_samples, random_accuracies, label='Random')
   plt.xlabel('Number of Labeled Samples')
   plt.ylabel('Test Accuracy')
   plt.legend()

**Area Under Learning Curve (AULC)**:
   Higher is better

**Reduction ratio**:
   How much data saved to reach target accuracy

Best Practices
--------------

1. **Start with uncertainty sampling**
      Simple, effective baseline

2. **Use batch queries**
      Query 50-100 samples at a time for efficiency

3. **Consider diversity**
      Prevent querying redundant samples

4. **Retrain frequently**
      Model needs to adapt to new labels

5. **Use Bayesian methods when possible**
      BALD often outperforms simple uncertainty

6. **Compare to random baseline**
      Always benchmark against random sampling

Common Pitfalls
---------------

- **Querying only hardest samples**: Can lead to noisy/outlier labels
- **Not using diversity**: Queries may be redundant
- **Infrequent retraining**: Model doesn't benefit from new labels
- **Wrong initial set**: Cold start matters - use stratified sampling

Advanced Topics
---------------

**Query by committee**:
   Use ensemble disagreement:

.. code-block:: python

   from incerto.active import QueryByCommittee

   # Committee of models
   committee = [model1, model2, model3]
   strategy = QueryByCommittee(committee, batch_size=100)

   indices = strategy.query(unlabeled_data, model=committee[0])

References
----------

1. Settles, "Active Learning Literature Survey" (2009)
2. Houlsby et al., "Bayesian Active Learning for Classification" (AIStats 2011)
3. Gal et al., "Deep Bayesian Active Learning" (ICML 2017)
4. Ash et al., "Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds" (ICLR 2020)

See Also
--------

- :doc:`../api/active` - Complete API reference
- :doc:`bayesian` - Bayesian uncertainty for BALD
- :doc:`selective` - Selective prediction
