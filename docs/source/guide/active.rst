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

Uncertainty Sampling
^^^^^^^^^^^^^^^^^^^^

**Best for**: Starting point, simple and effective

Query samples where model is most uncertain:

.. code-block:: python

   from incerto.active import entropy_acquisition, UncertaintySampling

   strategy = UncertaintySampling(
       model,
       acquisition_fn=entropy_acquisition
   )

   # Query most uncertain samples
   query_indices = strategy.query(
       unlabeled_pool,
       n_samples=100
   )

   # Label these samples
   labeled_samples = label_samples(unlabeled_pool[query_indices])

**Variants**:
   - Least confidence: Query samples with lowest max probability
   - Margin sampling: Query samples with smallest difference between top-2 classes
   - Entropy: Query samples with highest entropy

Entropy Acquisition
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from incerto.active import entropy_acquisition

   # Compute entropy for each sample
   logits = model(unlabeled_data)
   probs = F.softmax(logits, dim=-1)
   entropy_scores = entropy_acquisition(probs)

   # Higher entropy = more uncertain = higher priority
   top_k_indices = torch.argsort(entropy_scores, descending=True)[:k]

Least Confidence
^^^^^^^^^^^^^^^^

.. code-block:: python

   from incerto.active import least_confidence_acquisition

   logits = model(unlabeled_data)
   probs = F.softmax(logits, dim=-1)
   confidence_scores = least_confidence_acquisition(probs)

   # Lower confidence = higher priority
   top_k_indices = torch.argsort(confidence_scores)[:k]

Margin Sampling
^^^^^^^^^^^^^^^

.. code-block:: python

   from incerto.active import margin_acquisition

   logits = model(unlabeled_data)
   probs = F.softmax(logits, dim=-1)
   margin_scores = margin_acquisition(probs)

   # Smaller margin = more uncertain = higher priority
   top_k_indices = torch.argsort(margin_scores)[:k]

BALD (Bayesian Active Learning by Disagreement)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Best for**: When using Bayesian methods (MC Dropout, ensembles)

Query samples with highest mutual information:

.. code-block:: python

   from incerto.active import BALDAcquisition
   from incerto.bayesian import MCDropout

   # Use MC Dropout for Bayesian uncertainty
   mc_dropout = MCDropout(model, n_samples=10)

   strategy = BALDAcquisition(mc_dropout)
   query_indices = strategy.query(unlabeled_pool, n_samples=100)

**Intuition**: Query where model weights disagree most (high epistemic uncertainty)

**Advantages**:
   - Theoretically motivated
   - Considers model uncertainty
   - Often outperforms entropy

**Disadvantages**:
   - Requires Bayesian model
   - More computationally expensive

**Reference**: Houlsby et al., "Bayesian Active Learning for Classification" (AIStats 2011)

Complete Active Learning Loop
------------------------------

.. code-block:: python

   import torch
   from incerto.active import UncertaintySampling, entropy_acquisition

   # Initial setup
   labeled_data = initial_labeled_set  # Small labeled set
   unlabeled_pool = large_unlabeled_set
   budget = 1000  # Number of labels we can afford

   # Initial model
   model = train_model(labeled_data)

   # Active learning loop
   n_queries = budget // 100  # Query 100 samples at a time

   for round in range(n_queries):
       print(f"Round {round + 1}/{n_queries}")

       # 1. Select samples to label
       strategy = UncertaintySampling(
           model,
           acquisition_fn=entropy_acquisition
       )

       query_indices = strategy.query(
           unlabeled_pool,
           n_samples=100
       )

       # 2. Get labels (human annotation or oracle)
       query_samples = unlabeled_pool[query_indices]
       query_labels = get_labels(query_samples)  # Human labeling

       # 3. Add to labeled set
       labeled_data.add(query_samples, query_labels)

       # 4. Remove from unlabeled pool
       unlabeled_pool.remove(query_indices)

       # 5. Retrain model
       model = train_model(labeled_data)

       # 6. Evaluate
       accuracy = evaluate(model, test_set)
       print(f"Accuracy: {accuracy:.2%}")
       print(f"Labeled samples: {len(labeled_data)}")

Practical Tips
--------------

**Batch mode**:
   Query multiple samples at once for efficiency

.. code-block:: python

   # Query 100 samples in batch
   query_indices = strategy.query(unlabeled_pool, n_samples=100)

**Diversity**:
   Combine uncertainty with diversity to avoid querying similar samples

.. code-block:: python

   from incerto.active import diverse_batch_query

   # Select diverse AND uncertain samples
   query_indices = diverse_batch_query(
       model,
       unlabeled_pool,
       n_samples=100,
       diversity_weight=0.5
   )

**Cold start**:
   Begin with random sampling or stratified sampling

.. code-block:: python

   # Initial random sample
   initial_size = 100
   initial_indices = torch.randperm(len(dataset))[:initial_size]
   labeled_data = dataset[initial_indices]

**Stopping criteria**:
   Stop when performance plateaus or budget exhausted

.. code-block:: python

   if accuracy > target_accuracy:
       print("Target accuracy reached!")
       break

   if len(labeled_data) >= max_budget:
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

.. code-block:: python

   # E.g., active learning reaches 95% accuracy with 1000 samples
   # Random sampling needs 5000 samples for same accuracy
   # Reduction ratio = 5000 / 1000 = 5x

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

7. **Monitor labeling quality**
      Human labels may be noisy or biased

Common Pitfalls
---------------

❌ **Querying only hardest samples**
      Can lead to noisy/outlier labels

❌ **Not using diversity**
      Queries may be redundant

❌ **Infrequent retraining**
      Model doesn't benefit from new labels

❌ **Wrong initial set**
      Cold start matters - use stratified sampling

❌ **Ignoring label noise**
      Uncertain samples may have unreliable labels

Advanced Topics
---------------

**Query by committee**:
   Use ensemble disagreement instead of single model uncertainty

**Expected model change**:
   Query samples that change model most

**Expected error reduction**:
   Query samples that reduce expected error most

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
