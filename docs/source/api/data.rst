Data Utilities
==============

The data module provides dataset loaders, OOD benchmarks, and data utilities.

.. currentmodule:: incerto.data

Vision Datasets
---------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   VisionDataset
   MNIST
   FashionMNIST
   CIFAR10
   CIFAR100
   SVHN

OOD Benchmarks
--------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   OODBenchmark
   get_ood_benchmark
   MNIST_vs_FashionMNIST
   CIFAR10_vs_CIFAR100
   CIFAR10_vs_SVHN
   MNIST_vs_NotMNIST
   SubclassOOD
   CorruptedDataOOD

Data Loaders
------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   create_dataloaders
   create_balanced_dataloader
   create_ood_dataloader
   create_calibration_loaders
   InfiniteDataLoader
   get_dataloader_stats

Dataset Utilities
-----------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   split_dataset
   filter_dataset_by_class
   get_class_balanced_subset
   compute_dataset_statistics
   create_imbalanced_dataset
   TransformDataset
   LabelNoiseDataset
   merge_datasets
   subsample_dataset
