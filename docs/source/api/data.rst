Data Utilities
==============

The data module provides dataset loaders, OOD benchmarks, and data utilities.

.. currentmodule:: incerto.data

Dataset Loaders
---------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   get_mnist
   get_fashion_mnist
   get_cifar10
   get_cifar100
   get_svhn

OOD Benchmarks
--------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   get_ood_benchmark
   OODBenchmark

Data Loaders
------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   create_calibration_split
   balanced_dataloader
