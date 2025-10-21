Installation
============

Requirements
------------

- Python 3.10 or higher
- PyTorch 2.0 or higher
- NumPy
- scikit-learn
- scipy

Install from PyPI
-----------------

.. note::
   PyPI release coming soon!

Once released, you can install incerto using pip:

.. code-block:: bash

   pip install incerto

Install from Source
-------------------

For development or to get the latest features:

.. code-block:: bash

   git clone https://github.com/steverab/incerto.git
   cd incerto
   pip install -e .

Development Installation
------------------------

To install with development dependencies (for contributing):

.. code-block:: bash

   git clone https://github.com/steverab/incerto.git
   cd incerto
   pip install -e ".[dev]"

Or if using uv:

.. code-block:: bash

   git clone https://github.com/steverab/incerto.git
   cd incerto
   uv pip install -e .

Verify Installation
-------------------

To verify that incerto is installed correctly:

.. code-block:: python

   import incerto
   print(incerto.__version__)

   # Test imports
   from incerto import calibration, ood, conformal, sp, llm, bayesian, active, shift
   print("All modules imported successfully!")

GPU Support
-----------

incerto automatically uses CUDA if PyTorch is installed with GPU support:

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")

Optional Dependencies
---------------------

Some features may require additional packages:

- **transformers**: For LLM uncertainty methods
- **matplotlib**: For visualization utilities
- **rich**: For pretty output formatting

These are included in the base installation.

Troubleshooting
---------------

**ImportError: No module named 'incerto'**
   Make sure you installed the package and are not in the source directory

**PyTorch version conflicts**
   Ensure PyTorch 2.0+ is installed: ``pip install --upgrade torch``

**CUDA errors**
   Check PyTorch CUDA compatibility: ``torch.cuda.is_available()``
