.. _minor-version-compatibility:

CUDA Minor Version Compatibility
================================

CUDA `Minor Version Compatibility
<https://docs.nvidia.com/deploy/cuda-compatibility/index.html#minor-version-compatibility>`_
(MVC) enables the use of a newer CUDA toolkit version than the CUDA version
supported by the driver, provided that the toolkit and driver both have the same
major version. For example, use of CUDA toolkit 11.5 with CUDA driver 450 (CUDA
version 11.0) is supported through MVC.

Numba supports MVC for CUDA 11 on Linux using the external ``cubinlinker`` and
``ptxcompiler`` packages, subject to the following limitations:

- Linking of archives is unsupported.
- Cooperative Groups are unsupported, because they require an archive to be
  linked.

MVC is not yet supported on CUDA 12 - this will be added in a future release of
Numba.

MVC is not supported on Windows.


Enabling MVC Support
--------------------

To use MVC support, the ``cubinlinker`` and ``ptxcompiler`` compiler packages
must be installed from the appropriate channels. To install using conda, use:

.. code:: bash

   conda install rapidsai::cubinlinker conda-forge::ptxcompiler

To install with pip, use the NVIDIA package index:

.. code:: bash

   pip install ptxcompiler-cu11 cubinlinker-cu11 --extra-index-url=https://pypi.nvidia.com

MVC support is enabled by setting the environment variable:

.. code:: bash

   export NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=1


or by setting a configuration variable prior to using any CUDA functionality in
Numba:

.. code:: python

   from numba import config
   config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY = True


References
----------

Further information about Minor Version Compatibility may be found in:

- The `CUDA Compatibility Guide
  <https://docs.nvidia.com/deploy/cuda-compatibility/index.html>`_.
- The `README for ptxcompiler
  <https://github.com/rapidsai/ptxcompiler/blob/main/README.md>`_.

