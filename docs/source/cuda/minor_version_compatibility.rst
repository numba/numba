.. _minor-version-compatibility:

Minor Version Compatiblity
==========================

`Minor Version Compatibility
<https://docs.nvidia.com/deploy/cuda-compatibility/index.html#minor-version-compatibility>`_
(MVC) enables the use of a newer CUDA toolkit version than the CUDA version
supported by the driver, provided that the toolkit and driver both have the same
major version. For example, use of CUDA toolkit 11.5 with CUDA driver 450 (CUDA
version 11.0) is supported through MVC.

Numba supports MVC for CUDA 11 on Linux through the use of external
``cubinlinker`` and ``ptxcompiler`` packages, subject to the following
limitations:

- Linking of archives is unsupported.
- Cooperative Groups are unsupported, because they require an archive to be
  linked.


Enabling MVC Support
--------------------

To use MVC support, the ``cubinlinker`` and ``ptxcompiler`` compiler packages
must be installed from the appropriate channels. To install using conda, use:

.. code:: bash

   conda install rapidsai::cubinlinker conda-forge::ptxcompiler

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

