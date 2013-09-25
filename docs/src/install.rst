Installation
============

NumbaPro is part of the `Anaconda Accelerate` product.  Please refer to the
`Anaconda Accelerate Documentation`_
for instructions on downloading and installation.

Purchasing a License
---------------------
Licenses can be purchased through the `online store`_.  Your
purchased license will be emailed to you with further instructions.

Update Instructions
-------------------
With Anaconda Accelerate already installed, first update
the conda_ package management tool to the latest version, then use conda
to update the NumbaPro module:

.. code-block:: bash

    $ conda update conda
    $ conda update numbapro
    
    
CUDA GPUs Setup
---------------

CUDA Driver
~~~~~~~~~~~~

NumbaPro does not ship the CUDA driver.  It is users responsibility to ensure
their systems are using the latest driver.
Currently, users should use the driver shipped with `CUDA 5.5 SDK`_.

CUDA Support & Detection
~~~~~~~~~~~~~~~~~~~~~~~~

NumbaPro currently supports `NVIDIA CUDA GPUs`_ with compute-capability 2+ only.
Users should check their hardware with the following:

.. code-block:: python
    
    import numbapro
    numbapro.check_cuda()

.. doctest::
    :hide:
    
    >>> import numbapro
    >>> callable(numbapro.check_cuda)
    True

A sample output looks like::

    ------------------------------libraries detection-------------------------------
    Finding cublas
    	located at /Users/.../lib/libcublas.dylib
    Finding cusparse
    	located at /Users/.../lib/libcusparse.dylib
    Finding cufft
    	located at /Users/.../lib/libcufft.dylib
    Finding curand
    	located at /Users/.../lib/libcurand.dylib
    Finding nvvm
    	located at /Users/.../lib/libnvvm.dylib
    In /Users/.../lib
    	finding libdevice.compute_20.bc 	ok
    	finding libdevice.compute_30.bc 	ok
    	finding libdevice.compute_35.bc 	ok
    -------------------------------hardware detection-------------------------------
    Found 1 CUDA devices
    id 0         GeForce GT 650M                              [SUPPORTED]
                          compute capability: 3.0
                               pci device id: 0
                                  pci bus id: 1
    Summary:
    	1/1 devices are supported

This performs CUDA library and GPU detection.
Discovered GPUs are listed with information for compute capability and whether
it is supported by NumbaPro.

.. _`Anaconda Accelerate Documentation`: http://docs.continuum.io/accelerate/index.html

.. _`online store`: https://store.continuum.io/cshop/accelerate

.. _conda: http://docs.continuum.io/conda/index.html

.. _`NVIDIA CUDA GPUs`: https://developer.nvidia.com/cuda-gpus

.. _`CUDA 5.5 SDK`: https://developer.nvidia.com/cuda-toolkit