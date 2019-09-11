========
Overview
========

Numba supports CUDA GPU programming by directly compiling a restricted subset
of Python code into CUDA kernels and device functions following the CUDA
execution model.  Kernels written in Numba appear to have direct access
to NumPy arrays.  NumPy arrays are transferred between the CPU and the
GPU automatically.


Terminology
===========

Several important terms in the topic of CUDA programming are listed here:

- *host*: the CPU
- *device*: the GPU
- *host memory*: the system main memory
- *device memory*: onboard memory on a GPU card
- *kernels*: a GPU function launched by the host and executed on the device
- *device function*: a GPU function executed on the device which can only be
  called from the device (i.e. from a kernel or another device function)


Programming model
=================

Most CUDA programming facilities exposed by Numba map directly to the CUDA
C language offered by NVidia.  Therefore, it is recommended you read the
official `CUDA C programming guide <http://docs.nvidia.com/cuda/cuda-c-programming-guide>`_.


Requirements
============

Supported GPUs
--------------

Numba supports CUDA-enabled GPU with compute capability 2.0 or above with an
up-to-data Nvidia driver.

Software
--------

You will need the CUDA toolkit installed.  If you are using Conda, just
type::

   $ conda install cudatoolkit

If you are not using Conda or if you want to use a different version of CUDA
toolkit, the following describe how Numba searches for a CUDA toolkit
installation.

.. _cudatoolkit-lookup:

Setting CUDA Installation Path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Numba searches for a CUDA toolkit installation in the following order:

1. Conda installed `cudatoolkit` package.
2. Environment variable ``CUDA_HOME``, which points to the directory of the
   installed CUDA toolkit (i.e. ``/home/user/cuda-10``)
3. System-wide installation at exactly ``/usr/local/cuda`` on Linux platforms.
   Versioned installation paths (i.e. ``/usr/local/cuda-10.0``) are intentionally
   ignored.  Users can use ``CUDA_HOME`` to select specific versions.

In addition to the CUDA toolkit libraries, which can be installed by conda into
an environment or installed system-wide by the `CUDA SDK installer
<(https://developer.nvidia.com/cuda-downloads)>`_, the CUDA target in Numba
also requires an up-to-date NVIDIA graphics driver.  Updated graphics drivers
are also installed by the CUDA SDK installer, so there is no need to do both.
Note that on macOS, the CUDA SDK must be installed to get the required driver,
and the driver is only supported on macOS prior to 10.14 (Mojave).  If the
``libcuda`` library is in a non-standard location, users can set environment
variable ``NUMBA_CUDA_DRIVER`` to the file path (not the directory path) of the
shared library file.


Missing CUDA Features
=====================

Numba does not implement all features of CUDA, yet.  Some missing features
are listed below:

* dynamic parallelism
* texture memory
