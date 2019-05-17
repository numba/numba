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
toolkit, the following describe how Numba search for a CUDA toolkit
installation.

.. _cudatoolkit-lookup:

Setting CUDA Installation Path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Numba search for CUDA toolkit installation in the following order:

1. Old and deprecated environment variables: ``NUMBAPRO_NVVM``,
   ``NUMBAPRO_LIBDEVICE``, and ``NUMBAPRO_CUDALIB``.
2. Conda installed `cudatoolkit` package.
3. Environment variable ``CUDA_HOME``, which points to the directory of the
   installed CUDA toolkit (i.e. ``/home/user/cuda-10``)
4. System-wide installation at exactly ``/usr/local/cuda`` on Linux platforms.
   Versioned installation paths (i.e. ``/usr/loca/cuda-10.0``) are intentionally
   ignored.  Users can use ``CUDA_HOME`` to select specific versions.

An up-to-date CUDA driver is required. It is typically installed by the CUDA
toolkit installer.  The CUDA driver is usually located in system default
location and Numba can find it automatically.  If the driver
is in a different location,  users can set environment variable
``NUMBA_CUDA_DRIVER`` to the file path (not the directory path) of the
driver shared library file.


Missing CUDA Features
=====================

Numba does not implement all features of CUDA, yet.  Some missing features
are listed below:

* dynamic parallelism
* texture memory
