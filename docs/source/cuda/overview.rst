========
Overview
========

.. cuda-deprecated::

.. _cuda-deprecation-status:

Built-in CUDA target deprecation and maintenance status
=======================================================

Numba's CUDA target is now maintained in a separate package, `numba-cuda
<https://nvidia.github.io/numba-cuda>`_. This enables improvements in the
development of the CUDA target:

* The time and effort required to incorporate new features and bug fixes into
  the CUDA target is decreased - its development cycle is now decoupled from the
  Numba development process, so "upstream" reviews from Numba maintainers and
  test runs on the Anaconda internal CI systems are no longer required as part
  of the standard process for CUDA target pull requests. This lightening of
  process enables development to proceed at an increased cadence.
* Similarly, releases of the CUDA target can be made independently of Numba
  releases, at a more frequent pace.
* Numba is sufficiently mature as a compiler platform to support out-of-tree
  targets. The CUDA target, whilst maintained upstream, had been migrated to
  use the externally-facing APIs for target implementation. The continued
  development of the CUDA target outside of the main Numba repository ensures
  the continued development and robustness of these target extension APIs.

With development proceeding outside of the main Numba package, the built-in CUDA
target is now deprecated, but is still supported in Numba 0.61. It will continue
to be provided by Numba through at least version 0.62, but no new functionality
is expected to be added to it. New functionality and bug fixes will be
implemented in the ``numba-cuda`` package.

Users are encouraged to install ``numba-cuda`` in addition to Numba when using
the CUDA target. No code changes are required - the ``numba-cuda`` package will
continue to implement functionality under the ``numba.cuda`` namespace. 

To install ``numba-cuda`` with ``pip``::

   pip install numba-cuda

To install ``numba-cuda`` with ``conda``, for example from the ``conda-forge``
channel::

   conda install conda-forge::numba-cuda

For further information, see the :ref:`deprecation notice and schedule
<cuda-builtin-target-deprecation-notice>`.

Introduction
============

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

Numba supports CUDA-enabled GPUs with Compute Capability 3.5 or greater.
Support for devices with Compute Capability less than 5.0 is deprecated, and
will be removed in a future Numba release.

Devices with Compute Capability 5.0 or greater include (but are not limited to):

- Embedded platforms: NVIDIA Jetson Nano, Jetson Orin Nano, TX1, TX2, Xavier
  NX, AGX Xavier, AGX Orin.
- Desktop / Server GPUs: All GPUs with Maxwell microarchitecture or later. E.g.
  GTX 9 / 10 / 16 series, RTX 20 / 30 / 40 series, Quadro / Tesla M / P / V /
  RTX series, RTX A series, RTX Ada / SFF, A / L series, H100.
- Laptop GPUs: All GPUs with Maxwell microarchitecture or later. E.g. MX series,
  Quadro M / P / T series (mobile), RTX 20 / 30 series (mobile), RTX A series
  (mobile).

Software
--------

Numba aims to support CUDA Toolkit versions released within the last 3 years.
Presently 11.2 is the minimum required toolkit version. An NVIDIA driver
sufficient for the toolkit version is also required (see also
:ref:`minor-version-compatibility`).

Conda users can install the CUDA Toolkit into a conda environment.

For CUDA 12, ``cuda-nvcc`` and ``cuda-nvrtc`` are required::

    $ conda install -c conda-forge cuda-nvcc cuda-nvrtc "cuda-version>=12.0"

For CUDA 11, ``cudatoolkit`` is required::

    $ conda install -c conda-forge cudatoolkit "cuda-version>=11.2,<12.0"

If you are not using Conda or if you want to use a different version of CUDA
toolkit, the following describes how Numba searches for a CUDA toolkit
installation.

.. _cuda-bindings:

CUDA Bindings
~~~~~~~~~~~~~

Numba supports interacting with the CUDA Driver API via the `NVIDIA CUDA Python
bindings <https://nvidia.github.io/cuda-python/>`_ and its own ctypes-based
bindings. Functionality is equivalent between the two bindings. The
ctypes-based bindings are presently the default, but the NVIDIA bindings will
be used by default (if they are available in the environment) in a future Numba
release.

You can install the NVIDIA bindings with::

   $ conda install -c conda-forge cuda-python

if you are using Conda, or::

   $ pip install cuda-python

if you are using pip.

The use of the NVIDIA bindings is enabled by setting the environment variable
:envvar:`NUMBA_CUDA_USE_NVIDIA_BINDING` to ``"1"``.

.. _cudatoolkit-lookup:

Setting CUDA Installation Path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Numba searches for a CUDA toolkit installation in the following order:

1. Conda installed CUDA Toolkit packages
2. Environment variable ``CUDA_HOME``, which points to the directory of the
   installed CUDA toolkit (i.e. ``/home/user/cuda-12``)
3. System-wide installation at exactly ``/usr/local/cuda`` on Linux platforms.
   Versioned installation paths (i.e. ``/usr/local/cuda-12.0``) are intentionally
   ignored.  Users can use ``CUDA_HOME`` to select specific versions.

In addition to the CUDA toolkit libraries, which can be installed by conda into
an environment or installed system-wide by the `CUDA SDK installer
<https://developer.nvidia.com/cuda-downloads>`_, the CUDA target in Numba
also requires an up-to-date NVIDIA graphics driver.  Updated graphics drivers
are also installed by the CUDA SDK installer, so there is no need to do both.
If the ``libcuda`` library is in a non-standard location, users can set
environment variable ``NUMBA_CUDA_DRIVER`` to the file path (not the directory
path) of the shared library file.


Missing CUDA Features
=====================

Numba does not implement all features of CUDA, yet.  Some missing features
are listed below:

* dynamic parallelism
* texture memory
