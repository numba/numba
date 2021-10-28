*****
Numba
*****

.. image:: https://badges.gitter.im/numba/numba.svg
   :target: https://gitter.im/numba/numba?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
   :alt: Gitter

.. image:: https://img.shields.io/badge/discuss-on%20discourse-blue
   :target: https://numba.discourse.group/
   :alt: Discourse

.. image:: https://zenodo.org/badge/3659275.svg
   :target: https://zenodo.org/badge/latestdoi/3659275
   :alt: Zenodo DOI

A Just-In-Time Compiler for Numerical Functions in Python
#########################################################

Numba is an open source, NumPy-aware optimizing compiler for Python sponsored
by Anaconda, Inc.  It uses the LLVM compiler project to generate machine code
from Python syntax.

Numba can compile a large subset of numerically-focused Python, including many
NumPy functions.  Additionally, Numba has support for automatic
parallelization of loops, generation of GPU-accelerated code, and creation of
ufuncs and C callbacks.

For more information about Numba, see the Numba homepage:
https://numba.pydata.org

Supported Platforms
===================

* Operating systems and CPUs:

  - Linux: x86 (32-bit), x86_64, ppc64le (POWER8 and 9), ARMv7 (32-bit),
    ARMv8 (64-bit).
  - Windows: x86, x86_64.
  - macOS: x86_64, (M1/Arm64, unofficial support only).
  - \*BSD: (unofficial support only).

* (Optional) Accelerators and GPUs:

  * NVIDIA GPUs (Kepler architecture or later) via CUDA driver on Linux,
    Windows, macOS (< 10.14).

Dependencies
============

* Python versions: 3.7-3.9
* llvmlite 0.38.*
* NumPy >=1.17 (can build with 1.11 for ABI compatibility).

Optionally:

* SciPy >=1.0.0 (for ``numpy.linalg`` support).


Installing
==========

The easiest way to install Numba and get updates is by using the Anaconda
Distribution: https://www.anaconda.com/download

::

   $ conda install numba

For more options, see the Installation Guide:
https://numba.readthedocs.io/en/stable/user/installing.html

Documentation
=============

https://numba.readthedocs.io/en/stable/index.html


Contact
=======

Numba has a discourse forum for discussions:

* https://numba.discourse.group



Continuous Integration
======================

.. image:: https://dev.azure.com/numba/numba/_apis/build/status/numba.numba?branchName=master
    :target: https://dev.azure.com/numba/numba/_build/latest?definitionId=1?branchName=master
    :alt: Azure Pipelines
