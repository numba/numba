****************
Python to SPIR-V
****************

An OpenCL / SPIR-V backend for Numba
####################################

This repository is a Proof of Concept on how to translate numpy-based Python to SPIR-V code.
It uses Numba to translate Python to LLVM and SPIRV-LLVM to generate the final SPIR-V.
Although most Numba tests are working, this is a prototype, only meant for experimentation.

**TODO** / requirements for future development:
 - An official LLVM backend targeting SPIR-V is critical requirement of this project
 - A clean-up / refactorization of the Numba .cuda, .hsa and .ocl backends is needed

This repository was originally forked from **Numba 0.33** and tested with **llvmlite 0.18**. The llvmlite lib is an augmented Python wrapper for **LLVM 4.0** that Numba uses internally to handle the LLVM IR code.

**Dependencies** / installation requirements:
 - llvmlite 0.18. Install with conda, pip or build manually (requires LLVM 4.0)
 - Intel OpenCL 2.1 and a recent Intel CPU supporting it (e.g. Haswell, Skylake)
 - LLVM 4.0 to SPIR-V translator. Use https://github.com/thewilsonator/llvm-target-spirv
 - LLVM 4.0 with which to build the translator. Use https://github.com/thewilsonator/llvm
 - SPIRV-Tools https://github.com/KhronosGroup/SPIRV-Tools, i.e. spirv-as/dis/opt/val

First of all, clone this repository and install Numba following http://numba.pydata.org/numba-doc/dev/user/installing.html#installing-from-source and http://numba.pydata.org/numba-doc/dev/developer/contributing.html#building-numba. Next install llvmlite. If building manually, follow http://llvmlite.pydata.org/en/latest/install/index.html#building-manually. Finally run the tests for both llvmlite and Numba to verify you have a working Numba installation.

Now the details for the OpenCL / SPIR-V backend. First install OpenCL 2.1 from the Intel website. A new OpenCL platform should appear by the name Experimental OpenCL 2.1 CPU Only Platform. Clone *thewilsonator/* *llvm-target-spirv* and *llvm*, place the files in llvm-target-spirv/* into llvm/lib/Target/SPIRV/, and build llvm with cmake to obtain the **llc** binary. Build the SPIRV-Tools binaries too, i.e. **spirv-as/dis/opt/val**. Finally place the binaries in a folder pointed by the environtment variable $SPIRVDIR, which defaults to /opt/spirv/

Verify the installation by running the examples available in /numba/examples/ocljit/.
Do not hesitate to open an issue for more information or help.

An effort by **StreamHPC** www.streamhpc.eu (formerly StreamComputing)

------------------------------------------------------------------------------------


A Just-In-Time(JIT) Compiler for Numerical Functions in Python.
#########################################################

Numba is an open source, NumPy-aware optimizing compiler for Python sponsored
by Anaconda, Inc.  It uses the LLVM compiler project to generate machine code
from Python syntax.

Numba can compile a large subset of numerically-focused Python, including many
NumPy functions.  Additionally, Numba has support for automatic
parallelization of loops, generation of GPU-accelerated code, and creation of
ufuncs and C callbacks.

For more information about Numba, see the Numba homepage: 
http://numba.pydata.org

Dependencies
============

* llvmlite (version 0.31.0 or higher)
* NumPy (version 1.9 or higher)
* funcsigs (for Python 2)

Supported Platforms
===================

* Operating systems and CPU:

  - Linux: x86 (32-bit), x86_64, ppc64le (POWER8 and 9), ARMv7 (32-bit),
    ARMv8 (64-bit)
  - Windows: x86, x86_64
  - macOS: x86_64
  
* Python versions: 2.7, 3.5-3.7
* NumPy: >= 1.11
* NVIDIA GPUs (Kepler architecture or later) via CUDA driver on Linux, Windows,
  macOS (< 10.14)
* AMD GPUs via ROCm driver on Linux
* llvmlite: >= 0.31.0


Installing
==========

The easiest way to install Numba and get updates is by using the Anaconda
Distribution: https://www.anaconda.com/download

::

   $ conda install numba

For more options, see the Installation Guide: http://numba.pydata.org/numba-doc/latest/user/installing.html

Documentation
=============

http://numba.pydata.org/numba-doc/latest/index.html


Mailing Lists
=============

Join the Numba mailing list numba-users@continuum.io:
https://groups.google.com/a/continuum.io/d/forum/numba-users

Some old archives are at: http://librelist.com/browser/numba/


Continuous Integration
======================

.. image:: https://travis-ci.org/numba/numba.svg?branch=master
    :target: https://travis-ci.org/numba/numba
    :alt: Travis CI

.. image:: https://dev.azure.com/numba/numba/_apis/build/status/numba.numba?branchName=master
    :target: https://dev.azure.com/numba/numba/_build/latest?definitionId=1?branchName=master
    :alt: Azure Pipelines
