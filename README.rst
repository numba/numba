Numba with PyDPPL
=================

========
1. What?
========

DPPL proof-of-concept backend for NUMBA to support compilation for Intel CPU and
GPU architectures. The present implementation of DPPL is based on OpenCL 2.1,
but is likely to change in the future to rely on Sycl/DPC++ or Intel Level-0
driver API.

===============
2. Perquisites?
===============

- Bash                 : In the system and not as default Shell
- Tar                  : To extract files
- Git                  : To fetch required dependencies listed below
- C/C++ compiler       : To build the dependencies
- Cmake                : For managing build process of dependencies
- Python3              : Version 3 is required
- Conda or miniconda   : Can be found at https://docs.conda.io/en/latest/miniconda.html
- OpenCL 2.1 driver    : DPPL currently works for both Intel GPUs and CPUs is a correct OpenCL driver version is found on the system.
Note. To use the GPU users should be added to "video" user group on Linux systems.


The following requisites will need to be present in the system. Refer to next section for more details.
*******************************************************************************************************

- NUMBA v0.51          : The DPPL backend has only been tested for NUMBA v0.51.
                         The included install script downloads and applies
                         the DPPy patch to the correct NUMBA version.

- LLVM-SPIRV translator: Used for SPIRV generation from LLVM IR.

- LLVMDEV              : To support LLVM IR generation.

- Others               : All existing dependencies for NUMBA, such as llvmlite, also apply to DPPL.

==================
3. How to install?
==================
Install Pre-requisites
**********************
Make sure the following dependencies of NUMBA-PyDPPL are installed
in your conda environemtn:

- llvmlite =0.33
- spirv-tools
- llvm-spirv
- llvmdev
- dpCtl =0.3

Make sure the dependencies are installed with consistent version of LLVM 10.

Install dpCtl backend
*********************
NUMBA-PyDPPL also depend on dpCtl backend. It can be found `here <https://github.com/IntelPython/dpCtl>`_.
Please install dpCtl from package.

Install NUMBA-PyDPPL
********************
After all the dependencies are installed please run ``build_for_develop.sh``
to get a local installation of NUMBA-PyDPPL.

================
4. Running tests
================

To make sure the installation was successful, try running the examples and the
test suite:

    $PATH_TO_NUMBA-PyDPPL/numba/dppl/examples/

To run the test suite execute the following:

.. code-block:: bash

    python -m numba.runtests numba.dppl.tests

===========================
5. How Tos and Known Issues
===========================

Refer the HowTo.rst guide for an overview of the programming semantics,
examples, supported functionalities, and known issues.

* Installing while Intel oneAPI Base Toolkit is activated have shown to throw error
while installation of NUMBA-PyDPPL because of incompatible TBB interface,
one way around that is to temporarily move env variable TBBROOT to something else*

===================
6. Reporting issues
===================

Please use https://github.com/IntelPython/numba/issues to report issues and bugs.
