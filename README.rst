
DPPY
====

========
1. What?
========

DPPy proof-of-concept backend for NUMBA to support compilation for Intel CPU and
GPU architectures. The present implementation of DPPy is based on OpenCL 2.1,
but is likely to change in the future to rely on Sycl/DPC++ or Intel Level-0
driver API.

===============
2. Prequisites?
===============

- Bash                 : In the system and not as default Shell
- Tar                  : To extract files
- Git                  : To fetch required dependencies listed below
- C/C++ compiler       : To build the dependencies
- Cmake                : For managing build process of dependencies
- Python3              : Version 3 is required
- Conda or miniconda   : Can be found at https://docs.conda.io/en/latest/miniconda.html
- OpenCL 2.1 driver    : DPPy currently works for both Intel GPUs and CPUs is a correct OpenCL driver version is found on the system. 
Note. To use the GPU users should be added to "video" user group on Linux systems.


The following requisites will need to be present in the system. Refer to next section for more details.
*******************************************************************************************************

- NUMBA v0.48          : The DPPy backend has only been tested for NUMBA v0.48. The included install script downloads and applies the DDPy patch to the correct NUMBA version.

- LLVM-SPIRV translator: Used for SPIRV generation from LLVM IR.

- SPIRV-Tools          : Used internally for code-generation.

- LLVMDEV              : To support LLVM IR generation.

- Others               : All existing dependecies for NUMBA, such as llvmlite, also apply to DPPy.

==================
3. How to install?
==================
Install Pre-requisites
*************************
Make sure the dependencies of NUMBA-DPPY are installed in the system, for convenience
and to make sure the dependencies are installed with consistent version of LLVM we provide
installation script that will create a CONDA environment and install LLVM-SPIRV translator,
SPIRV-Tools and llvmlite in that environment. **To use this CONDA has to be available in the system**.

The above mentioned installation script can be found `here <https://github.intel.com/SAT/numba-pvc-build-scripts>`_. Please follow the README to run the installation script. 

After successful installation the following message should be displayed:

    | #
    | # Use the following to activate the correct environment
    | #
    | # `    $ ``conda activate numba-dppy-env`` `
    | #
    | #  Use the following to deactivate environment
    | #
    | # `    $ ``conda deactivate`` `

The installer script creates a new conda environment called numba-dppy-env with
all the needed dependencies already installed. **Please activate the numba-dppy-env before proceeding**.


Install DPPY backend
***********************
NUMBA-DPPY also depend on DPPY backend. It can be found `here <https://github.intel.com/SAT/dppy>`_. Please run 
`build_for_conda.sh` to install DPPY backend.

Install NUMBA-DPPY
*********************
After all the dependencies are installed please run ``build_for_develop.sh`` to get a local installation of NUMBA-DPPY. **Both step 2 and 3 assumes CONDA environment with
the dependencies of NUMBA-DPPY installed in it, was activated**.

================
4. Running tests
================

To make sure the installation was successful, try running the examples and the
test suite:

    $PATH_TO_NUMBA-DPPY/numba/dppy/examples/

To run the test suite execute the following:

    $ ``python -m numba.runtests numba.dppy.tests``

===========================
5. How Tos and Known Issues
===========================

Refer the HowTo.rst guide for an overview of the programming semantics,
examples, supported functionalities, and known issues.


===================
6. Reporting issues
===================

Please email diptorup.deb@intel.com to report issues and bugs.
