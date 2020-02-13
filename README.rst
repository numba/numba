                                ****************
                                      DPPY
                                ****************
=========
1. What?
=========

DPPy proof-of-concept backend for NUMBA to support compilation for Intel CPU and
GPU architectures. The present implementation of DPPy is based on OpenCL 2.1,
but is likely to change in the future to reply on Sycl/DPC++ or Intel Level-0
driver API.

================
2. Prequisites?
================

Bash                 : In the system and not as default Shell
Tar                  : To extract files
Git                  : To fetch required dependencies listed below
C/C++ compiler       : To build the dependencies
Cmake                : For managing build process of dependencies
Python3              : Version 3 is required
Conda or miniconda   : Can be found at https://docs.conda.io/en/latest/miniconda.html

OpenCL 2.1 driver    : DPPy currently works for both Intel GPUs and CPUs is
                       a correct OpenCL driver version is found on the system.

                       Note. To use the GPU users should be added to "video"
                       user group on Linux systems.


The following requisites will be installed by the install script provided with
this package.

NUMBA v0.48          : The DPPy backend has only been tested for NUMBA v0.48.
                       The included install script downloads and applies the
                       DDPy patch to the correct NUMBA version.

LLVM-SPIRV translator: Used for SPIRV generation from LLVM IR.

SPIRV-Tools          : Used internally for code-generation. The provided install
                       script would handle downloading and installing the
                       required version.

LLVMDEV              : To support LLVM IR generation.

Others               : All existing dependecies for NUMBA, such as llvmlite,
                       also apply to DPPy.

==================
3. How to install?
==================

Extract the archive:

    tar -zxvf NUMBA-PVC-offline.tar.gz

Run the installer script:

    ./build_numba_dppy.sh --prefix $PATH_TO_INSTALL_NUMBA-DPPY

After successful installation the following message should be displayed:

    #
    #  Use the following to activate the correct environment
    #
    #    $ conda activate numba-dppy-env
    #
    #  Use the following to deactivate environment
    #
    #    $ conda deactivate

The installer script creates a new conda environment called numba-dppy-env with
all the needed dependencies already installed. To use the DPPy backend, please
activate the numba-dppy-env

================
4. Running tests
================

To make sure the installation was successful, try running the examples and the
test suite:

    $PATH_TO_INSTALL_NUMBA-DPPY/numba/dppy/examples/
    $PATH_TO_INSTALL_NUMBA-DPPY/numba/dppy/tests/dppy/

To run the test suite execute the following:

    $ python -m numba.runtests numba.dppy.tests

===========================
5. How Tos and Known Issues
===========================

Refer the HowTo.rst guide for an overview of the programming semantics,
examples, supported functionalities, and known issues.


===================
6. Reporting issues
===================

Please email diptorup.deb@intel.com to report issues and bugs.


