------------
CUDA Support
------------

.. NOTE:: Platforms supported: Linux, Windows and Mac OSX 32/64-bit.
          Only support CUDA devices with **compute capability 2.0 and above**.
          Please see `CUDA GPUs <https://developer.nvidia.com/cuda-gpus>`_ for 
          a list of CUDA GPUs and their compute capability.


.. NOTE:: As of version 0.12.2, `Anaconda <https://store.continuum.io/cshop/anaconda/>`_ is distributing a subset of CUDA 
          toolkit 5.5 libraries.  
          The following information is no longer necessary for most users.


Numba searches in the system shared library path for the CUDA drivers and CUDA libraries (e.g. cuRAND, cuBLAS).  Users can set environment variable **LD_LIBRARY_PATH** to the directory of the CUDA drivers to ensure that Numba can find them.  The instruction to do so is printed at the end of the CUDA SDK installation.

User can override the search path with the following environment variables:

- **NUMBAPRO_CUDA_DRIVER**
    path to CUDA driver shared library file
- **NUMBAPRO_NVVM**
    path to CUDA libNVVM shared library file
- **NUMBAPRO_LIBDEVICE**
    path to CUDA libNVVM libdevice directory which contains .bc files.
