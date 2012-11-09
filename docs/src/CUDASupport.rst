------------
CUDA Support
------------

All CUDA supports are dependent on `CUDA 5 <https://developer.nvidia.com/cuda-toolkit>`_ and `NVVM <https://developer.nvidia.com/cuda-llvm-compiler>`_.  NVVM is a CUDA LLVM Compiler that is only available to registered Nvidia Developer.

NumbaPro tries to locate the CUDA driver automatically.  Users can override the location by setting the environment variable NUMBAPRO_CUDA_DRIVER to point to the path of the CUDA driver shared library.

For the NVVM shared library, NumbaPro will try to find it in the current directory.  Users can override the location by providing the environment variable NUMBAPRO_NVVM that points to the path of the NVVM shared library.

All CUDA features are experimental. Computation speeds may have unexpected results.  CUDA cards with compute capability less than 1.3 do not support double-precision floating-point arithmetic.

**Supported**: Linux, Windows and Mac OSX 32/64-bit with CUDA 5 and NVVM.