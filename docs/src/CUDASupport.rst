------------
CUDA Support
------------

All CUDA support requires `CUDA 5 <https://developer.nvidia.com/cuda-toolkit>`_ and `NVVM <https://developer.nvidia.com/cuda-llvm-compiler>`_.  NVVM is a CUDA LLVM Compiler that is currently available to `NVIDIA CUDA Registered Developers <https://developer.nvidia.com/nvidia-registered-developer-program>`_.  NVVM will be integrated into the official CUDA Toolkit in a future release.

NumbaPro tries to locate the CUDA driver automatically.  Users can override the location by setting the environment variable NUMBAPRO_CUDA_DRIVER to point to the path of the CUDA driver shared library.

For the NVVM shared library, NumbaPro will try to find it in the current directory.  Users can override the location by providing the environment variable NUMBAPRO_NVVM that points to the path of the NVVM shared library.

All CUDA features are experimental. Computation speeds may have unexpected results.

**Supported**: Linux, Windows and Mac OSX 32/64-bit with CUDA 5 and NVVM.  Only support CUDA devices with compute capability 2.0 and above.

