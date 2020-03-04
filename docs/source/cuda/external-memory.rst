===========================================
External Memory Management plugin interface
===========================================

.. _cuda-emm-plugin:

By default, Numba allocates memory on CUDA devices by interacting with the CUDA
driver API to call functions such as ``cuMemAlloc`` and ``cuMemFree``. This is
suitable for many use cases. When Numba is used in conjunction with other
CUDA-aware libraries that also allocate memory,

Plugin interface
================

.. autoclass:: numba.cuda.BaseCUDAMemoryManager
   :members: __init__, memalloc, memhostalloc, mempin, initialize,
             get_ipc_handle, get_memory_info, reset, defer_cleanup,
             interface_version

.. autoclass:: numba.cuda.cudadrv.driver.MemoryInfo

Memory pointers
===============

.. autoclass:: numba.cuda.MemoryPointer

.. autoclass:: numba.cuda.MappedMemory

.. autoclass:: numba.cuda.PinnedMemory

IPC
===

.. autoclass:: numba.cuda.cudadrv.driver.IpcHandle
