===================
Sharing CUDA Memory
===================

.. _cuda-ipc-memory:

Sharing between process
=======================

.. warning:: This feature is limited to Linux only.

A device array can be shared with another process in the same machine using
the CUDA IPC API.  To do so, use the ``.get_ipc_handle()`` method on the device
array to get a ``IpcArrayHandle`` object, which can be transferred to another
process.


.. automethod:: numba.cuda.cudadrv.devicearray.DeviceNDArray.get_ipc_handle

.. autoclass:: numba.cuda.cudadrv.devicearray.IpcArrayHandle
    :members: open, close