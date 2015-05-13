Memory Management
=================

.. autofunction:: numba.cuda.to_device
.. autofunction:: numba.cuda.device_array
.. autofunction:: numba.cuda.device_array_like
.. autofunction:: numba.cuda.pinned_array
.. autofunction:: numba.cuda.mapped_array
.. autofunction:: numba.cuda.pinned
.. autofunction:: numba.cuda.mapped

Device Objects
--------------

.. autoclass:: numba.cuda.cudadrv.devicearray.DeviceNDArray
   :members: copy_to_device, copy_to_host, is_c_contiguous, is_f_contiguous,
              ravel, reshape, split
.. autoclass:: numba.cuda.cudadrv.devicearray.DeviceRecord
   :members: copy_to_device, copy_to_host
.. autoclass:: numba.cuda.cudadrv.devicearray.MappedNDArray
   :members: copy_to_device, copy_to_host, split
