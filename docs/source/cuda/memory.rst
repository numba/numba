=================
Memory management
=================

Data transfer
=============

Even though Numba can automatically transfer NumPy arrays to the device,
it can only do so conservatively by always transferring device memory back to
the host when a kernel finishes. To avoid the unnecessary transfer for
read-only arrays, you can use the following APIs to manually control the
transfer:

.. function:: numba.cuda.to_device(array, stream=0)

   Copy the Numpy *array* to device memory.  A device array reference
   is returned which can be passed as argument to a kernel expecting the
   same kind of array.

   If a CUDA *stream* is given, then the transfer will be made asynchronously
   as part as the given stream.  Otherwise, the transfer is synchronous:
   the function returns after the copy is finished.

   The lifetime of the allocated device memory is managed by Numba.  Once
   it isn't referred to anymore, it is automatically released.

Device arrays
-------------

Device array references have the following methods:

.. method:: copy_to_host(array=None, stream=0)

   Copy back contents of the device array to Numpy *array* on the host.
   If *array* is not given, a new array is allocated and returned.

   If a CUDA *stream* is given, then the transfer will be made asynchronously
   as part as the given stream.  Otherwise, the transfer is synchronous:
   the function returns after the copy is finished.

   Example::

      import numpy as np
      from numba import cuda

      arr = np.arange(1000)
      d_arr = cuda.to_device(arr)

      my_kernel[100, 100](d_arr)

      result_array = d_arr.copy_to_host()

.. method:: is_c_contiguous()

   Return whether the array is C-contiguous.

.. method:: is_f_contiguous()

   Return whether the array is Fortran-contiguous.

.. method:: ravel(order='C')

   Flatten the array without changing its contents, similarly to
   :meth:`numpy.ndarray.ravel`.

.. method:: reshape(*newshape, order='C')

   Change the array's shape without changing its contents, similarly to
   :meth:`numpy.ndarray.reshape`.  Example::

      d_arr = d_arr.reshape(20, 50, order='F')


CUDA streams
============

.. function:: numba.cuda.stream()

   Create and return a CUDA stream.  A CUDA stream acts as a command queue
   for the device.

   CUDA streams have the following method:

   .. method:: synchronize()

      Wait for all commands in this stream to execute.  This will commit
      any pending memory transfers.
