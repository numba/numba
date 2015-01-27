=================
Memory management
=================

.. _cuda-device-memory:

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

Device array references have the following methods.  These methods are to be
called on the host, not on the device.

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


Streams
=======

.. function:: numba.cuda.stream()

   Create and return a CUDA stream.  A CUDA stream acts as a command queue
   for the device.

   CUDA streams have the following method:

   .. method:: synchronize()

      Wait for all commands in this stream to execute.  This will commit
      any pending memory transfers.


.. _cuda-shared-memory:

Shared memory and thread synchronization
========================================

A limited amount of shared memory can be allocated on the device to speed
up access to data, when necessary.  That memory will be shared (i.e. both
readable and writable) amongst all threads belonging to a given block
and has faster access times than regular device memory.  It also allows
threads to cooperate on a given solution.  You can think of it as a
manually-managed data cache.

The memory is allocated once for the duration of the kernel, unlike
traditional dynamic memory management.

.. function:: numba.cuda.shared.array(shape, type)

   Allocate a shared array of the given *shape* and *type* on the device.
   This function must be called on the device (i.e. from a kernel or
   device function).  *shape* is either an integer or a tuple of integers
   representing the array's dimensions.  *type* is a :ref:`Numba type <numba-types>`
   of the elements needing to be stored in the array.

   The returned array-like object can be read and written to like any normal
   device array (e.g. through indexing).

   A common pattern is to have each thread populate one element in the
   shared array and then wait for all threads to finish using :func:`.syncthreads`.

.. function:: numba.cuda.syncthreads()

   Synchronize all threads in the same thread block.  This function
   implements the same pattern as `barriers <http://en.wikipedia.org/wiki/Barrier_%28computer_science%29>`_
   in traditional multi-threaded programming: this function waits
   until all threads in the block call it, at which point it returns
   control to all its callers.

.. seealso::
   :ref:`Matrix multiplication example <cuda-matmul>`.

.. _cuda-local-memory:

Local memory
============

Local memory is an area of memory private to each thread.  Using local
memory helps allocate some scratchpad area when scalar local variables
are not enough.  The memory is allocated once for the duration of the kernel,
unlike traditional dynamic memory management.

.. function:: numba.cuda.local.array(shape, type)

   Allocate a local array of the given *shape* and *type* on the device.
   The array is private to the current thread.  An array-like object is
   returned which can be read and written to like any standard array
   (e.g. through indexing).
