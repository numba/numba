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

.. autofunction:: numba.cuda.device_array
   :noindex:
.. autofunction:: numba.cuda.device_array_like
   :noindex:
.. autofunction:: numba.cuda.to_device
   :noindex:

Device arrays
-------------

Device array references have the following methods.  These methods are to be
called in host code, not within CUDA-jitted functions.

.. autoclass:: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :members: copy_to_host, is_c_contiguous, is_f_contiguous, ravel, reshape
    :noindex:

Pinned memory
=============

.. autofunction:: numba.cuda.pinned
   :noindex:
.. autofunction:: numba.cuda.pinned_array
   :noindex:

Streams
=======

.. autofunction:: numba.cuda.stream
   :noindex:

CUDA streams have the following methods:

.. autoclass:: numba.cuda.cudadrv.driver.Stream
    :members: synchronize, auto_synchronize
    :noindex:

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
   :noindex:

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
   :noindex:

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
   :noindex:

   Allocate a local array of the given *shape* and *type* on the device.
   The array is private to the current thread.  An array-like object is
   returned which can be read and written to like any standard array
   (e.g. through indexing).

SmartArrays (experimental)
==========================

Numba provides an Array-like data type that manages data movement to
and from the device automatically. It can be used as drop-in replacement for
`numpy.ndarray` in most cases, and is supported by Numba's JIT-compiler for both
'host' and 'cuda' target.

.. comment: function:: numba.SmartArray(obj=None, copy=True,
                               shape=None, dtype=None, order=None, where='host')

.. autoclass:: numba.SmartArray
   :members: __init__, host, host_changed, gpu, gpu_changed
	       

Thus, `SmartArray` objects may be passed as function arguments to jit-compiled
functions. Whenever a cuda.jit-compiled function is being executed, it will
trigger a data transfer to the GPU (unless the data are already there). But instead
of transferring the data back to the host after the function completes, it leaves
the data on the device and merely updates the host-side if there are any external
references to that.
Thus, if the next operation is another invocation of a cuda.jit-compiled function,
the data does not need to be transferred again, making the compound operation more
efficient (and making the use of the GPU advantagous even for smaller data sizes).
