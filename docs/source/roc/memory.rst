=================
Memory management
=================

.. _roc-device-memory:

Data transfer
=============

Even though Numba can automatically transfer NumPy arrays to the device,
it can only do so conservatively by always transferring device memory back to
the host when a kernel finishes. To avoid the unnecessary transfer for
read-only arrays, you can use the following APIs to manually control the
transfer:

.. autofunction:: numba.roc.device_array
   :noindex:
.. autofunction:: numba.roc.device_array_like
   :noindex:
.. autofunction:: numba.roc.to_device
   :noindex:

Device arrays
-------------

Device array references have the following methods.  These methods are to be
called in host code, not within ROC-jitted functions.

.. autoclass:: numba.roc.hsadrv.devicearray.DeviceNDArray
    :members: copy_to_host, is_c_contiguous, is_f_contiguous, ravel, reshape
    :noindex:


Data Registration
-----------------

The CPU and GPU do not share the same main memory, however, it is recommended to
register a memory allocation to the HSA runtime for as a performance optimisation
hint.

.. function:: roc.register(*arrays)

    Register every given array.  The function can be used in a *with-context*
    for automically deregistration::

        array_a = numpy.arange(10)
        array_b = numpy.arange(10)
        with roc.register(array_a, array_b):
            some_hsa_code(array_a, array_b)


.. function:: roc.deregister(*arrays)

   Deregister every given array


Streams
=======

.. autofunction:: numba.roc.stream
   :noindex:

ROC streams have the following methods:

.. autoclass:: numba.roc.hsadrv.driver.Stream
    :members: synchronize, auto_synchronize
    :noindex:


.. _roc-shared-memory:

Shared memory and thread synchronization
========================================

A limited amount of shared memory can be allocated on the device to speed
up access to data, when necessary.  That memory will be shared (i.e. both
readable and writable) amongst all workitems  belonging to a given group
and has faster access times than regular device memory.  It also allows
workitems to cooperate on a given solution.  You can think of it as a
manually-managed data cache.

The memory is allocated once for the duration of the kernel, unlike
traditional dynamic memory management.

.. function:: numba.roc.shared.array(shape, type)

   Allocate a shared array of the given *shape* and *type* on the device.
   This function must be called on the device (i.e. from a kernel or
   device function).  *shape* is either an integer or a tuple of integers
   representing the array's dimensions.  *type* is a :ref:`Numba type <numba-types>`
   of the elements needing to be stored in the array.

   The returned array-like object can be read and written to like any normal
   device array (e.g. through indexing).

   A common pattern is to have each workitem populate one element in the
   shared array and then wait for all workitems to finish using :func:`
   .barrier`.

.. function:: numba.roc.barrier(scope)

   The ``scope`` argument specifies the level of synchronization.  Set ``scope``
   to ``roc.CLK_GLOBAL_MEM_FENCE`` or ``roc.CLK_LOCAL_MEM_FENCE`` to synchronize
   all workitems across a workgroup when accessing global memory or local memory
   respectively.

.. function:: numba.roc.wavebarrier

    Creates an execution barrier across a wavefront to force a synchronization
    point.

.. seealso::
   :ref:`Matrix multiplication example <roc-matmul>`.
