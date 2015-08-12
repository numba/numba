=================
Memory management
=================

.. _hsa-device-memory:

The CPU and GPU in a APU share the same main memory.  There is no distinction
between CPU and GPU memory.  Even though a HSA kernel can directly consume any
data in the main memory, it is recommended to register a memory region to the
HSA runtime compatibility with HSA-compliant discrete GPUs.

.. function:: hsa.register(*arrays)

    Register every given array.  The function can be used in a *with-context*
    for automically deregistration::

        array_a = numpy.arange(10)
        array_b = numpy.arange(10)
        with hsa.register(array_a, array_b):
            some_hsa_code(array_a, array_b)


.. function:: hsa.deregister(*arrays)

   Deregister every given array

.. _hsa-shared-memory:

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

.. function:: numba.hsa.shared.array(shape, type)

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

.. function:: numba.hsa.barrier(scope)

   The ``scope`` argument specifies the level of synchronization.  Set ``scope``
   to ``1`` to synchronize all workitems in the same workgroup.

.. seealso::
   :ref:`Matrix multiplication example <hsa-matmul>`.

