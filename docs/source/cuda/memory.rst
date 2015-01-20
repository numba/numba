Memory management
=================

Even though Numba can automatically transfer NumPy arrays to the device,
it can only do so conservatively by always trasnferring device memory back to
the host when a kernel finishes. To avoid the unnecessary transfer for
read-only arrays, use memory management APIs to manually control the transfer.

To copy an NumPy array to the device, use ``cuda.to_device``::


    import numpy
    from numba import cuda

    arr = numpy.arange(100)
    d_arr = cuda.to_device(arr)  # copy to device

Numba implements a device array object.  When the refcount of the device
array drops to zero, the underlying device memory allocation is released.

The device array provides a ``.copy_to_host`` method for transferring data
back to the host::

    arr = d_arr.copy_to_host()  # creates a new numpy array
    d_arr.copy_to_host(arr)     # copy data to the specified array


