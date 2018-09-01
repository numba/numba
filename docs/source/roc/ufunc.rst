ROC Ufuncs and Generalized Ufuncs
==================================

This page describes the ROC ufunc-like object.

To support the programming pattern of ROC programs, ROC Vectorize and
GUVectorize cannot produce a conventional ufunc.  Instead, a ufunc-like
object is returned.  This object is a close analog but not fully
compatible with a regular NumPy ufunc.  The ROC ufunc adds support for
passing intra-device arrays (already on the GPU device) to reduce
traffic over the PCI-express bus.  It also accepts a `stream` keyword
for launching in asynchronous mode.

Basic ROC UFunc Example
-----------------------

::

    import math
    from numba import vectorize
    import numpy as np

    @vectorize(['float32(float32, float32, float32)',
                'float64(float64, float64, float64)'],
               target='roc')
    def roc_discriminant(a, b, c):
        return math.sqrt(b ** 2 - 4 * a * c)

    N = 10000
    dtype = np.float32

    # prepare the input
    A = np.array(np.random.sample(N), dtype=dtype)
    B = np.array(np.random.sample(N) + 10, dtype=dtype)
    C = np.array(np.random.sample(N), dtype=dtype)

    D = roc_discriminant(A, B, C)

    print(D)  # print result

Calling Device Functions from ROC UFuncs
----------------------------------------

All ROC ufunc kernels have the ability to call other ROC device functions::

    from numba import vectorize, roc

    # define a device function
    @roc.jit('float32(float32, float32, float32)', device=True)
    def roc_device_fn(x, y, z):
        return x ** y / z

    # define a ufunc that calls our device function
    @vectorize(['float32(float32, float32, float32)'], target='roc')
    def roc_ufunc(x, y, z):
        return roc_device_fn(x, y, z)


Generalized ROC ufuncs
----------------------

Generalized ufuncs may be executed on the GPU using ROC, analogous to
the ROC ufunc functionality.  This may be accomplished as follows::

    from numba import guvectorize

    @guvectorize(['void(float32[:,:], float32[:,:], float32[:,:])'], 
                 '(m,n),(n,p)->(m,p)', target='roc')
    def matmulcore(A, B, C):
        ...

.. seealso::
   :ref:`Matrix multiplication example <roc-matmul>`.


Async execution: A Chunk at a Time
----------------------------------

Partitioning your data into chunks allows computation and memory transfer
to be overlapped.  This can increase the throughput of your ufunc and
enables your ufunc to operate on data that is larger than the memory
capacity of your GPU.  For example::

    import math
    from numba import vectorize, roc
    import numpy as np

    # the ufunc kernel
    def discriminant(a, b, c):
        return math.sqrt(b ** 2 - 4 * a * c)

    roc_discriminant = vectorize(['float32(float32, float32, float32)'],
                                target='roc')(discriminant)

    N = int(1e+8)
    dtype = np.float32

    # prepare the input
    A = np.array(np.random.sample(N), dtype=dtype)
    B = np.array(np.random.sample(N) + 10, dtype=dtype)
    C = np.array(np.random.sample(N), dtype=dtype)
    D = np.zeros(A.shape, dtype=A.dtype)

    # create a ROC stream
    stream = roc.stream()

    chunksize = 1e+6
    chunkcount = N // chunksize

    # partition numpy arrays into chunks
    # no copying is performed
    sA = np.split(A, chunkcount)
    sB = np.split(B, chunkcount)
    sC = np.split(C, chunkcount)
    sD = np.split(D, chunkcount)

    device_ptrs = []

    # helper function, async requires operation on coarsegrain memory regions
    def async_array(arr):
        coarse_arr = roc.coarsegrain_array(shape=arr.shape, dtype=arr.dtype)
        coarse_arr[:] = arr
        return coarse_arr

    with stream.auto_synchronize():
        # every operation in this context with be launched asynchronously
        # by using the ROC stream

        dchunks = [] # holds the result chunks

        # for each chunk
        for a, b, c, d in zip(sA, sB, sC, sD):
            # create coarse grain arrays
            asyncA = async_array(a)
            asyncB = async_array(b)
            asyncC = async_array(c)
            asyncD = async_array(d)

            # transfer to device
            dA = roc.to_device(asyncA, stream=stream)
            dB = roc.to_device(asyncB, stream=stream)
            dC = roc.to_device(asyncC, stream=stream)
            dD = roc.to_device(asyncD, stream=stream, copy=False) # no copying

            # launch kernel
            roc_discriminant(dA, dB, dC, out=dD, stream=stream)

            # retrieve result
            dD.copy_to_host(asyncD, stream=stream)

            # store device pointers to prevent them from freeing before
            # the kernel is scheduled
            device_ptrs.extend([dA, dB, dC, dD])

            # store result reference
            dchunks.append(asyncD)

    # put result chunks into the output array 'D'
    for i, result in enumerate(dchunks):
        sD[i][:] = result[:]

    # data is ready at this point inside D
    print(D)


