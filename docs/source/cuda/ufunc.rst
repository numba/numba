CUDA Ufuncs and Generalized Ufuncs
==================================

This page describes the CUDA ufunc-like object.

To support the programming pattern of CUDA programs, CUDA Vectorize and
GUVectorize cannot produce a conventional ufunc.  Instead, a ufunc-like
object is returned.  This object is a close analog but not fully
compatible with a regular NumPy ufunc.  The CUDA ufunc adds support for
passing intra-device arrays (already on the GPU device) to reduce
traffic over the PCI-express bus.  It also accepts a `stream` keyword
for launching in asynchronous mode.

Example: Basic Example
------------------------

::

    import math
    from numba import vectorize, cuda
    import numpy as np

    @vectorize(['float32(float32, float32, float32)',
                'float64(float64, float64, float64)'],
               target='cuda')
    def cu_discriminant(a, b, c):
        return math.sqrt(b ** 2 - 4 * a * c)

    N = 1e+4
    dtype = np.float32

    # prepare the input
    A = np.array(np.random.sample(N), dtype=dtype)
    B = np.array(np.random.sample(N) + 10, dtype=dtype)
    C = np.array(np.random.sample(N), dtype=dtype)

    D = cu_discriminant(A, B, C)

    print(D)  # print result

Example: Calling Device Functions
----------------------------------

All CUDA ufunc kernels have the ability to call other CUDA device functions::

    from numba import vectorize, cuda

    # define a device function
    @cuda.jit('float32(float32, float32, float32)', device=True, inline=True)
    def cu_device_fn(x, y, z):
        return x ** y / z

    # define a ufunc that calls our device function
    @vectorize(['float32(float32, float32, float32)'], target='cuda')
    def cu_ufunc(x, y, z):
        return cu_device_fn(x, y, z)


Generalized CUDA ufuncs
-----------------------

Generalized ufuncs may be executed on the GPU using CUDA, analogous to
the CUDA ufunc functionality.  This may be accomplished as follows::

    from numba import guvectorize

    @guvectorize(['void(float32[:,:], float32[:,:], float32[:,:])'], 
                 '(m,n),(n,p)->(m,p)', target='cuda')
    def matmulcore(A, B, C):
        ...

There are times when the gufunc kernel uses too many of a GPU's
resources, which can cause the kernel launch to fail.  The user can
explicitly control the maximum size of the thread block by setting
the `max_blocksize` attribute on the compiled gufunc object.

::

    from numba import guvectorize

    @guvectorize(..., target='cuda')
    def very_complex_kernel(A, B, C):
        ...

    very_complex_kernel.max_blocksize = 32  # limits to 32 threads per block

.. comment

    Example: A Chunk at a Time
    ---------------------------

    Partitioning your data into chunks allows computation and memory transfer
    to be overlapped.  This can increase the throughput of your ufunc and
    enables your ufunc to operate on data that is larger than the memory
    capacity of your GPU.  For example:

    ::

        import math
        from numba import vectorize, cuda
        import numpy as np

        # the ufunc kernel
        def discriminant(a, b, c):
            return math.sqrt(b ** 2 - 4 * a * c)

        cu_discriminant = vectorize(['float32(float32, float32, float32)',
                                     'float64(float64, float64, float64)'],
                                    target='cuda')(discriminant)

        N = 1e+8
        dtype = np.float32

        # prepare the input
        A = np.array(np.random.sample(N), dtype=dtype)
        B = np.array(np.random.sample(N) + 10, dtype=dtype)
        C = np.array(np.random.sample(N), dtype=dtype)
        D = np.empty(A.shape, dtype=A.dtype)

        # create a CUDA stream
        stream = cuda.stream()

        chunksize = 1e+6
        chunkcount = N // chunksize

        # partition numpy arrays into chunks
        # no copying is performed
        sA = np.split(A, chunkcount)
        sB = np.split(B, chunkcount)
        sC = np.split(C, chunkcount)
        sD = np.split(D, chunkcount)

        device_ptrs = []

        with stream.auto_synchronize():
            # every operation in this context with be launched asynchronously
            # by using the CUDA stream

            # for each chunk
            for a, b, c, d in zip(sA, sB, sC, sD):
                # transfer to device
                dA = cuda.to_device(a, stream)
                dB = cuda.to_device(b, stream)
                dC = cuda.to_device(c, stream)
                dD = cuda.to_device(d, stream, copy=False) # no copying
                # launch kernel
                cu_discriminant(dA, dB, dC, out=dD, stream=stream)
                # retrieve result
                dD.copy_to_host(d, stream)
                # store device pointers to prevent them from freeing before
                # the kernel is scheduled
                device_ptrs.extend([dA, dB, dC, dD])

        # data is ready at this point inside D
