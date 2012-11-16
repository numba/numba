--------------
CUDA ufunc
--------------

This page describes the CUDA ufunc-like object returned from Vectorize.build_ufunc and GUVectorize.build_ufunc.

To support the programming pattern of CUDA programs, CUDA Vectorize and GUVectorize cannot produce regular ufunc.  Instead, a ufunc-like object is returned.  This object is mostly compatible with regular ufunc.  The CUDA ufunc adds support for passing in-device arrays to reduce traffic over the PCI-express.  It also accepts a `stream` keyword for launching in asynchronous mode.


Example: A Chunk at a Time
---------------------------

Partitioning your data into chunks allows computation and memory transfer to be overlapped.  This can increase the throughput of your ufunc, and enables your ufunc to operate on data that is larger than the memory capacity of your GPU.  For example::

    # the ufunc kernel
    def discriminant(a, b, c):
        return math.sqrt(b ** 2 - 4 * a * c)
        
    # create the ufunc
    cu_discriminant = vectorize([f4(f4, f4, f4), f8(f8, f8, f8)],
                                target='gpu')(discriminant)

    N = 1e+8
    
    # prepare the input
    A = np.array(np.random.sample(n), dtype=dtype)
    B = np.array(np.random.sample(n) + 10, dtype=dtype)
    C = np.array(np.random.sample(n), dtype=dtype)
    D = np.empty(A.shape, dtype=A.dtype)

    # create a CUDA stream
    stream = cuda.stream()

    chunksize = 1e+7
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
            dD.to_host(stream)
            # store device pointers to prevent them from freeing before
            # the kernel is scheduled
            device_ptrs.extend([dA, dB, dC, dD])

    # data is ready at this point inside D
    

Example: Calling Device Functions
----------------------------------

All CUDA ufunc kernels can call other CUDA device functions::
    
    from numbapro import vectorize
    from numba import *

    # define a device function
    @jit(f4(f4, f4, f4), device=True, inline=True, target='gpu')
    def cu_device_fn(x, y, z):
        return x ** y / z
        
    # define a ufunc that calls our device function
    @vectorize([f4(f4, f4, f4)], target='gpu')
    def cu_ufunc(x, y, z):
        return cu_device_fn(x, y, z)

