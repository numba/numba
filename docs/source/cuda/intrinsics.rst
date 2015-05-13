
Supported Atomic Operations
===========================

Numba provides access to some of the atomic operations supported in CUDA, in the
:class:`numba.cuda.atomic` class.

Those that are presently implemented are as follows:

.. automodule:: numba.cuda
    :members: atomic
    :noindex:

Example
'''''''

The following code demonstrates the use of :class:`numba.cuda.atomic.max` to
find the maximum value in an array. Note that this is not the most efficient way
of finding a maximum in this case, but that it serves as an example::

    from numba import cuda
    import numpy as np

    @cuda.jit
    def max_example(result, values):
        """Find the maximum value in values and store in result[0]"""
        tid = cuda.threadIdx.x
        bid = cuda.blockIdx.x
        bdim = cuda.blockDim.x
        i = (bid * bdim) + tid
        cuda.atomic.max(result, 0, values[i])


    arr = np.random.rand(16384)
    result = np.zeros(1, dtype=np.float64)

    max_example[256,64](result, arr)
    print(result[0]) # Found using cuda.atomic.max
    print(max(arr))  # Print max(arr) for comparision (should be equal!)

