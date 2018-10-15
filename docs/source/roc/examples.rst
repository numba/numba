
========
Examples
========

.. _roc-matmul:

Matrix multiplication
=====================

Here is a naive implementation of matrix multiplication using a HSA kernel::


    @roc.jit
    def matmul(A, B, C):
        i = roc.get_global_id(0)
        j = roc.get_global_id(1)

        if i >= C.shape[0] or j >= C.shape[1]:
            return

        tmp = 0

        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]

        C[i, j] = tmp

This implementation is straightforward and intuitive but performs poorly,
because the same matrix elements will be loaded multiple times from device
memory, which is slow (some devices may have transparent data caches, but
they may not be large enough to hold the entire inputs at once).

It will be faster if we use a blocked algorithm to reduce accesses to the
device memory.  HSA provides a fast :ref:`shared memory <roc-shared-memory>`
for workitems in a group to cooperatively compute on a task.  The following
implements a faster version of the square matrix multiplication using shared
memory::


    import numpy as np
    from numba import roc
    from numba import float32
    from time import time as timer

    blocksize = 16
    gridsize = 16

    @roc.jit('(float32[:,:], float32[:,:], float32[:,:])')
    def matmulfast(A, B, C):
        x = roc.get_global_id(0)
        y = roc.get_global_id(1)

        tx = roc.get_local_id(0)
        ty = roc.get_local_id(1)

        sA = roc.shared.array(shape=(blocksize, blocksize), dtype=float32)
        sB = roc.shared.array(shape=(blocksize, blocksize), dtype=float32)

        if x >= C.shape[0] or y >= C.shape[1]:
            return

        tmp = 0

        for i in range(gridsize):
            # preload
            sA[tx, ty] = A[x, ty + i * blocksize]
            sB[tx, ty] = B[tx + i * blocksize, y]
            # wait for preload to end
            roc.barrier(1)
            # compute loop
            for j in range(blocksize):
                tmp += sA[tx, j] * sB[j, ty]
            # wait for compute to end
            roc.barrier(1)

        C[x, y] = tmp

    N = gridsize * blocksize
    A = np.random.random((N, N)).astype(np.float32)
    B = np.random.random((N, N)).astype(np.float32)
    C = np.zeros_like(A)

    griddim = gridsize, gridsize
    blockdim = blocksize, blocksize

    with roc.register(A, B, C):
        ts = timer()
        matmulfast[griddim, blockdim](A, B, C)
        te = timer()
        print("1st GPU time:", te - ts)

    with roc.register(A, B, C):
        ts = timer()
        matmulfast[griddim, blockdim](A, B, C)
        te = timer()
        print("2nd GPU time:", te - ts)

    ts = timer()
    ans = np.dot(A, B)
    te = timer()
    print("CPU time:", te - ts)
    np.testing.assert_allclose(ans, C, rtol=1e-5)


Because the shared memory is a limited resource, the code preloads a small
block at a time from the input arrays.  Then, it calls
:func:`~numba.roc.barrier` to wait until all threads have finished
preloading before doing the computation on the shared memory.
It synchronizes again after the computation to ensure all threads
have finished with the data in shared memory before overwriting it
in the next loop iteration.


