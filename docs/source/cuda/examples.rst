
========
Examples
========

.. _cuda-matmul:

Matrix multiplication
=====================

Here is a naive implementation of matrix multiplication using a CUDA kernel::

    @cuda.jit
    def matmul(A, B, C):
        """Perform square matrix multiplication of C = A * B
        """
        i, j = cuda.grid(2)
        if i < C.shape[0] and j < C.shape[1]:
            tmp = 0.
            for k in range(A.shape[1]):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp


This implementation is straightforward and intuitive but performs poorly,
because the same matrix elements will be loaded multiple times from device
memory, which is slow (some devices may have transparent data caches, but
they may not be large enough to hold the entire inputs at once).

It will be faster if we use a blocked algorithm to reduce accesses to the
device memory.  CUDA provides a fast :ref:`shared memory <cuda-shared-memory>`
for threads in a block to cooperately compute on a task.  The following
implements a faster version of the square matrix multiplication using shared
memory::

    from numba import cuda, float32

    # Controls threads per block and shared memory usage.
    # The computation will be done on blocks of TPBxTPB elements.
    TPB = 16

    @cuda.jit
    def fast_matmul(A, B, C):
        # Define an array in the shared memory
        # The size and type of the arrays must be known at compile time
        sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
        sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

        x, y = cuda.grid(2)

        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bpg = cuda.gridDim.x    # blocks per grid

        if x >= C.shape[0] and y >= C.shape[1]:
            # Quit if (x, y) is outside of valid C boundary
            return

        # Each thread computes one element in the result matrix.
        # The dot product is chunked into dot products of TPB-long vectors.
        tmp = 0.
        for i in range(bpg):
            # Preload data into shared memory
            sA[tx, ty] = A[x, ty + i * TPB]
            sB[tx, ty] = B[tx + i * TPB, y]

            # Wait until all threads finish preloading
            cuda.syncthreads()

            # Computes partial product on the shared memory
            for j in range(TPB):
                tmp += sA[tx, j] * sB[j, ty]

            # Wait until all threads finish computing
            cuda.syncthreads()

        C[x, y] = tmp

Because the shared memory is a limited resources, the code preloads small
block at a time from the input arrays.  Then, it calls
:func:`~numba.cuda.syncthreads` to wait until all threads have finished
preloading and before doing the computation on the shared memory.
It synchronizes again after the computation to ensure all threads
have finished with the data in shared memory before overwriting it
in the next loop iteration.

