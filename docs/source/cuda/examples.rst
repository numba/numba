
========
Examples
========

.. _cuda-matmul:

Matrix multiplication
=====================

Here is a naive implementation of matrix multiplication using a CUDA kernel:

.. code-block:: python
  :linenos:

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
for threads in a block to cooperatively compute on a task.  The following
implements a faster version of the square matrix multiplication using shared memory:

.. code-block:: python
  :linenos:

    from numba import cuda, float32

    # Controls threads per block and shared memory usage.
    # The computation will be done on blocks of TPBxTPB elements.
    # TBP should not be larger than 32 in this example
    TPB = 16

    from numba import cuda, float32

    @cuda.jit
    def fast_matmul(A, B, C):
        """Based on corrected version by @RobertCrovella from https://stackoverflow.com/a/64198479/13697228
        """
        # Define an array in the shared memory
        # The size and type of the arrays must be known at compile time
        sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
        sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

        x, y = cuda.grid(2)

        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bpg = cuda.gridDim.x    # blocks per grid

        # Each thread computes one element in the result matrix.
        # The dot product is chunked into dot products of TPB-long vectors.
        tmp = float32(0.)
        for i in range(bpg):
            # Preload data into shared memory
            sA[tx, ty] = 0
            sB[tx, ty] = 0
            if x < A.shape[0] and (ty+i*TPB) < A.shape[1]:
              sA[tx, ty] = A[x, ty + i * TPB]
            if y < B.shape[1] and (tx+i*TPB) < B.shape[0]:
              sB[tx, ty] = B[tx + i * TPB, y]

            # Wait until all threads finish preloading
            cuda.syncthreads()

            # Computes partial product on the shared memory
            for j in range(TPB):
                tmp += sA[tx, j] * sB[j, ty]

            # Wait until all threads finish computing
            cuda.syncthreads()
        if x < C.shape[0] and y < C.shape[1]:
            C[x, y] = tmp

Because the shared memory is a limited resource, the code preloads small
block at a time from the input arrays.  Then, it calls
:func:`~numba.cuda.syncthreads` to wait until all threads have finished
preloading and before doing the computation on the shared memory.
It synchronizes again after the computation to ensure all threads
have finished with the data in shared memory before overwriting it
in the next loop iteration.

An example usage of this function is as follows:

.. code-block:: python
  :linenos:

    x_h = np.arange(16).reshape([4,4])
    y_h = np.ones([4,4])
    z_h = np.zeros([4,4])

    x_d = cuda.to_device(x_h)
    y_d = cuda.to_device(y_h)
    z_d = cuda.to_device(z_h)

    TPB = 3
    threadsperblock = (TPB, TPB)
    blockspergrid_x = math.ceil(z_h.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(z_h.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    fast_matmul[blockspergrid, threadsperblock](x_d, y_d, z_d)
    z_h = z_d.copy_to_host()
    print(z_h)
    print(x_h@y_h)
:ref:`the CUDA Simulator documentation <simulator>`
This passes a :ref:`CUDA memory check test <debugging-cuda-python-code>`:

.. code-block:: none

    $ cuda-memcheck python t49.py
    ========= CUDA-MEMCHECK
    [[ 6.  6.  6.  6.]
    [22. 22. 22. 22.]
    [38. 38. 38. 38.]
    [54. 54. 54. 54.]]
    [[ 6.  6.  6.  6.]
    [22. 22. 22. 22.]
    [38. 38. 38. 38.]
    [54. 54. 54. 54.]]
    ========= ERROR SUMMARY: 0 errors
