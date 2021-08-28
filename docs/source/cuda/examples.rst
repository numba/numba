
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
implements a faster version of the square matrix multiplication using shared
memory:

.. code-block:: python
  :linenos:

    from numba import cuda, float32

    # Controls threads per block and shared memory usage.
    # The computation will be done on blocks of TPBxTPB elements.
    # TPB should not be larger than 32 in this example
    TPB = 16

    @cuda.jit
    def fast_matmul(A, B, C):
    """Reference: https://stackoverflow.com/a/64198479/13697228
    by @RobertCrovella
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
            sA[ty, tx] = 0
            sB[ty, tx] = 0
            if y < A.shape[0] and (tx+i*TPB) < A.shape[1]:
              sA[ty, tx] = A[y, tx + i * TPB]
            if x < B.shape[1] and (ty+i*TPB) < B.shape[0]:
                sB[ty, tx] = B[ty + i * TPB, x]

            # Wait until all threads finish preloading
            cuda.syncthreads()

            # Computes partial product on the shared memory
            for j in range(TPB):
                tmp += sA[ty, j] * sB[j, tx]

            # Wait until all threads finish computing
            cuda.syncthreads()
        if y < C.shape[0] and x < C.shape[1]:
            C[y, x] = tmp

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
    print(x_h @ y_h)

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

.. note:: For high performance matrix multiplication in CUDA, see also the `CuPy implementation <https://docs.cupy.dev/en/stable/reference/generated/cupy.  matmul.html>`_.

The approach outlined here generalizes to non-square matrix multiplication as follows by adjusting the blockspergrid variable:

Again, here is an example usage:

.. code-block:: python
  :linenos:

  x_h = np.arange(115).reshape([5,23])
  y_h = np.ones([23,7])
  z_h = np.zeros([5,7])

  x_d = cuda.to_device(x_h)
  y_d = cuda.to_device(y_h)
  z_d = cuda.to_device(z_h)

  #TPB must be an integer between 1 and 32
  TPB = 32
  threadsperblock = (TPB, TPB)
  grid_y_max = max(x_h.shape[0],y_h.shape[0])
  grid_x_max = max(x_h.shape[1],y_h.shape[1])
  blockspergrid_x = math.ceil(grid_x_max / threadsperblock[0])
  blockspergrid_y = math.ceil(grid_y_max / threadsperblock[1])
  blockspergrid = (blockspergrid_x, blockspergrid_y)

  fast_matmul[blockspergrid, threadsperblock](x_d, y_d, z_d)
  z_h = z_d.copy_to_host()
  print(z_h)
  print(x_h@y_h)

and a corresponding memory check:

.. code-block:: none

  $ python nonsquare_matmul.py
  ========= CUDA-MEMCHECK
  [[ 253.  253.  253.  253.  253.  253.  253.]
  [ 782.  782.  782.  782.  782.  782.  782.]
  [1311. 1311. 1311. 1311. 1311. 1311. 1311.]
  [1840. 1840. 1840. 1840. 1840. 1840. 1840.]
  [2369. 2369. 2369. 2369. 2369. 2369. 2369.]]
  [[ 253.  253.  253.  253.  253.  253.  253.]
  [ 782.  782.  782.  782.  782.  782.  782.]
  [1311. 1311. 1311. 1311. 1311. 1311. 1311.]
  [1840. 1840. 1840. 1840. 1840. 1840. 1840.]
  [2369. 2369. 2369. 2369. 2369. 2369. 2369.]]
  ========= ERROR SUMMARY: 0 errors
