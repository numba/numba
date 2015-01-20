Writing CUDA Kernels
====================

CUDA has a very different execution model.
It defines a thread hierarchy of *grid*, *blocks* and *threads*.
When calling a CUDA kernel, a grid of blocks of threads is created.
The kernel code is executed by every thread once.
In the simplest case, each thread determines its position in the grid and block
and maps the position to an array index::

    from numba import cuda

    @cuda.jit                  # mark a function to be compile to a kernel
    def increment_by_one(an_array):
        tx = cuda.threadIdx.x  # thread id in a 1D block
        ty = cuda.blockIdx.x   # block id in a 1D grid
        bw = cuda.blockDim.x   # block width
        pos = tx + ty * bw     # flattened position
        if pos < an_array.size:
            an_array[pos] += 1

``cuda.threadIdx``, ``cuda.blockIdx``, ``cuda.blockDim`` and ``cuda.gridDim``
are special objects for the CUDA backend that holds the thread index in the
block, block index in a grid, block dimension and grid dimension.
These objects can be up 1D, 2D or 3D.
To access the value at each dimension, use the ``x``, ``y`` and ``z``
attributes of these objects.

To launch the kernel or create a grid, do::

    threadperblock = 32
    blockpergrid = (an_array.size + (threadperblock - 1)) // threadperblock
    increment_by_one[blockpergrid, threadperblock](an_array)

The ``threadperblock`` and ``blockpergrid`` can be an int or a tuple of 1, 2 or
3 ints corresponding to the x, y, z dimension of the block and grid.

Because the size of the array is not always divisible by the number of threads
per block, we could be launching more threads than there are array elements.
The kernel must check if the thread position in within the array bound.

Kernel Characteristics
----------------------

A kernel is unlike normal functions.  It cannot not return any value other than
``None`` (a bare return statement returns None).  Results should be store
into an array passed as an output parameter.

In older CUDA, a kernel can only be launched by the host.  Newer CUDA devices
support device side kernel launching.  This feature is called *dynamic
parallelism* but Numba does not support it currently.


Data Independent Parallelism
----------------------------

Using CUDA for data independent task is the easiest way to take advantage of
the GPU computation power.  Many NumPy functions perofrm data independent
elementwise operations.  The easiest way is to assign one thread per element
in the array.  Since this is very common, numba defines ``cuda.grid(dim)`` for
computing the flattened indices.  If ``dim == 1``, it returns an int
that represents the index along the x dimension of a grid.  If ``dim == 2``,
it returns a 2-tuple containing the ``(x, y)`` indices.  For example::

    @cuda.jit
    def simpler_increment_by_one(an_array):
        pos = cuda.grid(1)
        if pos < an_array.size:
            an_array[pos] += 1


For a 2D array and a 2D grid::

    @cuda.jit
    def increment_a_2D_array(an_array):
        x, y = cuda.grid(1)
        if x < an_array.shape[0] and y < an_array.shape[0]:
        an_array[x, y] += 1

The corresponding launch code::


    threadperblock = (16, 16)
    blockpergrid_x = (an_array.shape[0] + (threadperblock[0] - 1)) // threadperblock[0]
    blockpergrid_y = (an_array.shape[1] + (threadperblock[1] - 1)) // threadperblock[1]
    blockpergrid = (blockpergrid_x, blockpergrid_y)
    increment_by_one[blockpergrid, threadperblock](an_array)


To write a square matrix multiplication kernel::

    @cuda.jit
    def matmul(A, B, C):
        """Performs square matrix multiplication of C = A * B
        """
        i, j = cuda.grid(2)
        if i < C.shape[0] and j < C.shape[1]:
            tmp = 0
            for k in range(A.shape[1]):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp




Cooperative Parallelism
-----------------------

The previously shown ``matmul()`` kernel is very simple but not efficient
because the same element is being loaded from the device memory multiple times.
It will be faster if we use a blocked algorithm to reduce the access to the
device memory.
CUDA provides a fast *shared memory* for threads in a block to cooperately
compute on a task.
The following implements a faster version of the square matrix multiplication
using the shared memory::

    from numba import cuda, float32

    TPB = 16   # controls thread per block and shared memory usage

    @cuda.jit
    def fast_matmul(A, B, C):
        # Defines an array in the shared memory
        # The size and type of the array must be known at compile time
        sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
        sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

        x, y = cuda.grid(2)

        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bpg = cuda.gridDim.x  # block per grid

        tmp = 0.
        for i in range(bpg):
            # Preload into shared memory
            if x < n and y < n:
                sA[tx, ty] = A[x, ty + i * TPB]
                sB[tx, ty] = B[tx + i * TPB, y]

            # Wait until all threads finish preloading
            cuda.syncthreads()

            # Computes on the shared memory
            if x < n and y < n:
                for k in range(TPB):
                    tmp += sA[tx, k] * sB[k, ty]

            # Wait until all threads finish computing
            cuda.syncthreads()

        if x < n and y < n:
            C[x, y] = tmp

Because the shared memory is a limited resources, the code preloads small
block at a time from the input arrays.  Then, it calls ``cuda.syncthreads()``
to wait until all threads has finished preloading and before doing the
computation on the shared memory.  It synchronizes again after the
computation to ensure all threads have finished with the data in shared memory.


Further Reading
----------------

Please refer to the the `CUDA C Programming Guide`_ for a detailed discussion
of CUDA programming.



.. Links

.. _CUDA C Programming Guide: http://docs.nvidia.com/cuda/cuda-c-programming-guide
