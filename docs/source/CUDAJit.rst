Writing CUDA-Python
=========================

The CUDA JIT is a low-level entry point to the CUDA features in Numba.
It translates Python functions into `PTX
<http://en.wikipedia.org/wiki/Parallel_Thread_Execution>`_ code which execute on
the CUDA hardware.  The `jit` decorator is applied to Python functions written 
in our `Python dialect for CUDA <CUDAPySpec.html>`_.
Numba interacts with the `CUDA Driver API 
<http://docs.nvidia.com/cuda/cuda-driver-api/index.html>`_ to load the PTX onto
the CUDA device and execute.


Imports
-------

Most of the CUDA public API for CUDA features are exposed in the
``numba.cuda`` module::

	from numba import cuda

Compiling
-----------

CUDA kernels and device functions are compiled by decorating a Python
function with the jit or `autojit` decorators.

.. autofunction:: numba.cuda.jit

.. autofunction:: numba.cuda.autojit

Thread Identity by CUDA Intrinsics
------------------------------------

A set of CUDA intrinsics is used to identify the current execution thread.
These intrinsics are meaningful inside a CUDA kernel or device function only.
A common pattern to assign the computation of each element in the output array 
to a thread.

For a 1D grid::

    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = tx + bx * bw
    array[i] = something(i)

For a 2D grid::

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    x = tx + bx * bw
    y = ty + by * bh
    array[x, y] = something(x, y)


Since these patterns are so common, there is a shorthand function to produce
the same result.

For a 1D grid::

    i = cuda.grid(1)
    array[i] = something(i)

For a 2D grid::

    x, y = cuda.grid(2)
    array[x, y] = something(x, y)

Memory Transfer
---------------

By default, any NumPy arrays used as argument of a CUDA kernel is transferred
automatically to and from the device.  However, to achieve maximum performance 
and minimizing redundant memory transfer,
user should manage the memory transfer explicitly.

Host->device transfers are asynchronous to the host.
Device->host transfers are synchronous to the host.
If a non-zero `CUDA stream`_ is provided, the transfer becomes asynchronous.

.. autofunction:: numba.cuda.to_device

.. automethod:: numba.cuda.cudadrv.devicearray.DeviceNDArray.copy_to_host

The following are special DeviceNDArray factories:

.. autofunction:: numba.cuda.device_array

.. autofunction:: numba.cuda.pinned_array

.. autofunction:: numba.cuda.mapped_array

Memory Lifetime
-----------------

The live time of a device array is bound to the lifetime of the 
`DeviceNDArray` instance.


CUDA Stream
-----------

A CUDA stream is a command queue for the CUDA device.  By specifying a stream, 
the CUDA API calls become asynchronous, meaning that the call may return before
the command has been completed.  Memory transfer instructions and kernel 
invocation can use CUDA stream::

    stream = cuda.stream()
    devary = cuda.to_device(an_array, stream=stream)
    a_cuda_kernel[griddim, blockdim, stream](devary)
    devary.copy_to_host(an_array, stream=stream)
    # data may not be available in an_array
    stream.synchronize()
    # data available in an_array

.. autofunction:: numba.cuda.stream

.. automethod:: numba.cudadrv.driver.Stream.synchronize

An alternative syntax is available for use with a python context::

	stream = cuda.stream()
	with stream.auto_synchronize():
	    devary = cuda.to_device(an_array, stream=stream)
	    a_cuda_kernel[griddim, blockdim, stream](devary)
	    devary.copy_to_host(an_array, stream=stream)
	# data available in an_array

When the python ``with`` context exits, the stream is automatically synchronized.

Shared Memory
------------------

For maximum performance, a CUDA kernel needs to use shared memory for manual caching of data.  CUDA JIT supports the use of ``cuda.shared.array(shape, dtype)`` for specifying an NumPy-array-like object inside a kernel.

For example:::


    bpg = 50
    tpb = 32
    n = bpg * tpb

    @jit(argtypes=[float32[:,:], float32[:,:], float32[:,:]], target='gpu')
    def cu_square_matrix_mul(A, B, C):
        sA = cuda.shared.array(shape=(tpb, tpb), dtype=float32)
        sB = cuda.shared.array(shape=(tpb, tpb), dtype=float32)

        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        bw = cuda.blockDim.x
        bh = cuda.blockDim.y

        x = tx + bx * bw
        y = ty + by * bh

        acc = 0.
        for i in range(bpg):
            if x < n and y < n:
                sA[ty, tx] = A[y, tx + i * tpb]
                sB[ty, tx] = B[ty + i * tpb, x]

            cuda.syncthreads()

            if x < n and y < n:
                for j in range(tpb):
                    acc += sA[ty, j] * sB[j, tx]

            cuda.syncthreads()

        if x < n and y < n:
            C[y, x] = acc

The equivalent code in CUDA-C would be:

.. code-block:: c

    #define pos2d(Y, X, W) ((Y) * (W) + (X))

    const unsigned int BPG = 50;
    const unsigned int TPB = 32;
    const unsigned int N = BPG * TPB;

    __global__
    void cuMatrixMul(const float A[], const float B[], float C[]){
        __shared__ float sA[TPB * TPB];
        __shared__ float sB[TPB * TPB];

        unsigned int tx = threadIdx.x;
        unsigned int ty = threadIdx.y;
        unsigned int bx = blockIdx.x;
        unsigned int by = blockIdx.y;
        unsigned int bw = blockDim.x;
        unsigned int bh = blockDim.y;

        unsigned int x = tx + bx * bw;
        unsigned int y = ty + by * bh;

        float acc = 0.0;

        for (int i = 0; i < BPG; ++i) {
            if (x < N and y < N) {
                sA[pos2d(ty, tx, TPB)] = A[pos2d(y, tx + i * TPB, N)];
                sB[pos2d(ty, tx, TPB)] = B[pos2d(ty + i * TPB, x, N)];
            }
            __syncthreads();
            if (x < N and y < N) {
                for (int j = 0; j < TPB; ++j) {
                    acc += sA[pos2d(ty, j, TPB)] * sB[pos2d(j, tx, TPB)];
                }
            }
            __syncthreads();
        }

        if (x < N and y < N) {
            C[pos2d(y, x, N)] = acc;
        }
    }




The return value of ``cuda.shared.array`` is a NumPy-array-like object.  The ``shape`` argument  is similar as in NumPy API, with the requirement that it must contain a constant expression.  The `dtype` argument takes Numba types.


Synchronization Primitives
--------------------------

We currently support ``cuda.syncthreads()`` only.  It is the same as ``__syncthreads()`` in CUDA-C.

