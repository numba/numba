-------------
CUDA JIT
-------------

**Note: CUDA JIT is still in experimental stages.  Computation speeds may have unexpected results.**

**Warning: CUDA devices with compute capability less than 1.3 do not support double precision arithmetic.**

CUDA JIT translates Python function into a CUDA kernel.  It uses translated code from Numba and converts it to `PTX <http://en.wikipedia.org/wiki/Parallel_Thread_Execution>`_.  NumbaPro interacts with the CUDA Driver Libraries to load the PTX onto the CUDA device and execute.

Imports
-------

::

	import numpy as np
	from numbapro import *
	from numbapro import cuda


CUDA Kernel Definition
----------------------

A CUDA kernel is a special function that executes on a CUDA-enabled GPU device.
The kernel is executed once for every thread.  It does not return any value.
Results must be written to an array argument.  By default, all array arguments are copied
back to the host upon completion of the kernel.

::

	@jit(argtypes=[f4[:], f4[:], f4[:]], target='gpu')
	def cuda_sum(a, b, c):
		tid = cuda.threadIdx.x
		blkid = cuda.blockIdx.x
		blkdim = cuda.blockDim.x
		i = tid + blkid * blkdim
		c[i] = a[i] + b[i]

CUDA JIT enhances Numba translation by recognizing CUDA-C intrinsics, including threadIdx, blockIdx, blockDim, gridDim. All intrinsics are defined inside the `numbapro.cuda` module.

Since a CUDA kernel does not return any value, there is no `restype` for `jit` when the target is 'gpu'.

To invoke the CUDA kernel, it must be configured for the grid and block dimensions. By default, gridDim and blockDim are (1, 1, 1).

::

	griddim = 10, 1
	blockdim = 32, 1, 1
	cuda_sum_configured = cuda_sum[griddim, blockdim]

Above, we configured the kernel to use 10 blocks and 32 threads per block.

Lastly, we call `cuda_sum_configured` with three NumPy arrays as arguments::

	a = np.array(np.random.random(320), dtype=np.float32)
	b = np.array(np.random.random(320), dtype=np.float32)
	c = np.empty_like(a)
	cuda_sum_configured(a, b, c)

Users can also do the configuration and calling together::

	cuda_sum[griddim, blockdim](a, b, c)

This style looks closer to CUDA-C kernel<<<griddim, blockdim>>>(…).

**Note: All arrays are passed to the device without casting even if the array type does not match the signature of the CUDA kernel.  It is important to ensure all arguments have the correct type.**

Transferring Numpy Arrays to Device
------------------------------------

Numpy arrays can be transferred to the device by::

	device_array = cuda.to_device(array)

To retrieve the data, do::

	device_array.to_host()

This call copies the device memory back to the data buffer of `array`.

`device_array` is of type `DeviceNDArray`, which is a subclass of `numpy.ndarray`.  After the call, `device_array` contains the same buffer as `array`.  The lifetime of the device memory is tied to the `DeviceNDArray` instance.  When the `DeviceNDArray` is released, the device memory is also released.

CUDA Stream
-----------

A CUDA stream is a command queue for the CUDA device.  By specifying a stream, the CUDA API call become asynchronous, meaning that the call may return before the command has been completed.  Memory transfer instructions and kernel invocation can use CUDA stream::

	stream = cuda.stream()
	devary = cuda.to_device(an_array, stream=stream)
	a_cuda_kernel[griddim, blockdim, stream](devary)
	cuda.to_host(devary, stream=stream)
	stream.synchronize()
	# data available in an_array

Use `stream.synchronize()` to ensure all commands in the stream has been completed.

An alternative syntax is available for use with a python context::

	stream = cuda.stream()
	with stream.auto_synchronize():
	    devary = cuda.to_device(an_array, stream=stream)
	    a_cuda_kernel[griddim, blockdim, stream](devary)
	    cuda.to_host(devary)
	# data available in an_array

When the python "with" context exits, the stream is automatically synchronized.

Shared Memory
------------------

For maximum performance, a CUDA kernel needs to use shared memory for manual caching of data.  CUDA JIT supports the use of `cuda.shared.array(shape, dtype)` for specifying an numpy-array-like object inside a kernel.

For example:::


    bpg = 50
    tpb = 32
    n = bpg * tpb

    @jit(argtypes=[f4[:,:], f4[:,:], f4[:,:]], target='gpu')
    def cu_square_matrix_mul(A, B, C):
        sA = cuda.shared.array(shape=(tpb, tpb), dtype=f4)
        sB = cuda.shared.array(shape=(tpb, tpb), dtype=f4)

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

The same code in CUDA-C will be:::

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




The return value of `cuda.shared.array` is a numpy-array-like object.  The `shape` argument  is similar as in Numpy API, with the requirement that it must contain a constant expression.  The `dtype` argument takes Numba types.


Synchronization Primitives
--------------------------

We currently support the `cuda.syncthreads()` only.  It is the same as `__syncthreads()` in CUDA-C.
