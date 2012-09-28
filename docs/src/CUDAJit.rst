-------------
CUDA JIT
-------------

**Note: CUDA JIT is still in experimental stages.  Computation speeds may have unexpected results .**


CUDA JIT translates Python function into a CUDA kernel.  It uses translated code from Numba and converts it to `PTX <http://en.wikipedia.org/wiki/Parallel_Thread_Execution>`_.  Numbapro interacts with the CUDA Runtime Libraries to load the PTX onto the CUDA device and execute.  

Imports
-------

::

	import numpy as np
	from numba import *
	from numbapro import cuda



CUDA Kernel Definition
----------------------

A CUDA kernel is a special function that executes on a CUDA-enabled GPU device.  The kernel is executed once for every thread.  It does not return any value.  Results must be written to an array argument.  By default, all array arguments are copied-back to the host upon completion of the kernel.

::

	cuda.jit(argtypes=[f4[:], f4[:], f4[:]])
	def cuda_sum(a, b, c):
		tid = cuda.threadIdx.x
		blkid = cuda.blockIdx.x
		blkdim = cuda.blockDim.x
		i = tid + blkid * blkdim
		c[i] = a[i] + b[i]


CUDA JIT enhances Numba translation by recognizing CUDA intrinsics for `threadIdx`, `blockIdx`, `blockDim` and `gridIdx`.  These intrinsics are defined inside the `numbapro.cuda` module.

Similar to `numba.decorators.jit`, argument types are defined in `argtypes` for `cuda.jit`.  Since a CUDA kernel does not return any value, there are no `restype`.

To invoke the CUDA kernel, it must be configured for the grid and block dimensions.

::

	griddim = 10, 1
	blockdim = 32, 1, 1
	cuda_sum.configure(griddim, blockdim)

Above, we configured the kernel to use 10 blocks and 32 threads per block.

Lastly, we call cuda_sum with three NumPy arrays as arguments.

:: 

	a = np.array(np.random.random(320), dtype=np.float32)
	b = np.array(np.random.random(320), dtype=np.float32)
	c = np.empty_like(a)
	cuda_sum(a, b, c)
	
**Note: All arrays are passed to the device without casting even if the array type does not match the signature of the CUDA kernel.  It is important to ensure all arguments have the correct type.**

