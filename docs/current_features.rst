NumbaPro
########

.. contents::
   :maxdepth: 2

Current Features
================

This document describes the current features of NumbaPro.

Fast Vectorize
==============

There are 4 versions of vectorize for building native ufunc kernels from python functions.

They are:

1. Basic vectorize (`numbapro.vectorize.basic`) -- A simple wrapper for vectorize.
2. Stream vectorize (`numbapro.vectorize.stream`) -- A cache optimized version.  (Work in progress)
3. Parallel vectorize (`numbapro.vectorize.parallel`) -- A multi-threaded version of vectorize.
4. CUDA vectorize (`numbapro.vectorize.cuda`) -- A vectorizer that uses Nvidia CUDA GPU.
5. Mini vectorize (`numbapro.vectorize.minivectorize`) -- A vectorizer that works well with mixed
data orders (e.g. mixing C and Fortran ordered arrays).

All these vectorize versions use Numba to compile the functions into LLVM.
Then, NumbaPro wraps and enhances these LLVM functions into ufuncs.
Users may choose different versions of vectorize depending on the complexity of the workload.
Cache optimization, multithreading and GPU computation have their associated overheads.
Therefore, there may not a lot of benefit for smaller workloads.

User API
--------

All vectorizers may be easily imported from `numbapro.vectorize`::

    from numbapro.vectorize import *

The following vectorizer classes are available under the `numbapro.vectorize` namespace:

1. `BasicVectorize`
2. `StreamVectorize`
3. `ParallelVectorize`
4. `CudaVectorize`
5. `MiniVectorize` and `ParallelMiniVectorize`

All their basic usage are similar.  Here, we will use `BasicVectorize` as an example.

Let's say we have a python function to be used as the ufunc kernel::

    def vector_add(a, b):  # the ufunc kernel
        return a + b

First, create a BasicVectorize instance from `vector_add`::

    from numbapro.vectorize.basic import BasicVectorize
    bv = BasicVectorize(vector_add)

Second, add the supporting types of the ufunc using Numba types::

    from numba import *
    bv.add(restype=int32, argtypes=[int32, int32])  # integer
    bv.add(restype=f,     argtypes=[f, f])          # float
    bv.add(restype=d,     argtypes=[d, d])          # double

The code below defines integer, float and double versions of `vector_add`, which
allows the ufunc to be used on arrays of those types.

Finally, create the ufunc::

    vector_add_ufunc = bv.build_ufunc()

`bv.build_ufunc()` returns a python callable representing the ufunc.


Details on Basic Vectorize
--------------------------

This is the simplest vectorize.  It is suitable for small workloads in which the overhead for cache optimization, multithreading or GPU computation is too significant.


Details on Stream Vectorize
---------------------------

The stream vectorize implementation is work-in-progress. It aims to provide a cache optimized version of vectorize.  The current implementation works on a small chunk of data at a time.  It moves data from the array, which may be strided, into a local buffer before processing.  The stream vectorize is work-in-progress because it does not provide speedup, yet.  We may alter the technique to provide better performance gain in the future.

The `StreamVectorize.build_ufunc` method takes a keyword argument for controlling the granularity of the chunking (size of the chunk).  The default is `granularity=32`.


Details on Parallel Vectorize
-----------------------------

The parallel vectorize uses as many thread as is returned by `multiprocessing.cpu_count`.  The works are divided equally among all worker threads.  Once a worker thread completes its assigned works, it switches to work-stealing mode.  It tries to steal work from the end of the workqueue of other threads.  The implementation uses low-level atomic locks.  It requires the processor to have CAS instructions.

Work-stealing is a dynamic scheduling technique that only engages if necessary.  This balances inter-thread communication overhead between fixed scheduling and dynamic scheduling.

Details on CUDA Vectorize
-------------------------

The CUDA vectorize translates the Numba compiled ufunc kernel into Nvidia PTX representation. Then, it uses PyCUDA for access to the CUDA driver and send the PTX to device execution.

Due to the overhead for memory transfer between the host and the device, this is not suitable for small workloads.

Details on Mini Vectorize
-------------------------

The Mini vectorizer is an alternative vectorizer which can provide more stable performance for various
sorts of mixed data layouts by dispatching to different code specializations (for instance tiled
implementations). It may also give better performance than parallel vectorize when the inner
dimension is small or strided.

Fast Generalized Ufunc
======================

Numbapro enables users to write Numpy generalized ufuncs inside Python.
Regular ufuncs kernels created by users only take scalars as input and return scalars as output.
Generalized ufunc user kernels have the ability to take numpy arrays as input and output.
This mean the kernel can be applied on entire sub-arrays of larger arrays, while respecting all
NumPy rules such as broadcasting. For a detailed description of generalized ufuncs,
we refer the reader to http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html.

User API
--------

The fast generalized ufunc feature is implemented in `numbapro.vectorize.gufunc`, which
defines the `GUFuncVectorize` class (also available under `numbapro.vectorize.GUFuncVectorize`.
Its usage is similar to the other ufunc builder described above.
The only difference is that the constructor takes an additional signature of the generalized ufunc.

Example
-------

The following code implements `import numpy.core.umath_tests.matrix_multiply` using NumbaPro::

    def matmulcore(A, B, C):  # the generalized ufunc kernel
        m, n = A.shape
        n, p = B.shape
        for i in range(m):
            for j in range(p):
                C[i, j] = 0
                for k in range(n):
                    C[i, j] += A[i, k] * B[k, j]

    from numbapro.vectorize.gufunc import GUFuncVectorize

    gufunc = GUFuncVectorize(matmulcore, '(m,n),(n,p)->(m,p)')

    # specialize to 32-bit float
    gufunc.add(argtypes=[f[:,:], f[:,:], f[:,:]])

    # build the generalized ufunc
    gufunc = gufunc.build_ufunc()

    matrix_ct = 10
    A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2, 4)
    B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4, 5)

    # compute matrix-matrix multiply for 10 pairs of matrices
    C = gufunc(A, B)


Limitations
-----------

* Does not accept scalar input or output.


