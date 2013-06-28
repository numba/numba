.. NumbaPro documentation master file, created by
   sphinx-quickstart on Wed Aug 29 09:01:25 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


NumbaPro
========

NumbaPro is an enhanced version of Numba which adds premium features and
functionality that allow developers to rapidly create optimized code that integrates well with NumPy.

With NumbaPro, Python developers can define NumPy ufuncs and generalized ufuncs
in Python, which are compiled to machine code dynamically and loaded on the fly.
Additionally, NumbaPro offers developers the ability to target multicore and
GPU architectures with Python code for both ufuncs and general-purpose code.

Finally, with NumbaPro, high-level array-expressions (slicing, vectorized
math, reductions, linear-algebra operations, etc.) can be compiled directy to
machine code providing the fastest code using all the information available
about the calculation

For targeting the GPU, NumbaPro can either do the work automatically, doing
its best to optimize the code for the GPU architecture.  Alternatively,
CUDA-based API is provided for writing CUDA code specifically in Python for
ultimate control of the hardware (with thread and block identities).

Getting Started
---------------

Let's start with a simple function to add together all the pairwise values in two NumPy arrays.
Asking NumbaPro to compile this python function to vectorized machine code for execution
on the CPU is as simple as adding a single line of code (invoked via a decorator on the
function)::

    from numbapro import vectorize, float32

    @vectorize([float32(float32, float32)], target='cpu')
    def sum(a, b):
        return a + b

    # Invoke like:  result_array = sum(big_input_1, big_input_2)

Similarly, one can instead target the GPU for execution of the same python function by
modifying a single line in the above example::

    @vectorize([float32(float32, float32)], target='gpu')

Targeting the GPU for execution introduces the potential for numerous GPU-specific
optimizations so as a starting point for more complex scenarios, one can also target
the GPU with NumbaPro via its Just-In-Time (JIT) compiler::

    from numbapro import cuda, float32

    @cuda.jit(argtypes=[float32[:], float32[:], float32[:]])
    def sum(a, b, result):
        i = cuda.grid(1)
        result[i] = a[i] + b[i]

    # Invoke like:  sum[grid_dim, block_dim](big_input_1, big_input_2, result_array)


User Guide
----------

Major features of NumbaPro include support for (parallel) NumPy ufuncs and gufuncs,
CUDA support for GPU execution and a multi-threaded parallel range.


.. toctree::
   :maxdepth: 1

   install
   ufuncs
   generalizedufuncs
   prange
   CUDAPySpec
   CUDAufunc
   CUDAJit
   cudalib
   CUDADevice
   CUDASupport

Additional examples:
`GitHub repo of NumbaPro examples <https://github.com/ContinuumIO/numbapro-examples>`_

.. Developer Guide
   ---------------

Requirements
------------

* python 2.6 or 2.7 (support is not yet available for 3.x)
* LLVM (>= 3.2)
* nVidia CUDA SDK (>= 5.5rc)

Python modules:

* llvmpy (>= 0.12.0)
* numba 0.9

Release Notes
-------------

.. toctree::
    :maxdepth: 1

    release-notes

License Agreement
-----------------

.. toctree::
    :maxdepth: 1

    eula_numbapro

.. Indices and tables
   -------------------

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`

