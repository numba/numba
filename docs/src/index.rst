.. NumbaPro documentation master file, created by
   sphinx-quickstart on Wed Aug 29 09:01:25 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


NumbaPro
========

`Get started with the NumbaPro Quick Start [pdf] <https://store.continuum.io/static/img/NumbaPro-QuickStart.pdf>`_

`NumbaPro` is an enhanced version of Numba_ which adds premium features and
functionality that allow developers to rapidly create optimized code that 
integrates well with NumPy_.

With NumbaPro, Python developers can define NumPy ufuncs_ and `generalized ufuncs`_ (gufuncs)
in Python, which are compiled to machine code dynamically and loaded on the fly.
Additionally, NumbaPro offers developers the ability to target multicore and
GPU architectures with Python code for both ufuncs and general-purpose code.

For targeting the GPU, NumbaPro can either do the work automatically, doing
its best to optimize the code for the GPU architecture.  Alternatively,
CUDA-based API is provided for writing :ref:`CUDA <CUDA_int>` code specifically in Python for
ultimate control of the hardware (with thread and block identities).

Getting Started
---------------

Let's start with a simple function to add together all the pairwise values in two NumPy arrays.
Asking NumbaPro to compile this Python function to vectorized machine code for execution
on the CPU is as simple as adding a single line of code (invoked via a decorator on the
function):

.. testcode::

    from numbapro import vectorize

    @vectorize(['float32(float32, float32)'], target='cpu')
    def sum(a, b):
        return a + b


Similarly, one can instead target the GPU for execution of the same Python function by
modifying a single line in the above example:

.. code-block:: python

    @vectorize(['float32(float32, float32)'], target='gpu')

Targeting the GPU for execution introduces the potential for numerous GPU-specific
optimizations so as a starting point for more complex scenarios, one can also target
the GPU with NumbaPro via its `Just-In-Time` (JIT) compiler:

.. testcode::

    from numbapro import cuda

    @cuda.jit('void(float32[:], float32[:], float32[:])')
    def sum(a, b, result):
        i = cuda.grid(1)   # equals to threadIdx.x + blockIdx.x * blockDim.x
        result[i] = a[i] + b[i]

    # Invoke like:  sum[grid_dim, block_dim](big_input_1, big_input_2, result_array)


Features
---------

Here's a list of highlighted features:

* Portable data-parallel programming through ufuncs and gufuncs for single core CPU, multicore CPU and GPU
* Bindings to CUDA libraries: cuRAND, cuBLAS, cuFFT
* Python CUDA programming for maximum control of hardware resources



User Guide
----------

New users should first read the installation manual:

.. toctree::
   :maxdepth: 2

   install

Quick Start:

.. toctree::
   :maxdepth: 1

   decorators

High-level APIs for CPU/GPU:

.. toctree::
   :maxdepth: 1

   ufuncs
   generalizedufuncs
   
Python CUDA Programming

.. toctree::
   :maxdepth: 1

   CUDAintro
   CUDAufunc
   CUDAJit
   cudalib
   CUDADevice
   CUDASupport
   CUDAPySpec

Learn by Examples
------------------

The developer team maintains a public `GitHub repository of examples`_.
Many examples are designed to show off the potential performance gain by using
GPUs.


Requirements
------------

* Python 2.6 or 2.7 (support is not yet available for 3.x)
* LLVM (>= 3.2)
* Latest NVIDIA CUDA driver

Python modules:

* llvmpy (>= 0.12.0)
* numba 0.10.2

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

.. _Numba: http://docs.continuum.io/numba/index.html
.. _NumPy:  http://www.numpy.org
.. _`GitHub repository of examples`: https://github.com/ContinuumIO/numbapro-examples
.. _ufuncs: http://docs.scipy.org/doc/numpy/reference/ufuncs.html
.. _`generalized ufuncs`: http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
