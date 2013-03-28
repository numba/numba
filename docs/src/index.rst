.. NumbaPro documentation master file, created by
   sphinx-quickstart on Wed Aug 29 09:01:25 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


NumbaPro*
=========

NumbaPro is an enhanced version of Numba which adds premium features and
functionality that allow developers to rapidly create optimized code that integrates well with NumPy.

With NumbaPro Python developers can define NumPy ufuncs and generalized ufuncs
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

GPU support is rapidly improving but still an area where you may encounter
difficulties.   Please let us know if you have any trouble with our GPU
support.

See `release notes <releases.html>`_ for a summary of changes in each release.

Current Features
----------------

Current features of NumbaPro include support for (parallel) numpy ufuncs and gufuncs,
CUDA support and a multi-threaded parallel range.

Functionality of numbapro is made available through the numbapro namespace, as is
supposed to act as a drop-in replacement for numba::

    import numbapro as numba

.. NOTE:: Although we do not recommend using import star, we will often do so
          in our examples for brevity (``from numbapro import *``)

.. toctree::
   :maxdepth: 1

   ufuncs
   generalizedufuncs
   cu
   CUDAJit
   CUDAufunc
   CUDADevice
   CUDASupport
   cudalib
   prange


Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

