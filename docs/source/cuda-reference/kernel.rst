CUDA Kernel API
===============

Kernel declaration
------------------

The ``@cuda.jit`` decorator is used to create a CUDA kernel:

.. autofunction:: numba.cuda.jit

.. autoclass:: numba.cuda.compiler.AutoJitCUDAKernel
   :members: inspect_asm, inspect_llvm, inspect_types, specialize

Individual specialized kernels are instances of
:class:`numba.cuda.compiler.CUDAKernel`:

.. autoclass:: numba.cuda.compiler.CUDAKernel
   :members: bind, ptx, device, inspect_llvm, inspect_asm, inspect_types

Intrinsic Attributes and Functions
----------------------------------

The remainder of the attributes and functions in this section may only be called
from within a CUDA Kernel.

Thread Indexing
~~~~~~~~~~~~~~~

.. autofunction:: numba.cuda.threadIdx
.. autofunction:: numba.cuda.blockIdx
.. autofunction:: numba.cuda.blockDim
.. autofunction:: numba.cuda.gridDim

.. function:: numba.cuda.grid(ndim)

   Return the absolute position of the current thread in the entire
   grid of blocks.  *ndim* should correspond to the number of dimensions
   declared when instantiating the kernel.  If *ndim* is 1, a single integer
   is returned.  If *ndim* is 2 or 3, a tuple of the given number of
   integers is returned.

   Computation of the first integer is as follows::

      cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

   and is similar for the other two indices, but using the ``y`` and ``z``
   attributes.

.. function:: numba.cuda.gridsize(ndim)

   Return the absolute size (or shape) in threads of the entire grid of
   blocks. *ndim* should correspond to the number of dimensions declared when
   instantiating the kernel.

   Computation of the first integer is as follows::

       cuda.blockDim.x * cuda.gridDim.x

   and is similar for the other two indices, but using the ``y`` and ``z``
   attributes.

Memory Management
~~~~~~~~~~~~~~~~~

.. autoclass:: numba.cuda.shared
   :members: array
.. autoclass:: numba.cuda.local
   :members: array
.. autoclass:: numba.cuda.const
   :members: array_like

Synchronization and Atomic Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: numba.cuda.atomic
   :members: add, max
.. autofunction:: numba.cuda.syncthreads
