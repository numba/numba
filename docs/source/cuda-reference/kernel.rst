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

.. attribute:: numba.cuda.threadIdx

    The thread indices in the current thread block, accessed through the
    attributes ``x``, ``y``, and ``z``. Each index is an integer spanning the
    range from 0 inclusive to the corresponding value of the attribute in
    :attr:`numba.cuda.blockDim` exclusive.

.. attribute:: numba.cuda.blockIdx

    The block indices in the grid of thread blocks, accessed through the
    attributes ``x``, ``y``, and ``z``. Each index is an integer spanning the
    range from 0 inclusive to the corresponding value of the attribute in
    :attr:`numba.cuda.gridDim` exclusive.

.. attribute:: numba.cuda.blockDim

    The shape of a block of threads, as declared when instantiating the
    kernel.  This value is the same for all threads in a given kernel, even
    if they belong to different blocks (i.e. each block is "full").

.. attribute:: numba.cuda.gridDim

    The shape of the grid of blocks, accressed through the attributes ``x``,
    ``y``, and ``z``.

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

.. function:: numba.cuda.shared.array(shape, dtype)

   Creates an array in the local memory space of the CUDA kernel with
   the given ``shape`` and ``dtype``.

   Returns an array with its content uninitialized.

   .. note:: All threads in the same thread block sees the same array.

.. function:: numba.cuda.local.array(shape, dtype)

   Creates an array in the local memory space of the CUDA kernel with the
   given ``shape`` and ``dtype``.

   Returns an array with its content uninitialized.

   .. note:: Each thread sees a unique array.

.. function:: numba.cuda.const.array_like(ary)

   Copies the ``ary`` into constant memory space on the CUDA kernel at compile
   time.

   Returns an array like the ``ary`` argument.

   .. note:: All threads and blocks see the same array.

Synchronization and Atomic Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: numba.cuda.atomic.add(array, idx, value)

    Perform ``array[idx] += value``. Support int32, int64, float32 and
    float64 only. The ``idx`` argument can be an integer or a tuple of integer
    indices for indexing into multiple dimensional arrays. The number of element
    in ``idx`` must match the number of dimension of ``array``.

    Returns the value of ``array[idx]`` before the storing the new value.
    Behaves like an atomic load.

.. function:: numba.cuda.atomic.max(array, idx, value)

    Perform ``array[idx] = max(array[idx], value)``. Support int32, int64,
    float32 and float64 only. The ``idx`` argument can be an integer or a
    tuple of integer indices for indexing into multiple dimensional arrays.
    The number of element in ``idx`` must match the number of dimension of
    ``array``.

    Returns the value of ``array[idx]`` before the storing the new value.
    Behaves like an atomic load.


.. function:: numba.cuda.syncthreads

    Synchronize all threads in the same thread block.  This function implements
    the same pattern as barriers in traditional multi-threaded programming: this
    function waits until all threads in the block call it, at which point it
    returns control to all its callers.

    .. warning:: Must be called by every thread in the thread-block. Falling to
                 do so may result in undefined behavior.

Memory Fences
~~~~~~~~~~~~~~

The memory fences are used to guarantee the effect of memory operations
are visible by other threads within the same thread-block, the same GPU device,
and the same system (across GPUs on global memory). Memory loads and stores
are guaranteed to not move across the memory fences by optimization passes.

.. warning:: The memory fences are considered to be advanced API and most
             usercases should use the thread barrier (e.g. ``syncthreads()``).



.. function:: numba.cuda.threadfence

   A memory fence at device level (within the GPU).

.. function:: numba.cuda.threadfence_block

   A memory fence at thread block level.

.. function:: numba.cuda.threadfence_system

   A memory fence at system level (across GPUs)
