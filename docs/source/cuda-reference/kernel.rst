CUDA Kernel API
===============

Kernel declaration
------------------

The ``@cuda.jit`` decorator is used to create a CUDA dispatcher object that can
be configured and launched:

.. autofunction:: numba.cuda.jit


Dispatcher objects
------------------

The usual syntax for configuring a Dispatcher with a launch configuration uses
subscripting, with the arguments being as in the following:

.. code-block:: python

   # func is some function decorated with @cuda.jit
   func[griddim, blockdim, stream, sharedmem]


The ``griddim`` and ``blockdim`` arguments specify the size of the grid and
thread blocks, and may be either integers or tuples of length up to 3. The
``stream`` parameter is an optional stream on which the kernel will be launched,
and the ``sharedmem`` parameter specifies the size of dynamic shared memory in
bytes.

Subscripting the Dispatcher returns a configuration object that can be called
with the kernel arguments:

.. code-block:: python

   configured = func[griddim, blockdim, stream, sharedmem]
   configured(x, y, z)


However, it is more idiomatic to configure and call the kernel within a single
statement:

.. code-block:: python

   func[griddim, blockdim, stream, sharedmem](x, y, z)

This is similar to launch configuration in CUDA C/C++:

.. code-block:: cuda

   func<<<griddim, blockdim, sharedmem, stream>>>(x, y, z)

.. note:: The order of ``stream`` and ``sharedmem`` are reversed in Numba
   compared to in CUDA C/C++.

Dispatcher objects also provide several utility methods for inspection and
creating a specialized instance:

.. autoclass:: numba.cuda.dispatcher.CUDADispatcher
   :members: inspect_asm, inspect_llvm, inspect_sass, inspect_types,
             get_regs_per_thread, specialize, specialized, extensions, forall,
             get_shared_mem_per_block, get_max_threads_per_block,
             get_const_mem_size, get_local_mem_per_thread



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

    The shape of the grid of blocks, accessed through the attributes ``x``,
    ``y``, and ``z``.

.. attribute:: numba.cuda.laneid

    The thread index in the current warp, as an integer spanning the range
    from 0 inclusive to the :attr:`numba.cuda.warpsize` exclusive.

.. attribute:: numba.cuda.warpsize

    The size in threads of a warp on the GPU. Currently this is always 32.

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

    Returns the value of ``array[idx]`` before storing the new value.
    Behaves like an atomic load.

.. function:: numba.cuda.atomic.sub(array, idx, value)

    Perform ``array[idx] -= value``. Supports int32, int64, float32 and
    float64 only. The ``idx`` argument can be an integer or a tuple of integer
    indices for indexing into multi-dimensional arrays. The number of elements
    in ``idx`` must match the number of dimensions of ``array``.

    Returns the value of ``array[idx]`` before storing the new value.
    Behaves like an atomic load.

.. function:: numba.cuda.atomic.and_(array, idx, value)

    Perform ``array[idx] &= value``. Supports int32, uint32, int64,
    and uint64 only. The ``idx`` argument can be an integer or a tuple of
    integer indices for indexing into multi-dimensional arrays. The number
    of elements in ``idx`` must match the number of dimensions of ``array``.

    Returns the value of ``array[idx]`` before storing the new value.
    Behaves like an atomic load.

.. function:: numba.cuda.atomic.or_(array, idx, value)

    Perform ``array[idx] |= value``. Supports int32, uint32, int64,
    and uint64 only. The ``idx`` argument can be an integer or a tuple of
    integer indices for indexing into multi-dimensional arrays. The number
    of elements in ``idx`` must match the number of dimensions of ``array``.

    Returns the value of ``array[idx]`` before storing the new value.
    Behaves like an atomic load.

.. function:: numba.cuda.atomic.xor(array, idx, value)

    Perform ``array[idx] ^= value``. Supports int32, uint32, int64,
    and uint64 only. The ``idx`` argument can be an integer or a tuple of
    integer indices for indexing into multi-dimensional arrays. The number
    of elements in ``idx`` must match the number of dimensions of ``array``.

    Returns the value of ``array[idx]`` before storing the new value.
    Behaves like an atomic load.

.. function:: numba.cuda.atomic.exch(array, idx, value)

    Perform ``array[idx] = value``. Supports int32, uint32, int64,
    and uint64 only. The ``idx`` argument can be an integer or a tuple of
    integer indices for indexing into multi-dimensional arrays. The number
    of elements in ``idx`` must match the number of dimensions of ``array``.

    Returns the value of ``array[idx]`` before storing the new value.
    Behaves like an atomic load.

.. function:: numba.cuda.atomic.inc(array, idx, value)

    Perform ``array[idx] = (0 if array[idx] >= value else array[idx] + 1)``.
    Supports uint32, and uint64 only. The ``idx`` argument can be an integer
    or a tuple of integer indices for indexing into multi-dimensional arrays.
    The number of elements in ``idx`` must match the number of dimensions of
    ``array``.

    Returns the value of ``array[idx]`` before storing the new value.
    Behaves like an atomic load.

.. function:: numba.cuda.atomic.dec(array, idx, value)

    Perform ``array[idx] =
    (value if (array[idx] == 0) or (array[idx] > value) else array[idx] - 1)``.
    Supports uint32, and uint64 only. The ``idx`` argument can be an integer
    or a tuple of integer indices for indexing into multi-dimensional arrays.
    The number of elements in ``idx`` must match the number of dimensions of
    ``array``.

    Returns the value of ``array[idx]`` before storing the new value.
    Behaves like an atomic load.

.. function:: numba.cuda.atomic.max(array, idx, value)

    Perform ``array[idx] = max(array[idx], value)``. Support int32, int64,
    float32 and float64 only. The ``idx`` argument can be an integer or a
    tuple of integer indices for indexing into multiple dimensional arrays.
    The number of element in ``idx`` must match the number of dimension of
    ``array``.

    Returns the value of ``array[idx]`` before storing the new value.
    Behaves like an atomic load.

.. function:: numba.cuda.atomic.cas(array, idx, old, value)

    Perform ``if array[idx] == old: array[idx] = value``. Supports int32,
    int64, uint32, uint64 indexes only. The ``idx`` argument can be an integer
    or a tuple of integer indices for indexing into multi-dimensional arrays.
    The number of elements in ``idx`` must match the number of dimensions of
    ``array``.

    Returns the value of ``array[idx]`` before storing the new value.
    Behaves like an atomic compare and swap.


.. function:: numba.cuda.syncthreads

    Synchronize all threads in the same thread block.  This function implements
    the same pattern as barriers in traditional multi-threaded programming: this
    function waits until all threads in the block call it, at which point it
    returns control to all its callers.

.. function:: numba.cuda.syncthreads_count(predicate)

    An extension to :attr:`numba.cuda.syncthreads` where the return value is a count
    of the threads where ``predicate`` is true.

.. function:: numba.cuda.syncthreads_and(predicate)

    An extension to :attr:`numba.cuda.syncthreads` where 1 is returned if ``predicate`` is
    true for all threads or 0 otherwise.

.. function:: numba.cuda.syncthreads_or(predicate)

    An extension to :attr:`numba.cuda.syncthreads` where 1 is returned if ``predicate`` is
    true for any thread or 0 otherwise.

    .. warning:: All syncthreads functions must be called by every thread in the
                 thread-block. Falling to do so may result in undefined behavior.


Cooperative Groups
~~~~~~~~~~~~~~~~~~

.. function:: numba.cuda.cg.this_grid()

   Get the current grid group.

   :return: The current grid group
   :rtype: numba.cuda.cg.GridGroup

.. class:: numba.cuda.cg.GridGroup

   A grid group. Users should not construct a GridGroup directly - instead, get
   the current grid group using :func:`cg.this_grid() <numba.cuda.cg.this_grid>`.

   .. method:: sync()

      Synchronize the current grid group.


Memory Fences
~~~~~~~~~~~~~

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


   A memory fence at system level (across GPUs).

Warp Intrinsics
~~~~~~~~~~~~~~~

The argument ``membermask`` is a 32 bit integer mask with each bit
corresponding to a thread in the warp, with 1 meaning the thread is in the
subset of threads within the function call. The ``membermask`` must be all 1 if
the GPU compute capability is below 7.x.

.. function:: numba.cuda.syncwarp(membermask)

   Synchronize a masked subset of the threads in a warp.

.. function:: numba.cuda.all_sync(membermask, predicate)

    If the ``predicate`` is true for all threads in the masked warp, then
    a non-zero value is returned, otherwise 0 is returned.

.. function:: numba.cuda.any_sync(membermask, predicate)

    If the ``predicate`` is true for any thread in the masked warp, then
    a non-zero value is returned, otherwise 0 is returned.

.. function:: numba.cuda.eq_sync(membermask, predicate)

    If the boolean ``predicate`` is the same for all threads in the masked warp,
    then a non-zero value is returned, otherwise 0 is returned.

.. function:: numba.cuda.ballot_sync(membermask, predicate)

    Returns a mask of all threads in the warp whose ``predicate`` is true,
    and are within the given mask.

.. function:: numba.cuda.shfl_sync(membermask, value, src_lane)

    Shuffles ``value`` across the masked warp and returns the ``value``
    from ``src_lane``. If this is outside the warp, then the
    given ``value`` is returned.

.. function:: numba.cuda.shfl_up_sync(membermask, value, delta)

    Shuffles ``value`` across the masked warp and returns the ``value``
    from ``laneid - delta``. If this is outside the warp, then the
    given ``value`` is returned.

.. function:: numba.cuda.shfl_down_sync(membermask, value, delta)

    Shuffles ``value`` across the masked warp and returns the ``value``
    from ``laneid + delta``. If this is outside the warp, then the
    given ``value`` is returned.

.. function:: numba.cuda.shfl_xor_sync(membermask, value, lane_mask)

    Shuffles ``value`` across the masked warp and returns the ``value``
    from ``laneid ^ lane_mask``.

.. function:: numba.cuda.match_any_sync(membermask, value, lane_mask)

    Returns a mask of threads that have same ``value`` as the given ``value``
    from within the masked warp.

.. function:: numba.cuda.match_all_sync(membermask, value, lane_mask)

    Returns a tuple of (mask, pred), where mask is a mask of threads that have
    same ``value`` as the given ``value`` from within the masked warp, if they
    all have the same value, otherwise it is 0. And pred is a boolean of whether
    or not all threads in the mask warp have the same warp.

.. function:: numba.cuda.activemask()

    Returns a 32-bit integer mask of all currently active threads in the
    calling warp. The Nth bit is set if the Nth lane in the warp is active when
    activemask() is called. Inactive threads are represented by 0 bits in the
    returned mask. Threads which have exited the kernel are always marked as
    inactive.

.. function:: numba.cuda.lanemask_lt()

    Returns a 32-bit integer mask of all lanes (including inactive ones) with
    ID less than the current lane.


Integer Intrinsics
~~~~~~~~~~~~~~~~~~

A subset of the CUDA Math API's integer intrinsics are available. For further
documentation, including semantics, please refer to the `CUDA Toolkit
documentation
<https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html>`_.


.. function:: numba.cuda.popc(x)

   Returns the number of bits set in ``x``.

.. function:: numba.cuda.brev(x)

   Returns the reverse of the bit pattern of ``x``. For example, ``0b10110110``
   becomes ``0b01101101``.

.. function:: numba.cuda.clz(x)

   Returns the number of leading zeros in ``x``.

.. function:: numba.cuda.ffs(x)

   Returns the position of the first (least significant) bit set to 1 in ``x``,
   where the least significant bit position is 1. ``ffs(0)`` returns 0.


Floating Point Intrinsics
~~~~~~~~~~~~~~~~~~~~~~~~~

A subset of the CUDA Math API's floating point intrinsics are available. For further
documentation, including semantics, please refer to the `single
<https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html>`_ and
`double <https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html>`_
precision parts of the CUDA Toolkit documentation.


.. function:: numba.cuda.fma

   Perform the fused multiply-add operation. Named after the ``fma`` and ``fmaf`` in
   the C api, but maps to the ``fma.rn.f32`` and ``fma.rn.f64`` (round-to-nearest-even)
   PTX instructions.

.. function:: numba.cuda.cbrt (x)

   Perform the cube root operation, x ** (1/3). Named after the functions
   ``cbrt`` and ``cbrtf`` in the C api. Supports float32, and float64 arguments
   only.

16-bit Floating Point Intrinsics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The functions in the ``cuda.fp16`` module are used to operate on 16-bit
floating point operands. These functions return a 16-bit floating point result.

To determine whether Numba supports compiling code that uses the ``float16``
type in the current configuration, use:

   .. function:: numba.cuda.is_float16_supported ()

   Return ``True`` if 16-bit floats are supported, ``False`` otherwise.

To check whether a device supports ``float16``, use its
:attr:`supports_float16 <numba.cuda.cudadrv.driver.Device.supports_float16>`
attribute.

.. function:: numba.cuda.fp16.hfma (a, b, c)

   Perform the fused multiply-add operation ``(a * b) + c`` on 16-bit
   floating point arguments in round to nearest mode. Maps to the ``fma.rn.f16``
   PTX instruction.

   Returns the 16-bit floating point result of the fused multiply-add.

.. function:: numba.cuda.fp16.hadd (a, b)

   Perform the add operation ``a + b`` on 16-bit floating point arguments in
   round to nearest mode. Maps to the ``add.f16`` PTX instruction.

   Returns the 16-bit floating point result of the addition.

.. function:: numba.cuda.fp16.hsub (a, b)

   Perform the subtract operation ``a - b`` on 16-bit floating point arguments in
   round to nearest mode. Maps to the ``sub.f16`` PTX instruction.

   Returns the 16-bit floating point result of the subtraction.

.. function:: numba.cuda.fp16.hmul (a, b)

   Perform the multiply operation ``a * b`` on 16-bit floating point arguments in
   round to nearest mode. Maps to the ``mul.f16`` PTX instruction.

   Returns the 16-bit floating point result of the multiplication.

.. function:: numba.cuda.fp16.hdiv (a, b)

   Perform the divide operation ``a / b`` on 16-bit floating point arguments in
   round to nearest mode.

   Returns the 16-bit floating point result of the division.

.. function:: numba.cuda.fp16.hneg (a)

   Perform the negation operation ``-a`` on the 16-bit floating point argument.
   Maps to the ``neg.f16`` PTX instruction.

   Returns the 16-bit floating point result of the negation.

.. function:: numba.cuda.fp16.habs (a)

   Perform the absolute value operation ``|a|`` on the 16-bit floating point argument.

   Returns the 16-bit floating point result of the absolute value operation.

.. function:: numba.cuda.fp16.hsin (a)

   Calculates the trigonometry sine function of the 16-bit floating point argument.

   Returns the 16-bit floating point result of the sine operation.

.. function:: numba.cuda.fp16.hcos (a)

   Calculates the trigonometry cosine function of the 16-bit floating point argument.

   Returns the 16-bit floating point result of the cosine operation.

.. function:: numba.cuda.fp16.hlog (a)

   Calculates the natural logarithm of the 16-bit floating point argument.

   Returns the 16-bit floating point result of the natural log operation.

.. function:: numba.cuda.fp16.hlog10 (a)

   Calculates the base 10 logarithm of the 16-bit floating point argument.

   Returns the 16-bit floating point result of the log base 10 operation.

.. function:: numba.cuda.fp16.hlog2 (a)

   Calculates the base 2 logarithm on the 16-bit floating point argument.

   Returns the 16-bit floating point result of the log base 2 operation.

.. function:: numba.cuda.fp16.hexp (a)

   Calculates the natural exponential operation of the 16-bit floating point argument.

   Returns the 16-bit floating point result of the exponential operation.

.. function:: numba.cuda.fp16.hexp10 (a)

   Calculates the base 10 exponential of the 16-bit floating point argument.

   Returns the 16-bit floating point result of the exponential operation.

.. function:: numba.cuda.fp16.hexp2 (a)

   Calculates the base 2 exponential of the 16-bit floating point argument.

   Returns the 16-bit floating point result of the exponential operation.

.. function:: numba.cuda.fp16.hfloor (a)

   Calculates the floor operation, the largest integer less than or equal to ``a``,
   on the 16-bit floating point argument.

   Returns the 16-bit floating point result of the floor operation.

.. function:: numba.cuda.fp16.hceil (a)

   Calculates the ceiling operation, the smallest integer greater than or equal to ``a``,
   on the 16-bit floating point argument.

   Returns the 16-bit floating point result of the ceil operation.

.. function:: numba.cuda.fp16.hsqrt (a)

   Calculates the square root operation of the 16-bit floating point argument.

   Returns the 16-bit floating point result of the square root operation.

.. function:: numba.cuda.fp16.hrsqrt (a)

   Calculates the reciprocal of the square root of the 16-bit floating point argument.

   Returns the 16-bit floating point result of the reciprocal square root operation.

.. function:: numba.cuda.fp16.hrcp (a)

   Calculates the reciprocal of the 16-bit floating point argument.

   Returns the 16-bit floating point result of the reciprocal.

.. function:: numba.cuda.fp16.hrint (a)

   Round the input 16-bit floating point argument to nearest integer value.

   Returns the 16-bit floating point result of the rounding.

.. function:: numba.cuda.fp16.htrunc (a)

   Truncate the input 16-bit floating point argument to the nearest integer
   that does not exceed the input argument in magnitude.

   Returns the 16-bit floating point result of the truncation.

.. function:: numba.cuda.fp16.heq (a, b)

   Perform the comparison operation ``a == b`` on 16-bit floating point arguments.

   Returns a boolean.

.. function:: numba.cuda.fp16.hne (a, b)

   Perform the comparison operation ``a != b`` on 16-bit floating point arguments.

   Returns a boolean.

.. function:: numba.cuda.fp16.hgt (a, b)

   Perform the comparison operation ``a > b`` on 16-bit floating point arguments.

   Returns a boolean.

.. function:: numba.cuda.fp16.hge (a, b)

   Perform the comparison operation ``a >= b`` on 16-bit floating point arguments.

   Returns a boolean.

.. function:: numba.cuda.fp16.hlt (a, b)

   Perform the comparison operation ``a < b`` on 16-bit floating point arguments.

   Returns a boolean.

.. function:: numba.cuda.fp16.hle (a, b)

   Perform the comparison operation ``a <= b`` on 16-bit floating point arguments.

   Returns a boolean.

.. function:: numba.cuda.fp16.hmax (a, b)

   Perform the operation ``a if a > b else b.``

   Returns a 16-bit floating point value.

.. function:: numba.cuda.fp16.hmin (a, b)

   Perform the operation ``a if a < b else b.``

   Returns a 16-bit floating point value.

Control Flow Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

A subset of the CUDA's control flow instructions are directly available as
intrinsics. Avoiding branches is a key way to improve CUDA performance, and
using these intrinsics mean you don't have to rely on the ``nvcc`` optimizer
identifying and removing branches. For further documentation, including
semantics, please refer to the `relevant CUDA Toolkit documentation
<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions>`_.


.. function:: numba.cuda.selp

    Select between two expressions, depending on the value of the first
    argument. Similar to LLVM's ``select`` instruction.


Timer Intrinsics
~~~~~~~~~~~~~~~~

.. function:: numba.cuda.nanosleep(ns)

    Suspends the thread for a sleep duration approximately close to the delay
    ``ns``, specified in nanoseconds.
