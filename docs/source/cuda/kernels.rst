
====================
Writing CUDA Kernels
====================

Introduction
============

CUDA has an execution model unlike the traditional sequential model used
for programming CPUs.  In CUDA, the code you write will be executed by
multiple threads at once (often hundreds or thousands).  Your solution will
be modeled by defining a thread hierarchy of *grid*, *blocks* and *threads*.

Numba's CUDA support exposes facilities to declare and manage this
hierarchy of threads.  The facilities are largely similar to those
exposed by NVidia's CUDA C language.

Numba also exposes three kinds of GPU memory: global :ref:`device memory
<cuda-device-memory>` (the large, relatively slow
off-chip memory that's connected to the GPU itself), on-chip
:ref:`shared memory <cuda-shared-memory>` and :ref:`local memory <cuda-local-memory>`.
For all but the simplest algorithms, it is important that you carefully
consider how to use and access memory in order to minimize bandwidth
requirements and contention.


Kernel declaration
==================

A *kernel function* is a GPU function that is meant to be called from CPU
code (*).  It gives it two fundamental characteristics:

* kernels cannot explicitly return a value; all result data must be written
  to an array passed to the function (if computing a scalar, you will
  probably pass a one-element array);

* kernels explicitly declare their thread hierarchy when called: i.e.
  the number of thread blocks and the number of threads per block
  (note that while a kernel is compiled once, it can be called multiple
  times with different block sizes or grid sizes).

At first sight, writing a CUDA kernel with Numba looks very much like
writing a :term:`JIT function` for the CPU::

    @cuda.jit
    def increment_by_one(an_array):
        """
        Increment all array elements by one.
        """
        # code elided here; read further for different implementations

(*) Note: newer CUDA devices support device-side kernel launching; this feature
is called *dynamic parallelism* but Numba does not support it currently)


.. _cuda-kernel-invocation:

Kernel invocation
=================

A kernel is typically launched in the following way::

    threadsperblock = 32
    blockspergrid = (an_array.size + (threadsperblock - 1)) // threadperblock
    increment_by_one[blockspergrid, threadsperblock](an_array)

We notice two steps here:

* Instantiate the kernel proper, by specifying a number of blocks
  (or "blocks per grid"), and a number of threads per block.  The product
  of the two will give the total number of threads launched.  Kernel
  instantiation is done by taking the compiled kernel function
  (here ``increment_by_one``) and indexing it with a tuple of integers.

* Running the kernel, by passing it the input array (and any separate
  output arrays if necessary).  By default, running a kernel is synchronous:
  the function returns when the kernel has finished executing and the
  data is synchronized back.

Choosing the block size
-----------------------

It might seem curious to have a two-level hierarchy when declaring the
number of threads needed by a kernel.  The block size (i.e. number of
threads per block) is often crucial:

* On the software side, the block size determines how many threads
  share a given area of :ref:`shared memory <cuda-shared-memory>`.

* On the hardware side, the block size must be large enough for full
  occupation of execution units; recommendations can be found in the
  `CUDA C Programming Guide`_.

Multi-dimensional blocks and grids
----------------------------------

To help deal with multi-dimensional arrays, CUDA allows you to specify
multi-dimensional blocks and grids.  In the example above, you could
make ``blockspergrid`` and ``threadsperblock`` tuples of one, two
or three integers.  Compared to 1D declarations of equivalent sizes,
this doesn't change anything to the efficiency or behaviour of generated
code, but can help you write your algorithms in a more natural way.


Thread positioning
==================

When running a kernel, the kernel function's code is executed by every
thread once.  It therefore has to know which thread it is in, in order
to know which array element(s) it is responsible for (complex algorithms
may define more complex responsibilities, but the underlying principle
is the same).

One way is for the thread to determines its position in the grid and block
and manually compute the corresponding array position::

    @cuda.jit
    def increment_by_one(an_array):
        # Thread id in a 1D block
        tx = cuda.threadIdx.x
        # Block id in a 1D grid
        ty = cuda.blockIdx.x
        # Block width, i.e. number of threads per block
        bw = cuda.blockDim.x
        # Compute flattened index inside the array
        pos = tx + ty * bw
        if pos < an_array.size:  # Check array boundaries
            an_array[pos] += 1

.. note:: Unless you are sure the block size and grid size is a divisor
   of your array size, you **must** check boundaries as shown above.

:attr:`.threadIdx`, :attr:`.blockIdx`, :attr:`.blockDim` and :attr:`.gridDim`
are special objects provided by the CUDA backend for the sole purpose of
knowing the geometry of the thread hierarchy and the position of the
current thread within that geometry.

These objects can be 1D, 2D or 3D, depending on how the kernel was
:ref:`invoked <cuda-kernel-invocation>`.  To access the value at each
dimension, use the ``x``, ``y`` and ``z`` attributes of these objects,
respectively.

.. attribute:: numba.cuda.threadIdx
   :noindex:

   The thread indices in the current thread block.  For 1D blocks, the index
   (given by the ``x`` attribute) is an integer spanning the range from 0
   inclusive to :attr:`numba.cuda.blockDim` exclusive.  A similar rule
   exists for each dimension when more than one dimension is used.

.. attribute:: numba.cuda.blockDim
   :noindex:

   The shape of the block of threads, as declared when instantiating the
   kernel.  This value is the same for all threads in a given kernel, even
   if they belong to different blocks (i.e. each block is "full").

.. attribute:: numba.cuda.blockIdx
   :noindex:

   The block indices in the grid of threads launched a kernel.  For a 1D grid,
   the index (given by the ``x`` attribute) is an integer spanning the range
   from 0 inclusive to :attr:`numba.cuda.gridDim` exclusive.  A similar rule
   exists for each dimension when more than one dimension is used.

.. attribute:: numba.cuda.gridDim
   :noindex:

   The shape of the grid of blocks, i.e. the total number of blocks launched
   by this kernel invocation, as declared when instantiating the kernel.

Absolute positions
------------------

Simple algorithms will tend to always use thread indices in the
same way as shown in the example above.  Numba provides additional facilities
to automate such calculations:

.. function:: numba.cuda.grid(ndim)
   :noindex:

   Return the absolute position of the current thread in the entire
   grid of blocks.  *ndim* should correspond to the number of dimensions
   declared when instantiating the kernel.  If *ndim* is 1, a single integer
   is returned.  If *ndim* is 2 or 3, a tuple of the given number of
   integers is returned.

.. function:: numba.cuda.gridsize(ndim)
   :noindex:

   Return the absolute size (or shape) in threads of the entire grid of
   blocks.  *ndim* has the same meaning as in :func:`.grid` above.

With these functions, the incrementation example can become::

    @cuda.jit
    def increment_by_one(an_array):
        pos = cuda.grid(1)
        if pos < an_array.size:
            an_array[pos] += 1

The same example for a 2D array and grid of threads would be::

    @cuda.jit
    def increment_a_2D_array(an_array):
        x, y = cuda.grid(2)
        if x < an_array.shape[0] and y < an_array.shape[1]:
           an_array[x, y] += 1

Note the grid computation when instantiating the kernel must still be
done manually, for example::

    from __future__ import division  # for Python 2

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(an_array.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(an_array.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    increment_a_2D_array[blockspergrid, threadsperblock](an_array)


Further Reading
----------------

Please refer to the the `CUDA C Programming Guide`_ for a detailed discussion
of CUDA programming.


.. _CUDA C Programming Guide: http://docs.nvidia.com/cuda/cuda-c-programming-guide
