
====================
Writing HSA Kernels
====================

Introduction
============

HSA provides an execution model similar to OpenCL.  Instructions are executed
in parallel by a group of hardware threads.  In some way, this is similar to
*single-instruction-multiple-data* (SIMD) model but with the convenience that
the fine-grain scheduling is hidden from the programmer instead of programming
with SIMD vectors as a data structure.  In HSA, the code you write will be
executed by multiple threads at once (often hundreds or thousands).  Your
solution will
be modeled by defining a thread hierarchy of *grid*, *workgroup* and
*workitem*.

Numba's HSA support exposes facilities to declare and manage this
hierarchy of threads.


Introduction for CUDA Programmers
==================================

HSA execution model is similar to CUDA.  The main difference will be the
shared memory model employed by HSA so that there are no device memory.  The
GPU hardware uses the machine's main memory (or host memory in
CUDA term) directly.  Therefore, you will not need ``to_device()`` and
``copy_to_host()`` in HSA programming.

Here's a quick mapping of the CUDA terms to HSA:
* workitem is CUDA threads
* workgroup is CUDA thread block
* grid is CUDA grid


Kernel declaration
==================

A *kernel function* is a GPU function that is meant to be called from CPU
code.  It gives it two fundamental characteristics:

* kernels cannot explicitly return a value; all result data must be written
  to an array passed to the function (if computing a scalar, you will
  probably pass a one-element array);

* kernels explicitly declare their thread hierarchy when called: i.e.
  the number of workgroups and the number of workitems per workgroup
  (note that while a kernel is compiled once, it can be called multiple
  times with different workgroup sizes or grid sizes).

At first sight, writing a HSA kernel with Numba looks very much like
writing a :term:`JIT function` for the CPU::

    @hsa.jit
    def increment_by_one(an_array):
        """
        Increment all array elements by one.
        """
        # code elided here; read further for different implementations


.. _hsa-kernel-invocation:

Kernel invocation
=================

A kernel is typically launched in the following way::

    itempergroup = 32
    groupperrange = (an_array.size + (itempergroup - 1)) // itempergroup
    increment_by_one[groupperrange, itempergroup](an_array)

We notice two steps here:

* Instantiate the kernel proper, by specifying a number of workgroup
  (or "workgroup per grid"), and a number of workitems per workgroup.  The
  product of the two will give the total number of workitem launched.  Kernel
  instantiation is done by taking the compiled kernel function
  (here ``increment_by_one``) and indexing it with a tuple of integers.

* Running the kernel, by passing it the input array (and any separate
  output arrays if necessary).  By default, running a kernel is synchronous:
  the function returns when the kernel has finished executing and the
  data is synchronized back.

Choosing the workgroup size
---------------------------

It might seem curious to have a two-level hierarchy when declaring the
number of workitem needed by a kernel.  The workgroup size (i.e. number of
workitem per workgroup) is often crucial:

* On the software side, the workgroup size determines how many threads
  share a given area of :ref:`shared memory <hsa-shared-memory>`.
* On the hardware side, the workgroup size must be large enough for full
   occupation of execution units.

Multi-dimensional workgroup and grid
---------------------------------------

To help deal with multi-dimensional arrays, HSA allows you to specify
multi-dimensional workgroups and grids.  In the example above, you could
make ``itempergroup`` and ``groupperrange`` tuples of one, two
or three integers.  Compared to 1D declarations of equivalent sizes,
this doesn't change anything to the efficiency or behaviour of generated
code, but can help you write your algorithms in a more natural way.


WorkItem positioning
====================

When running a kernel, the kernel function's code is executed by every
thread once.  It therefore has to know which thread it is in, in order
to know which array element(s) it is responsible for (complex algorithms
may define more complex responsibilities, but the underlying principle
is the same).

One way is for the thread to determines its position in the grid and
workgroup and manually compute the corresponding array position::

    @hsa.jit
    def increment_by_one(an_array):
        # workitem id in a 1D workgroup
        tx = hsa.get_local_id(0)
        # workgroup id in a 1D grid
        ty = hsa.get_group_id(0)
        # workgroup size, i.e. number of workitem per workgroup
        bw = hsa.get_local_size(0)
        # Compute flattened index inside the array
        pos = tx + ty * bw
        # The above is equivalent to pos = hsa.get_global_id(0)
        if pos < an_array.size:  # Check array boundaries
            an_array[pos] += 1

.. note:: Unless you are sure the workgroup size and grid size is a divisor
   of your array size, you **must** check boundaries as shown above.

:func:`.get_local_id`, :func:`.get_local_size`, :func:`.get_group_id` and
:func:`.get_global_id` are special functions provided by the HSA backend for
the sole purpose of knowing the geometry of the thread hierarchy and the
position of the current workitem within that geometry.

.. function:: numba.hsa.get_local_id(dim)

   Takes the index of the dimension being queried

   Returns local workitem ID in the the current workgroup for the given
   dimension. For 1D workgroup, the index is an integer spanning the range
   from 0 inclusive to :func:`numba.hsa.get_local_size` exclusive.

.. function:: numba.hsa.get_local_size(dim)

   Takes the index of the dimension being queried

   Returns the size of the workgroup at the given dimension.
   The value is declared when instantiating the kernel.
   This value is the same for all workitems in a given kernel,
   even if they belong to different workgroups (i.e. each workgroups is "full").

.. function:: numba.hsa.get_group_id(dim)

   Takes the index of the dimension being queried

   Returns the workgroup ID in the grid of workgroup launched a kernel.

.. function:: numba.hsa.get_global_id(dim)

   Takes the index of the dimension being queried

   Returns the global workitem ID for the given dimension.  Unlike `numba.hsa
   .get_local_id()`, this number is unique for all workitems in a grid.


