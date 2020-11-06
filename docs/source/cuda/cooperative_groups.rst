==================
Cooperative Groups
==================

Supported features
------------------

Numba's Cooperative Groups support presently provides grid groups and grid
synchronization, along with cooperative kernel launches.

Cooperative groups are supported on Linux, and Windows for devices in `TCC
mode <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tesla-compute-cluster-mode-for-windows>`_.

Using Grid Groups
-----------------

To get the current grid group, use the :meth:`cg.this_grid()
<numba.cuda.cg.this_grid>` function:

.. code-block:: python

   g = cuda.cg.this_grid()

Synchronizing the grid is done with the :meth:`sync()
<numba.cuda.cg.GridGroup.sync>` method of the grid group:

.. code-block:: python

   g.sync()


Cooperative Launches
--------------------

Unlike the CUDA C/C++ API, a cooperative launch is invoked using the same syntax
as a normal kernel launch - Numba automatically determines whether a cooperative
launch is required based on whether a grid group is synchronized in the kernel.

The grid size limit for a cooperative launch is more restrictive than for a
normal launch - the grid must be no larger than the maximum number of active
blocks on the device on which it is launched. To get maximum grid size for a
cooperative launch of a kernel with a given block size and dynamic shared
memory requirement, use the ``max_cooperative_grid_blocks()`` method of kernel
definitions:

.. automethod:: numba.cuda.compiler._Kernel.max_cooperative_grid_blocks

This can be used to ensure that the kernel is launched with no more than the
maximum number of blocks. Exceeding the maximum number of blocks for the
cooperative launch will result in a ``CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE``
error. 


Applications and Example
------------------------

Grid group synchronization can be used to implement a global barrier across all
threads in the grid - applications of this include a global reduction to a
single value, or looping over rows of a large matrix sequentially using the
entire grid to operate on column elements in parallel.

In the following example, rows are written sequentially by the grid. Each thread
in the grid reads a value from the previous row written by it's *opposite*
thread. A grid sync is needed to ensure that threads in the grid don't run ahead
of threads in other blocks, or fail to see updates from their opposite thread.

.. code-block:: python

   from numba import cuda, int32, void
   import numpy as np

   @cuda.jit(void(int32[:,::1]))
   def sequential_rows(M):
       col = cuda.grid(1)
       g = cuda.cg.this_grid()

       rows = M.shape[0]
       cols = M.shape[1]

       for row in range(1, rows):
           opposite = cols - col - 1
           # Each row's elements are one greater than the previous row
           M[row, col] = M[row - 1, opposite] + 1
           # Wait until all threads have written their column element,
           # and that the write is visible to all other threads
           g.sync()

   # Empty input data
   A = np.zeros((1024, 1024), dtype=np.int32)
   # A somewhat arbitrary choice (one warp), but generally smaller block sizes
   # allow more blocks to be launched (noting that other limitations on
   # occupancy apply such as shared memory size)
   blockdim = 32
   griddim = A.shape[1] // blockdim

   # Kernel launch - this is implicitly a cooperative launch
   sequential_rows[griddim, blockdim](A)

   # Sanity check - are the results what we expect?
   reference = np.tile(np.arange(1024), (1024, 1)).T
   np.testing.assert_equal(A, reference)

   # What do the results look like?
   print(A)

   # [[   0    0    0 ...    0    0    0]
   #  [   1    1    1 ...    1    1    1]
   #  [   2    2    2 ...    2    2    2]
   #  ...
   #  [1021 1021 1021 ... 1021 1021 1021]
   #  [1022 1022 1022 ... 1022 1022 1022]
   #  [1023 1023 1023 ... 1023 1023 1023]]

The maximum grid size for ``sequential_rows`` can be enquired using:


.. code-block:: python

   defn = sequential_rows.definition
   max_blocks = defn.max_cooperative_grid_blocks(blockdim)
   print(max_blocks)
   # 1152 (e.g. on Quadro RTX 8000 with Numba 0.52.1 and CUDA 11.0)
