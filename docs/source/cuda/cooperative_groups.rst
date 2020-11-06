==================
Cooperative Groups
==================

Supported features
------------------

Numba's Cooperative Groups support presently provides grid groups and grid
synchronization, along with cooperative kernel launches.

Cooperative groups are supported on Linux, and Windows for devices in `TCC
mode
<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tesla-compute-cluster-mode-for-windows>`_.
Cooperative Groups also require the CUDA Device Runtime library, ``cudadevrt``,
to be available - for conda default channel-installed CUDA toolkit packages, it
is only available in versions 10.2 onwards. System-installed toolkits (e.g. from
NVIDIA distribution packages or runfiles) all include ``cudadevrt``.

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

First we'll define our kernel:

.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_cg.py
   :language: python
   :caption: from ``test_grid_sync`` of ``numba/cuda/tests/doc_example/test_cg.py``
   :start-after: magictoken.ex_grid_sync_kernel.begin
   :end-before: magictoken.ex_grid_sync_kernel.end
   :dedent: 8
   :linenos:

Then create some empty input data and determine the grid and block sizes:

.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_cg.py
   :language: python
   :caption: from ``test_grid_sync`` of ``numba/cuda/tests/doc_example/test_cg.py``
   :start-after: magictoken.ex_grid_sync_data.begin
   :end-before: magictoken.ex_grid_sync_data.end
   :dedent: 8
   :linenos:

Finally we launch the kernel and print the result:

.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_cg.py
   :language: python
   :caption: from ``test_grid_sync`` of ``numba/cuda/tests/doc_example/test_cg.py``
   :start-after: magictoken.ex_grid_sync_launch.begin
   :end-before: magictoken.ex_grid_sync_launch.end
   :dedent: 8
   :linenos:


The maximum grid size for ``sequential_rows`` can be enquired using:


.. code-block:: python

   defn = sequential_rows.definition
   max_blocks = defn.max_cooperative_grid_blocks(blockdim)
   print(max_blocks)
   # 1152 (e.g. on Quadro RTX 8000 with Numba 0.52.1 and CUDA 11.0)
