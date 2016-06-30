
.. _cudafaq:

=================================================
CUDA Frequently Asked Questions
=================================================

nvprof reports "No kernels were profiled"
-----------------------------------------

When using the ``nvprof`` tool to profile Numba jitted code for the CUDA
target, the output contains ``No kernels were profiled`` but there are clearly
running kernels present, what is going on?

This is quite likely due to the profiling data not being flushed on program
exit, see the `NVIDIA CUDA documentation
<http://docs.nvidia.com/cuda/profiler-users-guide/#flush-profile-data>`_ for
details. To fix this simply add a call to ``numba.cuda.profile_stop()`` prior
to the exit point in your program (or whereever you want to stop profiling).
For more on CUDA profiling support in Numba, see :ref:`cuda-profiling`.
