.. _CUDA_int: 

=====================================
CUDA Programming Introduction
=====================================

Numba now contains preliminary support for CUDA programming. Numba will
eventually provide multiple entry points for programmers of different levels
of expertise on CUDA. For now, Numba provides a Python dialect
for low-level programming on the CUDA hardware. It provides full control over
the hardware for fine tunning the performance of CUDA kernels.

A Very Brief Introduction to CUDA
----------------------------------

A `CUDA GPU <https://developer.nvidia.com/what-cuda>`_ contains one or more `streaming multiprocessors` (SMs). Each SM is
a manycore processor that is optimized for high throughput.  The `manycore`
architecture is very different from the common multicore CPU architecture.
Instead of having a large cache and complex logic for instruction level 
optimization, a manycore processor achieves high throughput by executing many
threads in parallel on many simpler cores.  It overcomes latency due to cache
miss or long operations by using zero-cost context switching.  It is common
to launch a CUDA kernel with hundreds or thousands of threads to keep the
GPU busy.

The CUDA programming model is similar to the SIMD vector model in
modern CPUs.  A CUDA SM schedules the same instruction from a *warp* 
of 32-threads at each issuing cycle.
The advantage of CUDA is that the programmer does not need to
handle the divergence of execution path in a warp, whereas a SIMD
programmer would be required to properly mask and shuffle the vectors.
The CUDA model decouples the data structure from the program logic.

To know more about CUDA, please refer to `NVIDIA CUDA-C Programming Guide
<http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>`_.



