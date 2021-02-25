
.. _cuda-fast-math:

CUDA Fast Math
==============

As noted in :ref:`fast-math`, for certain classes of applications that utilize
floating point, strict IEEE-754 conformance is not required. For this subset of
applications, performance speedups may be possible.

The CUDA target implements :ref:`fast-math` behavior with two differences.

* First, the ``fastmath`` argument to the :func:`@jit decorator
  <numba.cuda.jit>` is limited to the values ``True`` and ``False``.
  When ``True``, the following optimizations are enabled:

  - Flushing of denormals to zero.
  - Use of a fast approximation to the square root function.
  - Use of a fast approximation to the division operation.
  - Contraction of multiply and add operations into single fused multiply-add
    operations.

  See the `documentation for nvvmCompileProgram <https://docs.nvidia.com/cuda/libnvvm-api/group__compilation.html#group__compilation_1g76ac1e23f5d0e2240e78be0e63450346>`_ for more details of these optimizations.

* Secondly, calls to a subset of math module functions on ``float32`` operands
  will be implemented using fast approximate implementations from the libdevice
  library.

  - :func:`math.cos`: Implemented using `__nv_fast_cosf <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_cosf.html>`_.
  - :func:`math.sin`: Implemented using `__nv_fast_sinf <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_sinf.html>`_.
  - :func:`math.tan`: Implemented using `__nv_fast_tanf <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_tanf.html>`_.
  - :func:`math.exp`: Implemented using `__nv_fast_expf <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_expf.html>`_.
  - :func:`math.log2`: Implemented using `__nv_fast_log2f <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_log2f.html>`_.
  - :func:`math.log10`: Implemented using `__nv_fast_log10f <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_log10f.html>`_.
  - :func:`math.log`: Implemented using `__nv_fast_logf <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_logf.html>`_.
  - :func:`math.pow`: Implemented using `__nv_fast_powf <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_powf.html>`_.
