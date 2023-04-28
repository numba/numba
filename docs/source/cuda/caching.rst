On-disk Kernel Caching
======================

When the ``cache`` keyword argument of the :func:`@cuda.jit <numba.cuda.jit>`
decorator is ``True``, a file-based cache is enabled. This shortens compilation
times when the function was already compiled in a previous invocation.

The cache is maintained in the ``__pycache__`` subdirectory of the directory
containing the source file; if the current user is not allowed to write to it,
the cache implementation falls back to a platform-specific user-wide cache
directory (such as ``$HOME/.cache/numba`` on Unix platforms).


Compute capability considerations
---------------------------------

Separate cache files are maintained for each compute capability. When a cached
kernel is loaded, the compute capability of the device the kernel is first
launched on in the current run is used to determine which version to load.
Therefore, on systems that have multiple GPUs with differing compute
capabilities, the cached versions of kernels are only used for one compute
capability, and recompilation will occur for other compute capabilities.

For example: if a system has two GPUs, one of compute capability 7.5 and one of
8.0, then:

* If a cached kernel is first launched on the CC 7.5 device, then the cached
  version for CC 7.5 is used. If it is subsequently launched on the CC 8.0
  device, a recompilation will occur.
* If in a subsequent run the cached kernel is first launched on the CC 8.0
  device, then the cached version for CC 8.0 is used. A subsequent launch on
  the CC 7.5 device will require a recompilation.

This limitation is not expected to present issues in most practical scenarios,
as multi-GPU production systems tend to have identical GPUs within each node.
