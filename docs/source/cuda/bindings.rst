CUDA Bindings
=============

Numba supports two bindings to the CUDA Driver APIs: its own internal bindings
based on ctypes, and the official `NVIDIA CUDA Python bindings
<https://nvidia.github.io/cuda-python/>`_. Functionality is equivalent between
the two bindings, with two exceptions:

* the NVIDIA bindings presently do not support Per-Thread Default Streams
  (PTDS), and an exception will be raised on import if PTDS is enabled along
  with the NVIDIA bindings.
* The profiling APIs are not available with the NVIDIA bindings.

The internal bindings are used by default. If the NVIDIA bindings are installed,
then they can be used by setting the environment variable
``NUMBA_CUDA_USE_NVIDIA_BINDING`` to ``1`` prior to the import of Numba. Once
Numba has been imported, the selected binding cannot be changed.


Roadmap
-------

In future versions of Numba:

- The NVIDIA Bindings will be used by default, if they are installed.
- The internal bindings will be deprecated.
- The internal bindings will be removed.

It is expected that the NVIDIA bindings will be the default in Numba 0.56; at
present, no specific release is planned for the deprecation or removal of the
internal bindings.
