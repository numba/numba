CUDA Bindings
=============

Numba supports two bindings to the CUDA Driver APIs: its own internal bindings
based on ctypes, and the official `NVIDIA CUDA Python bindings
<https://nvidia.github.io/cuda-python/>`_. Functionality is equivalent between
the two bindings.

The internal bindings are used by default. If the NVIDIA bindings are installed,
then they can be used by setting the environment variable
``NUMBA_CUDA_USE_NVIDIA_BINDING`` to ``1`` prior to the import of Numba. Once
Numba has been imported, the selected binding cannot be changed.


Per-Thread Default Streams
--------------------------

Responsibility for handling Per-Thread Default Streams (PTDS) is delegated to
the NVIDIA bindings when they are in use. To use PTDS with the NVIDIA bindings,
set the environment variable ``CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM`` to
``1`` instead of Numba's environmnent variable
:envvar:`NUMBA_CUDA_PER_THREAD_DEFAULT_STREAM`.

.. seealso::

   The `Default Stream section
   <https://nvidia.github.io/cuda-python/release/11.6.0-notes.html#default-stream>`_
   in the NVIDIA Bindings documentation.


Roadmap
-------

In Numba 0.56, the NVIDIA Bindings will be used by default, if they are
installed.

In future versions of Numba:

- The internal bindings will be deprecated.
- The internal bindings will be removed.

At present, no specific release is planned for the deprecation or removal of
the internal bindings.
