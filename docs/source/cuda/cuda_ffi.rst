
.. _cuda_ffi:

Calling foreign functions from Python kernels
=============================================

Python kernels can call device functions written in other languages. CUDA C/C++,
PTX, and binary objects (cubins, fat binaries, etc.) are directly supported;
sources in other languages must be compiled to PTX first. The constituent parts
of a Python kernel call to a foreign device function are:

- The device function implementation in a foreign language (e.g. CUDA C).
- A declaration of the device function in Python.
- A kernel that links with and calls the foreign function.

.. _device-function-abi:

Device function ABI
-------------------

Numba's ABI for calling device functions defines the following prototype in
C/C++:

.. code:: C

   extern "C"
   __device__ int
   function(
     T* return_value,
     ...
   );


Components of the prototype are as follows:

- ``extern "C"`` is used to prevent name-mangling so that it is easy to declare
  the function in Python. It can be removed, but then the mangled name must be
  used in the declaration of the function in Python.
- ``__device__`` is required to define the function as a device function.
- The return value is always of type ``int``, and is used to signal whether a
  Python exception occurred. Since Python exceptions don't occur in foreign
  functions, this should always be set to 0 by the callee.
- The first argument is a pointer to the return value of type ``T``, which is
  allocated in the local address space [#f1]_ and passed in by the caller. If
  the function returns a value, the pointee should be set by the callee to
  store the return value.
- Subsequent arguments should match the types and order of arguments passed to
  the function from the Python kernel.

Functions written in other languages must compile to PTX that conforms to this
prototype specification.

A function that accepts two floats and returns a float would have the following
prototype:

.. code:: C

   extern "C"
   __device__ int
   mul_f32_f32(
     float* return_value,
     float x,
     float y
   );

.. rubric:: Notes

.. [#f1] Care must be taken to ensure that any operations on the return value
         are applicable to data in the local address space.  Some operations,
         such as atomics, cannot be performed on data in the local address
         space.

Declaration in Python
---------------------

To declare a foreign device function in Python, use :func:`declare_device()
<numba.cuda.declare_device>`:

.. autofunction:: numba.cuda.declare_device

The returned descriptor name need not match the name of the foreign function.
For example, when:

.. code::

   mul = cuda.declare_device('mul_f32_f32', 'float32(float32, float32)')

is declared, calling ``mul(a, b)`` inside a kernel will translate into a call to
``mul_f32_f32(a, b)`` in the compiled code.

Passing pointers
----------------

Numba's calling convention requires multiple values to be passed for array
arguments. These include the data pointer along with shape, stride, and other
information. This is incompatible with the expectations of most C/C++ functions,
which generally only expect a pointer to the data. To align the calling
conventions between C device code and Python kernels it is necessary to declare
array arguments using C pointer types.

For example, a function with the following prototype:

.. literalinclude:: ../../../numba/cuda/tests/doc_examples/ffi/functions.cu
   :language: C
   :caption: ``numba/cuda/tests/doc_examples/ffi/functions.cu``
   :start-after: magictoken.ex_sum_reduce_proto.begin
   :end-before: magictoken.ex_sum_reduce_proto.end
   :linenos:

would be declared as follows:

.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_ffi.py
   :language: python
   :caption: from ``test_ex_from_buffer`` in ``numba/cuda/tests/doc_examples/test_ffi.py``
   :start-after: magictoken.ex_from_buffer_decl.begin
   :end-before: magictoken.ex_from_buffer_decl.end
   :dedent: 8
   :linenos:

To obtain a pointer to array data for passing to foreign functions, use the
``from_buffer()`` method of a ``cffi.FFI`` instance. For example, a kernel using
the ``sum_reduce`` function could be defined as:

.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_ffi.py
   :language: python
   :caption: from ``test_ex_from_buffer`` in ``numba/cuda/tests/doc_examples/test_ffi.py``
   :start-after: magictoken.ex_from_buffer_kernel.begin
   :end-before: magictoken.ex_from_buffer_kernel.end
   :dedent: 8
   :linenos:

where ``result`` and ``array`` are both arrays of ``float32`` data.

Linking and Calling functions
-----------------------------

The ``link`` keyword argument of the :func:`@cuda.jit <numba.cuda.jit>`
decorator accepts a list of file names specified by absolute path or a path
relative to the current working directory. Files whose name ends in ``.cu``
will be compiled with the `NVIDIA Runtime Compiler (NVRTC)
<https://docs.nvidia.com/cuda/nvrtc/index.html>`_ and linked into the kernel as
PTX; other files will be passed directly to the CUDA Linker.

For example, the following kernel calls the ``mul()`` function declared above
with the implementation ``mul_f32_f32()`` in a file called ``functions.cu``:

.. code::

   @cuda.jit(link=['functions.cu'])
   def multiply_vectors(r, x, y):
       i = cuda.grid(1)

       if i < len(r):
           r[i] = mul(x[i], y[i])


C/C++ Support
-------------

Support for compiling and linking of CUDA C/C++ code is provided through the use
of NVRTC subject to the following considerations:

- It is only available when using the NVIDIA Bindings. See
  :envvar:`NUMBA_CUDA_USE_NVIDIA_BINDING`.
- A suitable version of the NVRTC library for the installed version of the
  NVIDIA CUDA Bindings must be available.
- The CUDA include path is assumed by default to be ``/usr/local/cuda/include``
  on Linux and ``$env:CUDA_PATH\include`` on Windows. It can be modified using
  the environment variable :envvar:`NUMBA_CUDA_INCLUDE_PATH`.
- The CUDA include directory will be made available to NVRTC on the include
  path; additional includes are not supported.


Complete Example
----------------

This example demonstrates calling a foreign function written in CUDA C to
multiply pairs of numbers from two arrays.

The foreign function is written as follows:

.. literalinclude:: ../../../numba/cuda/tests/doc_examples/ffi/functions.cu
   :language: C
   :caption: ``numba/cuda/tests/doc_examples/ffi/functions.cu``
   :start-after: magictoken.ex_mul_f32_f32.begin
   :end-before: magictoken.ex_mul_f32_f32.end
   :linenos:

The Python code and kernel are:

.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_ffi.py
   :language: python
   :caption: from ``test_ex_linking_cu`` in ``numba/cuda/tests/doc_examples/test_ffi.py``
   :start-after: magictoken.ex_linking_cu.begin
   :end-before: magictoken.ex_linking_cu.end
   :dedent: 8
   :linenos:

.. note::

  The example above is minimal in order to illustrate a foreign function call -
  it would not be expected to be particularly performant due to the small grid
  and light workload of the foreign function.
