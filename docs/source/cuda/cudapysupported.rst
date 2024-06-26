========================================
Supported Python features in CUDA Python
========================================

This page lists the Python features supported in the CUDA Python.  This includes
all kernel and device functions compiled with ``@cuda.jit`` and other higher
level Numba decorators that targets the CUDA GPU.

Language
========

Execution Model
---------------

CUDA Python maps directly to the *single-instruction multiple-thread*
execution (SIMT) model of CUDA.  Each instruction is implicitly
executed by multiple threads in parallel.  With this execution model, array
expressions are less useful because we don't want multiple threads to perform
the same task.  Instead, we want threads to perform a task in a cooperative
fashion.

For details please consult the
`CUDA Programming Guide
<http://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model>`_.

Floating Point Error Model
--------------------------

By default, CUDA Python kernels execute with the NumPy error model. In this
model, division by zero raises no exception and instead produces a result of
``inf``, ``-inf`` or ``nan``. This differs from the normal Python error model,
in which division by zero raises a ``ZeroDivisionError``.

When debug is enabled (by passing ``debug=True`` to the
:func:`@cuda.jit <numba.cuda.jit>` decorator), the Python error model is used.
This allows division-by-zero errors during kernel execution to be identified.

Constructs
----------

The following Python constructs are not supported:

* Exception handling (``try .. except``, ``try .. finally``)
* Context management (the ``with`` statement)
* Comprehensions (either list, dict, set or generator comprehensions)
* Generator (any ``yield`` statements)

The ``raise`` and ``assert`` statements are supported, with the following
constraints:

- They can only be used in kernels, not in device functions.
- They only have an effect when ``debug=True`` is passed to the
  :func:`@cuda.jit <numba.cuda.jit>` decorator. This is similar to the behavior
  of the ``assert`` keyword in CUDA C/C++, which is ignored unless compiling
  with device debug turned on.


Printing of strings, integers, and floats is supported, but printing is an
asynchronous operation - in order to ensure that all output is printed after a
kernel launch, it is necessary to call :func:`numba.cuda.synchronize`. Eliding
the call to ``synchronize`` is acceptable, but output from a kernel may appear
during other later driver operations (e.g. subsequent kernel launches, memory
transfers, etc.), or fail to appear before the program execution completes. Up
to 32 arguments may be passed to the ``print`` function - if more are passed
then a format string will be emitted instead and a warning will be produced.
This is due to a general limitation in CUDA printing, as outlined in the
`section on limitations in printing
<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#limitations>`_
in the CUDA C++ Programming Guide.


Recursion
---------

Self-recursive device functions are supported, with the constraint that
recursive calls must have the same argument types as the initial call to
the function. For example, the following form of recursion is supported:

.. code:: python

   @cuda.jit("int64(int64)", device=True)
   def fib(n):
       if n < 2:
           return n
       return fib(n - 1) + fib(n - 2)

(the ``fib`` function always has an ``int64`` argument), whereas the following
is unsupported:

.. code:: python

   # Called with x := int64, y := float64
   @cuda.jit
   def type_change_self(x, y):
       if x > 1 and y > 0:
           return x + type_change_self(x - y, y)
       else:
           return y

The outer call to ``type_change_self`` provides ``(int64, float64)`` arguments,
but the inner call uses ``(float64, float64)`` arguments (because ``x - y`` /
``int64 - float64`` results in a ``float64`` type). Therefore, this function is
unsupported.

Mutual recursion between functions (e.g. where a function ``func1()`` calls
``func2()`` which again calls ``func1()``) is unsupported.

.. note::

   The call stack in CUDA is typically quite limited in size, so it is easier
   to overflow it with recursive calls on CUDA devices than it is on CPUs.

   Stack overflow will result in an Unspecified Launch Failure (ULF) during
   kernel execution.  In order to identify whether a ULF is due to stack
   overflow, programs can be run under `Compute Sanitizer
   <https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html>`_,
   which explicitly states when stack overflow has occurred.

.. _cuda-built-in-types:

Built-in types
===============

The following built-in types support are inherited from CPU nopython mode.

* int
* float
* complex
* bool
* None
* tuple
* Enum, IntEnum

See :ref:`nopython built-in types <pysupported-builtin-types>`.

There is also some very limited support for character sequences (bytes and
unicode strings) used in NumPy arrays. Note that this support can only be used
with CUDA 11.2 onwards.

Built-in functions
==================

The following built-in functions are supported:

* :func:`abs`
* :class:`bool`
* :class:`complex`
* :func:`enumerate`
* :class:`float`
* :class:`int`: only the one-argument form
* :func:`len`
* :func:`min`: only the multiple-argument form
* :func:`max`: only the multiple-argument form
* :func:`pow`
* :class:`range`
* :func:`round`
* :func:`zip`


Standard library modules
========================


``cmath``
---------

The following functions from the :mod:`cmath` module are supported:

* :func:`cmath.acos`
* :func:`cmath.acosh`
* :func:`cmath.asin`
* :func:`cmath.asinh`
* :func:`cmath.atan`
* :func:`cmath.atanh`
* :func:`cmath.cos`
* :func:`cmath.cosh`
* :func:`cmath.exp`
* :func:`cmath.isfinite`
* :func:`cmath.isinf`
* :func:`cmath.isnan`
* :func:`cmath.log`
* :func:`cmath.log10`
* :func:`cmath.phase`
* :func:`cmath.polar`
* :func:`cmath.rect`
* :func:`cmath.sin`
* :func:`cmath.sinh`
* :func:`cmath.sqrt`
* :func:`cmath.tan`
* :func:`cmath.tanh`

``math``
--------

The following functions from the :mod:`math` module are supported:

* :func:`math.acos`
* :func:`math.asin`
* :func:`math.atan`
* :func:`math.acosh`
* :func:`math.asinh`
* :func:`math.atanh`
* :func:`math.cos`
* :func:`math.sin`
* :func:`math.tan`
* :func:`math.hypot`
* :func:`math.cosh`
* :func:`math.sinh`
* :func:`math.tanh`
* :func:`math.atan2`
* :func:`math.erf`
* :func:`math.erfc`
* :func:`math.exp`
* :func:`math.expm1`
* :func:`math.fabs`
* :func:`math.frexp`
* :func:`math.ldexp`
* :func:`math.gamma`
* :func:`math.lgamma`
* :func:`math.log`
* :func:`math.log2`
* :func:`math.log10`
* :func:`math.log1p`
* :func:`math.sqrt`
* :func:`math.remainder`
* :func:`math.pow`
* :func:`math.ceil`
* :func:`math.floor`
* :func:`math.copysign`
* :func:`math.fmod`
* :func:`math.modf`
* :func:`math.isnan`
* :func:`math.isinf`
* :func:`math.isfinite`


``operator``
------------

The following functions from the :mod:`operator` module are supported:

* :func:`operator.add`
* :func:`operator.and_`
* :func:`operator.eq`
* :func:`operator.floordiv`
* :func:`operator.ge`
* :func:`operator.gt`
* :func:`operator.iadd`
* :func:`operator.iand`
* :func:`operator.ifloordiv`
* :func:`operator.ilshift`
* :func:`operator.imod`
* :func:`operator.imul`
* :func:`operator.invert`
* :func:`operator.ior`
* :func:`operator.ipow`
* :func:`operator.irshift`
* :func:`operator.isub`
* :func:`operator.itruediv`
* :func:`operator.ixor`
* :func:`operator.le`
* :func:`operator.lshift`
* :func:`operator.lt`
* :func:`operator.mod`
* :func:`operator.mul`
* :func:`operator.ne`
* :func:`operator.neg`
* :func:`operator.not_`
* :func:`operator.or_`
* :func:`operator.pos`
* :func:`operator.pow`
* :func:`operator.rshift`
* :func:`operator.sub`
* :func:`operator.truediv`
* :func:`operator.xor`

.. _cuda_numpy_support:

NumPy support
=============

Due to the CUDA programming model, dynamic memory allocation inside a kernel is
inefficient and is often not needed.  Numba disallows any memory allocating features.
This disables a large number of NumPy APIs.  For best performance, users should write
code such that each thread is dealing with a single element at a time.

Supported NumPy features:

* accessing `ndarray` attributes `.shape`, `.strides`, `.ndim`, `.size`, etc..
* indexing and slicing works.
* A subset of ufuncs are supported, but the output array must be passed in as a
  positional argument (see :ref:`cuda_ufunc_call_example`). Note that ufuncs
  execute sequentially in each thread - there is no automatic parallelisation
  of ufuncs across threads over the elements of an input array.

  The following ufuncs are supported:

  * :func:`numpy.sin`
  * :func:`numpy.cos`
  * :func:`numpy.tan`
  * :func:`numpy.arcsin`
  * :func:`numpy.arccos`
  * :func:`numpy.arctan`
  * :func:`numpy.arctan2`
  * :func:`numpy.hypot`
  * :func:`numpy.sinh`
  * :func:`numpy.cosh`
  * :func:`numpy.tanh`
  * :func:`numpy.arcsinh`
  * :func:`numpy.arccosh`
  * :func:`numpy.arctanh`
  * :func:`numpy.deg2rad`
  * :func:`numpy.radians`
  * :func:`numpy.rad2deg`
  * :func:`numpy.degrees`
  * :func:`numpy.greater`
  * :func:`numpy.greater_equal`
  * :func:`numpy.less`
  * :func:`numpy.less_equal`
  * :func:`numpy.not_equal`
  * :func:`numpy.equal`
  * :func:`numpy.log`
  * :func:`numpy.log2`
  * :func:`numpy.log10`
  * :func:`numpy.logical_and`
  * :func:`numpy.logical_or`
  * :func:`numpy.logical_xor`
  * :func:`numpy.logical_not`
  * :func:`numpy.maximum`
  * :func:`numpy.minimum`
  * :func:`numpy.fmax`
  * :func:`numpy.fmin`
  * :func:`numpy.bitwise_and`
  * :func:`numpy.bitwise_or`
  * :func:`numpy.bitwise_xor`
  * :func:`numpy.invert`
  * :func:`numpy.bitwise_not`
  * :func:`numpy.left_shift`
  * :func:`numpy.right_shift`

Unsupported NumPy features:

* array creation APIs.
* array methods.
* functions that returns a new array.


CFFI support
============

The ``from_buffer()`` method of ``cffi.FFI`` objects is supported. This is
useful for obtaining a pointer that can be passed to external C / C++ / PTX
functions (see the :ref:`CUDA FFI documentation <cuda_ffi>`).
