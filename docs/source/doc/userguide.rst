************
User Guide
************

Numba compiles Python code with LLVM to code which can be natively executed
at runtime. This happens by decorating Python functions, which allows users
to create native functions for different input types, or to create them
on the fly::

    from numba import *

    @jit(restype=double, argtypes=[double[:, :]])
    def sum1d(my_double_array):
        sum = 0.0
        for i in range(my_double_array.shape[0]):
            sum += my_double_array[i]
        return sum

To make the above example work for any compatile input types automatically,
we can create a function that specializes automatically::

    @autojit()
    def sum1d(my_array):
        ...

Specifying Types
================
User elementary types are summarized in the table below and can be found
in the ``numba`` namespace. These types
can be further used to specify arrays in a similar manner to Cython's
memoryviews.

==========  ===================
Type Name   Result Type
==========  ===================
float\_     float32
double      float64
longdouble  float128

char        signed char
int8        int8 (char)
int16       int16
int32       int32
int64       int64

complex64   float complex
complex128  double complex
complex256  long double complex
==========  ===================

Unsigned integer counterparts are available under the name ``uint8`` etc.

Specifying Arrays
-----------------
Arrays may be specified strided or C or Fortran contiguous. For instance,
``float[:, :]`` specifies a strided 2D array of floats. ``float[:, ::1]``
specifies that the array is C contiguous (row-major), and ``float[::1, :]``
specifies that the array is Fortran contiguous (column-major).

Translator Backends
===================
Then ``autojit`` decorator takes ``backend`` as an optional argument, which
may be set to **bytecode** or **ast**. Both backends currently have different
capabilities, but the next release plans for the ast backend to supersede
the bytecode backend. The default is **bytecode**.

For instance the bytecode backend supports complex
numbers, whereas the ast backend has limited support for Python objects (
conversions to and from native data and objects, attribute access and calling
functions).

.. NOTE:: Currently the bytecode translator does not support returning arrays
          from the function. The ast backend does handle this.

Examples
========
For get a better feel of what numba can do, see :ref:`Examples <examples>`.

.. toctree::
   :maxdepth: 1

    examples.rst

