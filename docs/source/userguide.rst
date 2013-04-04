************
User Guide
************

Numba compiles Python code with LLVM to code which can be natively executed
at runtime. This happens by decorating Python functions, which allows users
to create native functions for different input types, or to create them
on the fly::

    @jit('f8(f8[:])')
    def sum1d(my_double_array):
        sum = 0.0
        for i in range(my_double_array.shape[0]):
            sum += my_double_array[i]
        return sum

To make the above example work for any compatible input types automatically,
we can create a function that specializes automatically::

    @autojit
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

Unsigned integer counterparts are available under the name ``uint8``
etc.   Also, short-names are available with the style '<char>N' where
char is 'b', 'i', 'u', 'f', and 'c' for boolean, integer, unsigned,
float and complex types respectively with 'N' indicating the number of
bytes in the type.    Thus, f8 is equivalent to float64, and c16 is
equivalent to double complex.

Native platform-dependent types are also available under names such as
``int_``, ``short``, ``ulonglong``, etc.

Types are names that can be imported from the numba namespace.
Alternatively, they can be specified in strings in the jit decorator. 

The jit decorator can take keyword arguments: restype, and argtypes to
specify the function signature.  Alternatively, the signature can be
expressed by passing a single argument to jit either as a string as
shown above or directly (assuming the type names have been imported
from the numba module)::

   from numba import f8, jit

   @jit(f8(f8[:]))
   def sum(arr):
       ...

Notice how the argument types are passed in as arguments to the return
type treated as a python function.    Previously, this same syntax was
used but embedded in a string which avoids having to import f8 from
numba directly.

Specifying Arrays
-----------------
Arrays may be specified strided or C or Fortran contiguous. For instance,
``float[:, :]`` specifies a strided 2D array of floats. ``float[:, ::1]``
specifies that the array is C contiguous (row-major), and ``float[::1, :]``
specifies that the array is Fortran contiguous (column-major).

Examples
========
For get a better feel of what numba can do, see :ref:`Examples <examples>`.

.. toctree::
   :maxdepth: 1

   examples.rst

