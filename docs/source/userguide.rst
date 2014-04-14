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

    @jit
    def sum1d(my_array):
        ...

Specifying Types
================
User elementary types are summarized in the table below and can be found
in the ``numba`` namespace. These types
can be further used to specify arrays in a similar manner to Cython's
memoryviews.

==========  =====  ===================
Type Name   Alias  Result Type
==========  =====  ===================
boolean     b1     uint8 (char)
bool\_      b1     uint8 (char)

byte        u1     unsigned char
uint8       u1     uint8 (char)
uint16      u2     uint16
uint32      u4     uint32
uint64      u8     uint64

char        i1     signed char
int8        i1     int8 (char)
int16       i2     int16
int32       i4     int32
int64       i8     int64

float\_     f4     float32
float32     f4     float32
double      f8     float64
float64     f8     float64

complex64   c8     float complex
complex128  c16    double complex
==========  =====  ===================

Native platform-dependent types are also available under names such as
``int_``, ``short``, ``ulonglong``, etc.

Types are names that can be imported from the numba namespace.
Alternatively, they can be specified in strings in the jit decorator. 

The function signature of the function to compile can be
expressed by passing a single argument to jit either as a string as
shown above or directly (assuming the type names have been imported
from the numba module)::

   from numba import f8, jit

   @jit('f8(f8[:])')
   def sum(arr):
       ...

   @jit(f8(f8[:]))
   def sum(arr):
       ...

In the first example, the argument and return types are embedded in a
string which avoids having to import f8 from numba.
In the second example, the argument types are passed in as
arguments to the return type treated as a python function. 

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

