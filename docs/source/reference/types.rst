.. _numba-types:

===========
Numba Types
===========

Basic Types
===========

"Basic" Numba types can be expressed through simple expressions.  The
symbols below refer to attributes of the main ``numba`` module (so if
you read "boolean", it means that symbol can be accessed as ``numba.boolean``).

Numbers
-------

The following table contains the elementary numeric types currently defined
by Numba, and their various aliases.

===================     =========        ===================================
Type name(s)            Shorthand        Comments
===================     =========        ===================================
boolean                 b1               represented as a byte
uint8, byte             u1               8-bit unsigned byte
uint16                  u2               16-bit unsigned integer
uint32                  u4               32-bit unsigned integer
uint64                  u8               64-bit unsigned integer

int8, char              i1               8-bit signed byte
int16                   i2               16-bit signed integer
int32                   i4               32-bit signed integer
int64                   i8               64-bit signed integer

float32                 f4               float32
float64, double         f8               float64

complex64               c8               single-precision complex number
complex128              c16              double-precision complex number
===================     =========        ===================================

Arrays
------

The easy way to declare array types is to subscript an elementary type
according to the number of dimensions.  For example a 1-dimension
single-precision array::

   >>> numba.float32[:]
   array(float32, 1d, A, nonconst)

or a 3-dimension array of the same underlying type::

   >>> numba.float32[:,:,:]
   array(float32, 3d, A, nonconst)

However, this is not enough to express all possibilities, such as a particular
contiguity or a structured array.


Advanced Types
==============

For more advanced declarations, you have to use constructors provided
by Numba.

.. warning::
   The APIs documented here are not guaranteed to be stable.  Unless
   necessary, it is recommended to let Numba infer argument types by using
   the :ref:`signature-less variant of @jit <jit-lazy>`.

.. A word of note: I only documented those types that can be genuinely
   useful to users, i.e. types that can be passed as parameters to a JIT
   function.  Other types such as tuple are only usable in type inference.


Numpy scalars
-------------

Non-trivial scalars such as structured types need to be constructed
programmatically.

.. function:: numba.from_dtype(dtype)

   Create a Numba type corresponding to the given Numpy *dtype*::

      >>> struct_dtype = np.dtype([('row', np.float64), ('col', np.float64)])
      >>> numba.from_dtype(struct_dtype)
      Record([('row', '<f8'), ('col', '<f8')])

.. class:: numba.types.NPDatetime(unit)

   Create a Numba type for Numpy datetimes of the given *unit*.  *unit*
   should be a string amongst the codes recognized by Numpy (e.g.
   ``Y``, ``M``, ``D``, etc.).

.. class:: numba.types.NPTimedelta(unit)

   Create a Numba type for Numpy timedeltas of the given *unit*.  *unit*
   should be a string amongst the codes recognized by Numpy (e.g.
   ``Y``, ``M``, ``D``, etc.).

   .. seealso::
      Numpy `datetime units <http://docs.scipy.org/doc/numpy/reference/arrays.datetime.html#datetime-units>`_.


Arrays
------

.. class:: numba.types.Array(dtype, ndim, layout)

   Create an array type.  *dtype* should be a Numba type.  *ndim* is the
   number of dimensions of the array (a positive integer).  *layout*
   is a string giving the layout of the array: ``A`` means any layout, ``C``
   means C-contiguous and ``F`` means Fortran-contiguous.


Optional types
--------------

.. class:: numba.optional(typ)

   Create an optional type based on the underlying Numba type *typ*.
   The optional type will allow any value of either *typ* or :const:`None`.

   ::

      >>> @jit((optional(intp),))
      ... def f(x):
      ...     return x is not None
      ...
      >>> f(0)
      True
      >>> f(None)
      False
