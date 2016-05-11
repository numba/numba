.. _numba-types:

====================
Types and signatures
====================

Rationale
=========

As an optimizing compiler, Numba needs to decide on the type of each
variable to generate efficient machine code.  Python's standard types
are not precise enough for that, so we had to develop our own fine-grained
type system.

You will encounter Numba types mainly when trying to inspect the results
of Numba's type inference, for :ref:`debugging <numba-envvars>` or
:ref:`educational <architecture>` purposes.  However, you need to use
types explicitly if compiling code :ref:`ahead-of-time <pycc>`.


Signatures
==========

A signature specifies the type of a function.  Exactly which kind
of signature is allowed depends on the context (:term:`AOT` or :term:`JIT`
compilation), but signatures always involve some representation of Numba
types to specifiy the concrete types for the function's arguments and,
if required, the function's return type.

An example function signature would be the string ``"f8(i4, i4)"``
(or the equivalent ``"float64(int32, int32)"``) which specifies a
function taking two 32-bit integers and returning a double-precision float.


Basic types
===========

The most basic types can be expressed through simple expressions.  The
symbols below refer to attributes of the main ``numba`` module (so if
you read "boolean", it means that symbol can be accessed as ``numba.boolean``).
Many types are available both as a canonical name and a shorthand alias,
following Numpy's conventions.

Numbers
-------

The following table contains the elementary numeric types currently defined
by Numba and their aliases.

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

intc                    --               C int-sized integer
uintc                   --               C int-sized unsigned integer
intp                    --               pointer-sized integer
uintp                   --               pointer-sized unsigned integer

float32                 f4               single-precision floating-point number
float64, double         f8               double-precision floating-point number

complex64               c8               single-precision complex number
complex128              c16              double-precision complex number
===================     =========        ===================================

Arrays
------

The easy way to declare array types is to subscript an elementary type
according to the number of dimensions.  For example a 1-dimension
single-precision array::

   >>> numba.float32[:]
   array(float32, 1d, A)

or a 3-dimension array of the same underlying type::

   >>> numba.float32[:, :, :]
   array(float32, 3d, A)

This syntax defines array types with no particular layout (producing code
that accepts both non-contiguous and contiguous arrays), but you can
specify a particular contiguity by using the ``::1`` index either at
the beginning or the end of the index specification::

   >>> numba.float32[::1]
   array(float32, 1d, C)
   >>> numba.float32[:, :, ::1]
   array(float32, 3d, C)
   >>> numba.float32[::1, :, :]
   array(float32, 3d, F)


Advanced types
==============

For more advanced declarations, you have to explicitly call helper
functions or classes provided by Numba.

.. warning::
   The APIs documented here are not guaranteed to be stable.  Unless
   necessary, it is recommended to let Numba infer argument types by using
   the :ref:`signature-less variant of @jit <jit-lazy>`.

.. A word of note: I only documented those types that can be genuinely
   useful to users, i.e. types that can be passed as parameters to a JIT
   function.  Other types such as tuple are only usable in type inference.


Inference
---------

.. function:: numba.typeof(value)

   Create a Numba type accurately describing the given Python *value*.
   ``None`` is returned if the value isn't supported in :term:`nopython mode`.

   ::

      >>> numba.typeof(np.empty(3))
      array(float64, 1d, C)
      >>> numba.typeof((1, 2.0))
      (int64, float64)
      >>> numba.typeof([0])
      reflected list(int64)


Numpy scalars
-------------

Instead of using :func:`~numba.typeof`, non-trivial scalars such as
structured types can also be constructed programmatically.

.. function:: numba.from_dtype(dtype)

   Create a Numba type corresponding to the given Numpy *dtype*::

      >>> struct_dtype = np.dtype([('row', np.float64), ('col', np.float64)])
      >>> ty = numba.from_dtype(struct_dtype)
      >>> ty
      Record([('row', '<f8'), ('col', '<f8')])
      >>> ty[:, :]
      unaligned array(Record([('row', '<f8'), ('col', '<f8')]), 2d, A)

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
