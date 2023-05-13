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
types to specify the concrete types for the function's arguments and,
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
following NumPy's conventions.

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
ssize_t                 --               C ssize_t
size_t                  --               C size_t

float32                 f4               single-precision floating-point number
float64, double         f8               double-precision floating-point number

complex64               c8               single-precision complex number
complex128              c16              double-precision complex number
===================     =========        ===================================

Arrays
------

The easy way to declare :class:`~numba.types.Array` types is to subscript an
elementary type according to the number of dimensions. For example a 
1-dimension single-precision array::

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

This style of type declaration is supported within Numba compiled-functions,
e.g. declaring the type of a :ref:`typed.List <feature-typed-list>`.::

    from numba import njit, types, typed

    @njit
    def example():
        return typed.List.empty_list(types.float64[:, ::1])

Note that this feature is only supported for simple numerical types. Application
to compound types, e.g. record types, is not supported.

Functions
---------

.. warning::
   The feature of considering functions as first-class type objects is
   under development.

Functions are often considered as certain transformations of
input arguments to output values. Within Numba :term:`JIT` compiled
functions, the functions can also be considered as objects, that is,
functions can be passed around as arguments or return values, or used
as items in sequences, in addition to being callable.

First-class function support is enabled for all Numba :term:`JIT`
compiled functions and Numba ``cfunc`` compiled functions except when:

- using a non-CPU compiler,
- the compiled function is a Python generator,
- the compiled function has Omitted arguments,
- or the compiled function returns Optional value.

To disable first-class function support, use ``no_cfunc_wrapper=True``
decorator option.

For instance, consider an example where the Numba :term:`JIT` compiled
function applies user-specified functions as a composition to an input
argument::

    >>> @numba.njit
    ... def composition(funcs, x):
    ...     r = x
    ...     for f in funcs[::-1]:
    ...         r = f(r)
    ...     return r
    ...
    >>> @numba.cfunc("double(double)")
    ... def a(x):
    ...     return x + 1.0
    ...
    >>> @numba.njit
    ... def b(x):
    ...     return x * x
    ...
    >>> composition((a, b), 0.5), 0.5 ** 2 + 1
    (1.25, 1.25)
    >>> composition((b, a, b, b, a), 0.5), b(a(b(b(a(0.5)))))
    (36.75390625, 36.75390625)

Here, ``cfunc`` compiled functions ``a`` and ``b`` are considered as
first-class function objects because these are passed in to the Numba
:term:`JIT` compiled function ``composition`` as arguments, that is, the
``composition`` is :term:`JIT` compiled independently from its argument function
objects (that are collected in the input argument ``funcs``).

Currently, first-class function objects can be Numba ``cfunc`` compiled
functions, :term:`JIT` compiled functions, and objects that implement the
Wrapper Address Protocol (WAP, see below) with the following restrictions:

========================   ============   ==============   ===========
Context                    JIT compiled   cfunc compiled   WAP objects
========================   ============   ==============   ===========
Can be used as arguments   yes            yes              yes
Can be called              yes            yes              yes
Can be used as items       yes\*          yes              yes
Can be returned            yes            yes              yes
Namespace scoping          yes            yes              yes
Automatic overload         yes            no               no
========================   ============   ==============   ===========

\* at least one of the items in a sequence of first-class function objects must
have a precise type.


Wrapper Address Protocol - WAP
++++++++++++++++++++++++++++++

Wrapper Address Protocol provides an API for making any Python object
a first-class function for Numba :term:`JIT` compiled functions. This assumes
that the Python object represents a compiled function that can be
called via its memory address (function pointer value) from Numba :term:`JIT`
compiled functions. The so-called WAP objects must define the
following two methods:

.. method:: __wrapper_address__(self) -> int

            Return the memory address of a first-class function. This
            method is used when a Numba :term:`JIT` compiled function tries to
            call the given WAP instance.

.. method:: signature(self) -> numba.typing.Signature

            Return the signature of the given first-class
            function. This method is used when passing in the given
            WAP instance to a Numba :term:`JIT` compiled function.

In addition, the WAP object may implement the ``__call__``
method. This is necessary when calling WAP objects from Numba
:term:`JIT` compiled functions in :term:`object mode`.

As an example, let us call the standard math library function ``cos``
within a Numba :term:`JIT` compiled function. The memory address of ``cos`` can
be established after loading the math library and using the ``ctypes``
package::

    >>> import numba, ctypes, ctypes.util, math
    >>> libm = ctypes.cdll.LoadLibrary(ctypes.util.find_library('m'))
    >>> class LibMCos(numba.types.WrapperAddressProtocol):
    ...     def __wrapper_address__(self):
    ...         return ctypes.cast(libm.cos, ctypes.c_voidp).value
    ...     def signature(self):
    ...         return numba.float64(numba.float64)
    ...
    >>> @numba.njit
    ... def foo(f, x):
    ...     return f(x)
    ...
    >>> foo(LibMCos(), 0.0)
    1.0
    >>> foo(LibMCos(), 0.5), math.cos(0.5)
    (0.8775825618903728, 0.8775825618903728)

Miscellaneous Types
-------------------

There are some non-numerical types that do not fit into the other categories.

===================   =================================================
Type name(s)          Comments
===================   =================================================
pyobject              generic Python object
voidptr               raw pointer, no operations can be performed on it
===================   =================================================

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
   ``ValueError`` is raised if the value isn't supported in
   :term:`nopython mode`.

   ::

      >>> numba.typeof(np.empty(3))
      array(float64, 1d, C)
      >>> numba.typeof((1, 2.0))
      (int64, float64)
      >>> numba.typeof([0])
      reflected list(int64)


NumPy scalars
-------------

Instead of using :func:`~numba.typeof`, non-trivial scalars such as
structured types can also be constructed programmatically.

.. function:: numba.from_dtype(dtype)

   Create a Numba type corresponding to the given NumPy *dtype*::

      >>> struct_dtype = np.dtype([('row', np.float64), ('col', np.float64)])
      >>> ty = numba.from_dtype(struct_dtype)
      >>> ty
      Record([('row', '<f8'), ('col', '<f8')])
      >>> ty[:, :]
      unaligned array(Record([('row', '<f8'), ('col', '<f8')]), 2d, A)

.. class:: numba.types.NPDatetime(unit)

   Create a Numba type for NumPy datetimes of the given *unit*.  *unit*
   should be a string amongst the codes recognized by NumPy (e.g.
   ``Y``, ``M``, ``D``, etc.).

.. class:: numba.types.NPTimedelta(unit)

   Create a Numba type for NumPy timedeltas of the given *unit*.  *unit*
   should be a string amongst the codes recognized by NumPy (e.g.
   ``Y``, ``M``, ``D``, etc.).

   .. seealso::
      NumPy `datetime units <http://docs.scipy.org/doc/numpy/reference/arrays.datetime.html#datetime-units>`_.


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


Type annotations
-----------------

.. function:: numba.extending.as_numba_type(py_type)

   Create a Numba type corresponding to the given Python *type annotation*.
   ``TypingError`` is raised if the type annotation can't be mapped to a Numba
   type.  This function is meant to be used at statically compile time to
   evaluate Python type annotations.  For runtime checking of Python objects
   see ``typeof`` above.

   For any numba type, ``as_numba_type(nb_type) == nb_type``.

      >>> numba.extending.as_numba_type(int)
      int64
      >>> import typing  # the Python library, not the Numba one
      >>> numba.extending.as_numba_type(typing.List[float])
      ListType[float64]
      >>> numba.extending.as_numba_type(numba.int32)
      int32

   ``as_numba_type`` is automatically updated to include any ``@jitclass``.

      >>> @jitclass
      ... class Counter:
      ...     x: int
      ...
      ...     def __init__(self):
      ...         self.x = 0
      ...
      ...     def inc(self):
      ...         old_val = self.x
      ...         self.x += 1
      ...         return old_val
      ...
      >>> numba.extending.as_numba_type(Counter)
      instance.jitclass.Counter#11bad4278<x:int64>

   Currently ``as_numba_type`` is only used to infer fields for ``@jitclass``.
