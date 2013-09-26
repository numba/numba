.. _types:

*******************
Types and Variables
*******************

Types can be used in Numba to compile functions directly with the ``jit``
function, and they can be used to declare local variables in both ``jit``
and ``autojit`` functions::

    @autojit(locals=dict(array=double[:, :], scalar1=double))
    def func(array):
        scalar1 = array[0, 0] # scalar is declared double
        scalar2 = double(array[0, 0])

Of course, declaring types in this example is unnecessary since the type
inferencer knows the input type of ``array``, and hence knows the type
of ``array[i, j]`` to be the dtype of ``array``.

.. NOTE:: Type declarations or casts can be useful in cases where the
          type inferencer doesn't know the type, or if you want to
          override the type inferencer's rules (e.g. force 32-bit floating
          point precision).

Cases where the type inferencer doesn't know the type is often when you call
a Python function or method that is not a numba function and numba doesn't
otherwise recognize.

Numba allows you to obtain the type of a expression or variable through
the typeof function in a Numba function. This type can then be used for
instance to cast other values::

    type = numba.typeof(x + y)
    value = type(value)

When used outside of a Numba function, it returns the type the type inferencer
would infer for that value::

    >>> numba.typeof(1.0)
    double
    >>> numba.typeof(cmath.sqrt(-1))
    complex128

.. _variables:

Variables declared in Locals
============================
Variables declared in the ``locals`` dict have a single type throughout
the entire function. However, any variable not declared in locals can
assume different types, just like in Python::

    @autojit
    def variable_ressign(arg):
        arg = 1.0
        arg = "hello"
        arg = object()
        var = arg
        var = "world"

However, there are some restrictions, namely that variables must have
a unifyable type at control flow merge points. For example, the following
code will not compile::

    @autojit
    def incompatible_types(arg):
        if arg > 10:
            x = "hello"
        else:
            x = arg

        return x        # ERROR! Inconsistent type for x!

This code is invalid because strings and integers are not compatible.
However, if we do not read ``x`` after the if block, the code will
compile fine, since it does not need to unify the type::

    @autojit
    def compatible_types(arg):
        if arg > 10:
            x = "hello"
        else:
            x = arg

        x = func()
        return x

The same goes for loop carried dependencies and variables escaping loops, e.g.::

    @autojit
    def incompatible_types2(N):
        x = "hello"
        for i in range(N):
            print x     # ERROR! Inconsistent type for x!
            x = i

        return x

    @autojit
    def incompatible_types3(N):
        x = "hello"
        for i in range(N):
            x = i
            print x

        return x        # ERROR! Inconsistent type for x if N <= 0

Specifying Aggregate Types
==========================
The numba type system goes far beyond the simple scalars (e.g. ``ushort``)
and arrays (e.g. ``float32[:, :]``) that we had previously covered.
We can also define structs, pointers, functions and strings.

.. _structtypes:

Structs/Records
---------------
Structs can be either aligned or unaligned (packed). Aligned structs are
the recommended default. Structs can be ordered, in case we need to interface
with non-numba code or NumPy record arrays::

    >>> import numba
    >>> numba.struct([('first_field', double), ('second_field', float_)], name='MyStruct')
    struct MyStruct { double first_field, float second_field }

The ``name`` argument is optional, but useful for debugging purposes and code clarity.
An unordered struct can be created using keyword arguments::

    >>> numba.struct(first_field=double, second_field=float_)
    struct { double first_field, float second_field }

Unordered structs order their fields by the size of their data type, and secondarily on
the field name. Since ``sizeof(double) > sizeof(float)``,
``first_field`` precedes ``second_field`` in the struct.

Inferface:

.. class:: numba.struct(fields, name=None, packed=False)

    .. attribute:: name

        Struct name or None

    .. attribute:: packed

        Whether the fields of the structs are aligned or packed.

    .. attribute:: fields

        List of 2-tuples in the order of the fields: ``[(field_name, field_type)]``.

    .. attribute:: fielddict

        Dict mapping field names to field types.

.. _pointertypes:

Pointers
--------
Each type has a ``pointer`` method that allows one to create a pointer type with that type
as the base type::

    >>> double.pointer()
    double *
    >>> numba.struct(first_field=double, second_field=float_).pointer()
    struct { double first_field, float second_field } *

Pointer support is still somewhat immature, but in the future it is likely ctypes and CFFI pointers
will be supported, and possibly pointers returned by Cython cdef functions or methods. Currently
pointers can be obtained from integers by using the ``Py_uintptr_t`` type, which is an integer large
enough to store any pointer::

    voidp = void.pointer()

    @autojit
    def test_compare_null_attribute():
        return voidp(Py_uintptr_t(0)) == numba.NULL

Note how we declare the type ``void *`` outside the function, since numba does not yet recognize
``void.pointer()`` inside a numba function as the type which it constitutes (it is also valid
to pass ``voidp`` in as an argument to the function).

Note also how we use numba.NULL, which represents the C NULL pointer, and may be compared to a
pointer of any type.

.. NOTE:: Type declarations inside numba functions itself is still immature, but at any time
          can types be passed into numba functions, or accessed as module attributes or globals.

Inferface:

.. class:: PointerType

    .. attribute:: base_type

        Base type of the pointer, i.e. what type after dereferencing the pointer.

.. _functiontypes:

Functions
---------
As we have already seen, functions can be easily specified by calling types::

    >>> void(double)
    void (*)(double)

Function types can also be created through the ``FunctionType`` class exposed in the ``numba`` namespace.
For instance, this allows you to omit a return type, and to have the type inferencer infer the return
type automatically::

    >>> numba.FunctionType(return_type=None, args=[], name="foo")
    None (*foo)()
    >>> numba.FunctionType(return_type=void, args=[], name="foo", is_vararg=True)
    void (*foo )(...)

Inferface:

.. class:: numba.FunctionType(return_type, args, name=None, is_vararg=False)

    .. attribute:: return_type

        Base type of the pointer, i.e. the type after dereferencing the pointer.

    .. attribute:: args

        The argument types.

    .. attribute:: is_vararg

        Whether it takes a variable number of arguments (compatible with C ABI).

.. _stringtypes:

Strings
-------
Strings may be specified through the ``c_string_type`` type, a name which is subject to change in the future.
This does not handle unicode, and is equivalent to ``char *``::

    >>> c_string_type
    const char *

C Arrays
--------

.. _datetimtypes:

DateTimes
---------
NumPy datetime and timedelta types are supported with numba.datetime and
numba.timedelta types. Internally a NumPy datetime or timedelta object is
converted to a struct type with a timestamp/timedelta field and a datetime
units field, so datetime and timedelta objects can be used in nopython mode.

NumPy datetime and timedelta scalars and arrays can be passed to a numba
function. Datetime components can also be accessed::

    @numba.jit(void(numba.datetime), nopython=True)
    def foo(x):
        year = date.year

    foo(numpy.datetime64('2014-01-01'))

New datetime and timedelta objects can be created inside a numba function and
returned. ::

    @numba.autojit(nopython=True)
    def foo():
        date = numpy.datetime64('2014-01-01')
        return date

Currently a datetime can be subtracted from another datetime to get a timedelta,
and a timedelta can be added to or subtracted from a datetime to get a new
datetime::

    @numba.jit(numba.timedelta(numba.datetime, numba.datetime), nopython=True)
    def foo1(date1, date2):
        return date1 - date2

    @numba.jit(numba.datetime(numba.datetime, numba.timedelta), nopython=True)
    def foo2(date, delta):
        return date + delta

Arrays of datetimes and timedeltas can be passed in and returned from a numba
function, and indexed within a numba function. When a datetime array type is
specified, the datetime units must also be specified (datetime units are the
same as NumPy datetime64 units)::

    @numba.jit(numba.datetime(numba.datetime(units='M')[:]))
    def foo(datetimes):
        return datetimes[0]

.. _templates:

Templates
=========
Templates allow the user to deal with types in a more abstract manner, which is useful when concrete types
are not available at the time of specification. This can be used in conjuction with the ``autojit`` decorator,
which determines the types based on the argument input values. For example, this allows one to access the
base type of a pointer, or the dtype of an array to declare variable types or perform casts::

    T = numba.template("T") # the name argument is optional

    @autojit(T(T[:, :]), locals=dict(scalar=T))
    def test_templates(array):
        scalar = array[0, 0]
        return scalar

This specifies that the function takes a 2D array of some dtype ``T``, and returns a value of type ``T``.
The local variable ``scalar`` also assumes type ``T``. In this example we could just as well have relied
on the type inferencer, but we have gained a constraint on the input type ``array``, namely that it is
a 2D array.

We can loosen the constaint a bit, and for instance allow any N-dimensional array to be passed in::

    T = numba.template()
    dtype = T.dtype

    @autojit(dtype(T), locals=dict(scalar=dtype))
    def test_template_generic(array):
        scalar = array[(0,) * array.ndim]
        return scalar

We can do a similar thing with pointers by accessing the ``base_type`` attribute, or with struct fields
by indexing ``fielddict``. E.g. we could write::

    pointer_type = T2.pointer()
    struct_type = numba.struct(array=T1[:], pointer=pointer_type, func=T1(pointer_type, T3))

    @autojit(void(struct_type))
    def process_struct(struct):
        arg = T3(0)
        array[0] = struct.func(struct.pointer, arg)

as::

    cast_type = T.fielddict["func"].args[1] # Get at the T3 type in the example above

    @autojit(void(T))
    def process_struct(struct):
        arg = cast_type(0)
        array[0] = struct.func(struct.pointer, arg)


