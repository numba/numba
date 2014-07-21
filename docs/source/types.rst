*****
Types
*****

Basic Types
===========

The following table contains the elementary types currently defined by Numba.

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

Types can be used to specify the signature of a function::

    @jit('f8(f8[:])')
    def sum1d(array):
        sum = 0.0
        for i in range(array.shape[0]):
            sum += array[i]
        return sum

Types can also be used in Numba to declare local variables in a function::

    @jit(locals=dict(array=double[:, :], scalar1=double))
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

Variables declared in the ``locals`` dict have a single type throughout
the entire function. However, any variable not declared in locals can
assume different types, just like in Python::

    @jit
    def variable_ressign(arg):
        arg = 1.0
        arg = "hello"
        arg = object()
        var = arg
        var = "world"

However, there are some restrictions, namely that variables must have
a unifyable type at control flow merge points. For example, the following
code will not compile::

    @jit
    def incompatible_types(arg):
        if arg > 10:
            x = "hello"
        else:
            x = 1

        return x        # ERROR! Inconsistent type for x!

This code is invalid because strings and integers are not compatible.
However, if we do not read ``x`` after the if block, the code will
compile fine, since it does not need to unify the type::

    @jit
    def compatible_types(arg):
        if arg > 10:
            x = "hello"
        else:
            x = arg

        x = func()
        return x

The same goes for loop carried dependencies and variables escaping loops, e.g.::

    @jit
    def incompatible_types2(N):
        x = "hello"
        for i in range(N):
            print x     # ERROR! Inconsistent type for x!
            x = i

        return x

    @jit
    def incompatible_types3(N):
        x = "hello"
        for i in range(N):
            x = i
            print x

        return x        # ERROR! Inconsistent type for x if N <= 0

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

More Complex Types
==================

Numba is in the process of being refactored to better define more complex types
such as structs, pointers, strings and user defined classes. More on this soon...

