***********
Quick Start
***********

Numba compiles Python code to LLVM IR which can be natively executed at runtime
much faster than pure Python code. The first step to using Numba is becoming
familiar with the ``jit`` decorator, which tells Numba which functions to
compile::

    from numba import jit

    @jit
    def sum(x, y):
        return x + y

The very basic example above is compiled for any compatible input types
automatically when the ``sum`` function is called. The result is a new function
with performance comparable to a compiled function written in C (assuming best
case scenario; more on that later). To compile for specific input types, we
can tell Numba what those input types are::

    @jit('f8(f8,f8)')
    def sum(x, y):
        return x + y

The string above passed to the ``jit`` decorator tells Numba the return type is an
8 byte float, and the two arguments passed in are also 8 byte floats. The
string takes the form ``returntype(arg1type, arg2type, ...)``.

One of the main features of Numba is it's support for NumPy arrays. The
following example shows how a function can be compiled that takes a NumPy array
of floats as an input::

    @jit('f8(f8[:])')
    def sum1d(array):
        sum = 0.0
        for i in range(array.shape[0]):
            sum += array[i]
        return sum

There are two main things to notice in the example above. The input argument is
specified by the string ``f8[:]``, which means a 1d array of 8 byte floats. A
2d array would be specified as ``f8[:, :]``, a 3d array as ``f8[:, :, :]``, and
so on. The other thing to take note of is the array indexing and shape method
call, and the fact that we're iterating over a NumPy array using Python.
Normally this would be terribly slow and would be cause for writing a NumPy
ufunc in C, but the performance of the code above is the same as NumPy's
``sum`` method.

Numba can also infer the array type automatically like other elementary types::

    @jit
    def sum1d(array):
        ...

Numba's elementary built in types in are summarized in the table below and can
be found in the ``numba`` namespace.

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

Function signatures can also be expressed with the type objects directly as
opposed to using strings. For example::

   from numba import jit, f8

   @jit(f8(f8[:]))
   def sum1d(array):
       ...

In the example above, the argument type object is passed in to the return type
object's constructor.

Numba attempts to compile everything down to LLVM IR, but in some cases this
isn't (yet) possible. If Numba can't infer the type of a variable or doesn't
support a particular data type, it falls back to using Python objects. This is
of course much slower. If you're having performance issues and suspect Python
objects are to blame, you can use the ``nopython`` flag to force Numba to abort
if it can't avoid using Python objects::

    @jit(nopython=True)
    def sum1d(array):
        ...

Another useful debugging tool is Numba's new ``inspect_types`` method. This can
be called for any Numba compiled function to get a listing of the Numba IR
generated from the Python code as well as the inferred types of each variable::

    >>> sum1d.inspect_types()
    sum1d (array(float64, 1d, A),) -> float64
    ``-----------------------------------------------------------------------``
    # --- LINE 5 --- 

    @jit('f8(f8[:])')

    # --- LINE 6 --- 

    def sum1d(array):

        # --- LINE 7 --- 
        # label 0
        #   $0.1 = const(<type 'float'>, 0.0)  :: float64
        #   sum = $0.1  :: float64

        sum = 0.0

        # --- LINE 8 --- 
        #   jump 6
        # label 6
        #   $6.1 = global(range: <built-in function range>)  :: range
        #   $6.2 = getattr(attr=shape, value=array)  :: (int64 x 1)
        #   $6.3 = const(<type 'int'>, 0)  :: int32
        #   $6.4 = getitem(index=$6.3, target=$6.2)  :: int64
        #   $6.5 = call $6.1($6.4, )  :: (int64,) -> range_state64
        #   $6.6 = getiter(value=$6.5)  :: range_iter64
        #   jump 26
        # label 26
        #   $26.1 = iternext(value=$6.6)  :: int64
        #   $26.2 = itervalid(value=$6.6)  :: bool
        #   branch $26.2, 29, 50
        # label 29
        #   $29.1 = $26.1  :: int64
        #   i = $29.1  :: int64

        for i in range(array.shape[0]):

            # --- LINE 9 --- 
            # label 49
            #   del $6.6
            #   $29.2 = getitem(index=i, target=array)  :: float64
            #   $29.3 = sum + $29.2  :: float64
            #   sum = $29.3  :: float64
            #   jump 26

            sum += array[i]

        # --- LINE 10 --- 
        #   jump 50
        # label 50
        #   return sum

        return sum

For get a better feel of what Numba can do, see :ref:`Examples <examples>`.

