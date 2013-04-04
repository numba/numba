.. _arrays:

******
Arrays
******

Numba has support for fast indexing and understands NumPy arrays and infers
types for various calls of the NumPy API.


**Limitations**:

Unfortunately, there are a few pitfalls. We hope to resolve these in the
future, and to document them in the meantime:

=============================   =============================
Operation                       Example
=============================   =============================
Boundschecking                  ``array[N]``, with N < 0 or N > array.shape[0]

Wraparound                      ``array[-1]``

=============================   =============================


Array Expressions
=================

.. toctree::
      :maxdepth: 2

Numba implements array expressions which provide a single pass
over the data with a fused expression. It also implements native slicing
and stack-allocated NumPy array views, which means slicing is very fast compared
to slicing in Python. It also means one can now slice an array in
a ``nopython`` context. Lets try a diffusion in numba with loops and with
array expressions::

    from numba import *
    import numpy as np

    mu = 0.1
    Lx, Ly = 101, 101
    N = 1000

    @autojit
    def diffuse_loops(iter_num):
        u = np.zeros((Lx, Ly), dtype=np.float64)
        temp_u = np.zeros_like(u)
        temp_u[Lx / 2, Ly / 2] = 1000.0

        for n in range(iter_num):
            for i in range(1, Lx - 1):
                for j in range(1, Ly - 1):
                    u[i, j] = mu * (temp_u[i + 1, j] + temp_u[i - 1, j] +
                                    temp_u[i, j + 1] + temp_u[i, j - 1] -
                                    4 * temp_u[i, j])

            temp = u
            u = temp_u
            temp_u = temp

        return u

    @autojit
    def diffuse_array_expressions(iter_num):
        u = np.zeros((Lx, Ly), dtype=np.float64)
        temp_u = np.zeros_like(u)
        temp_u[Lx / 2, Ly / 2] = 1000.0

        for i in range(iter_num):
            u[1:-1, 1:-1] = mu * (temp_u[2:, 1:-1] + temp_u[:-2, 1:-1] +
                                  temp_u[1:-1, 2:] + temp_u[1:-1, :-2] -
                                  4 * temp_u[1:-1, 1:-1])

            temp = u
            u = temp_u
            temp_u = temp

        return u

.. NOTE:: Correct handling of overlapping memory between the left-hand and
          right-hand side of expressions is not supported yet.

Broadcasting
------------
Array expressions also support broadcasting, raising an error if shapes do not match::

    @autojit
    def matrix_vector(M, v):
        return np.sum(M * v, axis=1)

    M = np.arange(90).reshape(9, 10)
    v = np.arange(10)
    print matrix_vector(M, v)
    print np.dot(M, v)

Prints::

    [ 285  735 1185 1635 2085 2535 2985 3435 3885]
    [ 285  735 1185 1635 2085 2535 2985 3435 3885]

Calling the function with incompatible shapes gives the following::

    In [0]: matrix_vector(M, np.arange(8))
    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)
        ...
    ValueError: Shape mismatch while broadcasting

.. NOTE:: Errors raised in a nopython context print an error message and abort the
          program.

New Arrays
----------
Expressions not containing a left-hand side automatically create a new array::

    @autojit
    def square(a):
        return a * a

    print square(np.arange(10)) # array([ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81])

Allocating new arrays is however not support yet in nopython mode::

    @autojit(nopython=True)
    def square(a):
        return a * a

    print square(np.arange(10)) # NumbaError: 1:0: Cannot allocate new memory in nopython context

Math
----
All NumPy math functions supported on scalars is also supported for
arrays. This includes most unary ufuncs::

    @autojit
    def tan(a):
        return np.sin(a) / np.cos(a)

Universal Functions (ufuncs)
============================

Numba's vectorize allows Numba functions taking scalar input arguments to be used as
NumPy ufuncs (see http://docs.scipy.org/doc/numpy/reference/ufuncs.html).

For the example codes we will assume the user has run the following import::

    from numba.vectorize import Vectorize, vectorize

ufunc Definition
-----------------
Ufunc arguments are scalars of a NumPy array. Function definitions can be arbitrary
mathematical expressions.

::

	def my_ufunc(a, b):
		return a+b+sqrt(a*cos(b))

Compilation requires type information.  Numba assumes no knowledge of type when building native
ufuncs.  We must therefore define argument and return dtypes for the defined ufunc.  We can add
many and various dtypes for a given BasicVectorize ufuncs, using Numba types, to create different
versions of the code depending on the inputs.

::

	v = Vectorize(my_ufunc)
	v.add(restype=int32, argtypes=[int32, int32])
	v.add(restype=uint32, argtypes=[uint32, uint32])
	v.add(restype=f4, argtypes=[f4, f4])
	v.add(restype=f8, argtypes=[f8, f8])

Above we are using signed and unsigned 32-bit ints, a float **f4**, and a double **f8**.

To compile our ufunc we issue the following command

::

	basic_ufunc = v.build_ufunc()

*bv.build_ufunc()* returns a python callable list of functions which are compiled by Numba.  *This work is normally accomplished by* `PyUFunc_FromFuncAndData <http://docs.scipy.org/doc/numpy/user/c-info.ufunc-tutorial.html>`_ and Numba takes care of it.* We've now registered a set of overload functions ready be used as NumPy ufuncs.

Lastly, we call basic_ufunc with two NumPy array as arguments

::

	data = np.array(np.random.random(100))
	result = basic_ufunc(data, data)

Since we defined a binary ufunc, we can use the various ufunc methods such as ``reduce``, ``accumulate``,
etc::

    data = np.array(np.arange(100), dtype=np.int32)
    print basic_ufunc.reduce(data)
    print basic_ufunc.accumulate(data)


Building ufuncs using @vectorize
--------------------------------

An alternative syntax is available through the use of the `vectorize` decorator::

    from numba import *
    import math

    pi = math.pi

    @vectorize([f4(f4), f8(f8)], target='cpu')
    def sinc(x):
        if x == 0.0:
            return 1.0
        else:
            return math.sin(x*pi)/(pi*x)

The `vectorize` decorator takes a list of function signature and an optional `target` keyword argument (default to 'cpu').  The example above generate a `sinc` ufunc that is overloaded to accept float and double.  This syntax replaces calls to `Vectorize.add` and `Vectorize.build_ufunc`.
