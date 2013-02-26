*******************
Arrays
*******************
Numba has support for fast indexing and understands NumPy arrays and infers
types for various calls of the NumPy API.


Limitations
-------------
Unfortunately, there are a few pitfalls. We hope to resolve these in the
future, and to document them in the meantime:

=============================   =============================
Operation                       Example
=============================   =============================
Boundschecking                  ``array[N]``, with N < 0 or N > array.shape[0]

Wraparound                      ``array[-1]``

Calls to imported functions     Importing things from ``numpy``

                                ::

                                    from numpy import zeros

                                    @autojit
                                    def func():
                                        array = zeros(...)

Calling without a dtype         Calling ``zeros``, ``ones`` or ``empty``
                                without a dtype or with lists

                                ::

                                    np.zeros((M, N))                  # No dtype!
                                    np.zeros([M, N], dtype=np.double) # Not a tuple or int!

=============================   =============================


Array Expressions
-----------------

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

.. NOTE:: The next release may support parallel array expressions and
          tiled array expressions for mixed C- and Fortran-like data layouts.
          The next release will also support array expressions on the GPU.

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

.. NOTE:: Error raised in a nopython context print an error message and abort the
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

