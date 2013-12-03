Quick Start
==============

Numba/NumbaPro uses decorators extensively to annotate function for
compilation.  This document explains the major decorators: ``jit``,
``autojit``, ``vectorize`` and ``guvectorize``.

Types
-----

Numba/NumbaPro decorators specializes generic python function to typed native
function.  All decorators except ``autojit`` requires type information to be
supplied.  Here is a table of common Numba type objects:

=================  ===============================
Kind                Numba Types
=================  ===============================
signed integer      int8, int16, int32, int64
unsigned integer    uint8, uint16, uint32, uint64
float-points        float32, float64
complex numbers     complex64, complex128
boolean             bool\_
others              void
=================  ===============================

Array Types
~~~~~~~~~~~

Array types are created by "creating slices" of Numba type objects.  For
example:

.. testcode::

   from numbapro import int32, float32
   print(int32[:])           # 1D int32 array
   print(float32[:,:])       # 2D float32 array
   print(int32[:,:,:,:])     # 4D int32 array

Output:

.. testoutput::

    int32[:]
    float32[:, :]
    int32[:, :, :, :]


Function Types
~~~~~~~~~~~~~~

The function type is created from "calling" a Numba type object.

.. testcode::

    from numbapro import void, int32, float32, complex64
    print(complex64(int32, float32, complex64))
    print(float32())                             # no arguments
    print(void(float32))                         # return nothing
    print(void(float32[:], int32[:]))

Output:

.. testoutput::

    complex64 (*)(int32, float32, complex64)
    float32 (*)()
    void (*)(float32)
    void (*)(float32[:], int32[:])

Alternatively, the function type can be provided as a string to
decorators for avoiding the import of the type objects::

    "complex64(int32, float32, complex64)"

``numbapro.jit``
----------------

The ``jit`` decorator annotate a function for runtime compilation given
the function type.

Example
~~~~~~~

.. testcode::

    from numbapro import jit, int32, float32, complex64

    @jit(complex64(int32, float32, complex64), target="cpu")
    def bar(a, b, c):
       return a + b  * c

    @jit(complex64(int32, float32, complex64)) # target kwarg defaults to "cpu"
    def foo(a, b, c):
       return a + b  * c


    print foo
    print foo(1, 2.0, 3.0j)

Output:

.. testoutput::

    <NumbaFunction foo at ...>
    (1+6j)

.. note:: The target keyword is discussed later.

``numbapro.autojit``
---------------------

The ``autojit`` decorator annotate a function for deferred compilation at
callsite.  The function signature is inferred from the arguments.  Each
function signature is compiled exactly once.  Later invocation with the
same function signature will reuse a cached copy of the compiled function.

Example
~~~~~~~

.. testcode::

    from numbapro import autojit

    @autojit(target="cpu")
    def bar(a, b, c):
        return a + b * c

    @autojit                    # target kwarg defaults to "cpu"
    def foo(a, b, c):
        return a + b * c

    print(foo)
    print(foo(1, 2.0, 3j))

Output:

.. testoutput::

    <specializing numba function(<function foo at ...>)>
    (1+6j)

.. note:: The target keyword is discussed later.

``numbapro.vectorize``
----------------------

The ``vectorize`` decorator produces a NumPy Universal function (ufunc) object
from a python function.  A ufunc can be overloaded to take multiple
combination parameter types.  User must provide a list of function types as
the first argument of ``vectorize``.

Example
~~~~~~~

.. testcode::

    from numbapro import vectorize
    from numpy import arange

    @vectorize(['float32(float32, float32)'], target='cpu') # default to 'cpu'
    def add2(a, b):
        return a + b

    X = arange(10, dtype='float32')
    Y = X * 2
    print add2(X, Y)
    print add2.reduce(X)

Output:

.. testoutput::

    [  0.   3.   6.   9.  12.  15.  18.  21.  24.  27.]
    45.0

``numbapro.guvectorize``
------------------------

The ``guvectorize`` decorator produces a NumPy Generalized Univesral function
(gufunc) object from a python function. While ``vectorize`` works on scalar
arguments, ``guvectorize`` works on array arguments.  This decorator takes an
extra argument for specifying gufunc signature.  Please refer to
`NumPy documentations <http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html>`_
for details of gufunc.


Example: Batch Matrix Multiplication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. testcode::

    from numbapro import guvectorize
    from numpy import arange

    @guvectorize(['void(float64[:,:], float64[:,:], float64[:,:])'],
                 '(m,n),(n,p)->(m,p)')
    def matmul(A, B, C):
        m, n = A.shape
        n, p = B.shape
        for i in range(m):
            for j in range(p):
                C[i, j] = 0
                for k in range(n):
                    C[i, j] += A[i, k] * B[k, j]

    w = 2
    A = arange(w**2).reshape(w, w)
    B = arange(w**2).reshape(w, w)
    C = matmul(A, B)
    print("A:\n%s" % A)
    print("B:\n%s" % B)
    print("C:\n%s" % C)

.. testoutput::

    A:
    [[0 1]
     [2 3]]
    B:
    [[0 1]
     [2 3]]
    C:
    [[  2.   3.]
     [  6.  11.]]


Example: 2D -> 1D
~~~~~~~~~~~~~~~~~

.. testcode::

    from numbapro import guvectorize
    from numpy import zeros, arange

    @guvectorize(['void(int32[:], int32[:])'], '(n)->()')
    def sum_row(inp, out):
        """
        Sum every row

        function type: two arrays
                       (note: scalar is represented as an array of length 1)
        signature: n elements to scalar
        """
        tmp = 0.
        for i in range(inp.shape[0]):
            tmp += inp[i]
        out[0] = tmp

    inp = arange(15, dtype='int32').reshape(5, 3)
    print(inp)

    # implicit output array
    out = sum_row(inp)
    print('imp: %s' % out)

    # explicit output array
    explicit_out = zeros(5, dtype='int32')
    sum_row(inp, out=explicit_out)
    print('exp: %s' % explicit_out)

Output:

.. testoutput::

    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]
     [12 13 14]]
    imp: [ 3 12 21 30 39]
    exp: [ 3 12 21 30 39]


Compiler Target ``target="..."``
---------------------------------

All decorators, ``jit``, ``autojit``, ``vectorize`` and ``guvectorize``,
have a target keyword argument to select the code generation
target.  User provides a string to name the target.  Numba supports only the
``"cpu"`` target. NumbaPro adds ``"parallel"`` and ``"gpu"``.  The
``"parallel"``
target is only available for ``vectorize``, which will distributes the work
across CPU threads.  The "gpu" offloads the computation to a Nvidia CUDA GPU.