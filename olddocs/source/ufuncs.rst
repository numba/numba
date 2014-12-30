******
UFuncs
******

Ufuncs
======

Numba's vectorize allows Numba functions taking scalar input arguments to be used as
NumPy ufuncs (see http://docs.scipy.org/doc/numpy/reference/ufuncs.html).
Creating a traditional NumPy ufunc is not the most difficult task in the world,
but it is also not the most straightforward process and involves writing some
C code. Numba makes this easy though. Using the vectorize decorator,
Numba can compile a Python function into a ufunc that operates over NumPy
arrays as fast as traditional ufuncs written in C.

Ufunc arguments are scalars of a NumPy array. Function definitions can be arbitrary
mathematical expressions. The vectorize decorator needs to know the argument
and return types of the ufunc. These are specified much like the jit decorator::

    import math

    @vectorize(['float64(float64, float64)'])
    def my_ufunc(x, y):
        return x+y+math.sqrt(x*math.cos(y))

    a = np.arange(1.0, 10.0)
    b = np.arange(1.0, 10.0)
    # Calls compiled version of my_ufunc for each element of a and b
    print(my_ufunc(a, b))

Multiple signatures can be specified to handle multiple input types::

    @vectorize(['int32(int32, int32)',
                'float64(float64, float64)'])
    def my_ufunc(x, y):
        return x+y+math.sqrt(x*math.cos(y))

    a = np.arange(1.0, 10.0, dtype='f8')
    b = np.arange(1.0, 10.0, dtype='f8')
    print(my_ufunc(a, b))

    a = np.arange(1, 10, dtype='i4')
    b = np.arange(1, 10, dtype='i4')
    print(my_ufunc(a, b))

The order of the signatures is important. Numba dispatches based on the input
array types and uses the first ufunc signature that the input types can be
safely cast to. In the example above, if the float64 signature had been listed
first, the call to sum with int32 arrays would have produced a float64 array
as the result.

An alternative syntax is to use the UFuncBuilder object to build a list of
function signatures::

    from numba.npyufunc.ufuncbuilder import UFuncBuilder

    def my_ufunc(x, y):
        return x+y+math.sqrt(x*math.cos(y))

    builder = UFuncBuilder(my_ufunc)
    builder.add(restype=i4, argtypes=[i4, i4])
    builder.add(restype=f8, argtypes=[f8, f8])

To compile our ufunc we call the build_ufunc method::

    compiled_ufunc = builder.build_ufunc()

    a = np.arange(1.0, 10.0, dtype='f8')
    b = np.arange(1.0, 10.0, dtype='f8')
    print(compiled_ufunc(a, b))

Since we defined a binary ufunc, we can use the various NumPy ufunc methods
such as ``reduce``, ``accumulate``, etc::

    a = np.arange(100)
    print(compiled_ufunc.reduce(a))
    print(compiled_ufunc.accumulate(a))

Generalized Ufuncs
==================

Numba also provides support for generalized ufuncs with the guvectorize decorator.
Traditional ufuncs perfom element-wise operations, whereas generalized ufuncs
operate on entire sub-arrays. In addition to the argument and return types,
the guvectorize decorator takes an additional signature which specifies the
shapes of the inner arrays we want to operate on::

    import math

    @guvectorize(['void(int32[:,:], int32[:,:], int32[:,:])',
                  'void(float64[:,:], float64[:,:], float64[:,:])'],
                  '(x, y),(x, y)->(x, y)')
    def my_gufunc(a, b, c):
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                c[i, j] = a[i, j] + b[i, j]

    a = np.arange(1.0, 10.0, dtype='f8').reshape(3,3)
    b = np.arange(1.0, 10.0, dtype='f8').reshape(3,3)
    # Calls compiled version of my_gufunc for each row of a and b
    print(my_gufunc(a, b))

Notice that we don't have a third argument in the gufunc call but the generalized
ufunc definition above has three arguments. The last argument of the generalized
ufunc is the output, which is automatically allocated with the shape specified
in the signature.


Generalized ufuncs also have an alternative syntax. We can use the GUFuncBuilder
object to build a list of function signatures and specify the shape of the arguments::
    
    from numba.npyufunc.ufuncbuilder import GUFuncBuilder

    def my_gufunc(a, b, c):
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                c[i, j] = a[i, j] + b[i, j]

    builder = GUFuncBuilder(my_ufunc, '(x, y),(x, y)->(x, y)')
    builder.add('void(int32[:,:], int32[:,:], int32[:,:])')
    builder.add('void(float64[:,:], float64[:,:], float64[:,:])')

To compile our ufunc we call the build_ufunc method::

    compiled_gufunc = builder.build_ufunc()

    a = np.arange(1.0, 10.0, dtype='f8').reshape(3,3)
    b = np.arange(1.0, 10.0, dtype='f8').reshape(3,3)
    print(my_gufunc(a, b))

