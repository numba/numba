Generalized Ufuncs
==================

The GUFuncVectorize module of NumbaPro this creates a fast "generalized ufunc" from numba-compiled code.
Traditional ufuncs perfom element-wise operations, whereas generalized ufuncs operate on entire
sub-arrays. Unlike other NumbaPro Vectorize classes, the GUFuncVectorize constructor takes
an additional signature of the generalized ufunc.


Imports
-------

::

	from numbapro import float32, float64, int32
	from numbapro.vectorize import GUVectorize
	import numpy as np

Generalized ufunc Definition
-----------------

GUFuncVectorize ufunc arguments are vectors of a NumPy array.  Function definitions can be arbitrary
mathematical expressions.

::

    def matmulcore(A, B, C):
        m, n = A.shape
        n, p = B.shape
        for i in range(m):
            for j in range(p):
                C[i, j] = 0
                for k in range(n):
                    C[i, j] += A[i, k] * B[k, j]

Compilation requires type information.  NumbaPro assumes no knowledge of type when building native ufuncs.  We must therefore define argument and return dtypes for the defined ufunc.  We can add many and various dtypes for a given GUFuncVectorize ufunc.  This is similar to `function overloading <http://en.wikipedia.org/wiki/Function_overloading>`_ in C++

::

    gufunc = GUVectorize(matmulcore, '(m,n),(n,p)->(m,p)')
    gufunc.add(argtypes=[float32[:,:], float32[:,:], float32[:,:]])
    gufunc.add(argtypes=[float64[:,:], float64[:,:], float64[:,:]])
    gufunc.add(argtypes=[int32[:,:], int32[:,:], int32[:,:]])

Above we are using a float **float32**, a double **float64**, and a signed **32-bit int**.  The GUVectorize calls `PyDynUFunc_FromFuncAndDataAndSignature <http://scipy-lectures.github.com/advanced/advanced_numpy/index.html#generalized-ufuncs>`_ which requires a the signature: *(m,n),(n,p)->(m,p)* in the constructor.  This signature defines the *"core dimensions"* of the generalized ufunc.


To compile our ufunc we issue the following command

::

    gufunc = gufunc.build_ufunc()

**pv.build_ufunc()** returns a python callable list of functions which are compiled by Numba.

Lastly, we call gufunc with two NumPy matrices

::

    matrix_ct = 10
    A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2, 4)
    B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4, 5)
    C = gufunc(A, B)


Notice that we don't have a third argument in the gufunc call but the generalized ufunc definition above has three arguments.  The last argument of the generalized ufunc is the the output.  Numba lacks the ability to return array objects.  A third object is implicitly defined with a shape defined by the signature.

Generalized ufuncs with Array Expressions
-----------------------------------------
If your generalized ufunc kernel contains array expressions, you will need to use the 'ast' Numba backend.
Array expressions are currently not supported in the bytecode backend::

    gufunc = GUVectorize(matmulcore, '(m,n),(n,p)->(m,p)', backend='ast')

It is currently not possible to combine ``target='gpu'`` and ``backend='ast'`` if the kernel
contains array expressions (this will fail when executing on the GPU).

Generalized CUDA ufuncs
-----------------------
Generalized ufuncs may also be executed on the GPU using CUDA, analogous to the CUDA ufunc functionality.
Jump to the `documentation for CUDA ufunc <CUDAufunc.html>`_ for continued discussion on generalized ufuncs.

