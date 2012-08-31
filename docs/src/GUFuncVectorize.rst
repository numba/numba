---------------
GUFuncVectorize
---------------

The GUFuncVectorize module of NumbaPro this creates a fast "generalized ufunc" from the numba-compiled code.  Unlike other NumbaPro Vectorize classes, the GUFuncVectorize constructor takes an additional signature of the generalized ufunc.


Imports
-------------------

::

	from numba.decorators import jit
	from numba import *
	import numpy as np
	import numpy.core.umath_tests as ut
	from numbapro.vectorize.gufunc import GUFuncVectorize

ufunc Definition
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

    gufunc = GUFuncVectorize(matmulcore, '(m,n),(n,p)->(m,p)')
    gufunc.add(arg_types=[f[:,:], f[:,:], f[:,:]])

::

	gufunc = gufunc.build_ufunc()

:: 

	matrix_ct = 10
    A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2, 4)
    B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4, 5)
  	C = gufunc(A, B)
    