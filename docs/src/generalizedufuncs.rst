Generalized Ufuncs
==================

The GUFuncVectorize module of NumbaPro this creates a fast "generalized ufunc" from numba-compiled code.
Traditional ufuncs perfom element-wise operations, whereas generalized ufuncs operate on entire
sub-arrays. Unlike other NumbaPro Vectorize classes, the GUFuncVectorize constructor takes
an additional signature of the generalized ufunc.


Imports
--------

::

	from numbapro import *
	import numpy as np
	from numbapro.vectorize import GUVectorize

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
    gufunc.add(argtypes=[f4[:,:], f4[:,:], f4[:,:]])
    gufunc.add(argtypes=[f8[:,:], f8[:,:], f8[:,:]])
    gufunc.add(argtypes=[int32[:,:], int32[:,:], int32[:,:]])

Above we are using a signed **32-bit int**, a float **f4**, and a double **f8**.  The GUVectorize calls `PyDynUFunc_FromFuncAndDataAndSignature <http://scipy-lectures.github.com/advanced/advanced_numpy/index.html#generalized-ufuncs>`_ which requires a the signature: *(m,n),(n,p)->(m,p)* in the constructor.  This signature defines the *"core dimensions"* of the generalized ufunc.


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

Generalized CUDA ufuncs
=======================
Generalized ufuncs may also be executed on the GPU using CUDA, analogous to the CUDA ufunc functionality.
This may be accomplished as follows::

    gufunc = GUVectorize(matmulcore, '(m,n),(n,p)->(m,p)', target='gpu')

Or, through the one line decorator syntax::

    from numbapro import guvectorize

    @guvectorize([void(f4[:,:], f4[:,:], f4[:,:])], '(m,n),(n,p)->(m,p)', target='gpu')
    def matmulcore(A, B, C):
        ...

**Note:** `GUVectorize(..., target='gpu').build_ufunc` returns an *ufunc-like* object.  See `documentation for CUDA ufunc <CUDAufunc.html>`_ for detail.

There are times when the gufunc kernel uses too much GPU resources, which can
can the kernel launch to fail.  User to explicitly control the maximum size of
thread block by setting setting the `max_blocksize` attribute on the compiled
gufunc object.  

::
    
    
    @guvectorize(..., target='gpu')
    def very_complex_kernel(A, B, C):
        ...
        
    very_complex_kernel.max_blocksize = 32  # limits to 32 threads per block


Generalized ufuncs with Array Expressions
=========================================
If your generalized ufunc kernel contains array expressions, you will need to use the 'ast' Numba backend.
Array expressions are currently not supported in the bytecode backend::

    gufunc = GUVectorize(matmulcore, '(m,n),(n,p)->(m,p)', backend='ast')

It is currently not possible to combine ``target='gpu'`` and ``backend='ast'`` if the kernel
contains array expressions (this will fail when executing on the GPU).
