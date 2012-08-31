-----------------
ParallelVectorize
-----------------

The ParallelVectorize module of NumbaPro targets multicore architectures.  It contains a set of `llvm-py <https://github.com/llvmpy/llvmpy>`_ code generators for creating multithreaded ufuncs. 

Imports
-------------------

::

	import numpy as np
	from numba import *
	from numbapro.vectorize.parallel import ParallelVectorize

ufunc Definition
-----------------

ParallelVectorize ufunc arguments are scalars of a NumPy array.  Function definitions can be arbitrary
mathematical expressions.

::	

	def my_ufunc(a, b, c, d):
		return a+b+sqrt(c*cos(d))
 


Compilation requires type information.  NumbaPro assumes no knowledge of type when building native ufuncs.  We must therefore define argument and return dtypes for the defined ufunc.  We can add many and various dtypes for a given ParallelVectorize ufunc.  This is similar to `function overloading <http://en.wikipedia.org/wiki/Function_overloading>`_ in C++

::

    pv = ParallelVectorize(vector_add)
	pv.add(ret_type=int32, arg_types=[int32, int32])
	pv.add(ret_type=uint32, arg_types=[uint32, uint32])
	pv.add(ret_type=f, arg_types=[f, f])
	pv.add(ret_type=d, arg_types=[d, d])

Above we are using signed and unsigned 32-bit ints, a float **f**, and a double **d**. 

To compile our ufunc we issue the following command

::

	para_ufunc = pv.build_ufunc()

*pv.build_ufunc()* returns a python callable list of functions which are compiled by Numba.  As mentioned above, `llvm-py <https://github.com/llvmpy/llvmpy>`_ creates a wrapper around the Numba complied functions and divides work evenly among the number of cores returned multiprocessing.cpu_count. We've now registered a set of overload functions ready be used as NumPy ufuncs.

Lastly, we call para_ufunc with two NumPy array as arguments

:: 

	data = np.array(np.random.random(100))
	result = para_ufunc(data, data)
