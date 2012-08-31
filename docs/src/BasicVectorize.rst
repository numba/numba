BasicVectorize
==============

This is the simplest vectorize.  It is suitable for small workloads in which the overhead for cache optimization, multithreading or GPU computation is too significant.

Module Dependencies
-------------------

::

	import numpy as np
	from numba import *
	from numbapro.vectorize.basic import BasicVectorize

ufunc Definition
-----------------

BasicVectorize ufunc arguments are scalars of a NumPy array.  Function definitions can be arbitrary
mathematical expressions.

::	

	def my_ufunc(a, b, c, d):
		return a+b+sqrt(c*cos(d))
 


Compilation requires type information.  NumbaPro assumes no knowledge of type when building native ufunc.  We must therefore define argument and return dtypes for the defined ufunc.  We can add many and various dtypes for a given BasicVectorize ufunc.  This is similar to `function overloading <http://en.wikipedia.org/wiki/Function_overloading>`_ in C++

::

	bv = BasicVectorize(my_ufunc)
	bv.add(ret_type=int32, arg_types=[int32, int32])
	bv.add(ret_type=uint32, arg_types=[uint32, uint32])
	bv.add(ret_type=f, arg_types=[f, f])
	bv.add(ret_type=d, arg_types=[d, d])

Above we are using signed and unsigned 32-bit, a float **f**, and a double **d**. 

To compile our ufunc we issue the following command

::

	basic_ufunc = bv.build_ufunc()

*bv.build_ufunc()* returns a python callable list of functions which are compiled by Numba.  *This work is normally accomplished by* `PyUFunc_FromFuncAndData <http://docs.scipy.org/doc/numpy/user/c-info.ufunc-tutorial.html>`_ and Numba takes care of it.* We've now registered a set of overload functions ready be used as NumPy ufuncs.

Lastly, we call basic_ufunc with two NumPy array as arguments

:: 

	data = np.array(np.random.random(100))
	result = basic_ufunc(data, data)
