---------------
StreamVectorize
---------------

**Note: StreamVectorize is still in experimental stages.  Computation speeds may have unexpected results .**

StreamVectorize aims to provide a cache optimized version of vectorize. The current implementation works on small chunks of data at a time.  StreamVectorize copies memory to aligned, contiguous buffers (small enough presumably to fit in cache) and then executes the code on those chunks.

Imports
-------

::

	import numpy as np
	from numba import *
	from numbapro.vectorize import StreamVectorize


ufunc Definition
-----------------

StreamVectorize ufunc arguments are scalars of a NumPy array.  Function definitions can be arbitrary
mathematical expressions.

::

	def my_ufunc(a, b, c, d):
		return a+b+sqrt(c*cos(d))



Compilation requires type information.  NumbaPro assumes no knowledge of type when building native ufuncs.  We must therefore define argument and return dtypes for the defined ufunc.  We can add many and various dtypes for a given StreamVectorize ufunc.  This is similar to `function overloading <http://en.wikipedia.org/wiki/Function_overloading>`_ in C++

::

    sv = StreamVectorize(my_ufun)
    sv.add(restype=int32, argtypes=[int32, int32])
    sv.add(restype=f4, argtypes=[f4, f4])
    sv.add(restype=f8, argtypes=[f8, f8])

Above we are using signed and unsigned 32-bit ints, a float **f4**, and a double **f8**. 

To compile our ufunc we issue the following command

::

	stream_ufunc = sv.build_ufunc(granularity=32)

*sv.build_ufunc(granularity=32)* returns a python callable list of functions which are compiled by Numba.  The argument, *granularity=32*, controls the granularity of the chunking in bytes.

Lastly, we call stream_ufunc with two NumPy array as arguments

:: 

	data = np.array(np.random.random(100))
	result = stream_ufunc(data, data))
