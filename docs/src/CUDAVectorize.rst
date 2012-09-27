-------------
CUDAVectorize
-------------

**Note: CUDAVectorize is still in experimental stages.  Computation speeds may have unexpected results .**


CUDAVectorize uses translated code from Numba and converts it to `PTX <http://en.wikipedia.org/wiki/Parallel_Thread_Execution>`_. `PyCUDA <http://documen.tician.de/pycuda/>`_ directives are called which load the PTX code on the device and calls it with array inputs.

Imports
-------

::

	import numpy as np
	from numba import *
	from numbapro.vectorize import CudaVectorize



ufunc Definition
-----------------

CudaVectorize ufunc arguments are scalars of a NumPy array.  Function definitions can be arbitrary
mathematical expressions.

::

	def my_ufunc(a, b, c, d):
		return a+b+sqrt(c*cos(d))



Compilation requires type information.  NumbaPro assumes no knowledge of type when building native ufunc.  We must therefore define argument and return dtypes for the defined ufunc.  We can add many and various dtypes for a given CudaVectorize ufunc.  This is similar to `function overloading <http://en.wikipedia.org/wiki/Function_overloading>`_ in C++

::

    cv = CudaVectorize(my_ufunc)
    cv.add(restype=int32, argtypes=[int32, int32])
    cv.add(restype=f, argtypes=[f, f])
    cv.add(restype=d, argtypes=[d, d])


Above we are using a signed 32-bit **int**, a float **f**, and a double **d**. 

To compile our ufunc we issue the following command

::

	cuda_ufunc = cv.build_ufunc()


*cuda_ufunc = cv.build_ufunc()* returns a python callable list of functions which are compiled by Numba.  The CUDA vectorize translates the Numba compiled ufunc kernel into an Nvidia PTX representation. Then, it uses PyCUDA for access to the CUDA driver and send the PTX to device execution.

Lastly, we call cuda_ufunc with two NumPy array as arguments

:: 

	data = np.array(np.random.random(100))
	result = cuda_ufunc(data, data))
