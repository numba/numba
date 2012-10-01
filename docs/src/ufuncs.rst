Universal Functions
===================

NumbaPro allows Numba functions taking scalar input arguments to be used as
NumPy ufuncs (see http://docs.scipy.org/doc/numpy/reference/ufuncs.html).

For the example codes we will assume the user has run the following import::

    from numbapro.vectorize import Vectorize

ufunc Definition
-----------------
Ufunc arguments are scalars of a NumPy array.  Function definitions can be arbitrary
mathematical expressions.

::

	def my_ufunc(a, b, c, d):
		return a+b+sqrt(c*cos(d))

Compilation requires type information.  NumbaPro assumes no knowledge of type when building native
ufuncs.  We must therefore define argument and return dtypes for the defined ufunc.  We can add
many and various dtypes for a given BasicVectorize ufuncs, using Numba types, to create different
versions of the code depending on the inputs.

::

	v = Vectorize(my_ufunc)
	v.add(restype=int32, argtypes=[int32, int32])
	v.add(restype=uint32, argtypes=[uint32, uint32])
	v.add(restype=f4, argtypes=[f4, f4])
	v.add(restype=f8, argtypes=[f8, f8])

Above we are using signed and unsigned 32-bit ints, a float **f4**, and a double **f8**.

To compile our ufunc we issue the following command

::

	basic_ufunc = v.build_ufunc()

*bv.build_ufunc()* returns a python callable list of functions which are compiled by Numba.  *This work is normally accomplished by* `PyUFunc_FromFuncAndData <http://docs.scipy.org/doc/numpy/user/c-info.ufunc-tutorial.html>`_ and Numba takes care of it.* We've now registered a set of overload functions ready be used as NumPy ufuncs.

Lastly, we call basic_ufunc with two NumPy array as arguments

::

	data = np.array(np.random.random(100))
	result = basic_ufunc(data, data)

Universal Function Types
------------------------
There are several vectorizer versions available. The different options are listed below:

=================       ===============================================================
Name                    Description
=================       ===============================================================
BasicVectorize          Single-threaded vectorization (default if no options given).

                        It is suitable
                        for small workloads in which
                        the overhead for cache optimization, and multithreading or GPU
                        computation is too significant.


                        **Usage:**

                        ::

                            vectorizer = Vectorize(numba_ufunc, target='cpu')

ParallelVectorize       Multi-core vectorization. It contains a set of
                        `llvm-py <https://github.com/llvmpy/llvmpy>`_ code generators
                        for creating multithreaded ufuncs.


                        **Usage:**

                        ::

                            vectorizer = Vectorize(numba_ufunc, target='parallel')

StreamVectorize         StreamVectorize aims to provide a cache optimized version
                        of vectorize. The current implementation works on small chunks
                        of data at a time.  StreamVectorize copies memory to aligned,
                        contiguous buffers (small enough presumably to fit in cache)
                        and then executes the code on those chunks.

                        **Note:** StreamVectorize is still in experimental stages. Computation speeds may vary.

                        **Usage:**

                        ::

                            vectorizer = Vectorize(numba_ufunc, target='stream')

CudaVectorize           CUDAVectorize uses translated code from Numba and converts it to
                        `PTX <http://en.wikipedia.org/wiki/Parallel_Thread_Execution>`_,
                        which is then compiled by CUDA and loaded on the device when called with array inputs.

                        **Usage:**

                        ::

                            vectorizer = Vectorize(numba_ufunc, target='gpu')

MiniVectorize           The Mini vectorizer is an alternative vectorizer which can provide more
                        stable performance for various sorts of mixed data layouts by dispatching
                        to different code specializations (for instance tiled
                        implementations). It may also give better performance than parallel
                        vectorize when the inner
                        dimension is small outer dimensions strided.

                        This vectorizer can be specified as a backend to ``Vectorize``,
                        and the ``target`` argument specifies whether it should run
                        single- or multi-threaded. The default ``target`` is "cpu".


                        **Usage:**

                        ::

                            single_threaded_vectorizer = Vectorize(numba_ufunc, backend='mini', target='cpu')
                            multi_threaded_vectorizer = Vectorize(numba_ufunc, backend='mini', target='parallel')

=================       ===============================================================

