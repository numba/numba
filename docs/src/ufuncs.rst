Universal Functions
===================

NumbaPro allows Numba functions taking scalar input arguments to be used as
NumPy ufuncs (see http://docs.scipy.org/doc/numpy/reference/ufuncs.html).

For the example codes we will assume the user has run the following import::

    from numbapro.vectorize import Vectorize

The numbapro support for ufuncs is an extension of the support in numba.
The basic vectorizer functionality was open sourced and put in numba.
NumbaPro provides parallel and streaming versions of those.

.. NOTE:: Documentation for using ufuncs can be found here: http://numba.pydata.org/numba-doc/dev/doc/arrays.html#universal-functions-ufuncs

In order to use the numbapro vectorizers, we recommend reviewing the above documentation. However, instead
of importing ``Vectorize`` from numba, we need to import it from numbapro.

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

                        **Note:** `vectorizer.build_ufunc` returns an *ufunc-like* object.  See `documentation for CUDA ufunc <CUDAufunc.html>`_ for detail.

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

