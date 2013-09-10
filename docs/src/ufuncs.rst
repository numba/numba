Universal Functions
===================

NumbaPro allows Numba functions taking scalar input arguments to be used as
NumPy ufuncs (see http://docs.scipy.org/doc/numpy/reference/ufuncs.html).

For the example codes we will assume the user has run the following import::

    from numbapro import vectorize
    
`@vectorize` is a decorator applied on a scalar function::

    from numbapro import vectorize, float32

    @vectorize([float32(float32, float32)], target='cpu')
    def sum(a, b):
        return a + b
        
To use multithreaded version, change the target to 'parallel'::


    from numbapro import vectorize, float32

    @vectorize([float32(float32, float32)], target='parallel')
    def sum(a, b):
        return a + b

The NumbaPro support for ufuncs is an extension of the support in Numba.
The basic vectorizer functionality was open sourced and put in Numba.
NumbaPro provides parallel, stream and CUDA versions of those.

.. NOTE:: Documentation for using ufuncs can be found here: http://numba.pydata.org/numba-doc/0.9/arrays.html#universal-functions-ufuncs

In order to use the NumbaPro vectorizers, we recommend reviewing the above documentation. However, instead
of importing ``vectorize`` from Numba, we need to import it from NumbaPro.

Universal Function Types
------------------------
There are several vectorizer versions available. The different options are listed below:

=================       ===============================================================
Target                    Description
=================       ===============================================================
cpu                     Single-threaded vectorization (default if target is not given).

                        Suitable for small workloads in which
                        the overhead for cache optimization, and multithreading or GPU
                        computation is too significant.


parallel                Multi-core vectorization. Best for heavy computation
                        over large dataset.


stream                  Build a ufunc that is cache optimized.
                        The current implementation works on small chunks
                        of data at a time.  StreamVectorize copies memory to aligned,
                        contiguous buffers (small enough presumably to fit in cache)
                        and then executes the code on those chunks.

                        .. NOTE:: StreamVectorize is still in experimental stages. Computation speeds may vary.


gpu                     Build a ufunc for CUDA GPU. 

                        .. NOTE:: This creats an *ufunc-like* object.  See `documentation for CUDA ufunc <CUDAufunc.html>`_ for detail.


=================       ===============================================================

