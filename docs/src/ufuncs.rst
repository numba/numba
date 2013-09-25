Universal Functions
===================

With NumbaPro, `universal functions`_ (ufuncs) can be created by applying
the `vectorize` decorator on to simple scalar functions.
A ufunc can operates on scalars or NumPy arrays.
When used on arrays, the ufunc apply the core scalar function to every group of
elements from each arguments in an element-wise fashion.
NumPy Broadcasting_ is applied to every argument with mismatching dimensions.

Example: Basic
----------------

Here is a simple example to perfoum element-wise addition:

.. testcode::
    
    import numpy
    from numbapro import vectorize

    # Create a ufunc
    @vectorize(['float32(float32, float32)',
                'float64(float64, float64)'])
    def sum(a, b):
        return a + b

    # Use the ufunc
    a = numpy.arange(10)
    b = numpy.arange(10)
    result = sum(a, b)      # call the ufunc

    print("a = %s" % a)
    print("b = %s" % b)
    print("sum = %s" % result)



The ufunc is compiled for to operate on `float32` and `float64`.
It is used to compute element-wise addition of array `a` and `b` which are arrays of `numpy.float64` with 10 elements.
The output:

.. testoutput::

    a = [0 1 2 3 4 5 6 7 8 9]
    b = [0 1 2 3 4 5 6 7 8 9]
    sum = [  0.   2.   4.   6.   8.  10.  12.  14.  16.  18.]


Usage
------

A generalization of the usage of the `vectorize` decorator is described in this section.

.. py:function:: vectorize(type_signatures[, target='cpu'])

    Returns a vectorizer object to be applied to python functions.
    
    :param type_signatures: an iterable of type signatures, which are either function type object or a string describing the function type.
    :param target: a string for hardware target; e.g. "cpu", "parallel", "gpu".
    :returns: a vectorizers object.

To use multithreaded version, change the target to "parallel":

.. testcode::

    from numbapro import vectorize

    @vectorize(['float32(float32, float32)'], target='parallel')
    def sum(a, b):
        return a + b
        
For CUDA target, use "gpu" for target:

.. testcode::

    from numbapro import vectorize
    
    @vectorize(['float32(float32, float32)'], target='gpu')
    def sum(a, b):
        return a + b

Performance Guideline
---------------------

A general guideline is to choose different targets for different data sizes
and algorithms.
The "cpu" target works well for small data sizes (approx. less than 1KB) and low compute intensity algorithms. It has the least amount of overhead.
The "parallel" target works well for medium data sizes (approx. less than 1MB).
Threading adds a small delay.
The "gpu" target works well for big data sizes (approx. greater than 1MB) and
high compute intensity algorithms.  Transfering memory to and from the GPU adds
significant overhead.

Universal Function Targets
---------------------------
There are several vectorizer versions available. The different options are listed below:

=================       ===============================================================
Target                    Description
=================       ===============================================================
cpu                     Single-threaded CPU


parallel                Multi-core CPU


stream                  Optimize for CPU cache

                        .. NOTE:: Experimental. Computation speeds may vary.


gpu                     CUDA GPU

                        .. NOTE:: This creats an *ufunc-like* object.  See `documentation for CUDA ufunc <CUDAufunc.html>`_ for detail.


=================       ===============================================================


.. _`universal functions`: http://docs.scipy.org/doc/numpy/reference/ufuncs.html
.. _Broadcasting: http://docs.scipy.org/doc/numpy/reference/ufuncs.html#broadcasting
