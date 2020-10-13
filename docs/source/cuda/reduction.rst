GPU Reduction
==============

Writing a reduction algorithm for CUDA GPU can be tricky.  Numba provides a
``@reduce`` decorator for converting a simple binary operation into a reduction
kernel. An example follows::

    import numpy
    from numba import cuda

    @cuda.reduce
    def sum_reduce(a, b):
        return a + b

    A = (numpy.arange(1234, dtype=numpy.float64)) + 1
    expect = A.sum()      # numpy sum reduction
    got = sum_reduce(A)   # cuda sum reduction
    assert expect == got

Lambda functions can also be used here::

    sum_reduce = cuda.reduce(lambda a, b: a + b)

The Reduce class
----------------

The ``reduce`` decorator creates an instance of the ``Reduce`` class.
Currently, ``reduce`` is an alias to ``Reduce``, but this behavior is not
guaranteed.

.. autoclass:: numba.cuda.Reduce
   :members: __init__, __call__
   :member-order: bysource
