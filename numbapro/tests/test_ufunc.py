import numpy as np

from numba import *
from numbapro.vectorize.basic import BasicVectorize

def add(a, b):
    return a + b

basic_vectorizer = BasicVectorize(add)
basic_vectorizer.add(ret_type=f, arg_types=[f, f])
basic_ufunc = basic_vectorizer.build_ufunc()

def test_ufunc_add(ufunc, dtype):
    """
    >>> test_ufunc_add(basic_ufunc, np.float32)
    """
    a = np.arange(80, dtype=dtype).reshape(8, 10)
    b = a.copy()
    assert np.all(ufunc(a, b) == a + b)
    assert ufunc.reduce(ufunc.reduce(a)) == np.sum(a)
    assert np.all(ufunc.accumulate(a) == np.add.accumulate(a))
    assert np.all(ufunc.outer(a, b) == np.add.outer(a, b))


def add(a, b, c, d):
    return a + b + c + d

basic_vectorizer = BasicVectorize(add)
basic_vectorizer.add(ret_type=f, arg_types=[f, f, f, f])
basic_ufunc_multi = basic_vectorizer.build_ufunc()

def test_multiple_args(ufunc, dtype):
    """
    >>> test_multiple_args(basic_ufunc_multi, np.float32)
    """
    a = np.arange(80, dtype=dtype).reshape(8, 10)
    b = a.copy()
    print ufunc(a, b, a, b)

if __name__ == '__main__':
    import doctest
    doctest.testmod()