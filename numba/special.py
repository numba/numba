from __future__ import print_function, division, absolute_import

from .typing.typeof import typeof
import numpy as np

def pndindex(*args):
    """ Provides an n-dimensional parallel iterator that generates index tuples
    for each iteration point. Sequentially, pndindex is identical to np.ndindex.
    """
    return np.ndindex(*args)

class prange(object):
    """ Provides a 1D parallel iterator that generates a sequence of integers.
    Sequentially, prange is identical to range.
    """
    def __new__(cls, *args):
        return range(*args)

__all__ = ['typeof', 'prange', 'pndindex']
