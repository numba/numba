from __future__ import print_function, division, absolute_import

from .typing.typeof import typeof
import numpy as np

def pndindex(*args):
    return np.ndindex(*args)

class prange(object):
    def __new__(cls, *args):
        return range(*args)

__all__ = ['typeof', 'prange', 'pndindex']
