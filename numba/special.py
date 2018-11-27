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

def gdb(*args):
    """
    Calling this function will invoke gdb and attach it to the current process
    at the call site. Arguments are strings in the gdb command language syntax
    which will be executed by gdb once initialisation has occurred.
    """
    pass

def gdb_breakpoint():
    """
    Calling this function will inject a breakpoint at the call site that is
    recognised by both `gdb` and `gdb_init`, this is to allow breaking at
    multiple points. gdb will stop in the user defined code just after the frame
    employed by the breakpoint returns.
    """
    pass

def gdb_init(*args):
    """
    Calling this function will invoke gdb and attach it to the current process
    at the call site, then continue executing the process under gdb's control.
    Arguments are strings in the gdb command language syntax which will be
    executed by gdb once initialisation has occurred.
    """
    pass

__all__ = ['typeof', 'prange', 'pndindex', 'gdb', 'gdb_breakpoint', 'gdb_init']
