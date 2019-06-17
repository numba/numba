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
    In non-parallel contexts, prange is identical to range.
    """
    def __new__(cls, *args):
        return range(*args)

def _gdb_python_call_gen(func_name, *args):
    # generates a call to a function containing a compiled in gdb command,
    # this is to make `numba.gdb*` work in the interpreter.
    import numba
    fn = getattr(numba, func_name)
    argstr = ','.join(['"%s"' for _ in args]) % args
    defn = """def _gdb_func_injection():\n\t%s(%s)\n
    """ % (func_name, argstr)
    l = {}
    numba.six.exec_(defn, {func_name: fn}, l)
    return numba.njit(l['_gdb_func_injection'])

def gdb(*args):
    """
    Calling this function will invoke gdb and attach it to the current process
    at the call site. Arguments are strings in the gdb command language syntax
    which will be executed by gdb once initialisation has occurred.
    """
    _gdb_python_call_gen('gdb', *args)()


def gdb_breakpoint():
    """
    Calling this function will inject a breakpoint at the call site that is
    recognised by both `gdb` and `gdb_init`, this is to allow breaking at
    multiple points. gdb will stop in the user defined code just after the frame
    employed by the breakpoint returns.
    """
    _gdb_python_call_gen('gdb_breakpoint')()


def gdb_init(*args):
    """
    Calling this function will invoke gdb and attach it to the current process
    at the call site, then continue executing the process under gdb's control.
    Arguments are strings in the gdb command language syntax which will be
    executed by gdb once initialisation has occurred.
    """
    _gdb_python_call_gen('gdb_init', *args)()


__all__ = ['typeof', 'prange', 'pndindex', 'gdb', 'gdb_breakpoint', 'gdb_init']
