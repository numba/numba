# -*- coding: utf-8 -*-

"""
Special compiler-recognized numba functions and attributes.
"""

from __future__ import print_function, division, absolute_import

__all__ = ['NULL', 'typeof', 'python', 'nopython', 'addressof', 'prange']

import ctypes

from numba import error

#------------------------------------------------------------------------
# Pointers
#------------------------------------------------------------------------

class NumbaDotNULL(object):
    "NULL pointer"

NULL = NumbaDotNULL()


def addressof(obj, propagate=True):
    """
    Take the address of a compiled jit function.

    :param obj: the jit function
    :param write_unraisable: whether to write uncaught exceptions to stderr
    :param propagate: whether to always propagate exceptions

    :return: ctypes function pointer
    """
    from numba import numbawrapper

    if not propagate:
        raise ValueError("Writing unraisable exceptions is not yet supported")

    if not isinstance(obj, (numbawrapper.NumbaCompiledWrapper,
                            numbawrapper.numbafunction_type)):
        raise TypeError("Object is not a jit function")

    if obj.lfunc_pointer is None:
        assert obj.lfunc is not None, obj
        from numba.codegen import llvmcontext
        llvm_context = llvmcontext.LLVMContextManager()
        obj.lfunc_pointer = llvm_context.get_pointer_to_function(obj.lfunc)

    ctypes_sig = obj.signature.to_ctypes()
    return ctypes.cast(obj.lfunc_pointer, ctypes_sig)

#------------------------------------------------------------------------
# Types
#------------------------------------------------------------------------

def typeof(value):
    """
    Get the type of a variable or value.

    Used outside of Numba code, infers the type for the object.
    """
    from numba import typesystem
    return typesystem.numba_typesystem.typeof(value)

#------------------------------------------------------------------------
# python/nopython context managers
#------------------------------------------------------------------------

class NoopContext(object):

    def __init__(self, name):
        self.name = name

    def __enter__(self, *args):
        return None

    def __exit__(self, *args):
        return None

    def __repr__(self):
        return self.name

python = NoopContext("python")
nopython = NoopContext("nopython")

#------------------------------------------------------------------------
# prange
#------------------------------------------------------------------------

def prange(start=0, stop=None, step=1):
    if stop is None:
        stop = start
        start = 0
    return range(start, stop, step)