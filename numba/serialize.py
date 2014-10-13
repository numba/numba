"""
Serialization support for compiled functions.
"""

from __future__ import print_function, division, absolute_import

import imp
import marshal
from types import FunctionType


#
# Pickle support
#

def _rebuild_reduction(cls, *args):
    """
    Global hook to rebuild a given class from its __reduce__ arguments.
    """
    return cls._rebuild(*args)


def _reduce_function(func):
    """
    Reduce a Python function to picklable components
    """
    assert not func.__closure__
    # XXX globals dict can contain any kind of unpicklable values
    # (such as... modules)
    return _reduce_code(func.__code__), {}, func.__name__

def _reduce_code(code):
    """
    Reduce a code object to picklable components.
    """
    return marshal.version, imp.get_magic(), marshal.dumps(code)

def _rebuild_function(code_reduced, globals, name):
    """
    Rebuild a function from its _reduce_function() results.
    """
    code = _rebuild_code(*code_reduced)
    return FunctionType(code, globals, name)

def _rebuild_code(marshal_version, bytecode_magic, marshalled):
    """
    Rebuild a code object from its _reduce_code() results.
    """
    if marshal.version != marshal_version:
        raise RuntimeError("incompatible marshal version: "
                           "interpreter has %r, marshalled code has %r"
                           % (marshal.version, marshal_version))
    if imp.get_magic() != bytecode_magic:
        raise RuntimeError("incompatible bytecode version")
    return marshal.loads(marshalled)

